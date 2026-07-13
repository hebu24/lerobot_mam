# Deploy MAM in lerobot

本文件只保留当前有效的 LIBERO-10 v3 DP/MAM 流程。旧的单任务和中间调试命令已删除，避免误用。

## 2026-07-12 LIBERO-10 DP/MAM 全流程 debug 最终结论

> 本节是当前唯一保留的 LIBERO-10 正式训练命令来源。

### 1. 当前产物红线

以下现有产物不可用于正式 MAM 训练：

- `outputs/datasets/libero10_mam_train` 和 `outputs/datasets/libero10_mam_eval` 是旧格式。旧转换器先逐元素 mask absolute axis-angle，再做 SE(3) relative 变换；该操作在旋转空间不成立，而且被遮掉的 MAS 数值无法恢复。
- 旧 MAM 数据实测 `max(abs(mam.mas_action_absolute - action)) = 3.14017`；当前 train/eval 为 431/48 个 episode，task 4 和 9 的 eval 只有 4 个，不满足每 task 45/5。
- 旧数据的 `action` stats 是 absolute stats，不是模型实际使用的 chunk-relative stats。
- 现有 task 2/8 STPM checkpoint 是旧 v1 架构。v2 修复了 token 顺序、时间位置、causal/padding mask 和 episode 泄漏，因此会明确拒绝加载 v1，必须重训。
- 当前没有可用于正式 LIBERO-10 的 MAM policy checkpoint。下面必须从原始 HDF5 重建数据，再训练 STPM 和 MAM。

### 2. 两条 pipeline 的最终不变量

#### 2.1 DP 原始 delta 过拟合路径

```text
官方 delta action
  -> DP normalize
  -> diffusion noise/loss
  -> denoise action chunk
  -> unnormalize
  -> LIBERO relative controller
```

- 该路径必须使用 `policy.use_relative_actions=false` 和 `env.control_mode=relative`。
- 这里的 `relative` 是 LIBERO 单步 delta controller，不是 chunk-relative SE(3) action。
- 已有单 demo DP checkpoint 属于该路径，不能做 relative-to-absolute SE(3) 后处理。

#### 2.2 DP 正式 chunk-relative 路径

```text
官方 delta action
  -> 在原 demo XML/状态中 replay，保存 absolute OSC controller goal
  -> 按最新 14D state 建立整段 action chunk 的 anchor
  -> position: p_rel = p_abs - p_anchor
  -> rotation: R_rel = R_abs @ R_anchor.T
  -> 用 chunk-relative action stats 做 min-max normalize
  -> diffusion 训练/去噪
  -> unnormalize 整段 chunk
  -> 用当前真实 state: p_abs = p_rel + p_anchor,
     R_abs = R_rel @ R_anchor
  -> 整段 absolute action 入队
  -> LIBERO absolute controller
```

- 14D state 的 quaternion 描述 Panda `right_hand` body，而 OSC 控制 `grip_site`。anchor rotation 已统一右乘 `Rz(-pi/2)` 转到 controller frame。
- relative stats 按训练时真实的 action delta indices 重算，并在 overfit subset 之后重算，禁止沿用 absolute stats。
- rollout 只在 action queue 为空时预测并转换整段 chunk，不能逐步重新 anchor。
- 固定评估从独立 eval dataset metadata 读取每个 task 的原始 `libero/init_state`。DP 的 `eval.n_episodes=5` 表示每 task 5 个。

#### 2.3 MAM 路径

```text
absolute action + 完整 absolute MAS + 独立 binary mask + progress
  -> 以 source episode 为单位先做 task-balanced train/eval split
  -> 如有多个 mask type，再在各 split 内展开
  -> action 和完整 MAS 使用同一个最新 state anchor 做 SE(3) chunk-relative
  -> 使用同一套 relative action stats normalize
  -> normalize 后才应用 mask
  -> 构造 B,S,T,D 的 long/short MAS window
  -> RGB/state/language/MAS conditioning -> diffusion U-Net
  -> known/unknown region-balanced loss，padding 不计入任一区域
  -> 可选逐 diffusion timestep 的正确 inpainting
  -> STPM 预测 progress
  -> 整段 relative action 用当前真实 state 转 absolute
  -> LIBERO absolute controller
```

- `mam.mas_action_absolute` 永远保存完整 absolute action；mask 只保存在 `mam.mas_action_mask`。
- 所有 observation history 共用当前 rollout 时刻的最新 anchor，但各自采样不同的时间窗口；不能用跨 chunk 的旧 MAM queue 代替。
- STPM 使用独立 camera/language/state token、time-major 排列、时间/模态位置编码、按时间 causal mask 和 padding mask。
- STPM train/val 按 `(task, libero/source_episode_id)` 分组，state normalization 只统计 train split，避免帧级泄漏和多 mask 轨迹泄漏。
- MAM 评估的 `eval.n_episodes=50` 表示所有 task 的总数，即每 task 5 个；这与通用 DP eval 的参数语义不同。

### 3. 正式数据重建命令

#### 3.1 环境

```bash
export MUJOCO_GL=egl
export LIBERO_ASSETS_PATH="$PWD/.cache/libero/assets"
export HF_HOME="$PWD/.hf-cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export UV_CACHE_DIR="$PWD/.uv-cache"
export MPLCONFIGDIR=/tmp/matplotlib-cache
```

#### 3.2 原始 HDF5 转完整 LeRobot delta 数据

正式训练不要设置 `--max-episodes-per-task`。

```bash
uv run python scripts/convert_libero10_hdf5_to_lerobot.py \
  --input-dir=outputs/source/libero_official/libero_10 \
  --output-root=outputs/datasets/libero10_full_v3 \
  --output-repo-id=local/libero10_full_v3 \
  --suite=libero_10 \
  --height=128 \
  --width=128 \
  --use-videos \
  --overwrite
```

#### 3.3 delta 转 absolute controller goal

正式数据必须使用 replay 和每条 demo 自带的 XML。`--no-replay` 仅允许做 codec 调试，不允许生成正式训练集。

```bash
uv run python scripts/convert_libero_delta_to_absolute.py \
  --input-root=outputs/datasets/libero10_full_v3 \
  --input-repo-id=local/libero10_full_v3 \
  --output-root=outputs/datasets/libero10_absolute_v3 \
  --output-repo-id=local/libero10_absolute_v3 \
  --task=libero_10 \
  --observation-height=128 \
  --observation-width=128 \
  --source-hdf5-dir=outputs/source/libero_official/libero_10 \
  --source-action-strategy=controller-goal \
  --replay \
  --overwrite
```

#### 3.4 生成严格 45/5 的 MAM train/eval split

```bash
uv run python scripts/convert_libero_absolute_to_mam.py \
  --input-root=outputs/datasets/libero10_absolute_v3 \
  --input-repo-id=local/libero10_absolute_v3 \
  --output-root=outputs/datasets/libero10_mam_v3 \
  --output-repo-id=local/libero10_mam_v3 \
  --eval-per-task=5 \
  --split-seed=0 \
  --mask-types=random_mask \
  --retain-ratio=0.2 \
  --n-obs-steps=2 \
  --horizon=32 \
  --overwrite
```

实际输出为 `libero10_mam_v3_train` 和 `libero10_mam_v3_eval`。使用多个 mask type 时，episode 数会乘以 mask type 数量，但 train/eval 的 source episode 仍严格隔离。

#### 3.5 数据硬审计

任何一项失败都必须停止训练。

```bash
uv run python scripts/audit_libero10_mam_dataset.py \
  --train-root=outputs/datasets/libero10_mam_v3_train \
  --train-repo-id=local/libero10_mam_v3_train \
  --eval-root=outputs/datasets/libero10_mam_v3_eval \
  --eval-repo-id=local/libero10_mam_v3_eval \
  --train-per-task=45 \
  --eval-per-task=5 \
  --n-obs-steps=2 \
  --horizon=32
```

审计覆盖 14D state、7D action、双 128x128 RGB、完整 MAS、binary mask、progress、relative stats、SE(3) roundtrip、task 计数和 source split 泄漏。

### 4. 训练 v2 STPM

每个 task 使用各自的 train episode；不要复用旧 task2/task8 checkpoint。

```bash
for TASK_ID in 0 1 2 3 4 5 6 7 8 9; do
  EPISODES=$(uv run python -c "from lerobot.datasets import LeRobotDatasetMetadata; m=LeRobotDatasetMetadata('local/libero10_mam_v3_train', root='outputs/datasets/libero10_mam_v3_train'); print([int(r['episode_index']) for r in m.episodes if int(r['libero/task_id']) == ${TASK_ID}])")
  uv run python -m lerobot.scripts.lerobot_train_stpm \
    --dataset.repo_id=local/libero10_mam_v3_train \
    --dataset.root=outputs/datasets/libero10_mam_v3_train \
    --episodes="${EPISODES}" \
    --output_dir=outputs/train/stpm_libero10_v2_task${TASK_ID} \
    --n_obs_steps=1 \
    --frame_gap=1 \
    --batch_size=32 \
    --num_workers=4 \
    --steps=10000 \
    --val_ratio=0.1 \
    --device=cuda \
    --require_cuda \
    --vision_ckpt=/home/hebu/code/mam/ManiSkill/pretrained/clip-vit-base-patch32
done
```

### 5. 正式 DP 训练与固定 split 评估

DP 可以直接读取 MAM train split；额外 MAM columns 会被忽略，action 仍是完整 absolute action。

```bash
CUDA_VISIBLE_DEVICES=0 \
NUM_GPUS=1 \
POLICY_DEVICE=cuda \
DATASET_REPO_ID=local/libero10_mam_v3_train \
DATASET_ROOT=outputs/datasets/libero10_mam_v3_train \
EVAL_DATASET_REPO_ID=local/libero10_mam_v3_eval \
EVAL_DATASET_ROOT=outputs/datasets/libero10_mam_v3_eval \
USE_RELATIVE_ACTIONS=true \
ENABLE_EVAL=true \
EVAL_FREQ=1000 \
EVAL_N_EPISODES=5 \
EVAL_BATCH_SIZE=1 \
ENV_TASK=libero_10 \
ENV_TASK_IDS='[0,1,2,3,4,5,6,7,8,9]' \
ENV_CONTROL_MODE=absolute \
ENV_OBSERVATION_HEIGHT=128 \
ENV_OBSERVATION_WIDTH=128 \
bash scripts/train_diffusion_libero_put_bowl_on_plate_multigpu.sh \
  --policy.horizon=32 \
  --policy.n_action_steps=15 \
  --policy.use_language_conditioning=true
```

这里 `EVAL_N_EPISODES=5` 是每 task 5 个。训练入口会在创建 env 前把 eval dataset 中记录的 init state 按 task 注入环境；若某 task 不足 5 个会立即报错。

### 6. 正式 MAM 训练与固定 split 评估

```bash
STPM_PATHS='{"libero_10/0":"outputs/train/stpm_libero10_v2_task0","libero_10/1":"outputs/train/stpm_libero10_v2_task1","libero_10/2":"outputs/train/stpm_libero10_v2_task2","libero_10/3":"outputs/train/stpm_libero10_v2_task3","libero_10/4":"outputs/train/stpm_libero10_v2_task4","libero_10/5":"outputs/train/stpm_libero10_v2_task5","libero_10/6":"outputs/train/stpm_libero10_v2_task6","libero_10/7":"outputs/train/stpm_libero10_v2_task7","libero_10/8":"outputs/train/stpm_libero10_v2_task8","libero_10/9":"outputs/train/stpm_libero10_v2_task9"}' \
CUDA_VISIBLE_DEVICES=0 \
NUM_GPUS=1 \
POLICY_DEVICE=cuda \
DATASET_REPO_ID=local/libero10_mam_v3_train \
DATASET_ROOT=outputs/datasets/libero10_mam_v3_train \
MAM_EVAL_DATASET_REPO_ID=local/libero10_mam_v3_eval \
MAM_EVAL_DATASET_ROOT=outputs/datasets/libero10_mam_v3_eval \
ENABLE_EVAL=true \
EVAL_FREQ=1000 \
EVAL_N_EPISODES=50 \
EVAL_BATCH_SIZE=1 \
ENV_TASK=libero_10 \
ENV_TASK_IDS='[0,1,2,3,4,5,6,7,8,9]' \
ENV_CONTROL_MODE=absolute \
ENV_OBSERVATION_HEIGHT=128 \
ENV_OBSERVATION_WIDTH=128 \
bash scripts/run_mam_libero10_conda.sh
```

这里 `EVAL_N_EPISODES=50` 是所有 task 的总数。评估使用同一个选中 episode 前缀同时配置 init state、MAS 和 task 分组，避免三者错位；STPM 只在 action queue 为空时运行一次。

### 7. 已完成验证

- rotation codec：病理 `pi` 旋转、10,000 个随机 SO(3)、14D state controller-frame roundtrip 均通过。
- 真实 MAM train 118,283 帧全量 codec 扫描：position 最大误差 `3.73e-9`，rotation matrix 最大误差 `7.15e-7`，gripper 误差 `0`。
- MAM 真实数据 CUDA smoke：单 batch preprocess、forward、backward、inference、relative-to-absolute 全通过；loss `1.118688`，action shape `(1,15,7)`，显存峰值 `551.2 MB`。
- STPM v2：真实 episode 的 1-step train/val、checkpoint reload 和 inference 通过。
- absolute expert replay：旧 eval episode 0 在 EGL absolute controller 下 `1/1 success`。
- DP 单 demo checkpoint：CUDA 静态 denoise loss `0.00500688`，native action MAE `0.0266374`；已有 step 2000 闭环日志为 `100%`。
- 定向单元/集成回归为 `75 passed, 4 skipped`；ruff、shell 语法和 `git diff --check` 全部通过。skip 为当前 CPU 测试设备下的既定 CUDA 分支，GPU 另由真实 CUDA smoke 覆盖。

### 8. 尚未执行的长任务

正式 500 episode 数据重建、10 个 STPM 的完整 10k-step 训练，以及 LIBERO-10 DP/MAM 的完整长训练尚未执行。原因不是代码路径仍有已知错误，而是现有 MAM/STPM 产物已被审计判定无效，必须先按本节重建；在新数据通过硬审计前，不应启动长训练。
