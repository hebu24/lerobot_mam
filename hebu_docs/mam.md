# Deploy MAM in lerobot

本文件只保留当前有效的 LIBERO-10 v3 DP/MAM 流程。旧的单任务和中间调试命令已删除，避免误用。

## LIBERO-10 v3 DP/MAM 当前流程

> 本节是当前唯一保留的 LIBERO-10 正式训练命令来源。

### 1. 唯一 v3 pipeline 不变量

#### 1.1 DP v3 chunk-relative 路径

```text
官方 delta action
  -> 在原 demo XML/状态中计算 absolute OSC controller goal
  -> 用 absolute controller 闭环 replay，并重新物化同一 rollout 的 RGB/state/action
  -> 按最新 14D state 建立整段 action chunk 的 anchor
  -> position: p_rel = p_abs - p_anchor
  -> rotation: R_rel = R_abs @ R_anchor.T
  -> 用 chunk-relative action stats 做 min-max normalize
  -> diffusion 训练/去噪
  -> unnormalize 整段 chunk
  -> 用当前真实 state: p_abs = p_rel + p_anchor,
     R_abs = R_rel @ R_anchor
  -> 整段 absolute action 入队
  -> LIBERO absolute controller (`env.control_mode=absolute`)
```

- 14D state 的 quaternion 描述 Panda `right_hand` body，而 OSC 控制 `grip_site`。anchor rotation 已统一右乘 `Rz(-pi/2)` 转到 controller frame。
- relative 训练数据必须同时声明 `observation_materialization=closed_loop_absolute_controller` 和 `relative_action_ready=true`。只替换 action 列、仍保留 delta-controller observation 的旧 v3 数据禁止训练。
- relative stats 按训练时真实的 action delta indices 重算，并在 overfit subset 之后重算，禁止沿用 absolute stats。
- rollout 只在 action queue 为空时预测并转换整段 chunk，不能逐步重新 anchor。
- 固定评估从独立 eval dataset metadata 读取每个 task 的原始 `libero/init_state`。DP 的 `eval.n_episodes=5` 表示每 task 5 个。

#### 1.2 MAM v3 路径

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

### 2. v3 K-task 过拟合

`K` 可取 `1..10`，默认选择前 K 个 task，每个 task 选 1 条 trajectory。训练和评估使用完全相同的 episode ID 及 init state。

```bash
bash scripts/run_diffusion_libero10_v3_overfit.sh 3
```

也可用 `K=5 bash scripts/run_diffusion_libero10_v3_overfit.sh`。该入口固定 `policy.use_relative_actions=true` 和 `env.control_mode=absolute`，默认训练 20000 steps；启动前会校验 v3 manifest、数据 schema、source identity、CUDA，并对所有选中 trajectory 运行 `1/4/n_action_steps/full` 的真实 runtime oracle。

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

正式训练不要设置 `--max-episodes-per-task`，也不要设置 `--use-videos`。闭环重物化必须原子替换两路图像，不能沿用 delta rollout 的旧视频。

```bash
uv run python scripts/convert_libero10_hdf5_to_lerobot.py \
  --input-dir=outputs/source/libero_official/libero_10 \
  --output-root=outputs/datasets/libero10_full_v3 \
  --output-repo-id=local/libero10_full_v3 \
  --suite=libero_10 \
  --height=128 \
  --width=128 \
  --overwrite
```

#### 3.3 delta 转 absolute controller goal

正式数据必须通过 `--source-hdf5-dir` 读取每条 demo 自带的 XML/状态，生成精确的 absolute controller goal，并在 absolute controller 下闭环重放、重新物化 RGB/state。只有转换完整结束后 manifest 才会写入 `relative_action_ready=true`；旧 action-only 输出必须用 `--overwrite` 重建。

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
  --auto-repair-failed-replays \
  --allow-unrepairable-episodes \
  --overwrite
```

#### 3.4 生成每个 task 固定 5 条 eval 的完整 MAM train/eval split

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
  --allow-source-exclusions \
  --overwrite
```

实际输出为 `libero10_mam_v3_train` 和 `libero10_mam_v3_eval`。无法通过真实闭环回放的 source episode 会被显式排除；当前 v3 全量审计保留 485 条，按 task 固定抽取 5 条 eval，得到 435 条 train 和 50 条 eval。使用多个 mask type 时，episode 数会乘以 mask type 数量，但 train/eval 的 source episode 仍严格隔离。

当前 mask 语义与 ManiSkill MAM 对齐：

- `pose` / `pose_motion_planning`：随机保留 `floor(T × retain_ratio)` 个时间步的完整 7D action。
- `points` / `3D_points`：随机保留若干时间步的 XY / XYZ。
- `random_mask`：在整个 `T × 7` action 矩阵中随机保留元素。
- `2D_video_trajectory` / `2D_image_trajectory`：保留所有时间步的 XY。
- `2D_partial_trajectory`：保留一个连续 XY 窗口，长度由 `--mask-seq-len` 指定。
- `local_planner`：保留整段 action，但遮挡一个连续的完整 7D 窗口。
- `mix0`（别名 `mix`）：保留全轨迹 XY，再保留一个完整 7D pose 和另外三个 XYZ 点。
- `none`：不提供任何已知 action。

生成 mixed 数据集有两种方式：

```bash
# 每条 source episode 为每种 mask 各复制一次。
--mask-types=pose,points,3D_points,random_mask,mix0 \
--mask-assign-mode=one_demo_multi_mask \
--retain-ratio=0.2

# 每条 source episode 只使用一种 mask，并按给定比例分配。
--mask-types=pose,points,3D_points,random_mask,mix0 \
--mask-assign-mode=composition \
--mask-composition=0.2,0.2,0.2,0.2,0.2 \
--retain-ratio=0.2
```

训练 mixed 数据集时设置与 manifest 顺序一致的
`MASK_TYPES=pose,points,3D_points,random_mask,mix0`。若评估集使用
`one_demo_multi_mask`，`EVAL_N_EPISODES` 需要包含展开后的 episode 数。

#### 3.5 数据硬审计

任何一项失败都必须停止训练。

```bash
uv run python scripts/audit_libero10_mam_dataset.py \
  --train-root=outputs/datasets/libero10_mam_v3_train \
  --train-repo-id=local/libero10_mam_v3_train \
  --eval-root=outputs/datasets/libero10_mam_v3_eval \
  --eval-repo-id=local/libero10_mam_v3_eval \
  --source-root=outputs/datasets/libero10_absolute_v3 \
  --source-repo-id=local/libero10_absolute_v3 \
  --eval-per-task=5 \
  --n-obs-steps=2 \
  --horizon=32 \
  --allow-source-exclusions
```

审计覆盖 14D state、7D action、双 128x128 RGB、完整 MAS、binary mask、progress、relative stats、SE(3) roundtrip、task 计数和 source split 泄漏。

随后必须在真实 eval runtime 中运行 relative-action oracle。下面命令按实际 DP 配置对完整 eval split 执行 `chunk=15`；任何失败都会返回非零退出码，禁止开始训练。

```bash
uv run python scripts/audit_libero_chunk_relative_oracle.py \
  --dataset-root=outputs/datasets/libero10_mam_v3_eval \
  --max-episodes=50 \
  --chunk-sizes=15 \
  --post-hold-steps=0 \
  --output-json=outputs/audit/libero10_mam_v3_eval_chunk15_oracle.json
```

### 4. 训练 v2 STPM

每个 task 使用各自的 train episode，checkpoint 必须由当前 v3 split 训练生成。

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
STPM_PATHS='{"libero_10/0":"outputs/train/stpm_libero10_v2_task0","bash scripts/run_mam_libero10_v3_overfit.sh 1libero_10/1":"outputs/train/stpm_libero10_v2_task1","libero_10/2":"outputs/train/stpm_libero10_v2_task2","libero_10/3":"outputs/train/stpm_libero10_v2_task3","libero_10/4":"outputs/train/stpm_libero10_v2_task4","libero_10/5":"outputs/train/stpm_libero10_v2_task5","libero_10/6":"outputs/train/stpm_libero10_v2_task6","libero_10/7":"outputs/train/stpm_libero10_v2_task7","libero_10/8":"outputs/train/stpm_libero10_v2_task8","libero_10/9":"outputs/train/stpm_libero10_v2_task9"}' \
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

### 7. 尚未执行的长任务

正式 500 episode 数据重建、10 个 STPM 的完整 10k-step 训练，以及 LIBERO-10 DP/MAM 的完整长训练尚未执行。必须先按本节重建正式数据；在新数据通过硬审计前，不应启动长训练。
