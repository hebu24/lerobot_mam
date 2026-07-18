# LIBERO-10 v3 全量数据集、Replay 与 Pipeline 排查报告

- 日期：2026-07-18
- 仓库：`/home/hebu/code/mam/lerobot_mam`
- 最终状态：可开始全量 Diffusion Policy 训练
- 训练动作表示：chunk-relative SE(3)
- 环境控制表示：absolute controller goal

> 2026-07-18 清理记录：已删除 `libero10_full_v3`、四个 `*_sample*` 数据集、sample split 以及非 final 的历史 audit，回收约 5.2G。当前保留 source HDF5、`libero10_absolute_v3`、最终 train/eval、最终 replay/oracle 和 Task6 source400 oracle。本文仍保留已删除历史实验的结果，便于解释排查过程。

## 1. 最终结论

本轮重新生成了 LIBERO-10 v3 全量训练集和固定评估集，并完成了数据结构、split、relative action、真实环境 replay、真实 inference 预处理/后处理和 CUDA 训练 smoke test。

| 项目 | 结果 |
| --- | --- |
| 原始 source episode | 500 |
| 闭环转换可用 episode | 485 |
| 显式排除 episode | 15 |
| Train | 435 episodes / 119,467 frames |
| Eval | 50 episodes / 13,665 frames |
| Train/Eval source 泄漏 | 0 |
| Eval runtime absolute replay | 50/50 成功 |
| Eval chunk=15 relative inference oracle | 50/50 成功 |
| 最大 runtime 位置偏差 | 0.834 mm |
| 最大 runtime 旋转偏差 | 0.001691 rad，约 0.097° |
| CUDA 训练 smoke test | forward/backward/optimizer 全部通过 |
| Task6 替代 demo 过拟合 | 4k、5k eval 均成功 |

最终数据：

- Train：[`outputs/datasets/libero10_mam_v3_train`](../outputs/datasets/libero10_mam_v3_train/meta/info.json)
- Eval：[`outputs/datasets/libero10_mam_v3_eval`](../outputs/datasets/libero10_mam_v3_eval/meta/info.json)
- Split manifest：[`outputs/datasets/libero10_mam_v3_split.json`](../outputs/datasets/libero10_mam_v3_split.json)
- Eval replay：[`summary.json`](../outputs/audit/libero10_mam_v3_eval_runtime_replay_final/summary.json)
- Relative oracle：[`libero10_mam_v3_eval_chunk15_oracle_final.json`](../outputs/audit/libero10_mam_v3_eval_chunk15_oracle_final.json)

## 2. v3 数据生成流程

### 2.1 环境变量

```bash
export MUJOCO_GL=egl
export LIBERO_ASSETS_PATH="$PWD/.cache/libero/assets"
export HF_HOME="$PWD/.hf-cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export UV_CACHE_DIR="$PWD/.uv-cache"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
```

`LIBERO_ASSETS_PATH` 必须指向完整 assets。之前出现的 `wine_rack.xml`、`akita_black_bowl.xml`、`cream_cheese.xml` 和 `scenes/libero_kitchen_tabletop_base_style.xml` 缺失属于 simulator assets 问题，不影响训练数据本身，但会阻止所有环境 replay/eval。

### 2.2 官方 HDF5 转 LeRobot delta 数据

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

输出包含 500 条原始 episode。

### 2.3 Delta 转 closed-loop absolute 数据

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

这一阶段不是只改 action 数值，而是：

1. 从每条 demo 的 HDF5 XML 和 init state 恢复 simulator。
2. 将原始 delta action 转成 absolute controller goal。
3. 在 absolute controller 下闭环执行。
4. 重新采集双路 RGB、14D state 和实际 controller goal。
5. 只有完整转换结束后才写入 `relative_action_ready=true`。

最终保留 485 条，以下 15 条显式标记为不可修复，不进入 train/eval：

```text
84, 104, 107, 114, 134, 145, 151, 191, 198, 264,
275, 415, 447, 480, 487
```

### 2.4 生成最终固定 train/eval split

最终 eval 使用 replay-certified 的固定 source episode，而不是每次随机切分：

```bash
uv run python scripts/convert_libero_absolute_to_mam.py \
  --input-root=outputs/datasets/libero10_absolute_v3 \
  --input-repo-id=local/libero10_absolute_v3 \
  --output-root=outputs/datasets/libero10_mam_v3 \
  --output-repo-id=local/libero10_mam_v3 \
  --eval-per-task=5 \
  --eval-episode-ids='252,305,6,59,366,453,403,201,152,121,254,307,25,71,369,455,416,203,164,130,269,320,26,79,377,483,418,222,178,133,272,331,32,50,386,490,424,236,180,140,295,337,33,96,395,496,434,243,199,146' \
  --split-seed=0 \
  --mask-types=random_mask \
  --retain-ratio=0.2 \
  --n-obs-steps=2 \
  --horizon=32 \
  --allow-source-exclusions \
  --overwrite
```

最终每个 task 的数量：

| Task | Train | Eval source episode |
| ---: | ---: | --- |
| 0 | 43 | 252, 254, 269, 272, 295 |
| 1 | 45 | 305, 307, 320, 331, 337 |
| 2 | 45 | 6, 25, 26, 32, 33 |
| 3 | 44 | 50, 59, 71, 79, 96 |
| 4 | 45 | 366, 369, 377, 386, 395 |
| 5 | 43 | 453, 455, 483, 490, 496 |
| 6 | 43 | 403, 416, 418, 424, 434 |
| 7 | 45 | 201, 203, 222, 236, 243 |
| 8 | 42 | 152, 164, 178, 180, 199 |
| 9 | 40 | 121, 130, 133, 140, 146 |

Task3 最初选择的 source 85 在同一环境连续 replay 中不稳定，因此将 source 85 留在 train，并用 source 50 替换进 eval。替换后 Task3 的 5 条 eval 连续 replay 全部通过。

## 3. Relative action pipeline

最终训练和 inference 的真实数据流如下：

```text
磁盘中的 closed-loop absolute action/state
    ↓
按 observation anchor 编码 chunk-relative SE(3) action
    ↓
relative action normalization / Diffusion Policy
    ↓
模型输出 relative action chunk
    ↓
unnormalization
    ↓
使用当前 live observation anchor 解码为 absolute controller goal
    ↓
截取当前可执行窗口并送入 LIBERO absolute controller
```

关键配置：

```text
n_obs_steps=2
horizon=32
n_action_steps=15
action_delta_indices=[-1, 0, 1, ..., 30]  # 共 32 个
policy.use_relative_actions=true
env.control_mode=absolute
```

### 3.1 为什么不能设置 horizon=15

曾出现：

```text
ValueError: Executable action window exceeds predicted chunk horizon:
start=1, end=16, horizon=15
```

原因是 `n_obs_steps=2` 时当前动作窗口从 `n_obs_steps - 1 = 1` 开始；执行 15 步需要访问 `[1, 16)`。因此必须满足：

```text
n_obs_steps - 1 + n_action_steps <= horizon
```

即这里至少需要 horizon 16。正式配置使用 horizon 32，并在启动脚本 GPU 初始化前做硬检查。

### 3.2 Relative stats 启动内存问题

全量训练 smoke test 曾发现 relative stats 计算直接读取完整 Hugging Face table，导致 RGB 图像也被解码，启动内存接近耗尽。

修复后：

- stats 计算只选择 `action`、`observation.state` 和 `episode_index` 数值列；
- 正常全量训练直接复用 manifest 中已经独立审计的 relative action stats；
- 单轨迹过拟合仍重新计算所选轨迹的 stats，避免使用全量分布；
- 启动脚本校验 stats 的 `n_obs_steps`、`horizon` 和 delta indices 与 policy 完全一致。

## 4. 数据静态审计

执行命令：

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

审计通过项目：

- pipeline version、closed-loop manifest 和 `relative_action_ready`；
- 14D state、7D action、双路 128×128 RGB；
- MAM binary mask、progress 和完整性；
- 485 条有效 source 被 train/eval 完整且仅覆盖一次；
- 15 条 exclusion 与 absolute manifest 一致；
- train/eval source episode 零交集；
- eval 每个 task 固定 5 条；
- init state fidelity；
- relative action stats 独立 parquet 重算一致；
- SE(3) encode/decode round-trip。

## 5. 环境 Replay 审计

### 5.1 Absolute action 标准 eval runtime replay

最终 eval split 在标准训练评估 runtime 中逐条执行磁盘中的 absolute action：

```bash
uv run python scripts/replay_libero_dataset_success_by_task.py \
  --dataset-root=outputs/datasets/libero10_mam_v3_eval \
  --action-key=action \
  --action-transform=none \
  --control-mode=absolute \
  --backend=direct \
  --max-episodes=50 \
  --seed=1000 \
  --num-steps-wait=0 \
  --post-noop-steps=0 \
  --output-dir=outputs/audit/libero10_mam_v3_eval_runtime_replay_final \
  --overwrite-output
```

结果：

```text
total=50
success=50
failure=0
success_rate=1.0
每个 task=5/5
```

### 5.2 预处理 → 伪模型输出 → 后处理 oracle

这一步不使用学习模型。它将数据集中的 absolute action 编码成 relative action，直接伪装成模型输出，再走真实 inference 后处理，用 live observation anchor 解码并执行。

```bash
uv run python scripts/audit_libero_chunk_relative_oracle.py \
  --dataset-root=outputs/datasets/libero10_mam_v3_eval \
  --max-episodes=50 \
  --chunk-sizes=15 \
  --seed=1000 \
  --num-steps-wait=0 \
  --post-hold-steps=0 \
  --output-json=outputs/audit/libero10_mam_v3_eval_chunk15_oracle_final.json
```

结果：

```text
success=50/50
max anchor position error=0.0008339367 m
max goal position error=0.0008339367 m
max anchor rotation error=0.0016238276 rad
max goal rotation error=0.0016908196 rad
```

这些非零值来自 simulator 中 live state 相对录制 state 的闭环物理漂移，不是 relative 编码/解码代数误差。所有 50 条在真实 chunk=15 inference 语义下仍成功。

### 5.3 `chunk=1/4/15/full` 的含义

- `chunk=1`：每步重新读取 live observation 并重新锚定，最强闭环反馈。
- `chunk=4`：一次解码并连续执行 4 个动作。
- `chunk=15`：正式 DP 的 `n_action_steps=15`，是训练前必须通过的核心检查。
- `chunk=full`：整条轨迹只在起点锚定一次，主要用于检查完整 relative 数学一致性，不代表正式 policy 执行方式。

部分 demo 在 chunk=4 或 chunk=15 下可能因接触、夹取和物体碰撞放大微小漂移，但 full 成功。只有最终固定 eval split 的 chunk=15 全部成功后才允许正式训练。

### 5.4 旧的 455/485 replay 数字为什么不作为最终结论

旧产物 `outputs/audit/libero10_mam_v3_full_replay` 得到 455/485。该诊断强制每条 episode 使用 source HDF5 内嵌 XML，与正式训练评估使用的标准 benchmark BDDL runtime 不完全相同，因此不能用它代表最终 inference 成功率。

最终门禁采用：

1. 全部 485 条的静态一致性、source XML closed-loop 转换和 SE(3) round-trip；
2. 固定 eval 50 条的标准 runtime absolute replay；
3. 固定 eval 50 条的真实 chunk=15 relative inference oracle。

需要注意，“train 可用”表示轨迹在自己的 source model 中闭环一致，可用于监督学习；不保证任意 train demo 都能在标准 eval model XML 中逐帧复现。因此进行单轨迹过拟合时，应先用 oracle selector 选择标准 runtime 也通过的轨迹。Eval split 则满足更严格的标准 runtime 50/50 门禁。

## 6. 全量训练 smoke test

最终 split 上执行过 1-step CUDA 训练：

```bash
STEPS=1 \
ENABLE_EVAL=false \
SAVE_FREQ=100000 \
OUTPUT_DIR=outputs/train/diffusion_libero10_v3_full_smoke_finalsplit_20260718 \
bash scripts/run_diffusion_libero10.sh
```

结果：

```text
CUDA forward: passed
backward: passed
optimizer step: passed
loss=1.172
grad_norm=11.410
step_time=16.94s
```

随后执行宿主机 dry-run：

```bash
DRY_RUN=true bash scripts/run_diffusion_libero10.sh
```

通过以下启动门禁：

- train=435 episodes / 435 unique sources；
- eval=50 episodes / 50 unique sources；
- CUDA GPU 数量正确；
- relative stats 与 policy horizon 匹配；
- action window 合法；
- train/eval 零泄漏；
- 完整 accelerate/lerobot_train 命令拼装成功。

相关代码回归结果：

```text
bash -n: passed
ruff: passed
pytest: 42 passed
```

## 7. Task6 单轨迹过拟合排查

Task6：

```text
put the white mug on the plate and put the chocolate pudding to the right of the plate
```

### 7.1 原 demo

| 字段 | 值 |
| --- | --- |
| source episode | 401 |
| 当前 train episode | 350 |
| 长度 | 325 frames |
| oracle replay | 成功 |
| 学习 eval | 1k–9k 均为 0 |
| 训练终止原因 | 10k checkpoint 保存时磁盘无空间 |

原 demo 本身不是损坏数据，relative oracle 也能执行成功。视频表现为：

- 杯子放置不稳定或接触后倾倒；
- 即使杯子阶段完成，也常未完成第二阶段的 pudding 放置；
- 训练 loss 已经很低，但闭环成功率仍为 0，说明逐动作拟合误差在接触阶段被放大。

这是一个较长的双阶段接触任务，单条轨迹对时序误差和动作分布非常敏感。

### 7.2 替代 demo

| 字段 | 值 |
| --- | --- |
| source episode | 400 |
| 当前 train episode | 349 |
| 长度 | 203 frames |
| chunk oracle | 1/4/15/full 全部成功 |
| 1k eval | 0% |
| 2k eval | 0% |
| 3k eval | 0% |
| 4k eval | 100% |
| 5k eval | 100% |
| 5k loss | 0.0044437 |

过程表现：

- 1k：抓杯后在 plate 附近失稳，未进入 pudding 阶段；
- 2k：杯子/plate 接触不稳定，动作停滞；
- 3k：杯子已放稳并开始接近 pudding，但第二阶段失败；
- 4k、5k：完整成功。

结论：

1. Task6 并非不可学习；
2. relative preprocessing/postprocessing pipeline 没有系统性错误；
3. 原 source 401 较长、接触更敏感，对单轨迹 DP 更难；
4. 旧实验还叠加了磁盘满导致的非正常终止；
5. 后续 Task6 过拟合优先使用 source 400，或先对候选轨迹执行 chunk=15 oracle。

本报告编写时 `outputs/train` 已被清理，因此旧 Task6 checkpoint/video 不再作为持久链接；上述指标来自完成的 5k 实验记录。

## 8. 已定位的其他故障

### 8.1 Checkpoint 保存失败

报错：

```text
safetensors_rust.SafetensorError:
I/O error: No space left on device (os error 28)
```

该错误发生在 optimizer state 序列化，不是模型训练或数据 pipeline 错误。一次 DP checkpoint 大约包含 1.1 GB model 和 2.2 GB optimizer state，频繁保存会迅速占满磁盘。

正式训练前建议：

```bash
df -h .
nvidia-smi
SAVE_FREQ=20000 bash scripts/run_diffusion_libero10.sh
```

### 8.2 Robosuite private macro warning

```text
No private macro file found!
```

这是 warning，不会导致 replay 或训练失败。缺失 LIBERO XML/assets 才是阻塞性错误。

## 9. 正式训练命令

启动脚本：[`scripts/run_diffusion_libero10.sh`](../scripts/run_diffusion_libero10.sh)

推荐在确认磁盘空间后启动：

```bash
SAVE_FREQ=20000 bash scripts/run_diffusion_libero10.sh
```

默认核心配置：

```text
steps=50000
batch_size=32 / GPU
num_workers=8
learning_rate=1e-4
n_obs_steps=2
horizon=32
n_action_steps=15
use_relative_actions=true
env.control_mode=absolute
eval_freq=5000
eval_n_episodes=5 per task
```

如需只检查命令而不启动训练：

```bash
DRY_RUN=true bash scripts/run_diffusion_libero10.sh
```

## 10. 训练前硬门禁

正式训练前必须同时满足：

- `LIBERO_ASSETS_PATH` 完整；
- train/eval metadata 和 split manifest 存在；
- train=435、eval=50；
- eval 每 task 5 条；
- train/eval source episode 零泄漏；
- `relative_action_ready=true`；
- `n_obs_steps=2`、`horizon=32` 与预计算 stats 一致；
- eval absolute replay 为 50/50；
- eval chunk=15 relative oracle 为 50/50；
- GPU 空闲；
- 磁盘空间足够保存 checkpoint。
