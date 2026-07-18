# Multi GPU


## 改动记录

1. 训练入口继续复用 `src/lerobot/scripts/lerobot_train.py` 的 LeRobot 标准流程，不新增训练循环。
2. 在 `lerobot_train.py` 中把 `cfg.trainable_config.device` 对齐到 `accelerator.device`，确保多卡时每个 rank 的 policy / processor 使用本 rank GPU。
3. pretrained processor override 的 `device_processor.device` 从 `cuda` 改为当前 rank 的完整设备字符串，例如 `cuda:1`。
4. 新增启动脚本：

```text
scripts/train_diffusion_libero_put_bowl_on_plate_multigpu.sh
```

该脚本用 `uv run accelerate launch -m lerobot.scripts.lerobot_train` 启动训练，只负责多卡启动和 LIBERO put_bowl_on_plate diffusion baseline 默认参数。

## 使用说明

安装依赖：

```bash
uv sync --locked --extra training --extra diffusion --extra libero
```

2 卡 baseline 训练：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NUM_GPUS=2 \
OUTPUT_DIR=outputs/train/diffusion_libero_put_bowl_on_plate_multigpu \
bash scripts/train_diffusion_libero_put_bowl_on_plate_multigpu.sh
```

默认训练数据：

```text
DATASET_REPO_ID=local/libero_put_bowl_on_plate
DATASET_ROOT=outputs/datasets/libero_put_bowl_on_plate
```

常用参数覆盖：

```bash
NUM_GPUS=4 \
BATCH_SIZE=4 \
STEPS=50000 \
SAVE_FREQ=10000 \
LOG_FREQ=200 \
MIXED_PRECISION=bf16 \
OUTPUT_DIR=outputs/train/diffusion_libero_put_bowl_on_plate_4gpu \
bash scripts/train_diffusion_libero_put_bowl_on_plate_multigpu.sh
```

说明：

- `BATCH_SIZE` 是每张 GPU 的 batch size。
- 有效 batch size = `NUM_GPUS * BATCH_SIZE`。
- LeRobot 不会自动缩放 learning rate 或 steps，需要手动调。
- 不设置 `OUTPUT_DIR` 时，LeRobot 会按默认规则生成时间戳目录。
- 继续传 CLI 参数会追加到训练命令末尾，例如：

```bash
NUM_GPUS=2 bash scripts/train_diffusion_libero_put_bowl_on_plate_multigpu.sh \
  --optimizer.lr=2e-4
```

## 可选：带在线 eval

baseline 默认 `EVAL_FREQ=0`，只做离线训练。需要 eval 时：

```bash
MUJOCO_GL=egl \
CUDA_VISIBLE_DEVICES=0,1 \
NUM_GPUS=2 \
ENABLE_EVAL=true \
EVAL_FREQ=5000 \
ENV_TASK=libero_goal \
ENV_TASK_IDS='[8]' \
ENV_CONTROL_MODE=relative \
bash scripts/train_diffusion_libero_put_bowl_on_plate_multigpu.sh
```

## 可选：relative-action 版本

如果训练 `libero_put_bowl_on_plate_absolute`，再启用 diffusion relative action：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NUM_GPUS=2 \
DATASET_REPO_ID=local/libero_put_bowl_on_plate_absolute \
DATASET_ROOT=outputs/datasets/libero_put_bowl_on_plate_absolute \
USE_RELATIVE_ACTIONS=true \
OUTPUT_DIR=outputs/train/diffusion_relative_libero_put_bowl_on_plate_multigpu \
bash scripts/train_diffusion_libero_put_bowl_on_plate_multigpu.sh
```

如果同时打开在线 eval，需额外设置 `ENV_CONTROL_MODE=absolute`。
