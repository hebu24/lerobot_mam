#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /cephfs/shared/Yanbang/envs/lerobot0.5.1

export LIBERO_ASSETS_PATH="${LIBERO_ASSETS_PATH:-/root/.cache/libero/assets}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NUM_GPUS="${NUM_GPUS:-4}"
export DATASET_REPO_ID="${DATASET_REPO_ID:-local/libero_put_bowl_on_plate_absolute}"
export DATASET_ROOT="${DATASET_ROOT:-/cephfs/shared/Yanbang/lerobot/mam_lerobot0.5.1/lerobot_mam/libero_put_bowl_on_plate_absolute}"
export USE_RELATIVE_ACTIONS="${USE_RELATIVE_ACTIONS:-true}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export STEPS="${STEPS:-50000}"
export SAVE_FREQ="${SAVE_FREQ:-2000}"
export EVAL_FREQ="${EVAL_FREQ:-2000}"
export LOG_FREQ="${LOG_FREQ:-2000}"
export ENABLE_EVAL="${ENABLE_EVAL:-true}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
export ENV_TASK="${ENV_TASK:-libero_goal}"
export ENV_TASK_IDS="${ENV_TASK_IDS:-[8]}"
export ENV_CONTROL_MODE="${ENV_CONTROL_MODE:-absolute}"
export MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/diffusion_relative_libero_put_bowl_on_plate_multigpu}"

bash scripts/train_diffusion_libero_put_bowl_on_plate_multigpu.sh \
  --policy.down_dims='[256,512,1024]' \
  --policy.diffusion_step_embed_dim=64 \
  --policy.spatial_softmax_num_keypoints=16
