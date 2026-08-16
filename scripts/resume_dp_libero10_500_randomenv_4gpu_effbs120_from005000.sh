#!/usr/bin/env bash
set -euo pipefail

cd /cephfs/shared/Yanbang/lerobot/mam_lerobot0.5.1/lerobot_mam

export PATH="/cephfs/shared/Yanbang/envs/lerobot0.5.1/bin:${PATH}"

# Resume the original 6 x 20 run on four A10s while preserving its effective
# global batch size: 4 x 30 = 120.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NUM_GPUS="${NUM_GPUS:-4}"
export BATCH_SIZE="${BATCH_SIZE:-30}"
export MIXED_PRECISION=fp16
export LEROBOT_DDP_TIMEOUT_S=7200
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

export DATASET_REPO_ID=local/libero10_500_train
export DATASET_ROOT=data/libero10_mam/libero10_500_train
export EVAL_ENV_MODE=random

export ENABLE_EVAL=true
export EVAL_FREQ=5000
export EVAL_N_EPISODES=5
export EVAL_BATCH_SIZE=1
export STEPS=150000
export SAVE_FREQ=10000
export LOG_FREQ=200

export REQUIRE_FULL_DATASET=false
export REQUIRE_IDLE_GPU=true
export RESUME=true

export OUTPUT_DIR=outputs/train/diffusion_libero10_500_randomenv_150k_6gpu_effbs120_20260731_192539
export RESUME_CONFIG_PATH="${OUTPUT_DIR}/checkpoints/005000/pretrained_model/train_config.json"

LOG_FILE="${LOG_FILE:-outputs/logs/diffusion_libero10_500_randomenv_150k_4gpu_effbs120_resume005000_20260801.log}"
mkdir -p "$(dirname "${LOG_FILE}")"

echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] Resuming DP from step 5000 on 4 GPUs (batch/GPU=30)." \
  | tee -a "${LOG_FILE}"

set +e
bash scripts/run_diffusion_libero10.sh --batch_size=30 2>&1 | tee -a "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e

echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] Resume process exited with status ${status}." \
  | tee -a "${LOG_FILE}"
exit "${status}"
