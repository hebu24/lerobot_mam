#!/usr/bin/env bash
set -euo pipefail

cd /cephfs/shared/Yanbang/lerobot/mam_lerobot0.5.1/lerobot_mam

export PATH="/cephfs/shared/Yanbang/envs/lerobot0.5.1/bin:${PATH}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export POLICY_DEVICE=cuda
export LEROBOT_DDP_TIMEOUT_S=7200

export DATASET_REPO_ID=local/libero10_500_train
export DATASET_ROOT=data/libero10_mam/libero10_500_train

# Updated random-env evaluation: 5 episodes per task, 50 total seeds.
export EVAL_ENV_MODE=random
export EVAL_START_SEED="${EVAL_START_SEED:-100000}"
export ENABLE_EVAL=true
export EVAL_FREQ=5000
export EVAL_N_EPISODES=5
export EVAL_BATCH_SIZE=1

# Eight A10 workers, batch 16 per rank: effective global batch 128.
export BATCH_SIZE=16
export STEPS=200000
export SAVE_FREQ=10000

export ENV_TASK=libero_10
export ENV_TASK_IDS='[0,1,2,3,4,5,6,7,8,9]'
export ENV_CONTROL_MODE=absolute
export ENV_OBSERVATION_HEIGHT=128
export ENV_OBSERVATION_WIDTH=128

export USE_RELATIVE_ACTIONS=true
export USE_LANGUAGE_CONDITIONING=true
export REQUIRE_FULL_DATASET=false
export REQUIRE_IDLE_GPU=true

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export JOB_NAME="${JOB_NAME:-diffusion_libero10_500_randomenv_200k_8gpu_effbs128_${RUN_ID}}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${JOB_NAME}}"

LOG_DIR=outputs/logs
LOG_FILE="${LOG_DIR}/${JOB_NAME}.log"
mkdir -p "${LOG_DIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] Starting ${JOB_NAME}"
echo "  train=${DATASET_ROOT}"
echo "  eval=random seeds ${EVAL_START_SEED}..$((EVAL_START_SEED + EVAL_N_EPISODES * 10 - 1)) total=${EVAL_N_EPISODES}x10"
echo "  GPUs=${NUM_GPUS} (${CUDA_VISIBLE_DEVICES}), batch/GPU=${BATCH_SIZE}, effective_batch=$((NUM_GPUS * BATCH_SIZE))"
echo "  steps=${STEPS}, eval_freq=${EVAL_FREQ}, save_freq=${SAVE_FREQ}"

set +e
bash scripts/run_diffusion_libero10.sh \
  --policy.horizon=32 \
  --policy.n_action_steps=15 \
  --policy.use_language_conditioning=true \
  --policy.language_tokenizer_name=/cephfs/shared/Yanbang/maniskill/pretrained/clip-vit-base-patch32 \
  2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e

echo "[$(date '+%Y-%m-%d %H:%M:%S %z')] ${JOB_NAME} exited with status ${status}" | tee -a "${LOG_FILE}"
exit "${status}"
