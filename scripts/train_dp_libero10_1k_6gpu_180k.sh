#!/usr/bin/env bash
set -euo pipefail

# No-mask Diffusion Policy baseline for the 1k two-stage MAM experiment.
# It trains once from scratch on the same 1k trajectories and evaluates on the
# same 50 raw LIBERO init states, without loading MAS, masks, progress, or STPM.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/cephfs/shared/Yanbang/envs/lerobot0.5.1}"
export PATH="${CONDA_ENV_PATH}/bin:${PATH}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export NUM_GPUS="${NUM_GPUS:-6}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
export PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"

export DATASET_REPO_ID="${DATASET_REPO_ID:-local/libero10_1000_train}"
export DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/hf_libero10_mam/libero10_1000_train}"
export EVAL_DATASET_REPO_ID="${EVAL_DATASET_REPO_ID:-local/libero10_100first50_refmix_eval}"
export EVAL_DATASET_ROOT="${EVAL_DATASET_ROOT:-${REPO_ROOT}/data/libero10_mam/libero10_100first50_refmix_eval}"
export EVAL_ENV_MODE=fixed
export REQUIRE_FULL_DATASET=false
export ALLOW_INDEPENDENT_EVAL_SOURCE=true

export STEPS="${STEPS:-180000}"
export SAVE_FREQ="${SAVE_FREQ:-10000}"
export EVAL_FREQ="${EVAL_FREQ:-5000}"
export LOG_FREQ="${LOG_FREQ:-200}"
export ENABLE_EVAL=true
# Generic DP evaluation interprets this value per task: 5/task = 50 total.
export EVAL_N_EPISODES="${EVAL_N_EPISODES:-5}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"

export N_OBS_STEPS="${N_OBS_STEPS:-2}"
export HORIZON="${HORIZON:-32}"
export N_ACTION_STEPS="${N_ACTION_STEPS:-15}"
export DENOISER_TYPE="${DENOISER_TYPE:-unet}"
export DOWN_DIMS="${DOWN_DIMS:-[512,1024,2048]}"
export SPATIAL_SOFTMAX_NUM_KEYPOINTS="${SPATIAL_SOFTMAX_NUM_KEYPOINTS:-32}"
export USE_LANGUAGE_CONDITIONING=true
export PRETRAINED_BACKBONE_WEIGHTS=null
export DO_MASK_LOSS_FOR_PADDING=true

export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
export WARMUP_STEPS="${WARMUP_STEPS:-500}"
export SEED="${SEED:-1000}"
export CUDNN_DETERMINISTIC=false
export PUSH_TO_HUB=false
export WANDB_ENABLE=false

export ENV_TASK_IDS='[0,1,2,3,4,5,6,7,8,9]'
export ENV_MAX_PARALLEL_TASKS=1
export ENV_OBSERVATION_HEIGHT=128
export ENV_OBSERVATION_WIDTH=128

export REQUIRE_IDLE_GPU="${REQUIRE_IDLE_GPU:-true}"
export MIN_FREE_GB="${MIN_FREE_GB:-20}"
export DRY_RUN="${DRY_RUN:-false}"

EXPECTED_HOST="${EXPECTED_HOST:-zhangchenyu3-0}"
EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-6}"
EXPECTED_GPU_NAME="${EXPECTED_GPU_NAME:-A10}"
if [[ "$(hostname)" != "${EXPECTED_HOST}" && "${ALLOW_OTHER_HOST:-false}" != "true" ]]; then
  echo "Expected host ${EXPECTED_HOST}; got $(hostname)." >&2
  exit 2
fi
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if (( ${#gpu_names[@]} != EXPECTED_GPU_COUNT )); then
  echo "Expected ${EXPECTED_GPU_COUNT} GPUs, found ${#gpu_names[@]}." >&2
  exit 2
fi
for gpu_name in "${gpu_names[@]}"; do
  if [[ "${gpu_name}" != *"${EXPECTED_GPU_NAME}"* ]]; then
    echo "Expected GPU names containing '${EXPECTED_GPU_NAME}', found: ${gpu_names[*]}" >&2
    exit 2
  fi
done

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export JOB_NAME="${JOB_NAME:-diffusion_libero10_1k_nomask_180k_6a10_${RUN_ID}}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${JOB_NAME}}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${JOB_NAME}.log}"
mkdir -p "${LOG_DIR}"

{
  echo "DP baseline protocol"
  echo "host=$(hostname) GPUs=${gpu_names[*]}"
  echo "train=${DATASET_ROOT} (1000 episodes; mam.* columns ignored)"
  echo "eval=${EVAL_DATASET_ROOT} (5 fixed init states/task; no MAS/mask/progress/STPM)"
  echo "steps=${STEPS} batch_per_gpu=${BATCH_SIZE} global_batch=$((NUM_GPUS * BATCH_SIZE))"
  echo "save_freq=${SAVE_FREQ} keep_all_checkpoints_after_step=100000 eval_freq=${EVAL_FREQ}"
  echo "output=${OUTPUT_DIR}"
} | tee "${LOG_FILE}"

set +e
bash scripts/libero/train/run_diffusion_libero10.sh \
  --keep_all_checkpoints_after_step=100000 \
  --optimizer.grad_clip_norm=10.0 \
  --policy.language_tokenizer_name=/cephfs/shared/Yanbang/maniskill/pretrained/clip-vit-base-patch32 \
  2>&1 | tee -a "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e

exit "${status}"
