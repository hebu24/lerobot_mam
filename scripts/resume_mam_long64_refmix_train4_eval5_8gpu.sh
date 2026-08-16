#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CONDA_ENV_PATH="${CONDA_ENV_PATH:-/cephfs/shared/Yanbang/envs/lerobot0.5.1}"
export PATH="${CONDA_ENV_PATH}/bin:${PATH}"

# Match the mixed-mask protocol used by the long52/long70 runs. The repeated
# 3D_points entry intentionally gives dense XYZ guidance two of four train slots.
export DATASET_REPO_ID="${DATASET_REPO_ID:-local/libero10_mam_v3_refmix_train}"
export DATASET_ROOT="${DATASET_ROOT:-data/libero10_mam/libero10_mam_v3_refmix_train}"
export MAM_EVAL_DATASET_REPO_ID="${MAM_EVAL_DATASET_REPO_ID:-local/libero10_mam_v3_refmix_eval}"
export MAM_EVAL_DATASET_ROOT="${MAM_EVAL_DATASET_ROOT:-data/libero10_mam/libero10_mam_v3_refmix_eval}"
export TRAIN_MASK_TYPES=points,3D_points,3D_points,pose_motion_planning
export EVAL_MASK_TYPES=points,3D_points,3D_points,pose_motion_planning,mix0

# Keep the current long64 experiment's optimization, window, STPM, and batch
# settings fixed so this continuation changes only the materialized mask protocol.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-12}"
export MIXED_PRECISION=bf16
export NUM_WORKERS=8
export PREFETCH_FACTOR=4
export PERSISTENT_WORKERS=true
export STEPS="${STEPS:-150000}"
export EVAL_FREQ=5000
export SAVE_FREQ=10000
export LOG_FREQ=200
export ENABLE_EVAL=true
export EVAL_N_EPISODES=50
export EVAL_BATCH_SIZE=1
export EVAL_USE_ASYNC_ENVS=false
export ENV_TASK=libero_10
export ENV_TASK_IDS='[0,1,2,3,4,5,6,7,8,9]'
export ENV_CONTROL_MODE=absolute
export ENV_OBSERVATION_HEIGHT=128
export ENV_OBSERVATION_WIDTH=128
export ENV_MAX_PARALLEL_TASKS=1

export MASK_LOSS_MODE=average
export MASK_KNOWN_REGION_WEIGHT=0.2
export MASK_INPAINTING=false
export MASK_PADDING_LOSS=true
export DO_MASK_LOSS_FOR_PADDING=true
export MAS_SHORT_WINDOW_HORIZON=0
export MAS_LONG_BACKWARD_LENGTH=0
export MAS_LONG_FORWARD_LENGTH=64
export MAS_LONG_FEATURE_DIM=128

export LEARNING_RATE=1e-4
export WEIGHT_DECAY=1e-6
export WARMUP_STEPS=500
export GRAD_CLIP_NORM=10.0
export SEED=1000
export CUDNN_DETERMINISTIC=false
export PRETRAINED_BACKBONE_WEIGHTS=null
export PUSH_TO_HUB=false
export WANDB_ENABLE=false

export STPM_BASE_DIR=outputs/train
export STPM_NAME_PREFIX="${STPM_NAME_PREFIX:-stpm_libero10_v4_maniskill_d768_l8_obs6_gap2_seed42_20260729_task}"
export LIBERO_ASSETS_PATH="${LIBERO_ASSETS_PATH:-/root/.cache/libero/assets}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/scripts/libero_config}"
export SKIP_PREFLIGHT=false
export ALLOW_INDEPENDENT_EVAL_SOURCE="${ALLOW_INDEPENDENT_EVAL_SOURCE:-false}"

SOURCE_JOB=mam_libero10_500train_100eval5ptask_150k_4gpu_maniskill_short0_long64_dim128_avgmse_seed1000_20260803_173414
RESUME_STEP="${RESUME_STEP:-90000}"
step_id="$(printf '%06d' "${RESUME_STEP}")"
export RESUME_CONFIG_PATH="${RESUME_CONFIG_PATH:-outputs/train/${SOURCE_JOB}/checkpoints/${step_id}/pretrained_model/train_config.json}"
if [[ ! -f "${RESUME_CONFIG_PATH}" ]]; then
  echo "Missing resume config: ${RESUME_CONFIG_PATH}" >&2
  exit 2
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export JOB_NAME="${JOB_NAME:-mam_libero10_refmix_train4_eval5_long64_avgmse_resume${step_id}_8gpu_${RUN_ID}}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${JOB_NAME}}"

echo "MAM long64 mixed-mask continuation"
echo "  resume=${RESUME_CONFIG_PATH}"
echo "  train=${DATASET_ROOT} masks=${TRAIN_MASK_TYPES}"
echo "  eval=${MAM_EVAL_DATASET_ROOT} masks=${EVAL_MASK_TYPES}"
echo "  GPUs=${NUM_GPUS}, batch/GPU=${BATCH_SIZE}, effective_batch=$((NUM_GPUS * BATCH_SIZE))"
echo "  output=${OUTPUT_DIR}"

exec bash scripts/run_mam_libero10_conda.sh \
  --resume=true \
  --config_path="${RESUME_CONFIG_PATH}" \
  --policy.allow_independent_eval_source="${ALLOW_INDEPENDENT_EVAL_SOURCE}" \
  --policy.language_tokenizer_name=/cephfs/shared/Yanbang/maniskill/pretrained/clip-vit-base-patch32
