#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CONDA_ENV_PATH="${CONDA_ENV_PATH:-/cephfs/shared/Yanbang/envs/lerobot0.5.1}"
export PATH="${CONDA_ENV_PATH}/bin:${PATH}"

# Match the previous 500-train / independent-50-eval scratch experiment, but
# evaluate every checkpoint with the LIBERO v3 STPM and train through 200k.
export DATASET_REPO_ID=local/libero10_500_refmix_train
export DATASET_ROOT="${DATASET_ROOT:-data/libero10_mam/libero10_500_refmix_train}"
export MAM_EVAL_DATASET_REPO_ID=local/libero10_100first50_refmix_eval
export MAM_EVAL_DATASET_ROOT="${MAM_EVAL_DATASET_ROOT:-data/libero10_mam/libero10_100first50_refmix_eval}"
export TRAIN_MASK_TYPES=points,3D_points,3D_points,pose_motion_planning
export EVAL_MASK_TYPES=points,3D_points,3D_points,pose_motion_planning,mix0

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-12}"
export MIXED_PRECISION=bf16
export NUM_WORKERS=8
export PREFETCH_FACTOR=4
export PERSISTENT_WORKERS=true
export STEPS="${STEPS:-200000}"
export EVAL_FREQ=5000
export SAVE_FREQ=10000
export KEEP_ALL_CHECKPOINTS_AFTER_STEP="${KEEP_ALL_CHECKPOINTS_AFTER_STEP:-100000}"
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
export ENV_EPISODE_LENGTH="${ENV_EPISODE_LENGTH:-520}"

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
export STPM_NAME_PREFIX=stpm_libero10_v3_large_d512_l4_task
export LIBERO_ASSETS_PATH="${LIBERO_ASSETS_PATH:-/root/.cache/libero/assets}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/scripts/libero_config}"
export SKIP_PREFLIGHT=false

if [[ "${NUM_GPUS}" -ne 8 ]]; then
  echo "This experiment requires NUM_GPUS=8, got ${NUM_GPUS}." >&2
  exit 2
fi
if [[ $((NUM_GPUS * BATCH_SIZE)) -ne 96 ]]; then
  echo "Expected effective batch size 96, got $((NUM_GPUS * BATCH_SIZE))." >&2
  exit 2
fi
if [[ "${STEPS}" -ne 200000 ]]; then
  echo "This experiment must end at step 200000, got ${STEPS}." >&2
  exit 2
fi
if [[ "${KEEP_ALL_CHECKPOINTS_AFTER_STEP}" -ne 100000 ]]; then
  echo "This experiment must retain scheduled checkpoints from step 100000, got ${KEEP_ALL_CHECKPOINTS_AFTER_STEP}." >&2
  exit 2
fi
if [[ "${ENV_EPISODE_LENGTH}" -ne 520 ]]; then
  echo "Fixed comparison protocol requires ENV_EPISODE_LENGTH=520, got ${ENV_EPISODE_LENGTH}." >&2
  exit 2
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export JOB_NAME="${JOB_NAME:-mam_libero10_500train_100first50eval_refmix_train4_eval5_long64_avgmse_stpmv3_scratch_8gpu_200k_${RUN_ID}}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${JOB_NAME}}"

echo "MAM long64 v3-STPM training from scratch"
echo "  train=${DATASET_ROOT} masks=${TRAIN_MASK_TYPES}"
echo "  eval=${MAM_EVAL_DATASET_ROOT} masks=${EVAL_MASK_TYPES}"
echo "  stpm=${STPM_BASE_DIR}/${STPM_NAME_PREFIX}{0..9}/checkpoints/reward_best.pt"
echo "  GPUs=${NUM_GPUS}, batch/GPU=${BATCH_SIZE}, effective_batch=$((NUM_GPUS * BATCH_SIZE))"
echo "  steps=${STEPS}, eval_freq=${EVAL_FREQ}, save_freq=${SAVE_FREQ}"
echo "  keep_all_checkpoints_after_step=${KEEP_ALL_CHECKPOINTS_AFTER_STEP}, episode_length=${ENV_EPISODE_LENGTH}"
echo "  output=${OUTPUT_DIR}"

exec bash scripts/run_mam_libero10_conda.sh \
  --keep_all_checkpoints_after_step="${KEEP_ALL_CHECKPOINTS_AFTER_STEP}" \
  --env.episode_length="${ENV_EPISODE_LENGTH}" \
  --policy.allow_independent_eval_source=true \
  --policy.language_tokenizer_name=/cephfs/shared/Yanbang/maniskill/pretrained/clip-vit-base-patch32
