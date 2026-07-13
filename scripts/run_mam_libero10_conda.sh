#!/usr/bin/env bash
set -euo pipefail

if [[ -f "${CONDA_PREFIX:-}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_PREFIX}/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "Could not find conda.sh. Activate the lerobot env before running this script." >&2
  exit 1
fi

if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
  conda activate "${CONDA_ENV_PATH}"
elif [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV_NAME:-lerobot}" ]]; then
  true
else
  conda activate "${CONDA_ENV_NAME:-lerobot}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export LIBERO_ASSETS_PATH="${LIBERO_ASSETS_PATH:-${REPO_ROOT}/.cache/libero/assets}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/.torch-cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_GPUS="${NUM_GPUS:-1}"

export DATASET_REPO_ID="${DATASET_REPO_ID:-local/libero10_mam_v3_train}"
export DATASET_ROOT="${DATASET_ROOT:-outputs/datasets/libero10_mam_v3_train}"
export MAM_EVAL_DATASET_REPO_ID="${MAM_EVAL_DATASET_REPO_ID:-local/libero10_mam_v3_eval}"
export MAM_EVAL_DATASET_ROOT="${MAM_EVAL_DATASET_ROOT:-outputs/datasets/libero10_mam_v3_eval}"

export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export STEPS="${STEPS:-10000}"
export SAVE_FREQ="${SAVE_FREQ:-1000}"
export EVAL_FREQ="${EVAL_FREQ:-1000}"
export LOG_FREQ="${LOG_FREQ:-200}"
export ENABLE_EVAL="${ENABLE_EVAL:-true}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
# MAM eval interprets n_episodes globally. Use five fixed episodes per LIBERO-10 task.
export EVAL_N_EPISODES="${EVAL_N_EPISODES:-50}"
export ENV_TASK="${ENV_TASK:-libero_10}"
export ENV_TASK_IDS="${ENV_TASK_IDS:-[0,1,2,3,4,5,6,7,8,9]}"
export ENV_CONTROL_MODE="${ENV_CONTROL_MODE:-absolute}"
export ENV_OBSERVATION_HEIGHT="${ENV_OBSERVATION_HEIGHT:-128}"
export ENV_OBSERVATION_WIDTH="${ENV_OBSERVATION_WIDTH:-128}"
export MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
export JOB_NAME="${JOB_NAME:-mam_libero10_${NUM_GPUS}gpu}"
export OUTPUT_DIR="${OUTPUT_DIR:-}"
export PRETRAINED_BACKBONE_WEIGHTS="${PRETRAINED_BACKBONE_WEIGHTS:-null}"

bash scripts/train_mam_libero_put_bowl_on_plate_multigpu.sh \
  --policy.horizon=32 \
  --policy.n_action_steps=15 \
  --policy.down_dims='[512,1024,2048]' \
  --policy.diffusion_step_embed_dim=128 \
  --policy.spatial_softmax_num_keypoints=32 \
  --policy.use_language_conditioning=true \
  "$@"
