#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CHECKPOINT="${CHECKPOINT:-outputs/checkpoints/dp_libero10_v3_filtered_best_sr94_step095000/pretrained_model}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-outputs/datasets/libero10_500_train}"
RAW_ROOT="${RAW_ROOT:-outputs/datasets/libero10_500_rollout_absolute_train}"
ROLLOUT_MAM_BASE="${ROLLOUT_MAM_BASE:-outputs/datasets/libero10_500_rollout_mam}"
ROLLOUT_MAM_ROOT="${ROLLOUT_MAM_ROOT:-${ROLLOUT_MAM_BASE}_train}"
FINAL_ROOT="${FINAL_ROOT:-outputs/datasets/libero10_1000_train}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
START_SEED="${START_SEED:-50}"
MAX_ATTEMPTS_PER_TASK="${MAX_ATTEMPTS_PER_TASK:-10000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
UPLOAD="${UPLOAD:-false}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/.uv-cache}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export LIBERO_ASSETS_PATH="${LIBERO_ASSETS_PATH:-${REPO_ROOT}/.cache/libero/assets}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

if ! rg -q '"total_episodes": 500' "${RAW_ROOT}/meta/info.json" 2>/dev/null; then
  uv run python scripts/data_collect/collect_libero10_dp_rollouts.py \
    --checkpoint "${CHECKPOINT}" \
    --output-root "${RAW_ROOT}" \
    --output-repo-id local/libero10_500_rollout_absolute_train \
    --reference-length-root "${OFFICIAL_ROOT}" \
    --episodes-per-task "${EPISODES_PER_TASK}" \
    --max-attempts-per-task "${MAX_ATTEMPTS_PER_TASK}" \
    --batch-size "${BATCH_SIZE}" \
    --start-seed "${START_SEED}"
fi

uv run python scripts/libero/data/convert_libero_absolute_to_mam.py \
  --input-root "${RAW_ROOT}" \
  --input-repo-id local/libero10_500_rollout_absolute_train \
  --output-root "${ROLLOUT_MAM_BASE}" \
  --output-repo-id local/libero10_500_rollout_mam \
  --eval-ratio 0 --only-split train \
  --train-mask-types random_mask --retain-ratio 0.2 \
  --overwrite

uv run python scripts/data_collect/merge_libero10_train.py \
  --official-root "${OFFICIAL_ROOT}" \
  --official-repo-id local/libero10_500_train \
  --rollout-root "${ROLLOUT_MAM_ROOT}" \
  --rollout-repo-id local/libero10_500_rollout_mam_train \
  --output-root "${FINAL_ROOT}" \
  --output-repo-id local/libero10_1000_train

if [[ "${UPLOAD}" == "true" ]]; then
  HF_HOME="${UPLOAD_HF_HOME:-${XDG_CACHE_HOME:-${HOME}/.cache}/huggingface}" \
    HF_HUB_OFFLINE=0 uv run python scripts/data_collect/upload_libero10_100_eval.py \
    --root "${FINAL_ROOT}" \
    --path-in-repo libero10_1000_train \
    --commit-message "Add 500 successful seeded DP rollouts to LIBERO-10 training data"
elif [[ "${UPLOAD}" != "false" ]]; then
  echo "UPLOAD must be true or false, got: ${UPLOAD}" >&2
  exit 2
fi
