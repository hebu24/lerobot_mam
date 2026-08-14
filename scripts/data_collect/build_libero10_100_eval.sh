#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CHECKPOINT="${CHECKPOINT:-outputs/checkpoints/dp_libero10_v3_filtered_best_sr94_step095000/pretrained_model}"
RAW_ROOT="${RAW_ROOT:-outputs/datasets/libero10_100_rollout_absolute_lpb}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/datasets/libero10_100_lpb}"
FINAL_ROOT="${FINAL_ROOT:-outputs/datasets/libero10_100_eval_lpb}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
START_SEED="${START_SEED:-${SEED:-100000}}"
MAX_ATTEMPTS_PER_TASK="${MAX_ATTEMPTS_PER_TASK:-1000}"
UPLOAD="${UPLOAD:-false}"
TOTAL_EPISODES=$((10 * EPISODES_PER_TASK))
EVAL_EPISODE_IDS="$(seq -s, 0 $((TOTAL_EPISODES - 1)))"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/.uv-cache}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export LIBERO_ASSETS_PATH="${LIBERO_ASSETS_PATH:-${REPO_ROOT}/.cache/libero/assets}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

uv run python scripts/data_collect/collect_libero10_dp_rollouts.py \
  --checkpoint "${CHECKPOINT}" \
  --output-root "${RAW_ROOT}" \
  --episodes-per-task "${EPISODES_PER_TASK}" \
  --max-attempts-per-task "${MAX_ATTEMPTS_PER_TASK}" \
  --batch-size "${BATCH_SIZE}" \
  --start-seed "${START_SEED}" \
  --output-repo-id local/libero10_100_rollout_absolute_lpb

uv run python scripts/libero/data/convert_libero_absolute_to_mam.py \
  --input-root "${RAW_ROOT}" \
  --input-repo-id local/libero10_100_rollout_absolute_lpb \
  --output-root "${OUTPUT_BASE}" \
  --output-repo-id local/libero10_100_lpb \
  --eval-output-root "${FINAL_ROOT}" \
  --eval-output-repo-id local/libero10_100_eval_lpb \
  --eval-episode-ids "${EVAL_EPISODE_IDS}" \
  --eval-per-task "${EPISODES_PER_TASK}" \
  --only-split eval \
  --eval-mask-types random_mask \
  --retain-ratio 0.2

uv run python scripts/data_collect/validate_libero10_mam_eval.py \
  --root "${FINAL_ROOT}" \
  --episodes-per-task "${EPISODES_PER_TASK}" \
  --min-rollout-seed "${START_SEED}"

if [[ "${UPLOAD}" == "true" ]]; then
  HF_HOME="${UPLOAD_HF_HOME:-${XDG_CACHE_HOME:-${HOME}/.cache}/huggingface}" \
    HF_HUB_OFFLINE=0 uv run python scripts/data_collect/upload_libero10_100_eval.py \
    --root "${FINAL_ROOT}" \
    --path-in-repo libero10_100_eval_lpb
elif [[ "${UPLOAD}" != "false" ]]; then
  echo "UPLOAD must be true or false, got: ${UPLOAD}" >&2
  exit 2
fi
