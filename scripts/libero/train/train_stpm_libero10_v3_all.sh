#!/usr/bin/env bash
set -euo pipefail

# Train one STPM progress model for each LIBERO-10 task. The output naming
# matches scripts/libero/train/run_mam_libero10_conda.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/.uv-cache}"
export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/.torch-cache}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET_REPO_ID="${DATASET_REPO_ID:-local/libero10_mam_v3_unfiltered_train}"
DATASET_ROOT="${DATASET_ROOT:-outputs/datasets/libero10_mam_v3_unfiltered_train}"
TASK_IDS="${TASK_IDS:-0,1,2,3,4,5,6,7,8,9}"
STPM_BASE_DIR="${STPM_BASE_DIR:-outputs/train}"
STPM_NAME_PREFIX="${STPM_NAME_PREFIX:-stpm_libero10_v2_task}"

N_OBS_STEPS="${N_OBS_STEPS:-1}"
FRAME_GAP="${FRAME_GAP:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
EPOCHS="${EPOCHS:-2}"
STEPS="${STEPS:-}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
VAL_RATIO="${VAL_RATIO:-0.1}"
DEVICE="${DEVICE:-cuda}"
REQUIRE_CUDA="${REQUIRE_CUDA:-true}"
VISION_CKPT="${VISION_CKPT:-/home/hebu/code/mam/ManiSkill/pretrained/clip-vit-base-patch32}"
CLIP_ENCODE_BATCH_SIZE="${CLIP_ENCODE_BATCH_SIZE:-64}"
D_MODEL="${D_MODEL:-256}"
N_LAYERS="${N_LAYERS:-2}"
N_HEADS="${N_HEADS:-4}"
DROPOUT="${DROPOUT:-0.1}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
DRY_RUN="${DRY_RUN:-false}"

for boolean_name in REQUIRE_CUDA SKIP_EXISTING DRY_RUN; do
  value="${!boolean_name}"
  if [[ "${value}" != "true" && "${value}" != "false" ]]; then
    echo "${boolean_name} must be true or false; got ${value}." >&2
    exit 2
  fi
done
if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Dataset root does not exist: ${DATASET_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${DATASET_ROOT}/meta/libero_pipeline.json" ]]; then
  echo "Missing LIBERO pipeline manifest: ${DATASET_ROOT}/meta/libero_pipeline.json" >&2
  exit 2
fi
if [[ ! -e "${VISION_CKPT}" ]]; then
  echo "Local CLIP checkpoint does not exist: ${VISION_CKPT}" >&2
  exit 2
fi
if [[ -n "${STEPS}" && -n "${EPOCHS}" ]]; then
  echo "STEPS overrides EPOCHS=${EPOCHS}." >&2
fi

IFS=',' read -r -a requested_task_ids <<< "${TASK_IDS}"
if [[ "${#requested_task_ids[@]}" -eq 0 ]]; then
  echo "TASK_IDS must contain at least one task id." >&2
  exit 2
fi

for raw_task_id in "${requested_task_ids[@]}"; do
  task_id="${raw_task_id//[[:space:]]/}"
  if [[ ! "${task_id}" =~ ^[0-9]+$ ]] || (( task_id < 0 || task_id > 9 )); then
    echo "LIBERO-10 task id must be in [0, 9]; got ${raw_task_id}." >&2
    exit 2
  fi

  output_dir="${STPM_BASE_DIR}/${STPM_NAME_PREFIX}${task_id}"
  if [[ "${SKIP_EXISTING}" == "true" \
        && -f "${output_dir}/config.yaml" \
        && -f "${output_dir}/checkpoints/reward_best.pt" ]]; then
    echo "[Skip] task ${task_id}: complete STPM already exists at ${output_dir}"
    continue
  fi

  episodes="$(
    uv run python -c '
import json
import sys

from lerobot.datasets import LeRobotDatasetMetadata

repo_id, root, task_id = sys.argv[1], sys.argv[2], int(sys.argv[3])
meta = LeRobotDatasetMetadata(repo_id, root=root)
episode_ids = [
    int(row["episode_index"])
    for row in meta.episodes
    if int(row["libero/task_id"]) == task_id
]
if not episode_ids:
    raise SystemExit(f"No episodes found for libero/task_id={task_id} in {root}.")
print(json.dumps(episode_ids, separators=(",", ":")))
' "${DATASET_REPO_ID}" "${DATASET_ROOT}" "${task_id}"
  )"

  train_cmd=(
    uv run python -m lerobot.scripts.lerobot_train_stpm
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.root="${DATASET_ROOT}"
    --output_dir="${output_dir}"
    --episodes="${episodes}"
    --n_obs_steps="${N_OBS_STEPS}"
    --frame_gap="${FRAME_GAP}"
    --batch_size="${BATCH_SIZE}"
    --num_workers="${NUM_WORKERS}"
    --prefetch_factor="${PREFETCH_FACTOR}"
    --lr="${LEARNING_RATE}"
    --val_ratio="${VAL_RATIO}"
    --device="${DEVICE}"
    --vision_ckpt="${VISION_CKPT}"
    --clip_encode_batch_size="${CLIP_ENCODE_BATCH_SIZE}"
    --d_model="${D_MODEL}"
    --n_layers="${N_LAYERS}"
    --n_heads="${N_HEADS}"
    --dropout="${DROPOUT}"
  )
  if [[ -n "${STEPS}" ]]; then
    train_cmd+=(--steps="${STEPS}")
  else
    train_cmd+=(--epochs="${EPOCHS}")
  fi
  if [[ "${REQUIRE_CUDA}" == "true" ]]; then
    train_cmd+=(--require_cuda)
  fi

  echo "[Task ${task_id}] episodes=${episodes}"
  echo "[Task ${task_id}] output=${output_dir}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf "%q " "${train_cmd[@]}"
    printf "\n"
    continue
  fi
  "${train_cmd[@]}"
done
