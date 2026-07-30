#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PATH="/cephfs/shared/Yanbang/envs/lerobot0.5.1/bin:${PATH}"
PYTHON="/cephfs/shared/Yanbang/envs/lerobot0.5.1/bin/python"

TRAIN_SOURCE_ROOT="outputs/datasets/libero10_mam_v3_unfiltered_train"
EVAL_SOURCE_ROOT="outputs/datasets/libero10_mam_v3_unfiltered_eval"
TRAIN_ROOT="data/libero10_mam/libero10_mam_v3_refmix_train"
EVAL_ROOT="data/libero10_mam/libero10_mam_v3_refmix_eval"
TRAIN_REPO_ID="local/libero10_mam_v3_refmix_train"
EVAL_REPO_ID="local/libero10_mam_v3_refmix_eval"

TRAIN_MASK_TYPES="points,3D_points,3D_points,pose_motion_planning"
TRAIN_RETAIN_RATIOS="1,1,0.2,0.2"
TRAIN_MASK_COMPOSITION="0.25,0.25,0.25,0.25"
EVAL_MASK_TYPES="points,3D_points,3D_points,pose_motion_planning,mix0"
EVAL_RETAIN_RATIOS="1,1,0.2,0.2,1"
EVAL_MASK_COMPOSITION="0.2,0.2,0.2,0.2,0.2"

mkdir -p "$(dirname "${TRAIN_ROOT}")" outputs/logs

manifest_uses_per_task_composition() {
  local manifest_path="$1"
  [[ -f "${manifest_path}" ]] && "${PYTHON}" -c '
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(
    0
    if manifest.get("mask_assign_mode") == "composition"
    and manifest.get("mask_composition_scope") == "per_task"
    else 1
)
' "${manifest_path}"
}

if ! manifest_uses_per_task_composition "${TRAIN_ROOT}/meta/libero_pipeline.json"; then
  train_overwrite_args=()
  if [[ -e "${TRAIN_ROOT}" ]]; then
    train_overwrite_args+=(--overwrite)
  fi
  "${PYTHON}" scripts/convert_libero_absolute_to_mam.py \
    --input-root="${TRAIN_SOURCE_ROOT}" \
    --input-repo-id=local/libero10_mam_v3_unfiltered_train \
    --output-root="${TRAIN_ROOT}" \
    --output-repo-id="${TRAIN_REPO_ID}" \
    --remask-existing-split \
    --train-mask-types="${TRAIN_MASK_TYPES}" \
    --train-retain-ratios="${TRAIN_RETAIN_RATIOS}" \
    --train-mask-assign-mode=composition \
    --train-mask-composition="${TRAIN_MASK_COMPOSITION}" \
    --n-obs-steps=2 \
    --horizon=32 \
    "${train_overwrite_args[@]}"
fi

if ! manifest_uses_per_task_composition "${EVAL_ROOT}/meta/libero_pipeline.json"; then
  eval_overwrite_args=()
  if [[ -e "${EVAL_ROOT}" ]]; then
    eval_overwrite_args+=(--overwrite)
  fi
  "${PYTHON}" scripts/convert_libero_absolute_to_mam.py \
    --input-root="${EVAL_SOURCE_ROOT}" \
    --input-repo-id=local/libero10_mam_v3_unfiltered_eval \
    --output-root="${EVAL_ROOT}" \
    --output-repo-id="${EVAL_REPO_ID}" \
    --remask-existing-split \
    --eval-mask-types="${EVAL_MASK_TYPES}" \
    --eval-retain-ratios="${EVAL_RETAIN_RATIOS}" \
    --eval-mask-assign-mode=composition \
    --eval-mask-composition="${EVAL_MASK_COMPOSITION}" \
    --n-obs-steps=2 \
    --horizon=32 \
    "${eval_overwrite_args[@]}"
fi

export CONDA_PREFIX=/root/miniconda3
export CONDA_ENV_PATH=/cephfs/shared/Yanbang/envs/lerobot0.5.1
export LIBERO_ASSETS_PATH=/root/.cache/libero/assets
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export NUM_GPUS=6

export DATASET_REPO_ID="${TRAIN_REPO_ID}"
export DATASET_ROOT="${TRAIN_ROOT}"
export MAM_EVAL_DATASET_REPO_ID="${EVAL_REPO_ID}"
export MAM_EVAL_DATASET_ROOT="${EVAL_ROOT}"
export TRAIN_MASK_TYPES
export EVAL_MASK_TYPES
export STPM_BASE_DIR=outputs/train
export STPM_NAME_PREFIX=stpm_libero10_v3_large_d512_l4_task

export STEPS=150000
export BATCH_SIZE=16
export NUM_WORKERS=8
export PREFETCH_FACTOR=4
export PERSISTENT_WORKERS=true
export MIXED_PRECISION=bf16

export ENABLE_EVAL=true
export EVAL_FREQ=5000
export SAVE_FREQ=10000
export LOG_FREQ=200
export EVAL_N_EPISODES=50
export EVAL_BATCH_SIZE=1
export EVAL_USE_ASYNC_ENVS=false
export ENV_TASK=libero_10
export ENV_TASK_IDS='[0,1,2,3,4,5,6,7,8,9]'
export ENV_CONTROL_MODE=absolute
export ENV_OBSERVATION_HEIGHT=128
export ENV_OBSERVATION_WIDTH=128
export ENV_MAX_PARALLEL_TASKS=1

export LEARNING_RATE=1e-4
export WEIGHT_DECAY=1e-6
export WARMUP_STEPS=500
export GRAD_CLIP_NORM=10.0
export MASK_LOSS_MODE=weighted
export MASK_KNOWN_REGION_WEIGHT=0.2
export MASK_INPAINTING=false
export MASK_PADDING_LOSS=true
export DO_MASK_LOSS_FOR_PADDING=true
export PRETRAINED_BACKBONE_WEIGHTS=null
export PUSH_TO_HUB=false
export WANDB_ENABLE=false

export JOB_NAME="${JOB_NAME:?JOB_NAME must be set by the tmux launcher}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${JOB_NAME}}"

exec bash scripts/run_mam_libero10_conda.sh \
  --policy.language_tokenizer_name=/cephfs/shared/Yanbang/maniskill/pretrained/clip-vit-base-patch32
