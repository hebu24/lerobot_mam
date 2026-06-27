#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU launcher for MAM on LIBERO put_bowl_on_plate.
# The dataset must be materialized with scripts/convert_libero_absolute_to_mam.py
# so that mam.mas_action_absolute, mam.mas_action_mask, and mam.progress exist.

if [[ -z "${NUM_GPUS:-}" ]]; then
  NUM_GPUS="$(python -c 'import torch; print(torch.cuda.device_count())')"
fi

if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "No CUDA GPU detected. Set NUM_GPUS or run on a GPU machine." >&2
  exit 1
fi

MIXED_PRECISION="${MIXED_PRECISION:-no}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/libero_put_bowl_on_plate_mam_train}"
DATASET_ROOT="${DATASET_ROOT:-outputs/datasets/libero_put_bowl_on_plate_mam_train}"
MAM_EVAL_DATASET_REPO_ID="${MAM_EVAL_DATASET_REPO_ID:-local/libero_put_bowl_on_plate_mam_eval}"
MAM_EVAL_DATASET_ROOT="${MAM_EVAL_DATASET_ROOT:-outputs/datasets/libero_put_bowl_on_plate_mam_eval}"
JOB_NAME="${JOB_NAME:-mam_libero_put_bowl_on_plate_${NUM_GPUS}gpu}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
STEPS="${STEPS:-50000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-200}"
EVAL_FREQ="${EVAL_FREQ:-0}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"

launch_args=(
  --num_processes="${NUM_GPUS}"
  --mixed_precision="${MIXED_PRECISION}"
)
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  launch_args=(--multi_gpu "${launch_args[@]}")
fi

train_args=(
  --policy.type=mam
  --policy.device="${POLICY_DEVICE}"
  --policy.push_to_hub="${PUSH_TO_HUB}"
  --policy.mam_eval_dataset_repo_id="${MAM_EVAL_DATASET_REPO_ID}"
  --policy.mam_eval_dataset_root="${MAM_EVAL_DATASET_ROOT}"
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_ROOT}"
  --job_name="${JOB_NAME}"
  --batch_size="${BATCH_SIZE}"
  --num_workers="${NUM_WORKERS}"
  --prefetch_factor="${PREFETCH_FACTOR}"
  --persistent_workers="${PERSISTENT_WORKERS}"
  --steps="${STEPS}"
  --save_freq="${SAVE_FREQ}"
  --eval_freq="${EVAL_FREQ}"
  --log_freq="${LOG_FREQ}"
  --wandb.enable="${WANDB_ENABLE}"
)

if [[ -n "${OUTPUT_DIR:-}" ]]; then
  train_args+=(--output_dir="${OUTPUT_DIR}")
fi

if [[ -n "${DATASET_EPISODES:-}" ]]; then
  train_args+=(--dataset.episodes="${DATASET_EPISODES}")
fi

if [[ -n "${MAM_EVAL_EPISODES:-}" ]]; then
  train_args+=(--policy.mam_eval_episodes="${MAM_EVAL_EPISODES}")
fi

if [[ "${ENABLE_EVAL:-false}" == "true" ]]; then
  train_args+=(
    --env.type=libero
    --env.task="${ENV_TASK:-libero_goal}"
    --env.task_ids="${ENV_TASK_IDS:-[8]}"
    --env.control_mode="${ENV_CONTROL_MODE:-absolute}"
    --env.observation_height="${ENV_OBSERVATION_HEIGHT:-256}"
    --env.observation_width="${ENV_OBSERVATION_WIDTH:-256}"
    --env.max_parallel_tasks="${ENV_MAX_PARALLEL_TASKS:-1}"
    --eval.batch_size="${EVAL_BATCH_SIZE:-1}"
    --eval.use_async_envs="${EVAL_USE_ASYNC_ENVS:-false}"
  )
fi

echo "Launching MAM with ${NUM_GPUS} process(es); per-GPU batch=${BATCH_SIZE}; effective batch=$((NUM_GPUS * BATCH_SIZE))"
accelerate launch "${launch_args[@]}" -m lerobot.scripts.lerobot_train "${train_args[@]}" "$@"
