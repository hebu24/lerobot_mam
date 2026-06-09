#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU launcher for the baseline Diffusion Policy on LIBERO put_bowl_on_plate.
# It keeps the actual training path inside lerobot.scripts.lerobot_train and only
# supplies an Accelerate launch wrapper plus task defaults.

if [[ -z "${NUM_GPUS:-}" ]]; then
  NUM_GPUS="$(uv run python -c 'import torch; print(torch.cuda.device_count())')"
fi

if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "No CUDA GPU detected. Set NUM_GPUS or run on a GPU machine." >&2
  exit 1
fi

MIXED_PRECISION="${MIXED_PRECISION:-no}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/libero_put_bowl_on_plate}"
DATASET_ROOT="${DATASET_ROOT:-outputs/datasets/libero_put_bowl_on_plate}"
JOB_NAME="${JOB_NAME:-diffusion_libero_put_bowl_on_plate_${NUM_GPUS}gpu}"
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
USE_RELATIVE_ACTIONS="${USE_RELATIVE_ACTIONS:-false}"

launch_args=(
  --num_processes="${NUM_GPUS}"
  --mixed_precision="${MIXED_PRECISION}"
)
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  launch_args=(--multi_gpu "${launch_args[@]}")
fi

train_args=(
  --policy.type=diffusion
  --policy.device="${POLICY_DEVICE}"
  --policy.push_to_hub="${PUSH_TO_HUB}"
  --policy.use_relative_actions="${USE_RELATIVE_ACTIONS}"
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

if [[ "${ENABLE_EVAL:-false}" == "true" ]]; then
  train_args+=(
    --env.type=libero
    --env.task="${ENV_TASK:-libero_goal}"
    --env.task_ids="${ENV_TASK_IDS:-[8]}"
    --env.control_mode="${ENV_CONTROL_MODE:-relative}"
    --env.observation_height="${ENV_OBSERVATION_HEIGHT:-256}"
    --env.observation_width="${ENV_OBSERVATION_WIDTH:-256}"
    --env.max_parallel_tasks="${ENV_MAX_PARALLEL_TASKS:-1}"
    --eval.batch_size="${EVAL_BATCH_SIZE:-1}"
    --eval.use_async_envs="${EVAL_USE_ASYNC_ENVS:-false}"
  )
fi

echo "Launching ${NUM_GPUS} process(es); per-GPU batch=${BATCH_SIZE}; effective batch=$((NUM_GPUS * BATCH_SIZE))"
uv run accelerate launch "${launch_args[@]}" -m lerobot.scripts.lerobot_train "${train_args[@]}" "$@"
