#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU launcher for the baseline Diffusion Policy on LIBERO datasets.
# The filename is kept for compatibility with existing local commands.
# It keeps the actual training path inside lerobot.scripts.lerobot_train and only
# supplies an Accelerate launch wrapper plus task defaults.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -z "${NUM_GPUS:-}" ]]; then
  NUM_GPUS="$(python -c 'import torch; print(torch.cuda.device_count())')"
fi

if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "No CUDA GPU detected. Set NUM_GPUS or run on a GPU machine." >&2
  exit 1
fi

MIXED_PRECISION="${MIXED_PRECISION:-no}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/libero10_mam_v3_train}"
DATASET_ROOT="${DATASET_ROOT:-outputs/datasets/libero10_mam_v3_train}"
JOB_NAME="${JOB_NAME:-diffusion_libero10_v3_${NUM_GPUS}gpu}"
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
USE_RELATIVE_ACTIONS="${USE_RELATIVE_ACTIONS:-true}"

if [[ "${POLICY_DEVICE}" == cuda* ]]; then
  python - <<'PY'
import sys

import torch

if not torch.cuda.is_available():
    print(
        "CUDA is not available to PyTorch; refusing to run a CUDA training job on CPU.",
        file=sys.stderr,
    )
    print(f"python={sys.executable}", file=sys.stderr)
    print(f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}", file=sys.stderr)
    print("Fix the NVIDIA driver / PyTorch CUDA wheel mismatch, or set POLICY_DEVICE=cpu explicitly.", file=sys.stderr)
    raise SystemExit(1)

print(f"CUDA OK: {torch.cuda.device_count()} visible GPU(s); using {torch.cuda.get_device_name(0)}")
PY
fi

launch_args=(
  --num_processes="${NUM_GPUS}"
  --num_machines=1
  --mixed_precision="${MIXED_PRECISION}"
  --dynamo_backend=no
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

if [[ -n "${OVERFIT_TEST:-}" ]]; then
  train_args+=(--overfit_test="${OVERFIT_TEST}")
fi

if [[ -n "${OVERFIT_PER_TASK:-}" ]]; then
  train_args+=(--overfit_per_task="${OVERFIT_PER_TASK}")
fi

if [[ -n "${NUM_OVERFIT:-}" ]]; then
  train_args+=(--num_overfit="${NUM_OVERFIT}")
fi

if [[ -n "${NUM_OVERFIT_PER_TASK:-}" ]]; then
  train_args+=(--num_overfit_per_task="${NUM_OVERFIT_PER_TASK}")
fi

if [[ -n "${OUTPUT_DIR:-}" ]]; then
  train_args+=(--output_dir="${OUTPUT_DIR}")
fi

if [[ -n "${DATASET_EPISODES:-}" ]]; then
  train_args+=(--dataset.episodes="${DATASET_EPISODES}")
fi

if [[ "${ENABLE_EVAL:-false}" == "true" ]]; then
  train_args+=(
    --env.type=libero
    --env.task="${ENV_TASK:-libero_10}"
    --env.task_ids="${ENV_TASK_IDS:-[0,1,2,3,4,5,6,7,8,9]}"
    --env.control_mode="${ENV_CONTROL_MODE:-absolute}"
    --env.observation_height="${ENV_OBSERVATION_HEIGHT:-128}"
    --env.observation_width="${ENV_OBSERVATION_WIDTH:-128}"
    --env.num_steps_wait="${ENV_NUM_STEPS_WAIT:-0}"
    --env.max_parallel_tasks="${ENV_MAX_PARALLEL_TASKS:-1}"
    --eval.n_episodes="${EVAL_N_EPISODES:-5}"
    --eval.batch_size="${EVAL_BATCH_SIZE:-1}"
    --eval.use_async_envs="${EVAL_USE_ASYNC_ENVS:-false}"
  )
  if [[ -n "${EVAL_DATASET_REPO_ID:-}" ]]; then
    train_args+=(--eval.dataset_repo_id="${EVAL_DATASET_REPO_ID}")
  fi
  if [[ -n "${EVAL_DATASET_ROOT:-}" ]]; then
    train_args+=(--eval.dataset_root="${EVAL_DATASET_ROOT}")
  fi
  if [[ -n "${EVAL_DATASET_EPISODES:-}" ]]; then
    train_args+=(--eval.dataset_episodes="${EVAL_DATASET_EPISODES}")
  fi
fi

echo "Launching ${NUM_GPUS} process(es); per-GPU batch=${BATCH_SIZE}; effective batch=$((NUM_GPUS * BATCH_SIZE))"
accelerate launch "${launch_args[@]}" -m lerobot.scripts.lerobot_train "${train_args[@]}" "$@"
