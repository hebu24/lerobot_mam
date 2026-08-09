#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU launcher for MAM on LIBERO datasets.
# The filename is kept for compatibility with existing local commands.
# The dataset must be materialized with scripts/convert_libero_absolute_to_mam.py
# so that mam.mas_action_absolute, mam.mas_action_mask, and mam.progress exist.

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
DATASET_REPO_ID="${DATASET_REPO_ID:-local/libero10_mam_v3_unfiltered_train}"
DATASET_ROOT="${DATASET_ROOT:-outputs/datasets/libero10_mam_v3_unfiltered_train}"
MAM_EVAL_DATASET_REPO_ID="${MAM_EVAL_DATASET_REPO_ID:-local/libero10_mam_v3_unfiltered_eval}"
MAM_EVAL_DATASET_ROOT="${MAM_EVAL_DATASET_ROOT:-outputs/datasets/libero10_mam_v3_unfiltered_eval}"
JOB_NAME="${JOB_NAME:-mam_libero10_v3_${NUM_GPUS}gpu}"
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
PRETRAINED_BACKBONE_WEIGHTS="${PRETRAINED_BACKBONE_WEIGHTS:-}"
DENOISER_TYPE="${DENOISER_TYPE:-unet}"
# U-Net-only parameters.
DOWN_DIMS="${DOWN_DIMS:-[512,1024,2048]}"
UNET_KERNEL_SIZE="${UNET_KERNEL_SIZE:-5}"
UNET_N_GROUPS="${UNET_N_GROUPS:-8}"
DIFFUSION_STEP_EMBED_DIM="${DIFFUSION_STEP_EMBED_DIM:-128}"
UNET_USE_FILM_SCALE_MODULATION="${UNET_USE_FILM_SCALE_MODULATION:-true}"
# DiT-only parameters. The DiT feed-forward dimension is 4 * DIT_HIDDEN_DIM.
DIT_HIDDEN_DIM="${DIT_HIDDEN_DIM:-512}"
DIT_NUM_LAYERS="${DIT_NUM_LAYERS:-6}"
DIT_NUM_HEADS="${DIT_NUM_HEADS:-8}"
DIT_DROPOUT="${DIT_DROPOUT:-0.1}"
DIT_TIMESTEP_EMBED_DIM="${DIT_TIMESTEP_EMBED_DIM:-256}"
DIT_USE_POSITIONAL_ENCODING="${DIT_USE_POSITIONAL_ENCODING:-false}"
DIT_USE_ROPE="${DIT_USE_ROPE:-true}"
DIT_ROPE_BASE="${DIT_ROPE_BASE:-10000.0}"

if [[ "${DENOISER_TYPE}" != "unet" && "${DENOISER_TYPE}" != "dit" ]]; then
  echo "DENOISER_TYPE must be unet or dit, got ${DENOISER_TYPE}." >&2
  exit 2
fi

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
  --policy.denoiser_type="${DENOISER_TYPE}"
  --policy.do_mask_loss_for_padding="${DO_MASK_LOSS_FOR_PADDING:-true}"
  --policy.loss_mode="${MASK_LOSS_MODE:-weighted}"
  --policy.loss_mask_area_weight="${MASK_KNOWN_REGION_WEIGHT:-0.2}"
  --policy.inpainting="${MASK_INPAINTING:-false}"
  --policy.mas_short_window_horizon="${MAS_SHORT_WINDOW_HORIZON:-15}"
  --policy.mas_long_backward_length="${MAS_LONG_BACKWARD_LENGTH:-0}"
  --policy.mas_long_forward_length="${MAS_LONG_FORWARD_LENGTH:-32}"
  --policy.mas_long_feature_dim="${MAS_LONG_FEATURE_DIM:-64}"
  --policy.mam_eval_dataset_repo_id="${MAM_EVAL_DATASET_REPO_ID}"
  --policy.mam_eval_dataset_root="${MAM_EVAL_DATASET_ROOT}"
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_ROOT}"
  --job_name="${JOB_NAME}"
  --batch_size="${BATCH_SIZE}"
  --num_workers="${NUM_WORKERS}"
  --steps="${STEPS}"
  --save_freq="${SAVE_FREQ}"
  --eval_freq="${EVAL_FREQ}"
  --log_freq="${LOG_FREQ}"
  --wandb.enable="${WANDB_ENABLE}"
)

if [[ "${DENOISER_TYPE}" == "unet" ]]; then
  train_args+=(
    --policy.down_dims="${DOWN_DIMS}"
    --policy.kernel_size="${UNET_KERNEL_SIZE}"
    --policy.n_groups="${UNET_N_GROUPS}"
    --policy.diffusion_step_embed_dim="${DIFFUSION_STEP_EMBED_DIM}"
    --policy.use_film_scale_modulation="${UNET_USE_FILM_SCALE_MODULATION}"
  )
else
  train_args+=(
    --policy.dit_hidden_dim="${DIT_HIDDEN_DIM}"
    --policy.dit_num_layers="${DIT_NUM_LAYERS}"
    --policy.dit_num_heads="${DIT_NUM_HEADS}"
    --policy.dit_dropout="${DIT_DROPOUT}"
    --policy.dit_timestep_embed_dim="${DIT_TIMESTEP_EMBED_DIM}"
    --policy.dit_use_positional_encoding="${DIT_USE_POSITIONAL_ENCODING}"
    --policy.dit_use_rope="${DIT_USE_ROPE}"
    --policy.dit_rope_base="${DIT_ROPE_BASE}"
  )
fi

if [[ "${NUM_WORKERS}" -gt 0 ]]; then
  train_args+=(
    --prefetch_factor="${PREFETCH_FACTOR}"
    --persistent_workers="${PERSISTENT_WORKERS}"
  )
else
  train_args+=(--persistent_workers=false)
fi

if [[ -n "${OUTPUT_DIR:-}" ]]; then
  train_args+=(--output_dir="${OUTPUT_DIR}")
fi

if [[ -n "${DATASET_EPISODES:-}" ]]; then
  train_args+=(--dataset.episodes="${DATASET_EPISODES}")
fi

if [[ -n "${MAM_EVAL_EPISODES:-}" ]]; then
  train_args+=(--policy.mam_eval_episodes="${MAM_EVAL_EPISODES}")
fi

if [[ -n "${STPM_PATHS:-}" ]]; then
  train_args+=(--policy.stpm_paths="${STPM_PATHS}")
fi

if [[ -n "${STPM_CHECKPOINT_PATHS:-}" ]]; then
  train_args+=(--policy.stpm_checkpoint_paths="${STPM_CHECKPOINT_PATHS}")
fi

if [[ -n "${STPM_CONFIG_PATHS:-}" ]]; then
  train_args+=(--policy.stpm_config_paths="${STPM_CONFIG_PATHS}")
fi

if [[ -n "${PRETRAINED_BACKBONE_WEIGHTS:-}" ]]; then
  train_args+=(--policy.pretrained_backbone_weights="${PRETRAINED_BACKBONE_WEIGHTS}")
fi

if [[ "${ENABLE_EVAL:-false}" == "true" ]]; then
  train_args+=(
    --env.type=libero
    --env.task="${ENV_TASK:-libero_10}"
    --env.task_ids="${ENV_TASK_IDS:-[0,1,2,3,4,5,6,7,8,9]}"
    --env.control_mode="${ENV_CONTROL_MODE:-absolute}"
    --env.observation_height="${ENV_OBSERVATION_HEIGHT:-128}"
    --env.observation_width="${ENV_OBSERVATION_WIDTH:-128}"
    --env.max_parallel_tasks="${ENV_MAX_PARALLEL_TASKS:-1}"
    --eval.n_episodes="${EVAL_N_EPISODES:-50}"
    --eval.batch_size="${EVAL_BATCH_SIZE:-1}"
    --eval.use_async_envs="${EVAL_USE_ASYNC_ENVS:-false}"
  )
fi

echo "Launching MAM with ${NUM_GPUS} process(es); denoiser=${DENOISER_TYPE}; per-GPU batch=${BATCH_SIZE}; effective batch=$((NUM_GPUS * BATCH_SIZE))"
accelerate launch "${launch_args[@]}" -m lerobot.scripts.lerobot_train "${train_args[@]}" "$@"
