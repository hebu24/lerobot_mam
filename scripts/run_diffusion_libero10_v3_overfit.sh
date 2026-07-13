#!/usr/bin/env bash
set -euo pipefail

# Strict LIBERO-10 v3 Diffusion Policy overfit.
# Usage: bash scripts/run_diffusion_libero10_v3_overfit.sh 3
#        K=5 DEMO_RANK=0 bash scripts/run_diffusion_libero10_v3_overfit.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${K:-}" && ${#} -gt 0 && "${1}" != --* ]]; then
  K="${1}"
  shift
else
  K="${K:-3}"
fi
if (( ${#} > 0 )); then
  echo "Only one optional positional argument (K) is supported; configure other knobs with environment variables." >&2
  exit 2
fi
DEMO_RANK="${DEMO_RANK:-0}"
TASK_IDS_CSV="${TASK_IDS:-}"
DATASET_ROOT="${DATASET_ROOT:-outputs/datasets/libero10_mam_v3_sample_train}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/libero10_mam_v3_sample_train}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/diffusion_libero10_v3_overfit_k${K}}"
PLAN_PATH="${PLAN_PATH:-${OUTPUT_DIR}.selection.json}"

STEPS="${STEPS:-15000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
EVAL_FREQ="${EVAL_FREQ:-1000}"
LOG_FREQ="${LOG_FREQ:-200}"
N_ACTION_STEPS="${N_ACTION_STEPS:-15}"
SEED="${SEED:-1000}"
DRY_RUN="${DRY_RUN:-false}"

if ! [[ "${K}" =~ ^[0-9]+$ ]] || (( K < 1 || K > 10 )); then
  echo "K must be an integer in [1, 10], got ${K}." >&2
  exit 2
fi
if ! [[ "${STEPS}" =~ ^[0-9]+$ ]] || (( STEPS <= 0 )); then
  echo "STEPS must be a positive integer, got ${STEPS}." >&2
  exit 2
fi
if ! [[ "${EVAL_FREQ}" =~ ^[0-9]+$ ]] || (( EVAL_FREQ <= 0 )); then
  echo "EVAL_FREQ must be a positive integer, got ${EVAL_FREQ}." >&2
  exit 2
fi
if (( STEPS < EVAL_FREQ || STEPS % EVAL_FREQ != 0 )); then
  echo "STEPS must be an EVAL_FREQ multiple so the final training step is evaluated." >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Output already exists; choose a new OUTPUT_DIR: ${OUTPUT_DIR}" >&2
  exit 2
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/.uv-cache}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/.torch-cache}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${REPO_ROOT}/.cache/numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
export LIBERO_ASSETS_PATH="${LIBERO_ASSETS_PATH:-${REPO_ROOT}/.cache/libero/assets}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ACCELERATE_MIXED_PRECISION="${MIXED_PRECISION:-fp16}"

selector=(
  uv run python scripts/prepare_libero10_v3_overfit.py
  --dataset-root="${DATASET_ROOT}"
  --dataset-repo-id="${DATASET_REPO_ID}"
  --k="${K}"
  --demo-rank="${DEMO_RANK}"
  --output-plan="${PLAN_PATH}"
)
if [[ -n "${TASK_IDS_CSV}" ]]; then
  selector+=(--task-ids="${TASK_IDS_CSV}")
fi
mapfile -t selection < <("${selector[@]}")
if (( ${#selection[@]} != 2 )); then
  echo "Trajectory selector returned an invalid response." >&2
  exit 2
fi
TASK_IDS_JSON="${selection[0]}"
EPISODES_JSON="${selection[1]}"

train_cmd=(
  uv run python -m lerobot.scripts.lerobot_train
  --policy.type=diffusion
  --policy.device=cuda
  --policy.push_to_hub=false
  --policy.use_relative_actions=true
  --policy.horizon=32
  --policy.n_action_steps="${N_ACTION_STEPS}"
  --policy.down_dims='[512,1024,2048]'
  --policy.diffusion_step_embed_dim=128
  --policy.spatial_softmax_num_keypoints=32
  --policy.use_language_conditioning=true
  --policy.do_mask_loss_for_padding=true
  --policy.pretrained_backbone_weights=null
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_ROOT}"
  --dataset.episodes="${EPISODES_JSON}"
  --env.type=libero
  --env.task=libero_10
  --env.task_ids="${TASK_IDS_JSON}"
  --env.control_mode=absolute
  --env.observation_height=128
  --env.observation_width=128
  --env.num_steps_wait=0
  --env.max_parallel_tasks=1
  --output_dir="${OUTPUT_DIR}"
  --job_name="diffusion_libero10_v3_overfit_k${K}"
  --batch_size="${BATCH_SIZE}"
  --num_workers="${NUM_WORKERS}"
  --steps="${STEPS}"
  --save_freq="${SAVE_FREQ}"
  --eval_freq="${EVAL_FREQ}"
  --log_freq="${LOG_FREQ}"
  --seed="${SEED}"
  --wandb.enable=false
  --cudnn_deterministic=true
  --overfit_test=true
  --overfit_per_task=true
  --num_overfit_per_task=1
  --eval.dataset_repo_id="${DATASET_REPO_ID}"
  --eval.dataset_root="${DATASET_ROOT}"
  --eval.dataset_episodes="${EPISODES_JSON}"
  --eval.n_episodes=1
  --eval.batch_size=1
  --eval.use_async_envs=false
)
if (( NUM_WORKERS > 0 )); then
  train_cmd+=(--prefetch_factor="${PREFETCH_FACTOR}" --persistent_workers=true)
else
  train_cmd+=(--persistent_workers=false)
fi

echo "LIBERO-10 v3 overfit: K=${K}, task_ids=${TASK_IDS_JSON}, episodes=${EPISODES_JSON}"
echo "Train/eval trajectory ids are identical; plan=${PLAN_PATH}"
if [[ "${DRY_RUN}" == "true" ]]; then
  printf '%q ' "${train_cmd[@]}"
  printf '\n'
  exit 0
fi

nvidia-smi >/dev/null
uv run python -c 'import sys, torch; print(f"CUDA={torch.cuda.is_available()} GPUs={torch.cuda.device_count()}"); sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)'
exec "${train_cmd[@]}"
