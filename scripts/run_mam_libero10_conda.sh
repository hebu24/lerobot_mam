#!/usr/bin/env bash
set -euo pipefail

# Formal LIBERO-10 MAM launcher. Environment variables may override every
# important dataset, mask, optimization, STPM, and evaluation setting below.

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
cd "${REPO_ROOT}"

export LIBERO_ASSETS_PATH="${LIBERO_ASSETS_PATH:-${REPO_ROOT}/.cache/libero/assets}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/.uv-cache}"
export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/.torch-cache}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${REPO_ROOT}/.cache/numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_GPUS="${NUM_GPUS:-1}"

export DATASET_REPO_ID="${DATASET_REPO_ID:-local/libero10_mam_v3_unfiltered_train}"
export DATASET_ROOT="${DATASET_ROOT:-outputs/datasets/libero10_mam_v3_unfiltered_train}"
export MAM_EVAL_DATASET_REPO_ID="${MAM_EVAL_DATASET_REPO_ID:-local/libero10_mam_v3_unfiltered_eval}"
export MAM_EVAL_DATASET_ROOT="${MAM_EVAL_DATASET_ROOT:-outputs/datasets/libero10_mam_v3_unfiltered_eval}"

# Mask/MAM contract. MASK_TYPES accepts a comma-separated mixed-mask list and is
# validated against the materialized dataset. MASK_TYPE remains the single-mask
# compatibility default.
export MASK_TYPE="${MASK_TYPE:-random_mask}"
export MASK_TYPES="${MASK_TYPES:-${MASK_TYPE}}"
export TRAIN_MASK_TYPES="${TRAIN_MASK_TYPES:-${MASK_TYPES}}"
export EVAL_MASK_TYPES="${EVAL_MASK_TYPES:-${TRAIN_MASK_TYPES}}"
export MASK_LOSS_MODE="${MASK_LOSS_MODE:-weighted}"
export MASK_KNOWN_REGION_WEIGHT="${MASK_KNOWN_REGION_WEIGHT:-0.2}"
export MASK_INPAINTING="${MASK_INPAINTING:-false}"
export MASK_PADDING_LOSS="${MASK_PADDING_LOSS:-true}"
export DO_MASK_LOSS_FOR_PADDING="${DO_MASK_LOSS_FOR_PADDING:-${MASK_PADDING_LOSS}}"
export MAS_SHORT_WINDOW_HORIZON="${MAS_SHORT_WINDOW_HORIZON:-15}"
export MAS_LONG_BACKWARD_LENGTH="${MAS_LONG_BACKWARD_LENGTH:-0}"
export MAS_LONG_FORWARD_LENGTH="${MAS_LONG_FORWARD_LENGTH:-32}"
export MAS_LONG_FEATURE_DIM="${MAS_LONG_FEATURE_DIM:-64}"

export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
export PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
export STEPS="${STEPS:-50000}"
export SAVE_FREQ="${SAVE_FREQ:-5000}"
export EVAL_FREQ="${EVAL_FREQ:-5000}"
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
export ENV_MAX_PARALLEL_TASKS="${ENV_MAX_PARALLEL_TASKS:-1}"
export EVAL_USE_ASYNC_ENVS="${EVAL_USE_ASYNC_ENVS:-false}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
export JOB_NAME="${JOB_NAME:-mam_libero10_v3_unfiltered_${NUM_GPUS}gpu}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${JOB_NAME}}"
export PRETRAINED_BACKBONE_WEIGHTS="${PRETRAINED_BACKBONE_WEIGHTS:-null}"
export PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
export WANDB_ENABLE="${WANDB_ENABLE:-false}"

# Action denoiser. U-Net and DiT use separate architecture parameters.
export DENOISER_TYPE="${DENOISER_TYPE:-unet}"
# U-Net-only parameters.
export DOWN_DIMS="${DOWN_DIMS:-[512,1024,2048]}"
export UNET_KERNEL_SIZE="${UNET_KERNEL_SIZE:-5}"
export UNET_N_GROUPS="${UNET_N_GROUPS:-8}"
export DIFFUSION_STEP_EMBED_DIM="${DIFFUSION_STEP_EMBED_DIM:-128}"
export UNET_USE_FILM_SCALE_MODULATION="${UNET_USE_FILM_SCALE_MODULATION:-true}"
# DiT-only parameters. The DiT feed-forward dimension is 4 * DIT_HIDDEN_DIM.
export DIT_HIDDEN_DIM="${DIT_HIDDEN_DIM:-512}"
export DIT_NUM_LAYERS="${DIT_NUM_LAYERS:-6}"
export DIT_NUM_HEADS="${DIT_NUM_HEADS:-8}"
export DIT_DROPOUT="${DIT_DROPOUT:-0.1}"
export DIT_TIMESTEP_EMBED_DIM="${DIT_TIMESTEP_EMBED_DIM:-256}"
export DIT_USE_POSITIONAL_ENCODING="${DIT_USE_POSITIONAL_ENCODING:-false}"
export DIT_USE_ROPE="${DIT_USE_ROPE:-true}"
export DIT_ROPE_BASE="${DIT_ROPE_BASE:-10000.0}"

export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
export WARMUP_STEPS="${WARMUP_STEPS:-500}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-10.0}"
export SEED="${SEED:-1000}"
export CUDNN_DETERMINISTIC="${CUDNN_DETERMINISTIC:-false}"

export STPM_BASE_DIR="${STPM_BASE_DIR:-outputs/train}"
export STPM_NAME_PREFIX="${STPM_NAME_PREFIX:-stpm_libero10_v2_task}"
if [[ -z "${STPM_PATHS:-}" ]]; then
  STPM_PATHS="{"
  for task_id in 0 1 2 3 4 5 6 7 8 9; do
    if [[ "${STPM_PATHS}" != "{" ]]; then
      STPM_PATHS+=","
    fi
    STPM_PATHS+="\"libero_10/${task_id}\":\"${STPM_BASE_DIR}/${STPM_NAME_PREFIX}${task_id}\""
  done
  STPM_PATHS+="}"
fi
export STPM_PATHS

export SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-false}"
export DRY_RUN="${DRY_RUN:-false}"

for boolean_name in \
  ENABLE_EVAL MASK_INPAINTING MASK_PADDING_LOSS DO_MASK_LOSS_FOR_PADDING \
  PUSH_TO_HUB WANDB_ENABLE CUDNN_DETERMINISTIC \
  UNET_USE_FILM_SCALE_MODULATION DIT_USE_POSITIONAL_ENCODING DIT_USE_ROPE \
  SKIP_PREFLIGHT DRY_RUN; do
  value="${!boolean_name}"
  if [[ "${value}" != "true" && "${value}" != "false" ]]; then
    echo "${boolean_name} must be true or false; got ${value}." >&2
    exit 2
  fi
done
if [[ "${DENOISER_TYPE}" != "unet" && "${DENOISER_TYPE}" != "dit" ]]; then
  echo "DENOISER_TYPE must be unet or dit; got ${DENOISER_TYPE}." >&2
  exit 2
fi
if [[ "${MASK_LOSS_MODE}" != "average" && "${MASK_LOSS_MODE}" != "weighted" ]]; then
  echo "MASK_LOSS_MODE must be average or weighted; got ${MASK_LOSS_MODE}." >&2
  exit 2
fi
if [[ "${ENV_CONTROL_MODE}" != "absolute" ]]; then
  echo "MAM relative-action decoding requires ENV_CONTROL_MODE=absolute." >&2
  exit 2
fi

if [[ "${SKIP_PREFLIGHT}" != "true" ]]; then
  python -c '
import json
import sys
from pathlib import Path

train_root, eval_root = map(Path, sys.argv[1:3])
train_mask_types = [item.strip() for item in sys.argv[3].split(",") if item.strip()]
eval_mask_types = [item.strip() for item in sys.argv[4].split(",") if item.strip()]
if not train_mask_types or not eval_mask_types:
    raise SystemExit("TRAIN_MASK_TYPES and EVAL_MASK_TYPES must each contain at least one mask type")
expected_mask_types = {"train": train_mask_types, "eval": eval_mask_types}
expected_indices = list(range(-1, 31))
payloads = {}
for split, root in (("train", train_root), ("eval", eval_root)):
    path = root / "meta" / "libero_pipeline.json"
    if not path.is_file():
        raise SystemExit(f"missing dataset manifest: {path}")
    data = json.loads(path.read_text())
    expected = {
        "stage": "absolute_to_mam",
        "dataset_split": split,
        "action_representation": "osc_pose_absolute_goal",
        "policy_action_representation": "chunk_relative_se3",
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise SystemExit(f"{path}: expected {key}={value!r}, got {data.get(key)!r}")
    if data.get("relative_action_ready") is not True:
        raise SystemExit(f"{path}: relative_action_ready must be true")
    if data.get("relative_action_stats_action_delta_indices") != expected_indices:
        raise SystemExit(f"{path}: relative action stats do not match n_obs_steps=2/horizon=32")
    if data.get("mask_types") != expected_mask_types[split]:
        raise SystemExit(
            f"{path}: expected mask_types={expected_mask_types[split]!r}, "
            f"got {data.get('mask_types')!r}"
        )
    if (
        data.get("mask_assign_mode") == "composition"
        and data.get("mask_composition_scope") != "per_task"
    ):
        raise SystemExit(
            f"{path}: composition masks must be assigned per task; "
            f"got mask_composition_scope={data.get('mask_composition_scope')!r}"
        )
    payloads[split] = data
overlap = set(payloads["train"].get("source_episode_ids", [])) & set(
    payloads["eval"].get("source_episode_ids", [])
)
if overlap:
    raise SystemExit(f"train/eval source leakage: {sorted(overlap)}")
' "${DATASET_ROOT}" "${MAM_EVAL_DATASET_ROOT}" "${TRAIN_MASK_TYPES}" "${EVAL_MASK_TYPES}"

  if [[ "${ENABLE_EVAL}" == "true" ]]; then
    python -c '
import json
import sys
from pathlib import Path

mapping = json.loads(sys.argv[1])
missing_keys = [f"libero_10/{task_id}" for task_id in range(10) if f"libero_10/{task_id}" not in mapping]
if missing_keys:
    raise SystemExit(f"STPM_PATHS is missing keys: {missing_keys}")
missing_files = []
for key in (f"libero_10/{task_id}" for task_id in range(10)):
    root = Path(mapping[key])
    for relative in ("config.yaml", "checkpoints/reward_best.pt"):
        path = root / relative
        if not path.is_file():
            missing_files.append(str(path))
if missing_files:
    raise SystemExit(
        "missing STPM artifacts; run scripts/train_stpm_libero10_v3_all.sh first:\n"
        + "\n".join(missing_files)
    )
' "${STPM_PATHS}"
  fi
fi

train_cmd=(
  bash scripts/train_mam_libero_put_bowl_on_plate_multigpu.sh
  --seed="${SEED}"
  --cudnn_deterministic="${CUDNN_DETERMINISTIC}"
  --optimizer.grad_clip_norm="${GRAD_CLIP_NORM}"
  --policy.n_obs_steps=2
  --policy.horizon=32
  --policy.n_action_steps=15
  --policy.use_relative_actions=true
  --policy.spatial_softmax_num_keypoints=32
  --policy.use_language_conditioning=true
  --policy.pretrained_backbone_weights="${PRETRAINED_BACKBONE_WEIGHTS}"
  --policy.loss_mode="${MASK_LOSS_MODE}"
  --policy.loss_mask_area_weight="${MASK_KNOWN_REGION_WEIGHT}"
  --policy.inpainting="${MASK_INPAINTING}"
  --policy.do_mask_loss_for_padding="${DO_MASK_LOSS_FOR_PADDING}"
  --policy.mas_short_window_horizon="${MAS_SHORT_WINDOW_HORIZON}"
  --policy.mas_long_backward_length="${MAS_LONG_BACKWARD_LENGTH}"
  --policy.mas_long_forward_length="${MAS_LONG_FORWARD_LENGTH}"
  --policy.mas_long_feature_dim="${MAS_LONG_FEATURE_DIM}"
  --policy.optimizer_lr="${LEARNING_RATE}"
  --policy.optimizer_weight_decay="${WEIGHT_DECAY}"
  --policy.scheduler_warmup_steps="${WARMUP_STEPS}"
  "$@"
)

echo "MAM dataset: ${DATASET_ROOT}"
echo "MAM eval dataset: ${MAM_EVAL_DATASET_ROOT}"
echo "Denoiser: ${DENOISER_TYPE}"
echo "Mask: train_types=${TRAIN_MASK_TYPES}, eval_types=${EVAL_MASK_TYPES}, loss=${MASK_LOSS_MODE}, known_weight=${MASK_KNOWN_REGION_WEIGHT}, inpainting=${MASK_INPAINTING}"
echo "STPM paths: ${STPM_PATHS}"
if [[ "${DRY_RUN}" == "true" ]]; then
  printf "%q " "${train_cmd[@]}"
  printf "\n"
  exit 0
fi

exec "${train_cmd[@]}"
