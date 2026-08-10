#!/usr/bin/env bash
set -euo pipefail

# Full LIBERO-10 v3 Diffusion Policy training launcher.
#
# Common overrides:
#   STEPS=100000 BATCH_SIZE=16 LEARNING_RATE=1e-4 \
#     bash scripts/run_diffusion_libero10.sh
#
# Evaluation environment:
#   EVAL_ENV_MODE=random bash scripts/run_diffusion_libero10.sh
#   EVAL_ENV_MODE=random EVAL_START_SEED=100000 EVAL_N_EPISODES=50 \
#     bash scripts/run_diffusion_libero10.sh
#   EVAL_ENV_MODE=fixed bash scripts/run_diffusion_libero10.sh
#
# Resume:
#   RESUME=true OUTPUT_DIR=outputs/train/diffusion_libero10_v3_full \
#     STEPS=100000 bash scripts/run_diffusion_libero10.sh
#
# Print the command without using the GPU:
#   DRY_RUN=true bash scripts/run_diffusion_libero10.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# -----------------------------------------------------------------------------
# Data and output
# -----------------------------------------------------------------------------
DATASET_REPO_ID="${DATASET_REPO_ID:-local/libero10_mam_v3_train}"
DATASET_ROOT="${DATASET_ROOT:-outputs/datasets/libero10_mam_v3_train}"
EVAL_DATASET_REPO_ID="${EVAL_DATASET_REPO_ID:-local/libero10_mam_v3_eval}"
EVAL_DATASET_ROOT="${EVAL_DATASET_ROOT:-outputs/datasets/libero10_mam_v3_eval}"
EVAL_ENV_MODE="${EVAL_ENV_MODE:-${eval_env_mode:-fixed}}"
# random/fixed
DATASET_EPISODES="${DATASET_EPISODES:-}"
EVAL_DATASET_EPISODES="${EVAL_DATASET_EPISODES:-}"
JOB_NAME="${JOB_NAME:-diffusion_libero10_v3_full}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${JOB_NAME}}"

# -----------------------------------------------------------------------------
# Hardware and dataloader
# BATCH_SIZE is per GPU; effective batch = BATCH_SIZE * NUM_GPUS.
# -----------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS="${NUM_GPUS:-}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"

# -----------------------------------------------------------------------------
# Training and optimizer
# -----------------------------------------------------------------------------
STEPS="${STEPS:-50000}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
SEED="${SEED:-1000}"
CUDNN_DETERMINISTIC="${CUDNN_DETERMINISTIC:-false}"
LOG_FREQ="${LOG_FREQ:-200}"
SAVE_FREQ="${SAVE_FREQ:-10000}"

# -----------------------------------------------------------------------------
# Diffusion Policy
# The relative-action and absolute-controller switches are intentionally fixed:
# they are invariants of the audited LIBERO-10 v3 pipeline.
# -----------------------------------------------------------------------------
N_OBS_STEPS="${N_OBS_STEPS:-2}"
HORIZON="${HORIZON:-32}"
N_ACTION_STEPS="${N_ACTION_STEPS:-15}"
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
SPATIAL_SOFTMAX_NUM_KEYPOINTS="${SPATIAL_SOFTMAX_NUM_KEYPOINTS:-32}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}"
USE_LANGUAGE_CONDITIONING="${USE_LANGUAGE_CONDITIONING:-true}"
PRETRAINED_BACKBONE_WEIGHTS="${PRETRAINED_BACKBONE_WEIGHTS:-null}"
DO_MASK_LOSS_FOR_PADDING="${DO_MASK_LOSS_FOR_PADDING:-true}"

# -----------------------------------------------------------------------------
# Simulator evaluation
# EVAL_N_EPISODES is per task for generic DP evaluation.
# -----------------------------------------------------------------------------
ENABLE_EVAL="${ENABLE_EVAL:-true}"
EVAL_FREQ="${EVAL_FREQ:-5000}"
if [[ "${EVAL_ENV_MODE}" == "random" ]]; then
  # Match LPB: 50 deterministic held-out resets per task by default.
  EVAL_N_EPISODES="${EVAL_N_EPISODES:-50}"
else
  EVAL_N_EPISODES="${EVAL_N_EPISODES:-5}"
fi
EVAL_START_SEED="${EVAL_START_SEED:-100000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
ENV_TASK_IDS="${ENV_TASK_IDS:-[0,1,2,3,4,5,6,7,8,9]}"
ENV_MAX_PARALLEL_TASKS="${ENV_MAX_PARALLEL_TASKS:-1}"
ENV_OBSERVATION_HEIGHT="${ENV_OBSERVATION_HEIGHT:-128}"
ENV_OBSERVATION_WIDTH="${ENV_OBSERVATION_WIDTH:-128}"

# -----------------------------------------------------------------------------
# Logging, resume and safety checks
# -----------------------------------------------------------------------------
WANDB_ENABLE="${WANDB_ENABLE:-false}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
RESUME="${RESUME:-false}"
RESUME_CONFIG_PATH="${RESUME_CONFIG_PATH:-${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json}"
REQUIRE_FULL_DATASET="${REQUIRE_FULL_DATASET:-true}"
REQUIRE_IDLE_GPU="${REQUIRE_IDLE_GPU:-true}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
DRY_RUN="${DRY_RUN:-false}"

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
export CUDA_VISIBLE_DEVICES

is_bool() {
  [[ "$1" == "true" || "$1" == "false" ]]
}

for value_name in \
  ENABLE_EVAL RESUME REQUIRE_FULL_DATASET REQUIRE_IDLE_GPU DRY_RUN \
  UNET_USE_FILM_SCALE_MODULATION DIT_USE_POSITIONAL_ENCODING DIT_USE_ROPE; do
  value="${!value_name}"
  if ! is_bool "${value}"; then
    echo "${value_name} must be true or false, got: ${value}" >&2
    exit 2
  fi
done

if [[ "${DENOISER_TYPE}" != "unet" && "${DENOISER_TYPE}" != "dit" ]]; then
  echo "DENOISER_TYPE must be unet or dit, got: ${DENOISER_TYPE}" >&2
  exit 2
fi

case "${EVAL_ENV_MODE}" in
  fixed|random) ;;
  *)
    echo "EVAL_ENV_MODE must be fixed or random, got: ${EVAL_ENV_MODE}" >&2
    exit 2
    ;;
esac

for value_name in \
  STEPS BATCH_SIZE NUM_WORKERS SAVE_FREQ LOG_FREQ HORIZON N_OBS_STEPS N_ACTION_STEPS \
  EVAL_N_EPISODES; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "${value_name} must be a non-negative integer, got: ${value}" >&2
    exit 2
  fi
done

if (( STEPS <= 0 || BATCH_SIZE <= 0 || SAVE_FREQ <= 0 || HORIZON <= 0 || N_OBS_STEPS <= 0 || N_ACTION_STEPS <= 0 || EVAL_N_EPISODES <= 0 )); then
  echo "STEPS, BATCH_SIZE, SAVE_FREQ, HORIZON, N_OBS_STEPS, N_ACTION_STEPS and EVAL_N_EPISODES must be positive." >&2
  exit 2
fi
if (( N_OBS_STEPS - 1 + N_ACTION_STEPS > HORIZON )); then
  echo "Invalid action window: N_OBS_STEPS - 1 + N_ACTION_STEPS must be <= HORIZON." >&2
  exit 2
fi
if [[ "${ENABLE_EVAL}" == "true" ]]; then
  if ! [[ "${EVAL_FREQ}" =~ ^[0-9]+$ ]] || (( EVAL_FREQ <= 0 )); then
    echo "EVAL_FREQ must be positive when ENABLE_EVAL=true." >&2
    exit 2
  fi
else
  EVAL_FREQ=0
fi
if [[ "${EVAL_ENV_MODE}" == "random" ]]; then
  if ! [[ "${EVAL_START_SEED}" =~ ^[0-9]+$ ]]; then
    echo "EVAL_START_SEED must be a non-negative integer, got: ${EVAL_START_SEED}" >&2
    exit 2
  fi
  eval_end_seed=$((EVAL_START_SEED + EVAL_N_EPISODES - 1))
  if (( EVAL_START_SEED <= 49 && eval_end_seed >= 0 )); then
    echo "Random eval seeds must not overlap 0..49, got: ${EVAL_START_SEED}..${eval_end_seed}" >&2
    exit 2
  fi
fi

if [[ "${RESUME}" == "true" ]]; then
  if [[ ! -f "${RESUME_CONFIG_PATH}" ]]; then
    echo "Resume config not found: ${RESUME_CONFIG_PATH}" >&2
    exit 2
  fi
elif [[ -e "${OUTPUT_DIR}" ]]; then
  echo "OUTPUT_DIR already exists: ${OUTPUT_DIR}" >&2
  echo "Choose another OUTPUT_DIR or set RESUME=true." >&2
  exit 2
fi

if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  echo "Full v3 train dataset is missing:" >&2
  echo "  train=${DATASET_ROOT}" >&2
  exit 2
fi
if [[ "${EVAL_ENV_MODE}" == "fixed" && ! -f "${EVAL_DATASET_ROOT}/meta/info.json" ]]; then
  echo "Fixed evaluation dataset is missing:" >&2
  echo "  eval=${EVAL_DATASET_ROOT}" >&2
  echo "Build them with the commands in mam.md section 3 before training." >&2
  exit 2
fi

# Validate that this is the closed-loop v3 dataset and that train/eval source
# trajectories are disjoint. By default, require all replay-certified source
# trajectories, with exactly five eval trajectories per task.
uv run python - \
  "${DATASET_ROOT}" "${EVAL_DATASET_ROOT}" "${REQUIRE_FULL_DATASET}" \
  "${EVAL_ENV_MODE}" "${N_OBS_STEPS}" "${HORIZON}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def read_dataset(root_text: str, expected_split: str):
    root = Path(root_text)
    manifest_path = root / "meta" / "libero_pipeline.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing v3 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "pipeline_version": "v3",
        "observation_materialization": "closed_loop_absolute_controller",
        "relative_action_ready": True,
        "relative_action_stats": True,
        "conversion_complete": True,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise SystemExit(f"{root}: manifest {key}={manifest.get(key)!r}, expected {expected!r}")
    n_obs_steps = int(sys.argv[5])
    horizon = int(sys.argv[6])
    expected_delta_indices = list(range(1 - n_obs_steps, 1 - n_obs_steps + horizon))
    certified_delta_indices = manifest.get("relative_action_stats_action_delta_indices")
    if certified_delta_indices != expected_delta_indices:
        raise SystemExit(
            f"{root}: relative stats do not match N_OBS_STEPS={n_obs_steps}, HORIZON={horizon}; "
            f"dataset indices={certified_delta_indices}, expected={expected_delta_indices}"
        )
    split = manifest.get("dataset_split")
    if split is not None and split != expected_split:
        raise SystemExit(f"{root}: dataset_split={split!r}, expected {expected_split!r}")

    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    features = info.get("features", {})
    if features.get("action", {}).get("shape") != [7]:
        raise SystemExit(f"{root}: action must be 7D")
    if features.get("observation.state", {}).get("shape") != [14]:
        raise SystemExit(f"{root}: observation.state must be 14D")
    image_shapes = sorted(
        value.get("shape")
        for key, value in features.items()
        if key.startswith("observation.images.")
    )
    if image_shapes != [[128, 128, 3], [128, 128, 3]]:
        raise SystemExit(f"{root}: expected two 128x128 RGB observations, got {image_shapes}")

    episode_paths = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if not episode_paths:
        raise SystemExit(f"{root}: no episode metadata")
    rows = []
    for path in episode_paths:
        rows.extend(pq.read_table(path).to_pylist())
    task_key = "libero/task_id" if "libero/task_id" in rows[0] else "task_id"
    source_key = (
        "libero/source_episode_id"
        if "libero/source_episode_id" in rows[0]
        else "source_episode_id"
        if "source_episode_id" in rows[0]
        else None
    )
    if source_key is None:
        raise SystemExit(f"{root}: source episode identity is missing")
    sources = {(int(row[task_key]), int(row[source_key])) for row in rows}
    counts = Counter(task for task, _ in sources)
    if sorted(counts) != list(range(10)):
        raise SystemExit(f"{root}: expected LIBERO-10 task ids 0..9, got {sorted(counts)}")
    return info, manifest, sources, counts


train_info, train_manifest, train_sources, train_counts = read_dataset(sys.argv[1], "train")
if sys.argv[4] == "random":
    print(
        "Dataset preflight OK: "
        f"train={train_info['total_episodes']} episodes/{sum(train_counts.values())} sources, "
        "eval=random environment seeds"
    )
    raise SystemExit(0)

eval_info, eval_manifest, eval_sources, eval_counts = read_dataset(sys.argv[2], "eval")
overlap = train_sources & eval_sources
if overlap:
    raise SystemExit(f"Train/eval source leakage detected: {sorted(overlap)[:10]}")

if sys.argv[3] == "true":
    expected_eval = {task: 5 for task in range(10)}
    if dict(eval_counts) != expected_eval:
        raise SystemExit(f"Eval is not the full 5/task source split: {dict(sorted(eval_counts.items()))}")
    train_source_root = train_manifest.get("source_root")
    eval_source_root = eval_manifest.get("source_root")
    if not train_source_root or train_source_root != eval_source_root:
        raise SystemExit(
            f"Train/eval source roots differ or are missing: {train_source_root!r}, {eval_source_root!r}"
        )
    source_manifest_path = Path(train_source_root) / "meta" / "libero_pipeline.json"
    if not source_manifest_path.is_file():
        raise SystemExit(f"Missing source replay manifest: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    valid_ids = source_manifest.get("valid_absolute_episode_ids")
    selected_ids = {source_id for _, source_id in train_sources | eval_sources}
    if valid_ids is not None:
        valid_ids = {int(value) for value in valid_ids}
        excluded_ids = {int(value) for value in source_manifest.get("unrepairable_episode_ids", [])}
        if not excluded_ids or valid_ids & excluded_ids:
            raise SystemExit("Source replay manifest has invalid valid/excluded episode sets")
        if selected_ids != valid_ids:
            raise SystemExit(
                "Final split does not cover every replay-certified source episode: "
                f"missing={sorted(valid_ids - selected_ids)[:10]}, "
                f"unexpected={sorted(selected_ids - valid_ids)[:10]}"
            )
    else:
        expected_train = {task: 45 for task in range(10)}
        if dict(train_counts) != expected_train:
            raise SystemExit(
                f"Train is not the full 45/task source split: {dict(sorted(train_counts.items()))}"
            )

print(
    "Dataset preflight OK: "
    f"train={train_info['total_episodes']} episodes/{sum(train_counts.values())} sources, "
    f"eval={eval_info['total_episodes']} episodes/{sum(eval_counts.values())} sources"
)
PY

if ! [[ "${MIN_FREE_GB}" =~ ^[0-9]+$ ]]; then
  echo "MIN_FREE_GB must be a non-negative integer, got: ${MIN_FREE_GB}" >&2
  exit 2
fi
available_kb="$(df -Pk "${REPO_ROOT}" | awk 'NR==2 {print $4}')"
available_gb=$((available_kb / 1024 / 1024))
if (( available_gb < MIN_FREE_GB )); then
  echo "Insufficient disk space: ${available_gb} GiB free, require at least ${MIN_FREE_GB} GiB." >&2
  exit 2
fi

if [[ -z "${NUM_GPUS}" ]]; then
  NUM_GPUS="$(uv run python -c 'import torch; print(torch.cuda.device_count())')"
fi
if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] || (( NUM_GPUS < 1 )); then
  echo "NUM_GPUS must be a positive integer, got: ${NUM_GPUS}" >&2
  exit 2
fi

launch_args=(
  --num_processes="${NUM_GPUS}"
  --num_machines=1
  --mixed_precision="${MIXED_PRECISION}"
  --dynamo_backend=no
)
if (( NUM_GPUS > 1 )); then
  launch_args=(--multi_gpu "${launch_args[@]}")
fi

if [[ "${RESUME}" == "true" ]]; then
  train_args=(
    --config_path="${RESUME_CONFIG_PATH}"
    --resume=true
    --steps="${STEPS}"
    --save_freq="${SAVE_FREQ}"
    --eval_freq="${EVAL_FREQ}"
    --eval.n_episodes="${EVAL_N_EPISODES}"
    --log_freq="${LOG_FREQ}"
  )
  if [[ "${EVAL_ENV_MODE}" == "random" ]]; then
    train_args+=(
      --eval.dataset_repo_id=null
      --eval.dataset_root=null
      --eval.dataset_episodes=null
      --eval.start_seed="${EVAL_START_SEED}"
      --env.init_states=false
    )
  fi
else
  train_args=(
    --policy.type=diffusion
    --policy.device=cuda
    --policy.push_to_hub="${PUSH_TO_HUB}"
    --policy.use_relative_actions=true
    --policy.n_obs_steps="${N_OBS_STEPS}"
    --policy.horizon="${HORIZON}"
    --policy.n_action_steps="${N_ACTION_STEPS}"
    --policy.denoiser_type="${DENOISER_TYPE}"
    --policy.spatial_softmax_num_keypoints="${SPATIAL_SOFTMAX_NUM_KEYPOINTS}"
    --policy.use_language_conditioning="${USE_LANGUAGE_CONDITIONING}"
    --policy.do_mask_loss_for_padding="${DO_MASK_LOSS_FOR_PADDING}"
    --policy.pretrained_backbone_weights="${PRETRAINED_BACKBONE_WEIGHTS}"
    --policy.optimizer_lr="${LEARNING_RATE}"
    --policy.optimizer_weight_decay="${WEIGHT_DECAY}"
    --policy.scheduler_name="${LR_SCHEDULER}"
    --policy.scheduler_warmup_steps="${WARMUP_STEPS}"
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.root="${DATASET_ROOT}"
    --output_dir="${OUTPUT_DIR}"
    --job_name="${JOB_NAME}"
    --batch_size="${BATCH_SIZE}"
    --num_workers="${NUM_WORKERS}"
    --steps="${STEPS}"
    --save_freq="${SAVE_FREQ}"
    --eval_freq="${EVAL_FREQ}"
    --log_freq="${LOG_FREQ}"
    --seed="${SEED}"
    --cudnn_deterministic="${CUDNN_DETERMINISTIC}"
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
  if (( NUM_WORKERS > 0 )); then
    train_args+=(
      --prefetch_factor="${PREFETCH_FACTOR}"
      --persistent_workers="${PERSISTENT_WORKERS}"
    )
  else
    train_args+=(--persistent_workers=false)
  fi
  if [[ -n "${DATASET_EPISODES}" ]]; then
    train_args+=(--dataset.episodes="${DATASET_EPISODES}")
  fi
  if [[ -n "${NUM_INFERENCE_STEPS}" ]]; then
    train_args+=(--policy.num_inference_steps="${NUM_INFERENCE_STEPS}")
  fi
  if [[ "${ENABLE_EVAL}" == "true" ]]; then
    train_args+=(
      --env.type=libero
      --env.task=libero_10
      --env.task_ids="${ENV_TASK_IDS}"
      --env.control_mode=absolute
      --env.observation_height="${ENV_OBSERVATION_HEIGHT}"
      --env.observation_width="${ENV_OBSERVATION_WIDTH}"
      --env.num_steps_wait=0
      --env.max_parallel_tasks="${ENV_MAX_PARALLEL_TASKS}"
      --eval.n_episodes="${EVAL_N_EPISODES}"
      --eval.batch_size="${EVAL_BATCH_SIZE}"
      --eval.use_async_envs=false
    )
    if [[ "${EVAL_ENV_MODE}" == "fixed" ]]; then
      train_args+=(
        --env.init_states=true
        --eval.dataset_repo_id="${EVAL_DATASET_REPO_ID}"
        --eval.dataset_root="${EVAL_DATASET_ROOT}"
      )
      if [[ -n "${EVAL_DATASET_EPISODES}" ]]; then
        train_args+=(--eval.dataset_episodes="${EVAL_DATASET_EPISODES}")
      fi
    else
      train_args+=(
        --env.init_states=false
        --eval.start_seed="${EVAL_START_SEED}"
      )
    fi
  fi
fi

echo "LIBERO-10 v3 full DP"
echo "  train=${DATASET_ROOT}"
echo "  eval_env_mode=${EVAL_ENV_MODE}"
if [[ "${EVAL_ENV_MODE}" == "fixed" ]]; then
  echo "  eval=${EVAL_DATASET_ROOT}"
else
  echo "  eval=random LPB seeds ${EVAL_START_SEED}..$((EVAL_START_SEED + EVAL_N_EPISODES - 1)) per task"
fi
echo "  GPUs=${NUM_GPUS} (${CUDA_VISIBLE_DEVICES}), batch/GPU=${BATCH_SIZE}, effective_batch=$((NUM_GPUS * BATCH_SIZE))"
echo "  steps=${STEPS}, lr=${LEARNING_RATE}, horizon=${HORIZON}, n_action_steps=${N_ACTION_STEPS}"
echo "  eval_freq=${EVAL_FREQ}, save_freq=${SAVE_FREQ}, output=${OUTPUT_DIR}"

cmd=(uv run accelerate launch "${launch_args[@]}" -m lerobot.scripts.lerobot_train "${train_args[@]}" "$@")
if [[ "${DRY_RUN}" == "true" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

nvidia-smi >/dev/null
uv run python -c 'import sys, torch; print(f"CUDA={torch.cuda.is_available()} GPUs={torch.cuda.device_count()}"); sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)'
if [[ "${REQUIRE_IDLE_GPU}" == "true" ]]; then
  active_gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
  if [[ -n "${active_gpu_pids}" ]]; then
    echo "GPU is already used by compute process(es): ${active_gpu_pids//$'\n'/, }" >&2
    echo "Wait for them to finish, or explicitly set REQUIRE_IDLE_GPU=false." >&2
    exit 2
  fi
fi

exec "${cmd[@]}"
