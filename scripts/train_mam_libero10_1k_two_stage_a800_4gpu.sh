#!/usr/bin/env bash
set -euo pipefail

# Two-stage LIBERO-10 1k MAM training. Defaults preserve the original
# single-node 4xA800 protocol; topology checks and batch size can be overridden
# for an equivalent-global-batch run on another VM.
#   1. random_mask from scratch through global step 90k
#   2. resume the new 90k checkpoint with the four-slot refmix composition
#      through global step 180k

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/cephfs/shared/Yanbang/envs/lerobot0.5.1}"
export CONDA_ENV_PATH
export PATH="${CONDA_ENV_PATH}/bin:${PATH}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-outputs/train/mam_libero10_1k_random090k_refmix180k_4a800_${RUN_ID}}"
PHASE1_OUTPUT="${PHASE1_OUTPUT:-${RUN_ROOT}/phase1_random_000000to090000}"
PHASE2_OUTPUT="${PHASE2_OUTPUT:-${RUN_ROOT}/phase2_refmix_090000to180000}"

RANDOM_TRAIN_ROOT="${RANDOM_TRAIN_ROOT:-${REPO_ROOT}/data/hf_libero10_mam/libero10_1000_train}"
REFMIX_TRAIN_ROOT="${REFMIX_TRAIN_ROOT:-${REPO_ROOT}/data/hf_libero10_mam/libero10_1000_refmix_train}"
RANDOM_EVAL_ROOT="${RANDOM_EVAL_ROOT:-${REPO_ROOT}/data/hf_libero10_mam/libero10_100_eval}"
REFMIX_EVAL_ROOT="${REFMIX_EVAL_ROOT:-${REPO_ROOT}/data/libero10_mam/libero10_100first50_refmix_eval}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NUM_GPUS="${NUM_GPUS:-4}"
export BATCH_SIZE="${BATCH_SIZE:-24}"
export MIXED_PRECISION=bf16
export NUM_WORKERS=8
export PREFETCH_FACTOR=4
export PERSISTENT_WORKERS=true

export SAVE_FREQ=10000
export EVAL_FREQ=5000
export LOG_FREQ=200
export ENABLE_EVAL=true
export EVAL_N_EPISODES=50
export EVAL_BATCH_SIZE=1
export EVAL_USE_ASYNC_ENVS=false
export ENV_TASK=libero_10
export ENV_TASK_IDS='[0,1,2,3,4,5,6,7,8,9]'
export ENV_CONTROL_MODE=absolute
export ENV_OBSERVATION_HEIGHT=128
export ENV_OBSERVATION_WIDTH=128
export ENV_MAX_PARALLEL_TASKS=1

export MASK_LOSS_MODE=average
export MASK_KNOWN_REGION_WEIGHT=0.2
export MASK_INPAINTING=false
export MASK_PADDING_LOSS=true
export DO_MASK_LOSS_FOR_PADDING=true
export MAS_SHORT_WINDOW_HORIZON=0
export MAS_LONG_BACKWARD_LENGTH=0
export MAS_LONG_FORWARD_LENGTH=64
export MAS_LONG_FEATURE_DIM=128

export LEARNING_RATE=1e-4
export WEIGHT_DECAY=1e-6
export WARMUP_STEPS=500
export GRAD_CLIP_NORM=10.0
export SEED=1000
export CUDNN_DETERMINISTIC=false
export PRETRAINED_BACKBONE_WEIGHTS=null
export PUSH_TO_HUB=false
export WANDB_ENABLE=false

export STPM_BASE_DIR="${STPM_BASE_DIR:-outputs/train}"
export STPM_NAME_PREFIX="${STPM_NAME_PREFIX:-stpm_libero10_v4_maniskill_d768_l8_obs6_gap2_seed42_20260729_task}"
export LIBERO_ASSETS_PATH="${LIBERO_ASSETS_PATH:-/root/.cache/libero/assets}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${HOME}/.libero}"
export SKIP_PREFLIGHT=false
export DRY_RUN=false

PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-false}"
if [[ "${PREFLIGHT_ONLY}" != "true" && "${PREFLIGHT_ONLY}" != "false" ]]; then
  echo "PREFLIGHT_ONLY must be true or false, got ${PREFLIGHT_ONLY}." >&2
  exit 2
fi

KEEP_ALL_CHECKPOINTS_AFTER_STEP=90000
PHASE1_END_STEP=90000
PHASE2_END_STEP=180000

EXPECTED_HOST="${EXPECTED_HOST:-zhangchenyu4-0}"
EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-4}"
EXPECTED_GPU_NAME="${EXPECTED_GPU_NAME:-A800}"

if [[ "$(hostname)" != "${EXPECTED_HOST}" && "${ALLOW_OTHER_HOST:-false}" != "true" ]]; then
  echo "Expected host ${EXPECTED_HOST}; got $(hostname)." >&2
  exit 2
fi

mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if (( ${#gpu_names[@]} != EXPECTED_GPU_COUNT )); then
  echo "Expected ${EXPECTED_GPU_COUNT} GPUs, found ${#gpu_names[@]}." >&2
  exit 2
fi
for gpu_name in "${gpu_names[@]}"; do
  if [[ -n "${EXPECTED_GPU_NAME}" && "${gpu_name}" != *"${EXPECTED_GPU_NAME}"* ]]; then
    echo "Expected GPU names containing '${EXPECTED_GPU_NAME}', found: ${gpu_names[*]}" >&2
    exit 2
  fi
done

IFS=',' read -r -a visible_gpu_ids <<<"${CUDA_VISIBLE_DEVICES}"
if (( ${#visible_gpu_ids[@]} != NUM_GPUS )); then
  echo "NUM_GPUS=${NUM_GPUS}, but CUDA_VISIBLE_DEVICES exposes ${#visible_gpu_ids[@]} devices." >&2
  exit 2
fi

active_gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
if [[ -n "${active_gpu_pids}" ]]; then
  echo "GPU(s) are already used by process(es): ${active_gpu_pids//$'\n'/, }" >&2
  exit 2
fi

"${CONDA_ENV_PATH}/bin/python" - \
  "${RANDOM_TRAIN_ROOT}" "${REFMIX_TRAIN_ROOT}" \
  "${RANDOM_EVAL_ROOT}" "${REFMIX_EVAL_ROOT}" \
  "${STPM_BASE_DIR}" "${STPM_NAME_PREFIX}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

random_train, refmix_train, random_eval, refmix_eval, stpm_base, stpm_prefix = map(Path, sys.argv[1:])


def validate_dataset(root: Path, split: str, episodes: int, masks: list[str], per_task: int) -> None:
    info_path = root / "meta" / "info.json"
    manifest_path = root / "meta" / "libero_pipeline.json"
    if not info_path.is_file() or not manifest_path.is_file():
        raise SystemExit(f"Missing dataset metadata under {root}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "stage": "absolute_to_mam",
        "dataset_split": split,
        "action_representation": "osc_pose_absolute_goal",
        "policy_action_representation": "chunk_relative_se3",
        "relative_action_ready": True,
        "relative_action_stats_action_delta_indices": list(range(-1, 31)),
        "mask_types": masks,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise SystemExit(f"{manifest_path}: expected {key}={value!r}, got {manifest.get(key)!r}")
    if int(info.get("total_episodes", -1)) != episodes:
        raise SystemExit(f"{root}: expected {episodes} episodes, got {info.get('total_episodes')}")
    rows = []
    for path in sorted((root / "meta" / "episodes").glob("**/*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    task_counts = Counter(int(row["libero/task_id"]) for row in rows)
    if task_counts != Counter({task: per_task for task in range(10)}):
        raise SystemExit(f"{root}: unexpected task counts {dict(sorted(task_counts.items()))}")
    if any(row.get("libero/init_state") is None for row in rows):
        raise SystemExit(f"{root}: every episode must contain libero/init_state")


validate_dataset(random_train, "train", 1000, ["random_mask"], 100)
validate_dataset(
    refmix_train,
    "train",
    1000,
    ["points", "3D_points", "3D_points", "pose_motion_planning"],
    100,
)
validate_dataset(random_eval, "eval", 100, ["random_mask"], 10)
validate_dataset(
    refmix_eval,
    "eval",
    50,
    ["points", "3D_points", "3D_points", "pose_motion_planning", "mix0"],
    5,
)

missing = []
for task in range(10):
    root = stpm_base / f"{stpm_prefix}{task}"
    for relative in ("config.yaml", "checkpoints/reward_best.pt"):
        path = root / relative
        if not path.is_file():
            missing.append(str(path))
if missing:
    raise SystemExit("Missing STPM artifacts:\n" + "\n".join(missing))

print("Preflight OK: 1k train=100/task; fixed eval=5/task; random and refmix masks audited")
PY

mkdir -p "${RUN_ROOT}"
{
  echo "run_id=${RUN_ID}"
  echo "host=$(hostname)"
  echo "gpus=${gpu_names[*]}"
  echo "phase1=random_mask scratch 0..${PHASE1_END_STEP}"
  echo "phase2=points,3D_points,3D_points,pose_motion_planning resume ${PHASE1_END_STEP}..${PHASE2_END_STEP}"
  echo "batch_per_gpu=${BATCH_SIZE} global_batch=$((NUM_GPUS * BATCH_SIZE))"
  echo "save_freq=${SAVE_FREQ} keep_all_checkpoints_after_step=${KEEP_ALL_CHECKPOINTS_AFTER_STEP}"
} >"${RUN_ROOT}/protocol.txt"

if [[ "${PREFLIGHT_ONLY}" == "true" ]]; then
  echo "Preflight-only check completed successfully: ${RUN_ROOT}"
  exit 0
fi

run_phase() {
  local phase="$1"
  local output_dir="$2"
  local end_step="$3"
  local train_repo_id="$4"
  local train_root="$5"
  local eval_repo_id="$6"
  local eval_root="$7"
  local train_masks="$8"
  local eval_masks="$9"
  local source_config="${10:-}"

  export STEPS="${end_step}"
  export DATASET_REPO_ID="${train_repo_id}"
  export DATASET_ROOT="${train_root}"
  export MAM_EVAL_DATASET_REPO_ID="${eval_repo_id}"
  export MAM_EVAL_DATASET_ROOT="${eval_root}"
  export MASK_TYPE="${train_masks%%,*}"
  export MASK_TYPES="${train_masks}"
  export TRAIN_MASK_TYPES="${train_masks}"
  export EVAL_MASK_TYPES="${eval_masks}"
  export JOB_NAME="$(basename "${output_dir}")"
  export OUTPUT_DIR="${output_dir}"

  local resume_config=""
  if [[ -f "${output_dir}/checkpoints/last/pretrained_model/train_config.json" ]]; then
    resume_config="${output_dir}/checkpoints/last/pretrained_model/train_config.json"
    echo "[${phase}] resuming interrupted phase from ${resume_config}"
  elif [[ -n "${source_config}" ]]; then
    resume_config="${source_config}"
    echo "[${phase}] starting from ${resume_config}"
  elif [[ -e "${output_dir}" ]]; then
    echo "[${phase}] output exists without a resumable checkpoint: ${output_dir}" >&2
    exit 2
  else
    echo "[${phase}] starting from scratch"
  fi

  local resume_args=()
  if [[ -n "${resume_config}" ]]; then
    if [[ ! -f "${resume_config}" ]]; then
      echo "[${phase}] missing resume config: ${resume_config}" >&2
      exit 2
    fi
    resume_args+=(--resume=true --config_path="${resume_config}")
  fi

  echo "[${phase}] train=${train_root} masks=${train_masks}"
  echo "[${phase}] eval=${eval_root} masks=${eval_masks}"
  echo "[${phase}] target=${end_step} output=${output_dir}"

  bash scripts/libero/train/run_mam_libero10_conda.sh \
    --keep_all_checkpoints_after_step="${KEEP_ALL_CHECKPOINTS_AFTER_STEP}" \
    --policy.allow_independent_eval_source=true \
    --policy.language_tokenizer_name=/cephfs/shared/Yanbang/maniskill/pretrained/clip-vit-base-patch32 \
    "${resume_args[@]}"
}

phase1_checkpoint="${PHASE1_OUTPUT}/checkpoints/090000/pretrained_model/train_config.json"
if [[ -f "${phase1_checkpoint}" ]]; then
  echo "[phase1] completed checkpoint already exists: ${phase1_checkpoint}"
else
  run_phase \
    phase1_random "${PHASE1_OUTPUT}" "${PHASE1_END_STEP}" \
    local/libero10_1000_train "${RANDOM_TRAIN_ROOT}" \
    local/libero10_100_eval "${RANDOM_EVAL_ROOT}" \
    random_mask random_mask
fi
if [[ ! -f "${phase1_checkpoint}" ]]; then
  echo "Phase 1 ended without the required 90k checkpoint: ${phase1_checkpoint}" >&2
  exit 1
fi

phase2_checkpoint="${PHASE2_OUTPUT}/checkpoints/180000/pretrained_model/train_config.json"
if [[ -f "${phase2_checkpoint}" ]]; then
  echo "[phase2] completed checkpoint already exists: ${phase2_checkpoint}"
else
  run_phase \
    phase2_refmix "${PHASE2_OUTPUT}" "${PHASE2_END_STEP}" \
    local/libero10_1000_refmix_train "${REFMIX_TRAIN_ROOT}" \
    local/libero10_100first50_refmix_eval "${REFMIX_EVAL_ROOT}" \
    points,3D_points,3D_points,pose_motion_planning \
    points,3D_points,3D_points,pose_motion_planning,mix0 \
    "${phase1_checkpoint}"
fi
if [[ ! -f "${phase2_checkpoint}" ]]; then
  echo "Phase 2 ended without the required 180k checkpoint: ${phase2_checkpoint}" >&2
  exit 1
fi

echo "Two-stage MAM training completed: ${RUN_ROOT}"
