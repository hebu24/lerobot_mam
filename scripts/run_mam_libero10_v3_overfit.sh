#!/usr/bin/env bash
set -euo pipefail

# Strict LIBERO-10 v3 MAM overfit with a mode-specific demonstration count.
# Modes:
#   1: one task, 5 demos
#   2: two tasks, 5 demos per task
#   3: five tasks, 2 demos per task
# Examples:
#   bash scripts/run_mam_libero10_v3_overfit.sh 3
#   TASK_IDS=2,4 bash scripts/run_mam_libero10_v3_overfit.sh 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# =============================================================================
# USER-EDITABLE MASK / STPM CONFIG
# Edit the values in this block directly before starting an overfit run.
# MASK_TYPE must already exist in the preprocessed MAM dataset.
# =============================================================================
MASK_TYPE="random_mask"
MASK_LOSS_MODE="weighted"
MASK_KNOWN_REGION_WEIGHT="0.2"
MASK_INPAINTING="false"
MASK_PADDING_LOSS="true"

# Task 0 uses the latest 6-epoch-equivalent STPM best checkpoint in this root.
STPM_TASK0_ROOT="outputs/train/stpm_libero10_v2_task0_plus4epoch"

MODE="${MODE:-3}"
if (( ${#} > 0 )) && [[ "${1}" != --* ]]; then
  MODE="${1}"
  shift
fi

case "${MODE}" in
  1)
    TASK_COUNT=1
    DEMOS_PER_TASK=5
    DEFAULT_TASK_IDS="0"
    ;;
  2)
    TASK_COUNT=2
    DEMOS_PER_TASK=2
    DEFAULT_TASK_IDS="0,1"
    ;;
  3)
    TASK_COUNT=5
    DEMOS_PER_TASK=2
    DEFAULT_TASK_IDS="0,1,2,3,4"
    ;;
  *)
    echo "MODE must be 1, 2, or 3; got ${MODE}." >&2
    exit 2
    ;;
esac
TOTAL_DEMOS=$((TASK_COUNT * DEMOS_PER_TASK))

IFS=',' read -r -a TASK_ID_ARRAY <<< "${TASK_IDS:-${DEFAULT_TASK_IDS}}"
if (( ${#TASK_ID_ARRAY[@]} != TASK_COUNT )); then
  echo "Mode ${MODE} requires exactly ${TASK_COUNT} task id(s)." >&2
  exit 2
fi

declare -A SEEN_TASK_IDS=()
TASK_IDS_CSV=""
TASK_IDS_JSON="["
TASK_TAG=""
for raw_task_id in "${TASK_ID_ARRAY[@]}"; do
  task_id="${raw_task_id//[[:space:]]/}"
  if ! [[ "${task_id}" =~ ^[0-9]$ ]]; then
    echo "LIBERO-10 task ids must be unique integers in [0, 9]; got ${raw_task_id}." >&2
    exit 2
  fi
  if [[ -n "${SEEN_TASK_IDS[${task_id}]:-}" ]]; then
    echo "Duplicate task id: ${task_id}." >&2
    exit 2
  fi
  SEEN_TASK_IDS[${task_id}]=1
  TASK_IDS_CSV+="${TASK_IDS_CSV:+,}${task_id}"
  if [[ "${TASK_IDS_JSON}" != "[" ]]; then
    TASK_IDS_JSON+=","
  fi
  TASK_IDS_JSON+="${task_id}"
  TASK_TAG+="${TASK_TAG:+-}${task_id}"
done
TASK_IDS_JSON+="]"

DEFAULT_DATASET_ROOT="outputs/datasets/libero10_mam_v3_train"
DEFAULT_DATASET_REPO_ID="local/libero10_mam_v3_train"
if [[ -z "${DATASET_ROOT:-}" \
  && ! -f "${DEFAULT_DATASET_ROOT}/meta/libero_pipeline.json" \
  && -f "outputs/datasets/libero10_mam_v3_unfiltered_train/meta/libero_pipeline.json" ]]; then
  DEFAULT_DATASET_ROOT="outputs/datasets/libero10_mam_v3_unfiltered_train"
  DEFAULT_DATASET_REPO_ID="local/libero10_mam_v3_unfiltered_train"
fi
DATASET_ROOT="${DATASET_ROOT:-${DEFAULT_DATASET_ROOT}}"
DATASET_REPO_ID="${DATASET_REPO_ID:-${DEFAULT_DATASET_REPO_ID}}"
DEMO_RANK="${DEMO_RANK:-0}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/mam_libero10_v3_overfit_mode${MODE}_tasks${TASK_TAG}_${TOTAL_DEMOS}demos}"
PLAN_PATH="${PLAN_PATH:-${OUTPUT_DIR}.selection.json}"
STPM_BASE_DIR="${STPM_BASE_DIR:-outputs/train}"
STPM_NAME_PREFIX="${STPM_NAME_PREFIX:-stpm_libero10_v2_task}"
DRY_RUN="${DRY_RUN:-false}"

if ! [[ "${DEMO_RANK}" =~ ^[0-9]+$ ]]; then
  echo "DEMO_RANK must be a non-negative integer; got ${DEMO_RANK}." >&2
  exit 2
fi
if [[ -z "${MASK_TYPE}" ]]; then
  echo "MASK_TYPE must be non-empty." >&2
  exit 2
fi
if [[ "${MASK_LOSS_MODE}" != "average" && "${MASK_LOSS_MODE}" != "weighted" ]]; then
  echo "MASK_LOSS_MODE must be average or weighted; got ${MASK_LOSS_MODE}." >&2
  exit 2
fi
for value_name in MASK_INPAINTING MASK_PADDING_LOSS; do
  value="${!value_name}"
  if [[ "${value}" != "true" && "${value}" != "false" ]]; then
    echo "${value_name} must be true or false; got ${value}." >&2
    exit 2
  fi
done
if [[ "${DRY_RUN}" != "true" && "${DRY_RUN}" != "false" ]]; then
  echo "DRY_RUN must be true or false; got ${DRY_RUN}." >&2
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

selector=(
  uv run python scripts/prepare_libero10_v3_overfit.py
  --dataset-root="${DATASET_ROOT}"
  --dataset-repo-id="${DATASET_REPO_ID}"
  --k="${TASK_COUNT}"
  --task-ids="${TASK_IDS_CSV}"
  --demo-rank="${DEMO_RANK}"
  --demos-per-task="${DEMOS_PER_TASK}"
  --mask-type="${MASK_TYPE}"
  --output-plan="${PLAN_PATH}"
)
if ! selection_output="$("${selector[@]}")"; then
  echo "Trajectory selector rejected the dataset." >&2
  exit 2
fi
mapfile -t selection <<< "${selection_output}"
if (( ${#selection[@]} != 2 )); then
  echo "Trajectory selector returned an invalid response." >&2
  exit 2
fi
SELECTED_TASK_IDS_JSON="${selection[0]}"
EPISODES_JSON="${selection[1]}"

STPM_PATHS="{"
for task_id in "${TASK_ID_ARRAY[@]}"; do
  task_id="${task_id//[[:space:]]/}"
  if [[ "${task_id}" == "0" ]]; then
    stpm_root="${STPM_TASK0_ROOT}"
  else
    stpm_root="${STPM_BASE_DIR}/${STPM_NAME_PREFIX}${task_id}"
  fi
  if [[ ! -f "${stpm_root}/config.yaml" || ! -f "${stpm_root}/checkpoints/reward_best.pt" ]]; then
    echo "Missing STPM artifacts for task ${task_id}: ${stpm_root}" >&2
    exit 2
  fi
  if [[ "${STPM_PATHS}" != "{" ]]; then
    STPM_PATHS+=","
  fi
  STPM_PATHS+="\"libero_10/${task_id}\":\"${stpm_root}\""
done
STPM_PATHS+="}"

export DATASET_REPO_ID
export DATASET_ROOT
export DATASET_EPISODES="${EPISODES_JSON}"
export MAM_EVAL_DATASET_REPO_ID="${DATASET_REPO_ID}"
export MAM_EVAL_DATASET_ROOT="${DATASET_ROOT}"
export MAM_EVAL_EPISODES="${EPISODES_JSON}"
export STPM_PATHS
export ENABLE_EVAL="${ENABLE_EVAL:-true}"
export EVAL_N_EPISODES="${TOTAL_DEMOS}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export EVAL_FREQ="${EVAL_FREQ:-1000}"
export ENV_TASK="libero_10"
export ENV_TASK_IDS="${SELECTED_TASK_IDS_JSON}"
export ENV_CONTROL_MODE="absolute"
export ENV_OBSERVATION_HEIGHT=128
export ENV_OBSERVATION_WIDTH=128
export OUTPUT_DIR
export JOB_NAME="${JOB_NAME:-mam_libero10_v3_overfit_mode${MODE}_tasks${TASK_TAG}_${TOTAL_DEMOS}demos}"
export STEPS="${STEPS:-40000}"
export SAVE_FREQ="${SAVE_FREQ:-1000}"
export LOG_FREQ="${LOG_FREQ:-200}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export PRETRAINED_BACKBONE_WEIGHTS="${PRETRAINED_BACKBONE_WEIGHTS:-null}"
export DO_MASK_LOSS_FOR_PADDING="${MASK_PADDING_LOSS}"

train_cmd=(
  bash scripts/run_mam_libero10_conda.sh
  --overfit_test=true
  --overfit_per_task=false
  --num_overfit="${TOTAL_DEMOS}"
  --cudnn_deterministic=true
  --policy.loss_mode="${MASK_LOSS_MODE}"
  --policy.loss_mask_area_weight="${MASK_KNOWN_REGION_WEIGHT}"
  --policy.inpainting="${MASK_INPAINTING}"
  "$@"
)

echo "MAM overfit mode ${MODE}: tasks=${SELECTED_TASK_IDS_JSON}, demos_per_task=${DEMOS_PER_TASK}, mask_type=${MASK_TYPE}"
echo "Mask policy: loss_mode=${MASK_LOSS_MODE}, known_region_weight=${MASK_KNOWN_REGION_WEIGHT}, inpainting=${MASK_INPAINTING}, padding_loss=${MASK_PADDING_LOSS}"
echo "Train/eval episodes=${EPISODES_JSON}; STPM paths=${STPM_PATHS}; plan=${PLAN_PATH}"
if [[ "${DRY_RUN}" == "true" ]]; then
  printf '%q ' "${train_cmd[@]}"
  printf '\n'
  exit 0
fi

exec "${train_cmd[@]}"
