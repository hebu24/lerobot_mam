#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PATH="/cephfs/shared/Yanbang/envs/lerobot0.5.1/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

VARIANT_LIST="${VARIANT_LIST:-obs3_gap1,obs6_gap1,obs6_gap2}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
RUN_ID="${RUN_ID:-stpm_v3_obsgap_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-outputs/logs/${RUN_ID}}"
REQUIRE_IDLE_GPU="${REQUIRE_IDLE_GPU:-true}"
DRY_RUN="${DRY_RUN:-false}"

export DATASET_REPO_ID="${DATASET_REPO_ID:-local/libero10_mam_v3_unfiltered_train}"
export DATASET_ROOT="${DATASET_ROOT:-outputs/datasets/libero10_mam_v3_unfiltered_train}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-6}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
export EPOCHS="${EPOCHS:-6}"
export STEPS="${STEPS:-}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
export WARMUP_STEPS="${WARMUP_STEPS:-0}"
export SCHEDULER_TOTAL_STEPS="${SCHEDULER_TOTAL_STEPS:-0}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0}"
export SEED="${SEED:-0}"
export VAL_RATIO="${VAL_RATIO:-0.1}"
export VISION_CKPT="${VISION_CKPT:-/cephfs/shared/Yanbang/maniskill/pretrained/clip-vit-base-patch32}"
export CLIP_ENCODE_BATCH_SIZE="${CLIP_ENCODE_BATCH_SIZE:-64}"
export D_MODEL="${D_MODEL:-512}"
export N_LAYERS="${N_LAYERS:-4}"
export N_HEADS="${N_HEADS:-8}"
export DROPOUT="${DROPOUT:-0.1}"
export SKIP_EXISTING="${SKIP_EXISTING:-true}"
export REQUIRE_CUDA="${REQUIRE_CUDA:-true}"

declare -A OBS_STEPS=(
  [obs3_gap1]=3
  [obs6_gap1]=6
  [obs6_gap2]=6
)
declare -A FRAME_GAPS=(
  [obs3_gap1]=1
  [obs6_gap1]=1
  [obs6_gap2]=2
)
declare -A PREFIXES=(
  [obs3_gap1]="stpm_libero10_v3_d512_l4_obs3_gap1_seed0_6epoch_task"
  [obs6_gap1]="stpm_libero10_v3_d512_l4_obs6_gap1_seed0_6epoch_task"
  [obs6_gap2]="stpm_libero10_v3_d512_l4_obs6_gap2_seed0_6epoch_task"
)

for name in REQUIRE_IDLE_GPU DRY_RUN; do
  value="${!name}"
  if [[ "${value}" != "true" && "${value}" != "false" ]]; then
    echo "${name} must be true or false; got ${value}." >&2
    exit 2
  fi
done

IFS=',' read -r -a gpu_ids <<<"${GPU_IDS}"
if (( ${#gpu_ids[@]} < 1 || ${#gpu_ids[@]} > 10 )); then
  echo "GPU_IDS must contain between 1 and 10 GPU ids; got ${GPU_IDS}." >&2
  exit 2
fi

IFS=',' read -r -a variants <<<"${VARIANT_LIST}"
if (( ${#variants[@]} == 0 )); then
  echo "VARIANT_LIST must contain at least one variant." >&2
  exit 2
fi
for raw_variant in "${variants[@]}"; do
  variant="${raw_variant//[[:space:]]/}"
  if [[ -z "${OBS_STEPS[${variant}]:-}" ]]; then
    echo "Unknown variant: ${variant}" >&2
    echo "Available variants: ${!OBS_STEPS[*]}" >&2
    exit 2
  fi
done

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Dataset root does not exist: ${DATASET_ROOT}" >&2
  exit 2
fi

if [[ "${REQUIRE_IDLE_GPU}" == "true" && "${DRY_RUN}" != "true" ]]; then
  for gpu_id in "${gpu_ids[@]}"; do
    active="$(
      nvidia-smi --id="${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits |
        sed '/^[[:space:]]*$/d'
    )"
    if [[ -n "${active}" ]]; then
      echo "GPU ${gpu_id} is already in use by PID(s): ${active//$'\n'/, }." >&2
      exit 2
    fi
  done
fi

mkdir -p "${LOG_ROOT}"

variant_complete() {
  local prefix="$1"
  for task_id in {0..9}; do
    local root="outputs/train/${prefix}${task_id}"
    for artifact in config.yaml state_norm.json checkpoints/reward_best.pt checkpoints/reward_best_endpoint.pt checkpoints/reward_final.pt; do
      if [[ ! -f "${root}/${artifact}" ]]; then
        return 1
      fi
    done
  done
  return 0
}

run_wave() {
  local variant="$1"
  shift
  local tasks=("$@")
  local prefix="${PREFIXES[${variant}]}"
  local pids=()
  local logs=()
  local status=0

  for index in "${!tasks[@]}"; do
    local task_id="${tasks[$index]}"
    local gpu_id="${gpu_ids[$index]}"
    local log="${LOG_ROOT}/${variant}_task${task_id}.log"
    echo "[Launch] variant=${variant} task=${task_id} gpu=${gpu_id} log=${log}"
    if [[ "${DRY_RUN}" == "true" ]]; then
      printf 'CUDA_VISIBLE_DEVICES=%q TASK_IDS=%q STPM_NAME_PREFIX=%q N_OBS_STEPS=%q FRAME_GAP=%q D_MODEL=512 N_LAYERS=4 N_HEADS=8 bash scripts/train_stpm_libero10_v3_all.sh\n' \
        "${gpu_id}" "${task_id}" "${prefix}" "${OBS_STEPS[${variant}]}" "${FRAME_GAPS[${variant}]}"
      continue
    fi
    (
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      export TASK_IDS="${task_id}"
      export STPM_NAME_PREFIX="${prefix}"
      export N_OBS_STEPS="${OBS_STEPS[${variant}]}"
      export FRAME_GAP="${FRAME_GAPS[${variant}]}"
      bash scripts/train_stpm_libero10_v3_all.sh
    ) >"${log}" 2>&1 &
    pids+=("$!")
    logs+=("${log}")
  done

  if [[ "${DRY_RUN}" == "true" ]]; then
    return
  fi

  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "[Failure] STPM worker failed: ${logs[$index]}" >&2
      tail -120 "${logs[$index]}" >&2 || true
      status=1
    fi
  done
  return "${status}"
}

for raw_variant in "${variants[@]}"; do
  variant="${raw_variant//[[:space:]]/}"
  prefix="${PREFIXES[${variant}]}"
  if variant_complete "${prefix}"; then
    echo "[Skip] variant=${variant}: complete STPM exists for all 10 tasks"
    continue
  fi

  echo "[Variant] ${variant}: prefix=${prefix}, obs=${OBS_STEPS[${variant}]}, gap=${FRAME_GAPS[${variant}]}"
  for ((start=0; start<10; start+=${#gpu_ids[@]})); do
    wave=()
    for ((offset=0; offset<${#gpu_ids[@]} && start+offset<10; offset++)); do
      wave+=("$((start + offset))")
    done
    run_wave "${variant}" "${wave[@]}"
  done

  if [[ "${DRY_RUN}" != "true" ]] && ! variant_complete "${prefix}"; then
    echo "Variant ${variant} did not produce all required artifacts." >&2
    exit 1
  fi
done

echo "STPM v3 obs/gap variants complete. Logs: ${LOG_ROOT}"
