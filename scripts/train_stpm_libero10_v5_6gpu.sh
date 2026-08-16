#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PATH="/cephfs/shared/Yanbang/envs/lerobot0.5.1/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export DATASET_REPO_ID="${DATASET_REPO_ID:-local/libero10_mam_v3_unfiltered_train}"
export DATASET_ROOT="${DATASET_ROOT:-outputs/datasets/libero10_mam_v3_unfiltered_train}"
export STPM_NAME_PREFIX="${STPM_NAME_PREFIX:-stpm_libero10_v5_d768_l8_obs6_gap2_seed0_6epoch_task}"
export N_OBS_STEPS="${N_OBS_STEPS:-6}"
export FRAME_GAP="${FRAME_GAP:-2}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-6}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
export EPOCHS="${EPOCHS:-6}"
export STEPS="${STEPS:-}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
export ADAM_BETA1="${ADAM_BETA1:-0.9}"
export ADAM_BETA2="${ADAM_BETA2:-0.999}"
export ADAM_EPS="${ADAM_EPS:-1e-8}"
export WARMUP_STEPS="${WARMUP_STEPS:-0}"
export SCHEDULER_TOTAL_STEPS="${SCHEDULER_TOTAL_STEPS:-0}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0}"
export SEED="${SEED:-0}"
export VAL_RATIO="${VAL_RATIO:-0.1}"
export VISION_CKPT="${VISION_CKPT:-/cephfs/shared/Yanbang/maniskill/pretrained/clip-vit-base-patch32}"
export CLIP_ENCODE_BATCH_SIZE="${CLIP_ENCODE_BATCH_SIZE:-64}"
export D_MODEL="${D_MODEL:-768}"
export N_LAYERS="${N_LAYERS:-8}"
export N_HEADS="${N_HEADS:-12}"
export DROPOUT="${DROPOUT:-0.1}"
export SKIP_EXISTING="${SKIP_EXISTING:-true}"
export REQUIRE_CUDA="${REQUIRE_CUDA:-true}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5}"
REQUIRE_IDLE_GPU="${REQUIRE_IDLE_GPU:-true}"
RUN_ID="${RUN_ID:-stpm_v5_d768_l8_obs6_gap2_seed0_6epoch_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-outputs/logs/${RUN_ID}}"

if [[ "${REQUIRE_IDLE_GPU}" != "true" && "${REQUIRE_IDLE_GPU}" != "false" ]]; then
  echo "REQUIRE_IDLE_GPU must be true or false; got ${REQUIRE_IDLE_GPU}." >&2
  exit 2
fi
IFS=',' read -r -a gpu_ids <<< "${GPU_IDS}"
if (( ${#gpu_ids[@]} != 6 )); then
  echo "Exactly six GPU ids are required; got ${GPU_IDS}." >&2
  exit 2
fi
if [[ "${REQUIRE_IDLE_GPU}" == "true" ]]; then
  active_gpu_pids="$(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d'
  )"
  if [[ -n "${active_gpu_pids}" ]]; then
    echo "GPU compute processes already exist: ${active_gpu_pids//$'\n'/, }" >&2
    exit 2
  fi
fi

mkdir -p "${LOG_DIR}"
assignments=("8" "9" "0,5" "7,3" "2,6" "1,4")
pids=()
logs=()
for index in "${!gpu_ids[@]}"; do
  gpu_id="${gpu_ids[$index]}"
  tasks="${assignments[$index]}"
  log="${LOG_DIR}/gpu${gpu_id}_tasks_${tasks//,/_}.log"
  echo "[Launch] gpu=${gpu_id} tasks=${tasks} log=${log}"
  (
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    TASK_IDS="${tasks}" \
    bash scripts/train_stpm_libero10_v3_all.sh
  ) >"${log}" 2>&1 &
  pids+=("$!")
  logs+=("${log}")
done

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "[Failure] worker log=${logs[$index]}" >&2
    tail -120 "${logs[$index]}" >&2 || true
    status=1
  fi
done
if (( status != 0 )); then
  echo "One or more STPM workers failed." >&2
  exit "${status}"
fi

for task_id in {0..9}; do
  root="outputs/train/${STPM_NAME_PREFIX}${task_id}"
  for artifact in \
    config.yaml \
    state_norm.json \
    checkpoints/reward_best.pt \
    checkpoints/reward_best_endpoint.pt \
    checkpoints/reward_final.pt; do
    if [[ ! -f "${root}/${artifact}" ]]; then
      echo "Missing completed task ${task_id} artifact: ${root}/${artifact}" >&2
      exit 1
    fi
  done
done
echo "All ten STPM tasks completed. Logs: ${LOG_DIR}"
