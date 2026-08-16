#!/usr/bin/env bash
set -euo pipefail

# Queue a single-node 4-GPU Franka DP run on the current VM. This script is
# intended to live in tmux while an earlier four-GPU job finishes.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOSTFILE="${HOSTFILE:-${REPO_ROOT}/configs/deepspeed/franka_dp_4gpu.hostfile}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-true}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29531}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-FrankaReal_pick_up_front_dualcam_dualdino_aug_unet_dp_15hz_obs2_act8_pred16_4gpu_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/logs}"

TOTAL_ITERS="${TOTAL_ITERS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-5e-5}"
OBS_HORIZON="${OBS_HORIZON:-2}"
ACT_HORIZON="${ACT_HORIZON:-8}"
PRED_HORIZON="${PRED_HORIZON:-16}"
CONTROL_FREQUENCY_HZ="${CONTROL_FREQUENCY_HZ:-15}"
NUM_DATALOAD_WORKERS="${NUM_DATALOAD_WORKERS:-2}"
SAVE_START_ITER="${SAVE_START_ITER:-30000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-1000}"

case "${WAIT_FOR_GPUS}" in
  true|false) ;;
  *) echo "WAIT_FOR_GPUS must be true or false, got ${WAIT_FOR_GPUS}" >&2; exit 2 ;;
esac
if [[ ! "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "POLL_SECONDS must be a positive integer, got ${POLL_SECONDS}" >&2
  exit 2
fi
if [[ ! -f "${HOSTFILE}" ]]; then
  echo "Hostfile not found: ${HOSTFILE}" >&2
  exit 1
fi

gpu_is_idle() {
  test -z "$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d')"
}

echo "[controller] exp=${EXP_NAME}"
echo "[controller] host=$(hostname); GPUs=4; master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[controller] per_gpu_batch=${BATCH_SIZE}; global_batch=$((BATCH_SIZE * 4))"
echo "[controller] control_frequency=${CONTROL_FREQUENCY_HZ}Hz; obs=${OBS_HORIZON} action=${ACT_HORIZON} prediction=${PRED_HORIZON}"

if [[ "${WAIT_FOR_GPUS}" == "true" ]]; then
  while ! gpu_is_idle; do
    echo "[controller] $(date '+%F %T') GPUs are busy; retrying in ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
  done
elif ! gpu_is_idle; then
  echo "[controller] GPUs are currently busy and WAIT_FOR_GPUS=false" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
NODE_LOG="${LOG_DIR}/${EXP_NAME}.node0.log"
echo "[controller] $(date '+%F %T') starting four-GPU training; node log: ${NODE_LOG}"

STATUS=0
env \
  HOSTFILE="${HOSTFILE}" \
  MASTER_ADDR="${MASTER_ADDR}" \
  MASTER_PORT="${MASTER_PORT}" \
  EXP_NAME="${EXP_NAME}" \
  TOTAL_ITERS="${TOTAL_ITERS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  LR="${LR}" \
  OBS_HORIZON="${OBS_HORIZON}" \
  ACT_HORIZON="${ACT_HORIZON}" \
  PRED_HORIZON="${PRED_HORIZON}" \
  CONTROL_FREQUENCY_HZ="${CONTROL_FREQUENCY_HZ}" \
  NUM_DATALOAD_WORKERS="${NUM_DATALOAD_WORKERS}" \
  SAVE_START_ITER="${SAVE_START_ITER}" \
  SAVE_FREQ="${SAVE_FREQ}" \
  LOG_FREQ="${LOG_FREQ}" \
  bash "${REPO_ROOT}/scripts/run_dp_franka_real_deepspeed_node.sh" 0 \
  >"${NODE_LOG}" 2>&1 || STATUS=$?

if (( STATUS != 0 )); then
  echo "[controller] training failed with status ${STATUS}; inspect ${NODE_LOG}" >&2
  exit "${STATUS}"
fi
echo "[controller] $(date '+%F %T') training completed successfully"
