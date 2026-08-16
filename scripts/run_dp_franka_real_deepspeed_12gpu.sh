#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOSTFILE="${HOSTFILE:-${REPO_ROOT}/configs/deepspeed/franka_dp_12gpu.hostfile}"
REMOTE_USER="${REMOTE_USER:-root}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-true}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MASTER_PORT="${MASTER_PORT:-29531}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-FrankaReal_pick_up_front_dualcam_dualdino_aug_unet_dp_15hz_obs2_act8_pred16_12gpu_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/logs}"

TOTAL_ITERS="${TOTAL_ITERS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-10}"
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

mapfile -t HOSTS < <(awk '!/^[[:space:]]*(#|$)/ {print $1}' "${HOSTFILE}")
mapfile -t SLOTS < <(awk '!/^[[:space:]]*(#|$)/ {for (i=2; i<=NF; i++) if ($i ~ /^slots=/) {sub(/^slots=/, "", $i); print $i}}' "${HOSTFILE}")
if [[ ${#HOSTS[@]} -ne 2 || ${#SLOTS[@]} -ne 2 ]]; then
  echo "Expected exactly two valid hosts in ${HOSTFILE}" >&2
  exit 2
fi
if [[ "${SLOTS[0]}" != "8" || "${SLOTS[1]}" != "4" ]]; then
  echo "Expected an 8+4 GPU hostfile, got ${SLOTS[0]}+${SLOTS[1]}" >&2
  exit 2
fi

ssh_target() {
  local host="$1"
  if [[ "${host}" == *"@"* ]]; then
    printf '%s' "${host}"
  else
    printf '%s@%s' "${REMOTE_USER}" "${host}"
  fi
}

gpu_is_idle() {
  local target="$1"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "${target}" \
    'test -z "$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | sed "/^[[:space:]]*$/d")"'
}

TARGET0="$(ssh_target "${HOSTS[0]}")"
TARGET1="$(ssh_target "${HOSTS[1]}")"
MASTER_ADDR="${MASTER_ADDR:-${HOSTS[0]#*@}}"

echo "[controller] exp=${EXP_NAME}"
echo "[controller] hosts=${TARGET0}(${SLOTS[0]}) ${TARGET1}(${SLOTS[1]}); master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[controller] per_gpu_batch=${BATCH_SIZE}; global_batch=$((BATCH_SIZE * (SLOTS[0] + SLOTS[1])))"
echo "[controller] control_frequency=${CONTROL_FREQUENCY_HZ}Hz; obs=${OBS_HORIZON} action=${ACT_HORIZON} prediction=${PRED_HORIZON}"

if [[ "${WAIT_FOR_GPUS}" == "true" ]]; then
  while ! gpu_is_idle "${TARGET0}" || ! gpu_is_idle "${TARGET1}"; do
    echo "[controller] $(date '+%F %T') GPUs are busy; retrying in ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
  done
elif ! gpu_is_idle "${TARGET0}" || ! gpu_is_idle "${TARGET1}"; then
  echo "[controller] GPUs are currently busy and WAIT_FOR_GPUS=false" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
NODE0_LOG="${LOG_DIR}/${EXP_NAME}.node0.log"
NODE1_LOG="${LOG_DIR}/${EXP_NAME}.node1.log"

REMOTE_ENV=(
  "HOSTFILE=${HOSTFILE}"
  "MASTER_ADDR=${MASTER_ADDR}"
  "MASTER_PORT=${MASTER_PORT}"
  "EXP_NAME=${EXP_NAME}"
  "TOTAL_ITERS=${TOTAL_ITERS}"
  "BATCH_SIZE=${BATCH_SIZE}"
  "LR=${LR}"
  "OBS_HORIZON=${OBS_HORIZON}"
  "ACT_HORIZON=${ACT_HORIZON}"
  "PRED_HORIZON=${PRED_HORIZON}"
  "CONTROL_FREQUENCY_HZ=${CONTROL_FREQUENCY_HZ}"
  "NUM_DATALOAD_WORKERS=${NUM_DATALOAD_WORKERS}"
  "SAVE_START_ITER=${SAVE_START_ITER}"
  "SAVE_FREQ=${SAVE_FREQ}"
  "LOG_FREQ=${LOG_FREQ}"
)

printf -v NODE0_CMD '%q ' env "${REMOTE_ENV[@]}" bash "${REPO_ROOT}/scripts/run_dp_franka_real_deepspeed_node.sh" 0
printf -v NODE1_CMD '%q ' env "${REMOTE_ENV[@]}" bash "${REPO_ROOT}/scripts/run_dp_franka_real_deepspeed_node.sh" 1

echo "[controller] $(date '+%F %T') starting rank launchers; logs: ${NODE0_LOG}, ${NODE1_LOG}"
ssh -o BatchMode=yes "${TARGET0}" "${NODE0_CMD}" >"${NODE0_LOG}" 2>&1 &
PID0=$!
ssh -o BatchMode=yes "${TARGET1}" "${NODE1_CMD}" >"${NODE1_LOG}" 2>&1 &
PID1=$!

cleanup() {
  kill "${PID0}" "${PID1}" 2>/dev/null || true
  wait "${PID0}" "${PID1}" 2>/dev/null || true
}
trap cleanup INT TERM

STATUS0=0
STATUS1=0
wait "${PID0}" || STATUS0=$?
wait "${PID1}" || STATUS1=$?
trap - INT TERM

if (( STATUS0 != 0 || STATUS1 != 0 )); then
  echo "[controller] launch failed: node0=${STATUS0}, node1=${STATUS1}" >&2
  exit 1
fi
echo "[controller] $(date '+%F %T') training completed successfully"
