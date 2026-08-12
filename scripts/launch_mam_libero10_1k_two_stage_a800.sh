#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARGET="${TARGET:-root@10.233.118.2}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SESSION="${SESSION:-mam1k_${RUN_ID}}"
RUN_ROOT="${RUN_ROOT:-outputs/train/mam_libero10_1k_random090k_refmix180k_4a800_${RUN_ID}}"
LOG_PATH="${LOG_PATH:-${RUN_ROOT}/orchestrator.log}"
WORKER="${REPO_ROOT}/scripts/train_mam_libero10_1k_two_stage_a800_4gpu.sh"

if [[ ! "${RUN_ID}" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  echo "RUN_ID must use YYYYMMDD_HHMMSS, got ${RUN_ID}." >&2
  exit 2
fi
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "${TARGET}" true; then
  echo "Cannot reach ${TARGET} with non-interactive SSH." >&2
  exit 1
fi

remote_gpu_summary="$(
  ssh -o BatchMode=yes "${TARGET}" \
    'hostname; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader; nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed "/^[[:space:]]*$/d"'
)"
if [[ "${remote_gpu_summary}" != *zhangchenyu4-0* ]]; then
  echo "Target is not the expected A800 VM: ${remote_gpu_summary}" >&2
  exit 2
fi
if (( $(grep -c 'NVIDIA A800' <<<"${remote_gpu_summary}") != 4 )); then
  echo "Expected four A800 GPUs on ${TARGET}:" >&2
  echo "${remote_gpu_summary}" >&2
  exit 2
fi
if ssh -o BatchMode=yes "${TARGET}" \
  'test -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed "/^[[:space:]]*$/d")"'; then
  echo "A800 GPUs are busy on ${TARGET}; refusing to start." >&2
  exit 1
fi
if ssh -o BatchMode=yes "${TARGET}" "tmux has-session -t '${SESSION}'" 2>/dev/null; then
  echo "tmux session already exists on ${TARGET}: ${SESSION}" >&2
  exit 2
fi

ssh -o BatchMode=yes "${TARGET}" "mkdir -p '${REPO_ROOT}/${RUN_ROOT}'"
ssh -o BatchMode=yes "${TARGET}" \
  "tmux new-session -d -s '${SESSION}' \"cd '${REPO_ROOT}' && env RUN_ID='${RUN_ID}' RUN_ROOT='${RUN_ROOT}' bash '${WORKER}' >'${LOG_PATH}' 2>&1\""

if ! ssh -o BatchMode=yes "${TARGET}" "tmux has-session -t '${SESSION}'"; then
  echo "Failed to create tmux session ${SESSION} on ${TARGET}." >&2
  exit 1
fi

echo "Started LIBERO-10 1k two-stage MAM training"
echo "  target=${TARGET} (4xA800)"
echo "  session=${SESSION}"
echo "  run_root=${REPO_ROOT}/${RUN_ROOT}"
echo "  log=${REPO_ROOT}/${LOG_PATH}"
echo "Monitor: ssh ${TARGET} \"tail -f '${REPO_ROOT}/${LOG_PATH}'\""
