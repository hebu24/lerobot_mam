#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

WAIT_TMUX_SESSION="${WAIT_TMUX_SESSION:-mam_stpm_reeval_20260809}"
RUN_ID="${RUN_ID:-20260809_v3_obsgap_after_current}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
POLL_SECONDS="${POLL_SECONDS:-60}"

STPM_VARIANTS="${STPM_VARIANTS:-obs3_gap1,obs6_gap1,obs6_gap2}"
REEVAL_VARIANTS="${REEVAL_VARIANTS:-v3_obs3_gap1_d512_l4_seed0_6epoch_best,v3_obs3_gap1_d512_l4_seed0_6epoch_endpoint,v3_obs6_gap1_d512_l4_seed0_6epoch_best,v3_obs6_gap1_d512_l4_seed0_6epoch_endpoint,v3_obs6_gap2_d512_l4_seed0_6epoch_best,v3_obs6_gap2_d512_l4_seed0_6epoch_endpoint}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval/mam_stpm_reeval_v3_obsgap_from100k_8gpu_${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-outputs/logs/stpm_v3_obsgap_8gpu_${RUN_ID}}"

echo "[Queue] waiting for tmux session to finish: ${WAIT_TMUX_SESSION}"
while tmux has-session -t "${WAIT_TMUX_SESSION}" 2>/dev/null; do
  sleep "${POLL_SECONDS}"
done

echo "[Queue] current reeval session ended; training missing v3 obs/gap STPM variants"
RUN_ID="stpm_v3_obsgap_${RUN_ID}" \
LOG_ROOT="${LOG_ROOT}" \
GPU_IDS="${GPU_IDS}" \
VARIANT_LIST="${STPM_VARIANTS}" \
bash scripts/train_stpm_libero10_v3_obsgap_variants_8gpu.sh

echo "[Queue] reeval v3 obs/gap STPM variants"
RUN_ID="${RUN_ID}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
GPU_IDS="${GPU_IDS}" \
VARIANT_LIST="${REEVAL_VARIANTS}" \
bash scripts/reeval_mam_stpm_variants_from100k_8gpu.sh

echo "[Queue] done: ${OUTPUT_ROOT}"
