#!/usr/bin/env bash
set -euo pipefail

cd /cephfs/shared/Yanbang/lerobot/mam_lerobot0.5.1/lerobot_mam

TRAIN_SESSION=dp_libero10_hf_4gpu_effbs64_multirankeval_100k_20260721_220743
JOB_NAME=diffusion_libero10_v3_hf_100k_4gpu_effbs64_multirankeval_20260721_220743
OUTPUT_DIR=outputs/train/${JOB_NAME}
TRAIN_LOG=outputs/logs/${JOB_NAME}.log
WATCH_LOG=outputs/logs/${JOB_NAME}_watch.log
TRAIN_METRICS=${OUTPUT_DIR}/logs/train_metrics.jsonl
EVAL_METRICS=${OUTPUT_DIR}/logs/eval_metrics.jsonl
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"

mkdir -p "$(dirname "${WATCH_LOG}")"

while true; do
  {
    echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
    if tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
      echo "tmux=running ${TRAIN_SESSION}"
    else
      echo "tmux=missing ${TRAIN_SESSION}"
    fi

    echo "-- latest train metric --"
    tail -1 "${TRAIN_METRICS}" 2>/dev/null || true

    echo "-- latest eval metric --"
    tail -1 "${EVAL_METRICS}" 2>/dev/null || true

    echo "-- latest rank eval files --"
    latest_rank_dir="$(find "${OUTPUT_DIR}/eval" -maxdepth 1 -type d -name 'rank_metrics_step_*' 2>/dev/null | sort | tail -1 || true)"
    if [[ -n "${latest_rank_dir}" ]]; then
      echo "${latest_rank_dir}"
      find "${latest_rank_dir}" -maxdepth 1 \( -name 'rank_*.json' -o -name 'merged.json' \) -type f 2>/dev/null | sort || true
    fi

    echo "-- gpu --"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

    if ! pgrep -af "${JOB_NAME}" >/dev/null; then
      echo "-- issue --"
      echo "training process for ${JOB_NAME} not found"
      echo "-- train log tail --"
      tail -160 "${TRAIN_LOG}" 2>/dev/null || true
    fi
    echo
  } >>"${WATCH_LOG}" 2>&1

  sleep "${INTERVAL_SECONDS}"
done
