#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

STPM_SESSION="${STPM_SESSION:-stpm_maniskill_4gpu}"
STPM_NAME_PREFIX="${STPM_NAME_PREFIX:-stpm_libero10_v4_maniskill_d768_l8_obs6_gap2_seed42_20260729_task}"
JOB_NAME="${JOB_NAME:-mam_libero10_v4_refmix_150k_4gpu_maniskill_stpm_d768_l8_obs6gap2_short0_long64_dim128_avgmse_seed1000_multirankeval_20260729_211900}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${JOB_NAME}}"
LOG_PATH="${LOG_PATH:-outputs/logs/${JOB_NAME}.log}"

mkdir -p "$(dirname "${LOG_PATH}")"

while tmux has-session -t "${STPM_SESSION}" 2>/dev/null; do
  printf '[%s] Waiting for STPM tmux session %s to finish.\n' "$(date '+%F %T')" "${STPM_SESSION}"
  sleep 30
done

for task_id in {0..9}; do
  root="outputs/train/${STPM_NAME_PREFIX}${task_id}"
  for relative in config.yaml checkpoints/reward_best.pt checkpoints/reward_final.pt; do
    if [[ ! -f "${root}/${relative}" ]]; then
      echo "Missing completed STPM artifact: ${root}/${relative}" >&2
      exit 1
    fi
  done
done

if pgrep -f 'lerobot.scripts.lerobot_train_stpm' >/dev/null; then
  echo "STPM tmux ended but STPM training processes still exist; refusing to start MAM." >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "MAM output already exists: ${OUTPUT_DIR}" >&2
  exit 1
fi

printf '[%s] STPM complete. Starting 4-GPU MAM job %s.\n' "$(date '+%F %T')" "${JOB_NAME}"
exec env \
  CONDA_ENV_PATH="${CONDA_ENV_PATH:-${REPO_ROOT}/.venv}" \
  LIBERO_ASSETS_PATH="${LIBERO_ASSETS_PATH:-${REPO_ROOT}/.cache/libero/assets}" \
  LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/scripts/libero_config}" \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  NUM_GPUS=4 \
  STPM_NAME_PREFIX="${STPM_NAME_PREFIX}" \
  JOB_NAME="${JOB_NAME}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  MAS_SHORT_WINDOW_HORIZON=0 \
  MAS_LONG_BACKWARD_LENGTH=0 \
  MAS_LONG_FORWARD_LENGTH=64 \
  MAS_LONG_FEATURE_DIM=128 \
  MASK_LOSS_MODE=average \
  SEED=1000 \
  CUDNN_DETERMINISTIC=false \
  PYTHONUNBUFFERED=1 \
  bash scripts/run_mam_libero10_refmask_6gpu.sh >>"${LOG_PATH}" 2>&1
