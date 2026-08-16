#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PATH="/cephfs/shared/Yanbang/envs/lerobot0.5.1/bin:${PATH}"

RUN_ID="${RUN_ID:-stpm_maniskill_4gpu_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-outputs/logs/${RUN_ID}}"
STPM_NAME_PREFIX="${STPM_NAME_PREFIX:-stpm_libero10_v4_maniskill_d768_l8_obs6_gap2_seed42_20260729_task}"
mkdir -p "${LOG_DIR}"

# Balance the historical per-task runtimes while keeping each STPM single-GPU.
assignments=("8,9" "0,1" "5,7,2" "3,4,6")
pids=()
for gpu in 0 1 2 3; do
  tasks="${assignments[$gpu]}"
  log_path="${LOG_DIR}/gpu${gpu}_tasks_${tasks//,/_}.log"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" \
    TASK_IDS="${tasks}" \
    EPOCHS=2 \
    STEPS= \
    DATASET_REPO_ID=local/libero10_mam_v3_unfiltered_train \
    DATASET_ROOT=outputs/datasets/libero10_mam_v3_unfiltered_train \
    STPM_NAME_PREFIX="${STPM_NAME_PREFIX}" \
    N_OBS_STEPS=6 \
    FRAME_GAP=2 \
    BATCH_SIZE=32 \
    NUM_WORKERS=6 \
    PREFETCH_FACTOR=2 \
    LEARNING_RATE=5e-5 \
    WEIGHT_DECAY=5e-3 \
    ADAM_BETA1=0.9 \
    ADAM_BETA2=0.95 \
    ADAM_EPS=1e-8 \
    WARMUP_STEPS=1000 \
    SCHEDULER_TOTAL_STEPS=100000 \
    GRAD_CLIP_NORM=1.0 \
    SEED=42 \
    VAL_RATIO=0.1 \
    VISION_CKPT=/cephfs/shared/Yanbang/maniskill/pretrained/clip-vit-base-patch32 \
    CLIP_ENCODE_BATCH_SIZE=64 \
    D_MODEL=768 \
    N_LAYERS=8 \
    N_HEADS=12 \
    DROPOUT=0.1 \
    SKIP_EXISTING=true \
    bash scripts/train_stpm_libero10_v3_all.sh
  ) >"${log_path}" 2>&1 &
  pids+=("$!")
  printf "gpu=%s tasks=%s pid=%s log=%s\n" "${gpu}" "${tasks}" "$!" "${log_path}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if (( status != 0 )); then
  echo "At least one STPM worker failed. Inspect ${LOG_DIR}." >&2
  exit "${status}"
fi

for task_id in {0..9}; do
  output_dir="outputs/train/${STPM_NAME_PREFIX}${task_id}"
  test -f "${output_dir}/config.yaml"
  test -f "${output_dir}/checkpoints/reward_best.pt"
  test -f "${output_dir}/checkpoints/reward_final.pt"
done
echo "All ManiSkill-parameter STPM checkpoints completed."
