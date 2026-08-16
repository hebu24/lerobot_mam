#!/usr/bin/env bash
set -euo pipefail

# This worker is started once per GPU by the DeepSpeed launcher. DeepSpeed is
# used only as the heterogeneous multi-node process launcher; the existing
# Franka trainer owns the PyTorch DDP model, optimizer, and checkpoints.

MANISKILL_ROOT="${MANISKILL_ROOT:-/cephfs/shared/Yanbang/maniskill}"
PYTHON_BIN="${PYTHON_BIN:-/cephfs/shared/Yanbang/envs/maniskill_py311/bin/python}"
DEMO_PATH="${DEMO_PATH:-franka_train/data/franka_mam/pick_up_front_baseline_train.h5}"
DINO_MODEL_PATH="${DINO_MODEL_PATH:-Dino/dinov2-small}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${MANISKILL_ROOT}"
export PYTHONPATH="${MANISKILL_ROOT}:${MANISKILL_ROOT}/examples/baselines/diffusion_policy${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-maniskill}"
export LD_LIBRARY_PATH="$(dirname "${PYTHON_BIN}")/../lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

EXP_NAME="${EXP_NAME:-FrankaReal_pick_up_front_dp_multinode}"
SEED="${SEED:-1}"
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

FREQUENCY_ARGS=(
  --dataset "${MANISKILL_ROOT}/${DEMO_PATH}"
  --expected-hz "${CONTROL_FREQUENCY_HZ}"
  --obs-horizon "${OBS_HORIZON}"
  --action-horizon "${ACT_HORIZON}"
  --prediction-horizon "${PRED_HORIZON}"
)
if [[ "${RANK:-0}" == "0" ]]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/validate_franka_dp_frequency.py" \
    "${FREQUENCY_ARGS[@]}" \
    --manifest "${MANISKILL_ROOT}/runs/${EXP_NAME}/frequency_manifest.json"
  echo "[franka-dp] exp=${EXP_NAME} world_size=${WORLD_SIZE:-1} per_gpu_batch=${BATCH_SIZE} global_batch=$((BATCH_SIZE * ${WORLD_SIZE:-1}))"
  echo "[franka-dp] rate=${CONTROL_FREQUENCY_HZ}Hz obs=${OBS_HORIZON} action=${ACT_HORIZON} prediction=${PRED_HORIZON} steps=${TOTAL_ITERS}"
else
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/validate_franka_dp_frequency.py" \
    "${FREQUENCY_ARGS[@]}" --quiet
fi

exec "${PYTHON_BIN}" franka_train/train_baseline_franka_multigpu.py \
  --exp-name "${EXP_NAME}" \
  --seed "${SEED}" \
  --torch-deterministic \
  --cuda \
  --no-track \
  --no-capture-video \
  --env-id FrankaReal-v1 \
  --action-dim 7 \
  --demo-path "${DEMO_PATH}" \
  --action-norm-path "${DEMO_PATH}" \
  --noise-model Unet \
  --vision-encoder dino2 \
  --dino-model-path "${DINO_MODEL_PATH}" \
  --dino-data-aug \
  --total-iters "${TOTAL_ITERS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --obs-horizon "${OBS_HORIZON}" \
  --act-horizon "${ACT_HORIZON}" \
  --pred-horizon "${PRED_HORIZON}" \
  --diffusion-step-embed-dim 64 \
  --unet-dims 256 512 1024 \
  --n-groups 8 \
  --obs-mode rgb \
  --log-freq "${LOG_FREQ}" \
  --eval-freq 0 \
  --valid-freq 0 \
  --num-validation-set 0 \
  --num-eval-episodes 0 \
  --num-eval-envs 0 \
  --save-start-iter "${SAVE_START_ITER}" \
  --save-freq "${SAVE_FREQ}" \
  --num-dataload-workers "${NUM_DATALOAD_WORKERS}" \
  --control-mode pd_ee_pose \
  --sim-backend physx_cpu \
  --demo-type franka_real_baseline
