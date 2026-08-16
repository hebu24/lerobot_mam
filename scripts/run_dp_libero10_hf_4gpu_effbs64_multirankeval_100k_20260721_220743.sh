#!/usr/bin/env bash
set -euo pipefail

cd /cephfs/shared/Yanbang/lerobot/mam_lerobot0.5.1/lerobot_mam

export PATH="/cephfs/shared/Yanbang/envs/lerobot0.5.1/bin:${PATH}"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_GPUS=4
export POLICY_DEVICE=cuda
export LEROBOT_DDP_TIMEOUT_S=7200

export DATASET_REPO_ID=local/libero10_mam_v3_train
export DATASET_ROOT=outputs/datasets/libero10_mam_v3_train
export EVAL_DATASET_REPO_ID=local/libero10_mam_v3_eval
export EVAL_DATASET_ROOT=outputs/datasets/libero10_mam_v3_eval

export USE_RELATIVE_ACTIONS=true
export ENABLE_EVAL=true
export EVAL_FREQ=5000
export EVAL_N_EPISODES=5
export EVAL_BATCH_SIZE=1

export BATCH_SIZE=16
export STEPS=100000
export SAVE_FREQ=10000

export ENV_TASK=libero_10
export ENV_TASK_IDS='[0,1,2,3,4,5,6,7,8,9]'
export ENV_CONTROL_MODE=absolute
export ENV_OBSERVATION_HEIGHT=128
export ENV_OBSERVATION_WIDTH=128

export USE_LANGUAGE_CONDITIONING=true
export REQUIRE_FULL_DATASET=false
export REQUIRE_IDLE_GPU=true

export JOB_NAME=diffusion_libero10_v3_hf_100k_4gpu_effbs64_multirankeval_20260721_220743
export OUTPUT_DIR=outputs/train/${JOB_NAME}

LOG_DIR=outputs/logs
LOG_FILE=${LOG_DIR}/${JOB_NAME}.log
mkdir -p "${LOG_DIR}"

set +e
bash scripts/run_diffusion_libero10.sh \
  --policy.horizon=32 \
  --policy.n_action_steps=15 \
  --policy.use_language_conditioning=true \
  --policy.language_tokenizer_name=/cephfs/shared/Yanbang/maniskill/pretrained/clip-vit-base-patch32 \
  2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e

exit "${status}"
