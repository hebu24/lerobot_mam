#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CONDA_ENV_PATH="${CONDA_ENV_PATH:-/cephfs/shared/Yanbang/envs/lerobot0.5.1}"
export PATH="${CONDA_ENV_PATH}/bin:${PATH}"
PYTHON="${CONDA_ENV_PATH}/bin/python"

export DATASET_REPO_ID=local/libero10_500_train
export DATASET_ROOT=/cephfs/shared/Yanbang/lerobot/mam_lerobot0.5.1/lerobot_mam/data/libero10_mam/libero10_500_train
export MAM_EVAL_DATASET_REPO_ID=local/libero10_100_eval
export MAM_EVAL_DATASET_ROOT=/cephfs/shared/Yanbang/lerobot/mam_lerobot0.5.1/lerobot_mam/data/libero10_mam/libero10_100_eval

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NUM_GPUS="${NUM_GPUS:-4}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export MIXED_PRECISION=bf16
export NUM_WORKERS=8
export PREFETCH_FACTOR=4
export PERSISTENT_WORKERS=true

export STEPS=150000
export EVAL_FREQ=5000
export SAVE_FREQ=10000
export LOG_FREQ=200
export ENABLE_EVAL=true
# MAM interprets this globally. The selected prefix is audited below to contain
# exactly five episodes for each of the ten LIBERO tasks.
export EVAL_N_EPISODES=50
export EVAL_BATCH_SIZE=1
export EVAL_USE_ASYNC_ENVS=false
export ENV_TASK=libero_10
export ENV_TASK_IDS='[0,1,2,3,4,5,6,7,8,9]'
export ENV_CONTROL_MODE=absolute
export ENV_OBSERVATION_HEIGHT=128
export ENV_OBSERVATION_WIDTH=128
export ENV_MAX_PARALLEL_TASKS=1

export MASK_TYPE=random_mask
export MASK_TYPES=random_mask
export TRAIN_MASK_TYPES=random_mask
export EVAL_MASK_TYPES=random_mask
export MASK_LOSS_MODE=average
export MASK_INPAINTING=false
export MASK_PADDING_LOSS=true
export DO_MASK_LOSS_FOR_PADDING=true
export MAS_SHORT_WINDOW_HORIZON=0
export MAS_LONG_BACKWARD_LENGTH=0
export MAS_LONG_FORWARD_LENGTH=64
export MAS_LONG_FEATURE_DIM=128

export LEARNING_RATE=1e-4
export WEIGHT_DECAY=1e-6
export WARMUP_STEPS=500
export GRAD_CLIP_NORM=10.0
export SEED=1000
export CUDNN_DETERMINISTIC=false
export PRETRAINED_BACKBONE_WEIGHTS=null
export PUSH_TO_HUB=false
export WANDB_ENABLE=false

export STPM_BASE_DIR=outputs/train
export STPM_NAME_PREFIX="${STPM_NAME_PREFIX:-stpm_libero10_v4_maniskill_d768_l8_obs6_gap2_seed42_20260729_task}"
export LIBERO_ASSETS_PATH="${LIBERO_ASSETS_PATH:-/root/.cache/libero/assets}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/scripts/libero_config}"
export REQUIRE_IDLE_GPU="${REQUIRE_IDLE_GPU:-true}"
export RESUME_CONFIG_PATH="${RESUME_CONFIG_PATH:-}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export JOB_NAME="${JOB_NAME:-mam_libero10_500train_100eval5ptask_150k_4gpu_maniskill_short0_long64_dim128_avgmse_seed1000_${RUN_ID}}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${JOB_NAME}}"

resume_args=()
if [[ -n "${RESUME_CONFIG_PATH}" ]]; then
  if [[ ! -f "${RESUME_CONFIG_PATH}" ]]; then
    echo "Resume config does not exist: ${RESUME_CONFIG_PATH}" >&2
    exit 2
  fi
  if [[ ! -d "${OUTPUT_DIR}" ]]; then
    echo "Resume output does not exist: ${OUTPUT_DIR}" >&2
    exit 2
  fi
  resume_args+=(--resume=true --config_path="${RESUME_CONFIG_PATH}")
elif [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Output already exists: ${OUTPUT_DIR}" >&2
  exit 2
fi
if [[ "${REQUIRE_IDLE_GPU}" != true && "${REQUIRE_IDLE_GPU}" != false ]]; then
  echo "REQUIRE_IDLE_GPU must be true or false, got ${REQUIRE_IDLE_GPU}" >&2
  exit 2
fi

"${PYTHON}" - "${DATASET_ROOT}" "${MAM_EVAL_DATASET_ROOT}" "${STPM_BASE_DIR}" "${STPM_NAME_PREFIX}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

train_root, eval_root, stpm_base, stpm_prefix = map(Path, sys.argv[1:])


def load_split(root: Path, expected_split: str, expected_episodes: int):
    info = json.loads((root / "meta" / "info.json").read_text())
    manifest = json.loads((root / "meta" / "libero_pipeline.json").read_text())
    expected = {
        "stage": "absolute_to_mam",
        "dataset_split": expected_split,
        "action_representation": "osc_pose_absolute_goal",
        "policy_action_representation": "chunk_relative_se3",
        "relative_action_ready": True,
        "relative_action_stats_action_delta_indices": list(range(-1, 31)),
        "mask_types": ["random_mask"],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise SystemExit(f"{root}: expected {key}={value!r}, got {manifest.get(key)!r}")
    if info.get("total_episodes") != expected_episodes:
        raise SystemExit(
            f"{root}: expected total_episodes={expected_episodes}, got {info.get('total_episodes')}"
        )
    rows = []
    for path in sorted((root / "meta" / "episodes").glob("**/*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    if len(rows) != expected_episodes:
        raise SystemExit(f"{root}: expected {expected_episodes} episode rows, got {len(rows)}")
    return manifest, sorted(rows, key=lambda row: int(row["episode_index"]))


train_manifest, train_rows = load_split(train_root, "train", 500)
eval_manifest, eval_rows = load_split(eval_root, "eval", 100)
if Counter(int(row["libero/task_id"]) for row in train_rows) != Counter({task: 50 for task in range(10)}):
    raise SystemExit("Train dataset must contain exactly 50 episodes per LIBERO task")
if Counter(int(row["libero/task_id"]) for row in eval_rows) != Counter({task: 10 for task in range(10)}):
    raise SystemExit("Eval dataset must contain exactly 10 episodes per LIBERO task")
selected_eval = eval_rows[:50]
if Counter(int(row["libero/task_id"]) for row in selected_eval) != Counter({task: 5 for task in range(10)}):
    raise SystemExit("The first 50 eval episodes must contain exactly five episodes per task")
if any(row.get("libero/init_state") is None for row in selected_eval):
    raise SystemExit("Every selected eval episode must contain a raw libero/init_state")
if (
    train_manifest.get("source_root") == eval_manifest.get("source_root")
    and train_manifest.get("source_repo_id") == eval_manifest.get("source_repo_id")
):
    raise SystemExit("This launcher expects an independently sourced eval dataset")

missing = []
for task in range(10):
    root = stpm_base / f"{stpm_prefix}{task}"
    for relative in ("config.yaml", "checkpoints/reward_best.pt"):
        path = root / relative
        if not path.is_file():
            missing.append(str(path))
if missing:
    raise SystemExit("Missing STPM artifacts:\n" + "\n".join(missing))

print(
    "Dataset/STPM preflight OK: train=500 (50/task), eval=100; "
    "selected eval prefix=50 (5/task), independent source provenance"
)
PY

if [[ "${REQUIRE_IDLE_GPU}" == true ]]; then
  active_gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
  if [[ -n "${active_gpu_pids}" ]]; then
    echo "GPU is already used by compute process(es): ${active_gpu_pids//$'\n'/, }" >&2
    exit 2
  fi
fi

# The generic launcher predates independently recorded eval datasets and its
# shell-only overlap check compares source IDs without their source namespace.
# The stricter namespaced checks above replace that preflight for this run.
export SKIP_PREFLIGHT=true

echo "MAM long64 fixed-eval training"
echo "  train=${DATASET_ROOT}"
echo "  eval=${MAM_EVAL_DATASET_ROOT} (first 50 = 5/task)"
echo "  GPUs=${NUM_GPUS} (${CUDA_VISIBLE_DEVICES}), batch/GPU=${BATCH_SIZE}, effective_batch=$((NUM_GPUS * BATCH_SIZE))"
echo "  steps=${STEPS}, eval_freq=${EVAL_FREQ}, long_forward=${MAS_LONG_FORWARD_LENGTH}"
echo "  output=${OUTPUT_DIR}"
if [[ -n "${RESUME_CONFIG_PATH}" ]]; then
  echo "  resume=${RESUME_CONFIG_PATH}"
fi

exec bash scripts/run_mam_libero10_conda.sh \
  --policy.allow_independent_eval_source=true \
  --policy.language_tokenizer_name=/cephfs/shared/Yanbang/maniskill/pretrained/clip-vit-base-patch32 \
  "${resume_args[@]}"
