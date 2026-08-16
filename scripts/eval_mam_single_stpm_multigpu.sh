#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PATH="/cephfs/shared/Yanbang/envs/lerobot0.5.1/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/.torch-cache}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${REPO_ROOT}/.cache/numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
export LIBERO_ASSETS_PATH="${LIBERO_ASSETS_PATH:-${REPO_ROOT}/.cache/libero/assets}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

: "${POLICY_PATH:?Set POLICY_PATH to a MAM pretrained_model checkpoint}"
STPM_PREFIX="${STPM_PREFIX:-outputs/train/stpm_libero10_v3_large_d512_l4_task}"
STPM_CHECKPOINT_NAME="${STPM_CHECKPOINT_NAME:-reward_best.pt}"
EVAL_DATASET_REPO_ID="${EVAL_DATASET_REPO_ID:-local/libero10_100first50_refmix_eval}"
EVAL_DATASET_ROOT="${EVAL_DATASET_ROOT:-data/libero10_mam/libero10_100first50_refmix_eval}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-5}"
SEED="${SEED:-1000}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
TASK_IDS="${TASK_IDS:-0,1,2,3,4,5,6,7,8,9}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval/mam_single_stpm_$(date +%Y%m%d_%H%M%S)}"
REQUIRE_IDLE_GPU="${REQUIRE_IDLE_GPU:-true}"
DRY_RUN="${DRY_RUN:-false}"
RESUME="${RESUME:-false}"

if [[ ! -d "${POLICY_PATH}" ]]; then
  echo "Policy path does not exist: ${POLICY_PATH}" >&2
  exit 2
fi
POLICY_PATH="$(cd "${POLICY_PATH}" && pwd)"
if [[ ! -f "${EVAL_DATASET_ROOT}/meta/info.json" ]]; then
  echo "Eval dataset does not exist: ${EVAL_DATASET_ROOT}" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" && "${RESUME}" != true ]]; then
  echo "Refusing to overwrite existing output: ${OUTPUT_DIR}" >&2
  exit 2
fi
if ! [[ "${EPISODES_PER_TASK}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPISODES_PER_TASK must be positive; got ${EPISODES_PER_TASK}." >&2
  exit 2
fi
if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "SEED must be non-negative; got ${SEED}." >&2
  exit 2
fi
for name in REQUIRE_IDLE_GPU DRY_RUN RESUME; do
  value="${!name}"
  if [[ "${value}" != true && "${value}" != false ]]; then
    echo "${name} must be true or false; got ${value}." >&2
    exit 2
  fi
done

IFS=',' read -r -a gpu_ids <<<"${GPU_IDS}"
if (( ${#gpu_ids[@]} < 1 || ${#gpu_ids[@]} > 10 )); then
  echo "GPU_IDS must contain between 1 and 10 GPU ids; got ${GPU_IDS}." >&2
  exit 2
fi
IFS=',' read -r -a task_ids <<<"${TASK_IDS}"
if (( ${#task_ids[@]} < 1 || ${#task_ids[@]} > 10 )); then
  echo "TASK_IDS must contain between 1 and 10 task ids; got ${TASK_IDS}." >&2
  exit 2
fi
declare -A seen_task_ids=()
for task_id in "${task_ids[@]}"; do
  if ! [[ "${task_id}" =~ ^[0-9]$ ]]; then
    echo "TASK_IDS entries must be integers from 0 through 9; got ${task_id}." >&2
    exit 2
  fi
  if [[ -n "${seen_task_ids[${task_id}]:-}" ]]; then
    echo "TASK_IDS contains duplicate task ${task_id}." >&2
    exit 2
  fi
  seen_task_ids["${task_id}"]=1
done
if [[ "${REQUIRE_IDLE_GPU}" == true && "${DRY_RUN}" != true ]]; then
  for gpu_id in "${gpu_ids[@]}"; do
    active="$(nvidia-smi --id="${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
    if [[ -n "${active}" ]]; then
      echo "GPU ${gpu_id} is already in use by PID(s): ${active//$'\n'/, }." >&2
      exit 2
    fi
  done
fi

mkdir -p "${OUTPUT_DIR}/preflight"
uv run python - \
  "${STPM_PREFIX}" "${STPM_CHECKPOINT_NAME}" \
  "${EVAL_DATASET_REPO_ID}" "${EVAL_DATASET_ROOT}" \
  "${EPISODES_PER_TASK}" "${OUTPUT_DIR}/preflight" <<'PY'
import json
import sys
from pathlib import Path

from lerobot.datasets import LeRobotDatasetMetadata

prefix, checkpoint_name, repo_id, root_text, per_task_text, output_text = sys.argv[1:]
root = Path(root_text)
output = Path(output_text)
per_task = int(per_task_text)
roots = {}
checkpoints = {}
for task_id in range(10):
    stpm_root = Path(f"{prefix}{task_id}")
    config_path = stpm_root / "config.yaml"
    state_norm_path = stpm_root / "state_norm.json"
    checkpoint_path = stpm_root / "checkpoints" / checkpoint_name
    for path in (config_path, state_norm_path, checkpoint_path):
        if not path.is_file():
            raise SystemExit(f"Missing STPM task {task_id} artifact: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    identity = config.get("split_identity", {})
    if identity.get("tasks") != [str(task_id)]:
        raise SystemExit(
            f"STPM task identity mismatch for task {task_id}: {identity.get('tasks')!r}"
        )
    key = f"libero_10/{task_id}"
    roots[key] = str(stpm_root)
    checkpoints[key] = str(checkpoint_path)

manifest = json.loads((root / "meta" / "libero_pipeline.json").read_text(encoding="utf-8"))
expected_masks = ["points", "3D_points", "3D_points", "pose_motion_planning", "mix0"]
if manifest.get("mask_types") != expected_masks:
    raise SystemExit(
        f"Eval mask protocol mismatch: expected {expected_masks}, got {manifest.get('mask_types')}"
    )
metadata = LeRobotDatasetMetadata(repo_id, root=root)
by_task = {}
for row in metadata.episodes:
    task_id = int(row["libero/task_id"])
    by_task.setdefault(task_id, []).append(int(row["episode_index"]))
for task_id in range(10):
    episodes = sorted(by_task.get(task_id, []))[:per_task]
    if len(episodes) != per_task:
        raise SystemExit(f"Task {task_id} has only {len(episodes)} eval episodes; need {per_task}.")
    (output / f"episodes_task{task_id}.json").write_text(
        json.dumps(episodes, separators=(",", ":")), encoding="utf-8"
    )
(output / "stpm_roots.json").write_text(
    json.dumps(roots, separators=(",", ":")), encoding="utf-8"
)
(output / "stpm_checkpoints.json").write_text(
    json.dumps(checkpoints, separators=(",", ":")), encoding="utf-8"
)
PY

roots="$(<"${OUTPUT_DIR}/preflight/stpm_roots.json")"
checkpoints="$(<"${OUTPUT_DIR}/preflight/stpm_checkpoints.json")"

run_wave() {
  local tasks=("$@")
  local pids=()
  local logs=()
  local status=0
  for index in "${!tasks[@]}"; do
    task_id="${tasks[$index]}"
    gpu_id="${gpu_ids[$index]}"
    episodes="$(<"${OUTPUT_DIR}/preflight/episodes_task${task_id}.json")"
    worker_dir="${OUTPUT_DIR}/task_${task_id}"
    worker_log="${OUTPUT_DIR}/task_${task_id}.log"
    mkdir -p "${worker_dir}"
    cmd=(
      uv run python -m lerobot.scripts.lerobot_eval
      --policy.path="${POLICY_PATH}"
      --policy.device=cuda
      --policy.stpm_paths="${roots}"
      --policy.stpm_checkpoint_paths="${checkpoints}"
      --policy.mam_eval_dataset_repo_id="${EVAL_DATASET_REPO_ID}"
      --policy.mam_eval_dataset_root="${EVAL_DATASET_ROOT}"
      --policy.mam_eval_episodes="${episodes}"
      --policy.use_language_conditioning=true
      --policy.language_tokenizer_name=/cephfs/shared/Yanbang/maniskill/pretrained/clip-vit-base-patch32
      --env.type=libero
      --env.task=libero_10
      --env.task_ids="[${task_id}]"
      --env.control_mode=absolute
      --env.observation_height=128
      --env.observation_width=128
      --env.num_steps_wait=0
      --env.max_parallel_tasks=1
      --eval.n_episodes="${EPISODES_PER_TASK}"
      --eval.batch_size=1
      --eval.use_async_envs=false
      --output_dir="${worker_dir}"
      --job_name="mam_single_stpm_seed${SEED}_task${task_id}"
      --seed="${SEED}"
    )
    echo "task=${task_id} gpu=${gpu_id} episodes=${episodes}"
    if [[ "${DRY_RUN}" == true ]]; then
      printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu_id}"
      printf '%q ' "${cmd[@]}"
      printf '\n'
    else
      (export CUDA_VISIBLE_DEVICES="${gpu_id}"; "${cmd[@]}") >"${worker_log}" 2>&1 &
      pids+=("$!")
      logs+=("${worker_log}")
    fi
  done
  if [[ "${DRY_RUN}" == true ]]; then
    return
  fi
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "Eval worker failed: ${logs[$index]}" >&2
      tail -120 "${logs[$index]}" >&2 || true
      status=1
    fi
  done
  if (( status != 0 )); then
    return "${status}"
  fi
  if grep -Eni "using rollout step ratio as progress fallback|No STPM configured" "${logs[@]}"; then
    echo "Detected forbidden STPM fallback." >&2
    return 1
  fi
}

for ((start=0; start<${#task_ids[@]}; start+=${#gpu_ids[@]})); do
  wave=()
  for ((offset=0; offset<${#gpu_ids[@]} && start+offset<${#task_ids[@]}; offset++)); do
    wave+=("${task_ids[$((start + offset))]}")
  done
  run_wave "${wave[@]}"
done

if [[ "${DRY_RUN}" == true ]]; then
  echo "DRY_RUN completed."
  exit 0
fi

uv run python - "${OUTPUT_DIR}" "${POLICY_PATH}" "${STPM_PREFIX}" "${SEED}" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

output, policy_path, stpm_prefix, seed_text = sys.argv[1:]
output = Path(output)
rows = []
per_task = {}
eval_seconds = 0.0
for task_id in range(10):
    path = output / f"task_{task_id}" / "eval_info.json"
    if not path.is_file():
        raise SystemExit(f"Missing worker result: {path}")
    info = json.loads(path.read_text(encoding="utf-8"))
    task_rows = info.get("per_episode", [])
    if not task_rows:
        raise SystemExit(f"No per_episode records in {path}")
    rows.extend(task_rows)
    eval_seconds += float(info["overall"]["eval_s"])
    per_task[str(task_id)] = 100.0 * sum(bool(row["success"]) for row in task_rows) / len(task_rows)

by_type = defaultdict(list)
by_slot = defaultdict(list)
for row in rows:
    by_type[str(row["mask_type"])].append(bool(row["success"]))
    by_slot[str(row["mask_type_slot"])].append(bool(row["success"]))
successes = [bool(row["success"]) for row in rows]
summary = {
    "policy_path": policy_path,
    "stpm_prefix": stpm_prefix,
    "seed": int(seed_text),
    "overall": {
        "pc_success": 100.0 * sum(successes) / len(successes),
        "n_episodes": len(successes),
        "sum_worker_eval_s": eval_seconds,
        "mean_episode_s": eval_seconds / len(successes),
    },
    "per_task_success": per_task,
    "per_mask_type_success": {
        key: 100.0 * sum(values) / len(values) for key, values in sorted(by_type.items())
    },
    "per_mask_slot_success": {
        key: 100.0 * sum(values) / len(values) for key, values in sorted(by_slot.items())
    },
    "per_episode": rows,
}
(output / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
