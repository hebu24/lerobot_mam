#!/usr/bin/env bash
set -euo pipefail

cd /cephfs/shared/Yanbang/lerobot/mam_lerobot0.5.1/lerobot_mam

export PATH="/cephfs/shared/Yanbang/envs/lerobot0.5.1/bin:${PATH}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PWD}/.uv-cache}"
export HF_HOME="${HF_HOME:-${PWD}/.hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TORCH_HOME="${TORCH_HOME:-${PWD}/.torch-cache}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${PWD}/.cache/numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PWD}/.cache/matplotlib}"
export LIBERO_ASSETS_PATH="${LIBERO_ASSETS_PATH:-${PWD}/.cache/libero/assets}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-outputs/train/diffusion_libero10_v3_hf_100k_4gpu_effbs64_noeval_20260721_212245}"
POLICY_PATH="${POLICY_PATH:-${TRAIN_OUTPUT_DIR}/checkpoints/last/pretrained_model}"

EVAL_DATASET_REPO_ID="${EVAL_DATASET_REPO_ID:-local/libero10_mam_v3_eval}"
EVAL_DATASET_ROOT="${EVAL_DATASET_ROOT:-outputs/datasets/libero10_mam_v3_eval}"
EVAL_N_EPISODES="${EVAL_N_EPISODES:-5}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
SEED="${SEED:-1000}"
GPU_IDS_TEXT="${GPU_IDS:-0,1,2,3}"
TASK_GROUPS_TEXT="${TASK_GROUPS:-0,1,2 3,4,5 6,7 8,9}"
JOB_NAME="${JOB_NAME:-parallel_eval_libero10_4gpu_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval/${JOB_NAME}}"
REQUIRE_IDLE_GPU="${REQUIRE_IDLE_GPU:-true}"
DRY_RUN="${DRY_RUN:-false}"

is_bool() {
  [[ "$1" == "true" || "$1" == "false" ]]
}

for value_name in REQUIRE_IDLE_GPU DRY_RUN; do
  value="${!value_name}"
  if ! is_bool "${value}"; then
    echo "${value_name} must be true or false, got: ${value}" >&2
    exit 2
  fi
done

if ! [[ "${EVAL_N_EPISODES}" =~ ^[0-9]+$ ]] || (( EVAL_N_EPISODES <= 0 )); then
  echo "EVAL_N_EPISODES must be positive, got: ${EVAL_N_EPISODES}" >&2
  exit 2
fi
if ! [[ "${EVAL_BATCH_SIZE}" =~ ^[0-9]+$ ]] || (( EVAL_BATCH_SIZE <= 0 )); then
  echo "EVAL_BATCH_SIZE must be positive, got: ${EVAL_BATCH_SIZE}" >&2
  exit 2
fi

IFS=',' read -r -a GPU_IDS_ARRAY <<< "${GPU_IDS_TEXT}"
read -r -a TASK_GROUPS_ARRAY <<< "${TASK_GROUPS_TEXT}"
if (( ${#GPU_IDS_ARRAY[@]} != ${#TASK_GROUPS_ARRAY[@]} )); then
  echo "GPU_IDS and TASK_GROUPS must have the same number of entries." >&2
  echo "  GPU_IDS=${GPU_IDS_TEXT}" >&2
  echo "  TASK_GROUPS=${TASK_GROUPS_TEXT}" >&2
  exit 2
fi

if [[ ! -d "${POLICY_PATH}" ]]; then
  echo "Policy path not found: ${POLICY_PATH}" >&2
  echo "Pass POLICY_PATH=.../checkpoints/<step>/pretrained_model after a checkpoint is available." >&2
  exit 2
fi
if [[ ! -f "${EVAL_DATASET_ROOT}/meta/info.json" ]]; then
  echo "Eval dataset not found: ${EVAL_DATASET_ROOT}" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" && "${DRY_RUN}" != "true" ]]; then
  echo "OUTPUT_DIR already exists: ${OUTPUT_DIR}" >&2
  exit 2
fi

if [[ "${REQUIRE_IDLE_GPU}" == "true" && "${DRY_RUN}" != "true" ]]; then
  active_gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
  if [[ -n "${active_gpu_pids}" ]]; then
    echo "GPU is already used by compute process(es): ${active_gpu_pids//$'\n'/, }" >&2
    echo "Run this evaluator when training is stopped/finished, or set REQUIRE_IDLE_GPU=false intentionally." >&2
    exit 2
  fi
fi

mkdir -p "${OUTPUT_DIR}"
SPLIT_DIR="${OUTPUT_DIR}/splits"
mkdir -p "${SPLIT_DIR}"

uv run python - "${EVAL_DATASET_ROOT}" "${EVAL_N_EPISODES}" "${SPLIT_DIR}" "${TASK_GROUPS_ARRAY[@]}" <<'PY'
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
n_episodes = int(sys.argv[2])
split_dir = Path(sys.argv[3])
task_groups = sys.argv[4:]

episode_paths = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
if not episode_paths:
    raise SystemExit(f"No episode metadata found under {root / 'meta' / 'episodes'}")

rows = []
for path in episode_paths:
    rows.extend(pq.read_table(path).to_pylist())
if not rows:
    raise SystemExit("Eval episode metadata is empty")

columns = set(rows[0])
task_key = "libero/task_id" if "libero/task_id" in columns else "task_id"
if task_key not in columns:
    raise SystemExit(f"No task id column found in eval metadata: {sorted(columns)}")

by_task: dict[int, list[int]] = {}
for row in rows:
    by_task.setdefault(int(row[task_key]), []).append(int(row["episode_index"]))
for task_id, episode_ids in by_task.items():
    by_task[task_id] = sorted(episode_ids)

summary = []
for worker_id, group_text in enumerate(task_groups):
    task_ids = [int(value) for value in group_text.split(",") if value != ""]
    if not task_ids:
        raise SystemExit(f"Empty task group at worker {worker_id}")
    selected_episodes: list[int] = []
    for task_id in task_ids:
        episode_ids = by_task.get(task_id, [])
        if len(episode_ids) < n_episodes:
            raise SystemExit(
                f"Task {task_id} has {len(episode_ids)} eval episode(s), need {n_episodes}"
            )
        selected_episodes.extend(episode_ids[:n_episodes])

    task_ids_json = json.dumps(task_ids, separators=(",", ":"))
    episodes_json = json.dumps(selected_episodes, separators=(",", ":"))
    (split_dir / f"tasks_{worker_id}.txt").write_text(task_ids_json, encoding="utf-8")
    (split_dir / f"episodes_{worker_id}.txt").write_text(episodes_json, encoding="utf-8")
    summary.append(
        {
            "worker_id": worker_id,
            "task_ids": task_ids,
            "episode_indices": selected_episodes,
            "n_episodes": len(selected_episodes),
        }
    )

(split_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False))
PY

echo "Parallel LIBERO-10 eval"
echo "  policy=${POLICY_PATH}"
echo "  output=${OUTPUT_DIR}"
echo "  GPUs=${GPU_IDS_TEXT}"
echo "  task_groups=${TASK_GROUPS_TEXT}"
echo "  eval_n_episodes_per_task=${EVAL_N_EPISODES}, eval_batch_size=${EVAL_BATCH_SIZE}"

pids=()
worker_logs=()
for worker_id in "${!GPU_IDS_ARRAY[@]}"; do
  gpu_id="${GPU_IDS_ARRAY[${worker_id}]}"
  task_ids="$(<"${SPLIT_DIR}/tasks_${worker_id}.txt")"
  episodes="$(<"${SPLIT_DIR}/episodes_${worker_id}.txt")"
  task_label="${task_ids//[\[\],]/_}"
  task_label="${task_label##_}"
  task_label="${task_label%%_}"
  worker_output_dir="${OUTPUT_DIR}/gpu${gpu_id}_tasks_${task_label}"
  worker_log="${OUTPUT_DIR}/gpu${gpu_id}_tasks_${task_label}.log"
  worker_logs+=("${worker_log}")
  mkdir -p "${worker_output_dir}"

  cmd=(
    uv run python -m lerobot.scripts.lerobot_eval
    --policy.path="${POLICY_PATH}"
    --policy.device=cuda
    --env.type=libero
    --env.task=libero_10
    --env.task_ids="${task_ids}"
    --env.control_mode=absolute
    --env.observation_height=128
    --env.observation_width=128
    --env.num_steps_wait=0
    --env.max_parallel_tasks=1
    --eval.dataset_repo_id="${EVAL_DATASET_REPO_ID}"
    --eval.dataset_root="${EVAL_DATASET_ROOT}"
    --eval.dataset_episodes="${episodes}"
    --eval.n_episodes="${EVAL_N_EPISODES}"
    --eval.batch_size="${EVAL_BATCH_SIZE}"
    --eval.use_async_envs=false
    --output_dir="${worker_output_dir}"
    --job_name="${JOB_NAME}_gpu${gpu_id}"
    --seed="${SEED}"
    --policy.use_language_conditioning=true
    --policy.language_tokenizer_name=/cephfs/shared/Yanbang/maniskill/pretrained/clip-vit-base-patch32
  )

  printf 'GPU %s tasks %s episodes %s\n' "${gpu_id}" "${task_ids}" "${episodes}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu_id}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    (
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      "${cmd[@]}"
    ) >"${worker_log}" 2>&1 &
    pids+=("$!")
  fi
done

if [[ "${DRY_RUN}" == "true" ]]; then
  exit 0
fi

status=0
for worker_id in "${!pids[@]}"; do
  if ! wait "${pids[${worker_id}]}"; then
    echo "Eval worker ${worker_id} failed. Log tail:" >&2
    tail -120 "${worker_logs[${worker_id}]}" >&2 || true
    status=1
  fi
done
if (( status != 0 )); then
  exit "${status}"
fi

uv run python - "${OUTPUT_DIR}" "${POLICY_PATH}" <<'PY'
import json
import math
import sys
import time
from pathlib import Path

output_dir = Path(sys.argv[1])
policy_path = sys.argv[2]
infos = []
for info_path in sorted(output_dir.glob("gpu*_tasks_*/eval_info.json")):
    info = json.loads(info_path.read_text(encoding="utf-8"))
    infos.append((info_path, info))
if not infos:
    raise SystemExit(f"No worker eval_info.json files found under {output_dir}")

sum_rewards = []
max_rewards = []
successes = []
per_task = []
video_paths = []
worker_summaries = []

for info_path, info in infos:
    overall = info["overall"]
    worker_summaries.append(
        {
            "worker": str(info_path.parent.name),
            "info_path": str(info_path),
            "pc_success": overall.get("pc_success"),
            "avg_sum_reward": overall.get("avg_sum_reward"),
            "n_episodes": overall.get("n_episodes"),
            "eval_s": overall.get("eval_s"),
        }
    )
    for task in info.get("per_task", []):
        metrics = task["metrics"]
        task_successes = [bool(value) for value in metrics.get("successes", [])]
        task_sum_rewards = [float(value) for value in metrics.get("sum_rewards", [])]
        task_max_rewards = [float(value) for value in metrics.get("max_rewards", [])]
        successes.extend(task_successes)
        sum_rewards.extend(task_sum_rewards)
        max_rewards.extend(task_max_rewards)
        video_paths.extend(metrics.get("video_paths", []))
        per_task.append(
            {
                "task_group": task.get("task_group"),
                "task_id": task.get("task_id"),
                "successes": sum(task_successes),
                "n_episodes": len(task_successes),
                "pc_success": (sum(task_successes) / len(task_successes) * 100)
                if task_successes
                else math.nan,
                "avg_sum_reward": sum(task_sum_rewards) / len(task_sum_rewards)
                if task_sum_rewards
                else math.nan,
                "avg_max_reward": sum(task_max_rewards) / len(task_max_rewards)
                if task_max_rewards
                else math.nan,
            }
        )

summary = {
    "mode": "parallel_eval",
    "time": time.time(),
    "policy_path": policy_path,
    "output_dir": str(output_dir),
    "overall": {
        "avg_sum_reward": sum(sum_rewards) / len(sum_rewards) if sum_rewards else math.nan,
        "avg_max_reward": sum(max_rewards) / len(max_rewards) if max_rewards else math.nan,
        "pc_success": sum(successes) / len(successes) * 100 if successes else math.nan,
        "successes": sum(successes),
        "n_episodes": len(successes),
        "worker_eval_s_max": max(
            (float(worker["eval_s"]) for worker in worker_summaries if worker["eval_s"] is not None),
            default=math.nan,
        ),
    },
    "per_task": sorted(per_task, key=lambda item: (str(item["task_group"]), int(item["task_id"]))),
    "workers": worker_summaries,
    "video_paths": video_paths,
}

summary_path = output_dir / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary["overall"], indent=2))
print(f"summary={summary_path}")
PY
