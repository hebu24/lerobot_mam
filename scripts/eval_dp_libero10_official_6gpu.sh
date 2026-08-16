#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/cephfs/shared/Yanbang/lerobot/mam_lerobot0.5.1/lerobot_mam}"
cd "${REPO_ROOT}"

export PATH="/cephfs/shared/Yanbang/envs/lerobot0.5.1/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/.uv-cache}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/.torch-cache}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${REPO_ROOT}/.cache/numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
export LIBERO_ASSETS_PATH="${LIBERO_ASSETS_PATH:-${REPO_ROOT}/.cache/libero/assets}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-outputs/train/diffusion_libero10_500_randomenv_200k_8gpu_effbs128_20260810_170932}"
POLICY_PATH="${POLICY_PATH:-${TRAIN_OUTPUT_DIR}/checkpoints/best/pretrained_model}"
GPU_IDS_TEXT="${GPU_IDS:-0,1,2,3,4,5}"
TASK_GROUPS_TEXT="${TASK_GROUPS:-0,1 2,3 4,5 6,7 8 9}"
N_EPISODES="${N_EPISODES:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPISODE_LENGTH="${EPISODE_LENGTH:-600}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-5}"
SEED="${SEED:-1000}"
JOB_NAME="${JOB_NAME:-dp_libero10_official_init20_step600_best160k_6gpu_$(date +%Y%m%d_%H%M%S)}"
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

for value_name in N_EPISODES BATCH_SIZE EPISODE_LENGTH NUM_STEPS_WAIT; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value <= 0 )); then
    echo "${value_name} must be a positive integer, got: ${value}" >&2
    exit 2
  fi
done

if (( N_EPISODES != 20 )); then
  echo "Official LIBERO protocol requires N_EPISODES=20, got: ${N_EPISODES}" >&2
  exit 2
fi
if (( EPISODE_LENGTH != 600 )); then
  echo "Official LIBERO protocol requires EPISODE_LENGTH=600, got: ${EPISODE_LENGTH}" >&2
  exit 2
fi
if (( NUM_STEPS_WAIT != 5 )); then
  echo "Official LIBERO protocol requires NUM_STEPS_WAIT=5, got: ${NUM_STEPS_WAIT}" >&2
  exit 2
fi

IFS=',' read -r -a GPU_IDS_ARRAY <<< "${GPU_IDS_TEXT}"
read -r -a TASK_GROUPS_ARRAY <<< "${TASK_GROUPS_TEXT}"
if (( ${#GPU_IDS_ARRAY[@]} != ${#TASK_GROUPS_ARRAY[@]} )); then
  echo "GPU_IDS and TASK_GROUPS must have the same number of entries." >&2
  exit 2
fi

if [[ ! -d "${POLICY_PATH}" ]]; then
  echo "Policy path not found: ${POLICY_PATH}" >&2
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
    exit 2
  fi
fi

mkdir -p "${OUTPUT_DIR}"

uv run python - "${OUTPUT_DIR}" "${POLICY_PATH}" "${N_EPISODES}" "${EPISODE_LENGTH}" \
  "${NUM_STEPS_WAIT}" "${BATCH_SIZE}" "${SEED}" "${TASK_GROUPS_ARRAY[@]}" <<'PY'
import json
import sys
from pathlib import Path

from lerobot.envs.libero import _get_suite, get_task_init_states

output_dir = Path(sys.argv[1])
policy_path = Path(sys.argv[2]).resolve()
n_episodes = int(sys.argv[3])
episode_length = int(sys.argv[4])
num_steps_wait = int(sys.argv[5])
batch_size = int(sys.argv[6])
seed = int(sys.argv[7])
task_groups = sys.argv[8:]

task_ids = [int(value) for group in task_groups for value in group.split(",") if value]
if sorted(task_ids) != list(range(10)) or len(task_ids) != len(set(task_ids)):
    raise SystemExit(f"TASK_GROUPS must cover LIBERO-10 task ids 0..9 exactly once, got {task_ids}")

suite = _get_suite("libero_10")
tasks = []
for task_id in range(10):
    task = suite.get_task(task_id)
    init_states = get_task_init_states(suite, task_id)
    if len(init_states) < n_episodes:
        raise SystemExit(
            f"Task {task_id} only has {len(init_states)} official init states; need {n_episodes}"
        )
    tasks.append(
        {
            "task_id": task_id,
            "name": task.name,
            "language": task.language,
            "init_states_file": task.init_states_file,
            "available_init_states": len(init_states),
            "evaluated_init_state_ids": list(range(n_episodes)),
        }
    )

protocol = {
    "benchmark": "LIBERO_10",
    "method": "official fixed pruned init states",
    "policy_path": str(policy_path),
    "n_episodes_per_task": n_episodes,
    "total_episodes": n_episodes * 10,
    "max_policy_steps": episode_length,
    "dummy_steps_after_init": num_steps_wait,
    "batch_size": batch_size,
    "seed": seed,
    "task_groups": task_groups,
    "tasks": tasks,
}
(output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
print(json.dumps({key: protocol[key] for key in protocol if key != "tasks"}, indent=2))
PY

echo "Official LIBERO-10 DP evaluation"
echo "  policy=$(realpath "${POLICY_PATH}")"
echo "  output=${OUTPUT_DIR}"
echo "  GPUs=${GPU_IDS_TEXT}, task_groups=${TASK_GROUPS_TEXT}"
echo "  protocol=first ${N_EPISODES} .pruned_init states/task, ${EPISODE_LENGTH} policy steps, ${NUM_STEPS_WAIT} dummy steps"

pids=()
worker_logs=()
for worker_id in "${!GPU_IDS_ARRAY[@]}"; do
  gpu_id="${GPU_IDS_ARRAY[${worker_id}]}"
  task_ids_text="${TASK_GROUPS_ARRAY[${worker_id}]}"
  task_ids="[${task_ids_text}]"
  task_label="${task_ids_text//,/_}"
  worker_output_dir="${OUTPUT_DIR}/gpu${gpu_id}_tasks_${task_label}"
  worker_log="${OUTPUT_DIR}/gpu${gpu_id}_tasks_${task_label}.log"
  worker_logs+=("${worker_log}")

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
    --env.init_states=true
    --env.episode_length="${EPISODE_LENGTH}"
    --env.num_steps_wait="${NUM_STEPS_WAIT}"
    --env.max_parallel_tasks=1
    --eval.n_episodes="${N_EPISODES}"
    --eval.batch_size="${BATCH_SIZE}"
    --eval.use_async_envs=false
    --eval.start_seed="${SEED}"
    --output_dir="${worker_output_dir}"
    --job_name="${JOB_NAME}_gpu${gpu_id}"
    --seed="${SEED}"
  )

  printf 'GPU %s tasks %s\n' "${gpu_id}" "${task_ids}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu_id}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    mkdir -p "${worker_output_dir}"
    (
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      exec "${cmd[@]}"
    ) >"${worker_log}" 2>&1 &
    pids+=("$!")
    printf '%s\n' "$!" >"${worker_output_dir}/worker.pid"
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
policy_path = str(Path(sys.argv[2]).resolve())
infos = []
for info_path in sorted(output_dir.glob("gpu*_tasks_*/eval_info.json")):
    infos.append((info_path, json.loads(info_path.read_text(encoding="utf-8"))))
if len(infos) != 6:
    raise SystemExit(f"Expected 6 worker eval_info.json files, found {len(infos)}")

sum_rewards = []
max_rewards = []
successes = []
per_task = []
workers = []
for info_path, info in infos:
    overall = info["overall"]
    workers.append(
        {
            "worker": info_path.parent.name,
            "pc_success": overall.get("pc_success"),
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
        per_task.append(
            {
                "task_id": int(task["task_id"]),
                "successes": sum(task_successes),
                "n_episodes": len(task_successes),
                "pc_success": sum(task_successes) / len(task_successes) * 100,
                "avg_sum_reward": sum(task_sum_rewards) / len(task_sum_rewards),
                "avg_max_reward": sum(task_max_rewards) / len(task_max_rewards),
            }
        )

if len(successes) != 200:
    raise SystemExit(f"Expected 200 rollout results, found {len(successes)}")
if sorted(item["task_id"] for item in per_task) != list(range(10)):
    raise SystemExit(f"Expected task ids 0..9, got {[item['task_id'] for item in per_task]}")

summary = {
    "mode": "official_libero10_fixed_pruned_init",
    "time": time.time(),
    "policy_path": policy_path,
    "output_dir": str(output_dir.resolve()),
    "overall": {
        "pc_success": sum(successes) / len(successes) * 100,
        "successes": sum(successes),
        "n_episodes": len(successes),
        "avg_sum_reward": sum(sum_rewards) / len(sum_rewards) if sum_rewards else math.nan,
        "avg_max_reward": sum(max_rewards) / len(max_rewards) if max_rewards else math.nan,
        "worker_eval_s_max": max(float(worker["eval_s"]) for worker in workers),
    },
    "per_task": sorted(per_task, key=lambda item: item["task_id"]),
    "workers": workers,
}
summary_path = output_dir / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary["overall"], indent=2))
print(f"summary={summary_path}")
PY
