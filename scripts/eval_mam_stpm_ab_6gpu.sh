#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
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

POLICY_PATH="${POLICY_PATH:-outputs/train/mam_libero10_v3_refmix_150k_6gpu_large_stpm_short0_long64_dim128_avgmse_seed1000_cudnnbench_multirankeval_20260728_142023/checkpoints/180000/pretrained_model}"
BASELINE_STPM_PREFIX="${BASELINE_STPM_PREFIX:-outputs/train/stpm_libero10_v3_large_d512_l4_task}"
CANDIDATE_STPM_PREFIX="${CANDIDATE_STPM_PREFIX:-outputs/train/stpm_libero10_v5_d768_l8_obs6_gap2_seed0_6epoch_task}"
BASELINE_CHECKPOINT_NAME="${BASELINE_CHECKPOINT_NAME:-reward_best.pt}"
CANDIDATE_CHECKPOINT_NAME="${CANDIDATE_CHECKPOINT_NAME:-reward_best_endpoint.pt}"
EVAL_DATASET_REPO_ID="${EVAL_DATASET_REPO_ID:-local/libero10_mam_v3_refmix_eval}"
EVAL_DATASET_ROOT="${EVAL_DATASET_ROOT:-data/libero10_mam/libero10_mam_v3_refmix_eval}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-5}"
SEEDS="${SEEDS:-${SEED:-1000}}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval/mam_stpm_ab_$(date +%Y%m%d_%H%M%S)}"
REUSE_BASELINE_DIR="${REUSE_BASELINE_DIR:-}"
REQUIRE_IDLE_GPU="${REQUIRE_IDLE_GPU:-true}"
DRY_RUN="${DRY_RUN:-false}"

for boolean_name in REQUIRE_IDLE_GPU DRY_RUN; do
  value="${!boolean_name}"
  if [[ "${value}" != "true" && "${value}" != "false" ]]; then
    echo "${boolean_name} must be true or false; got ${value}." >&2
    exit 2
  fi
done
if ! [[ "${EPISODES_PER_TASK}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPISODES_PER_TASK must be positive; got ${EPISODES_PER_TASK}." >&2
  exit 2
fi
if [[ ! -d "${POLICY_PATH}" ]]; then
  echo "Policy path does not exist: ${POLICY_PATH}" >&2
  exit 2
fi
POLICY_PATH="$(cd "${POLICY_PATH}" && pwd)"
if [[ ! -f "${EVAL_DATASET_ROOT}/meta/info.json" ]]; then
  echo "MAM eval dataset does not exist: ${EVAL_DATASET_ROOT}" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUTPUT_DIR}" >&2
  exit 2
fi

IFS=',' read -r -a gpu_ids <<< "${GPU_IDS}"
if (( ${#gpu_ids[@]} != 6 )); then
  echo "Exactly six GPU ids are required; got ${GPU_IDS}." >&2
  exit 2
fi
seeds_text="${SEEDS//,/ }"
read -r -a seeds <<< "${seeds_text}"
if (( ${#seeds[@]} == 0 )); then
  echo "SEEDS must contain at least one integer." >&2
  exit 2
fi
for seed in "${seeds[@]}"; do
  if ! [[ "${seed}" =~ ^[0-9]+$ ]]; then
    echo "Each seed must be a non-negative integer; got ${seed}." >&2
    exit 2
  fi
done

if [[ "${REQUIRE_IDLE_GPU}" == "true" && "${DRY_RUN}" != "true" ]]; then
  active_gpu_pids="$(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d'
  )"
  if [[ -n "${active_gpu_pids}" ]]; then
    echo "A requested evaluation GPU is already in use: ${active_gpu_pids//$'\n'/, }" >&2
    exit 2
  fi
fi

mkdir -p "${OUTPUT_DIR}/preflight"
uv run python - \
  "${BASELINE_STPM_PREFIX}" \
  "${CANDIDATE_STPM_PREFIX}" \
  "${BASELINE_CHECKPOINT_NAME}" \
  "${CANDIDATE_CHECKPOINT_NAME}" \
  "${EVAL_DATASET_REPO_ID}" \
  "${EVAL_DATASET_ROOT}" \
  "${EPISODES_PER_TASK}" \
  "${OUTPUT_DIR}/preflight" <<'PY'
import json
import sys
from pathlib import Path

from lerobot.datasets import LeRobotDatasetMetadata

baseline_prefix = sys.argv[1]
candidate_prefix = sys.argv[2]
baseline_checkpoint = sys.argv[3]
candidate_checkpoint = sys.argv[4]
eval_repo_id = sys.argv[5]
eval_root = Path(sys.argv[6])
episodes_per_task = int(sys.argv[7])
output = Path(sys.argv[8])

roots = {"baseline": {}, "candidate": {}}
checkpoints = {"baseline": {}, "candidate": {}}
all_stpm_sources = set()
for task_id in range(10):
    expected_tasks = [str(task_id)]
    configs = {}
    for variant, prefix, checkpoint_name in (
        ("baseline", baseline_prefix, baseline_checkpoint),
        ("candidate", candidate_prefix, candidate_checkpoint),
    ):
        root = Path(f"{prefix}{task_id}")
        config_path = root / "config.yaml"
        state_norm_path = root / "state_norm.json"
        checkpoint_path = root / "checkpoints" / checkpoint_name
        for path in (config_path, state_norm_path, checkpoint_path):
            if not path.is_file():
                raise SystemExit(f"Missing {variant} task {task_id} artifact: {path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        identity = config.get("split_identity", {})
        if identity.get("tasks") != expected_tasks:
            raise SystemExit(
                f"{variant} task identity mismatch for task {task_id}: {identity.get('tasks')!r}"
            )
        train = {
            (str(row["task"]), str(row["source_episode_id"]))
            for row in identity.get("train_groups", [])
        }
        val = {
            (str(row["task"]), str(row["source_episode_id"]))
            for row in identity.get("val_groups", [])
        }
        if not train or not val or train & val:
            raise SystemExit(f"Invalid {variant} train/val split for task {task_id}.")
        all_stpm_sources.update(train | val)
        configs[variant] = config
        key = f"libero_10/{task_id}"
        roots[variant][key] = str(root)
        checkpoints[variant][key] = str(checkpoint_path)
    if configs["baseline"]["split_identity"]["train_groups"] != configs["candidate"]["split_identity"]["train_groups"]:
        raise SystemExit(f"Task {task_id} baseline/candidate train source split differs.")
    if configs["baseline"]["split_identity"]["val_groups"] != configs["candidate"]["split_identity"]["val_groups"]:
        raise SystemExit(f"Task {task_id} baseline/candidate validation source split differs.")

metadata = LeRobotDatasetMetadata(eval_repo_id, root=eval_root)
by_task = {}
eval_sources = set()
for row in metadata.episodes:
    task_id = int(row["libero/task_id"])
    by_task.setdefault(task_id, []).append(int(row["episode_index"]))
    eval_sources.add((str(task_id), str(row["libero/source_episode_id"])))
overlap = all_stpm_sources & eval_sources
if overlap:
    raise SystemExit(f"STPM source episodes overlap MAM eval sources: {sorted(overlap)[:20]}")
for task_id in range(10):
    episode_ids = sorted(by_task.get(task_id, []))
    if len(episode_ids) < episodes_per_task:
        raise SystemExit(
            f"Task {task_id} has {len(episode_ids)} MAM eval episodes; need {episodes_per_task}."
        )
    (output / f"episodes_task{task_id}.json").write_text(
        json.dumps(episode_ids[:episodes_per_task], separators=(",", ":")),
        encoding="utf-8",
    )
for variant in ("baseline", "candidate"):
    (output / f"{variant}_roots.json").write_text(
        json.dumps(roots[variant], separators=(",", ":")),
        encoding="utf-8",
    )
    (output / f"{variant}_checkpoints.json").write_text(
        json.dumps(checkpoints[variant], separators=(",", ":")),
        encoding="utf-8",
    )
(output / "summary.json").write_text(
    json.dumps(
        {
            "episodes_per_task": episodes_per_task,
            "stpm_source_groups": len(all_stpm_sources),
            "mam_eval_source_groups": len(eval_sources),
            "source_overlap": 0,
            "episodes_by_task": {
                str(task): sorted(by_task[task])[:episodes_per_task] for task in range(10)
            },
        },
        indent=2,
    ),
    encoding="utf-8",
)
PY

variants=(baseline candidate)
if [[ -n "${REUSE_BASELINE_DIR}" ]]; then
  if [[ ! -d "${REUSE_BASELINE_DIR}" ]]; then
    echo "Baseline reuse directory does not exist: ${REUSE_BASELINE_DIR}" >&2
    exit 2
  fi
  REUSE_BASELINE_DIR="$(cd "${REUSE_BASELINE_DIR}" && pwd)"
  for preflight_name in summary.json baseline_roots.json baseline_checkpoints.json; do
    if ! cmp -s \
      "${REUSE_BASELINE_DIR}/preflight/${preflight_name}" \
      "${OUTPUT_DIR}/preflight/${preflight_name}"; then
      echo "Reused baseline preflight differs: ${preflight_name}" >&2
      exit 2
    fi
  done
  policy_marker="'pretrained_path': PosixPath('${POLICY_PATH}')"
  if ! grep -Fq \
    "${policy_marker}" \
    "${REUSE_BASELINE_DIR}/seed_${seeds[0]}/baseline/task_0.log"; then
    echo "Reused baseline was not evaluated with policy ${POLICY_PATH}." >&2
    exit 2
  fi
  for seed in "${seeds[@]}"; do
    for task_id in {0..9}; do
      reused_info="${REUSE_BASELINE_DIR}/seed_${seed}/baseline/task_${task_id}/eval_info.json"
      if [[ ! -f "${reused_info}" ]]; then
        echo "Missing reused baseline result: ${reused_info}" >&2
        exit 2
      fi
      worker_dir="${OUTPUT_DIR}/seed_${seed}/baseline/task_${task_id}"
      mkdir -p "${worker_dir}"
      cp "${reused_info}" "${worker_dir}/eval_info.json"
    done
  done
  printf '%s\n' "${REUSE_BASELINE_DIR}" >"${OUTPUT_DIR}/preflight/reused_baseline_dir.txt"
  variants=(candidate)
  echo "Reusing baseline results from ${REUSE_BASELINE_DIR}; only candidate waves will run."
fi

run_wave() {
  local variant="$1"
  local seed="$2"
  shift 2
  local tasks=("$@")
  local roots checkpoints
  roots="$(<"${OUTPUT_DIR}/preflight/${variant}_roots.json")"
  checkpoints="$(<"${OUTPUT_DIR}/preflight/${variant}_checkpoints.json")"
  local pids=()
  local logs=()
  local status=0

  for index in "${!tasks[@]}"; do
    local task_id="${tasks[$index]}"
    local gpu_id="${gpu_ids[$index]}"
    local episodes
    episodes="$(<"${OUTPUT_DIR}/preflight/episodes_task${task_id}.json")"
    local worker_dir="${OUTPUT_DIR}/seed_${seed}/${variant}/task_${task_id}"
    local worker_log="${OUTPUT_DIR}/seed_${seed}/${variant}/task_${task_id}.log"
    mkdir -p "${worker_dir}"
    local cmd=(
      uv run python -m lerobot.scripts.lerobot_eval
      --policy.path="${POLICY_PATH}"
      --policy.device=cuda
      --policy.stpm_paths="${roots}"
      --policy.stpm_checkpoint_paths="${checkpoints}"
      --policy.mam_eval_dataset_repo_id="${EVAL_DATASET_REPO_ID}"
      --policy.mam_eval_dataset_root="${EVAL_DATASET_ROOT}"
      --policy.mam_eval_episodes="${episodes}"
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
      --job_name="mam_stpm_ab_${variant}_seed${seed}_task${task_id}"
      --seed="${seed}"
      --policy.use_language_conditioning=true
      --policy.language_tokenizer_name=/cephfs/shared/Yanbang/maniskill/pretrained/clip-vit-base-patch32
    )
    echo "[${variant}] seed=${seed} task=${task_id} gpu=${gpu_id} episodes=${episodes}"
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
      logs+=("${worker_log}")
    fi
  done

  if [[ "${DRY_RUN}" == "true" ]]; then
    return
  fi
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "MAM A/B worker failed; log=${logs[$index]}" >&2
      tail -120 "${logs[$index]}" >&2 || true
      status=1
    fi
  done
  if (( status != 0 )); then
    return "${status}"
  fi
  if grep -Eni "using rollout step ratio as progress fallback|No STPM configured" "${logs[@]}"; then
    echo "Detected forbidden progress fallback in ${variant}, seed ${seed}." >&2
    return 1
  fi
}

for seed in "${seeds[@]}"; do
  for variant in "${variants[@]}"; do
    run_wave "${variant}" "${seed}" 0 1 2 3 4 5
    run_wave "${variant}" "${seed}" 6 7 8 9
  done
done

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "DRY_RUN completed; commands were not executed."
  exit 0
fi

uv run python - "${OUTPUT_DIR}" "${seeds[@]}" <<'PY'
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

output = Path(sys.argv[1])
seeds = [int(value) for value in sys.argv[2:]]
variants = ("baseline", "candidate")
summary = {"seeds": seeds, "variants": {}, "paired_comparison": {}}
episode_rows = {}

for variant in variants:
    variant_runs = {}
    seed_rates = []
    for seed in seeds:
        per_episode = []
        eval_seconds = 0.0
        per_task = {}
        for task_id in range(10):
            info_path = output / f"seed_{seed}" / variant / f"task_{task_id}" / "eval_info.json"
            if not info_path.is_file():
                raise SystemExit(f"Missing worker result: {info_path}")
            info = json.loads(info_path.read_text(encoding="utf-8"))
            rows = info.get("per_episode", [])
            if not rows:
                raise SystemExit(f"No per_episode records in {info_path}")
            for row in rows:
                row["task_id"] = int(row.get("task_id", task_id))
                row["seed"] = seed
                per_episode.append(row)
                key = (seed, task_id, int(row["source_episode_id"]))
                if key in episode_rows.setdefault(variant, {}):
                    raise SystemExit(f"Duplicate paired episode key: {variant} {key}")
                episode_rows[variant][key] = row
            overall = info["overall"]
            eval_seconds += float(overall["eval_s"])
            per_task[str(task_id)] = {
                "pc_success": 100.0 * sum(bool(row["success"]) for row in rows) / len(rows),
                "n_episodes": len(rows),
            }
        successes = [bool(row["success"]) for row in per_episode]
        by_mask_type = defaultdict(list)
        by_mask_slot = defaultdict(list)
        for row in per_episode:
            by_mask_type[str(row["mask_type"])].append(bool(row["success"]))
            by_mask_slot[str(row["mask_type_slot"])].append(bool(row["success"]))
        rate = 100.0 * sum(successes) / len(successes)
        seed_rates.append(rate)
        variant_runs[str(seed)] = {
            "overall": {
                "pc_success": rate,
                "n_episodes": len(successes),
                "sum_worker_eval_s": eval_seconds,
                "mean_episode_s": eval_seconds / len(successes),
            },
            "per_task": per_task,
            "per_mask_type_success": {
                key: 100.0 * sum(values) / len(values) for key, values in sorted(by_mask_type.items())
            },
            "per_mask_slot_success": {
                key: 100.0 * sum(values) / len(values) for key, values in sorted(by_mask_slot.items())
            },
            "per_episode": per_episode,
        }
    summary["variants"][variant] = {
        "runs": variant_runs,
        "overall_across_seeds": {
            "mean_pc_success": statistics.mean(seed_rates),
            "std_pc_success": statistics.pstdev(seed_rates) if len(seed_rates) > 1 else 0.0,
        },
    }

baseline_keys = set(episode_rows["baseline"])
candidate_keys = set(episode_rows["candidate"])
if baseline_keys != candidate_keys:
    raise SystemExit("Baseline/candidate paired episode keys differ.")
paired = []
for key in sorted(baseline_keys):
    baseline_success = bool(episode_rows["baseline"][key]["success"])
    candidate_success = bool(episode_rows["candidate"][key]["success"])
    paired.append(
        {
            "seed": key[0],
            "task_id": key[1],
            "source_episode_id": key[2],
            "baseline_success": baseline_success,
            "candidate_success": candidate_success,
            "delta": int(candidate_success) - int(baseline_success),
        }
    )
deltas = [row["delta"] for row in paired]
summary["paired_comparison"] = {
    "n_pairs": len(paired),
    "candidate_wins": sum(delta > 0 for delta in deltas),
    "baseline_wins": sum(delta < 0 for delta in deltas),
    "ties": sum(delta == 0 for delta in deltas),
    "pc_success_delta": 100.0 * sum(deltas) / len(deltas),
    "per_episode": paired,
}
(output / "comparison.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(summary["paired_comparison"], indent=2))
print(f"Wrote {output / 'comparison.json'}")
PY
