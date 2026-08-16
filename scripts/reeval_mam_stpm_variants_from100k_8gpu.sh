#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PATH="/cephfs/shared/Yanbang/envs/lerobot0.5.1/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

SOURCE_RUN="${SOURCE_RUN:-outputs/train/mam_libero10_500train_100first50eval_refmix_train4_eval5_long64_avgmse_stpmv3_scratch_8gpu_200k_keep100k_20260807_222122}"
STEP_LIST="${STEP_LIST:-100000,110000,120000,130000,140000,150000,160000,170000,180000,190000,200000}"
VARIANT_LIST="${VARIANT_LIST:-v4_obs6_gap2_d768_l8_seed42_best,v5_obs6_gap2_d768_l8_seed0_best,v5_obs6_gap2_d768_l8_seed0_endpoint}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval/mam_stpm_reeval_from100k_8gpu_${RUN_ID}}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-5}"
SEED="${SEED:-1000}"
REQUIRE_IDLE_GPU="${REQUIRE_IDLE_GPU:-true}"
DRY_RUN="${DRY_RUN:-false}"

declare -A STPM_PREFIXES=(
  [v3_obs1_gap1_d512_l4_best]="outputs/train/stpm_libero10_v3_large_d512_l4_task"
  [v4_obs6_gap2_d768_l8_seed42_best]="outputs/train/stpm_libero10_v4_maniskill_d768_l8_obs6_gap2_seed42_20260729_task"
  [v5_obs6_gap2_d768_l8_seed0_best]="outputs/train/stpm_libero10_v5_d768_l8_obs6_gap2_seed0_6epoch_task"
  [v5_obs6_gap2_d768_l8_seed0_endpoint]="outputs/train/stpm_libero10_v5_d768_l8_obs6_gap2_seed0_6epoch_task"
  [v6_obs6_gap2_d544_l5_seed0_best]="outputs/train/stpm_libero10_v6_d544_l5_obs6_gap2_seed0_6epoch_task"
  [v6_obs6_gap2_d544_l5_seed0_endpoint]="outputs/train/stpm_libero10_v6_d544_l5_obs6_gap2_seed0_6epoch_task"
  [v3_obs3_gap1_d512_l4_seed0_6epoch_best]="outputs/train/stpm_libero10_v3_d512_l4_obs3_gap1_seed0_6epoch_task"
  [v3_obs3_gap1_d512_l4_seed0_6epoch_endpoint]="outputs/train/stpm_libero10_v3_d512_l4_obs3_gap1_seed0_6epoch_task"
  [v3_obs6_gap1_d512_l4_seed0_6epoch_best]="outputs/train/stpm_libero10_v3_d512_l4_obs6_gap1_seed0_6epoch_task"
  [v3_obs6_gap1_d512_l4_seed0_6epoch_endpoint]="outputs/train/stpm_libero10_v3_d512_l4_obs6_gap1_seed0_6epoch_task"
  [v3_obs6_gap2_d512_l4_seed0_6epoch_best]="outputs/train/stpm_libero10_v3_d512_l4_obs6_gap2_seed0_6epoch_task"
  [v3_obs6_gap2_d512_l4_seed0_6epoch_endpoint]="outputs/train/stpm_libero10_v3_d512_l4_obs6_gap2_seed0_6epoch_task"
)

declare -A STPM_CKPTS=(
  [v3_obs1_gap1_d512_l4_best]="reward_best.pt"
  [v4_obs6_gap2_d768_l8_seed42_best]="reward_best.pt"
  [v5_obs6_gap2_d768_l8_seed0_best]="reward_best.pt"
  [v5_obs6_gap2_d768_l8_seed0_endpoint]="reward_best_endpoint.pt"
  [v6_obs6_gap2_d544_l5_seed0_best]="reward_best.pt"
  [v6_obs6_gap2_d544_l5_seed0_endpoint]="reward_best_endpoint.pt"
  [v3_obs3_gap1_d512_l4_seed0_6epoch_best]="reward_best.pt"
  [v3_obs3_gap1_d512_l4_seed0_6epoch_endpoint]="reward_best_endpoint.pt"
  [v3_obs6_gap1_d512_l4_seed0_6epoch_best]="reward_best.pt"
  [v3_obs6_gap1_d512_l4_seed0_6epoch_endpoint]="reward_best_endpoint.pt"
  [v3_obs6_gap2_d512_l4_seed0_6epoch_best]="reward_best.pt"
  [v3_obs6_gap2_d512_l4_seed0_6epoch_endpoint]="reward_best_endpoint.pt"
)

declare -A STPM_DESCRIPTIONS=(
  [v3_obs1_gap1_d512_l4_best]="v3 baseline, obs=1, gap=1, d512/l4, reward_best"
  [v4_obs6_gap2_d768_l8_seed42_best]="v4, obs=6, gap=2, d768/l8, seed42, reward_best"
  [v5_obs6_gap2_d768_l8_seed0_best]="v5, obs=6, gap=2, d768/l8, seed0, 6epoch, reward_best"
  [v5_obs6_gap2_d768_l8_seed0_endpoint]="v5, obs=6, gap=2, d768/l8, seed0, 6epoch, reward_best_endpoint"
  [v6_obs6_gap2_d544_l5_seed0_best]="v6, obs=6, gap=2, d544/l5, seed0, 6epoch, reward_best"
  [v6_obs6_gap2_d544_l5_seed0_endpoint]="v6, obs=6, gap=2, d544/l5, seed0, 6epoch, reward_best_endpoint"
  [v3_obs3_gap1_d512_l4_seed0_6epoch_best]="v3, obs=3, gap=1, d512/l4, seed0, 6epoch, reward_best"
  [v3_obs3_gap1_d512_l4_seed0_6epoch_endpoint]="v3, obs=3, gap=1, d512/l4, seed0, 6epoch, reward_best_endpoint"
  [v3_obs6_gap1_d512_l4_seed0_6epoch_best]="v3, obs=6, gap=1, d512/l4, seed0, 6epoch, reward_best"
  [v3_obs6_gap1_d512_l4_seed0_6epoch_endpoint]="v3, obs=6, gap=1, d512/l4, seed0, 6epoch, reward_best_endpoint"
  [v3_obs6_gap2_d512_l4_seed0_6epoch_best]="v3, obs=6, gap=2, d512/l4, seed0, 6epoch, reward_best"
  [v3_obs6_gap2_d512_l4_seed0_6epoch_endpoint]="v3, obs=6, gap=2, d512/l4, seed0, 6epoch, reward_best_endpoint"
)

for name in REQUIRE_IDLE_GPU DRY_RUN; do
  value="${!name}"
  if [[ "${value}" != "true" && "${value}" != "false" ]]; then
    echo "${name} must be true or false; got ${value}." >&2
    exit 2
  fi
done

if [[ ! -d "${SOURCE_RUN}" ]]; then
  echo "SOURCE_RUN does not exist: ${SOURCE_RUN}" >&2
  exit 2
fi

IFS=',' read -r -a steps <<<"${STEP_LIST}"
IFS=',' read -r -a variants <<<"${VARIANT_LIST}"
if (( ${#steps[@]} == 0 )); then
  echo "STEP_LIST must contain at least one step." >&2
  exit 2
fi
if (( ${#variants[@]} == 0 )); then
  echo "VARIANT_LIST must contain at least one variant." >&2
  exit 2
fi

for raw_step in "${steps[@]}"; do
  step="${raw_step//[[:space:]]/}"
  if ! [[ "${step}" =~ ^[0-9]+$ ]]; then
    echo "Invalid step in STEP_LIST: ${raw_step}" >&2
    exit 2
  fi
  step_id="$(printf "%06d" "${step}")"
  policy_path="${SOURCE_RUN}/checkpoints/${step_id}/pretrained_model"
  if [[ ! -d "${policy_path}" ]]; then
    echo "Missing policy checkpoint: ${policy_path}" >&2
    exit 2
  fi
done

for raw_variant in "${variants[@]}"; do
  variant="${raw_variant//[[:space:]]/}"
  if [[ -z "${STPM_PREFIXES[${variant}]:-}" ]]; then
    echo "Unknown variant: ${variant}" >&2
    echo "Available variants:" >&2
    printf '  %s\n' "${!STPM_PREFIXES[@]}" >&2
    exit 2
  fi
  prefix="${STPM_PREFIXES[${variant}]}"
  ckpt="${STPM_CKPTS[${variant}]}"
  for task_id in {0..9}; do
    root="${prefix}${task_id}"
    for artifact in "config.yaml" "state_norm.json" "checkpoints/${ckpt}"; do
      if [[ ! -f "${root}/${artifact}" ]]; then
        echo "Missing STPM artifact for ${variant}: ${root}/${artifact}" >&2
        exit 2
      fi
    done
  done
done

mkdir -p "${OUTPUT_ROOT}"
{
  echo "source_run=${SOURCE_RUN}"
  echo "step_list=${STEP_LIST}"
  echo "variant_list=${VARIANT_LIST}"
  echo "gpu_ids=${GPU_IDS}"
  echo "episodes_per_task=${EPISODES_PER_TASK}"
  echo "seed=${SEED}"
  echo "started_at=$(date -Is)"
  for variant in "${variants[@]}"; do
    variant="${variant//[[:space:]]/}"
    echo "variant.${variant}.prefix=${STPM_PREFIXES[${variant}]}"
    echo "variant.${variant}.checkpoint=${STPM_CKPTS[${variant}]}"
    echo "variant.${variant}.description=${STPM_DESCRIPTIONS[${variant}]}"
  done
} >"${OUTPUT_ROOT}/run_manifest.txt"

echo "[Reeval] source=${SOURCE_RUN}"
echo "[Reeval] output=${OUTPUT_ROOT}"
echo "[Reeval] steps=${STEP_LIST}"
echo "[Reeval] variants=${VARIANT_LIST}"

for raw_step in "${steps[@]}"; do
  step="${raw_step//[[:space:]]/}"
  step_id="$(printf "%06d" "${step}")"
  policy_path="${SOURCE_RUN}/checkpoints/${step_id}/pretrained_model"
  for raw_variant in "${variants[@]}"; do
    variant="${raw_variant//[[:space:]]/}"
    output_dir="${OUTPUT_ROOT}/${variant}/step_${step_id}"
    if [[ -f "${output_dir}/summary.json" ]]; then
      echo "[Skip] variant=${variant} step=${step_id}: ${output_dir}/summary.json exists"
      continue
    fi
    echo "[Run] variant=${variant} step=${step_id}"
    POLICY_PATH="${policy_path}" \
    STPM_PREFIX="${STPM_PREFIXES[${variant}]}" \
    STPM_CHECKPOINT_NAME="${STPM_CKPTS[${variant}]}" \
    OUTPUT_DIR="${output_dir}" \
    GPU_IDS="${GPU_IDS}" \
    EPISODES_PER_TASK="${EPISODES_PER_TASK}" \
    SEED="${SEED}" \
    REQUIRE_IDLE_GPU="${REQUIRE_IDLE_GPU}" \
    RESUME=true \
    DRY_RUN="${DRY_RUN}" \
    bash scripts/eval_mam_single_stpm_multigpu.sh
  done
done

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[Reeval] dry run complete; no aggregation written."
  exit 0
fi

uv run python - "${OUTPUT_ROOT}" "${SOURCE_RUN}" "${STEP_LIST}" "${VARIANT_LIST}" <<'PY'
import csv
import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
source_run = sys.argv[2]
steps = [f"{int(value):06d}" for value in sys.argv[3].split(",") if value.strip()]
variants = [value.strip() for value in sys.argv[4].split(",") if value.strip()]

rows = []
missing = []
for variant in variants:
    for step in steps:
        summary_path = output_root / variant / f"step_{step}" / "summary.json"
        if not summary_path.is_file():
            missing.append(str(summary_path))
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {
            "variant": variant,
            "step": step,
            "pc_success": summary["overall"]["pc_success"],
            "n_episodes": summary["overall"]["n_episodes"],
            "mean_episode_s": summary["overall"]["mean_episode_s"],
        }
        for task_id in range(10):
            row[f"task_{task_id}"] = summary["per_task_success"].get(str(task_id))
        for slot in range(5):
            row[f"mask_slot_{slot}"] = summary["per_mask_slot_success"].get(str(slot))
        for mask_type in ("points", "3D_points", "pose_motion_planning", "mix0"):
            row[f"mask_type_{mask_type}"] = summary["per_mask_type_success"].get(mask_type)
        rows.append(row)

rows.sort(key=lambda item: (item["variant"], int(item["step"])))
fieldnames = [
    "variant",
    "step",
    "pc_success",
    "n_episodes",
    "mean_episode_s",
    *[f"task_{task_id}" for task_id in range(10)],
    *[f"mask_slot_{slot}" for slot in range(5)],
    "mask_type_points",
    "mask_type_3D_points",
    "mask_type_pose_motion_planning",
    "mask_type_mix0",
]
csv_path = output_root / "summary_table.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

best_overall = max(rows, key=lambda item: item["pc_success"]) if rows else None
best_by_variant = {}
for variant in variants:
    variant_rows = [row for row in rows if row["variant"] == variant]
    if variant_rows:
        best_by_variant[variant] = max(variant_rows, key=lambda item: item["pc_success"])

comparison = {
    "source_run": source_run,
    "output_root": str(output_root),
    "n_rows": len(rows),
    "missing": missing,
    "best_overall": best_overall,
    "best_by_variant": best_by_variant,
    "rows": rows,
}
(output_root / "comparison.json").write_text(
    json.dumps(comparison, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps({"best_overall": best_overall, "best_by_variant": best_by_variant, "missing": missing}, ensure_ascii=False, indent=2))
print(f"Wrote {csv_path}")
print(f"Wrote {output_root / 'comparison.json'}")
PY

echo "[Reeval] complete: ${OUTPUT_ROOT}"
