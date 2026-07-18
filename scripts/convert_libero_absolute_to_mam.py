#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.compute_stats import compute_libero_relative_action_stats
from lerobot.datasets.io_utils import write_stats
from lerobot.datasets.libero_pipeline import (
    LIBERO_ABSOLUTE_ACTION,
    LIBERO_CHUNK_RELATIVE_ACTION,
    LIBERO_CLOSED_LOOP_ABSOLUTE_MATERIALIZATION,
    LIBERO_PIPELINE_VERSION,
    LIBERO_STATE_14D,
    read_libero_pipeline_manifest,
    require_libero_v3_relative_ready_dataset,
    write_libero_pipeline_manifest,
)
from lerobot.utils.constants import ACTION, DEFAULT_FEATURES

MAM_MAS_ACTION_ABSOLUTE = "mam.mas_action_absolute"
MAM_MAS_ACTION_MASK = "mam.mas_action_mask"
MAM_PROGRESS = "mam.progress"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create materialized MAM LeRobot datasets from LIBERO absolute actions."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--input-repo-id", type=str, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-repo-id", type=str, required=True)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--eval-per-task", type=int, default=None)
    parser.add_argument(
        "--eval-episode-ids",
        type=str,
        default=None,
        help="Explicit comma-separated source episode ids for a replay-certified eval split.",
    )
    parser.add_argument(
        "--task-stratified-eval-ratio",
        action="store_true",
        help="When --eval-per-task is unset, split by task while matching --eval-ratio globally.",
    )
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--only-split", choices=("both", "train", "eval"), default="both")
    parser.add_argument("--mask-type", type=str, default="random_mask")
    parser.add_argument(
        "--mask-types",
        type=str,
        default=None,
        help="Comma-separated mask types. Each source episode is materialized once per entry.",
    )
    parser.add_argument("--retain-ratio", type=float, default=0.2)
    parser.add_argument("--mask-value", type=float, default=0.0)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument(
        "--skip-relative-action-stats",
        action="store_true",
        help="Skip the expensive HF-datasets reload used to compute relative action stats.",
    )
    parser.add_argument(
        "--allow-source-exclusions",
        action="store_true",
        help=(
            "Allow a completed v3 replay audit source with explicit unrepairable_episode_ids. "
            "Only valid_absolute_episode_ids are eligible for the train/eval split."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve_mask_types(args: argparse.Namespace) -> list[str]:
    raw_mask_types = getattr(args, "mask_types", None)
    if raw_mask_types is None:
        mask_types = [str(args.mask_type).strip()]
    else:
        mask_types = [item.strip() for item in str(raw_mask_types).split(",")]
    if not mask_types or any(not item for item in mask_types):
        raise ValueError("--mask-types must contain one or more non-empty comma-separated values.")
    return mask_types


def _repo_id_from_root(root: Path) -> str:
    return f"local/{root.name}"


def _selected_episode_ids(total: int, eval_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    ids = np.arange(total, dtype=np.int64)
    if total <= 1:
        return ids.tolist(), []
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    eval_count = max(1, int(round(total * float(eval_ratio))))
    eval_ids = sorted(ids[:eval_count].astype(int).tolist())
    train_ids = sorted(ids[eval_count:].astype(int).tolist())
    if len(train_ids) == 0:
        train_ids, eval_ids = eval_ids, []
    return train_ids, eval_ids


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (KeyError, TypeError):
        return default


def _as_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    # MuJoCo init states are contact-sensitive float64 values. Downcasting here
    # makes an "overfit eval on the training trajectory" reset non-identical.
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return None
    return array.tolist()


def _episode_rows(dataset: LeRobotDataset) -> dict[int, Any]:
    return {int(row["episode_index"]): row for row in dataset.meta.episodes}


def _column_names(dataset: LeRobotDataset) -> set[str]:
    return set(getattr(dataset.meta.episodes, "column_names", []) or [])


def _round_robin_task_episode_ids(task_episode_ids: dict[int, list[int]]) -> list[int]:
    ordered: list[int] = []
    sorted_groups = {task_id: sorted(ids) for task_id, ids in sorted(task_episode_ids.items())}
    max_len = max((len(ids) for ids in sorted_groups.values()), default=0)
    for slot in range(max_len):
        for task_id in sorted(sorted_groups):
            ids = sorted_groups[task_id]
            if slot < len(ids):
                ordered.append(ids[slot])
    return ordered


def _selected_explicit_eval_ids(
    source: LeRobotDataset,
    text: str,
    *,
    eval_per_task: int | None,
    eligible_episode_ids: set[int] | None,
) -> tuple[list[int], list[int]]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("--eval-episode-ids must contain unique episode ids.")
    eligible = (
        set(eligible_episode_ids)
        if eligible_episode_ids is not None
        else {int(row["episode_index"]) for row in source.meta.episodes}
    )
    eval_set = set(values)
    if not eval_set <= eligible:
        raise ValueError(
            f"--eval-episode-ids contains ineligible ids: {sorted(eval_set - eligible)}"
        )

    columns = _column_names(source)
    task_key = next((key for key in ("libero/task_id", "task_id") if key in columns), None)
    if task_key is None:
        raise ValueError("Explicit eval split requires task id episode metadata.")
    groups: dict[int, list[int]] = {}
    for row in source.meta.episodes:
        episode_id = int(row["episode_index"])
        if episode_id in eval_set:
            groups.setdefault(int(row[task_key]), []).append(episode_id)
    if eval_per_task is not None:
        counts = {task_id: len(ids) for task_id, ids in groups.items()}
        expected = {task_id: int(eval_per_task) for task_id in range(10)}
        if counts != expected:
            raise ValueError(
                f"Explicit eval split must contain {eval_per_task} episode(s) per task: {counts}"
            )
    return sorted(eligible - eval_set), _round_robin_task_episode_ids(groups)


def _selected_episode_ids_by_task(
    source: LeRobotDataset,
    eval_per_task: int,
    seed: int,
    eligible_episode_ids: set[int] | None = None,
) -> tuple[list[int], list[int]]:
    if eval_per_task <= 0:
        raise ValueError(f"eval_per_task must be positive, got {eval_per_task}.")

    columns = _column_names(source)
    task_key = (
        "libero/task_id" if "libero/task_id" in columns else "task_id" if "task_id" in columns else None
    )
    if task_key is None:
        raise ValueError(
            "Task-balanced split requires episode metadata column 'libero/task_id' or 'task_id'."
        )

    groups: dict[int, list[int]] = {}
    for row in source.meta.episodes:
        episode_index = int(row["episode_index"])
        if eligible_episode_ids is not None and episode_index not in eligible_episode_ids:
            continue
        groups.setdefault(int(row[task_key]), []).append(episode_index)
    if not groups:
        raise ValueError("No episodes found for task-balanced split.")

    rng = np.random.default_rng(seed)
    train_ids: list[int] = []
    eval_ids_by_task: dict[int, list[int]] = {}
    for task_id in sorted(groups):
        ids = np.asarray(sorted(groups[task_id]), dtype=np.int64)
        if len(ids) <= eval_per_task:
            raise ValueError(
                f"Task {task_id} has only {len(ids)} episode(s), "
                f"cannot reserve {eval_per_task} eval episode(s)."
            )
        rng.shuffle(ids)
        eval_ids_by_task[task_id] = ids[:eval_per_task].astype(int).tolist()
        train_ids.extend(sorted(ids[eval_per_task:].astype(int).tolist()))
    return sorted(train_ids), _round_robin_task_episode_ids(eval_ids_by_task)


def _selected_episode_ids_by_task_ratio(
    source: LeRobotDataset,
    eval_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    columns = _column_names(source)
    task_key = (
        "libero/task_id" if "libero/task_id" in columns else "task_id" if "task_id" in columns else None
    )
    if task_key is None:
        raise ValueError(
            "Task-stratified split requires episode metadata column 'libero/task_id' or 'task_id'."
        )

    groups: dict[int, list[int]] = {}
    for row in source.meta.episodes:
        groups.setdefault(int(row[task_key]), []).append(int(row["episode_index"]))
    if not groups:
        raise ValueError("No episodes found for task-stratified split.")

    total = sum(len(ids) for ids in groups.values())
    eval_total = max(1, int(round(total * float(eval_ratio)))) if total > 1 else 0
    quotas = {task_id: len(ids) * float(eval_ratio) for task_id, ids in groups.items()}
    eval_counts = {task_id: int(np.floor(quota)) for task_id, quota in quotas.items()}
    remaining = eval_total - sum(eval_counts.values())
    if remaining > 0:
        ranked = sorted(
            groups,
            key=lambda task_id: (quotas[task_id] - eval_counts[task_id], len(groups[task_id]), -task_id),
            reverse=True,
        )
        for task_id in ranked[:remaining]:
            eval_counts[task_id] += 1
    elif remaining < 0:
        ranked = sorted(
            groups,
            key=lambda task_id: (quotas[task_id] - eval_counts[task_id], len(groups[task_id]), -task_id),
        )
        for task_id in ranked[:-remaining]:
            eval_counts[task_id] -= 1

    rng = np.random.default_rng(seed)
    train_ids: list[int] = []
    eval_ids_by_task: dict[int, list[int]] = {}
    for task_id in sorted(groups):
        ids = np.asarray(sorted(groups[task_id]), dtype=np.int64)
        rng.shuffle(ids)
        count = eval_counts[task_id]
        if count <= 0 or count >= len(ids):
            raise ValueError(f"Task {task_id} has {len(ids)} episode(s), invalid eval count {count}.")
        eval_ids_by_task[task_id] = ids[:count].astype(int).tolist()
        train_ids.extend(sorted(ids[count:].astype(int).tolist()))
    return sorted(train_ids), _round_robin_task_episode_ids(eval_ids_by_task)


def _apply_mask(
    action: np.ndarray,
    mask_type: str,
    retain_ratio: float,
    mask_value: float,
    rng,
) -> tuple[np.ndarray, np.ndarray]:
    action = np.asarray(action, dtype=np.float32)
    mask = np.zeros_like(action, dtype=np.float32)
    if mask_type == "none":
        return np.full_like(action, mask_value), mask
    if mask_type == "full":
        mask[:] = 1.0
    elif mask_type == "random_mask":
        total = action.size
        keep = int(total * float(retain_ratio))
        if keep > 0:
            idx = np.arange(total)
            rng.shuffle(idx)
            mask.reshape(-1)[idx[:keep]] = 1.0
    elif mask_type == "3D_points":
        keep = int(action.shape[0] * float(retain_ratio))
        idx = np.arange(action.shape[0])
        rng.shuffle(idx)
        mask[idx[:keep], :3] = 1.0
    elif mask_type == "points":
        keep = int(action.shape[0] * float(retain_ratio))
        idx = np.arange(action.shape[0])
        rng.shuffle(idx)
        mask[idx[:keep], :2] = 1.0
    else:
        raise ValueError(f"Unsupported mask_type={mask_type!r} for LeRobot MAM conversion.")
    masked = np.full_like(action, mask_value)
    masked[mask > 0.5] = action[mask > 0.5]
    return masked.astype(np.float32), mask.astype(np.float32)


def _to_frame_value(value, feature: dict):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if feature["dtype"] in {"image", "video"} and value.ndim == 3:
        h, w, c = feature["shape"]
        if value.shape == (c, h, w):
            value = np.transpose(value, (1, 2, 0))
    return value


def _patch_episode_metadata(root: Path, rows: dict[int, dict]) -> None:
    for parquet_path in sorted((root / "meta" / "episodes").glob("**/*.parquet")):
        df = pd.read_parquet(parquet_path)
        for key in (
            "source_episode_id",
            "mask_type",
            "mask_type_slot",
            "libero/init_state_id",
            "libero/init_state",
            "libero/suite",
            "libero/task_id",
            "libero/task_name",
            "libero/source_episode_id",
            "libero/source_file",
            "libero/source_demo",
        ):
            values = []
            for episode_index in df["episode_index"].astype(int).tolist():
                values.append(rows.get(episode_index, {}).get(key))
            df[key] = values
        df.to_parquet(parquet_path)


def _write_split(
    source: LeRobotDataset,
    episode_ids: list[int],
    root: Path,
    repo_id: str,
    args: argparse.Namespace,
) -> None:
    if root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{root} exists; pass --overwrite")
        shutil.rmtree(root)

    features = {key: value for key, value in source.meta.features.items() if key not in DEFAULT_FEATURES}
    features[MAM_MAS_ACTION_ABSOLUTE] = {"dtype": "float32", "shape": (7,), "names": None}
    features[MAM_MAS_ACTION_MASK] = {"dtype": "float32", "shape": (7,), "names": None}
    features[MAM_PROGRESS] = {"dtype": "float32", "shape": (1,), "names": None}

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=source.meta.fps,
        robot_type=source.meta.robot_type,
        features=features,
        use_videos=len(source.meta.video_keys) > 0,
    )

    episode_meta_rows = {}
    source_episode_rows = {int(row["episode_index"]): row for row in source.meta.episodes}
    mask_types = _resolve_mask_types(args)
    local_episode_index = 0

    for source_episode_id in episode_ids:
        source_row = source_episode_rows.get(int(source_episode_id), {})
        original_source_episode_id = _row_get(
            source_row,
            "libero/source_episode_id",
            _row_get(source_row, "source_episode_id", source_episode_id),
        )
        start = _row_get(source_row, "dataset_from_index")
        end = _row_get(source_row, "dataset_to_index")
        if start is None or end is None:
            raise ValueError(
                f"Episode {source_episode_id} is missing dataset_from_index/dataset_to_index; "
                "cannot stream materialize MAM split safely."
            )
        frames = [source[idx] for idx in range(int(start), int(end))]
        actions = np.stack([np.asarray(frame[ACTION], dtype=np.float32) for frame in frames], axis=0)
        init_state_id = source_row.get(
            "libero/init_state_id",
            source_row.get("init_state_id", source_episode_id),
        )
        suite = _row_get(source_row, "libero/suite")
        task_id = _row_get(source_row, "libero/task_id")
        task_name = _row_get(source_row, "libero/task_name")
        init_state = _as_float_list(_row_get(source_row, "libero/init_state"))
        denom = max(len(frames) - 1, 1)

        for mask_type_slot, mask_type in enumerate(mask_types):
            rng = np.random.default_rng(
                np.random.SeedSequence([int(args.split_seed), int(source_episode_id), int(mask_type_slot)])
            )
            _, mask = _apply_mask(
                actions,
                mask_type,
                args.retain_ratio,
                args.mask_value,
                rng,
            )
            for frame_index, item in enumerate(frames):
                out = {"task": item["task"]}
                for key, ft in features.items():
                    if key in {MAM_MAS_ACTION_ABSOLUTE, MAM_MAS_ACTION_MASK, MAM_PROGRESS}:
                        continue
                    out[key] = _to_frame_value(item[key], ft)
                out[MAM_MAS_ACTION_ABSOLUTE] = actions[frame_index]
                out[MAM_MAS_ACTION_MASK] = mask[frame_index]
                out[MAM_PROGRESS] = np.asarray([frame_index / denom], dtype=np.float32)
                dataset.add_frame(out)
            dataset.save_episode()

            episode_meta_rows[local_episode_index] = {
                "source_episode_id": int(original_source_episode_id),
                "mask_type": mask_type,
                "mask_type_slot": int(mask_type_slot),
                "libero/init_state_id": int(init_state_id),
                "libero/init_state": init_state,
                "libero/suite": None if suite is None else str(suite),
                "libero/task_id": None if task_id is None else int(task_id),
                "libero/task_name": None if task_name is None else str(task_name),
                "libero/source_episode_id": int(original_source_episode_id),
                "libero/source_file": _row_get(source_row, "libero/source_file"),
                "libero/source_demo": _row_get(source_row, "libero/source_demo"),
            }
            local_episode_index += 1

    dataset.finalize()
    _patch_episode_metadata(root, episode_meta_rows)

    if not args.skip_relative_action_stats:
        reopened = LeRobotDataset(repo_id=repo_id, root=root, return_uint8=True)
        stats = dict(reopened.meta.stats or {})
        action_delta_indices = list(range(1 - args.n_obs_steps, 1 - args.n_obs_steps + args.horizon))
        stats[ACTION] = compute_libero_relative_action_stats(
            hf_dataset=reopened.hf_dataset,
            action_delta_indices=action_delta_indices,
            num_workers=0,
        )
        write_stats(stats, root)

    write_libero_pipeline_manifest(
        root,
        {
            "pipeline_version": LIBERO_PIPELINE_VERSION,
            "stage": "absolute_to_mam",
            "conversion_complete": True,
            "dataset_split": "train" if root.name.endswith("_train") else "eval",
            "action_representation": LIBERO_ABSOLUTE_ACTION,
            "policy_action_representation": LIBERO_CHUNK_RELATIVE_ACTION,
            "relative_action_stats": not args.skip_relative_action_stats,
            "relative_action_stats_n_obs_steps": int(args.n_obs_steps),
            "relative_action_stats_horizon": int(args.horizon),
            "relative_action_stats_action_delta_indices": list(
                range(1 - args.n_obs_steps, 1 - args.n_obs_steps + args.horizon)
            ),
            "observation_materialization": LIBERO_CLOSED_LOOP_ABSOLUTE_MATERIALIZATION,
            "relative_action_ready": True,
            "state_representation": LIBERO_STATE_14D,
            "source_root": str(args.input_root.resolve()),
            "source_repo_id": args.input_repo_id or _repo_id_from_root(args.input_root),
            "source_episode_ids": [int(episode_id) for episode_id in episode_ids],
            "mask_types": _resolve_mask_types(args),
        },
    )


def main() -> None:
    args = parse_args()
    input_manifest = read_libero_pipeline_manifest(args.input_root)
    excluded_episode_ids: set[int] = set()
    valid_absolute_episode_ids: set[int] | None = None
    if input_manifest.get("relative_action_ready") is True:
        input_manifest = require_libero_v3_relative_ready_dataset(args.input_root)
    elif getattr(args, "allow_source_exclusions", False):
        excluded_episode_ids = {
            int(value) for value in input_manifest.get("unrepairable_episode_ids", [])
        }
        valid_absolute_episode_ids = {
            int(value) for value in input_manifest.get("valid_absolute_episode_ids", [])
        }
        if (
            input_manifest.get("pipeline_version") != LIBERO_PIPELINE_VERSION
            or input_manifest.get("stage") != "delta_to_absolute"
            or input_manifest.get("conversion_complete") is not True
            or input_manifest.get("audit_complete") is not True
            or input_manifest.get("observation_materialization")
            != LIBERO_CLOSED_LOOP_ABSOLUTE_MATERIALIZATION
            or not excluded_episode_ids
            or not valid_absolute_episode_ids
            or excluded_episode_ids & valid_absolute_episode_ids
        ):
            raise ValueError(
                "--allow-source-exclusions requires a completed v3 delta_to_absolute replay audit "
                "with disjoint non-empty valid_absolute_episode_ids and unrepairable_episode_ids."
            )
    else:
        input_manifest = require_libero_v3_relative_ready_dataset(args.input_root)
    if input_manifest.get("stage") != "delta_to_absolute":
        raise ValueError(
            "MAM conversion input must be the closed-loop-rematerialized delta_to_absolute v3 dataset, "
            f"got stage={input_manifest.get('stage')!r}."
        )
    mask_types = _resolve_mask_types(args)
    input_repo_id = args.input_repo_id or _repo_id_from_root(args.input_root)
    source = LeRobotDataset(input_repo_id, root=args.input_root, return_uint8=True)
    source_episode_ids = {
        int(row["episode_index"]) for row in source.meta.episodes
    }
    if valid_absolute_episode_ids is not None:
        if valid_absolute_episode_ids | excluded_episode_ids != source_episode_ids:
            raise ValueError(
                "Replay-audit valid/excluded episode ids do not exactly cover source metadata."
            )
        if args.eval_per_task is None:
            raise ValueError(
                "An excluded replay-audit source requires --eval-per-task for a task-balanced split."
            )
    if args.eval_episode_ids is not None:
        train_ids, eval_ids = _selected_explicit_eval_ids(
            source,
            args.eval_episode_ids,
            eval_per_task=args.eval_per_task,
            eligible_episode_ids=valid_absolute_episode_ids,
        )
    elif args.eval_per_task is None:
        if args.task_stratified_eval_ratio:
            train_ids, eval_ids = _selected_episode_ids_by_task_ratio(
                source,
                args.eval_ratio,
                args.split_seed,
            )
        else:
            train_ids, eval_ids = _selected_episode_ids(
                source.meta.total_episodes,
                args.eval_ratio,
                args.split_seed,
            )
    else:
        train_ids, eval_ids = _selected_episode_ids_by_task(
            source,
            args.eval_per_task,
            args.split_seed,
            eligible_episode_ids=valid_absolute_episode_ids,
        )

    train_root = args.output_root.with_name(f"{args.output_root.name}_train")
    eval_root = args.output_root.with_name(f"{args.output_root.name}_eval")
    train_repo = f"{args.output_repo_id}_train"
    eval_repo = f"{args.output_repo_id}_eval"
    if args.only_split in {"both", "train"}:
        _write_split(source, train_ids, train_root, train_repo, args)
    if eval_ids and args.only_split in {"both", "eval"}:
        _write_split(source, eval_ids, eval_root, eval_repo, args)
    manifest = {
        "input_root": str(args.input_root),
        "input_repo_id": input_repo_id,
        "split_seed": args.split_seed,
        "eval_ratio": args.eval_ratio,
        "eval_per_task": args.eval_per_task,
        "explicit_eval_episode_ids": args.eval_episode_ids,
        "mask_types": mask_types,
        "train_episode_ids": train_ids,
        "eval_episode_ids": eval_ids,
        "train_expanded_episode_count": len(train_ids) * len(mask_types),
        "eval_expanded_episode_count": len(eval_ids) * len(mask_types),
        "source_excluded_episode_ids": sorted(excluded_episode_ids),
    }
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    (args.output_root.with_name(f"{args.output_root.name}_split.json")).write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote MAM datasets: train={train_root}, eval={eval_root if eval_ids else 'N/A'}")


if __name__ == "__main__":
    main()
