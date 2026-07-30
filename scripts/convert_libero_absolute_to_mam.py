#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.compute_stats import (
    aggregate_stats,
    compute_episode_stats,
    compute_libero_relative_action_stats,
)
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

MASK_TYPES_REQUIRING_RATIO = {
    "pose",
    "pose_motion_planning",
    "points",
    "3D_points",
    "random_mask",
}
MASK_TYPES_REQUIRING_SEQ_LEN = {"2D_partial_trajectory", "local_planner"}
SUPPORTED_MASK_TYPES = {
    "none",
    "2D_video_trajectory",
    "2D_image_trajectory",
    "mix",
    "mix0",
    "2D_partial_trajectory",
    "pose",
    "pose_motion_planning",
    "points",
    "3D_points",
    "local_planner",
    "random_mask",
}


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
    parser.add_argument(
        "--remask-existing-split",
        action="store_true",
        help=(
            "Treat an existing absolute_to_mam train or eval dataset as the source, preserve all "
            "episodes in that split, and write --output-root directly with new masks."
        ),
    )
    parser.add_argument("--mask-type", type=str, default="random_mask")
    parser.add_argument(
        "--mask-types",
        type=str,
        default=None,
        help="Comma-separated mask types used to build a mixed-mask dataset.",
    )
    parser.add_argument(
        "--mask-assign-mode",
        choices=("one_demo_multi_mask", "composition"),
        default="one_demo_multi_mask",
        help=(
            "one_demo_multi_mask duplicates every source episode for every mask type; "
            "composition assigns exactly one mask type to each source episode while "
            "maintaining the requested proportions independently within every task."
        ),
    )
    parser.add_argument(
        "--mask-composition",
        type=str,
        default=None,
        help="Comma-separated per-type fractions for composition mode. Defaults to equal fractions.",
    )
    parser.add_argument("--retain-ratio", type=float, default=0.2)
    parser.add_argument(
        "--retain-ratios",
        type=str,
        default=None,
        help="Comma-separated retain ratios aligned with --mask-types. A single value is broadcast.",
    )
    parser.add_argument("--mask-seq-len", type=int, default=20)
    parser.add_argument(
        "--mask-seq-lens",
        type=str,
        default=None,
        help="Comma-separated sequence lengths aligned with --mask-types. A single value is broadcast.",
    )
    for split in ("train", "eval"):
        parser.add_argument(
            f"--{split}-mask-types",
            type=str,
            default=None,
            help=f"Comma-separated mask types for the {split} split. Defaults to --mask-types.",
        )
        parser.add_argument(
            f"--{split}-mask-assign-mode",
            choices=("one_demo_multi_mask", "composition"),
            default=None,
            help=f"Mask assignment mode for the {split} split.",
        )
        parser.add_argument(
            f"--{split}-mask-composition",
            type=str,
            default=None,
            help=f"Comma-separated mask fractions for the {split} split.",
        )
        parser.add_argument(
            f"--{split}-retain-ratios",
            type=str,
            default=None,
            help=f"Comma-separated retain ratios for the {split} split.",
        )
        parser.add_argument(
            f"--{split}-mask-seq-lens",
            type=str,
            default=None,
            help=f"Comma-separated mask sequence lengths for the {split} split.",
        )
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


def _resolve_mask_types(args: argparse.Namespace, split: str | None = None) -> list[str]:
    raw_mask_types = (
        getattr(args, f"{split}_mask_types", None)
        if split is not None
        else None
    )
    if raw_mask_types is None:
        raw_mask_types = getattr(args, "mask_types", None)
    if raw_mask_types is None:
        mask_types = [str(args.mask_type).strip()]
    else:
        mask_types = [item.strip() for item in str(raw_mask_types).split(",")]
    if not mask_types or any(not item for item in mask_types):
        raise ValueError("--mask-types must contain one or more non-empty comma-separated values.")
    unsupported = sorted(set(mask_types) - SUPPORTED_MASK_TYPES)
    if unsupported:
        raise ValueError(
            f"Unsupported mask type(s): {unsupported}. Supported: {sorted(SUPPORTED_MASK_TYPES)}"
        )
    return mask_types


def _resolve_aligned_values(
    raw_values: Any,
    count: int,
    default: Any,
    caster,
    option_name: str,
) -> list[Any]:
    if raw_values is None:
        return [caster(default)] * count
    values = [caster(item.strip()) for item in str(raw_values).split(",") if item.strip()]
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(f"{option_name} expects 1 or {count} value(s), got {len(values)}.")
    return values


def _resolve_split_option(args: argparse.Namespace, split: str | None, name: str) -> Any:
    if split is not None:
        split_value = getattr(args, f"{split}_{name}", None)
        if split_value is not None:
            return split_value
    return getattr(args, name, None)


def _resolve_mask_assign_mode(args: argparse.Namespace, split: str | None = None) -> str:
    value = _resolve_split_option(args, split, "mask_assign_mode")
    return "one_demo_multi_mask" if value is None else str(value)


def _resolve_mask_specs(
    args: argparse.Namespace,
    split: str | None = None,
) -> list[dict[str, Any]]:
    mask_types = _resolve_mask_types(args, split=split)
    retain_ratios = _resolve_aligned_values(
        _resolve_split_option(args, split, "retain_ratios"),
        len(mask_types),
        getattr(args, "retain_ratio", 0.2),
        float,
        "--retain-ratios",
    )
    mask_seq_lens = _resolve_aligned_values(
        _resolve_split_option(args, split, "mask_seq_lens"),
        len(mask_types),
        getattr(args, "mask_seq_len", 20),
        int,
        "--mask-seq-lens",
    )
    raw_composition = _resolve_split_option(args, split, "mask_composition")
    if raw_composition is None:
        composition = [1.0 / len(mask_types)] * len(mask_types)
    else:
        composition = _resolve_aligned_values(
            raw_composition,
            len(mask_types),
            1.0,
            float,
            "--mask-composition",
        )
    if any(ratio < 0.0 or ratio > 1.0 for ratio in retain_ratios):
        raise ValueError(f"retain ratios must be in [0, 1], got {retain_ratios}.")
    if any(length <= 0 for length in mask_seq_lens):
        raise ValueError(f"mask sequence lengths must be positive, got {mask_seq_lens}.")
    if any(weight < 0.0 for weight in composition) or not np.isclose(sum(composition), 1.0):
        raise ValueError(f"mask composition must be non-negative and sum to 1, got {composition}.")

    return [
        {
            "mask_type": mask_type,
            "mask_type_slot": slot,
            "composition": float(composition[slot]),
            "retain_ratio": (
                float(retain_ratios[slot]) if mask_type in MASK_TYPES_REQUIRING_RATIO else None
            ),
            "mask_seq_len": (
                int(mask_seq_lens[slot]) if mask_type in MASK_TYPES_REQUIRING_SEQ_LEN else None
            ),
        }
        for slot, mask_type in enumerate(mask_types)
    ]


def _largest_remainder_counts(total: int, weights: list[float]) -> list[int]:
    exact = np.asarray(weights, dtype=np.float64) * int(total)
    counts = np.floor(exact).astype(np.int64)
    remaining = int(total) - int(counts.sum())
    if remaining > 0:
        order = sorted(range(len(weights)), key=lambda idx: (-(exact[idx] - counts[idx]), idx))
        for idx in order[:remaining]:
            counts[idx] += 1
    return [int(value) for value in counts]


def _assign_mask_specs(
    episode_ids: list[int],
    mask_specs: list[dict[str, Any]],
    assign_mode: str,
    seed: int,
    task_ids_by_episode: dict[int, int] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    if assign_mode == "one_demo_multi_mask":
        return {int(episode_id): mask_specs for episode_id in episode_ids}
    if assign_mode != "composition":
        raise ValueError(f"Unsupported mask_assign_mode={assign_mode!r}.")
    if task_ids_by_episode is None:
        raise ValueError("composition mask assignment requires a task id for every episode.")
    missing_episode_ids = sorted(set(episode_ids) - set(task_ids_by_episode))
    if missing_episode_ids:
        raise ValueError(
            f"composition mask assignment is missing task ids for episode(s): {missing_episode_ids}."
        )

    episode_ids_by_task: dict[int, list[int]] = {}
    for episode_id in episode_ids:
        task_id = int(task_ids_by_episode[int(episode_id)])
        episode_ids_by_task.setdefault(task_id, []).append(int(episode_id))

    assigned: dict[int, list[dict[str, Any]]] = {}
    weights = [float(spec["composition"]) for spec in mask_specs]
    for task_id in sorted(episode_ids_by_task):
        task_episode_ids = episode_ids_by_task[task_id]
        counts = _largest_remainder_counts(len(task_episode_ids), weights)
        shuffled_ids = np.random.default_rng(np.random.SeedSequence([int(seed), int(task_id)])).permutation(
            np.asarray(sorted(task_episode_ids), dtype=np.int64)
        )
        cursor = 0
        for spec, count in zip(mask_specs, counts, strict=True):
            for episode_id in shuffled_ids[cursor : cursor + count]:
                assigned[int(episode_id)] = [spec]
            cursor += count
    if len(assigned) != len(episode_ids):
        raise AssertionError("Per-task mask composition did not assign every source episode.")
    return assigned


def _repo_id_from_root(root: Path) -> str:
    return f"local/{root.name}"


def _absolute_source_provenance(args: argparse.Namespace) -> tuple[str, str]:
    input_manifest = read_libero_pipeline_manifest(args.input_root)
    if input_manifest.get("stage") == "absolute_to_mam":
        source_root = input_manifest.get("source_root")
        source_repo_id = input_manifest.get("source_repo_id")
        if not source_root or not source_repo_id:
            raise ValueError(
                "An existing MAM split must retain its absolute source_root and source_repo_id."
            )
        return str(source_root), str(source_repo_id)
    return (
        str(args.input_root.resolve()),
        args.input_repo_id or _repo_id_from_root(args.input_root),
    )


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


def _task_ids_by_episode(
    episode_rows: dict[int, Any],
    episode_ids: list[int],
) -> dict[int, int]:
    task_ids: dict[int, int] = {}
    missing_episode_ids: list[int] = []
    for episode_id in episode_ids:
        row = episode_rows.get(int(episode_id), {})
        task_id = _row_get(row, "libero/task_id", _row_get(row, "task_id"))
        if task_id is None:
            missing_episode_ids.append(int(episode_id))
        else:
            task_ids[int(episode_id)] = int(task_id)
    if missing_episode_ids:
        raise ValueError(
            "Per-task mask composition requires episode metadata column "
            "'libero/task_id' or 'task_id'; missing for episode(s): "
            f"{missing_episode_ids}."
        )
    return task_ids


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
    retain_ratio: float | None,
    mask_value: float,
    rng,
    mask_seq_len: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    action = np.asarray(action, dtype=np.float32)
    if action.ndim != 2 or action.shape[1] != 7:
        raise ValueError(f"MAM actions must have shape (T, 7), got {action.shape}.")
    if mask_type not in SUPPORTED_MASK_TYPES:
        raise ValueError(f"Unsupported mask_type={mask_type!r} for LeRobot MAM conversion.")
    if mask_type in MASK_TYPES_REQUIRING_RATIO and retain_ratio is None:
        raise ValueError(f"mask_type={mask_type!r} requires retain_ratio.")
    if retain_ratio is not None and not 0.0 <= float(retain_ratio) <= 1.0:
        raise ValueError(f"retain_ratio must be in [0, 1], got {retain_ratio}.")
    if mask_type in MASK_TYPES_REQUIRING_SEQ_LEN and (
        mask_seq_len is None or int(mask_seq_len) <= 0
    ):
        raise ValueError(f"mask_type={mask_type!r} requires a positive mask_seq_len.")

    n, _ = action.shape
    mask = np.zeros_like(action, dtype=np.float32)
    if mask_type == "none":
        return np.full_like(action, mask_value), mask
    if n == 0:
        return np.full_like(action, mask_value), mask

    if mask_type in {"2D_video_trajectory", "2D_image_trajectory"}:
        mask[:, :2] = 1.0
    elif mask_type in {"mix", "mix0"}:
        if n < 4:
            raise ValueError(f"{mask_type} requires trajectory length >= 4, got {n}.")
        mask[:, :2] = 1.0
        idx = np.arange(n)
        rng.shuffle(idx)
        mask[idx[0], :] = 1.0
        mask[idx[1:4], :3] = 1.0
    elif mask_type == "2D_partial_trajectory":
        if int(mask_seq_len) >= n:
            raise ValueError(
                f"mask_seq_len ({mask_seq_len}) must be smaller than trajectory length ({n})."
            )
        start = int(rng.integers(0, n - int(mask_seq_len) + 1))
        mask[start : start + int(mask_seq_len), :2] = 1.0
    elif mask_type in {"pose", "pose_motion_planning"}:
        keep = int(n * float(retain_ratio))
        idx = np.arange(n)
        rng.shuffle(idx)
        mask[idx[:keep], :] = 1.0
    elif mask_type == "random_mask":
        total = action.size
        keep = int(total * float(retain_ratio))
        if keep > 0:
            idx = np.arange(total)
            rng.shuffle(idx)
            mask.reshape(-1)[idx[:keep]] = 1.0
    elif mask_type == "3D_points":
        keep = int(n * float(retain_ratio))
        idx = np.arange(n)
        rng.shuffle(idx)
        mask[idx[:keep], :3] = 1.0
    elif mask_type == "points":
        keep = int(n * float(retain_ratio))
        idx = np.arange(n)
        rng.shuffle(idx)
        mask[idx[:keep], :2] = 1.0
    elif mask_type == "local_planner":
        if int(mask_seq_len) >= n:
            raise ValueError(
                f"mask_seq_len ({mask_seq_len}) must be smaller than trajectory length ({n})."
            )
        mask[:, :] = 1.0
        start = int(rng.integers(0, n - int(mask_seq_len) + 1))
        mask[start : start + int(mask_seq_len), :] = 0.0
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
            "retain_ratio",
            "mask_seq_len",
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
        temporary_path = parquet_path.with_suffix(f"{parquet_path.suffix}.tmp")
        df.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, parquet_path)


def _copy_existing_split_for_remask(source_root: Path, output_root: Path) -> None:
    """Clone a materialized MAM split while avoiding another image decode/encode pass."""

    source_data_root = (source_root.resolve() / "data")

    def copy_file(source_path: str, destination_path: str) -> str:
        source = Path(source_path).resolve()
        if source.is_relative_to(source_data_root):
            os.link(source_path, destination_path)
            return destination_path
        return shutil.copy2(source_path, destination_path)

    shutil.copytree(source_root, output_root, copy_function=copy_file)


def _write_existing_split_with_new_masks(
    source: LeRobotDataset,
    root: Path,
    args: argparse.Namespace,
) -> None:
    """Fast path for composition remasking: preserve every column except the mask."""

    dataset_split = str(read_libero_pipeline_manifest(args.input_root)["dataset_split"])
    episode_ids = sorted(int(row["episode_index"]) for row in source.meta.episodes)
    source_episode_rows = {int(row["episode_index"]): row for row in source.meta.episodes}
    mask_specs = _resolve_mask_specs(args, split=dataset_split)
    mask_assign_mode = _resolve_mask_assign_mode(args, split=dataset_split)
    if mask_assign_mode != "composition":
        raise ValueError(
            "Fast --remask-existing-split currently requires mask_assign_mode=composition "
            "because one_demo_multi_mask changes the episode count."
        )
    assigned_mask_specs = _assign_mask_specs(
        episode_ids,
        mask_specs,
        mask_assign_mode,
        args.split_seed,
        task_ids_by_episode=(
            _task_ids_by_episode(source_episode_rows, episode_ids)
            if mask_assign_mode == "composition"
            else None
        ),
    )

    if root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{root} exists; pass --overwrite")
        shutil.rmtree(root)
    _copy_existing_split_for_remask(args.input_root, root)

    episode_meta_rows: dict[int, dict[str, Any]] = {}
    episode_mask_stats: dict[int, dict[str, np.ndarray]] = {}
    mask_feature = {MAM_MAS_ACTION_MASK: source.meta.features[MAM_MAS_ACTION_MASK]}
    seen_episode_ids: set[int] = set()

    for parquet_path in sorted((root / "data").glob("**/*.parquet")):
        table = pq.read_table(parquet_path)
        actions = np.asarray(table[MAM_MAS_ACTION_ABSOLUTE].to_pylist(), dtype=np.float32)
        parquet_episode_ids = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        masks = np.zeros_like(actions, dtype=np.float32)

        for episode_id in np.unique(parquet_episode_ids):
            episode_id = int(episode_id)
            if episode_id in seen_episode_ids:
                raise ValueError(
                    f"Episode {episode_id} spans more than one parquet file; "
                    "cannot use the fast remask path safely."
                )
            seen_episode_ids.add(episode_id)
            row_indices = np.flatnonzero(parquet_episode_ids == episode_id)
            mask_spec = assigned_mask_specs[episode_id][0]
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [int(args.split_seed), episode_id, int(mask_spec["mask_type_slot"])]
                )
            )
            _, episode_mask = _apply_mask(
                actions[row_indices],
                str(mask_spec["mask_type"]),
                mask_spec["retain_ratio"],
                args.mask_value,
                rng,
                mask_seq_len=mask_spec["mask_seq_len"],
            )
            masks[row_indices] = episode_mask
            episode_mask_stats[episode_id] = compute_episode_stats(
                {MAM_MAS_ACTION_MASK: episode_mask},
                mask_feature,
            )[MAM_MAS_ACTION_MASK]

            source_row = source_episode_rows[episode_id]
            original_source_episode_id = int(
                _row_get(
                    source_row,
                    "libero/source_episode_id",
                    _row_get(source_row, "source_episode_id", episode_id),
                )
            )
            episode_meta_rows[episode_id] = {
                "source_episode_id": original_source_episode_id,
                "mask_type": str(mask_spec["mask_type"]),
                "mask_type_slot": int(mask_spec["mask_type_slot"]),
                "retain_ratio": mask_spec["retain_ratio"],
                "mask_seq_len": mask_spec["mask_seq_len"],
                "libero/init_state_id": int(
                    _row_get(source_row, "libero/init_state_id", episode_id)
                ),
                "libero/init_state": _as_float_list(_row_get(source_row, "libero/init_state")),
                "libero/suite": _row_get(source_row, "libero/suite"),
                "libero/task_id": _row_get(source_row, "libero/task_id"),
                "libero/task_name": _row_get(source_row, "libero/task_name"),
                "libero/source_episode_id": original_source_episode_id,
                "libero/source_file": _row_get(source_row, "libero/source_file"),
                "libero/source_demo": _row_get(source_row, "libero/source_demo"),
            }

        mask_array = pa.FixedSizeListArray.from_arrays(
            pa.array(masks.reshape(-1), type=pa.float32()),
            actions.shape[1],
        )
        mask_column_index = table.schema.get_field_index(MAM_MAS_ACTION_MASK)
        table = table.set_column(
            mask_column_index,
            table.schema.field(mask_column_index),
            mask_array,
        )
        temporary_path = parquet_path.with_suffix(f"{parquet_path.suffix}.tmp")
        pq.write_table(table, temporary_path)
        os.replace(temporary_path, parquet_path)

    if seen_episode_ids != set(episode_ids):
        raise ValueError(
            "Fast remask did not cover exactly the source episodes: "
            f"missing={sorted(set(episode_ids) - seen_episode_ids)}, "
            f"extra={sorted(seen_episode_ids - set(episode_ids))}."
        )

    _patch_episode_metadata(root, episode_meta_rows)
    for parquet_path in sorted((root / "meta" / "episodes").glob("**/*.parquet")):
        df = pd.read_parquet(parquet_path)
        for stat_name in ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"):
            column = f"stats/{MAM_MAS_ACTION_MASK}/{stat_name}"
            df[column] = [
                episode_mask_stats[int(episode_id)][stat_name].tolist()
                for episode_id in df["episode_index"].astype(int)
            ]
        temporary_path = parquet_path.with_suffix(f"{parquet_path.suffix}.tmp")
        df.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, parquet_path)

    stats = dict(source.meta.stats or {})
    stats[MAM_MAS_ACTION_MASK] = aggregate_stats(
        [
            {MAM_MAS_ACTION_MASK: episode_mask_stats[episode_id]}
            for episode_id in episode_ids
        ]
    )[MAM_MAS_ACTION_MASK]
    write_stats(stats, root)

    manifest_source_episode_ids = [
        int(
            _row_get(
                source_episode_rows[episode_id],
                "libero/source_episode_id",
                _row_get(source_episode_rows[episode_id], "source_episode_id", episode_id),
            )
        )
        for episode_id in episode_ids
    ]
    absolute_source_root, absolute_source_repo_id = _absolute_source_provenance(args)
    write_libero_pipeline_manifest(
        root,
        {
            "pipeline_version": LIBERO_PIPELINE_VERSION,
            "stage": "absolute_to_mam",
            "conversion_complete": True,
            "dataset_split": dataset_split,
            "action_representation": LIBERO_ABSOLUTE_ACTION,
            "policy_action_representation": LIBERO_CHUNK_RELATIVE_ACTION,
            "relative_action_stats": True,
            "relative_action_stats_n_obs_steps": int(args.n_obs_steps),
            "relative_action_stats_horizon": int(args.horizon),
            "relative_action_stats_action_delta_indices": list(
                range(1 - args.n_obs_steps, 1 - args.n_obs_steps + args.horizon)
            ),
            "observation_materialization": LIBERO_CLOSED_LOOP_ABSOLUTE_MATERIALIZATION,
            "relative_action_ready": True,
            "state_representation": LIBERO_STATE_14D,
            "source_root": absolute_source_root,
            "source_repo_id": absolute_source_repo_id,
            "source_episode_ids": manifest_source_episode_ids,
            "mask_types": [str(spec["mask_type"]) for spec in mask_specs],
            "mask_assign_mode": mask_assign_mode,
            "mask_composition_scope": "per_task",
            "mask_specs": mask_specs,
            "source_episode_count": len(episode_ids),
            "expanded_episode_count": len(episode_ids),
        },
    )


def _write_split(
    source: LeRobotDataset,
    episode_ids: list[int],
    root: Path,
    repo_id: str,
    args: argparse.Namespace,
) -> None:
    dataset_split = "train" if root.name.endswith("_train") else "eval"
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
    manifest_source_episode_ids = [
        int(
            _row_get(
                source_episode_rows.get(int(episode_id), {}),
                "libero/source_episode_id",
                _row_get(
                    source_episode_rows.get(int(episode_id), {}),
                    "source_episode_id",
                    episode_id,
                ),
            )
        )
        for episode_id in episode_ids
    ]
    mask_specs = _resolve_mask_specs(args, split=dataset_split)
    mask_assign_mode = _resolve_mask_assign_mode(args, split=dataset_split)
    assigned_mask_specs = _assign_mask_specs(
        episode_ids,
        mask_specs,
        mask_assign_mode,
        args.split_seed,
        task_ids_by_episode=(
            _task_ids_by_episode(source_episode_rows, episode_ids)
            if mask_assign_mode == "composition"
            else None
        ),
    )
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

        for mask_spec in assigned_mask_specs[int(source_episode_id)]:
            mask_type = str(mask_spec["mask_type"])
            mask_type_slot = int(mask_spec["mask_type_slot"])
            rng = np.random.default_rng(
                np.random.SeedSequence([int(args.split_seed), int(source_episode_id), int(mask_type_slot)])
            )
            _, mask = _apply_mask(
                actions,
                mask_type,
                mask_spec["retain_ratio"],
                args.mask_value,
                rng,
                mask_seq_len=mask_spec["mask_seq_len"],
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
                "retain_ratio": mask_spec["retain_ratio"],
                "mask_seq_len": mask_spec["mask_seq_len"],
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

    absolute_source_root, absolute_source_repo_id = _absolute_source_provenance(args)
    write_libero_pipeline_manifest(
        root,
        {
            "pipeline_version": LIBERO_PIPELINE_VERSION,
            "stage": "absolute_to_mam",
            "conversion_complete": True,
            "dataset_split": dataset_split,
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
            "source_root": absolute_source_root,
            "source_repo_id": absolute_source_repo_id,
            "source_episode_ids": manifest_source_episode_ids,
            "mask_types": [str(spec["mask_type"]) for spec in mask_specs],
            "mask_assign_mode": mask_assign_mode,
            "mask_composition_scope": (
                "per_task" if mask_assign_mode == "composition" else "all_source_episodes"
            ),
            "mask_specs": mask_specs,
            "source_episode_count": len(episode_ids),
            "expanded_episode_count": (
                len(episode_ids) * len(mask_specs)
                if mask_assign_mode == "one_demo_multi_mask"
                else len(episode_ids)
            ),
        },
    )


def main() -> None:
    args = parse_args()
    input_manifest = read_libero_pipeline_manifest(args.input_root)
    if getattr(args, "remask_existing_split", False):
        dataset_split = input_manifest.get("dataset_split")
        if (
            input_manifest.get("stage") != "absolute_to_mam"
            or dataset_split not in {"train", "eval"}
            or input_manifest.get("action_representation") != LIBERO_ABSOLUTE_ACTION
        ):
            raise ValueError(
                "--remask-existing-split requires an absolute_to_mam train or eval dataset "
                "with absolute actions."
            )
        if args.input_root.resolve() == args.output_root.resolve():
            raise ValueError("Remasking requires a distinct --output-root; refusing in-place overwrite.")
        input_repo_id = args.input_repo_id or _repo_id_from_root(args.input_root)
        source = LeRobotDataset(input_repo_id, root=args.input_root, return_uint8=True)
        episode_ids = sorted(int(row["episode_index"]) for row in source.meta.episodes)
        if _resolve_mask_assign_mode(args, split=str(dataset_split)) == "composition":
            _write_existing_split_with_new_masks(source, args.output_root, args)
        else:
            _write_split(source, episode_ids, args.output_root, args.output_repo_id, args)
        print(
            f"Remasked {dataset_split} dataset: source={args.input_root}, "
            f"output={args.output_root}, episodes={len(episode_ids)}"
        )
        return

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
    train_mask_specs = _resolve_mask_specs(args, split="train")
    eval_mask_specs = _resolve_mask_specs(args, split="eval")
    train_mask_types = [str(spec["mask_type"]) for spec in train_mask_specs]
    eval_mask_types = [str(spec["mask_type"]) for spec in eval_mask_specs]
    train_mask_assign_mode = _resolve_mask_assign_mode(args, split="train")
    eval_mask_assign_mode = _resolve_mask_assign_mode(args, split="eval")
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
        "train_mask_types": train_mask_types,
        "eval_mask_types": eval_mask_types,
        "train_mask_assign_mode": train_mask_assign_mode,
        "eval_mask_assign_mode": eval_mask_assign_mode,
        "train_mask_composition_scope": (
            "per_task" if train_mask_assign_mode == "composition" else "all_source_episodes"
        ),
        "eval_mask_composition_scope": (
            "per_task" if eval_mask_assign_mode == "composition" else "all_source_episodes"
        ),
        "train_mask_specs": train_mask_specs,
        "eval_mask_specs": eval_mask_specs,
        "train_episode_ids": train_ids,
        "eval_episode_ids": eval_ids,
        "train_expanded_episode_count": (
            len(train_ids) * len(train_mask_types)
            if train_mask_assign_mode == "one_demo_multi_mask"
            else len(train_ids)
        ),
        "eval_expanded_episode_count": (
            len(eval_ids) * len(eval_mask_types)
            if eval_mask_assign_mode == "one_demo_multi_mask"
            else len(eval_ids)
        ),
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
