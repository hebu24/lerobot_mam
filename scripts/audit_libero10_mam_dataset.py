#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.compute_stats import compute_libero_relative_action_stats
from lerobot.datasets.libero_pipeline import (
    LIBERO_ABSOLUTE_ACTION,
    LIBERO_CHUNK_RELATIVE_ACTION,
    require_libero_v3_action_dataset,
)
from lerobot.processor.libero_relative_action_processor import (
    absolute_to_chunk_relative,
    axis_angle_to_matrix,
    chunk_relative_to_absolute,
)
from lerobot.utils.constants import ACTION, OBS_STATE

MAM_ACTION = "mam.mas_action_absolute"
MAM_MASK = "mam.mas_action_mask"
MAM_PROGRESS = "mam.progress"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit final LIBERO-10 MAM train/eval artifacts.")
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--train-repo-id", default=None)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--eval-repo-id", default=None)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--source-repo-id", default=None)
    parser.add_argument("--train-per-task", type=int, default=45)
    parser.add_argument("--eval-per-task", type=int, default=5)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--roundtrip-samples", type=int, default=2048)
    parser.add_argument("--skip-stats-recompute", action="store_true")
    return parser.parse_args()


def _repo_id(root: Path, configured: str | None) -> str:
    return configured or f"local/{root.name}"


def _episode_value(row, keys: tuple[str, ...]):
    columns = set(getattr(row, "column_names", []) or [])
    for key in keys:
        if not columns or key in columns:
            try:
                value = row[key]
            except (KeyError, TypeError):
                continue
            if value is not None:
                return value
    return None


def _episode_summary(dataset: LeRobotDataset) -> tuple[dict[int, int], set[int]]:
    counts: dict[int, int] = {}
    sources: set[int] = set()
    for row in dataset.meta.episodes:
        task_id = _episode_value(row, ("libero/task_id", "task_id"))
        source_id = _episode_value(
            row,
            ("libero/source_episode_id", "source_episode_id", "episode_index"),
        )
        if task_id is None or source_id is None:
            raise AssertionError("Episode metadata must contain task_id and source_episode_id.")
        counts[int(task_id)] = counts.get(int(task_id), 0) + 1
        sources.add(int(source_id))
    return counts, sources


def _check_schema(dataset: LeRobotDataset, root: Path) -> None:
    features = dataset.meta.features
    required = {ACTION, OBS_STATE, MAM_ACTION, MAM_MASK, MAM_PROGRESS}
    missing = sorted(required - set(features))
    if missing:
        raise AssertionError(f"{root}: missing features {missing}")
    if tuple(features[OBS_STATE]["shape"]) != (14,):
        raise AssertionError(f"{root}: expected 14D state, got {features[OBS_STATE]['shape']}")
    if tuple(features[ACTION]["shape"]) != (7,):
        raise AssertionError(f"{root}: expected 7D action, got {features[ACTION]['shape']}")
    camera_keys = list(dataset.meta.camera_keys)
    if len(camera_keys) != 2:
        raise AssertionError(f"{root}: expected two cameras, got {camera_keys}")
    for key in camera_keys:
        shape = tuple(features[key]["shape"])
        if shape not in {(128, 128, 3), (3, 128, 128)}:
            raise AssertionError(f"{root}: camera {key} must be 128x128 RGB, got {shape}")


def _check_mam_columns(dataset: LeRobotDataset, root: Path) -> None:
    table = dataset.hf_dataset.select_columns([ACTION, MAM_ACTION, MAM_MASK, MAM_PROGRESS])
    action = np.asarray(table[ACTION], dtype=np.float32)
    mas = np.asarray(table[MAM_ACTION], dtype=np.float32)
    mask = np.asarray(table[MAM_MASK], dtype=np.float32)
    progress = np.asarray(table[MAM_PROGRESS], dtype=np.float32)
    if not np.allclose(mas, action, atol=1e-6):
        error = float(np.max(np.abs(mas - action)))
        raise AssertionError(
            f"{root}: MAS must store complete absolute actions before masking; max error={error:.6g}"
        )
    if not np.all((mask == 0.0) | (mask == 1.0)):
        raise AssertionError(f"{root}: MAS mask must be binary.")
    if not np.isfinite(progress).all() or progress.min() < 0.0 or progress.max() > 1.0:
        raise AssertionError(f"{root}: progress must be finite and within [0, 1].")


def _check_init_state_fidelity(
    dataset: LeRobotDataset,
    root: Path,
    source: LeRobotDataset,
) -> None:
    source_rows = {int(row["episode_index"]): row for row in source.meta.episodes}
    for row in dataset.meta.episodes:
        source_id = _episode_value(
            row,
            ("libero/source_episode_id", "source_episode_id"),
        )
        split_state = _episode_value(row, ("libero/init_state", "init_state"))
        if source_id is None or split_state is None:
            raise AssertionError(f"{root}: episode metadata lacks source id or raw init_state.")
        source_row = source_rows.get(int(source_id))
        if source_row is None:
            raise AssertionError(f"{root}: source episode {source_id} does not exist.")
        source_state = _episode_value(source_row, ("libero/init_state", "init_state"))
        if source_state is None:
            raise AssertionError(f"Source episode {source_id} lacks raw init_state.")
        lhs = np.asarray(split_state, dtype=np.float64)
        rhs = np.asarray(source_state, dtype=np.float64)
        if not np.array_equal(lhs, rhs):
            error = float(np.max(np.abs(lhs - rhs)))
            raise AssertionError(
                f"{root}: source episode {source_id} init_state changed; max error={error:.6g}"
            )


def _check_relative_stats(
    dataset: LeRobotDataset,
    root: Path,
    action_delta_indices: list[int],
) -> None:
    expected = compute_libero_relative_action_stats(
        dataset.hf_dataset,
        action_delta_indices=action_delta_indices,
        num_workers=0,
    )
    actual = (dataset.meta.stats or {}).get(ACTION)
    if actual is None:
        raise AssertionError(f"{root}: missing action stats.")
    for key in ("min", "max", "mean", "std", "q01", "q99"):
        lhs = np.asarray(actual[key], dtype=np.float64)
        rhs = np.asarray(expected[key], dtype=np.float64)
        if not np.allclose(lhs, rhs, rtol=1e-5, atol=1e-6):
            error = float(np.max(np.abs(lhs - rhs)))
            raise AssertionError(f"{root}: action stats {key} are not chunk-relative; max error={error:.6g}")


def _check_roundtrip(
    dataset: LeRobotDataset,
    root: Path,
    action_delta_indices: list[int],
    max_samples: int,
) -> None:
    table = dataset.hf_dataset.select_columns([ACTION, OBS_STATE, "episode_index"])
    actions = np.asarray(table[ACTION], dtype=np.float32)
    states = np.asarray(table[OBS_STATE], dtype=np.float32)
    episode_ids = np.asarray(table["episode_index"], dtype=np.int64)
    offsets = np.asarray(action_delta_indices, dtype=np.int64)
    anchors = np.arange(len(actions), dtype=np.int64)
    indices = anchors[:, None] + offsets[None, :]
    valid = (indices >= 0) & (indices < len(actions))
    safe = np.clip(indices, 0, max(len(actions) - 1, 0))
    valid &= episode_ids[safe] == episode_ids[anchors, None]
    anchors = anchors[np.all(valid, axis=1)]
    if len(anchors) == 0:
        raise AssertionError(f"{root}: no valid action chunks for roundtrip.")
    if len(anchors) > max_samples:
        anchors = anchors[np.linspace(0, len(anchors) - 1, max_samples, dtype=np.int64)]
    chunks = actions[anchors[:, None] + offsets[None, :]]
    relative = absolute_to_chunk_relative(chunks, states[anchors])
    restored = np.asarray(chunk_relative_to_absolute(relative, states[anchors]), dtype=np.float32)
    pos_grip_error = float(np.max(np.abs(restored[..., [0, 1, 2, 6]] - chunks[..., [0, 1, 2, 6]])))
    expected_rot = np.asarray(axis_angle_to_matrix(chunks[..., 3:6]))
    actual_rot = np.asarray(axis_angle_to_matrix(restored[..., 3:6]))
    rot_error = float(np.max(np.abs(expected_rot - actual_rot)))
    if pos_grip_error > 2e-5 or rot_error > 2e-5:
        raise AssertionError(
            f"{root}: SE(3) roundtrip failed: position/gripper={pos_grip_error:.6g}, rotation={rot_error:.6g}"
        )


def main() -> None:
    args = parse_args()
    train_manifest = require_libero_v3_action_dataset(
        args.train_root,
        action_representation=LIBERO_ABSOLUTE_ACTION,
    )
    eval_manifest = require_libero_v3_action_dataset(
        args.eval_root,
        action_representation=LIBERO_ABSOLUTE_ACTION,
    )
    for root, manifest, expected_split in (
        (args.train_root, train_manifest, "train"),
        (args.eval_root, eval_manifest, "eval"),
    ):
        if manifest.get("stage") != "absolute_to_mam":
            raise AssertionError(f"{root}: expected absolute_to_mam manifest stage.")
        if manifest.get("dataset_split") != expected_split:
            raise AssertionError(f"{root}: expected dataset_split={expected_split!r}.")
        if manifest.get("policy_action_representation") != LIBERO_CHUNK_RELATIVE_ACTION:
            raise AssertionError(f"{root}: missing chunk-relative policy action declaration.")

    train = LeRobotDataset(_repo_id(args.train_root, args.train_repo_id), root=args.train_root)
    eval_dataset = LeRobotDataset(_repo_id(args.eval_root, args.eval_repo_id), root=args.eval_root)
    source = None
    if args.source_root is not None:
        require_libero_v3_action_dataset(
            args.source_root,
            action_representation=LIBERO_ABSOLUTE_ACTION,
        )
        source = LeRobotDataset(_repo_id(args.source_root, args.source_repo_id), root=args.source_root)
    action_delta_indices = list(range(1 - args.n_obs_steps, 1 - args.n_obs_steps + args.horizon))

    for dataset, root in ((train, args.train_root), (eval_dataset, args.eval_root)):
        _check_schema(dataset, root)
        _check_mam_columns(dataset, root)
        _check_roundtrip(dataset, root, action_delta_indices, args.roundtrip_samples)
        if source is not None:
            _check_init_state_fidelity(dataset, root, source)
        if not args.skip_stats_recompute:
            _check_relative_stats(dataset, root, action_delta_indices)

    train_counts, train_sources = _episode_summary(train)
    eval_counts, eval_sources = _episode_summary(eval_dataset)
    expected_tasks = set(range(10))
    if set(train_counts) != expected_tasks or set(eval_counts) != expected_tasks:
        raise AssertionError(f"Expected task ids 0..9, got train={train_counts}, eval={eval_counts}")
    if any(count != args.train_per_task for count in train_counts.values()):
        raise AssertionError(f"Train task counts mismatch: {train_counts}")
    if any(count != args.eval_per_task for count in eval_counts.values()):
        raise AssertionError(f"Eval task counts mismatch: {eval_counts}")
    overlap = sorted(train_sources & eval_sources)
    if overlap:
        raise AssertionError(f"Train/eval source episode leakage: {overlap[:20]}")

    print(
        "PASS: v3 manifest, schema, complete MAS, split, init_state fidelity, "
        "relative stats, and SE(3) roundtrip; "
        f"train={train_counts}, eval={eval_counts}"
    )


if __name__ == "__main__":
    main()
