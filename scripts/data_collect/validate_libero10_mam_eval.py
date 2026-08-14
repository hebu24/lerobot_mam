#!/usr/bin/env python
"""Validate a fixed LIBERO-10 MAM evaluation dataset before upload."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.libero_pipeline import require_libero_v3_relative_ready_dataset

MAM_FEATURES = {"mam.mas_action_absolute", "mam.mas_action_mask", "mam.progress"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/datasets/libero10_100_eval_lpb"))
    parser.add_argument("--repo-id", default="local/libero10_100_eval_lpb")
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--min-rollout-seed", type=int, default=100_000)
    parser.add_argument(
        "--train-root",
        type=Path,
        default=Path("outputs/datasets/libero10_1000_train"),
        help="Training dataset whose fixed initial states must not overlap the eval states.",
    )
    return parser.parse_args()


def _episode_rows(root: Path) -> list[dict]:
    paths = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata found under {root}.")
    rows = []
    for path in paths:
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def _state_key(row: dict) -> tuple[int, bytes]:
    return (
        int(row["libero/task_id"]),
        np.asarray(row["libero/init_state"], dtype=np.float64).reshape(-1).tobytes(),
    )


def main() -> None:
    args = parse_args()
    manifest = require_libero_v3_relative_ready_dataset(args.root)
    if manifest.get("dataset_split") != "eval" or manifest.get("stage") != "absolute_to_mam":
        raise ValueError(f"Unexpected MAM eval manifest: {manifest}")
    info = json.loads((args.root / "meta" / "info.json").read_text(encoding="utf-8"))
    expected_total = 10 * args.episodes_per_task
    if int(info["total_episodes"]) != expected_total:
        raise ValueError(f"Expected {expected_total} episodes, got {info['total_episodes']}.")
    missing = MAM_FEATURES - set(info["features"])
    if missing:
        raise ValueError(f"Missing MAM features: {sorted(missing)}")

    rows = _episode_rows(args.root)
    counts = Counter(int(row["libero/task_id"]) for row in rows)
    expected_counts = dict.fromkeys(range(10), args.episodes_per_task)
    if dict(sorted(counts.items())) != expected_counts:
        raise ValueError(f"Unexpected per-task counts: {dict(sorted(counts.items()))}")
    if any(not row.get("libero/init_state") for row in rows):
        raise ValueError("Every fixed evaluation episode must contain a raw init state.")
    if any(row.get("libero/rollout_seed") is None for row in rows):
        raise ValueError("Every LPB evaluation episode must preserve libero/rollout_seed.")
    seeds_by_task: dict[int, list[int]] = {
        task_id: sorted(
            int(row["libero/rollout_seed"]) for row in rows if int(row["libero/task_id"]) == task_id
        )
        for task_id in range(10)
    }
    if any(seed < args.min_rollout_seed for seeds in seeds_by_task.values() for seed in seeds):
        raise ValueError(f"Evaluation rollout seeds must be >= {args.min_rollout_seed}: {seeds_by_task}")
    if any(len(seeds) != len(set(seeds)) for seeds in seeds_by_task.values()):
        raise ValueError(f"Evaluation rollout seeds must be unique within every task: {seeds_by_task}")
    if any(int(row["libero/init_state_id"]) != int(row["libero/rollout_seed"]) for row in rows):
        raise ValueError("Native-seeded episodes must use rollout_seed as their stable init_state_id.")

    unique_states = {_state_key(row) for row in rows}
    if len(unique_states) != expected_total:
        raise ValueError("Fixed evaluation init states are not unique within tasks.")
    train_states = {
        _state_key(row) for row in _episode_rows(args.train_root) if row.get("libero/init_state") is not None
    }
    overlap = unique_states & train_states
    if overlap:
        raise ValueError(f"Found {len(overlap)} eval init states that exactly overlap the train set.")

    source_root = Path(manifest["source_root"])
    source_manifest = json.loads((source_root / "meta" / "libero_pipeline.json").read_text())
    if (
        source_manifest.get("collection_protocol") != "lpb_native_seeded_reset"
        or int(source_manifest.get("test_start_seed", -1)) != args.min_rollout_seed
    ):
        raise ValueError(f"Unexpected source collection protocol: {source_manifest}")

    dataset = LeRobotDataset(args.repo_id, root=args.root, return_uint8=True)
    sample = dataset[0]
    for key in ("action", "observation.state", *sorted(MAM_FEATURES)):
        if key not in sample:
            raise ValueError(f"Dataset sample is missing {key}.")
    print(
        json.dumps(
            {
                "root": str(args.root),
                "episodes": info["total_episodes"],
                "frames": info["total_frames"],
                "per_task": dict(sorted(counts.items())),
                "seeds_by_task": seeds_by_task,
                "train_init_state_overlap": 0,
                "features": sorted(info["features"]),
                "status": "ok",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
