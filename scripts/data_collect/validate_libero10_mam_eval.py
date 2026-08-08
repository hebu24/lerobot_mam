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
    parser.add_argument("--root", type=Path, default=Path("outputs/datasets/libero10_100_eval"))
    parser.add_argument("--repo-id", default="local/libero10_100_eval")
    parser.add_argument("--episodes-per-task", type=int, default=10)
    return parser.parse_args()


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

    rows = []
    for path in sorted((args.root / "meta" / "episodes").glob("**/*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    counts = Counter(int(row["libero/task_id"]) for row in rows)
    expected_counts = dict.fromkeys(range(10), args.episodes_per_task)
    if dict(sorted(counts.items())) != expected_counts:
        raise ValueError(f"Unexpected per-task counts: {dict(sorted(counts.items()))}")
    if any(not row.get("libero/init_state") for row in rows):
        raise ValueError("Every fixed evaluation episode must contain a raw init state.")
    unique_states = {
        (int(row["libero/task_id"]), np.asarray(row["libero/init_state"]).tobytes()) for row in rows
    }
    if len(unique_states) != expected_total:
        raise ValueError("Fixed evaluation init states are not unique within tasks.")

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
                "features": sorted(info["features"]),
                "status": "ok",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
