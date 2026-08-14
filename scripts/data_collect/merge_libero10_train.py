#!/usr/bin/env python
"""Merge the official and DP-rollout LIBERO-10 MAM training splits."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.libero_pipeline import (
    LIBERO_ABSOLUTE_ACTION,
    LIBERO_CHUNK_RELATIVE_ACTION,
    LIBERO_CLOSED_LOOP_ABSOLUTE_MATERIALIZATION,
    LIBERO_PIPELINE_VERSION,
    LIBERO_STATE_14D,
    require_libero_v3_relative_ready_dataset,
    write_libero_pipeline_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--official-repo-id", required=True)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--rollout-repo-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-repo-id", required=True)
    parser.add_argument("--episodes-per-task", type=int, default=100)
    return parser.parse_args()


def _episode_rows(root: Path) -> list[dict]:
    paths = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata found under {root}.")
    return [row for path in paths for row in pq.read_table(path).to_pylist()]


def _validate_source(root: Path, expected_episodes: int, label: str) -> dict:
    manifest = require_libero_v3_relative_ready_dataset(root)
    if manifest.get("dataset_split") != "train":
        raise ValueError(f"{label} must be a train split, got {manifest.get('dataset_split')!r}.")
    rows = _episode_rows(root)
    counts = Counter(int(row["libero/task_id"]) for row in rows)
    if (
        len(rows) != expected_episodes
        or set(counts) != set(range(10))
        or any(count != expected_episodes // 10 for count in counts.values())
    ):
        raise ValueError(f"Unexpected {label} task distribution: episodes={len(rows)}, counts={counts}")
    return manifest


def main() -> None:
    args = parse_args()
    if args.episodes_per_task <= 0:
        raise ValueError("--episodes-per-task must be positive.")
    official_manifest = _validate_source(args.official_root, 500, "official source")
    rollout_manifest = _validate_source(args.rollout_root, 500, "rollout source")
    if args.output_root.exists():
        raise FileExistsError(f"Output already exists: {args.output_root}")

    merged = merge_datasets(
        [
            LeRobotDataset(args.official_repo_id, root=args.official_root),
            LeRobotDataset(args.rollout_repo_id, root=args.rollout_root),
        ],
        output_repo_id=args.output_repo_id,
        output_dir=args.output_root,
    )
    rows = _episode_rows(args.output_root)
    counts = Counter(int(row["libero/task_id"]) for row in rows)
    expected_total = 10 * args.episodes_per_task
    if (
        merged.meta.total_episodes != expected_total
        or len(rows) != expected_total
        or any(counts[task_id] != args.episodes_per_task for task_id in range(10))
    ):
        raise RuntimeError(f"Merged dataset has unexpected task distribution: {counts}")

    write_libero_pipeline_manifest(
        args.output_root,
        {
            "pipeline_version": LIBERO_PIPELINE_VERSION,
            "stage": "absolute_to_mam",
            "conversion_complete": True,
            "dataset_split": "train",
            "action_representation": LIBERO_ABSOLUTE_ACTION,
            "policy_action_representation": LIBERO_CHUNK_RELATIVE_ACTION,
            "relative_action_ready": True,
            "relative_action_stats": True,
            "relative_action_stats_n_obs_steps": 2,
            "relative_action_stats_horizon": 32,
            "relative_action_stats_action_delta_indices": list(range(-1, 31)),
            "observation_materialization": LIBERO_CLOSED_LOOP_ABSOLUTE_MATERIALIZATION,
            "state_representation": LIBERO_STATE_14D,
            "mask_types": ["random_mask"],
            "merged_from": [str(args.official_root.resolve()), str(args.rollout_root.resolve())],
            "merged_source_episode_counts": [500, 500],
            "official_source_manifest": official_manifest,
            "rollout_source_manifest": rollout_manifest,
        },
    )
    print(
        {
            "root": str(args.output_root),
            "episodes": merged.meta.total_episodes,
            "frames": merged.meta.total_frames,
            "per_task": dict(sorted(counts.items())),
        }
    )


if __name__ == "__main__":
    main()
