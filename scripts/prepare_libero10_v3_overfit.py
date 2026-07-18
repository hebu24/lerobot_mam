#!/usr/bin/env python

"""Select and certify the exact LIBERO-10 v3 trajectories used for DP overfit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.datasets.libero_pipeline import (
    LIBERO_CHUNK_RELATIVE_ACTION,
    require_libero_v3_relative_ready_dataset,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    try:
        value = row[key]
    except (KeyError, TypeError):
        return default
    return default if value is None else value


def _parse_task_ids(text: str | None, k: int) -> list[int]:
    task_ids = list(range(k)) if text is None else [int(item.strip()) for item in text.split(",")]
    if len(task_ids) != k or len(set(task_ids)) != k:
        raise ValueError(f"Expected exactly K={k} unique task ids, got {task_ids}.")
    if any(task_id < 0 or task_id > 9 for task_id in task_ids):
        raise ValueError(f"LIBERO-10 task ids must be in [0, 9], got {task_ids}.")
    return task_ids


def select_overfit_episodes(
    rows: Iterable[Any],
    *,
    task_ids: list[int],
    demo_rank: int,
) -> list[dict[str, Any]]:
    if demo_rank < 0:
        raise ValueError(f"demo_rank must be non-negative, got {demo_rank}.")

    by_task_source: dict[int, dict[int, list[Any]]] = {task_id: {} for task_id in task_ids}
    for row in rows:
        task_id = int(_row_value(row, "libero/task_id", -1))
        if task_id not in by_task_source:
            continue
        source_id = _row_value(row, "libero/source_episode_id")
        if source_id is None:
            source_id = _row_value(row, "source_episode_id")
        if source_id is None:
            raise ValueError(f"Task {task_id} has an episode without source trajectory identity.")
        by_task_source[task_id].setdefault(int(source_id), []).append(row)

    selections: list[dict[str, Any]] = []
    for task_id in task_ids:
        source_groups = by_task_source[task_id]
        source_ids = sorted(source_groups)
        if demo_rank >= len(source_ids):
            raise ValueError(
                f"Task {task_id} has {len(source_ids)} unique source trajectories; "
                f"demo_rank={demo_rank} is unavailable."
            )
        source_id = source_ids[demo_rank]
        candidates = sorted(
            source_groups[source_id],
            key=lambda row: (
                int(_row_value(row, "mask_type_slot", 0)),
                int(row["episode_index"]),
            ),
        )
        row = candidates[0]
        init_state = _row_value(row, "libero/init_state")
        source_file = _row_value(row, "libero/source_file")
        source_demo = _row_value(row, "libero/source_demo")
        suite = str(_row_value(row, "libero/suite", ""))
        if suite != "libero_10" or init_state is None or source_file is None or source_demo is None:
            raise ValueError(
                f"Episode {int(row['episode_index'])} lacks exact v3 trajectory metadata "
                "(suite/raw init_state/source_file/source_demo)."
            )
        state = np.asarray(init_state, dtype="<f8").reshape(-1)
        if state.size == 0 or not np.isfinite(state).all():
            raise ValueError(f"Episode {int(row['episode_index'])} has an invalid raw init_state.")
        selections.append(
            {
                "task_id": task_id,
                "episode_index": int(row["episode_index"]),
                "source_episode_id": source_id,
                "source_file": str(source_file),
                "source_demo": str(source_demo),
                "init_state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                "mask_type_slot": int(_row_value(row, "mask_type_slot", 0)),
            }
        )
    return selections


def _validate_schema(meta: LeRobotDatasetMetadata) -> None:
    features = meta.features
    if ACTION not in features or tuple(features[ACTION]["shape"]) != (7,):
        raise ValueError("LIBERO v3 overfit requires a 7D action feature.")
    if OBS_STATE not in features or tuple(features[OBS_STATE]["shape"]) != (14,):
        raise ValueError("LIBERO v3 overfit requires a 14D observation.state feature.")
    camera_keys = list(meta.camera_keys)
    if len(camera_keys) != 2:
        raise ValueError(f"LIBERO v3 overfit requires two cameras, got {camera_keys}.")
    for key in camera_keys:
        if tuple(features[key]["shape"]) not in {(128, 128, 3), (3, 128, 128)}:
            raise ValueError(f"Camera {key} is not 128x128 RGB: {features[key]['shape']}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--task-ids", default=None, help="Optional comma-separated task ids; length must equal K."
    )
    parser.add_argument("--demo-rank", type=int, default=0)
    parser.add_argument("--output-plan", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k < 1 or args.k > 10:
        raise ValueError(f"K must be in [1, 10], got {args.k}.")
    manifest = require_libero_v3_relative_ready_dataset(args.dataset_root)
    if manifest.get("stage") != "absolute_to_mam" or manifest.get("dataset_split") != "train":
        raise ValueError("Overfit input must be the train split produced by absolute_to_mam.")
    if manifest.get("policy_action_representation") != LIBERO_CHUNK_RELATIVE_ACTION:
        raise ValueError("Dataset manifest does not declare chunk-relative SE(3) policy actions.")

    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    _validate_schema(metadata)
    task_ids = _parse_task_ids(args.task_ids, args.k)
    selections = select_overfit_episodes(
        metadata.episodes,
        task_ids=task_ids,
        demo_rank=args.demo_rank,
    )
    episode_ids = [item["episode_index"] for item in selections]
    plan = {
        "version": 2,
        "pipeline_version": manifest["pipeline_version"],
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_repo_id": args.dataset_repo_id,
        "k": args.k,
        "demo_rank": args.demo_rank,
        "task_ids": task_ids,
        "train_episode_ids": episode_ids,
        "eval_episode_ids": episode_ids,
        "same_train_eval_trajectories": True,
        "policy_action_representation": LIBERO_CHUNK_RELATIVE_ACTION,
        "observation_materialization": manifest["observation_materialization"],
        "relative_action_ready": True,
        "environment_control_mode": "absolute",
        "selections": selections,
    }
    if args.output_plan is not None:
        args.output_plan.parent.mkdir(parents=True, exist_ok=True)
        args.output_plan.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    # Stable two-line machine interface consumed by the bash launcher.
    print(json.dumps(task_ids, separators=(",", ":")))
    print(json.dumps(episode_ids, separators=(",", ":")))


if __name__ == "__main__":
    main()
