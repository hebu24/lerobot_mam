#!/usr/bin/env python3
"""Validate that Franka DP training data keeps the requested control rate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-hz", type=float, default=15.0)
    parser.add_argument("--tolerance-hz", type=float, default=0.25)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--prediction-horizon", type=int, default=16)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def decode_scalar(value: object) -> object:
    value = np.asarray(value).item()
    return value.decode("utf-8") if isinstance(value, bytes) else value


def trajectory_names(dataset: h5py.File) -> list[str]:
    return sorted(
        (key for key in dataset if key.startswith("traj_")),
        key=lambda key: int(key.split("_")[-1]),
    )


def timestamp_stats(
    source: h5py.File, key: str, expected_hz: float, tolerance_hz: float
) -> dict[str, float | list[float]]:
    all_dt: list[np.ndarray] = []
    trajectory_hz: list[float] = []
    for trajectory_name in trajectory_names(source):
        timestamps = np.asarray(source[trajectory_name][key], dtype=np.float64)
        dt = np.diff(timestamps)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size == 0:
            raise ValueError(f"{trajectory_name}/{key} has no positive timestamp intervals")
        all_dt.append(dt)
        trajectory_hz.append(float(1.0 / np.median(dt)))

    merged_dt = np.concatenate(all_dt)
    median_hz = float(1.0 / np.median(merged_dt))
    mean_hz = float(1.0 / np.mean(merged_dt))
    if abs(median_hz - expected_hz) > tolerance_hz:
        raise ValueError(f"{key} median rate is {median_hz:.6f} Hz, expected {expected_hz:.6f} Hz")
    outliers = [hz for hz in trajectory_hz if abs(hz - expected_hz) > tolerance_hz]
    if outliers:
        raise ValueError(f"{key} has {len(outliers)} trajectories outside the rate tolerance")
    return {
        "median_hz": median_hz,
        "mean_hz": mean_hz,
        "dt_ms_percentiles_1_50_99": [float(value) for value in np.percentile(merged_dt, [1, 50, 99]) * 1000],
        "trajectory_median_hz_min": min(trajectory_hz),
        "trajectory_median_hz_max": max(trajectory_hz),
    }


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.resolve()
    with h5py.File(dataset_path, "r") as dataset:
        if "meta/control_frequency_hz" not in dataset:
            raise ValueError(f"{dataset_path}: missing meta/control_frequency_hz")
        metadata_hz = float(dataset["meta/control_frequency_hz"][()])
        if abs(metadata_hz - args.expected_hz) > 1e-6:
            raise ValueError(
                f"{dataset_path}: control_frequency_hz={metadata_hz}, expected {args.expected_hz}"
            )
        source_path = Path(str(decode_scalar(dataset["meta/source_h5"][()])))
        processed_names = trajectory_names(dataset)

        with h5py.File(source_path, "r") as source:
            source_hz = float(source["meta/control_frequency_hz"][()])
            if abs(source_hz - args.expected_hz) > 1e-6:
                raise ValueError(
                    f"{source_path}: control_frequency_hz={source_hz}, expected {args.expected_hz}"
                )
            stats = {
                key: timestamp_stats(source, key, args.expected_hz, args.tolerance_hz)
                for key in ("obs_timestamps", "action_timestamps")
            }

            for processed_name in processed_names:
                processed = dataset[processed_name]
                source_episode_id = int(
                    processed["source_episode_id"][()]
                    if "source_episode_id" in processed
                    else processed_name.split("_")[-1]
                )
                source_trajectory = source[f"traj_{source_episode_id}"]
                for key in ("obs_timestamps", "action_timestamps"):
                    if key not in processed:
                        raise ValueError(f"{processed_name}: missing preserved {key}")
                    if not np.array_equal(processed[key][()], source_trajectory[key][()]):
                        raise ValueError(f"{processed_name}: {key} changed during preprocessing")

    result = {
        "dataset": str(dataset_path),
        "source_dataset": str(source_path),
        "control_frequency_hz": args.expected_hz,
        "control_period_seconds": 1.0 / args.expected_hz,
        "observation_horizon_steps": args.obs_horizon,
        "action_horizon_steps": args.action_horizon,
        "prediction_horizon_steps": args.prediction_horizon,
        "action_chunk_seconds": args.action_horizon / args.expected_hz,
        "prediction_chunk_seconds": args.prediction_horizon / args.expected_hz,
        "model_replan_frequency_hz": args.expected_hz / args.action_horizon,
        "timestamp_stats": stats,
    }
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if not args.quiet:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
