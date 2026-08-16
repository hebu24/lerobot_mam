#!/usr/bin/env python
"""Compare two task-scoped STPM families on identical held-out anchor frames."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.stpm import FrameLeRobotDataset, STPMEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_prefix", type=Path, required=True)
    parser.add_argument("--candidate_prefix", type=Path, required=True)
    parser.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--reference_checkpoint_name", default="reward_best.pt")
    parser.add_argument("--candidate_checkpoint_name", default="reward_best_endpoint.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--rollout_stride", type=int, default=15)
    parser.add_argument("--mam_eval_dataset_repo_id", default="local/libero10_mam_v3_refmix_eval")
    parser.add_argument(
        "--mam_eval_dataset_root",
        type=Path,
        default=Path("data/libero10_mam/libero10_mam_v3_refmix_eval"),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def _load_config(root: Path) -> dict[str, Any]:
    config_path = root / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing STPM config: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _source_groups(config: dict[str, Any]) -> set[tuple[str, str]]:
    identity = config.get("split_identity", {})
    return {
        (str(row["task"]), str(row["source_episode_id"]))
        for split in ("train_groups", "val_groups")
        for row in identity.get(split, [])
    }


def _validate_task_pair(
    task_id: int,
    reference_root: Path,
    candidate_root: Path,
    reference_config: dict[str, Any],
    candidate_config: dict[str, Any],
    reference_checkpoint_name: str,
    candidate_checkpoint_name: str,
) -> None:
    expected_tasks = [str(task_id)]
    for label, root, config, checkpoint_name in (
        ("reference", reference_root, reference_config, reference_checkpoint_name),
        ("candidate", candidate_root, candidate_config, candidate_checkpoint_name),
    ):
        checkpoint = root / "checkpoints" / checkpoint_name
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing {label} checkpoint: {checkpoint}")
        if not (root / "state_norm.json").is_file():
            raise FileNotFoundError(f"Missing {label} state normalization: {root / 'state_norm.json'}")
        identity = config.get("split_identity", {})
        if identity.get("tasks") != expected_tasks:
            raise ValueError(f"{label} task identity mismatch for task {task_id}: {identity.get('tasks')!r}")
        train_groups = {
            (str(row["task"]), str(row["source_episode_id"])) for row in identity.get("train_groups", [])
        }
        val_groups = {
            (str(row["task"]), str(row["source_episode_id"])) for row in identity.get("val_groups", [])
        }
        if not train_groups or not val_groups:
            raise ValueError(f"{label} task {task_id} has an empty train or validation split.")
        overlap = train_groups & val_groups
        if overlap:
            raise ValueError(f"{label} task {task_id} train/val source overlap: {sorted(overlap)}")

    if reference_config["repo_id"] != candidate_config["repo_id"]:
        raise ValueError(f"Task {task_id} dataset repo_id differs between A/B.")
    if Path(reference_config["root"]) != Path(candidate_config["root"]):
        raise ValueError(f"Task {task_id} dataset root differs between A/B.")
    for field in ("train_groups", "val_groups"):
        reference_groups = reference_config["split_identity"][field]
        candidate_groups = candidate_config["split_identity"][field]
        if reference_groups != candidate_groups:
            raise ValueError(f"Task {task_id} {field} differs between A/B.")
    if reference_config["val_episode_ids"] != candidate_config["val_episode_ids"]:
        raise ValueError(f"Task {task_id} local validation episode ids differ between A/B.")


def _assert_eval_source_isolation(
    configs: list[dict[str, Any]],
    repo_id: str,
    root: Path,
) -> dict[str, Any]:
    stpm_groups = set().union(*(_source_groups(config) for config in configs))
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    eval_groups = {
        (str(row["libero/task_id"]), str(row["libero/source_episode_id"])) for row in metadata.episodes
    }
    overlap = stpm_groups & eval_groups
    if overlap:
        raise ValueError(
            f"STPM train/validation sources overlap MAM evaluation sources: {sorted(overlap)[:20]}"
        )
    return {
        "stpm_source_groups": len(stpm_groups),
        "mam_eval_source_groups": len(eval_groups),
        "overlap": 0,
        "mam_eval_repo_id": repo_id,
        "mam_eval_root": str(root),
    }


def _pearson(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.size < 2 or np.std(prediction) == 0 or np.std(target) == 0:
        return float("nan")
    return float(np.corrcoef(prediction, target)[0, 1])


def _metrics(
    records: list[dict[str, float | int]],
    *,
    rollout_stride: int,
) -> dict[str, Any]:
    prediction = np.asarray([row["prediction"] for row in records], dtype=np.float64)
    target = np.asarray([row["target"] for row in records], dtype=np.float64)
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(target)):
        raise ValueError("STPM audit produced NaN or Inf.")
    error = prediction - target
    episodes: dict[int, list[dict[str, float | int]]] = defaultdict(list)
    for row in records:
        episodes[int(row["episode_index"])].append(row)

    dense_differences: list[float] = []
    rollout_differences: list[float] = []
    starts: list[float] = []
    ends: list[float] = []
    episode_rmse: list[float] = []
    for episode_records in episodes.values():
        ordered = sorted(episode_records, key=lambda row: int(row["anchor_index"]))
        values = np.asarray([row["prediction"] for row in ordered], dtype=np.float64)
        episode_targets = np.asarray([row["target"] for row in ordered], dtype=np.float64)
        dense_differences.extend(np.diff(values).tolist())
        rollout_differences.extend(np.diff(values[::rollout_stride]).tolist())
        starts.append(float(values[0]))
        ends.append(float(values[-1]))
        episode_rmse.append(float(np.sqrt(np.mean(np.square(values - episode_targets)))))

    dense = np.asarray(dense_differences, dtype=np.float64)
    rollout = np.asarray(rollout_differences, dtype=np.float64)
    return {
        "n_frames": int(prediction.size),
        "n_episodes": len(episodes),
        "mse": float(np.mean(np.square(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "pearson_correlation": _pearson(prediction, target),
        "macro_episode_rmse": float(np.mean(episode_rmse)),
        "backward_rate_dense": float(np.mean(dense < 0)) if dense.size else float("nan"),
        "backward_rate_rollout_stride": (float(np.mean(rollout < 0)) if rollout.size else float("nan")),
        "rollout_stride": rollout_stride,
        "mean_start_prediction": float(np.mean(starts)),
        "mean_end_prediction": float(np.mean(ends)),
    }


@torch.no_grad()
def _predict(
    root: Path,
    config: dict[str, Any],
    checkpoint_name: str,
    val_episode_ids: list[int],
    *,
    device: str,
    batch_size: int,
    num_workers: int,
    description: str,
) -> tuple[list[dict[str, float | int]], dict[str, Any]]:
    dataset = FrameLeRobotDataset(
        repo_id=config["repo_id"],
        root=config["root"],
        episodes=val_episode_ids,
        n_obs_steps=int(config["n_obs_steps"]),
        frame_gap=int(config["frame_gap"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=str(device).startswith("cuda"),
        persistent_workers=num_workers > 0,
    )
    checkpoint_path = root / "checkpoints" / checkpoint_name
    encoder = STPMEncoder(checkpoint_path, root / "config.yaml", device=device)
    parameter_count = sum(parameter.numel() for parameter in encoder.reward_model.parameters())
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    records: list[dict[str, float | int]] = []
    for batch in tqdm(loader, desc=description, leave=False):
        prediction = encoder.predict_progress(
            batch["image_frames"],
            batch["state"],
            list(batch["task"]),
        )
        target = batch["targets"][:, -1]
        for pred, truth, episode_index, anchor_index in zip(
            prediction.cpu().tolist(),
            target.cpu().tolist(),
            batch["episode_index"].tolist(),
            batch["anchor_index"].tolist(),
            strict=True,
        ):
            records.append(
                {
                    "episode_index": int(episode_index),
                    "anchor_index": int(anchor_index),
                    "prediction": float(pred),
                    "target": float(truth),
                }
            )
    metadata = {
        "root": str(root),
        "config_path": str(root / "config.yaml"),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": checkpoint_payload.get("step"),
        "checkpoint_selection_metric": checkpoint_payload.get("selection_metric", "legacy"),
        "parameter_count": parameter_count,
        "n_obs_steps": int(config["n_obs_steps"]),
        "frame_gap": int(config["frame_gap"]),
    }
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records, metadata


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    if args.batch_size <= 0 or args.rollout_stride <= 0:
        raise ValueError("--batch_size and --rollout_stride must be positive.")
    task_ids = [int(value) for value in args.tasks.split(",") if value.strip()]
    if sorted(set(task_ids)) != sorted(task_ids) or any(task < 0 or task > 9 for task in task_ids):
        raise ValueError(f"--tasks must be unique LIBERO-10 task ids, got {task_ids}.")

    pairs: dict[int, tuple[Path, Path, dict[str, Any], dict[str, Any]]] = {}
    all_configs: list[dict[str, Any]] = []
    for task_id in task_ids:
        reference_root = Path(f"{args.reference_prefix}{task_id}")
        candidate_root = Path(f"{args.candidate_prefix}{task_id}")
        reference_config = _load_config(reference_root)
        candidate_config = _load_config(candidate_root)
        _validate_task_pair(
            task_id,
            reference_root,
            candidate_root,
            reference_config,
            candidate_config,
            args.reference_checkpoint_name,
            args.candidate_checkpoint_name,
        )
        pairs[task_id] = (
            reference_root,
            candidate_root,
            reference_config,
            candidate_config,
        )
        all_configs.extend((reference_config, candidate_config))

    isolation = _assert_eval_source_isolation(
        all_configs,
        args.mam_eval_dataset_repo_id,
        args.mam_eval_dataset_root,
    )
    args.output_dir.mkdir(parents=True)
    results: dict[str, Any] = {
        "reference_prefix": str(args.reference_prefix),
        "candidate_prefix": str(args.candidate_prefix),
        "source_isolation": isolation,
        "tasks": {},
    }
    pooled: dict[str, list[dict[str, float | int]]] = {"reference": [], "candidate": []}
    for task_id, (reference_root, candidate_root, reference_config, candidate_config) in pairs.items():
        val_episode_ids = [int(value) for value in reference_config["val_episode_ids"]]
        task_result: dict[str, Any] = {}
        key_sets = {}
        for label, root, config, checkpoint_name in (
            ("reference", reference_root, reference_config, args.reference_checkpoint_name),
            ("candidate", candidate_root, candidate_config, args.candidate_checkpoint_name),
        ):
            records, metadata = _predict(
                root,
                config,
                checkpoint_name,
                val_episode_ids,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                description=f"task {task_id} {label}",
            )
            keys = {(int(row["episode_index"]), int(row["anchor_index"])) for row in records}
            key_sets[label] = keys
            pooled[label].extend(records)
            task_result[label] = {
                **metadata,
                "metrics": _metrics(records, rollout_stride=args.rollout_stride),
            }
        if key_sets["reference"] != key_sets["candidate"]:
            raise ValueError(f"Task {task_id} A/B anchor frames do not match.")
        reference_metrics = task_result["reference"]["metrics"]
        candidate_metrics = task_result["candidate"]["metrics"]
        task_result["delta_candidate_minus_reference"] = {
            metric: candidate_metrics[metric] - reference_metrics[metric]
            for metric in ("rmse", "mae", "bias", "backward_rate_dense", "backward_rate_rollout_stride")
        }
        results["tasks"][str(task_id)] = task_result
        print(
            f"task={task_id} reference_rmse={reference_metrics['rmse']:.6f} "
            f"candidate_rmse={candidate_metrics['rmse']:.6f} "
            f"delta={candidate_metrics['rmse'] - reference_metrics['rmse']:+.6f}"
        )

    results["overall"] = {
        label: {"metrics": _metrics(records, rollout_stride=args.rollout_stride)}
        for label, records in pooled.items()
    }
    reference_overall = results["overall"]["reference"]["metrics"]
    candidate_overall = results["overall"]["candidate"]["metrics"]
    results["overall"]["delta_candidate_minus_reference"] = {
        metric: candidate_overall[metric] - reference_overall[metric]
        for metric in ("rmse", "mae", "bias", "backward_rate_dense", "backward_rate_rollout_stride")
    }
    safe_results = _json_safe(results)
    (args.output_dir / "summary.json").write_text(
        json.dumps(safe_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"overall reference_rmse={reference_overall['rmse']:.6f} "
        f"candidate_rmse={candidate_overall['rmse']:.6f} "
        f"output={args.output_dir / 'summary.json'}"
    )


if __name__ == "__main__":
    main()
