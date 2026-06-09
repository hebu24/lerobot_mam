#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.compute_stats import compute_libero_relative_action_stats
from lerobot.datasets.io_utils import write_stats
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
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--mask-type", type=str, default="random_mask")
    parser.add_argument("--retain-ratio", type=float, default=0.2)
    parser.add_argument("--mask-value", type=float, default=0.0)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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
        for key in ("source_episode_id", "mask_type", "mask_type_slot", "libero/init_state_id"):
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

    features = {
        key: value
        for key, value in source.meta.features.items()
        if key not in DEFAULT_FEATURES
    }
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
    source_by_episode = {}
    for idx in range(len(source)):
        item = source[idx]
        source_by_episode.setdefault(int(item["episode_index"]), []).append(item)

    for local_episode_index, source_episode_id in enumerate(episode_ids):
        frames = source_by_episode[int(source_episode_id)]
        actions = np.stack([np.asarray(frame[ACTION], dtype=np.float32) for frame in frames], axis=0)
        rng = np.random.default_rng(args.split_seed + int(source_episode_id))
        masked_actions, mask = _apply_mask(actions, args.mask_type, args.retain_ratio, args.mask_value, rng)
        denom = max(len(frames) - 1, 1)
        for frame_index, item in enumerate(frames):
            out = {"task": item["task"]}
            for key, ft in features.items():
                if key in {MAM_MAS_ACTION_ABSOLUTE, MAM_MAS_ACTION_MASK, MAM_PROGRESS}:
                    continue
                out[key] = _to_frame_value(item[key], ft)
            out[MAM_MAS_ACTION_ABSOLUTE] = masked_actions[frame_index]
            out[MAM_MAS_ACTION_MASK] = mask[frame_index]
            out[MAM_PROGRESS] = np.asarray([frame_index / denom], dtype=np.float32)
            dataset.add_frame(out)
        dataset.save_episode()

        source_row = source_episode_rows.get(int(source_episode_id), {})
        init_state_id = source_row.get(
            "libero/init_state_id",
            source_row.get("init_state_id", source_episode_id),
        )
        episode_meta_rows[local_episode_index] = {
            "source_episode_id": int(source_episode_id),
            "mask_type": args.mask_type,
            "mask_type_slot": 0,
            "libero/init_state_id": int(init_state_id),
        }

    dataset.finalize()
    _patch_episode_metadata(root, episode_meta_rows)

    reopened = LeRobotDataset(repo_id=repo_id, root=root, return_uint8=True)
    stats = dict(reopened.meta.stats or {})
    action_delta_indices = list(range(1 - args.n_obs_steps, 1 - args.n_obs_steps + args.horizon))
    stats[ACTION] = compute_libero_relative_action_stats(
        hf_dataset=reopened.hf_dataset,
        action_delta_indices=action_delta_indices,
        num_workers=0,
    )
    write_stats(stats, root)


def main() -> None:
    args = parse_args()
    input_repo_id = args.input_repo_id or _repo_id_from_root(args.input_root)
    source = LeRobotDataset(input_repo_id, root=args.input_root, return_uint8=True)
    train_ids, eval_ids = _selected_episode_ids(source.meta.total_episodes, args.eval_ratio, args.split_seed)

    train_root = args.output_root.with_name(f"{args.output_root.name}_train")
    eval_root = args.output_root.with_name(f"{args.output_root.name}_eval")
    train_repo = f"{args.output_repo_id}_train"
    eval_repo = f"{args.output_repo_id}_eval"
    _write_split(source, train_ids, train_root, train_repo, args)
    if eval_ids:
        _write_split(source, eval_ids, eval_root, eval_repo, args)
    print(f"Wrote MAM datasets: train={train_root}, eval={eval_root if eval_ids else 'N/A'}")


if __name__ == "__main__":
    main()
