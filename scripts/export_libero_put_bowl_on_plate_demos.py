#!/usr/bin/env python

"""Export visual demo videos for the LIBERO task "put the bowl on the plate"."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import torch

from extract_libero_put_bowl_on_plate import (
    OUTPUT_REPO_ID,
    OUTPUT_ROOT,
)


OUTPUT_DIR = Path("outputs/demos/libero_put_bowl_on_plate")


def tensor_image_to_hwc_uint8(image: torch.Tensor) -> np.ndarray:
    """Convert a LeRobot image tensor to HWC uint8 RGB."""
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image tensor, got shape={tuple(image.shape)}")
    if image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    array = image.detach().cpu().numpy()
    if array.dtype != np.uint8:
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return array


def stack_camera_frames(frames: list[np.ndarray]) -> np.ndarray:
    if len(frames) == 1:
        return frames[0]
    min_height = min(frame.shape[0] for frame in frames)
    cropped = [frame[:min_height] for frame in frames]
    return np.concatenate(cropped, axis=1)


def select_episode_indices(args: argparse.Namespace) -> list[int]:
    from lerobot.datasets import LeRobotDatasetMetadata

    if not args.dataset_root.exists():
        raise FileNotFoundError(
            f"Filtered dataset not found: {args.dataset_root}. "
            "Run scripts/extract_libero_put_bowl_on_plate.py first."
        )

    logging.info("Loading metadata from filtered dataset: %s", args.dataset_root)
    meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    return [int(ep["episode_index"]) for ep in meta.episodes]


def export_episode_video(dataset, episode_index: int, output_path: Path, frame_stride: int) -> None:
    from lerobot.utils.io_utils import write_video

    camera_keys = dataset.meta.camera_keys
    if not camera_keys:
        raise ValueError("Dataset has no camera keys to export.")

    frames: list[np.ndarray] = []
    episode_frame_idx = 0
    for idx in range(len(dataset)):
        item = dataset[idx]
        item_episode = int(item["episode_index"].item())
        if item_episode != episode_index:
            continue
        if episode_frame_idx % frame_stride != 0:
            episode_frame_idx += 1
            continue
        camera_frames = [tensor_image_to_hwc_uint8(item[key]) for key in camera_keys]
        frames.append(stack_camera_frames(camera_frames))
        episode_frame_idx += 1
    if not frames:
        raise ValueError(f"No frames found for episode {episode_index}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = max(1, round(dataset.fps / frame_stride))
    write_video(output_path, frames, fps=fps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export 3-5 visual demos for the LIBERO task 'put the bowl on the plate'."
    )
    parser.add_argument("--dataset-repo-id", default=OUTPUT_REPO_ID)
    parser.add_argument("--dataset-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--num-demos", type=int, default=5)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--force-cache-sync", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.num_demos < 1 or args.num_demos > 5:
        raise ValueError("--num-demos should be between 1 and 5.")
    if args.frame_stride < 1:
        raise ValueError("--frame-stride should be >= 1.")
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)

    from lerobot.datasets import LeRobotDataset

    episode_indices = select_episode_indices(args)[: args.num_demos]
    logging.info("Exporting episodes: %s", episode_indices)

    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        episodes=episode_indices,
        revision=args.revision,
        force_cache_sync=args.force_cache_sync,
        return_uint8=True,
    )

    for demo_idx, episode_index in enumerate(episode_indices):
        output_path = args.output_dir / f"demo_{demo_idx:02d}_episode_{episode_index:06d}.mp4"
        if output_path.exists() and not args.overwrite:
            logging.info("Skip existing demo: %s", output_path)
            continue
        logging.info("Writing %s", output_path)
        export_episode_video(dataset, episode_index, output_path, args.frame_stride)

    logging.info("Done. Demo videos are in %s", args.output_dir)


if __name__ == "__main__":
    main()
