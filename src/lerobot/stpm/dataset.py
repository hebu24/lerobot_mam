from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.constants import OBS_STATE


class FrameLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        n_obs_steps: int = 1,
        frame_gap: int = 1,
        image_names: list[str] | None = None,
        task_description: str | None = None,
    ):
        self.meta = LeRobotDatasetMetadata(repo_id, root=root)
        self.n_obs_steps = int(n_obs_steps)
        self.frame_gap = int(frame_gap)
        self.sequence_length = self.n_obs_steps + 1
        self.relative_indices = list(range(-self.n_obs_steps * self.frame_gap, 1, self.frame_gap))
        self.camera_keys = image_names or list(self.meta.camera_keys)
        self.task_description = task_description
        delta_timestamps = {
            OBS_STATE: [i / self.meta.fps for i in self.relative_indices],
            **{key: [i / self.meta.fps for i in self.relative_indices] for key in self.camera_keys},
        }
        self.dataset = LeRobotDataset(
            repo_id,
            root=root,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            return_uint8=True,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _as_chw_sequence(images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(
                f"Expected image sequence shape (T,C,H,W) or (T,H,W,C), got {tuple(images.shape)}"
            )
        if images.shape[1] in (3, 4):
            return images[:, :3]
        if images.shape[-1] in (3, 4):
            return images[..., :3].permute(0, 3, 1, 2)
        raise ValueError(f"Cannot infer channel dimension from image shape {tuple(images.shape)}")

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        state = item[OBS_STATE].float()
        image_frames = torch.stack([self._as_chw_sequence(item[key]) for key in self.camera_keys], dim=1)
        ep_idx = int(item["episode_index"])
        frame_idx = int(item["frame_index"])
        ep = self.dataset.meta.episodes[ep_idx]
        ep_len = int(ep["length"])
        sampled = torch.tensor(
            [min(max(frame_idx + rel, 0), ep_len - 1) for rel in self.relative_indices],
            dtype=torch.float32,
        )
        targets = sampled / float(max(ep_len - 1, 1))
        task = self.task_description if self.task_description is not None else item["task"]
        return {
            "image_frames": image_frames,
            "state": state,
            "targets": targets,
            "lengths": torch.tensor(self.sequence_length, dtype=torch.long),
            "task": task,
            "episode_index": ep_idx,
            "anchor_index": frame_idx,
        }
