from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from .modeling import FrozenCLIPEncoder, RewardTransformer
from .normalizer import load_state_normalizer


def _shape_to_hw(shape: list[int] | tuple[int, ...]) -> tuple[int, int]:
    if len(shape) != 3:
        raise ValueError(f"Expected image shape with 3 dims, got {shape}")
    if int(shape[0]) in {1, 3, 4}:
        return int(shape[1]), int(shape[2])
    return int(shape[0]), int(shape[1])


class STPMEncoder(nn.Module):
    def __init__(
        self,
        ckpt_path: str | Path,
        config_path: str | Path,
        device: str | torch.device | None = None,
    ):
        super().__init__()
        self.config_path = Path(config_path)
        with open(self.config_path, encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.device = torch.device(device or self.cfg.get("device", "cpu"))
        self.camera_names = list(self.cfg["camera_names"])
        self.image_shape = self.cfg.get("image_shape")
        self.state_dim = int(self.cfg["state_dim"])
        self.n_obs_steps = int(self.cfg["n_obs_steps"])
        self.frame_gap = int(self.cfg["frame_gap"])
        self.task_description = str(self.cfg.get("task_description", ""))
        self.clip_encoder = FrozenCLIPEncoder(self.cfg.get("vision_ckpt", ""), self.device)
        self.reward_model = RewardTransformer(
            d_model=int(self.cfg.get("d_model", 256)),
            state_dim=self.state_dim,
            n_layers=int(self.cfg.get("n_layers", 2)),
            n_heads=int(self.cfg.get("n_heads", 4)),
            dropout=float(self.cfg.get("dropout", 0.1)),
            num_cameras=len(self.camera_names),
        ).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.reward_model.load_state_dict(ckpt["model"])
        self.state_normalizer = load_state_normalizer(
            self.cfg["state_norm_path"],
            self.device,
            state_dim=self.state_dim,
        )
        self.eval()
        self.requires_grad_(False)

    @torch.no_grad()
    def predict_progress(
        self,
        rgbd: torch.Tensor,
        state: torch.Tensor,
        tasks: list[str] | str | None = None,
    ) -> torch.Tensor:
        if rgbd.ndim == 5:
            rgbd = rgbd.unsqueeze(2)
        if rgbd.ndim != 6:
            raise ValueError(f"rgbd must be (B,T,N,3,H,W), got {tuple(rgbd.shape)}")
        if state.ndim != 3 or state.shape[-1] != self.state_dim:
            raise ValueError(f"state must be (B,T,{self.state_dim}), got {tuple(state.shape)}")
        b, t, n = rgbd.shape[:3]
        if n != len(self.camera_names):
            raise ValueError(f"Expected {len(self.camera_names)} cameras, got {n}")
        if self.image_shape is not None:
            expected_hw = [_shape_to_hw(shape) for shape in self.image_shape]
            actual_hw = (int(rgbd.shape[-2]), int(rgbd.shape[-1]))
            for camera_name, camera_hw in zip(self.camera_names, expected_hw, strict=True):
                if camera_hw != actual_hw:
                    raise ValueError(
                        f"Camera '{camera_name}' expected image HW={camera_hw}, got {actual_hw}"
                    )
        if tasks is None:
            task_list = [self.task_description] * b
        elif isinstance(tasks, str):
            task_list = [tasks] * b
        else:
            task_list = list(tasks)
        if self.task_description and any(str(task) != self.task_description for task in task_list):
            raise ValueError(
                "STPM task description mismatch: "
                f"expected {self.task_description!r}, got {task_list}"
            )
        rgb = rgbd[:, :, :, :3].to(self.device)
        flat = rgb.permute(2, 0, 1, 3, 4, 5).reshape(n * b * t, 3, rgb.shape[-2], rgb.shape[-1])
        img_emb = self.clip_encoder.encode_image(flat)
        img_emb = img_emb.view(n, b, t, -1).permute(1, 0, 2, 3)
        text_emb = self.clip_encoder.encode_text(task_list)
        norm_state = self.state_normalizer.normalize(state.to(self.device))
        lengths = torch.full((b,), t, dtype=torch.long, device=self.device)
        progress = self.reward_model(img_emb, text_emb, norm_state, lengths)
        return progress[:, -1]
