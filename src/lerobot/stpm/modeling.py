from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from lerobot.utils.import_utils import require_package


class FrozenCLIPEncoder(nn.Module):
    """Frozen CLIP encoder.

    An empty checkpoint keeps the previous lightweight deterministic fallback for
    tests. Any non-empty checkpoint/model id loads a real Hugging Face CLIP model.
    """

    def __init__(
        self,
        ckpt_path: str | Path | None = None,
        device: torch.device | str | None = None,
        emb_dim: int = 512,
    ):
        super().__init__()
        self.device = torch.device(device or "cpu")
        self.emb_dim = int(emb_dim)
        self.ckpt_path = str(ckpt_path or "")
        self.uses_clip = bool(self.ckpt_path)

        if self.uses_clip:
            require_package("transformers", extra="mam")
            from transformers import CLIPModel, CLIPProcessor

            resolved_ckpt = self._resolve_ckpt_path(self.ckpt_path)
            self.model = CLIPModel.from_pretrained(resolved_ckpt).to(self.device).eval()
            try:
                self.processor = CLIPProcessor.from_pretrained(resolved_ckpt, backend="pil")
            except TypeError:
                self.processor = CLIPProcessor.from_pretrained(resolved_ckpt, use_fast=False)
            image_processor = self.processor.image_processor
            image_size = image_processor.crop_size
            if hasattr(image_size, "height"):
                self.image_size = int(image_size.height)
            elif isinstance(image_size, dict):
                self.image_size = int(image_size.get("height") or image_size.get("shortest_edge") or 224)
            else:
                self.image_size = int(image_size)
            mean = torch.tensor(image_processor.image_mean, dtype=torch.float32, device=self.device).view(
                1, 3, 1, 1
            )
            std = torch.tensor(image_processor.image_std, dtype=torch.float32, device=self.device).view(
                1, 3, 1, 1
            )
            self.register_buffer("image_mean", mean, persistent=False)
            self.register_buffer("image_std", std, persistent=False)
            self.emb_dim = int(self.model.config.projection_dim)
            for param in self.model.parameters():
                param.requires_grad_(False)
            self.eval()
            return

        self.image_net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, self.emb_dim),
        ).to(self.device)
        self.requires_grad_(False)
        self.eval()

    @staticmethod
    def _resolve_ckpt_path(ckpt_path: str) -> str:
        path = Path(ckpt_path).expanduser()
        if path.exists():
            return str(path)

        repo_relative = Path.cwd() / path
        if repo_relative.exists():
            return str(repo_relative)

        return ckpt_path

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device, dtype=torch.float32)
        if images.ndim != 4:
            raise ValueError(f"images must be (B,3,H,W), got {tuple(images.shape)}")
        if images.shape[1] < 3:
            raise ValueError(f"Expected at least 3 image channels, got {tuple(images.shape)}")
        images = images[:, :3]
        if images.max() > 2.0:
            images = images / 255.0

        if self.uses_clip:
            if images.shape[-2:] != (self.image_size, self.image_size):
                images = F.interpolate(
                    images,
                    size=(self.image_size, self.image_size),
                    mode="bicubic",
                    align_corners=False,
                )
            pixel_values = (images - self.image_mean) / self.image_std
            vision_outputs = self.model.vision_model(pixel_values=pixel_values)
            return self.model.visual_projection(vision_outputs.pooler_output)

        return self.image_net(images)

    @torch.no_grad()
    def encode_text(self, tasks: list[str]) -> torch.Tensor:
        if self.uses_clip:
            inputs = self.processor.tokenizer(
                tasks,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            text_outputs = self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            return self.model.text_projection(text_outputs.pooler_output)

        vectors = []
        for task in tasks:
            digest = hashlib.sha256(str(task).encode("utf-8")).digest()
            raw = torch.tensor(list(digest), dtype=torch.float32, device=self.device)
            raw = raw.repeat((self.emb_dim + raw.numel() - 1) // raw.numel())[: self.emb_dim]
            vectors.append((raw / 127.5) - 1.0)
        return torch.stack(vectors, dim=0)


class RewardTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        vis_emb_dim: int = 512,
        text_emb_dim: int = 512,
        state_dim: int = 0,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        num_cameras: int = 1,
    ):
        super().__init__()
        self.num_cameras = int(num_cameras)
        self.state_dim = int(state_dim)
        self.vis_proj = nn.Linear(vis_emb_dim * self.num_cameras, d_model)
        self.text_proj = nn.Linear(text_emb_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model) if state_dim > 0 else None
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(
        self,
        image_emb: torch.Tensor,
        text_emb: torch.Tensor,
        state: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if image_emb.ndim != 4:
            raise ValueError(f"image_emb must be (B,N,T,D), got {tuple(image_emb.shape)}")
        b, n, t, d = image_emb.shape
        visual = image_emb.permute(0, 2, 1, 3).reshape(b, t, n * d)
        x = self.vis_proj(visual) + self.text_proj(text_emb).unsqueeze(1)
        if self.state_proj is not None:
            x = x + self.state_proj(state)
        encoded = self.encoder(x)
        return self.head(encoded).squeeze(-1)
