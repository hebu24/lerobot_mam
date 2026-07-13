from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as functional
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
                images = functional.interpolate(
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
    """Causal progress model over independent camera, language, and state tokens.

    Tokens are ordered by time, then modality. The causal mask is defined on the
    time index instead of the flattened token index, so all modalities at the
    current timestep can interact without observing any future timestep.
    """

    ARCHITECTURE_VERSION = 2
    _VERSION_KEY = "_architecture_version"

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
        if num_cameras <= 0:
            raise ValueError(f"num_cameras must be positive, got {num_cameras}.")
        if state_dim < 0:
            raise ValueError(f"state_dim must be non-negative, got {state_dim}.")
        self.num_cameras = int(num_cameras)
        self.state_dim = int(state_dim)
        self.d_model = int(d_model)
        self.num_modalities = self.num_cameras + 1 + int(self.state_dim > 0)
        self.visual_proj = nn.Linear(vis_emb_dim, d_model)
        self.lang_proj = nn.Linear(text_emb_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model) if state_dim > 0 else None
        self.modality_pos = nn.Parameter(torch.empty(1, 1, self.num_modalities, d_model))
        nn.init.normal_(self.modality_pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.fusion_net = nn.Sequential(
            nn.LayerNorm(d_model * self.num_modalities),
            nn.Linear(d_model * self.num_modalities, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.register_buffer(
            self._VERSION_KEY,
            torch.tensor(self.ARCHITECTURE_VERSION, dtype=torch.int64),
        )

    @staticmethod
    def _time_encoding(
        length: int,
        d_model: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a sinusoidal time encoding with no fixed sequence limit."""
        position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        even_dims = torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        frequencies = torch.exp(even_dims * (-math.log(10_000.0) / d_model))
        angles = position * frequencies
        encoding = torch.zeros(length, d_model, device=device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(angles)
        if d_model > 1:
            encoding[:, 1::2] = torch.cos(angles[:, : d_model // 2])
        return encoding.to(dtype=dtype).view(1, length, 1, d_model)

    @classmethod
    def validate_checkpoint_state_dict(cls, state_dict: Mapping[str, torch.Tensor]) -> None:
        version = state_dict.get(cls._VERSION_KEY)
        if version is None:
            legacy_keys = {"vis_proj.weight", "text_proj.weight", "encoder.layers.0.self_attn.in_proj_weight"}
            if legacy_keys.intersection(state_dict):
                raise RuntimeError(
                    "Incompatible legacy STPM checkpoint (architecture v1): the old model fused cameras, "
                    "text, and state before temporal attention. Retrain STPM with architecture v2."
                )
            raise RuntimeError(
                "STPM checkpoint has no architecture version and cannot be loaded safely. "
                "Retrain STPM with architecture v2."
            )
        loaded_version = int(torch.as_tensor(version).item())
        if loaded_version != cls.ARCHITECTURE_VERSION:
            raise RuntimeError(
                f"Incompatible STPM checkpoint architecture v{loaded_version}; "
                f"this code requires v{cls.ARCHITECTURE_VERSION}."
            )

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        self.validate_checkpoint_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(
        self,
        image_emb: torch.Tensor,
        text_emb: torch.Tensor,
        state: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if image_emb.ndim != 4:
            raise ValueError(f"image_emb must be (B,N,T,D), got {tuple(image_emb.shape)}")
        b, n, t, _ = image_emb.shape
        if n != self.num_cameras:
            raise ValueError(f"Expected {self.num_cameras} cameras, got {n}.")
        if text_emb.ndim != 2 or text_emb.shape[0] != b:
            raise ValueError(f"text_emb must be (B,D), got {tuple(text_emb.shape)}")
        if state.ndim != 3 or state.shape[:2] != (b, t):
            raise ValueError(f"state must be (B,T,D), got {tuple(state.shape)}")
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state_dim={self.state_dim}, got {state.shape[-1]}.")

        if lengths is None:
            lengths = torch.full((b,), t, dtype=torch.long, device=image_emb.device)
        else:
            lengths = lengths.to(device=image_emb.device, dtype=torch.long)
        if lengths.shape != (b,):
            raise ValueError(f"lengths must be (B,), got {tuple(lengths.shape)}")
        if torch.any(lengths < 1) or torch.any(lengths > t):
            raise ValueError(f"lengths values must be in [1, {t}], got {lengths.tolist()}.")

        visual = self.visual_proj(image_emb).permute(0, 2, 1, 3)
        language = self.lang_proj(text_emb).view(b, 1, 1, self.d_model).expand(-1, t, -1, -1)
        tokens = [visual, language]
        if self.state_proj is not None:
            tokens.append(self.state_proj(state).unsqueeze(2))
        x = torch.cat(tokens, dim=2)
        x = x + self._time_encoding(
            t,
            self.d_model,
            device=x.device,
            dtype=x.dtype,
        )
        x = x + self.modality_pos

        time_ids = torch.arange(t, device=x.device).repeat_interleave(self.num_modalities)
        causal_mask = time_ids.unsqueeze(0) > time_ids.unsqueeze(1)
        padding_mask = torch.arange(t, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        padding_mask = padding_mask.repeat_interleave(self.num_modalities, dim=1)

        encoded = self.transformer(
            x.reshape(b, t * self.num_modalities, self.d_model),
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        features = encoded.reshape(b, t, self.num_modalities * self.d_model)
        progress = self.fusion_net(features).squeeze(-1)
        valid_steps = torch.arange(t, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        return progress.masked_fill(~valid_steps, 0.0)
