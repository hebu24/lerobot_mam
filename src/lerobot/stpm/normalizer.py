from __future__ import annotations

import json
from pathlib import Path

import torch


class StateNormalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std.clamp_min(1e-6)

    def to(self, device: torch.device | str) -> "StateNormalizer":
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def normalize(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self.mean.to(state.device, state.dtype)) / self.std.to(state.device, state.dtype)


def load_state_normalizer(path: str | Path, device: torch.device | str, state_dim: int) -> StateNormalizer:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    stats = data["norm_stats"]["state"]
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=device)
    if mean.numel() != state_dim:
        raise ValueError(f"state_norm dim={mean.numel()} does not match state_dim={state_dim}")
    return StateNormalizer(mean, std)


def save_state_norm(path: str | Path, states: torch.Tensor, meta: dict) -> None:
    path = Path(path)
    flat = states.reshape(-1, states.shape[-1]).float()
    out = {
        "norm_stats": {
            "state": {
                "mean": flat.mean(dim=0).tolist(),
                "std": flat.std(dim=0).clamp_min(1e-6).tolist(),
            }
        },
        "meta": meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
