#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from lerobot.stpm import FrameLeRobotDataset, FrozenCLIPEncoder, RewardTransformer
from lerobot.stpm.normalizer import save_state_norm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train STPM progress model on a LeRobot dataset.")
    parser.add_argument("--dataset.repo_id", dest="repo_id", required=True)
    parser.add_argument("--dataset.root", dest="root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--episodes", type=str, default=None)
    parser.add_argument("--n_obs_steps", type=int, default=1)
    parser.add_argument("--frame_gap", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--task_description", default="")
    parser.add_argument("--vision_ckpt", default="openai/clip-vit-base-patch32")
    parser.add_argument("--reward_ckpt", type=Path, default=None)
    parser.add_argument("--allow_partial_reward_ckpt", action="store_true")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    return parser.parse_args()


def _parse_episodes(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    return [int(x) for x in raw.strip("[]").split(",") if x.strip()]


def _resolve_device(device_arg: str, require_cuda: bool) -> torch.device:
    device = torch.device(device_arg)
    if require_cuda and device.type != "cuda":
        raise ValueError(f"--require_cuda needs a cuda device, got {device_arg!r}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is false. "
            "Check NVIDIA driver, CUDA runtime, container GPU passthrough, and CUDA_VISIBLE_DEVICES."
        )
    return device


def _load_reward_checkpoint(
    model: torch.nn.Module,
    ckpt_path: Path,
    device: torch.device,
    *,
    allow_partial: bool,
) -> None:
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if not allow_partial:
        model.load_state_dict(state_dict)
        print(f"[Init] Loaded STPM reward checkpoint: {ckpt_path}")
        return

    current = model.state_dict()
    matched = {
        key: value
        for key, value in state_dict.items()
        if key in current and tuple(value.shape) == tuple(current[key].shape)
    }
    skipped = sorted(set(state_dict) - set(matched))
    current.update(matched)
    model.load_state_dict(current)
    print(
        f"[Init] Partially loaded STPM reward checkpoint: {ckpt_path} "
        f"({len(matched)} tensors loaded, {len(skipped)} skipped)"
    )


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device, args.require_cuda)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"[Init] Using device: {device}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset = FrameLeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
        episodes=_parse_episodes(args.episodes),
        n_obs_steps=args.n_obs_steps,
        frame_gap=args.frame_gap,
        task_description=args.task_description or None,
    )
    all_states = torch.stack([dataset[i]["state"] for i in range(len(dataset))], dim=0)
    state_norm_path = args.output_dir / "state_norm.json"
    save_state_norm(
        state_norm_path,
        all_states,
        meta={
            "source_root": str(args.root),
            "repo_id": args.repo_id,
            "camera_names": dataset.camera_keys,
            "state_dim": int(all_states.shape[-1]),
        },
    )

    val_len = int(len(dataset) * args.val_ratio)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(0),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        )
        if val_len
        else None
    )

    clip_encoder = FrozenCLIPEncoder(args.vision_ckpt, device=device)
    model = RewardTransformer(
        d_model=args.d_model,
        vis_emb_dim=clip_encoder.emb_dim,
        text_emb_dim=clip_encoder.emb_dim,
        state_dim=int(all_states.shape[-1]),
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        num_cameras=len(dataset.camera_keys),
    ).to(device)
    if args.reward_ckpt is not None:
        _load_reward_checkpoint(
            model,
            args.reward_ckpt,
            device,
            allow_partial=args.allow_partial_reward_ckpt,
        )
    state_mean = all_states.reshape(-1, all_states.shape[-1]).mean(dim=0).to(device)
    state_std = all_states.reshape(-1, all_states.shape[-1]).std(dim=0).clamp_min(1e-6).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    cfg = {
        "repo_id": args.repo_id,
        "root": str(args.root),
        "device": str(device),
        "camera_names": dataset.camera_keys,
        "image_shape": [dataset.meta.features[key]["shape"] for key in dataset.camera_keys],
        "state_dim": int(all_states.shape[-1]),
        "n_obs_steps": args.n_obs_steps,
        "frame_gap": args.frame_gap,
        "task_description": args.task_description,
        "state_norm_path": str(state_norm_path),
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
        "vision_ckpt": args.vision_ckpt,
        "reward_ckpt": str(args.reward_ckpt) if args.reward_ckpt is not None else "",
    }
    with open(args.output_dir / "config.yaml", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    best_val = float("inf")
    loader_iter = iter(train_loader)
    pbar = tqdm(range(args.steps), desc="STPM")
    for step in pbar:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)
        model.train()
        images = batch["image_frames"].to(device)
        state = batch["state"].to(device)
        targets = batch["targets"].to(device)
        b, t, n = images.shape[:3]
        flat = images.permute(2, 0, 1, 3, 4, 5).reshape(n * b * t, 3, images.shape[-2], images.shape[-1])
        img_emb = clip_encoder.encode_image(flat).view(n, b, t, -1).permute(1, 0, 2, 3)
        text_emb = clip_encoder.encode_text(list(batch["task"]))
        norm_state = (state - state_mean) / state_std
        pred = model(img_emb, text_emb, norm_state, batch["lengths"].to(device))
        loss = F.mse_loss(pred, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()
        pbar.set_postfix(loss=float(loss.item()))

        if step % 500 == 0 or step == args.steps - 1:
            val_loss = float(loss.item())
            if val_loader is not None:
                model.eval()
                losses = []
                with torch.no_grad():
                    for val_batch in val_loader:
                        images = val_batch["image_frames"].to(device)
                        state = val_batch["state"].to(device)
                        targets = val_batch["targets"].to(device)
                        b, t, n = images.shape[:3]
                        flat = images.permute(2, 0, 1, 3, 4, 5).reshape(
                            n * b * t, 3, images.shape[-2], images.shape[-1]
                        )
                        img_emb = clip_encoder.encode_image(flat).view(n, b, t, -1).permute(1, 0, 2, 3)
                        text_emb = clip_encoder.encode_text(list(val_batch["task"]))
                        norm_state = (state - state_mean) / state_std
                        losses.append(F.mse_loss(model(img_emb, text_emb, norm_state), targets).item())
                val_loss = float(sum(losses) / max(len(losses), 1))
            if val_loss <= best_val:
                best_val = val_loss
                torch.save(
                    {"model": model.state_dict(), "step": step, "val_loss": val_loss},
                    ckpt_dir / "reward_best.pt",
                )
    torch.save(
        {"model": model.state_dict(), "step": args.steps, "val_loss": best_val},
        ckpt_dir / "reward_final.pt",
    )


if __name__ == "__main__":
    main()
