#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from lerobot.stpm import FrameLeRobotDataset, FrozenCLIPEncoder, RewardTransformer
from lerobot.stpm.normalizer import save_state_norm
from lerobot.utils.constants import OBS_STATE


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
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="If set, overrides --steps with len(train_loader) * epochs.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument(
        "--scheduler_total_steps",
        type=int,
        default=0,
        help="Warmup/cosine scheduler horizon. Set to 0 to disable scheduling.",
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=0.0,
        help="Maximum gradient norm. Set to 0 to disable clipping.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument(
        "--task_description",
        default=None,
        help="Deprecated and ignored. STPM task text is always read from the dataset.",
    )
    parser.add_argument("--vision_ckpt", default="openai/clip-vit-base-patch32")
    parser.add_argument("--reward_ckpt", type=Path, default=None)
    parser.add_argument("--allow_partial_reward_ckpt", action="store_true")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--clip_encode_batch_size", type=int, default=64)
    return parser.parse_args()


def _parse_episodes(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    return [int(x) for x in raw.strip("[]").split(",") if x.strip()]


def _load_all_states(dataset: FrameLeRobotDataset, indices: list[int] | None = None) -> torch.Tensor:
    states = dataset.dataset.select_columns(OBS_STATE)[OBS_STATE]
    if indices is not None:
        states = [states[index] for index in indices]
    if len(states) == 0:
        raise ValueError("Cannot compute STPM state normalization from an empty train split.")
    return torch.stack(
        [
            state.float() if hasattr(state, "float") else torch.as_tensor(state, dtype=torch.float32)
            for state in states
        ],
        dim=0,
    )


def _split_dataset_by_episode(
    dataset: FrameLeRobotDataset,
    val_ratio: float,
    *,
    seed: int = 0,
) -> tuple[Subset, Subset, list[int], list[int], dict[str, object]]:
    """Split frames by source trajectory, falling back to local episodes."""
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"--val_ratio must be in [0, 1), got {val_ratio}.")

    frame_episode_ids = [
        int(value) for value in dataset.dataset.select_columns("episode_index")["episode_index"]
    ]
    if len(frame_episode_ids) != len(dataset):
        raise ValueError(
            "Episode metadata length does not match the STPM frame dataset: "
            f"{len(frame_episode_ids)} != {len(dataset)}."
        )
    local_episode_ids = sorted(set(frame_episode_ids))
    if not local_episode_ids:
        raise ValueError("STPM dataset contains no episodes.")

    episode_rows = getattr(dataset, "episode_rows", {})
    source_field = next(
        (
            field
            for field in ("libero/source_episode_id", "source_episode_id")
            if all(
                episode_rows.get(episode_id, {}).get(field) is not None for episode_id in local_episode_ids
            )
        ),
        None,
    )
    task_field = next(
        (
            field
            for field in ("libero/task_id", "task_index", "task_id", "libero/task_name", "tasks")
            if all(
                episode_rows.get(episode_id, {}).get(field) is not None for episode_id in local_episode_ids
            )
        ),
        None,
    )

    episode_groups: dict[int, tuple[str, str]] = {}
    for episode_id in local_episode_ids:
        row = episode_rows.get(episode_id, {})
        source_identity = row[source_field] if source_field is not None else episode_id
        task_identity = row[task_field] if task_field is not None else ""
        episode_groups[episode_id] = (str(task_identity), str(source_identity))

    group_ids = sorted(set(episode_groups.values()))
    val_groups: set[tuple[str, str]] = set()
    if val_ratio > 0.0 and len(group_ids) > 1:
        val_group_count = min(
            len(group_ids) - 1,
            max(1, math.ceil(len(group_ids) * val_ratio)),
        )
        permutation = torch.randperm(
            len(group_ids),
            generator=torch.Generator().manual_seed(seed),
        ).tolist()
        val_groups = {group_ids[index] for index in permutation[:val_group_count]}

    val_episode_ids = sorted(
        episode_id for episode_id, group_id in episode_groups.items() if group_id in val_groups
    )
    val_episode_set = set(val_episode_ids)
    train_episode_ids = sorted(set(local_episode_ids) - val_episode_set)
    train_indices = [
        index for index, episode_id in enumerate(frame_episode_ids) if episode_id not in val_episode_set
    ]
    val_indices = [
        index for index, episode_id in enumerate(frame_episode_ids) if episode_id in val_episode_set
    ]
    split_identity = {
        "source_field": source_field or "episode_index",
        "task_field": task_field,
        "tasks": sorted({task for task, _ in group_ids}),
        "train_groups": [
            {"task": task, "source_episode_id": source}
            for task, source in sorted(set(group_ids) - val_groups)
        ],
        "val_groups": [{"task": task, "source_episode_id": source} for task, source in sorted(val_groups)],
    }
    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        train_episode_ids,
        val_episode_ids,
        split_identity,
    )


def _masked_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError(
            "STPM prediction and target must have matching (B,T) shapes, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}."
        )
    lengths = lengths.to(device=prediction.device, dtype=torch.long)
    if lengths.shape != (prediction.shape[0],):
        raise ValueError(f"lengths must be (B,), got {tuple(lengths.shape)}.")
    valid = torch.arange(prediction.shape[1], device=prediction.device).unsqueeze(0) < lengths.unsqueeze(1)
    if not torch.any(valid):
        raise ValueError("STPM batch contains no valid timesteps.")
    return functional.mse_loss(prediction[valid], target[valid])


def _endpoint_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Return MSE at each sequence's last valid (rollout anchor) timestep."""
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError(
            "STPM prediction and target must have matching (B,T) shapes, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}."
        )
    lengths = lengths.to(device=prediction.device, dtype=torch.long)
    if lengths.shape != (prediction.shape[0],):
        raise ValueError(f"lengths must be (B,), got {tuple(lengths.shape)}.")
    if torch.any(lengths <= 0) or torch.any(lengths > prediction.shape[1]):
        raise ValueError(f"lengths must be in [1, {prediction.shape[1]}], got {lengths.tolist()}.")
    endpoint_indices = lengths - 1
    batch_indices = torch.arange(prediction.shape[0], device=prediction.device)
    return functional.mse_loss(
        prediction[batch_indices, endpoint_indices],
        target[batch_indices, endpoint_indices],
    )


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
    model: RewardTransformer,
    ckpt_path: Path,
    device: torch.device,
    *,
    allow_partial: bool,
) -> None:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"STPM checkpoint must contain a state dict, got {type(state_dict).__name__}.")
    model.validate_checkpoint_state_dict(state_dict)
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


def _encode_images_in_chunks(
    clip_encoder: FrozenCLIPEncoder,
    images: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    if chunk_size <= 0 or images.shape[0] <= chunk_size:
        return clip_encoder.encode_image(images)
    return torch.cat([clip_encoder.encode_image(chunk) for chunk in images.split(chunk_size)], dim=0)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _resolve_device(args.device, args.require_cuda)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"[Init] Using device: {device}")
    if args.task_description:
        print(
            "[Warn] --task_description is deprecated and ignored; "
            "STPM uses per-frame task text from the dataset."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    selected_episodes = _parse_episodes(args.episodes)
    dataset = FrameLeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
        episodes=selected_episodes,
        n_obs_steps=args.n_obs_steps,
        frame_gap=args.frame_gap,
    )
    train_set, val_set, train_episode_ids, val_episode_ids, split_identity = _split_dataset_by_episode(
        dataset,
        args.val_ratio,
        seed=args.seed,
    )
    train_indices = list(train_set.indices)
    all_states = _load_all_states(dataset, train_indices)
    state_norm_path = args.output_dir / "state_norm.json"
    save_state_norm(
        state_norm_path,
        all_states,
        meta={
            "source_root": str(args.root),
            "repo_id": args.repo_id,
            "camera_names": dataset.camera_keys,
            "state_dim": int(all_states.shape[-1]),
            "train_episode_ids": train_episode_ids,
            "split_identity": split_identity,
        },
    )

    train_len = len(train_set)
    val_len = len(val_set)
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
    steps_per_epoch = len(train_loader)
    if steps_per_epoch <= 0:
        raise ValueError(
            f"STPM train split is empty: dataset_len={len(dataset)}, train_len={train_len}, val_len={val_len}."
        )
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError(f"--epochs must be positive, got {args.epochs}.")
        total_steps = steps_per_epoch * args.epochs
        print(
            f"[Init] Training for {args.epochs} epoch(s): "
            f"{steps_per_epoch} steps/epoch -> {total_steps} total steps"
        )
    else:
        if args.steps <= 0:
            raise ValueError(f"--steps must be positive, got {args.steps}.")
        total_steps = args.steps
        print(f"[Init] Training for {total_steps} fixed steps ({steps_per_epoch} steps/epoch).")

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
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.scheduler_total_steps < 0:
        raise ValueError(f"--scheduler_total_steps must be non-negative, got {args.scheduler_total_steps}.")
    if args.warmup_steps < 0:
        raise ValueError(f"--warmup_steps must be non-negative, got {args.warmup_steps}.")
    if args.scheduler_total_steps > 0:
        if args.warmup_steps >= args.scheduler_total_steps:
            raise ValueError(
                "--warmup_steps must be smaller than --scheduler_total_steps, got "
                f"{args.warmup_steps} and {args.scheduler_total_steps}."
            )
        if args.warmup_steps > 0:
            if args.lr <= 0:
                raise ValueError(f"--lr must be positive when warmup is enabled, got {args.lr}.")
            warmup = LinearLR(
                optim,
                start_factor=1e-6 / args.lr,
                end_factor=1.0,
                total_iters=args.warmup_steps,
            )
            cosine = CosineAnnealingLR(
                optim,
                T_max=args.scheduler_total_steps - args.warmup_steps,
                eta_min=0.0,
            )
            scheduler = SequentialLR(
                optim,
                schedulers=[warmup, cosine],
                milestones=[args.warmup_steps],
            )
        else:
            scheduler = CosineAnnealingLR(
                optim,
                T_max=args.scheduler_total_steps,
                eta_min=0.0,
            )

    cfg = {
        "repo_id": args.repo_id,
        "root": str(args.root),
        "device": str(device),
        "camera_names": dataset.camera_keys,
        "image_shape": [dataset.meta.features[key]["shape"] for key in dataset.camera_keys],
        "state_dim": int(all_states.shape[-1]),
        "n_obs_steps": args.n_obs_steps,
        "frame_gap": args.frame_gap,
        "selected_episode_ids": sorted(train_episode_ids + val_episode_ids),
        "train_episode_ids": train_episode_ids,
        "val_episode_ids": val_episode_ids,
        "split_identity": split_identity,
        "split_seed": args.seed,
        "steps": total_steps,
        "epochs": args.epochs,
        "steps_per_epoch": steps_per_epoch,
        "train_len": train_len,
        "val_len": val_len,
        "batch_size": args.batch_size,
        "val_ratio": args.val_ratio,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "adam_betas": [args.adam_beta1, args.adam_beta2],
        "adam_eps": args.adam_eps,
        "warmup_steps": args.warmup_steps,
        "scheduler_total_steps": args.scheduler_total_steps,
        "grad_clip_norm": args.grad_clip_norm,
        "seed": args.seed,
        "task_description": "",
        "task_source": "dataset",
        "state_norm_path": str(state_norm_path),
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
        "vision_ckpt": args.vision_ckpt,
        "reward_ckpt": str(args.reward_ckpt) if args.reward_ckpt is not None else "",
        "clip_encode_batch_size": args.clip_encode_batch_size,
        "architecture_version": RewardTransformer.ARCHITECTURE_VERSION,
        "checkpoint_selection": {
            "reward_best.pt": "validation_sequence_mse",
            "reward_best_endpoint.pt": "validation_endpoint_mse",
        },
    }
    with open(args.output_dir / "config.yaml", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    best_val = float("inf")
    best_endpoint_val = float("inf")
    final_val = float("inf")
    final_endpoint_val = float("inf")
    loader_iter = iter(train_loader)
    pbar = tqdm(range(total_steps), desc="STPM")
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
        img_emb = (
            _encode_images_in_chunks(clip_encoder, flat, chunk_size=args.clip_encode_batch_size)
            .view(n, b, t, -1)
            .permute(1, 0, 2, 3)
        )
        text_emb = clip_encoder.encode_text(list(batch["task"]))
        norm_state = (state - state_mean) / state_std
        lengths = batch["lengths"].to(device)
        pred = model(img_emb, text_emb, norm_state, lengths)
        loss = _masked_mse_loss(pred, targets, lengths)
        optim.zero_grad()
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optim.step()
        if scheduler is not None:
            scheduler.step()
        pbar.set_postfix(loss=float(loss.item()))

        if step % 500 == 0 or step == total_steps - 1:
            val_loss = float(loss.item())
            endpoint_val_loss = float(_endpoint_mse_loss(pred, targets, lengths).item())
            if val_loader is not None:
                model.eval()
                sequence_squared_error = 0.0
                sequence_count = 0
                endpoint_squared_error = 0.0
                endpoint_count = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        images = val_batch["image_frames"].to(device)
                        state = val_batch["state"].to(device)
                        targets = val_batch["targets"].to(device)
                        b, t, n = images.shape[:3]
                        flat = images.permute(2, 0, 1, 3, 4, 5).reshape(
                            n * b * t, 3, images.shape[-2], images.shape[-1]
                        )
                        img_emb = (
                            _encode_images_in_chunks(
                                clip_encoder, flat, chunk_size=args.clip_encode_batch_size
                            )
                            .view(n, b, t, -1)
                            .permute(1, 0, 2, 3)
                        )
                        text_emb = clip_encoder.encode_text(list(val_batch["task"]))
                        norm_state = (state - state_mean) / state_std
                        lengths = val_batch["lengths"].to(device)
                        prediction = model(img_emb, text_emb, norm_state, lengths)
                        valid = torch.arange(prediction.shape[1], device=device).unsqueeze(
                            0
                        ) < lengths.unsqueeze(1)
                        sequence_squared_error += float(
                            torch.square(prediction[valid] - targets[valid]).sum().item()
                        )
                        sequence_count += int(valid.sum().item())
                        endpoint_indices = lengths - 1
                        batch_indices = torch.arange(prediction.shape[0], device=device)
                        endpoint_squared_error += float(
                            torch.square(
                                prediction[batch_indices, endpoint_indices]
                                - targets[batch_indices, endpoint_indices]
                            )
                            .sum()
                            .item()
                        )
                        endpoint_count += int(prediction.shape[0])
                val_loss = sequence_squared_error / max(sequence_count, 1)
                endpoint_val_loss = endpoint_squared_error / max(endpoint_count, 1)
            final_val = val_loss
            final_endpoint_val = endpoint_val_loss
            if val_loss <= best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "architecture_version": RewardTransformer.ARCHITECTURE_VERSION,
                        "step": step,
                        "val_loss": val_loss,
                        "val_sequence_mse": val_loss,
                        "val_endpoint_mse": endpoint_val_loss,
                        "selection_metric": "validation_sequence_mse",
                    },
                    ckpt_dir / "reward_best.pt",
                )
            if endpoint_val_loss <= best_endpoint_val:
                best_endpoint_val = endpoint_val_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "architecture_version": RewardTransformer.ARCHITECTURE_VERSION,
                        "step": step,
                        "val_loss": val_loss,
                        "val_sequence_mse": val_loss,
                        "val_endpoint_mse": endpoint_val_loss,
                        "selection_metric": "validation_endpoint_mse",
                    },
                    ckpt_dir / "reward_best_endpoint.pt",
                )
    torch.save(
        {
            "model": model.state_dict(),
            "architecture_version": RewardTransformer.ARCHITECTURE_VERSION,
            "step": total_steps - 1,
            "val_loss": final_val,
            "val_sequence_mse": final_val,
            "val_endpoint_mse": final_endpoint_val,
            "best_val_sequence_mse": best_val,
            "best_val_endpoint_mse": best_endpoint_val,
            "selection_metric": "final_training_step",
        },
        ckpt_dir / "reward_final.pt",
    )


if __name__ == "__main__":
    main()
