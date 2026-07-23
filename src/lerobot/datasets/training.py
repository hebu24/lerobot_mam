from __future__ import annotations

import logging
from typing import Any

import torch

from lerobot.configs.train import TrainPipelineConfig


def prepare_training_config(cfg: TrainPipelineConfig) -> None:
    """Apply dataset-selection settings before constructing the dataset."""
    if not cfg.overfit_test:
        return
    if cfg.overfit_per_task:
        cfg.eval.n_episodes = cfg.num_overfit_per_task
        cfg.eval.batch_size = (
            min(cfg.eval.batch_size, cfg.num_overfit_per_task)
            if cfg.eval.batch_size
            else cfg.num_overfit_per_task
        )
        from .libero_training import apply_overfit_per_task_episode_selection

        apply_overfit_per_task_episode_selection(cfg)
        return

    overfit_episodes = (
        list(cfg.dataset.episodes) if cfg.dataset.episodes is not None else list(range(cfg.num_overfit))
    )
    if not overfit_episodes:
        raise ValueError("overfit_test=True requires at least one selected episode.")
    cfg.dataset.episodes = overfit_episodes
    cfg.eval.n_episodes = len(overfit_episodes)
    cfg.eval.batch_size = (
        min(cfg.eval.batch_size, len(overfit_episodes)) if cfg.eval.batch_size else len(overfit_episodes)
    )
    logging.info(
        "Overfit test enabled: training episodes=%s, eval.n_episodes=%d, eval.batch_size=%d",
        overfit_episodes,
        cfg.eval.n_episodes,
        cfg.eval.batch_size,
    )


def prepare_dataset_for_training(
    cfg: TrainPipelineConfig,
    dataset: Any,
    *,
    configure_eval: bool,
) -> None:
    """Apply generic and dataset-specific preparation after dataset construction."""
    env_type = getattr(cfg.env, "type", None) if cfg.env is not None else None
    robot_type = getattr(dataset.meta, "robot_type", None)
    if robot_type == "libero" or env_type in {"libero", "libero_plus"}:
        from .libero_training import (
            apply_diffusion_relative_action_stats,
            apply_overfit_eval_init_state_ids,
            validate_libero_v3_training_dataset,
        )

        validate_libero_v3_training_dataset(cfg, dataset)
    apply_overfit_subset_stats(cfg, dataset)
    if robot_type == "libero" or env_type in {"libero", "libero_plus"}:
        apply_diffusion_relative_action_stats(cfg, dataset)
        if configure_eval:
            apply_overfit_eval_init_state_ids(cfg, dataset)


def apply_overfit_subset_stats(cfg: TrainPipelineConfig, dataset: Any) -> None:
    """Recompute numeric policy-feature stats on a strict overfit subset."""
    if not cfg.overfit_test or cfg.dataset.episodes is None:
        return
    if cfg.dataset.streaming:
        raise ValueError("overfit_test=True does not support subset-stat recomputation for streaming data.")

    active_cfg = cfg.trainable_config
    policy_features = {
        **(getattr(active_cfg, "input_features", None) or {}),
        **(getattr(active_cfg, "output_features", None) or {}),
    }
    feature_keys = set(policy_features) or set(dataset.meta.features)
    metadata_keys = {"index", "episode_index", "task_index", "frame_index", "timestamp"}
    hf_dataset = getattr(getattr(dataset, "reader", None), "hf_dataset", None)
    if hf_dataset is None:
        raise ValueError("Overfit subset stats require a loaded random-access dataset.")

    dataset.meta.stats = dict(dataset.meta.stats or {})
    updated: list[str] = []
    quantiles = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}
    frame_count: int | None = None
    for key in sorted(feature_keys):
        if key in metadata_keys or key not in dataset.meta.features:
            continue
        dtype = dataset.meta.features[key]["dtype"]
        if dtype in {"image", "video", "string", "language"}:
            continue
        arrow_data = getattr(hf_dataset, "data", None)
        column = (
            arrow_data.column(key).to_pylist()
            if arrow_data is not None and key in arrow_data.column_names
            else hf_dataset[key]
        )
        if len(column) == 0:
            raise ValueError(f"Cannot compute overfit stats for empty feature {key!r}.")
        values = torch.stack([torch.as_tensor(value) for value in column]).to(dtype=torch.float32)
        frame_count = values.shape[0]
        if not torch.isfinite(values).all():
            raise ValueError(f"Cannot compute overfit stats for non-finite feature {key!r}.")

        feature_stats = {
            "min": values.amin(dim=0),
            "max": values.amax(dim=0),
            "mean": values.mean(dim=0),
            "std": values.std(dim=0, unbiased=False),
            "count": torch.tensor([values.shape[0]], dtype=torch.int64),
        }
        for name, quantile in quantiles.items():
            feature_stats[name] = torch.quantile(values, quantile, dim=0)
        dataset.meta.stats[key] = feature_stats
        updated.append(key)

    logging.info(
        "Overfit normalization stats recomputed from %d selected frame(s): %s",
        frame_count or 0,
        updated,
    )


def get_sampler_episode_boundaries(dataset: Any) -> tuple[list[int], list[int]]:
    """Return episode boundaries in the index space expected by DatasetReader.get_item."""
    if dataset.episodes is None:
        return (
            [int(idx) for idx in dataset.meta.episodes["dataset_from_index"]],
            [int(idx) for idx in dataset.meta.episodes["dataset_to_index"]],
        )

    selected_episodes = {int(ep) for ep in dataset.episodes}
    selected_rows = [row for row in dataset.meta.episodes if int(row["episode_index"]) in selected_episodes]
    missing = selected_episodes - {int(row["episode_index"]) for row in selected_rows}
    if missing:
        raise ValueError(f"Sampler episodes missing from dataset metadata: {sorted(missing)}")

    selected_rows.sort(key=lambda row: int(row["dataset_from_index"]))
    from_indices: list[int] = []
    to_indices: list[int] = []
    cursor = 0
    for row in selected_rows:
        episode_length = int(row["dataset_to_index"]) - int(row["dataset_from_index"])
        from_indices.append(cursor)
        cursor += episode_length
        to_indices.append(cursor)

    if cursor != dataset.num_frames:
        logging.warning(
            "Sampler frame count (%d) differs from loaded dataset.num_frames (%d).",
            cursor,
            dataset.num_frames,
        )

    return from_indices, to_indices
