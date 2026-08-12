#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train a policy.

Requires: pip install 'lerobot[training]'  (includes dataset + accelerate + wandb extras)
"""

import dataclasses
import json
import logging
import os
import time
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from accelerate import Accelerator

import torch
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.common.train_utils import (
    append_jsonl,
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    prepare_checkpoint_dir_for_save,
    prune_checkpoint_training_state_if_not_last,
    prune_checkpoints_keep,
    restore_resume_eval_state,
    save_checkpoint,
    save_policy_checkpoint,
    trim_jsonl_after_step,
    update_best_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import EpisodeAwareSampler, make_dataset
from lerobot.datasets.training import (
    get_sampler_episode_boundaries,
    prepare_dataset_for_training,
    prepare_training_config,
)
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import (
    PreTrainedPolicy,
    make_policy,
    make_pre_post_processors,
    prepare_policy_evaluation,
    run_policy_evaluation,
)
from lerobot.rewards import make_reward_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.tmux_eval_recorder import append_tmux_eval_result
from lerobot.utils.utils import (
    cycle,
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)

from .lerobot_eval import eval_policy_all


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "__fspath__"):
        return str(value)
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _empty_eval_info(rank: int, task_ids: list[int]) -> dict[str, Any]:
    return {
        "per_task": [],
        "per_group": {},
        "overall": {
            "avg_sum_reward": float("nan"),
            "avg_max_reward": float("nan"),
            "pc_success": float("nan"),
            "n_episodes": 0,
            "eval_s": 0.0,
            "eval_ep_s": float("nan"),
            "video_paths": [],
        },
        "rank_eval": [{"rank": rank, "task_ids": task_ids, "n_episodes": 0, "eval_s": 0.0}],
    }


def _merge_distributed_eval_infos(rank_infos: list[dict[str, Any]]) -> dict[str, Any]:
    sum_rewards: list[float] = []
    max_rewards: list[float] = []
    successes: list[bool] = []
    video_paths: list[str] = []
    per_task: list[dict[str, Any]] = []
    per_episode: list[dict[str, Any]] = []
    rank_eval: list[dict[str, Any]] = []
    group_acc: dict[str, dict[str, list[Any]]] = {}

    def group_state(group: str) -> dict[str, list[Any]]:
        return group_acc.setdefault(
            group,
            {"sum_rewards": [], "max_rewards": [], "successes": [], "video_paths": []},
        )

    for rank_info in sorted(rank_infos, key=lambda item: int(item["rank"])):
        rank = int(rank_info["rank"])
        task_ids = [int(task_id) for task_id in rank_info.get("task_ids", [])]
        info = rank_info["eval_info"]
        overall = info.get("overall", {})
        rank_eval.append(
            {
                "rank": rank,
                "task_ids": task_ids,
                "n_episodes": int(overall.get("n_episodes", 0) or 0),
                "pc_success": overall.get("pc_success"),
                "avg_sum_reward": overall.get("avg_sum_reward"),
                "eval_s": overall.get("eval_s", 0.0),
            }
        )

        task_infos = info.get("per_task", []) or []
        if not task_infos:
            video_paths.extend(str(path) for path in overall.get("video_paths", []) or [])
        for episode_info in info.get("per_episode", []) or []:
            episode_record = dict(episode_info)
            episode_record["rank"] = rank
            per_episode.append(episode_record)

        for task_info in task_infos:
            task_group = str(task_info.get("task_group"))
            metrics = task_info.get("metrics", {})
            task_sum_rewards = [float(value) for value in metrics.get("sum_rewards", []) or []]
            task_max_rewards = [float(value) for value in metrics.get("max_rewards", []) or []]
            task_successes = [bool(value) for value in metrics.get("successes", []) or []]
            task_video_paths = [str(path) for path in metrics.get("video_paths", []) or []]

            sum_rewards.extend(task_sum_rewards)
            max_rewards.extend(task_max_rewards)
            successes.extend(task_successes)
            video_paths.extend(task_video_paths)

            acc = group_state(task_group)
            acc["sum_rewards"].extend(task_sum_rewards)
            acc["max_rewards"].extend(task_max_rewards)
            acc["successes"].extend(task_successes)
            acc["video_paths"].extend(task_video_paths)

            task_record = dict(task_info)
            task_record["rank"] = rank
            per_task.append(task_record)

    per_group = {
        group: {
            "avg_sum_reward": _mean([float(value) for value in acc["sum_rewards"]]),
            "avg_max_reward": _mean([float(value) for value in acc["max_rewards"]]),
            "pc_success": _mean([1.0 if value else 0.0 for value in acc["successes"]]) * 100
            if acc["successes"]
            else float("nan"),
            "n_episodes": len(acc["sum_rewards"]),
            "video_paths": list(acc["video_paths"]),
        }
        for group, acc in sorted(group_acc.items())
    }
    eval_s = max((float(rank.get("eval_s") or 0.0) for rank in rank_eval), default=0.0)
    n_episodes = len(successes)
    for episode_ix, episode_info in enumerate(per_episode):
        episode_info["episode_ix"] = episode_ix

    def success_by(key: str) -> dict[str, float]:
        buckets: dict[str, list[bool]] = {}
        for episode_info in per_episode:
            if key not in episode_info:
                continue
            buckets.setdefault(str(episode_info[key]), []).append(bool(episode_info["success"]))
        return {
            bucket: _mean([1.0 if value else 0.0 for value in values]) * 100
            for bucket, values in sorted(buckets.items())
        }

    return {
        "per_task": sorted(
            per_task,
            key=lambda item: (str(item.get("task_group")), int(item.get("task_id", -1))),
        ),
        "per_episode": per_episode,
        "per_mask_type_success": success_by("mask_type"),
        "per_mask_slot_success": success_by("mask_type_slot"),
        "per_group": per_group,
        "overall": {
            "avg_sum_reward": _mean(sum_rewards),
            "avg_max_reward": _mean(max_rewards),
            "pc_success": _mean([1.0 if value else 0.0 for value in successes]) * 100
            if successes
            else float("nan"),
            "n_episodes": n_episodes,
            "eval_s": eval_s,
            "eval_ep_s": eval_s / n_episodes if n_episodes else float("nan"),
            "video_paths": video_paths,
        },
        "rank_eval": rank_eval,
    }


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: "Accelerator",
    lr_scheduler=None,
    lock=None,
    sample_weighter=None,
) -> tuple[MetricsTracker, dict | None]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
        sample_weighter: Optional SampleWeighter instance for per-sample loss weighting.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Compute sample weights if a weighter is provided
    sample_weights = None
    weight_stats = None
    if sample_weighter is not None:
        sample_weights, weight_stats = sample_weighter.compute_batch_weights(batch)

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        if sample_weights is not None:
            # Use per-sample loss for weighted training
            # Note: Policies supporting sample weighting must implement forward(batch, reduction="none")
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")

            # Weighted loss: each sample's contribution is scaled by its weight.
            # We divide by weight sum (not batch size) so that if some weights are zero,
            # the remaining samples contribute proportionally more, preserving gradient scale.
            # Weights are pre-normalized to sum to batch_size for stable training dynamics.
            epsilon = 1e-6
            loss = (per_sample_loss * sample_weights).sum() / (sample_weights.sum() + epsilon)

            # Log weighting statistics
            if output_dict is None:
                output_dict = {}
            for key, value in weight_stats.items():
                output_dict[f"sample_weight_{key}"] = value
        else:
            loss, output_dict = policy.forward(batch)

        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: "Accelerator | None" = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    from lerobot.utils.import_utils import require_package

    require_package("accelerate", extra="training")
    from accelerate import Accelerator

    cfg.validate()
    if cfg.env is not None:
        cfg.env.validate_policy_compatibility(cfg.trainable_config)
    prepare_training_config(cfg)

    if cfg.env is not None and cfg.eval_freq > 0:
        cfg.env.validate_runtime_assets()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        ddp_timeout_s = int(os.environ.get("LEROBOT_DDP_TIMEOUT_S", "7200"))
        process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=ddp_timeout_s))
        # Accelerate auto-detects the device based on the available hardware and ignores the policy.device setting.
        # Force the device to be CPU when the active config's device is set to CPU (works for both policy and reward model training).
        force_cpu = cfg.trainable_config.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs, process_group_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    # Keep model and processor construction on the local rank device selected by Accelerate.
    # This matters for launchers that expose multiple CUDA devices to every worker.
    active_cfg = cfg.trainable_config
    active_cfg.device = str(device)
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset(cfg)

    should_evaluate = cfg.env is not None and cfg.eval_freq > 0
    prepare_dataset_for_training(cfg, dataset, configure_eval=should_evaluate)

    evaluation_runtime = prepare_policy_evaluation(cfg) if should_evaluate else None
    if evaluation_runtime is None and should_evaluate:
        cfg.env.prepare_evaluation(cfg)

    # Evaluation owns and closes each environment batch at the corresponding step.
    eval_env = None

    if cfg.is_reward_model_training:
        if is_main_process:
            logging.info("Creating reward model")
        from lerobot.rewards import make_reward_model

        policy = make_reward_model(
            cfg=cfg.reward_model,
            dataset_stats=dataset.meta.stats,
            dataset_meta=dataset.meta,
        )
        if not policy.is_trainable:
            raise ValueError(
                f"Reward model '{policy.name}' is zero-shot and cannot be trained via lerobot-train. "
                "Use it directly for inference via compute_reward() (e.g. offline precompute)."
            )
    else:
        if is_main_process:
            logging.info("Creating policy")
        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
            rename_map=cfg.rename_map,
        )

    if cfg.peft is not None:
        if cfg.is_reward_model_training:
            raise ValueError("PEFT is only supported for policy training. ")
        from peft import PeftModel

        if isinstance(policy, PeftModel):
            logging.info("PEFT adapter already loaded from checkpoint, skipping wrap_with_peft.")
        else:
            logging.info("Using PEFT! Wrapping model.")
            peft_cli_overrides = dataclasses.asdict(cfg.peft)
            policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    # Wait for all processes to finish model creation before continuing
    accelerator.wait_for_everyone()

    processor_pretrained_path = active_cfg.pretrained_path
    if (
        getattr(active_cfg, "use_relative_actions", False)
        and processor_pretrained_path is not None
        and not cfg.resume
    ):
        logging.warning(
            "use_relative_actions=true with pretrained processors can skip relative transforms if "
            "the checkpoint processors do not define them. Building processors from current policy config."
        )
        processor_pretrained_path = None

    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (processor_pretrained_path and not cfg.resume) or not processor_pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.is_reward_model_training:
        processor_kwargs["dataset_meta"] = dataset.meta

    if not cfg.is_reward_model_training and processor_pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": str(device)},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    if cfg.is_reward_model_training:
        preprocessor, postprocessor = make_reward_pre_post_processors(
            cfg.reward_model,
            **processor_kwargs,
        )
    else:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=processor_pretrained_path,
            **processor_kwargs,
            **postprocessor_kwargs,
        )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Create sample weighter if configured (e.g., for RA-BC training)
    sample_weighter = None
    if cfg.sample_weighting is not None:
        from lerobot.utils.sample_weighting import make_sample_weighter

        if is_main_process:
            logging.info(f"Creating sample weighter: {cfg.sample_weighting.type}")
        sample_weighter = make_sample_weighter(
            cfg.sample_weighting,
            policy,
            device,
            dataset_root=cfg.dataset.root,
            dataset_repo_id=cfg.dataset.repo_id,
        )

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    env_preprocessor = None
    env_postprocessor = None
    if cfg.env is not None:
        if is_main_process:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=cfg.env, policy_cfg=cfg.policy
        )

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(active_cfg, "drop_n_last_frames"):
        shuffle = False
        sampler_from_indices, sampler_to_indices = get_sampler_episode_boundaries(dataset)
        balance_overfit_episodes = cfg.overfit_test and dataset.num_episodes > 1
        sampler = EpisodeAwareSampler(
            sampler_from_indices,
            sampler_to_indices,
            drop_n_last_frames=active_cfg.drop_n_last_frames,
            shuffle=True,
            balance_episodes=balance_overfit_episodes,
        )
        if balance_overfit_episodes and is_main_process:
            logging.info(
                "Overfit sampler balances selected episodes to %d samples each.",
                len(sampler) // len(sampler.episode_indices),
            )
    else:
        shuffle = True
        sampler = None

    # Only swap in the language-aware collate when the dataset actually
    # declares language columns; otherwise stay on PyTorch's default
    # collate so non-language training runs are unaffected.
    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Keep global batch size for logging; MetricsTracker handles world size internally.
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    best_eval_score: tuple[float, float] | None = None
    best_checkpoint_dir: Path | None = None
    last_checkpoint_dir: Path | None = cfg.checkpoint_path if cfg.resume else None
    if is_main_process:
        train_log_path = cfg.output_dir / "logs" / "train_metrics.jsonl"
        eval_log_path = cfg.output_dir / "logs" / "eval_metrics.jsonl"
        if cfg.resume:
            trim_jsonl_after_step(train_log_path, step)
            eval_records = trim_jsonl_after_step(eval_log_path, step)
            best_eval_score, best_checkpoint_dir = restore_resume_eval_state(
                eval_records,
                output_dir=cfg.output_dir,
                total_steps=cfg.steps,
                checkpoint_path=cfg.checkpoint_path,
            )
            logging.info(
                "Resume restored best eval state: score=%s checkpoint=%s",
                best_eval_score,
                best_checkpoint_dir,
            )
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )
        logging.info("Training metrics will be appended to %s", train_log_path)
        if cfg.env is not None and cfg.eval_freq > 0:
            logging.info("Eval metrics will be appended to %s", eval_log_path)

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        for cam_key in dataset.meta.camera_keys:
            if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            sample_weighter=sample_weighter,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        if is_main_process:
            progbar.update(1)
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            log_dict = train_tracker.to_dict()
            if output_dict:
                log_dict.update(output_dict)
            if sample_weighter is not None:
                weighter_stats = sample_weighter.get_stats()
                log_dict.update({f"sample_weighting/{k}": v for k, v in weighter_stats.items()})
            append_jsonl(
                train_log_path,
                {
                    "mode": "train",
                    "step": step,
                    "time": time.time(),
                    "metrics": log_dict,
                },
            )
            if wandb_logger:
                wandb_logger.log_dict(log_dict, step)
            train_tracker.reset_averages()

        # An eval step persists the same weights after evaluation below. Save
        # non-eval save points here so save_freq and eval_freq remain independent.
        if cfg.save_checkpoint and is_saving_step and not (cfg.env and is_eval_step):
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                prune_checkpoint_training_state_if_not_last(best_checkpoint_dir, last_checkpoint_dir)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                last_checkpoint_dir = checkpoint_dir
                prune_checkpoint_training_state_if_not_last(best_checkpoint_dir, last_checkpoint_dir)
                if cfg.env is not None and cfg.eval_freq > 0:
                    prune_checkpoints_keep(
                        checkpoint_dir.parent,
                        keep_checkpoint_dirs=[best_checkpoint_dir, last_checkpoint_dir],
                        keep_scheduled_from_step=cfg.keep_all_checkpoints_after_step,
                        save_freq=cfg.save_freq,
                        total_steps=cfg.steps,
                    )
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            eval_info = None
            eval_start_seed = cfg.eval.start_seed if cfg.eval.start_seed is not None else cfg.seed
            eval_task_ids = list(getattr(cfg.env, "task_ids", None) or [])
            random_eval_seed_task_ids = (
                eval_task_ids
                if (
                    eval_start_seed is not None
                    and eval_task_ids
                    and not bool(getattr(cfg.env, "init_states", True))
                    and not getattr(cfg.eval, "dataset_repo_id", None)
                    and not getattr(cfg.eval, "dataset_root", None)
                )
                else None
            )
            distributed_policy_runtime = (
                evaluation_runtime is not None
                and accelerator.num_processes > 1
                and bool(eval_task_ids)
                and hasattr(evaluation_runtime, "shard")
            )
            if distributed_policy_runtime:
                rank_task_ids = eval_task_ids[accelerator.process_index :: accelerator.num_processes]
                rank_runtime = evaluation_runtime.shard(rank_task_ids, cfg.eval.n_episodes)
                rank_n_episodes = int(rank_runtime.num_episodes)
                rank_info_dir = cfg.output_dir / "eval" / f"rank_metrics_step_{step_id}"
                if rank_task_ids and rank_n_episodes:
                    rank_env_cfg = deepcopy(cfg.env)
                    rank_env_cfg.task_ids = rank_task_ids
                    logging.info(
                        "Rank %d/%d MAM eval at step %s for task_ids=%s, n_episodes=%d",
                        accelerator.process_index,
                        accelerator.num_processes,
                        step,
                        rank_task_ids,
                        rank_n_episodes,
                    )
                    logging.info("Creating eval env")
                    eval_env = make_env(
                        rank_env_cfg,
                        n_envs=min(cfg.eval.batch_size, rank_n_episodes),
                        use_async_envs=cfg.eval.use_async_envs,
                    )
                    try:
                        with torch.no_grad(), accelerator.autocast():
                            eval_info = run_policy_evaluation(
                                rank_runtime,
                                eval_policy_all,
                                envs=eval_env,
                                policy=accelerator.unwrap_model(policy),
                                env_preprocessor=env_preprocessor,
                                env_postprocessor=env_postprocessor,
                                preprocessor=preprocessor,
                                postprocessor=postprocessor,
                                n_episodes=rank_n_episodes,
                                videos_dir=cfg.output_dir
                                / "eval"
                                / f"videos_step_{step_id}"
                                / f"rank_{accelerator.process_index}",
                                max_episodes_rendered=4 if is_main_process else 0,
                                start_seed=(
                                    None
                                    if eval_start_seed is None
                                    else eval_start_seed + accelerator.process_index * cfg.eval.n_episodes
                                ),
                                max_parallel_tasks=rank_env_cfg.max_parallel_tasks,
                            )
                    finally:
                        if eval_env is not None:
                            close_envs(eval_env)
                        eval_env = None
                else:
                    eval_info = _empty_eval_info(accelerator.process_index, rank_task_ids)

                rank_info_dir.mkdir(parents=True, exist_ok=True)
                rank_info_path = rank_info_dir / f"rank_{accelerator.process_index}.json"
                rank_info_path.write_text(
                    json.dumps(
                        _json_safe(
                            {
                                "rank": accelerator.process_index,
                                "task_ids": rank_task_ids,
                                "eval_info": eval_info,
                            }
                        ),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                accelerator.wait_for_everyone()
                if is_main_process:
                    rank_infos = [
                        json.loads((rank_info_dir / f"rank_{rank}.json").read_text(encoding="utf-8"))
                        for rank in range(accelerator.num_processes)
                    ]
                    eval_info = _merge_distributed_eval_infos(rank_infos)
                    (rank_info_dir / "merged.json").write_text(
                        json.dumps(_json_safe(eval_info), ensure_ascii=False, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
            elif evaluation_runtime is not None:
                if is_main_process:
                    logging.info(f"Eval policy at step {step}")
                    logging.info("Creating eval env")
                    eval_env = make_env(
                        cfg.env,
                        n_envs=cfg.eval.batch_size,
                        use_async_envs=cfg.eval.use_async_envs,
                    )
                    try:
                        with torch.no_grad(), accelerator.autocast():
                            eval_info = run_policy_evaluation(
                                evaluation_runtime,
                                eval_policy_all,
                                envs=eval_env,
                                policy=accelerator.unwrap_model(policy),
                                env_preprocessor=env_preprocessor,
                                env_postprocessor=env_postprocessor,
                                preprocessor=preprocessor,
                                postprocessor=postprocessor,
                                n_episodes=cfg.eval.n_episodes,
                                videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                                max_episodes_rendered=4,
                                start_seed=eval_start_seed,
                                max_parallel_tasks=cfg.env.max_parallel_tasks,
                            )
                    finally:
                        if eval_env is not None:
                            close_envs(eval_env)
                        eval_env = None
            else:
                if accelerator.num_processes > 1 and eval_task_ids:
                    rank_task_ids = eval_task_ids[accelerator.process_index :: accelerator.num_processes]
                    rank_info_dir = cfg.output_dir / "eval" / f"rank_metrics_step_{step_id}"
                    if rank_task_ids:
                        rank_env_cfg = deepcopy(cfg.env)
                        rank_env_cfg.task_ids = rank_task_ids
                        logging.info(
                            "Rank %d/%d eval policy at step %s for task_ids=%s",
                            accelerator.process_index,
                            accelerator.num_processes,
                            step,
                            rank_task_ids,
                        )
                        logging.info("Creating eval env")
                        eval_env = make_env(
                            rank_env_cfg,
                            n_envs=cfg.eval.batch_size,
                            use_async_envs=cfg.eval.use_async_envs,
                        )
                        try:
                            with torch.no_grad(), accelerator.autocast():
                                eval_info = eval_policy_all(
                                    envs=eval_env,
                                    policy=accelerator.unwrap_model(policy),
                                    env_preprocessor=env_preprocessor,
                                    env_postprocessor=env_postprocessor,
                                    preprocessor=preprocessor,
                                    postprocessor=postprocessor,
                                    n_episodes=cfg.eval.n_episodes,
                                    videos_dir=cfg.output_dir
                                    / "eval"
                                    / f"videos_step_{step_id}"
                                    / f"rank_{accelerator.process_index}",
                                    max_episodes_rendered=4 if is_main_process else 0,
                                    start_seed=eval_start_seed,
                                    start_seed_task_ids=random_eval_seed_task_ids,
                                    max_parallel_tasks=cfg.env.max_parallel_tasks,
                                )
                        finally:
                            if eval_env is not None:
                                close_envs(eval_env)
                            eval_env = None
                    else:
                        eval_info = _empty_eval_info(accelerator.process_index, rank_task_ids)

                    rank_info_dir.mkdir(parents=True, exist_ok=True)
                    rank_info_path = rank_info_dir / f"rank_{accelerator.process_index}.json"
                    rank_info_path.write_text(
                        json.dumps(
                            _json_safe(
                                {
                                    "rank": accelerator.process_index,
                                    "task_ids": rank_task_ids,
                                    "eval_info": eval_info,
                                }
                            ),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        encoding="utf-8",
                    )
                    accelerator.wait_for_everyone()
                    if is_main_process:
                        rank_infos = [
                            json.loads(
                                (rank_info_dir / f"rank_{rank}.json").read_text(encoding="utf-8")
                            )
                            for rank in range(accelerator.num_processes)
                        ]
                        eval_info = _merge_distributed_eval_infos(rank_infos)
                        (rank_info_dir / "merged.json").write_text(
                            json.dumps(_json_safe(eval_info), ensure_ascii=False, indent=2, sort_keys=True),
                            encoding="utf-8",
                        )
                elif is_main_process:
                    logging.info(f"Eval policy at step {step}")
                    logging.info("Creating eval env")
                    eval_env = make_env(
                        cfg.env,
                        n_envs=cfg.eval.batch_size,
                        use_async_envs=cfg.eval.use_async_envs,
                    )
                    try:
                        with torch.no_grad(), accelerator.autocast():
                            eval_info = eval_policy_all(
                                envs=eval_env,
                                policy=accelerator.unwrap_model(policy),
                                env_preprocessor=env_preprocessor,
                                env_postprocessor=env_postprocessor,
                                preprocessor=preprocessor,
                                postprocessor=postprocessor,
                                n_episodes=cfg.eval.n_episodes,
                                videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                                max_episodes_rendered=4,
                                start_seed=eval_start_seed,
                                start_seed_task_ids=random_eval_seed_task_ids,
                                max_parallel_tasks=cfg.env.max_parallel_tasks,
                            )
                    finally:
                        if eval_env is not None:
                            close_envs(eval_env)
                        eval_env = None

            if is_main_process:
                if eval_info is None:
                    raise RuntimeError("Main process did not produce eval metrics.")
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]
                eval_score = (float(aggregated["pc_success"]), float(aggregated["avg_sum_reward"]))

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                if cfg.save_checkpoint:
                    checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                    is_new_best = best_eval_score is None or eval_score > best_eval_score
                    if is_new_best:
                        logging.info(
                            "New best eval checkpoint at step %s: pc_success=%.1f, avg_sum_reward=%.3f",
                            step,
                            eval_score[0],
                            eval_score[1],
                        )
                    else:
                        logging.info(
                            "Eval did not improve best checkpoint: current pc_success=%.1f, "
                            "avg_sum_reward=%.3f; best pc_success=%.1f, avg_sum_reward=%.3f",
                            eval_score[0],
                            eval_score[1],
                            best_eval_score[0],
                            best_eval_score[1],
                        )
                    # Keep eval and resume checkpoints separate: full checkpoints
                    # carry a large optimizer state, while best-only eval points
                    # only need policy artifacts for later inference.
                    should_save_full_checkpoint = is_saving_step or last_checkpoint_dir is None
                    if should_save_full_checkpoint:
                        prune_checkpoint_training_state_if_not_last(best_checkpoint_dir, last_checkpoint_dir)
                        save_checkpoint(
                            checkpoint_dir=checkpoint_dir,
                            step=step,
                            cfg=cfg,
                            policy=accelerator.unwrap_model(policy),
                            optimizer=optimizer,
                            scheduler=lr_scheduler,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                        )
                        update_last_checkpoint(checkpoint_dir)
                        last_checkpoint_dir = checkpoint_dir
                        prune_checkpoint_training_state_if_not_last(best_checkpoint_dir, last_checkpoint_dir)
                    elif is_new_best:
                        prepare_checkpoint_dir_for_save(checkpoint_dir)
                        save_policy_checkpoint(
                            checkpoint_dir=checkpoint_dir,
                            cfg=cfg,
                            policy=accelerator.unwrap_model(policy),
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                        )
                    if is_new_best:
                        update_best_checkpoint(checkpoint_dir)
                        best_eval_score = eval_score
                        best_checkpoint_dir = checkpoint_dir
                        if wandb_logger:
                            wandb_logger.log_policy(checkpoint_dir)
                    if should_save_full_checkpoint or is_new_best:
                        prune_checkpoints_keep(
                            checkpoint_dir.parent,
                            keep_checkpoint_dirs=[best_checkpoint_dir, last_checkpoint_dir],
                            keep_scheduled_from_step=cfg.keep_all_checkpoints_after_step,
                            save_freq=cfg.save_freq,
                            total_steps=cfg.steps,
                        )

                eval_timestamp = time.time()
                append_jsonl(
                    eval_log_path,
                    {
                        "mode": "eval",
                        "step": step,
                        "time": eval_timestamp,
                        "metrics": eval_info,
                    },
                )
                try:
                    shared_record_path = append_tmux_eval_result(
                        cfg=cfg,
                        step=step,
                        eval_time=eval_timestamp,
                        eval_info=eval_info,
                    )
                    if shared_record_path is not None:
                        logging.info("Recorded tmux eval result in %s", shared_record_path)
                except Exception:
                    # A shared-ledger failure must never interrupt model training. The complete
                    # result remains available in the run-local eval_metrics.jsonl as a fallback.
                    logging.exception("Failed to record tmux eval result in the shared ledger")

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    video_paths = eval_info["overall"].get("video_paths", [])
                    if video_paths:
                        wandb_logger.log_video(video_paths[0], step, mode="eval")

            accelerator.wait_for_everyone()

    if is_main_process:
        progbar.close()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if getattr(active_cfg, "push_to_hub", False):
            unwrapped_model = accelerator.unwrap_model(policy)
            # PEFT only applies when training a policy — reward models use the plain path.
            if not cfg.is_reward_model_training and cfg.policy.use_peft:
                unwrapped_model.push_model_to_hub(cfg, peft_model=unwrapped_model)
            else:
                unwrapped_model.push_model_to_hub(cfg)
            preprocessor.push_to_hub(active_cfg.repo_id)
            postprocessor.push_to_hub(active_cfg.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
