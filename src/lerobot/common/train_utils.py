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
import json
import logging
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lerobot.configs.train import TrainPipelineConfig
from lerobot.optim import (
    load_optimizer_state,
    load_scheduler_state,
    save_optimizer_state,
    save_scheduler_state,
)
from lerobot.policies import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
    TRAINING_STEP,
)
from lerobot.utils.io_utils import load_json, write_json
from lerobot.utils.random_utils import load_rng_state, save_rng_state


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


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True) + "\n")


def trim_jsonl_after_step(path: Path, max_step: int) -> list[dict[str, Any]]:
    """Drop stale records written after the checkpoint used for resume."""
    if not path.exists():
        return []

    kept_lines: list[str] = []
    kept_records: list[dict[str, Any]] = []
    dropped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            record_step = int(record["step"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            dropped += 1
            continue
        if record_step <= max_step:
            kept_lines.append(line)
            kept_records.append(record)
        else:
            dropped += 1

    if dropped:
        tmp_path = path.with_suffix(f"{path.suffix}.resume_tmp")
        tmp_path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")
        tmp_path.replace(path)
        logging.info("Resume trimmed %d stale/malformed record(s) from %s", dropped, path)
    return kept_records


def restore_resume_eval_state(
    eval_records: list[dict[str, Any]],
    *,
    output_dir: Path,
    total_steps: int,
    checkpoint_path: Path | None,
) -> tuple[tuple[float, float] | None, Path | None]:
    """Restore best-eval bookkeeping when resuming a run."""
    scored_records: list[tuple[tuple[float, float], int]] = []
    for record in eval_records:
        overall = record.get("metrics", {}).get("overall", {})
        try:
            score = (float(overall["pc_success"]), float(overall.get("avg_sum_reward", 0.0)))
            scored_records.append((score, int(record["step"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not scored_records:
        return None, None

    best_score, best_step = max(scored_records, key=lambda item: item[0])
    best_checkpoint_dir = get_step_checkpoint_dir(output_dir, total_steps, best_step)
    if not best_checkpoint_dir.exists():
        best_link = output_dir / "checkpoints" / "best"
        if best_link.exists():
            best_checkpoint_dir = best_link.resolve()
        elif checkpoint_path is not None and checkpoint_path.exists():
            best_checkpoint_dir = checkpoint_path
        else:
            logging.warning(
                "Resume recovered best eval score %s at step %d, but no corresponding checkpoint exists.",
                best_score,
                best_step,
            )
            best_checkpoint_dir = None
    return best_score, best_checkpoint_dir


def get_step_identifier(step: int, total_steps: int) -> str:
    num_digits = max(6, len(str(total_steps)))
    return f"{step:0{num_digits}d}"


def get_step_checkpoint_dir(output_dir: Path, total_steps: int, step: int) -> Path:
    """Returns the checkpoint sub-directory corresponding to the step number."""
    step_identifier = get_step_identifier(step, total_steps)
    return output_dir / CHECKPOINTS_DIR / step_identifier


def save_training_step(step: int, save_dir: Path) -> None:
    write_json({"step": step}, save_dir / TRAINING_STEP)


def load_training_step(save_dir: Path) -> int:
    training_step = load_json(save_dir / TRAINING_STEP)
    return training_step["step"]


def update_last_checkpoint(checkpoint_dir: Path) -> Path:
    last_checkpoint_dir = checkpoint_dir.parent / LAST_CHECKPOINT_LINK
    if last_checkpoint_dir.is_symlink():
        last_checkpoint_dir.unlink()
    relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
    last_checkpoint_dir.symlink_to(relative_target)


def update_best_checkpoint(checkpoint_dir: Path) -> Path:
    best_checkpoint_dir = checkpoint_dir.parent / "best"
    if best_checkpoint_dir.is_symlink():
        best_checkpoint_dir.unlink()
    elif best_checkpoint_dir.exists():
        shutil.rmtree(best_checkpoint_dir)
    relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
    best_checkpoint_dir.symlink_to(relative_target)
    return best_checkpoint_dir


def prune_checkpoints_keep(checkpoints_dir: Path, keep_checkpoint_dirs: Iterable[Path | None]) -> None:
    if not checkpoints_dir.exists():
        return

    keep_names = {
        checkpoint_dir.name for checkpoint_dir in keep_checkpoint_dirs if checkpoint_dir is not None
    }
    for child in checkpoints_dir.iterdir():
        if child.name in {LAST_CHECKPOINT_LINK, "best"} or child.name in keep_names:
            continue
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def prune_checkpoints(checkpoints_dir: Path, keep_checkpoint_dir: Path | None) -> None:
    prune_checkpoints_keep(checkpoints_dir, keep_checkpoint_dirs=[keep_checkpoint_dir])


def prune_checkpoint_training_state(checkpoint_dir: Path | None) -> None:
    if checkpoint_dir is None:
        return

    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR
    if training_state_dir.exists():
        shutil.rmtree(training_state_dir)


def prune_checkpoint_training_state_if_not_last(
    checkpoint_dir: Path | None, last_checkpoint_dir: Path | None
) -> None:
    if checkpoint_dir is None or last_checkpoint_dir is None:
        return
    if checkpoint_dir.resolve(strict=False) == last_checkpoint_dir.resolve(strict=False):
        return

    prune_checkpoint_training_state(checkpoint_dir)


def prepare_checkpoint_dir_for_save(checkpoint_dir: Path) -> None:
    if checkpoint_dir.exists():
        if checkpoint_dir.is_symlink() or checkpoint_dir.is_file():
            checkpoint_dir.unlink()
        elif checkpoint_dir.is_dir():
            shutil.rmtree(checkpoint_dir)


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    cfg: TrainPipelineConfig,
    policy: PreTrainedPolicy,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    preprocessor: PolicyProcessorPipeline | None = None,
    postprocessor: PolicyProcessorPipeline | None = None,
) -> None:
    """Save a full resumable checkpoint: policy, processors, optimizer, scheduler, RNG, and step."""
    prepare_checkpoint_dir_for_save(checkpoint_dir)
    save_policy_checkpoint(
        checkpoint_dir=checkpoint_dir,
        cfg=cfg,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )
    save_training_state(checkpoint_dir, step, optimizer, scheduler)


def save_policy_checkpoint(
    checkpoint_dir: Path,
    cfg: TrainPipelineConfig,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline | None = None,
    postprocessor: PolicyProcessorPipeline | None = None,
) -> None:
    """Save policy and processor artifacts without optimizer/scheduler training state.

    005000/  #  training step at checkpoint
    ├── pretrained_model/
    │   ├── config.json  # policy config
    │   ├── model.safetensors  # policy weights
    │   ├── train_config.json  # train config
    │   ├── processor.json  # processor config (if preprocessor provided)
    │   └── step_*.safetensors  # processor state files (if any)

    Args:
        cfg (TrainPipelineConfig): The training config used for this run.
        policy (PreTrainedPolicy): The policy to save.
        preprocessor: The preprocessor/pipeline to save. Defaults to None.
        postprocessor: The postprocessor/pipeline to save. Defaults to None.
    """
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    policy.save_pretrained(pretrained_dir)
    cfg.save_pretrained(pretrained_dir)
    if cfg.peft is not None:
        # When using PEFT, policy.save_pretrained will only write the adapter weights + config, not the
        # policy config which we need for loading the model. In this case we'll write it ourselves.
        policy.config.save_pretrained(pretrained_dir)
    if preprocessor is not None:
        preprocessor.save_pretrained(pretrained_dir)
    if postprocessor is not None:
        postprocessor.save_pretrained(pretrained_dir)


def save_training_state(
    checkpoint_dir: Path,
    train_step: int,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
) -> None:
    """
    Saves the training step, optimizer state, scheduler state, and rng state.

    Args:
        save_dir (Path): The directory to save artifacts to.
        train_step (int): Current training step.
        optimizer (Optimizer | None, optional): The optimizer from which to save the state_dict.
            Defaults to None.
        scheduler (LRScheduler | None, optional): The scheduler from which to save the state_dict.
            Defaults to None.
    """
    save_dir = checkpoint_dir / TRAINING_STATE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    save_training_step(train_step, save_dir)
    save_rng_state(save_dir)
    if optimizer is not None:
        save_optimizer_state(optimizer, save_dir)
    if scheduler is not None:
        save_scheduler_state(scheduler, save_dir)


def load_training_state(
    checkpoint_dir: Path, optimizer: Optimizer, scheduler: LRScheduler | None
) -> tuple[int, Optimizer, LRScheduler | None]:
    """
    Loads the training step, optimizer state, scheduler state, and rng state.
    This is used to resume a training run.

    Args:
        checkpoint_dir (Path): The checkpoint directory. Should contain a 'training_state' dir.
        optimizer (Optimizer): The optimizer to load the state_dict to.
        scheduler (LRScheduler | None): The scheduler to load the state_dict to (can be None).

    Raises:
        NotADirectoryError: If 'checkpoint_dir' doesn't contain a 'training_state' dir

    Returns:
        tuple[int, Optimizer, LRScheduler | None]: training step, optimizer and scheduler with their
            state_dict loaded.
    """
    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR
    if not training_state_dir.is_dir():
        raise NotADirectoryError(training_state_dir)

    load_rng_state(training_state_dir)
    step = load_training_step(training_state_dir)
    optimizer = load_optimizer_state(optimizer, training_state_dir)
    if scheduler is not None:
        scheduler = load_scheduler_state(scheduler, training_state_dir)

    return step, optimizer, scheduler
