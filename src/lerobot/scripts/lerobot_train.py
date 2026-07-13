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
import time
from contextlib import nullcontext
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
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    prune_checkpoints_keep,
    save_checkpoint,
    update_best_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import EpisodeAwareSampler, make_dataset
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.rewards import make_reward_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    cycle,
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)

from .lerobot_eval import (
    configure_fixed_libero_eval_from_dataset,
    eval_policy_all,
    validate_libero_action_semantics,
)

LIBERO_INIT_STATE_ID_KEYS = (
    "libero/init_state_id",
    "init_state_id",
)
LIBERO_INIT_STATE_VALUE_KEYS = (
    "libero/init_state",
    "init_state",
)
LIBERO_TASK_ID_KEYS = (
    "libero/task_id",
    "task_id",
)
LIBERO_SUITE_KEYS = (
    "libero/suite",
    "suite",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "__fspath__"):
        return str(value)
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def _append_jsonl(path: Any, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True) + "\n")


def _trim_jsonl_after_step(path: Path, max_step: int) -> list[dict[str, Any]]:
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


def _restore_resume_eval_state(
    eval_records: list[dict[str, Any]],
    *,
    output_dir: Path,
    total_steps: int,
    checkpoint_path: Path | None,
) -> tuple[tuple[float, float] | None, Path | None]:
    """Restore best-eval bookkeeping instead of treating resume as a fresh run."""
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


def apply_overfit_test_config(cfg: TrainPipelineConfig) -> None:
    """Restrict training/eval to the first fixed demos for pipeline debugging."""
    if not cfg.overfit_test:
        return
    if cfg.overfit_per_task:
        cfg.eval.n_episodes = cfg.num_overfit_per_task
        cfg.eval.batch_size = (
            min(cfg.eval.batch_size, cfg.num_overfit_per_task)
            if cfg.eval.batch_size
            else cfg.num_overfit_per_task
        )
        logging.info(
            "Task-aware overfit test enabled: selecting %d episode(s) per task.",
            cfg.num_overfit_per_task,
        )
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


def _metadata_column_names(episodes: Any) -> set[str]:
    return set(getattr(episodes, "column_names", []) or [])


def _first_existing_column(columns: set[str], candidates: tuple[str, ...]) -> str | None:
    return next((key for key in candidates if key in columns), None)


def _row_get(row: Any, key: str | None, default: Any = None) -> Any:
    if key is None:
        return default
    try:
        return row[key]
    except (KeyError, TypeError):
        return default


def _as_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        return None
    return [float(item) for item in value]


def apply_overfit_per_task_episode_selection(cfg: TrainPipelineConfig) -> None:
    """Select fixed demos by LIBERO task before constructing the training dataset."""
    if not cfg.overfit_test or not cfg.overfit_per_task:
        return

    from lerobot.datasets import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision)
    columns = _metadata_column_names(meta.episodes)
    task_id_key = _first_existing_column(columns, LIBERO_TASK_ID_KEYS)
    if task_id_key is None:
        raise ValueError(
            f"overfit_per_task=True requires episode metadata with one of: {', '.join(LIBERO_TASK_ID_KEYS)}"
        )
    suite_key = _first_existing_column(columns, LIBERO_SUITE_KEYS)

    candidates = set(cfg.dataset.episodes) if cfg.dataset.episodes is not None else None
    rows_by_task: dict[int, list[tuple[int, str]]] = {}
    for row in meta.episodes:
        episode_index = int(row["episode_index"])
        if candidates is not None and episode_index not in candidates:
            continue
        task_id = int(row[task_id_key])
        suite = str(_row_get(row, suite_key, getattr(cfg.env, "task", "")))
        rows_by_task.setdefault(task_id, []).append((episode_index, suite))

    if not rows_by_task:
        raise ValueError("overfit_per_task=True did not match any dataset episodes.")

    selected: list[int] = []
    suites: set[str] = set()
    for task_id in sorted(rows_by_task):
        rows = sorted(rows_by_task[task_id], key=lambda item: item[0])
        if len(rows) < cfg.num_overfit_per_task:
            raise ValueError(
                f"Task {task_id} has only {len(rows)} episode(s), cannot select {cfg.num_overfit_per_task}."
            )
        chosen = rows[: cfg.num_overfit_per_task]
        selected.extend(ep for ep, _ in chosen)
        suites.update(suite for _, suite in chosen if suite)

    cfg.dataset.episodes = selected
    if cfg.env is not None and getattr(cfg.env, "type", None) in {"libero", "libero_plus"}:
        if hasattr(cfg.env, "task_ids"):
            cfg.env.task_ids = sorted(rows_by_task)
        if len(suites) == 1 and hasattr(cfg.env, "task"):
            cfg.env.task = next(iter(suites))
    logging.info("Task-aware overfit selected dataset episodes=%s", selected)


def apply_overfit_eval_init_state_ids(cfg: TrainPipelineConfig, dataset: Any) -> None:
    """Read fixed LIBERO init states from the selected training demos and pass them to eval envs."""
    if not cfg.overfit_test or cfg.env is None:
        return
    if getattr(cfg.env, "type", None) not in {"libero", "libero_plus"}:
        return
    if not hasattr(cfg.env, "init_state_ids") and not hasattr(cfg.env, "init_state_values"):
        logging.warning(
            "overfit_test=True but env has no fixed init-state fields; skip fixed init-state wiring."
        )
        return

    overfit_episodes = cfg.dataset.episodes or list(range(cfg.num_overfit))
    episode_rows = {int(row["episode_index"]): row for row in dataset.meta.episodes}
    missing = [ep for ep in overfit_episodes if ep not in episode_rows]
    if missing:
        raise ValueError(f"Overfit episodes missing from dataset metadata: {missing}")

    column_names = _metadata_column_names(dataset.meta.episodes)
    init_state_value_key = _first_existing_column(column_names, LIBERO_INIT_STATE_VALUE_KEYS)
    init_state_key = _first_existing_column(column_names, LIBERO_INIT_STATE_ID_KEYS)
    if init_state_value_key is None and init_state_key is None:
        logging.warning(
            "Dataset episode metadata has no LIBERO init-state column (%s) or init-state id column (%s). "
            "Falling back to episode_index as init_state_id.",
            ", ".join(LIBERO_INIT_STATE_VALUE_KEYS),
            ", ".join(LIBERO_INIT_STATE_ID_KEYS),
        )

    if cfg.overfit_per_task:
        task_id_key = _first_existing_column(column_names, LIBERO_TASK_ID_KEYS)
        if task_id_key is None:
            raise ValueError(
                "overfit_per_task=True requires episode metadata with one of: "
                f"{', '.join(LIBERO_TASK_ID_KEYS)}"
            )
        suite_key = _first_existing_column(column_names, LIBERO_SUITE_KEYS)
        rows_by_task: dict[int, list[Any]] = {}
        for episode_index in overfit_episodes:
            row = episode_rows[int(episode_index)]
            rows_by_task.setdefault(int(row[task_id_key]), []).append(row)

        init_state_values_by_task: dict[str, list[list[float]]] = {}
        init_state_ids_by_task: dict[str, list[int]] = {}
        task_ids: set[int] = set()
        eval_episode_ids_by_task: dict[int, list[int]] = {}
        for task_id in sorted(rows_by_task):
            rows = sorted(rows_by_task[task_id], key=lambda item: int(item["episode_index"]))
            if len(rows) < cfg.num_overfit_per_task:
                raise ValueError(
                    f"Task {task_id} has only {len(rows)} selected training episode(s) for eval; "
                    f"need {cfg.num_overfit_per_task}."
                )
            chosen = rows[: cfg.num_overfit_per_task]
            eval_episode_ids_by_task[task_id] = [int(row["episode_index"]) for row in chosen]
            for row in chosen:
                suite = str(_row_get(row, suite_key, getattr(cfg.env, "task", "libero_10")))
                task_key = f"{suite}/{task_id}"
                if init_state_value_key is not None and hasattr(cfg.env, "init_state_values_by_task"):
                    init_state_value = _as_float_list(_row_get(row, init_state_value_key))
                    if init_state_value is None:
                        raise ValueError(
                            f"Episode {int(row['episode_index'])} has invalid {init_state_value_key}."
                        )
                    init_state_values_by_task.setdefault(task_key, []).append(init_state_value)
                else:
                    init_state_id = (
                        int(row[init_state_key]) if init_state_key is not None else int(row["episode_index"])
                    )
                    init_state_ids_by_task.setdefault(task_key, []).append(init_state_id)
            task_ids.add(task_id)
        cfg.env.task_ids = sorted(task_ids)
        if init_state_values_by_task:
            cfg.env.init_state_values_by_task = init_state_values_by_task
            if hasattr(cfg.env, "init_state_ids_by_task"):
                cfg.env.init_state_ids_by_task = None
            if hasattr(cfg.env, "num_steps_wait"):
                cfg.env.num_steps_wait = 0
            logging.info(
                "Overfit eval fixed LIBERO raw init_state_values_by_task for %d task(s).",
                len(init_state_values_by_task),
            )
        elif hasattr(cfg.env, "init_state_ids_by_task"):
            cfg.env.init_state_ids_by_task = init_state_ids_by_task
            logging.info("Overfit eval fixed LIBERO init_state_ids_by_task=%s", init_state_ids_by_task)
        logging.info(
            "Task-aware overfit eval uses selected training dataset episodes by task=%s",
            eval_episode_ids_by_task,
        )
        return

    if init_state_value_key is not None and hasattr(cfg.env, "init_state_values"):
        init_state_values = []
        for ep in overfit_episodes:
            init_state_value = _as_float_list(_row_get(episode_rows[ep], init_state_value_key))
            if init_state_value is None:
                raise ValueError(f"Episode {ep} has invalid {init_state_value_key}.")
            init_state_values.append(init_state_value)
        cfg.env.init_state_values = init_state_values
        if hasattr(cfg.env, "init_state_ids"):
            cfg.env.init_state_ids = None
        if hasattr(cfg.env, "num_steps_wait"):
            cfg.env.num_steps_wait = 0
        logging.info(
            "Overfit eval fixed LIBERO raw init_state_values from %d selected demo(s).",
            len(init_state_values),
        )
        return

    init_state_ids = [
        (
            int(episode_rows[ep][init_state_key])
            if init_state_key is not None
            else int(episode_rows[ep]["episode_index"])
        )
        for ep in overfit_episodes
    ]
    cfg.env.init_state_ids = init_state_ids
    logging.info("Overfit eval fixed LIBERO init_state_ids=%s", init_state_ids)


def apply_diffusion_relative_action_stats(cfg: TrainPipelineConfig, dataset: Any) -> None:
    """Use stats from the same chunk-relative action space consumed by DP/MAM."""
    active_cfg = cfg.trainable_config
    policy_type = getattr(active_cfg, "type", None)
    if policy_type not in {"diffusion", "mam"}:
        return
    if policy_type == "diffusion" and not getattr(active_cfg, "use_relative_actions", False):
        return
    if ACTION not in dataset.meta.features or OBS_STATE not in dataset.meta.features:
        raise ValueError(
            "Diffusion use_relative_actions=True requires action and observation.state features."
        )

    from lerobot.datasets.compute_stats import compute_libero_relative_action_stats

    dataset.meta.stats = dataset.meta.stats or {}
    dataset.meta.stats[ACTION] = compute_libero_relative_action_stats(
        hf_dataset=dataset.hf_dataset,
        action_delta_indices=active_cfg.action_delta_indices,
        num_workers=cfg.num_workers,
    )
    logging.info("Using LIBERO chunk-relative action stats for %s normalization.", policy_type)


def validate_libero_v3_training_dataset(cfg: TrainPipelineConfig, dataset: Any) -> None:
    """Certify the dataset boundary for LIBERO chunk-relative Diffusion training."""
    active_cfg = cfg.trainable_config
    if getattr(active_cfg, "type", None) != "diffusion" or not getattr(
        active_cfg, "use_relative_actions", False
    ):
        return
    if getattr(dataset.meta, "robot_type", None) != "libero":
        return

    from lerobot.datasets.libero_pipeline import (
        LIBERO_ABSOLUTE_ACTION,
        LIBERO_CHUNK_RELATIVE_ACTION,
        require_libero_v3_action_dataset,
    )

    train_root_value = getattr(dataset, "root", None) or cfg.dataset.root
    if train_root_value is None:
        raise ValueError("LIBERO v3 relative-action training requires an explicit dataset.root.")
    train_root = Path(train_root_value)
    train_manifest = require_libero_v3_action_dataset(
        train_root,
        action_representation=LIBERO_ABSOLUTE_ACTION,
    )
    if (
        train_manifest.get("stage") != "absolute_to_mam"
        or train_manifest.get("dataset_split") != "train"
        or train_manifest.get("policy_action_representation") != LIBERO_CHUNK_RELATIVE_ACTION
    ):
        raise ValueError(
            "LIBERO v3 relative-action training requires an absolute_to_mam/train dataset "
            "with chunk-relative SE(3) policy actions."
        )

    eval_repo_id = getattr(cfg.eval, "dataset_repo_id", None)
    eval_root_value = getattr(cfg.eval, "dataset_root", None)
    eval_episodes = getattr(cfg.eval, "dataset_episodes", None)
    if cfg.overfit_test:
        same_root = eval_root_value is not None and Path(eval_root_value).resolve() == train_root.resolve()
        if (
            eval_repo_id != cfg.dataset.repo_id
            or not same_root
            or list(eval_episodes or []) != list(cfg.dataset.episodes or [])
        ):
            raise ValueError(
                "LIBERO v3 overfit requires eval repo/root/episodes to be exactly identical "
                "to the selected training trajectories."
            )
        return

    if (
        cfg.env is not None
        and cfg.eval_freq > 0
        and getattr(cfg.env, "type", None) == "libero"
        and (eval_repo_id is None or eval_root_value is None)
    ):
        raise ValueError(
            "Normal LIBERO v3 online evaluation requires an explicit eval dataset repo/root."
        )
    if eval_repo_id is None:
        return
    if eval_root_value is None:
        raise ValueError("LIBERO v3 fixed evaluation requires an explicit eval.dataset_root.")
    eval_manifest = require_libero_v3_action_dataset(
        eval_root_value,
        action_representation=LIBERO_ABSOLUTE_ACTION,
    )
    if (
        eval_manifest.get("stage") != "absolute_to_mam"
        or eval_manifest.get("dataset_split") != "eval"
        or eval_manifest.get("policy_action_representation") != LIBERO_CHUNK_RELATIVE_ACTION
    ):
        raise ValueError(
            "Normal LIBERO v3 evaluation requires an absolute_to_mam/eval dataset "
            "with chunk-relative SE(3) policy actions."
        )


def apply_overfit_subset_stats(cfg: TrainPipelineConfig, dataset: Any) -> None:
    """Recompute numeric policy-feature stats on the episodes used by an overfit run.

    ``LeRobotDataset(..., episodes=...)`` filters frames but intentionally keeps the
    repository-wide metadata stats. That is desirable for normal training subsets,
    but undermines a strict overfit diagnostic: normalization can still be dominated
    by tasks and episodes that are not being trained. Visual stats are left unchanged
    (in particular, ImageNet normalization remains ImageNet normalization).
    """
    if not cfg.overfit_test or cfg.dataset.episodes is None:
        return
    if cfg.dataset.streaming:
        raise ValueError("overfit_test=True does not support subset-stat recomputation for streaming data.")

    active_cfg = cfg.trainable_config
    policy_features = {
        **(getattr(active_cfg, "input_features", None) or {}),
        **(getattr(active_cfg, "output_features", None) or {}),
    }
    feature_keys = set(policy_features)
    if not feature_keys:
        # Fresh CLI configs receive input/output feature objects later, inside
        # make_policy(). At this earlier stage derive numeric candidates from the
        # dataset schema instead of silently skipping subset normalization.
        feature_keys = set(dataset.meta.features)
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
    validate_libero_action_semantics(cfg)
    apply_overfit_test_config(cfg)
    apply_overfit_per_task_episode_selection(cfg)

    # LIBERO env construction is lazy, so missing simulator assets would otherwise
    # surface only at the first evaluation step after potentially hours of training.
    if (
        cfg.env is not None
        and cfg.eval_freq > 0
        and getattr(cfg.env, "type", None) in {"libero", "libero_plus"}
    ):
        from lerobot.envs.libero_assets import validate_libero_assets

        validate_libero_assets()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # Accelerate auto-detects the device based on the available hardware and ignores the policy.device setting.
        # Force the device to be CPU when the active config's device is set to CPU (works for both policy and reward model training).
        force_cpu = cfg.trainable_config.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
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

    validate_libero_v3_training_dataset(cfg, dataset)
    apply_overfit_subset_stats(cfg, dataset)
    apply_diffusion_relative_action_stats(cfg, dataset)

    if is_main_process:
        apply_overfit_eval_init_state_ids(cfg, dataset)
        mam_eval_episodes = None
        if getattr(cfg.trainable_config, "type", None) != "mam" and cfg.env is not None and cfg.eval_freq > 0:
            configure_fixed_libero_eval_from_dataset(cfg)
        mam_online_eval = (
            getattr(cfg.trainable_config, "type", None) == "mam" and cfg.env is not None and cfg.eval_freq > 0
        )
        if mam_online_eval and not getattr(cfg.trainable_config, "mam_eval_dataset_repo_id", None):
            raise ValueError(
                "MAM online eval requires policy.mam_eval_dataset_repo_id; generic eval cannot provide "
                "aligned MAS, progress, task-specific STPM, or fixed LIBERO init states."
            )
        if mam_online_eval:
            from lerobot.policies.mam.eval_mam import (
                configure_mam_eval_init_state_ids,
                load_mam_eval_episodes,
            )

            mam_eval_episodes = load_mam_eval_episodes(
                repo_id=cfg.trainable_config.mam_eval_dataset_repo_id,
                root=getattr(cfg.trainable_config, "mam_eval_dataset_root", None),
                episodes=getattr(cfg.trainable_config, "mam_eval_episodes", None),
            )
            configure_mam_eval_init_state_ids(cfg, mam_eval_episodes, cfg.eval.n_episodes)
            logging.info(
                "MAM eval fixed LIBERO init states: ids=%s ids_by_task=%s",
                getattr(cfg.env, "init_state_ids", None),
                getattr(cfg.env, "init_state_ids_by_task", None),
            )
    else:
        mam_eval_episodes = None

    # Create evaluation environments lazily at each eval step. `eval_policy_all`
    # closes the vector envs it receives, so reusing one across eval steps would
    # leave LIBERO's wrapped robosuite env in a partially closed state.
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

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
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
    if is_main_process:
        train_log_path = cfg.output_dir / "logs" / "train_metrics.jsonl"
        eval_log_path = cfg.output_dir / "logs" / "eval_metrics.jsonl"
        if cfg.resume:
            _trim_jsonl_after_step(train_log_path, step)
            eval_records = _trim_jsonl_after_step(eval_log_path, step)
            best_eval_score, best_checkpoint_dir = _restore_resume_eval_state(
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
            _append_jsonl(
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
                if cfg.env is not None and cfg.eval_freq > 0:
                    prune_checkpoints_keep(
                        checkpoint_dir.parent,
                        keep_checkpoint_dirs=[best_checkpoint_dir, checkpoint_dir],
                    )
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                logging.info("Creating eval env")
                eval_env = make_env(
                    cfg.env,
                    n_envs=cfg.eval.batch_size,
                    use_async_envs=cfg.eval.use_async_envs,
                )
                try:
                    with torch.no_grad(), accelerator.autocast():
                        if mam_eval_episodes is not None:
                            from lerobot.policies.mam.eval_mam import eval_mam_policy_all

                            eval_info = eval_mam_policy_all(
                                envs=eval_env,
                                policy=accelerator.unwrap_model(policy),
                                env_preprocessor=env_preprocessor,
                                env_postprocessor=env_postprocessor,
                                preprocessor=preprocessor,
                                postprocessor=postprocessor,
                                episodes=mam_eval_episodes,
                                n_episodes=cfg.eval.n_episodes,
                                start_seed=cfg.seed,
                            )
                        else:
                            eval_info = eval_policy_all(
                                envs=eval_env,  # dict[suite][task_id] -> vec_env
                                policy=accelerator.unwrap_model(policy),
                                env_preprocessor=env_preprocessor,
                                env_postprocessor=env_postprocessor,
                                preprocessor=preprocessor,
                                postprocessor=postprocessor,
                                n_episodes=cfg.eval.n_episodes,
                                videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                                max_episodes_rendered=4,
                                start_seed=cfg.seed,
                                max_parallel_tasks=cfg.env.max_parallel_tasks,
                            )
                finally:
                    if eval_env is not None:
                        close_envs(eval_env)
                    eval_env = None
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
                    # Persist before publishing the eval record: the incremental
                    # runner uses that record as its signal to stop a successful run.
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
                    if is_new_best:
                        update_best_checkpoint(checkpoint_dir)
                        best_eval_score = eval_score
                        best_checkpoint_dir = checkpoint_dir
                        if wandb_logger:
                            wandb_logger.log_policy(checkpoint_dir)
                    prune_checkpoints_keep(
                        checkpoint_dir.parent,
                        keep_checkpoint_dirs=[best_checkpoint_dir, checkpoint_dir],
                    )

                _append_jsonl(
                    eval_log_path,
                    {
                        "mode": "eval",
                        "step": step,
                        "time": time.time(),
                        "metrics": eval_info,
                    },
                )

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
