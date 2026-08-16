from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.constants import ACTION, OBS_STATE

from .libero_pipeline import (
    LIBERO_CHUNK_RELATIVE_ACTION,
    read_libero_pipeline_manifest,
    require_libero_v3_relative_ready_dataset,
)

LIBERO_INIT_STATE_ID_KEYS = ("libero/init_state_id", "init_state_id")
LIBERO_INIT_STATE_VALUE_KEYS = ("libero/init_state", "init_state")
LIBERO_TASK_ID_KEYS = ("libero/task_id", "task_id")
LIBERO_SUITE_KEYS = ("libero/suite", "suite")


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
    """Select fixed demos by LIBERO task before constructing the dataset."""
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
    """Use selected training demos as fixed LIBERO evaluation resets."""
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

    cfg.env.init_state_ids = [
        int(episode_rows[ep][init_state_key])
        if init_state_key is not None
        else int(episode_rows[ep]["episode_index"])
        for ep in overfit_episodes
    ]
    logging.info("Overfit eval fixed LIBERO init_state_ids=%s", cfg.env.init_state_ids)


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

    dataset_root = getattr(dataset, "root", None) or getattr(dataset.meta, "root", None)
    if not getattr(cfg, "overfit_test", False) and dataset_root is not None:
        try:
            manifest = read_libero_pipeline_manifest(dataset_root)
        except FileNotFoundError:
            manifest = {}
        expected_indices = [int(value) for value in active_cfg.action_delta_indices]
        certified_indices = manifest.get("relative_action_stats_action_delta_indices")
        if manifest.get("relative_action_stats") is True and certified_indices is not None:
            certified_indices = [int(value) for value in certified_indices]
            if certified_indices != expected_indices:
                raise ValueError(
                    "Dataset relative action stats do not match the policy horizon: "
                    f"dataset={certified_indices}, policy={expected_indices}."
                )
            action_stats = (dataset.meta.stats or {}).get(ACTION)
            required_stats = {"mean", "std", "min", "max", "q01", "q99"}
            if action_stats is None or not required_stats.issubset(action_stats):
                raise ValueError(
                    "Dataset manifest certifies relative action stats but meta/stats.json is incomplete."
                )
            logging.info("Using precomputed audited LIBERO chunk-relative action stats.")
            return

    from lerobot.datasets.compute_stats import compute_libero_relative_action_stats

    dataset.meta.stats = dataset.meta.stats or {}
    dataset.meta.stats[ACTION] = compute_libero_relative_action_stats(
        hf_dataset=dataset.hf_dataset,
        action_delta_indices=active_cfg.action_delta_indices,
        num_workers=cfg.num_workers,
    )
    logging.info("Using LIBERO chunk-relative action stats for %s normalization.", policy_type)


def _validate_libero_split_provenance(
    train_manifest: dict[str, Any],
    eval_manifest: dict[str, Any],
) -> None:
    train_source_root = train_manifest.get("source_root")
    eval_source_root = eval_manifest.get("source_root")
    train_source_repo = train_manifest.get("source_repo_id")
    eval_source_repo = eval_manifest.get("source_repo_id")
    if (
        not train_source_root
        or not eval_source_root
        or Path(train_source_root).resolve() != Path(eval_source_root).resolve()
        or not train_source_repo
        or train_source_repo != eval_source_repo
    ):
        raise ValueError("LIBERO v3 train/eval manifests must reference the same absolute source dataset.")

    train_source_ids = {int(value) for value in train_manifest.get("source_episode_ids", [])}
    eval_source_ids = {int(value) for value in eval_manifest.get("source_episode_ids", [])}
    if not train_source_ids or not eval_source_ids:
        raise ValueError("LIBERO v3 train/eval manifests must list non-empty source_episode_ids.")
    overlap = sorted(train_source_ids & eval_source_ids)
    if overlap:
        raise ValueError(f"LIBERO v3 train/eval source trajectory leakage: {overlap}.")


def validate_libero_v3_training_dataset(cfg: TrainPipelineConfig, dataset: Any) -> None:
    """Certify the dataset boundary for LIBERO chunk-relative DP/MAM training."""
    active_cfg = cfg.trainable_config
    policy_type = getattr(active_cfg, "type", None)
    if policy_type not in {"diffusion", "mam"} or not getattr(active_cfg, "use_relative_actions", False):
        return
    robot_type = getattr(dataset.meta, "robot_type", None)
    env_type = getattr(cfg.env, "type", None) if cfg.env is not None else None
    if robot_type != "libero":
        if env_type in {"libero", "libero_plus"}:
            raise ValueError(
                f"LIBERO v3 relative-action training requires dataset robot_type='libero', got {robot_type!r}."
            )
        return

    train_root_value = getattr(dataset, "root", None) or cfg.dataset.root
    if train_root_value is None:
        raise ValueError("LIBERO v3 relative-action training requires an explicit dataset.root.")
    train_root = Path(train_root_value)
    train_manifest = require_libero_v3_relative_ready_dataset(train_root)
    if (
        train_manifest.get("stage") != "absolute_to_mam"
        or train_manifest.get("dataset_split") != "train"
        or train_manifest.get("policy_action_representation") != LIBERO_CHUNK_RELATIVE_ACTION
    ):
        raise ValueError(
            "LIBERO v3 relative-action DP/MAM training requires an absolute_to_mam/train dataset "
            "with chunk-relative SE(3) policy actions."
        )

    if policy_type == "mam":
        eval_repo_id = getattr(active_cfg, "mam_eval_dataset_repo_id", None)
        eval_root_value = getattr(active_cfg, "mam_eval_dataset_root", None)
        eval_episodes = getattr(active_cfg, "mam_eval_episodes", None)
        if cfg.overfit_test:
            same_root = (
                eval_root_value is not None and Path(eval_root_value).resolve() == train_root.resolve()
            )
            if (
                eval_repo_id != cfg.dataset.repo_id
                or not same_root
                or list(eval_episodes or []) != list(cfg.dataset.episodes or [])
            ):
                raise ValueError(
                    "LIBERO v3 MAM overfit requires its eval repo/root/episodes to be exactly "
                    "identical to the selected training trajectories."
                )
            return
        if cfg.env is not None and cfg.eval_freq > 0:
            if eval_repo_id is None or eval_root_value is None:
                raise ValueError(
                    "Normal LIBERO v3 MAM online evaluation requires explicit "
                    "policy.mam_eval_dataset_repo_id/root."
                )
            eval_manifest = require_libero_v3_relative_ready_dataset(eval_root_value)
            if (
                eval_manifest.get("stage") != "absolute_to_mam"
                or eval_manifest.get("dataset_split") != "eval"
                or eval_manifest.get("policy_action_representation") != LIBERO_CHUNK_RELATIVE_ACTION
            ):
                raise ValueError(
                    "Normal LIBERO v3 MAM evaluation requires an absolute_to_mam/eval dataset "
                    "with chunk-relative SE(3) policy actions."
                )
            if not getattr(active_cfg, "allow_independent_eval_source", False):
                _validate_libero_split_provenance(train_manifest, eval_manifest)
            elif train_manifest.get("source_root") == eval_manifest.get("source_root") and train_manifest.get(
                "source_repo_id"
            ) == eval_manifest.get("source_repo_id"):
                raise ValueError(
                    "policy.allow_independent_eval_source=True requires train and eval to identify "
                    "different source datasets. Disable the override for a normal split."
                )
        return

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

    if eval_repo_id is None:
        return
    if eval_root_value is None:
        raise ValueError("LIBERO v3 fixed evaluation requires an explicit eval.dataset_root.")
    eval_manifest = require_libero_v3_relative_ready_dataset(eval_root_value)
    if (
        eval_manifest.get("stage") != "absolute_to_mam"
        or eval_manifest.get("dataset_split") != "eval"
        or eval_manifest.get("policy_action_representation") != LIBERO_CHUNK_RELATIVE_ACTION
    ):
        raise ValueError(
            "Normal LIBERO v3 evaluation requires an absolute_to_mam/eval dataset "
            "with chunk-relative SE(3) policy actions."
        )
    if not getattr(cfg.eval, "allow_independent_source", False):
        _validate_libero_split_provenance(train_manifest, eval_manifest)
    elif train_manifest.get("source_root") == eval_manifest.get("source_root") and train_manifest.get(
        "source_repo_id"
    ) == eval_manifest.get("source_repo_id"):
        raise ValueError(
            "eval.allow_independent_source=True requires train and eval to identify "
            "different source datasets. Disable the override for a normal split."
        )
