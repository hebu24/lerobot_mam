from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

from lerobot.datasets import LeRobotDatasetMetadata


def validate_libero_action_semantics(env_cfg: Any, policy_cfg: Any) -> None:
    """Reject a relative-chunk policy paired with a relative LIBERO controller."""
    if (
        bool(getattr(policy_cfg, "use_relative_actions", False))
        and getattr(env_cfg, "control_mode", None) != "absolute"
    ):
        raise ValueError(
            "LIBERO policy.use_relative_actions=True predicts chunk-relative SE(3) actions and "
            "requires env.control_mode='absolute'. A relative controller would silently execute "
            "the reconstructed absolute goals as deltas."
        )


def configure_fixed_libero_eval_from_dataset(cfg: Any) -> None:
    """Configure task-scoped LIBERO resets from an evaluation dataset split."""
    repo_id = getattr(cfg.eval, "dataset_repo_id", None)
    if not repo_id:
        return

    metadata = LeRobotDatasetMetadata(repo_id, root=getattr(cfg.eval, "dataset_root", None))
    requested = getattr(cfg.eval, "dataset_episodes", None)
    requested_set = set(requested) if requested is not None else None
    rows = [
        row
        for row in metadata.episodes
        if requested_set is None or int(row["episode_index"]) in requested_set
    ]
    found = {int(row["episode_index"]) for row in rows}
    if requested_set is not None and found != requested_set:
        raise ValueError(f"Fixed eval dataset is missing episode ids {sorted(requested_set - found)}.")

    columns = set(getattr(metadata.episodes, "column_names", []) or [])
    task_key = next((key for key in ("libero/task_id", "task_id") if key in columns), None)
    suite_key = next((key for key in ("libero/suite", "suite") if key in columns), None)
    init_value_key = next((key for key in ("libero/init_state", "init_state") if key in columns), None)
    init_id_key = next((key for key in ("libero/init_state_id", "init_state_id") if key in columns), None)
    if task_key is None or (init_value_key is None and init_id_key is None):
        raise ValueError(
            "Fixed LIBERO eval metadata requires task_id and either raw init_state or init_state_id."
        )

    by_task: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for row in rows:
        raw_suite = row[suite_key] if suite_key is not None else getattr(cfg.env, "task", "")
        suite = str(raw_suite) if raw_suite is not None else str(getattr(cfg.env, "task", ""))
        by_task[(suite, int(row[task_key]))].append(row)
    if not by_task:
        raise ValueError("Fixed LIBERO eval dataset selected no episodes.")

    per_task = int(cfg.eval.n_episodes)
    selected_by_task: dict[tuple[str, int], list[Any]] = {}
    for key, task_rows in sorted(by_task.items()):
        task_rows = sorted(task_rows, key=lambda row: int(row["episode_index"]))
        if len(task_rows) < per_task:
            raise ValueError(
                f"Fixed eval split has {len(task_rows)} episode(s) for {key[0]}/{key[1]}, "
                f"but eval.n_episodes={per_task} is interpreted per task."
            )
        selected_by_task[key] = task_rows[:per_task]

    suites = {suite for suite, _ in selected_by_task}
    if len(suites) != 1:
        raise ValueError(f"Fixed LIBERO eval currently requires one suite, got {sorted(suites)}.")
    cfg.env.task = next(iter(suites))
    cfg.env.task_ids = sorted(task_id for _, task_id in selected_by_task)
    all_have_raw_values = init_value_key is not None and all(
        row[init_value_key] is not None for task_rows in selected_by_task.values() for row in task_rows
    )
    if all_have_raw_values:
        cfg.env.init_state_values_by_task = {
            f"{suite}/{task_id}": [
                np.asarray(row[init_value_key], dtype=np.float64).reshape(-1).tolist() for row in task_rows
            ]
            for (suite, task_id), task_rows in selected_by_task.items()
        }
        cfg.env.init_state_ids_by_task = None
        cfg.env.init_state_values = None
        cfg.env.init_state_ids = None
        if hasattr(cfg.env, "num_steps_wait"):
            cfg.env.num_steps_wait = 0
    else:
        if init_id_key is None:
            raise ValueError(
                "Some fixed eval episodes lack raw init_state and no init_state_id is available."
            )
        cfg.env.init_state_ids_by_task = {
            f"{suite}/{task_id}": [int(row[init_id_key]) for row in task_rows]
            for (suite, task_id), task_rows in selected_by_task.items()
        }
        cfg.env.init_state_values_by_task = None
        cfg.env.init_state_values = None
        cfg.env.init_state_ids = None
    cfg.eval.batch_size = min(int(cfg.eval.batch_size), per_task)
    logging.info(
        "Fixed LIBERO eval uses %d episode(s) per task from %s: task_ids=%s",
        per_task,
        repo_id,
        cfg.env.task_ids,
    )
