#!/usr/bin/env python
"""Collect successful LIBERO-10 absolute-action rollouts from a DP checkpoint.

The collector deliberately separates rollout from LeRobot materialization:
accepted trajectories are first written as resumable NPZ staging files, then
materialized into a v3 absolute-action dataset that can be consumed by
``scripts/libero/data/convert_libero_absolute_to_mam.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.libero_pipeline import (
    LIBERO_ABSOLUTE_ACTION,
    LIBERO_CLOSED_LOOP_ABSOLUTE_MATERIALIZATION,
    LIBERO_PIPELINE_VERSION,
    LIBERO_STATE_14D,
    write_libero_pipeline_manifest,
)
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.libero import LiberoEnv, _get_suite
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.random_utils import seeded_context, set_seed

DEFAULT_CHECKPOINT = Path("outputs/checkpoints/dp_libero10_v3_filtered_best_sr94_step095000/pretrained_model")
DEFAULT_OUTPUT_ROOT = Path("outputs/datasets/libero10_100_rollout_absolute_lpb")
DEFAULT_REFERENCE_ROOT = Path("outputs/datasets/libero10_500_train")
DEFAULT_TOKENIZER_REPO = "openai/clip-vit-base-patch32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use random LIBERO reset/policy seeds and a DP checkpoint to collect only successful, "
            "length-filtered LIBERO-10 trajectories."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-repo-id", default="local/libero10_100_rollout_absolute_lpb")
    parser.add_argument("--staging-root", type=Path, default=None)
    parser.add_argument("--reference-length-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--max-attempts-per-task", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--start-seed",
        "--seed",
        dest="start_seed",
        type=int,
        default=100_000,
        help=(
            "First native LIBERO reset seed for every task. Seeds advance consecutively until "
            "the requested number of accepted trajectories is reached. --seed is a deprecated alias."
        ),
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Local CLIP tokenizer directory. Auto-detected from HF_HOME when omitted.",
    )
    parser.add_argument(
        "--max-trajectory-steps",
        type=int,
        default=None,
        help=(
            "Optional uniform accepted-length ceiling. By default the per-task maximum length "
            "from --reference-length-root is used."
        ),
    )
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--materialize-only", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def _task_ids(text: str) -> list[int]:
    values = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if not values or any(value < 0 or value > 9 for value in values):
        raise ValueError(f"--task-ids must be a non-empty subset of 0..9, got {values}.")
    return values


def _reference_length_limits(root: Path, task_ids: list[int]) -> dict[int, int]:
    episode_paths = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(f"No episode metadata found under {root}.")
    rows: list[dict[str, Any]] = []
    for path in episode_paths:
        table = pq.read_table(path)
        columns = set(table.column_names)
        task_key = "libero/task_id" if "libero/task_id" in columns else "task_id"
        rows.extend(table.select(["length", task_key]).to_pylist())
    limits: dict[int, int] = {}
    for task_id in task_ids:
        lengths = [int(row["length"]) for row in rows if int(row[task_key]) == task_id]
        if not lengths:
            raise ValueError(f"Reference dataset {root} has no trajectories for task {task_id}.")
        limits[task_id] = max(lengths)
    return limits


def _find_tokenizer_path(explicit: Path | None) -> Path:
    if explicit is not None:
        if not (explicit / "tokenizer.json").is_file():
            raise FileNotFoundError(f"Invalid tokenizer directory: {explicit}")
        return explicit.resolve()

    hf_home = Path(os.environ.get("HF_HOME", ".hf-cache"))
    snapshots = hf_home / "hub" / "models--openai--clip-vit-base-patch32" / "snapshots"
    candidates = sorted(snapshots.glob("*")) if snapshots.is_dir() else []
    candidates = [path for path in candidates if (path / "tokenizer.json").is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find cached {DEFAULT_TOKENIZER_REPO} tokenizer under {snapshots}. "
            "Pass --tokenizer-path explicitly."
        )
    return candidates[-1].resolve()


def _extract_successes(info: dict[str, Any], batch_size: int) -> np.ndarray:
    successes = np.zeros(batch_size, dtype=bool)
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, dict) and "is_success" in final_info:
            values = np.asarray(final_info["is_success"], dtype=bool).reshape(-1)
            successes[: min(batch_size, len(values))] |= values[:batch_size]
    if "is_success" in info:
        values = np.asarray(info["is_success"], dtype=bool).reshape(-1)
        successes[: min(batch_size, len(values))] |= values[:batch_size]
    return successes


def _images_to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
    value = image.detach().cpu().numpy()
    value = np.transpose(value, (0, 2, 3, 1))
    return np.clip(np.rint(value * 255.0), 0, 255).astype(np.uint8)


def _rollout_batch(
    *,
    task_suite: Any,
    task_id: int,
    rollout_seeds: list[int],
    max_accepted_steps: int,
    policy: Any,
    env_cfg: LiberoEnvConfig,
    env_preprocessor: Any,
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
) -> list[tuple[dict[str, np.ndarray] | None, dict[str, Any]]]:
    if not rollout_seeds:
        raise ValueError("A rollout batch must contain at least one seed.")
    batch_size = len(rollout_seeds)

    def make_env_fn():
        def make_single_env() -> LiberoEnv:
            return LiberoEnv(
                task_suite=task_suite,
                task_id=task_id,
                task_suite_name=env_cfg.task,
                camera_name=env_cfg.camera_name,
                obs_type=env_cfg.obs_type,
                observation_height=env_cfg.observation_height,
                observation_width=env_cfg.observation_width,
                init_states=False,
                n_envs=batch_size,
                num_steps_wait=0,
                control_mode="absolute",
            )

        return make_single_env

    env = gym.vector.SyncVectorEnv([make_env_fn() for _ in rollout_seeds])
    states: list[list[np.ndarray]] = [[] for _ in range(batch_size)]
    images: list[list[np.ndarray]] = [[] for _ in range(batch_size)]
    wrist_images: list[list[np.ndarray]] = [[] for _ in range(batch_size)]
    actions: list[list[np.ndarray]] = [[] for _ in range(batch_size)]
    reasons = ["environment_limit"] * batch_size
    successes = np.zeros(batch_size, dtype=bool)
    active = np.ones(batch_size, dtype=bool)
    try:
        policy.reset()
        observation, _ = env.reset(seed=rollout_seeds)
        # Save the exact state produced by the native seeded reset. The final
        # fixed MAM evaluator restores these values with set_init_state(), while
        # collection itself never touches LIBERO's official 0..49 init states.
        init_states = []
        for single_env in env.envs:
            if single_env._env is None:
                raise RuntimeError("LIBERO environment was not initialized by reset().")
            init_states.append(
                np.asarray(single_env._env.get_sim_state(), dtype=np.float64).reshape(-1).copy()
            )
        task_descriptions = [str(value) for value in env.call("task_description")]
        postprocessed_action_queue: deque[torch.Tensor] = deque()
        requires_context = bool(getattr(postprocessor, "requires_transition_context", False))
        environment_limit = int(env.call("_max_episode_steps")[0])

        with seeded_context(rollout_seeds[0]), torch.inference_mode():
            for _ in range(environment_limit):
                policy_observation = preprocess_observation(observation)
                policy_observation["task"] = task_descriptions
                policy_observation = env_preprocessor(policy_observation)
                if OBS_STATE not in policy_observation:
                    raise KeyError(
                        f"LIBERO environment preprocessing did not produce {OBS_STATE!r}; "
                        f"available keys={sorted(policy_observation)}"
                    )

                state_batch = policy_observation[OBS_STATE].detach().cpu().numpy().astype(np.float32)
                image_batch = _images_to_uint8_hwc(policy_observation[f"{OBS_IMAGES}.image"])
                wrist_batch = _images_to_uint8_hwc(policy_observation[f"{OBS_IMAGES}.image2"])
                for index in np.flatnonzero(active):
                    states[index].append(state_batch[index])
                    images[index].append(image_batch[index])
                    wrist_images[index].append(wrist_batch[index])

                context = deepcopy(policy_observation) if requires_context else None
                model_observation = preprocessor(policy_observation)
                if requires_context:
                    assert context is not None
                    model_observation = policy.update_observation_queue(model_observation)
                    if not postprocessed_action_queue:
                        action_chunk = policy.predict_action_chunk(model_observation)
                        action_chunk = postprocessor({**context, ACTION: action_chunk})
                        postprocessed_action_queue.extend(action_chunk.transpose(0, 1))
                    action = postprocessed_action_queue.popleft()
                else:
                    action = postprocessor(policy.select_action(model_observation))
                action = env_postprocessor({ACTION: action})[ACTION]
                action_numpy = action.detach().cpu().numpy().astype(np.float32)
                for index in np.flatnonzero(active):
                    actions[index].append(action_numpy[index])

                observation, _, terminated, truncated, info = env.step(action_numpy)
                step_successes = _extract_successes(info, batch_size)
                step_done = np.asarray(terminated | truncated, dtype=bool)
                for index in np.flatnonzero(active):
                    if step_successes[index]:
                        successes[index] = True
                        reasons[index] = (
                            "success" if len(actions[index]) <= max_accepted_steps else "overlong_success"
                        )
                        active[index] = False
                    elif step_done[index]:
                        reasons[index] = "terminated_without_success"
                        active[index] = False
                    elif len(actions[index]) > max_accepted_steps:
                        reasons[index] = "overlong"
                        active[index] = False
                if not np.any(active):
                    break
    finally:
        env.close()

    results = []
    task = task_suite.get_task(task_id)
    for index in range(batch_size):
        metadata = {
            "task_id": task_id,
            "task_name": task.name,
            "task_description": task.language,
            # Preserve the legacy integer field while making its meaning
            # explicit: for native resets the seed uniquely identifies the
            # captured initial state.
            "init_state_id": rollout_seeds[index],
            "init_state_source": "native_seed",
            "init_state": np.asarray(init_states[index], dtype=np.float64).reshape(-1).tolist(),
            "rollout_seed": rollout_seeds[index],
            "length": len(actions[index]),
            "max_accepted_steps": max_accepted_steps,
            "success": bool(successes[index]),
            "reason": reasons[index],
        }
        trajectory = None
        if reasons[index] == "success":
            trajectory = {
                ACTION: np.stack(actions[index]).astype(np.float32),
                OBS_STATE: np.stack(states[index]).astype(np.float32),
                f"{OBS_IMAGES}.image": np.stack(images[index]),
                f"{OBS_IMAGES}.image2": np.stack(wrist_images[index]),
            }
        results.append((trajectory, metadata))
    return results


def _staged_metadata(staging_root: Path, task_id: int) -> list[tuple[Path, dict[str, Any]]]:
    task_root = staging_root / f"task-{task_id:02d}"
    result = []
    for metadata_path in sorted(task_root.glob("episode-*.json")):
        npz_path = metadata_path.with_suffix(".npz")
        if npz_path.is_file():
            result.append((npz_path, json.loads(metadata_path.read_text(encoding="utf-8"))))
    return result


def _attempted_rollout_seeds(path: Path, task_id: int) -> set[int]:
    if not path.is_file():
        return set()
    attempted = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid attempt log JSON at {path}:{line_number}") from exc
        if int(row["task_id"]) == task_id:
            attempted.add(int(row["rollout_seed"]))
    return attempted


def _write_staged_trajectory(
    staging_root: Path,
    task_id: int,
    local_episode_id: int,
    trajectory: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    task_root = staging_root / f"task-{task_id:02d}"
    task_root.mkdir(parents=True, exist_ok=True)
    stem = task_root / f"episode-{local_episode_id:02d}"
    temporary_npz = stem.with_suffix(".npz.tmp")
    with temporary_npz.open("wb") as file:
        np.savez_compressed(file, **trajectory)
    temporary_npz.replace(stem.with_suffix(".npz"))
    temporary_json = stem.with_suffix(".json.tmp")
    temporary_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_json.replace(stem.with_suffix(".json"))


def collect(args: argparse.Namespace, task_ids: list[int], limits: dict[int, int]) -> Path:
    staging_root = args.staging_root or args.output_root.with_name(f"{args.output_root.name}_staging")
    staging_root.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"Invalid DP checkpoint: {checkpoint}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    tokenizer_path = _find_tokenizer_path(args.tokenizer_path)
    policy_cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
    policy_cfg.pretrained_path = checkpoint
    policy_cfg.device = args.device
    if hasattr(policy_cfg, "language_tokenizer_name"):
        policy_cfg.language_tokenizer_name = str(tokenizer_path)

    env_cfg = LiberoEnvConfig(
        task=args.suite,
        task_ids=task_ids,
        control_mode="absolute",
        observation_height=args.height,
        observation_width=args.width,
        num_steps_wait=0,
    )
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, policy_cfg)
    task_suite = _get_suite(args.suite)
    set_seed(args.start_seed)

    attempt_log_path = staging_root / "attempts.jsonl"
    for task_id in task_ids:
        staged = _staged_metadata(staging_root, task_id)
        if len(staged) > args.episodes_per_task:
            raise ValueError(f"Staging has too many task {task_id} episodes: {len(staged)}")
        accepted_seeds = {int(metadata["rollout_seed"]) for _, metadata in staged}
        attempted_seeds = _attempted_rollout_seeds(attempt_log_path, task_id)
        if accepted_seeds - attempted_seeds:
            raise ValueError(
                f"Task {task_id} staging contains accepted seeds missing from attempts.jsonl: "
                f"{sorted(accepted_seeds - attempted_seeds)}"
            )
        if any(seed < args.start_seed for seed in attempted_seeds):
            raise ValueError(f"Task {task_id} attempt log contains seeds below start seed {args.start_seed}.")
        next_seed = max(attempted_seeds, default=args.start_seed - 1) + 1

        logging.info(
            "Task %d: resume=%d target=%d length_limit=%d next_seed=%d",
            task_id,
            len(staged),
            args.episodes_per_task,
            limits[task_id],
            next_seed,
        )
        attempts = 0
        while len(staged) < args.episodes_per_task and attempts < args.max_attempts_per_task:
            batch_count = min(
                args.batch_size,
                args.max_attempts_per_task - attempts,
            )
            batch_seeds = list(range(next_seed, next_seed + batch_count))
            batch_results = _rollout_batch(
                task_suite=task_suite,
                task_id=task_id,
                rollout_seeds=batch_seeds,
                max_accepted_steps=limits[task_id],
                policy=policy,
                env_cfg=env_cfg,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
            )
            attempts += batch_count
            next_seed += batch_count
            for trajectory, metadata in batch_results:
                with attempt_log_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(metadata, sort_keys=True) + "\n")
                logging.info(
                    "Task %d attempt %d seed=%d length=%d result=%s",
                    task_id,
                    attempts,
                    metadata["rollout_seed"],
                    metadata["length"],
                    metadata["reason"],
                )
                if trajectory is not None and len(staged) < args.episodes_per_task:
                    local_episode_id = len(staged)
                    _write_staged_trajectory(staging_root, task_id, local_episode_id, trajectory, metadata)
                    staged = _staged_metadata(staging_root, task_id)

        if len(staged) != args.episodes_per_task:
            raise RuntimeError(
                f"Task {task_id}: collected {len(staged)}/{args.episodes_per_task} successful "
                f"trajectories after {attempts} new attempts. Re-run to resume or increase "
                "--max-attempts-per-task."
            )
    return staging_root


def _patch_episode_metadata(root: Path, rows: dict[int, dict[str, Any]]) -> None:
    keys = (
        "libero/suite",
        "libero/task_id",
        "libero/task_name",
        "libero/init_state_id",
        "libero/init_state",
        "libero/rollout_seed",
        "libero/policy_checkpoint",
        "libero/source_episode_id",
    )
    for parquet_path in sorted((root / "meta" / "episodes").glob("**/*.parquet")):
        frame = pd.read_parquet(parquet_path)
        for key in keys:
            frame[key] = [rows[int(ep)].get(key) for ep in frame["episode_index"].astype(int)]
        temporary_path = parquet_path.with_suffix(f"{parquet_path.suffix}.tmp")
        frame.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, parquet_path)


def materialize(
    args: argparse.Namespace,
    staging_root: Path,
    task_ids: list[int],
    limits: dict[int, int],
) -> None:
    staged = [item for task_id in task_ids for item in _staged_metadata(staging_root, task_id)]
    expected = len(task_ids) * args.episodes_per_task
    if len(staged) != expected:
        raise ValueError(f"Staging contains {len(staged)} trajectories; expected {expected}.")
    if args.output_root.exists():
        if not args.overwrite_output:
            raise FileExistsError(
                f"{args.output_root} exists. Pass --overwrite-output to replace it, or use --collect-only."
            )
        shutil.rmtree(args.output_root)

    features = {
        ACTION: {"dtype": "float32", "shape": (7,), "names": None},
        OBS_STATE: {"dtype": "float32", "shape": (14,), "names": None},
        f"{OBS_IMAGES}.image": {
            "dtype": "image",
            "shape": (args.height, args.width, 3),
            "names": ["height", "width", "channels"],
        },
        f"{OBS_IMAGES}.image2": {
            "dtype": "image",
            "shape": (args.height, args.width, 3),
            "names": ["height", "width", "channels"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=args.output_repo_id,
        root=args.output_root,
        fps=args.fps,
        robot_type="libero",
        features=features,
        use_videos=False,
        image_writer_threads=args.image_writer_threads,
    )
    episode_rows: dict[int, dict[str, Any]] = {}
    for episode_index, (npz_path, metadata) in enumerate(staged):
        with np.load(npz_path) as trajectory:
            # NPZ members are compressed independently. Cache each member once;
            # indexing ``trajectory[key]`` inside the frame loop would decompress
            # the complete image sequence again for every frame.
            arrays = {
                key: trajectory[key]
                for key in (
                    ACTION,
                    OBS_STATE,
                    f"{OBS_IMAGES}.image",
                    f"{OBS_IMAGES}.image2",
                )
            }
            length = len(arrays[ACTION])
            if length != int(metadata["length"]):
                raise ValueError(f"Staging length mismatch in {npz_path}.")
            for frame_index in range(length):
                dataset.add_frame(
                    {
                        ACTION: arrays[ACTION][frame_index],
                        OBS_STATE: arrays[OBS_STATE][frame_index],
                        f"{OBS_IMAGES}.image": arrays[f"{OBS_IMAGES}.image"][frame_index],
                        f"{OBS_IMAGES}.image2": arrays[f"{OBS_IMAGES}.image2"][frame_index],
                        "task": str(metadata["task_description"]),
                    }
                )
        dataset.save_episode()
        episode_rows[episode_index] = {
            "libero/suite": args.suite,
            "libero/task_id": int(metadata["task_id"]),
            "libero/task_name": str(metadata["task_name"]),
            "libero/init_state_id": int(metadata["init_state_id"]),
            "libero/init_state": metadata["init_state"],
            "libero/rollout_seed": int(metadata["rollout_seed"]),
            "libero/policy_checkpoint": str(args.checkpoint.resolve()),
            "libero/source_episode_id": episode_index,
        }
    dataset.finalize()
    _patch_episode_metadata(args.output_root, episode_rows)
    write_libero_pipeline_manifest(
        args.output_root,
        {
            "pipeline_version": LIBERO_PIPELINE_VERSION,
            "stage": "delta_to_absolute",
            "conversion_complete": True,
            "audit_complete": True,
            "action_representation": LIBERO_ABSOLUTE_ACTION,
            "state_representation": LIBERO_STATE_14D,
            "observation_materialization": LIBERO_CLOSED_LOOP_ABSOLUTE_MATERIALIZATION,
            "relative_action_ready": True,
            "collection_method": "seeded_dp_closed_loop_rollout",
            "suite": args.suite,
            "task_ids": task_ids,
            "episodes_per_task": args.episodes_per_task,
            "episode_count": expected,
            "source_episode_ids": list(range(expected)),
            "source_policy_checkpoint": str(args.checkpoint.resolve()),
            "collection_protocol": "lpb_native_seeded_reset",
            "test_start_seed": args.start_seed,
            "collection_seed": args.start_seed,
            "length_filter_by_task": {str(key): value for key, value in limits.items()},
            "reference_length_root": str(args.reference_length_root.resolve()),
            "staging_root": str(staging_root.resolve()),
            "image_size": [args.height, args.width],
        },
    )
    logging.info("Materialized %d trajectories at %s", expected, args.output_root)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.collect_only and args.materialize_only:
        raise ValueError("--collect-only and --materialize-only are mutually exclusive.")
    task_ids = _task_ids(args.task_ids)
    if (
        args.episodes_per_task <= 0
        or args.max_attempts_per_task <= 0
        or args.batch_size <= 0
        or args.image_writer_threads < 0
    ):
        raise ValueError("Episode, attempt, and batch counts must be positive; threads cannot be negative.")
    if args.start_seed < 50:
        raise ValueError("--start-seed must be at least 50 and must not overlap training seeds 0..49.")
    if args.max_trajectory_steps is None:
        limits = _reference_length_limits(args.reference_length_root, task_ids)
    else:
        if args.max_trajectory_steps <= 0:
            raise ValueError("--max-trajectory-steps must be positive.")
        limits = dict.fromkeys(task_ids, args.max_trajectory_steps)

    staging_root = args.staging_root or args.output_root.with_name(f"{args.output_root.name}_staging")
    if not args.materialize_only:
        staging_root = collect(args, task_ids, limits)
    if not args.collect_only:
        materialize(args, staging_root, task_ids, limits)


if __name__ == "__main__":
    main()
