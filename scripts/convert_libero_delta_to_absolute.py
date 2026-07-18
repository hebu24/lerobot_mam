#!/usr/bin/env python

"""Convert a LIBERO LeRobot dataset from OSC delta actions to absolute EEF targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.datasets.compute_stats import (
    aggregate_stats,
    compute_episode_stats,
    get_feature_stats,
    sample_indices,
)
from lerobot.datasets.dataset_tools import _write_parquet
from lerobot.datasets.io_utils import write_stats
from lerobot.datasets.libero_pipeline import (
    LIBERO_ABSOLUTE_ACTION,
    LIBERO_CLOSED_LOOP_ABSOLUTE_MATERIALIZATION,
    LIBERO_DELTA_ACTION,
    LIBERO_PIPELINE_VERSION,
    LIBERO_STATE_14D,
    read_libero_pipeline_manifest,
    require_libero_v3_action_dataset,
    write_libero_pipeline_manifest,
)
from lerobot.envs.libero_assets import rewrite_libero_demo_xml_paths
from lerobot.processor.libero_relative_action_processor import (
    matrix_to_axis_angle,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

AGENTVIEW_IMAGE = f"{OBS_IMAGES}.image"
WRIST_IMAGE = f"{OBS_IMAGES}.image2"
CLOSED_LOOP_OBSERVATION_KEYS = frozenset({OBS_STATE, AGENTVIEW_IMAGE, WRIST_IMAGE})


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory_sha256(root: Path, paths: list[Path]) -> str:
    """Fingerprint a file set cheaply enough to run before every resume."""
    inventory = []
    for path in sorted(paths):
        stat = path.stat()
        inventory.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    payload = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _episode_rows(meta: LeRobotDatasetMetadata) -> pd.DataFrame:
    episodes = meta.episodes
    if hasattr(episodes, "to_pandas"):
        return episodes.to_pandas()
    return pd.DataFrame(list(episodes))


def _init_state_id(row: pd.Series) -> int:
    for key in ("libero/init_state_id", "init_state_id"):
        if key in row and not pd.isna(row[key]):
            return int(row[key])
    return int(row["episode_index"])


def _init_state_value(row: pd.Series) -> list[float] | None:
    for key in ("libero/init_state", "init_state"):
        if key not in row:
            continue
        value = row[key]
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.size > 0:
            return array.tolist()
    return None


def _episode_suite(row: pd.Series, default_suite: str) -> str:
    for key in ("libero/suite", "suite"):
        if key in row and not pd.isna(row[key]):
            return str(row[key])
    return default_suite


def _episode_task_id(row: pd.Series, default_task_id: int) -> int:
    for key in ("libero/task_id", "task_id"):
        if key in row and not pd.isna(row[key]):
            return int(row[key])
    return int(default_task_id)


def _episode_text(row: pd.Series, key: str) -> str | None:
    if key not in row or pd.isna(row[key]):
        return None
    return str(row[key])


def _parse_episode_ids(text: str | None) -> set[int] | None:
    if not text:
        return None
    return {int(part.strip()) for part in text.split(",") if part.strip()}


def _validate_closed_loop_observation_schema(meta: LeRobotDatasetMetadata) -> None:
    """Reject dataset layouts that cannot be rematerialized atomically in parquet."""
    video_keys = set(meta.video_keys)
    if video_keys:
        raise ValueError(
            "Closed-loop LIBERO observation rematerialization does not support video-backed features: "
            f"{sorted(video_keys)}. Re-encode videos from the replayed frames first; copying the old videos "
            "would mix new state with stale images."
        )

    observation_keys = {key for key in meta.features if key.startswith("observation.")}
    if observation_keys != CLOSED_LOOP_OBSERVATION_KEYS:
        missing = sorted(CLOSED_LOOP_OBSERVATION_KEYS - observation_keys)
        unsupported = sorted(observation_keys - CLOSED_LOOP_OBSERVATION_KEYS)
        raise ValueError(
            "Closed-loop LIBERO rematerialization requires exactly observation.state and the two "
            f"image features. missing={missing}, unsupported={unsupported}."
        )
    if tuple(meta.features[OBS_STATE]["shape"]) != (14,):
        raise ValueError(
            f"Closed-loop LIBERO rematerialization requires a 14D state, got {meta.features[OBS_STATE]}."
        )
    for key in (AGENTVIEW_IMAGE, WRIST_IMAGE):
        if meta.features[key]["dtype"] != "image":
            raise ValueError(f"{key} must be an image feature, got {meta.features[key]}.")


def _validate_episode_frame_layout(ep_df: pd.DataFrame, expected_length: int) -> pd.DataFrame:
    """Return one complete episode in temporal order or fail before simulator replay."""
    ordered = ep_df.sort_values("frame_index")
    frame_indices = ordered["frame_index"].to_numpy(dtype=np.int64)
    expected = np.arange(expected_length, dtype=np.int64)
    if len(ordered) != expected_length or not np.array_equal(frame_indices, expected):
        raise ValueError(
            "Closed-loop observation rematerialization requires each episode to be wholly contained "
            f"in one parquet file with frame_index=0..{expected_length - 1}; got "
            f"length={len(ordered)}, frame range="
            f"{None if len(frame_indices) == 0 else (int(frame_indices[0]), int(frame_indices[-1]))}."
        )
    if not ordered.index.is_unique:
        raise ValueError("Closed-loop observation rematerialization requires unique parquet row indices.")
    return ordered


def _as_runtime_image(value: Any, *, key: str, height: int, width: int) -> np.ndarray:
    image = np.asarray(value)
    if image.shape != (height, width, 3) or image.dtype != np.uint8:
        raise ValueError(
            f"Runtime {key} must be uint8 HWC with shape {(height, width, 3)}, "
            f"got shape={image.shape}, dtype={image.dtype}."
        )
    return np.ascontiguousarray(image)


def _extract_runtime_observation(
    observation: Mapping[str, Any], *, height: int, width: int
) -> dict[str, np.ndarray]:
    """Convert one raw LiberoEnv observation to the exact v3 parquet features."""
    try:
        robot_state = observation["robot_state"]
        eef_pos = np.asarray(robot_state["eef"]["pos"], dtype=np.float32).reshape(-1)
        eef_quat = np.asarray(robot_state["eef"]["quat"], dtype=np.float32).reshape(-1)
        joint_pos = np.asarray(robot_state["joints"]["pos"], dtype=np.float32).reshape(-1)
        pixels = observation["pixels"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Runtime LIBERO observation is missing state or pixel fields.") from exc
    if eef_pos.shape != (3,) or eef_quat.shape != (4,) or joint_pos.shape != (7,):
        raise ValueError(
            "Runtime LIBERO state must contain eef_pos(3), eef_quat(4), joint_pos(7); "
            f"got {eef_pos.shape}, {eef_quat.shape}, {joint_pos.shape}."
        )
    quat_norm = float(np.linalg.norm(eef_quat))
    if not np.isfinite(quat_norm) or quat_norm < 1e-12:
        raise ValueError("Runtime LIBERO EEF quaternion is non-finite or degenerate.")
    state = np.concatenate((eef_pos, eef_quat / quat_norm, joint_pos)).astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError("Runtime LIBERO state contains non-finite values.")
    return {
        OBS_STATE: state,
        AGENTVIEW_IMAGE: _as_runtime_image(
            pixels["image"], key=AGENTVIEW_IMAGE, height=height, width=width
        ),
        WRIST_IMAGE: _as_runtime_image(
            pixels["image2"], key=WRIST_IMAGE, height=height, width=width
        ),
    }


def _record_closed_loop_episode(
    ordered_episode: pd.DataFrame,
    *,
    absolute_actions: Mapping[int, np.ndarray],
    runtime_env: Any,
    seed: int,
    observation_height: int,
    observation_width: int,
    post_hold_steps: int = 0,
) -> tuple[dict[int, dict[str, np.ndarray]], int]:
    """Replay absolute goals and capture the observation immediately before each action.

    The underlying OffScreenRenderEnv is stepped directly after reset. Unlike
    ``LiberoEnv.step``, it does not auto-reset when success is reached, so an early
    success cannot splice a second reset into an episode. We deliberately keep
    executing to the original length to preserve every episode boundary and index.
    """
    observation, _ = runtime_env.reset(seed=seed)
    if runtime_env._env is None:
        raise RuntimeError("LIBERO runtime env did not initialize during reset.")

    captured: dict[int, dict[str, np.ndarray]] = {}
    first_success_step: int | None = None
    rows = list(ordered_episode.itertuples(index=True))
    for position, row in enumerate(rows):
        row_index = int(row.Index)
        if row_index not in absolute_actions:
            raise ValueError(f"Missing materialized absolute action for parquet row {row_index}.")
        captured[row_index] = _extract_runtime_observation(
            observation,
            height=observation_height,
            width=observation_width,
        )
        action = np.asarray(absolute_actions[row_index], dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError(f"Invalid absolute action at parquet row {row_index}: shape={action.shape}.")

        raw_observation, _, done, _ = runtime_env._env.step(action)
        success = bool(runtime_env._env.check_success())
        if success and first_success_step is None:
            first_success_step = int(row.frame_index)
        if bool(done) and not success and position + 1 < len(rows):
            raise RuntimeError(
                f"LIBERO runtime terminated before episode completion at frame {int(row.frame_index)}."
            )
        if position + 1 < len(rows):
            observation = runtime_env._format_raw_obs(raw_observation)

    if first_success_step is None and post_hold_steps > 0:
        last_action = np.asarray(absolute_actions[int(rows[-1].Index)], dtype=np.float32)
        for hold_index in range(post_hold_steps):
            runtime_env._env.step(last_action)
            if bool(runtime_env._env.check_success()):
                first_success_step = int(rows[-1].frame_index) + hold_index + 1
                break

    if first_success_step is None:
        episode_index = int(ordered_episode["episode_index"].iloc[0])
        raise RuntimeError(
            f"Closed-loop absolute replay did not succeed for episode {episode_index}; "
            "refusing to materialize an invalid relative-action training trajectory."
        )
    return captured, first_success_step


def _apply_materialized_values(
    df: pd.DataFrame,
    *,
    absolute_actions: Mapping[int, np.ndarray],
    observations: Mapping[int, Mapping[str, np.ndarray]],
    target_indices: set[int],
) -> pd.DataFrame:
    """Replace action/observation columns without touching identity or boundary fields."""
    missing_actions = target_indices - set(absolute_actions)
    missing_observations = target_indices - set(observations)
    if missing_actions or missing_observations:
        raise ValueError(
            "Incomplete closed-loop materialization: "
            f"missing_actions={sorted(missing_actions)}, "
            f"missing_observations={sorted(missing_observations)}."
        )

    out = df.copy()
    out[ACTION] = [
        np.asarray(absolute_actions[int(idx)], dtype=np.float32)
        if int(idx) in target_indices
        else np.asarray(action, dtype=np.float32)
        for idx, action in zip(out.index, out[ACTION], strict=True)
    ]
    for key in CLOSED_LOOP_OBSERVATION_KEYS:
        out[key] = [
            observations[int(idx)][key] if int(idx) in target_indices else value
            for idx, value in zip(out.index, out[key], strict=True)
        ]
    return out


def _compute_embedded_image_stats(values: list[Any]) -> dict[str, np.ndarray]:
    """Match LeRobot image statistics while reading parquet-embedded PNG bytes."""
    sampled_images: list[np.ndarray] = []
    for index in sample_indices(len(values)):
        value = values[index]
        if not isinstance(value, Mapping) or value.get("bytes") is None:
            raise ValueError(
                "Closed-loop image statistics require parquet-embedded image bytes; "
                f"got {type(value).__name__} at sampled frame {index}."
            )
        with Image.open(BytesIO(value["bytes"])) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        sampled_images.append(np.transpose(array, (2, 0, 1)))

    stacked = np.stack(sampled_images, axis=0)
    stats = get_feature_stats(stacked, axis=(0, 2, 3), keepdims=True)
    return {
        key: value if key == "count" else np.squeeze(value / 255.0, axis=0)
        for key, value in stats.items()
    }


def _compute_materialized_episode_stats(
    episode: pd.DataFrame,
    features: Mapping[str, dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    """Compute every stored episode statistic from the rematerialized parquet rows."""
    numeric_features = {
        key: feature
        for key, feature in features.items()
        if feature["dtype"] not in {"image", "video", "string", "language"}
    }
    episode_data: dict[str, np.ndarray] = {}
    for key in numeric_features:
        if key not in episode:
            continue
        values = episode[key].to_numpy()
        episode_data[key] = (
            np.stack(values) if len(values) > 0 and hasattr(values[0], "__len__") else np.asarray(values)
        )
    stats = compute_episode_stats(episode_data, numeric_features)
    for key, feature in features.items():
        if feature["dtype"] == "image":
            stats[key] = _compute_embedded_image_stats(episode[key].tolist())
    return stats


def _recompute_materialized_stats(root: Path, meta: LeRobotDatasetMetadata) -> None:
    """Rewrite per-episode metadata stats and global stats from closed-loop data."""
    stats_by_episode: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for parquet_path in sorted((root / "data").glob("chunk-*/*.parquet")):
        data = pd.read_parquet(parquet_path)
        for episode_index, episode in data.groupby("episode_index", sort=True):
            episode_id = int(episode_index)
            if episode_id in stats_by_episode:
                raise ValueError(
                    f"Episode {episode_id} spans multiple parquet files; cannot rewrite stats atomically."
                )
            stats_by_episode[episode_id] = _compute_materialized_episode_stats(
                episode.sort_values("frame_index"),
                meta.features,
            )

    metadata_episode_ids: set[int] = set()
    for parquet_path in sorted((root / "meta" / "episodes").glob("**/*.parquet")):
        episodes = pd.read_parquet(parquet_path)
        episode_ids = episodes["episode_index"].astype(int).tolist()
        missing = set(episode_ids) - set(stats_by_episode)
        if missing:
            raise ValueError(f"Missing rematerialized statistics for episodes {sorted(missing)}.")
        metadata_episode_ids.update(episode_ids)
        for episode_id in episode_ids:
            for feature_key, feature_stats in stats_by_episode[episode_id].items():
                for stat_key in feature_stats:
                    column = f"stats/{feature_key}/{stat_key}"
                    if column not in episodes:
                        episodes[column] = None
        for column in [name for name in episodes if name.startswith("stats/")]:
            parts = column.split("/", maxsplit=2)
            if len(parts) != 3:
                continue
            _, feature_key, stat_key = parts
            episodes[column] = [
                np.asarray(stats_by_episode[episode_id][feature_key][stat_key]).tolist()
                for episode_id in episode_ids
            ]
        temp_path = parquet_path.with_suffix(".parquet.tmp")
        episodes.to_parquet(temp_path, index=False)
        temp_path.replace(parquet_path)

    if metadata_episode_ids != set(stats_by_episode):
        extra = sorted(set(stats_by_episode) - metadata_episode_ids)
        raise ValueError(f"Data contains episodes missing from metadata: {extra}.")
    ordered_stats = [stats_by_episode[index] for index in sorted(stats_by_episode)]
    write_stats(aggregate_stats(ordered_stats), root)


def _convert_file_with_replay(
    df: pd.DataFrame,
    *,
    episode_metadata: dict[int, dict[str, object]],
    task_suites: dict[str, object],
    observation_width: int,
    observation_height: int,
    source_hdf5_dir: Path | None,
    source_action_strategy: str,
    pose_offset: int,
    pose_gripper_offset: int,
    episode_strategy_overrides: dict[int, dict[str, object]],
    closed_loop_seed: int,
    selected_episode_ids: set[int] | None = None,
    auto_repair_failed_replays: bool = False,
    resolved_strategy_overrides: dict[int, dict[str, object]] | None = None,
    allow_unrepairable_episodes: bool = False,
    unrepairable_episode_ids: set[int] | None = None,
) -> pd.DataFrame:
    from libero.libero.envs import OffScreenRenderEnv

    from lerobot.envs.libero import LiberoEnv, sync_libero_controllers
    from lerobot.envs.libero_assets import get_libero_resource_path, validate_libero_assets

    converted_actions: dict[int, np.ndarray] = {}
    converted_observations: dict[int, dict[str, np.ndarray]] = {}
    target_indices: set[int] = set()
    if resolved_strategy_overrides is None:
        resolved_strategy_overrides = {}
    if unrepairable_episode_ids is None:
        unrepairable_episode_ids = set()

    for episode_index, ep_df in df.groupby("episode_index", sort=True):
        if selected_episode_ids is not None and int(episode_index) not in selected_episode_ids:
            continue
        ep_meta = episode_metadata[int(episode_index)]
        ordered_episode = _validate_episode_frame_layout(ep_df, int(ep_meta["length"]))
        episode_target_indices = {int(index) for index in ordered_episode.index}
        task_suite_name = str(ep_meta["suite"])
        task_id = int(ep_meta["task_id"])
        task_suite = task_suites[task_suite_name]
        init_state = ep_meta.get("init_state")
        if init_state is None:
            raise ValueError(
                f"Episode {episode_index} lacks the raw float64 init state required for closed-loop replay."
            )
        runtime_init_state = np.asarray(init_state, dtype=np.float64).reshape(-1)
        if source_hdf5_dir is not None:
            source_file = ep_meta.get("source_file")
            source_demo = ep_meta.get("source_demo")
            if source_file is None or source_demo is None or init_state is None:
                raise ValueError(
                    f"Episode {episode_index} lacks source_file, source_demo, or raw init_state metadata."
                )
            source_path = source_hdf5_dir / str(source_file)
            with h5py.File(source_path, "r") as source_h5:
                source_group = source_h5[f"data/{source_demo}"]
                model_xml = rewrite_libero_demo_xml_paths(str(source_group.attrs["model_file"]))
                source_init_state = np.asarray(source_group.attrs["init_state"], dtype=np.float64)
                source_actions = np.asarray(source_group["actions"], dtype=np.float32)
                source_states = np.asarray(source_group["states"], dtype=np.float64)
            if not np.array_equal(runtime_init_state, source_init_state):
                raise ValueError(
                    f"Episode {episode_index} metadata init state differs from {source_file}:{source_demo}."
                )
            if len(source_actions) != len(ordered_episode) or len(source_states) != len(ordered_episode):
                raise ValueError(
                    f"Episode {episode_index} length differs from its source demo: "
                    f"parquet={len(ordered_episode)}, actions={len(source_actions)}, states={len(source_states)}."
                )
            validate_libero_assets()
            task = task_suite.get_task(task_id)
            bddl_file = get_libero_resource_path("bddl_files") / task.problem_folder / task.bddl_file
            replay_env = OffScreenRenderEnv(
                bddl_file_name=str(bddl_file),
                use_camera_obs=False,
                camera_heights=1,
                camera_widths=1,
            )
            replay_env.reset()
            replay_env.reset_from_xml_string(model_xml)
            replay_env.sim.reset()
            replay_env.set_init_state(source_init_state)
            sync_libero_controllers(replay_env)
            close_env = replay_env
        else:
            source_actions = None
            source_states = None
            env = LiberoEnv(
                task_suite=task_suite,
                task_id=task_id,
                task_suite_name=task_suite_name,
                init_state_id=None if init_state is not None else int(ep_meta["init_state_id"]),
                init_state_values=None if init_state is None else [init_state],
                num_steps_wait=0 if init_state is not None else 10,
                observation_width=observation_width,
                observation_height=observation_height,
                control_mode="relative",
            )
            env.reset()
            assert env._env is not None
            replay_env = env._env
            close_env = env
        robot = replay_env.robots[0]
        replay_success = False
        episode_strategy = {
            "strategy": source_action_strategy,
            "pose_offset": pose_offset,
            "gripper_offset": pose_gripper_offset,
            **resolved_strategy_overrides.get(int(episode_index), {}),
            **episode_strategy_overrides.get(int(episode_index), {}),
        }
        strategy = str(episode_strategy["strategy"])
        strategy_pose_offset = int(episode_strategy["pose_offset"])
        strategy_gripper_offset = int(episode_strategy["gripper_offset"])
        strategy_post_hold_steps = int(episode_strategy.get("post_hold_steps", 0))
        fallback_action_candidates: list[
            tuple[dict[int, np.ndarray], dict[str, object]]
        ] = []

        try:
            for row in ordered_episode.itertuples(index=True):
                frame_index = int(row.frame_index)
                if source_states is not None:
                    if frame_index >= len(source_actions) or frame_index >= len(source_states):
                        raise IndexError(
                            f"Episode {episode_index} frame {frame_index} exceeds source demo "
                            f"lengths actions={len(source_actions)} states={len(source_states)}."
                        )
                    delta = source_actions[frame_index]
                    if strategy == "recorded-eef-pose":
                        pose_index = min(frame_index + strategy_pose_offset, len(source_states) - 1)
                        gripper_index = min(frame_index + strategy_gripper_offset, len(source_actions) - 1)
                        replay_env.sim.set_state_from_flattened(source_states[pose_index])
                        replay_env.sim.forward()
                        sync_libero_controllers(replay_env)
                        goal_axis_angle = matrix_to_axis_angle(
                            np.asarray(robot.controller.ee_ori_mat, dtype=np.float32)
                        )
                        converted_actions[int(row.Index)] = np.concatenate(
                            (
                                np.asarray(robot.controller.ee_pos, dtype=np.float32),
                                np.asarray(goal_axis_angle, dtype=np.float32),
                                np.asarray(
                                    source_actions[gripper_index][robot.controller.control_dim :],
                                    dtype=np.float32,
                                ),
                            )
                        )
                        continue
                    if strategy != "controller-goal":
                        raise ValueError(
                            f"Unsupported source action strategy {strategy!r} for episode {episode_index}."
                        )
                    # Raw LIBERO demos are not perfectly reproduced by open-loop
                    # delta replay for every episode. Re-anchor each controller
                    # goal computation at the recorded MuJoCo state so the
                    # materialized absolute trajectory follows the demonstration.
                    replay_env.sim.set_state_from_flattened(source_states[frame_index])
                    replay_env.sim.forward()
                    sync_libero_controllers(replay_env)
                    for robot_item in replay_env.robots:
                        robot_item.controller.use_delta = True
                else:
                    delta = np.asarray(getattr(row, ACTION), dtype=np.float32)
                gripper_action = delta[robot.controller.control_dim :]
                # Let robosuite set the controller goal through its normal control
                # path, then capture that exact goal. Calling set_goal() directly
                # triggers an incompatible MjSim.update path in current robosuite.
                replay_env.step(delta)
                replay_success = replay_success or bool(replay_env.check_success())
                goal_axis_angle = matrix_to_axis_angle(
                    np.asarray(robot.controller.goal_ori, dtype=np.float32)
                )
                converted_actions[int(row.Index)] = np.concatenate(
                    (
                        np.asarray(robot.controller.goal_pos, dtype=np.float32),
                        np.asarray(goal_axis_angle, dtype=np.float32),
                        gripper_action,
                    )
                )
            if (
                auto_repair_failed_replays
                and source_states is not None
                and source_actions is not None
                and strategy == "controller-goal"
            ):
                # The raw delta demos are occasionally too contact-sensitive to
                # survive conversion through controller goals.  Prepare a small,
                # deterministic set of recorded-pose alternatives while the exact
                # source model is still open.  A candidate is accepted only after
                # the same closed-loop absolute replay success check below.
                # (1, 0) is the normal next-pose/current-gripper convention. A
                # small explicit grid also covers demos whose pose or gripper
                # sample is shifted by one frame. Every candidate must still
                # pass the same closed-loop success check before acceptance.
                for candidate_pose_offset, candidate_gripper_offset in (
                    (1, 0),
                    (0, 0),
                    (1, 1),
                    (0, 1),
                ):
                    candidate_actions: dict[int, np.ndarray] = {}
                    for row in ordered_episode.itertuples(index=True):
                        frame_index = int(row.frame_index)
                        pose_index = min(
                            frame_index + candidate_pose_offset, len(source_states) - 1
                        )
                        gripper_index = min(
                            frame_index + candidate_gripper_offset, len(source_actions) - 1
                        )
                        replay_env.sim.set_state_from_flattened(source_states[pose_index])
                        replay_env.sim.forward()
                        sync_libero_controllers(replay_env)
                        goal_axis_angle = matrix_to_axis_angle(
                            np.asarray(robot.controller.ee_ori_mat, dtype=np.float32)
                        )
                        candidate_actions[int(row.Index)] = np.concatenate(
                            (
                                np.asarray(robot.controller.ee_pos, dtype=np.float32),
                                np.asarray(goal_axis_angle, dtype=np.float32),
                                np.asarray(
                                    source_actions[gripper_index][robot.controller.control_dim :],
                                    dtype=np.float32,
                                ),
                            )
                        )
                    for candidate_post_hold_steps in (0, 50):
                        fallback_action_candidates.append(
                            (
                                candidate_actions,
                                {
                                    "strategy": "recorded-eef-pose",
                                    "pose_offset": candidate_pose_offset,
                                    "gripper_offset": candidate_gripper_offset,
                                    "post_hold_steps": candidate_post_hold_steps,
                                },
                            )
                        )
                fallback_action_candidates.sort(
                    key=lambda item: int(item[1]["post_hold_steps"])
                )
            if source_hdf5_dir is not None and not replay_success:
                logging.warning(
                    "Source delta replay did not reach success for episode %s (%s/%s, %s:%s); "
                    "the converted absolute trajectory still requires final validation.",
                    episode_index,
                    ep_meta["suite"],
                    ep_meta["task_id"],
                    source_file,
                    source_demo,
                )
        finally:
            close_env.close()

        runtime_env = LiberoEnv(
            task_suite=task_suite,
            task_id=task_id,
            task_suite_name=task_suite_name,
            camera_name="agentview_image,robot0_eye_in_hand_image",
            obs_type="pixels_agent_pos",
            init_states=True,
            init_state_values=[runtime_init_state],
            n_envs=1,
            num_steps_wait=0,
            observation_width=observation_width,
            observation_height=observation_height,
            control_mode="absolute",
        )
        try:
            try:
                episode_observations, success_step = _record_closed_loop_episode(
                    ordered_episode,
                    absolute_actions=converted_actions,
                    runtime_env=runtime_env,
                    seed=closed_loop_seed,
                    observation_height=observation_height,
                    observation_width=observation_width,
                    post_hold_steps=strategy_post_hold_steps,
                )
            except RuntimeError as primary_error:
                if not fallback_action_candidates:
                    raise
                logging.warning(
                    "Primary absolute replay failed for episode %s; testing %d recorded-pose repair(s).",
                    episode_index,
                    len(fallback_action_candidates),
                )
                last_error: RuntimeError = primary_error
                for candidate_actions, candidate_strategy in fallback_action_candidates:
                    try:
                        episode_observations, success_step = _record_closed_loop_episode(
                            ordered_episode,
                            absolute_actions=candidate_actions,
                            runtime_env=runtime_env,
                            seed=closed_loop_seed,
                            observation_height=observation_height,
                            observation_width=observation_width,
                            post_hold_steps=int(candidate_strategy["post_hold_steps"]),
                        )
                    except RuntimeError as candidate_error:
                        last_error = candidate_error
                        logging.warning(
                            "Recorded-pose repair failed for episode %s with pose_offset=%s, "
                            "gripper_offset=%s, post_hold_steps=%s.",
                            episode_index,
                            candidate_strategy["pose_offset"],
                            candidate_strategy["gripper_offset"],
                            candidate_strategy["post_hold_steps"],
                        )
                        continue
                    converted_actions.update(candidate_actions)
                    resolved_strategy_overrides[int(episode_index)] = candidate_strategy
                    logging.warning(
                        "Accepted recorded-pose repair for episode %s: %s.",
                        episode_index,
                        candidate_strategy,
                    )
                    break
                else:
                    if allow_unrepairable_episodes:
                        unrepairable_episode_ids.add(int(episode_index))
                        for row_index in episode_target_indices:
                            converted_actions.pop(row_index, None)
                        logging.error(
                            "Excluding unrepairable episode %s after primary and all recorded-pose "
                            "repairs failed.",
                            episode_index,
                        )
                        continue
                    raise RuntimeError(
                        f"Primary and all recorded-pose repairs failed for episode {episode_index}."
                    ) from last_error
            target_indices.update(episode_target_indices)
            converted_observations.update(episode_observations)
        finally:
            runtime_env.close()
        logging.info(
            "Closed-loop rematerialized episode %s (%s/%s) at length %d; first success frame=%d.",
            episode_index,
            task_suite_name,
            task_id,
            len(ordered_episode),
            success_step,
        )

    return _apply_materialized_values(
        df,
        absolute_actions=converted_actions,
        observations=converted_observations,
        target_indices=target_indices,
    )


def convert_dataset(args: argparse.Namespace) -> None:
    input_manifest = require_libero_v3_action_dataset(
        args.input_root,
        action_representation=LIBERO_DELTA_ACTION,
    )
    if args.input_root.resolve() == args.output_root.resolve():
        raise ValueError(
            "LIBERO v3 conversion requires a separate output root; in-place conversion is unsafe."
        )
    if args.source_hdf5_dir is None:
        raise ValueError(
            "LIBERO v3 delta->absolute conversion requires --source-hdf5-dir "
            "so goals are materialized from each recorded model/state."
        )
    if args.resume and args.overwrite:
        raise ValueError("Pass either --resume or --overwrite, not both.")

    input_data_files = sorted((args.input_root / "data").glob("chunk-*/*.parquet"))
    input_episode_files = sorted((args.input_root / "meta" / "episodes").glob("**/*.parquet"))
    source_hdf5_files = sorted(
        [*args.source_hdf5_dir.glob("*.hdf5"), *args.source_hdf5_dir.glob("*.h5")]
    )
    if not input_data_files or not input_episode_files or not source_hdf5_files:
        raise FileNotFoundError(
            "v3 conversion requires input data parquet, episode metadata parquet, and source HDF5 files."
        )
    input_meta = LeRobotDatasetMetadata(args.input_repo_id, root=args.input_root)
    _validate_closed_loop_observation_schema(input_meta)

    conversion_config = {
        "input_root": str(args.input_root.resolve()),
        "input_repo_id": args.input_repo_id,
        "output_repo_id": args.output_repo_id,
        "source_hdf5_dir": str(args.source_hdf5_dir.resolve()),
        "source_action_strategy": args.source_action_strategy,
        "pose_offset": args.pose_offset,
        "pose_gripper_offset": args.pose_gripper_offset,
        "closed_loop_seed": args.closed_loop_seed,
        "auto_repair_failed_replays": args.auto_repair_failed_replays,
        "allow_unrepairable_episodes": args.allow_unrepairable_episodes,
        "observation_materialization": LIBERO_CLOSED_LOOP_ABSOLUTE_MATERIALIZATION,
        "input_manifest_sha256": _file_sha256(
            args.input_root / "meta" / "libero_pipeline.json"
        ),
        "input_data_inventory_sha256": _inventory_sha256(args.input_root, input_data_files),
        "input_episode_metadata_inventory_sha256": _inventory_sha256(
            args.input_root, input_episode_files
        ),
        "source_hdf5_inventory_sha256": _inventory_sha256(
            args.source_hdf5_dir, source_hdf5_files
        ),
        "input_episode_count": input_manifest.get("episode_count"),
        "episode_strategy_overrides": (
            None
            if args.episode_strategy_overrides is None
            else str(args.episode_strategy_overrides.resolve())
        ),
        "episode_strategy_overrides_sha256": (
            None
            if args.episode_strategy_overrides is None
            else _file_sha256(args.episode_strategy_overrides)
        ),
    }

    converted_files: set[str] = set()
    resolved_strategy_overrides: dict[int, dict[str, object]] = {}
    unrepairable_episode_ids: set[int] = set()
    expected_data_files = {
        path.relative_to(args.input_root).as_posix(): path for path in input_data_files
    }
    expected_relative_paths = set(expected_data_files)
    if args.output_root.exists():
        if args.resume:
            progress = read_libero_pipeline_manifest(args.output_root)
            if progress.get("stage") != "delta_to_absolute" or progress.get("config") != conversion_config:
                raise ValueError(
                    f"Cannot resume {args.output_root}: its v3 conversion manifest does not match this run."
                )
            converted_files = {str(path) for path in progress.get("converted_data_files", [])}
            resolved_strategy_overrides = {
                int(key): dict(value)
                for key, value in progress.get("resolved_episode_strategy_overrides", {}).items()
            }
            unrepairable_episode_ids = {
                int(value) for value in progress.get("unrepairable_episode_ids", [])
            }
            unexpected_markers = converted_files - expected_relative_paths
            if unexpected_markers:
                raise ValueError(
                    f"Resume manifest contains unknown converted files: {sorted(unexpected_markers)}."
                )
            for relative_path, source_path in expected_data_files.items():
                output_path = args.output_root / relative_path
                if not output_path.is_file():
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, output_path)
                    converted_files.discard(relative_path)
            logging.info(
                "Resuming conversion in %s (%d parquet file(s) already complete).",
                args.output_root,
                len(converted_files),
            )
        elif not args.overwrite:
            raise FileExistsError(f"{args.output_root} already exists. Use --overwrite to replace it.")
        else:
            shutil.rmtree(args.output_root)
            shutil.copytree(args.input_root, args.output_root)
    else:
        shutil.copytree(args.input_root, args.output_root)

    progress_manifest = {
        "pipeline_version": LIBERO_PIPELINE_VERSION,
        "stage": "delta_to_absolute",
        "conversion_complete": False,
        "action_representation": LIBERO_DELTA_ACTION,
        "state_representation": LIBERO_STATE_14D,
        "observation_materialization": LIBERO_CLOSED_LOOP_ABSOLUTE_MATERIALIZATION,
        "relative_action_ready": False,
        "config": conversion_config,
        "converted_data_files": sorted(converted_files),
        "resolved_episode_strategy_overrides": {
            str(key): value for key, value in sorted(resolved_strategy_overrides.items())
        },
        "unrepairable_episode_ids": sorted(unrepairable_episode_ids),
    }
    write_libero_pipeline_manifest(args.output_root, progress_manifest)

    meta = LeRobotDatasetMetadata(args.output_repo_id, root=args.output_root)
    episodes_df = _episode_rows(meta).set_index("episode_index", drop=False)
    episode_metadata = {
        int(row["episode_index"]): {
            "init_state_id": _init_state_id(row),
            "init_state": _init_state_value(row),
            "suite": _episode_suite(row, args.task),
            "task_id": _episode_task_id(row, args.task_id),
            "source_file": _episode_text(row, "libero/source_file"),
            "source_demo": _episode_text(row, "libero/source_demo"),
            "length": int(row["length"]),
        }
        for _, row in episodes_df.iterrows()
    }

    from lerobot.envs.libero import _get_suite

    task_suites: dict[str, object] = {}
    for suite_name in sorted({str(row["suite"]) for row in episode_metadata.values()}):
        task_suites[suite_name] = _get_suite(suite_name)
    episode_strategy_overrides: dict[int, dict[str, object]] = {}
    if args.episode_strategy_overrides is not None:
        raw_overrides = json.loads(args.episode_strategy_overrides.read_text(encoding="utf-8"))
        episode_strategy_overrides = {int(key): dict(value) for key, value in raw_overrides.items()}

    actual_relative_paths = {
        path.relative_to(args.output_root).as_posix()
        for path in (args.output_root / "data").glob("chunk-*/*.parquet")
    }
    if actual_relative_paths != expected_relative_paths:
        raise ValueError(
            "Output data inventory does not exactly match the immutable input: "
            f"missing={sorted(expected_relative_paths - actual_relative_paths)}, "
            f"unexpected={sorted(actual_relative_paths - expected_relative_paths)}."
        )
    all_data_files = [args.output_root / path for path in sorted(expected_relative_paths)]
    if args.start_file_index < 0:
        raise ValueError("--start-file-index must be non-negative.")
    if args.end_file_index is not None and args.end_file_index < args.start_file_index:
        raise ValueError("--end-file-index must be greater than or equal to --start-file-index.")
    data_files = all_data_files[args.start_file_index : args.end_file_index]
    if not data_files:
        raise ValueError("Selected parquet file range is empty.")

    for parquet_path in tqdm(data_files, desc="Converting action parquet files"):
        relative_path = parquet_path.relative_to(args.output_root).as_posix()
        if relative_path in converted_files:
            logging.info("Skipping already converted parquet during resume: %s", relative_path)
            continue
        # Always read the immutable delta source. This makes --resume idempotent,
        # including when a previously converted parquet is deliberately rerun.
        source_parquet_path = args.input_root / relative_path
        df = pd.read_parquet(source_parquet_path)
        df = _convert_file_with_replay(
            df,
            episode_metadata=episode_metadata,
            task_suites=task_suites,
            observation_width=args.observation_width,
            observation_height=args.observation_height,
            source_hdf5_dir=args.source_hdf5_dir,
            source_action_strategy=args.source_action_strategy,
            pose_offset=args.pose_offset,
            pose_gripper_offset=args.pose_gripper_offset,
            episode_strategy_overrides=episode_strategy_overrides,
            closed_loop_seed=args.closed_loop_seed,
            auto_repair_failed_replays=args.auto_repair_failed_replays,
            resolved_strategy_overrides=resolved_strategy_overrides,
            allow_unrepairable_episodes=args.allow_unrepairable_episodes,
            unrepairable_episode_ids=unrepairable_episode_ids,
        )
        _write_parquet(df, parquet_path, meta)
        converted_files.add(relative_path)
        progress_manifest["converted_data_files"] = sorted(converted_files)
        progress_manifest["resolved_episode_strategy_overrides"] = {
            str(key): value for key, value in sorted(resolved_strategy_overrides.items())
        }
        progress_manifest["unrepairable_episode_ids"] = sorted(unrepairable_episode_ids)
        write_libero_pipeline_manifest(args.output_root, progress_manifest)

    if converted_files != expected_relative_paths:
        remaining = sorted(expected_relative_paths - converted_files)
        logging.warning(
            "Partial conversion complete: %d/%d parquet files. Resume remaining files before use: %s",
            len(converted_files),
            len(expected_relative_paths),
            remaining,
        )
        return

    if unrepairable_episode_ids:
        valid_episode_ids = sorted(set(episode_metadata) - unrepairable_episode_ids)
        write_libero_pipeline_manifest(
            args.output_root,
            {
                **progress_manifest,
                "conversion_complete": True,
                "audit_complete": True,
                "action_representation": "absolute_controller_goal_with_exclusions",
                "observation_materialization": LIBERO_CLOSED_LOOP_ABSOLUTE_MATERIALIZATION,
                "relative_action_ready": False,
                "valid_absolute_episode_ids": valid_episode_ids,
                "unrepairable_episode_ids": sorted(unrepairable_episode_ids),
                "converted_data_files": sorted(converted_files),
            },
        )
        logging.warning(
            "Completed replay audit with %d valid and %d excluded episode(s). "
            "Materialize downstream datasets only from valid_absolute_episode_ids.",
            len(valid_episode_ids),
            len(unrepairable_episode_ids),
        )
        return

    _recompute_materialized_stats(args.output_root, meta)
    write_libero_pipeline_manifest(
        args.output_root,
        {
            **progress_manifest,
            "conversion_complete": True,
            "action_representation": LIBERO_ABSOLUTE_ACTION,
            "observation_materialization": LIBERO_CLOSED_LOOP_ABSOLUTE_MATERIALIZATION,
            "relative_action_ready": True,
            "converted_data_files": sorted(converted_files),
        },
    )
    logging.info("Wrote absolute-action dataset to %s", args.output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("outputs/datasets/libero10_full_v3"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/datasets/libero10_absolute_v3"))
    parser.add_argument("--input-repo-id", default="local/libero10_full_v3")
    parser.add_argument("--output-repo-id", default="local/libero10_absolute_v3")
    parser.add_argument("--task", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--observation-width", type=int, default=128)
    parser.add_argument("--observation-height", type=int, default=128)
    parser.add_argument(
        "--source-hdf5-dir",
        type=Path,
        default=None,
        help="Use each demo's recorded model XML from this directory for exact replay conversion.",
    )
    parser.add_argument(
        "--source-action-strategy",
        choices=("controller-goal", "recorded-eef-pose"),
        default="controller-goal",
        help="How to materialize absolute actions when --source-hdf5-dir is available.",
    )
    parser.add_argument(
        "--pose-offset",
        type=int,
        default=1,
        help="Frame offset for --source-action-strategy=recorded-eef-pose.",
    )
    parser.add_argument(
        "--pose-gripper-offset",
        type=int,
        default=0,
        help="Gripper action offset for --source-action-strategy=recorded-eef-pose.",
    )
    parser.add_argument(
        "--episode-strategy-overrides",
        type=Path,
        default=None,
        help=(
            "JSON mapping episode_index to strategy overrides, e.g. "
            '{"36":{"strategy":"recorded-eef-pose","pose_offset":0,"gripper_offset":0}}.'
        ),
    )
    parser.add_argument(
        "--closed-loop-seed",
        type=int,
        default=1000,
        help="Reset seed used while rematerializing observations in the current LIBERO runtime.",
    )
    parser.add_argument(
        "--auto-repair-failed-replays",
        action="store_true",
        help=(
            "When controller-goal replay fails, test deterministic recorded-eef-pose candidates. "
            "A repair is accepted only if closed-loop absolute replay succeeds and is recorded "
            "in the pipeline manifest."
        ),
    )
    parser.add_argument(
        "--allow-unrepairable-episodes",
        action="store_true",
        help=(
            "Continue the full replay audit after an episode fails every repair. Failed episode ids "
            "are explicitly excluded in the manifest and the intermediate dataset is not marked "
            "relative-action-ready."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue conversion in an existing output root instead of copying input or replacing output.",
    )
    parser.add_argument(
        "--start-file-index",
        type=int,
        default=0,
        help="Start converting from this index in the sorted parquet file list.",
    )
    parser.add_argument(
        "--end-file-index",
        type=int,
        default=None,
        help="Stop before this index in the sorted parquet file list.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    convert_dataset(parse_args())


if __name__ == "__main__":
    main()
