#!/usr/bin/env python

"""Convert a LIBERO LeRobot dataset from OSC delta actions to absolute EEF targets."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.dataset_tools import _write_parquet, recompute_stats
from lerobot.datasets.libero_pipeline import (
    LIBERO_ABSOLUTE_ACTION,
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
from lerobot.utils.constants import ACTION


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
    selected_episode_ids: set[int] | None = None,
) -> pd.DataFrame:
    from libero.libero.envs import OffScreenRenderEnv

    from lerobot.envs.libero import LiberoEnv, sync_libero_controllers
    from lerobot.envs.libero_assets import get_libero_resource_path, validate_libero_assets

    out = df.copy()
    converted_actions: dict[int, np.ndarray] = {}

    for episode_index, ep_df in out.groupby("episode_index", sort=True):
        if selected_episode_ids is not None and int(episode_index) not in selected_episode_ids:
            continue
        ep_meta = episode_metadata[int(episode_index)]
        task_suite_name = str(ep_meta["suite"])
        task_id = int(ep_meta["task_id"])
        task_suite = task_suites[task_suite_name]
        init_state = ep_meta.get("init_state")
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
            **episode_strategy_overrides.get(int(episode_index), {}),
        }
        strategy = str(episode_strategy["strategy"])
        strategy_pose_offset = int(episode_strategy["pose_offset"])
        strategy_gripper_offset = int(episode_strategy["gripper_offset"])

        try:
            for row in ep_df.sort_values("frame_index").itertuples(index=True):
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

    out[ACTION] = [
        converted_actions.get(int(idx), np.asarray(action, dtype=np.float32))
        for idx, action in zip(out.index, out[ACTION], strict=True)
    ]
    return out


def convert_dataset(args: argparse.Namespace) -> None:
    require_libero_v3_action_dataset(
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

    conversion_config = {
        "input_root": str(args.input_root.resolve()),
        "input_repo_id": args.input_repo_id,
        "output_repo_id": args.output_repo_id,
        "source_hdf5_dir": str(args.source_hdf5_dir.resolve()),
        "source_action_strategy": args.source_action_strategy,
        "pose_offset": args.pose_offset,
        "pose_gripper_offset": args.pose_gripper_offset,
        "episode_strategy_overrides": (
            None
            if args.episode_strategy_overrides is None
            else str(args.episode_strategy_overrides.resolve())
        ),
    }

    converted_files: set[str] = set()
    if args.output_root.exists():
        if args.resume:
            progress = read_libero_pipeline_manifest(args.output_root)
            if progress.get("stage") != "delta_to_absolute" or progress.get("config") != conversion_config:
                raise ValueError(
                    f"Cannot resume {args.output_root}: its v3 conversion manifest does not match this run."
                )
            converted_files = {str(path) for path in progress.get("converted_data_files", [])}
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
        "config": conversion_config,
        "converted_data_files": sorted(converted_files),
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

    all_data_files = sorted((args.output_root / "data").glob("chunk-*/*.parquet"))
    if not all_data_files:
        raise FileNotFoundError(f"No parquet data files found under {args.output_root / 'data'}")
    if args.start_file_index < 0:
        raise ValueError("--start-file-index must be non-negative.")
    if args.end_file_index is not None and args.end_file_index < args.start_file_index:
        raise ValueError("--end-file-index must be greater than or equal to --start-file-index.")
    data_files = all_data_files[args.start_file_index : args.end_file_index]
    if not data_files:
        raise ValueError("Selected parquet file range is empty.")

    for parquet_path in tqdm(data_files, desc="Converting action parquet files"):
        relative_path = parquet_path.relative_to(args.output_root).as_posix()
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
        )
        _write_parquet(df, parquet_path, meta)
        converted_files.add(relative_path)
        progress_manifest["converted_data_files"] = sorted(converted_files)
        write_libero_pipeline_manifest(args.output_root, progress_manifest)

    all_relative_paths = {path.relative_to(args.output_root).as_posix() for path in all_data_files}
    if converted_files != all_relative_paths:
        remaining = sorted(all_relative_paths - converted_files)
        logging.warning(
            "Partial conversion complete: %d/%d parquet files. Resume remaining files before use: %s",
            len(converted_files),
            len(all_relative_paths),
            remaining,
        )
        return

    dataset = LeRobotDataset(args.output_repo_id, root=args.output_root)
    recompute_stats(dataset, skip_image_video=True)
    write_libero_pipeline_manifest(
        args.output_root,
        {
            **progress_manifest,
            "conversion_complete": True,
            "action_representation": LIBERO_ABSOLUTE_ACTION,
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
