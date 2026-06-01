#!/usr/bin/env python

"""Convert a LIBERO LeRobot dataset from OSC delta actions to absolute EEF targets."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.dataset_tools import _write_parquet, recompute_stats
from lerobot.processor.libero_relative_action_processor import (
    axis_angle_to_matrix,
    delta_to_absolute_action,
)
from lerobot.utils.constants import ACTION, OBS_STATE


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


def _convert_from_recorded_state(delta: np.ndarray, state: np.ndarray) -> np.ndarray:
    eef_pos = state[:3]
    eef_mat = axis_angle_to_matrix(state[3:6])
    return np.asarray(delta_to_absolute_action(delta, eef_pos, eef_mat), dtype=np.float32)


def _convert_file_from_recorded_state(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[ACTION] = [
        _convert_from_recorded_state(np.asarray(delta, dtype=np.float32), np.asarray(state, dtype=np.float32))
        for delta, state in zip(out[ACTION], out[OBS_STATE], strict=True)
    ]
    return out


def _convert_file_with_replay(
    df: pd.DataFrame,
    *,
    task_suite,
    task_suite_name: str,
    task_id: int,
    init_state_ids: dict[int, int],
    observation_width: int,
    observation_height: int,
) -> pd.DataFrame:
    from lerobot.envs.libero import LiberoEnv

    out = df.copy()
    converted_actions: dict[int, np.ndarray] = {}

    for episode_index, ep_df in out.groupby("episode_index", sort=True):
        env = LiberoEnv(
            task_suite=task_suite,
            task_id=task_id,
            task_suite_name=task_suite_name,
            init_state_id=init_state_ids[int(episode_index)],
            observation_width=observation_width,
            observation_height=observation_height,
            control_mode="relative",
        )
        env.reset()
        assert env._env is not None
        robot = env._env.robots[0]

        try:
            for row in ep_df.sort_values("frame_index").itertuples(index=True):
                delta = np.asarray(getattr(row, ACTION), dtype=np.float32)
                eef_pos = np.asarray(robot.controller.ee_pos, dtype=np.float32)
                eef_mat = np.asarray(robot.controller.ee_ori_mat, dtype=np.float32)
                converted_actions[int(row.Index)] = np.asarray(
                    delta_to_absolute_action(delta, eef_pos, eef_mat), dtype=np.float32
                )
                env._env.step(delta)
        finally:
            env.close()

    out[ACTION] = [converted_actions[int(idx)] for idx in out.index]
    return out


def convert_dataset(args: argparse.Namespace) -> None:
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} already exists. Use --overwrite to replace it.")
        shutil.rmtree(args.output_root)

    shutil.copytree(args.input_root, args.output_root)
    meta = LeRobotDatasetMetadata(args.output_repo_id, root=args.output_root)
    episodes_df = _episode_rows(meta).set_index("episode_index", drop=False)
    init_state_ids = {int(row["episode_index"]): _init_state_id(row) for _, row in episodes_df.iterrows()}

    task_suite = None
    if args.replay:
        from lerobot.envs.libero import _get_suite

        task_suite = _get_suite(args.task)

    data_files = sorted((args.output_root / "data").glob("chunk-*/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet data files found under {args.output_root / 'data'}")

    for parquet_path in tqdm(data_files, desc="Converting action parquet files"):
        df = pd.read_parquet(parquet_path)
        if args.replay:
            df = _convert_file_with_replay(
                df,
                task_suite=task_suite,
                task_suite_name=args.task,
                task_id=args.task_id,
                init_state_ids=init_state_ids,
                observation_width=args.observation_width,
                observation_height=args.observation_height,
            )
        else:
            df = _convert_file_from_recorded_state(df)
        _write_parquet(df, parquet_path, meta)

    dataset = LeRobotDataset(args.output_repo_id, root=args.output_root)
    recompute_stats(dataset, skip_image_video=True)
    logging.info("Wrote absolute-action dataset to %s", args.output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("outputs/datasets/libero_put_bowl_on_plate"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/datasets/libero_put_bowl_on_plate_absolute")
    )
    parser.add_argument("--input-repo-id", default="local/libero_put_bowl_on_plate")
    parser.add_argument("--output-repo-id", default="local/libero_put_bowl_on_plate_absolute")
    parser.add_argument("--task", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument("--observation-width", type=int, default=256)
    parser.add_argument("--observation-height", type=int, default=256)
    parser.add_argument("--replay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    convert_dataset(parse_args())


if __name__ == "__main__":
    main()
