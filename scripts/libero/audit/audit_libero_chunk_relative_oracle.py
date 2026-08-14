#!/usr/bin/env python

"""Replay perfect chunk-relative LIBERO actions through the real eval runtime.

This is an inference-semantics audit, not a learned-policy evaluation. For each
queue fill it encodes the recorded absolute action chunk around the recorded
14D state, decodes it around the live runtime state, and executes the resulting
absolute goals. A closed-loop-rematerialized v3 dataset must pass for every
requested chunk size before Diffusion/MAM overfit results are meaningful.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LIBERO_ASSETS_PATH", str(REPO_ROOT / ".cache/libero/assets"))

from lerobot.datasets.libero_pipeline import (  # noqa: E402
    require_libero_v3_relative_ready_dataset,
)
from lerobot.processor.libero_relative_action_processor import (  # noqa: E402
    absolute_to_chunk_relative,
    axis_angle_to_matrix,
    chunk_relative_to_absolute,
    eef_body_quaternion_to_controller_matrix,
    matrix_to_axis_angle,
)
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402


@dataclass(frozen=True)
class EpisodeSpec:
    episode_index: int
    suite: str
    task_id: int
    init_state: list[float]
    source_episode_id: int | None
    source_file: str | None
    source_demo: str | None


@dataclass(frozen=True)
class OracleResult:
    episode_index: int
    suite: str
    task_id: int
    source_episode_id: int | None
    chunk_size: int
    episode_length: int
    success: bool
    success_step: int | None
    max_anchor_position_error_m: float
    max_anchor_rotation_error_rad: float
    max_goal_position_error_m: float
    max_goal_rotation_error_rad: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("outputs/datasets/libero10_mam_v3_sample_train"),
    )
    parser.add_argument(
        "--episodes",
        default=None,
        help="Optional comma-separated episode_index values; default uses sorted metadata order.",
    )
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument(
        "--chunk-sizes",
        default="1,4,15,full",
        help="Comma-separated positive sizes; 'full' means the complete episode.",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--num-steps-wait", type=int, default=0)
    parser.add_argument("--observation-width", type=int, default=128)
    parser.add_argument("--observation-height", type=int, default=128)
    parser.add_argument(
        "--post-hold-steps",
        type=int,
        default=0,
        help="Optional final-action hold steps. Strict trajectory validation defaults to zero.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/audit/libero_chunk_relative_oracle.json"),
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Write diagnostics but return exit code 0 even if an oracle replay fails.",
    )
    return parser.parse_args()


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _parse_episode_ids(text: str | None) -> set[int] | None:
    if text is None:
        return None
    values = {int(item.strip()) for item in text.split(",") if item.strip()}
    if not values:
        raise ValueError("--episodes selected no episode ids.")
    return values


def _parse_chunk_sizes(text: str, episode_length: int) -> list[int]:
    sizes: list[int] = []
    for item in text.split(","):
        token = item.strip().lower()
        if not token:
            continue
        size = episode_length if token == "full" else int(token)  # nosec B105
        if size <= 0:
            raise ValueError(f"Chunk sizes must be positive, got {size}.")
        if size not in sizes:
            sizes.append(size)
    if not sizes:
        raise ValueError("--chunk-sizes selected no sizes.")
    return sizes


def _load_episode_specs(root: Path, selected: set[int] | None, max_episodes: int | None) -> list[EpisodeSpec]:
    paths = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata parquet found under {root}.")

    specs: list[EpisodeSpec] = []
    for path in paths:
        for row in pq.read_table(path).to_pylist():
            episode_index = int(row["episode_index"])
            if selected is not None and episode_index not in selected:
                continue
            init_state = _first_present(row, ("libero/init_state", "init_state"))
            task_id = _first_present(row, ("libero/task_id", "task_id"))
            if init_state is None or task_id is None:
                raise ValueError(
                    f"Episode {episode_index} needs raw init_state and task_id for an exact oracle replay."
                )
            source_episode_id = _first_present(row, ("libero/source_episode_id", "source_episode_id"))
            specs.append(
                EpisodeSpec(
                    episode_index=episode_index,
                    suite=str(_first_present(row, ("libero/suite", "suite")) or "libero_10"),
                    task_id=int(task_id),
                    init_state=np.asarray(init_state, dtype=np.float64).reshape(-1).tolist(),
                    source_episode_id=(None if source_episode_id is None else int(source_episode_id)),
                    source_file=_first_present(row, ("libero/source_file", "source_file")),
                    source_demo=_first_present(row, ("libero/source_demo", "source_demo")),
                )
            )
    specs.sort(key=lambda item: item.episode_index)
    if selected is not None:
        found = {item.episode_index for item in specs}
        missing = selected - found
        if missing:
            raise ValueError(f"Dataset metadata is missing requested episodes {sorted(missing)}.")
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("--max-episodes must be positive.")
        specs = specs[:max_episodes]
    if not specs:
        raise ValueError("No episodes selected for oracle replay.")
    return specs


def _load_episode_data(root: Path, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
    rows: list[dict[str, Any]] = []
    paths = sorted((root / "data").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No data parquet found under {root}.")
    for path in paths:
        table = pq.read_table(
            path,
            columns=["episode_index", "frame_index", ACTION, OBS_STATE],
            filters=[("episode_index", "=", episode_index)],
        )
        rows.extend(table.to_pylist())
    rows.sort(key=lambda row: int(row["frame_index"]))
    frame_indices = [int(row["frame_index"]) for row in rows]
    if frame_indices != list(range(len(rows))):
        raise ValueError(f"Episode {episode_index} has non-contiguous frame_index values.")
    if not rows:
        raise ValueError(f"Episode {episode_index} has no data rows.")
    actions = np.asarray([row[ACTION] for row in rows], dtype=np.float32)
    states = np.asarray([row[OBS_STATE] for row in rows], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Episode {episode_index} has invalid action shape {actions.shape}.")
    if states.ndim != 2 or states.shape[1] != 14:
        raise ValueError(f"Episode {episode_index} has invalid state shape {states.shape}.")
    return actions, states


def _make_env(spec: EpisodeSpec, args: argparse.Namespace):
    from lerobot.envs.libero import LiberoEnv, _get_suite

    return LiberoEnv(
        task_suite=_get_suite(spec.suite),
        task_id=spec.task_id,
        task_suite_name=spec.suite,
        camera_name="agentview_image,robot0_eye_in_hand_image",
        obs_type="pixels_agent_pos",
        observation_width=args.observation_width,
        observation_height=args.observation_height,
        init_states=True,
        episode_index=0,
        init_state_values=[spec.init_state],
        n_envs=1,
        num_steps_wait=args.num_steps_wait,
        control_mode="absolute",
    )


def _state_from_observation(observation: dict[str, Any]) -> np.ndarray:
    robot_state = observation["robot_state"]
    state = np.concatenate(
        (
            np.asarray(robot_state["eef"]["pos"], dtype=np.float32),
            np.asarray(robot_state["eef"]["quat"], dtype=np.float32),
            np.asarray(robot_state["joints"]["pos"], dtype=np.float32),
        )
    )
    if state.shape != (14,):
        raise ValueError(f"Runtime produced invalid 14D state shape {state.shape}.")
    return state


def _rotation_error_rad(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_matrix = np.asarray(axis_angle_to_matrix(first), dtype=np.float64)
    second_matrix = np.asarray(axis_angle_to_matrix(second), dtype=np.float64)
    relative = first_matrix @ np.swapaxes(second_matrix, -1, -2)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(cosine)


def _anchor_rotation_error_rad(recorded: np.ndarray, live: np.ndarray) -> float:
    recorded_matrix = np.asarray(eef_body_quaternion_to_controller_matrix(recorded[3:7]), dtype=np.float64)
    live_matrix = np.asarray(eef_body_quaternion_to_controller_matrix(live[3:7]), dtype=np.float64)
    relative = recorded_matrix @ live_matrix.T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(cosine)


def _absolute_hold_action(env: Any) -> np.ndarray:
    from lerobot.envs.libero import get_libero_dummy_action

    if env._env is None:
        raise RuntimeError("LIBERO runtime is not initialized.")
    robot = env._env.robots[0]
    action = np.asarray(get_libero_dummy_action(), dtype=np.float32)
    action[:3] = np.asarray(robot.controller.ee_pos, dtype=np.float32)
    action[3:6] = np.asarray(
        matrix_to_axis_angle(np.asarray(robot.controller.ee_ori_mat, dtype=np.float32)),
        dtype=np.float32,
    )
    return action


def _run_oracle(
    env: Any,
    spec: EpisodeSpec,
    actions: np.ndarray,
    states: np.ndarray,
    chunk_size: int,
    args: argparse.Namespace,
) -> OracleResult:
    observation, _ = env.reset(seed=args.seed)
    success = False
    success_step: int | None = None
    anchor_position_errors: list[float] = []
    anchor_rotation_errors: list[float] = []
    goal_position_errors: list[float] = []
    goal_rotation_errors: list[float] = []

    for start in range(0, len(actions), chunk_size):
        stop = min(start + chunk_size, len(actions))
        recorded_anchor = states[start]
        live_anchor = _state_from_observation(observation)
        relative_chunk = absolute_to_chunk_relative(actions[start:stop], recorded_anchor)
        absolute_chunk = np.asarray(chunk_relative_to_absolute(relative_chunk, live_anchor), dtype=np.float32)

        anchor_position_errors.append(float(np.linalg.norm(live_anchor[:3] - recorded_anchor[:3])))
        anchor_rotation_errors.append(_anchor_rotation_error_rad(recorded_anchor, live_anchor))
        goal_position_errors.extend(
            np.linalg.norm(absolute_chunk[:, :3] - actions[start:stop, :3], axis=-1).tolist()
        )
        goal_rotation_errors.extend(
            _rotation_error_rad(absolute_chunk[:, 3:6], actions[start:stop, 3:6]).tolist()
        )

        for offset, action in enumerate(absolute_chunk):
            observation, _, _, _, info = env.step(action)
            if bool(info.get("is_success", False)):
                success = True
                success_step = start + offset
                break
        if success:
            break

    if not success:
        for offset in range(args.post_hold_steps):
            observation, _, _, _, info = env.step(_absolute_hold_action(env))
            if bool(info.get("is_success", False)):
                success = True
                success_step = len(actions) + offset
                break

    return OracleResult(
        episode_index=spec.episode_index,
        suite=spec.suite,
        task_id=spec.task_id,
        source_episode_id=spec.source_episode_id,
        chunk_size=chunk_size,
        episode_length=len(actions),
        success=success,
        success_step=success_step,
        max_anchor_position_error_m=max(anchor_position_errors, default=0.0),
        max_anchor_rotation_error_rad=max(anchor_rotation_errors, default=0.0),
        max_goal_position_error_m=max(goal_position_errors, default=0.0),
        max_goal_rotation_error_rad=max(goal_rotation_errors, default=0.0),
    )


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    manifest = require_libero_v3_relative_ready_dataset(root)
    specs = _load_episode_specs(root, _parse_episode_ids(args.episodes), args.max_episodes)

    results: list[OracleResult] = []
    for spec in specs:
        actions, states = _load_episode_data(root, spec.episode_index)
        env = _make_env(spec, args)
        try:
            for chunk_size in _parse_chunk_sizes(args.chunk_sizes, len(actions)):
                result = _run_oracle(env, spec, actions, states, chunk_size, args)
                results.append(result)
                print(
                    f"episode={spec.episode_index} task={spec.task_id} chunk={chunk_size} "
                    f"success={result.success} success_step={result.success_step} "
                    f"anchor_pos_max={result.max_anchor_position_error_m:.6f}m "
                    f"goal_pos_max={result.max_goal_position_error_m:.6f}m"
                )
        finally:
            env.close()

    payload = {
        "version": 2,
        "dataset_root": str(root),
        "pipeline_version": manifest["pipeline_version"],
        "observation_materialization": manifest["observation_materialization"],
        "relative_action_ready": manifest["relative_action_ready"],
        "seed": args.seed,
        "num_steps_wait": args.num_steps_wait,
        "post_hold_steps": args.post_hold_steps,
        "all_success": all(result.success for result in results),
        "results": [asdict(result) for result in results],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output_json}")
    if not payload["all_success"] and not args.allow_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
