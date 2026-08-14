#!/usr/bin/env python

"""Replay LIBERO LeRobot demos and report success rates by task.

The script is intentionally dataset-oriented: it replays stored actions from one
or more local LeRobot dataset roots, writes per-demo JSONL, and writes task-level
CSV/JSON summaries. It is useful for validating regenerated LIBERO datasets
where action conversion may make some demonstrations non-replayable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


sys.path.insert(0, str(_repo_root() / "src"))

from lerobot.datasets.libero_pipeline import (  # noqa: E402
    LIBERO_ABSOLUTE_ACTION,
    require_libero_v3_action_dataset,
)
from lerobot.processor.libero_relative_action_processor import (  # noqa: E402
    chunk_relative_to_absolute,
    matrix_to_axis_angle,
)


@dataclass(frozen=True)
class DemoRef:
    key: str
    root: Path
    root_index: int
    dataset: str
    episode_index: int
    source_episode_id: int | None
    suite: str
    task_id: int
    task: str
    length: int | None
    init_state: Any
    init_state_id: Any
    source_file: str | None
    source_demo: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        action="append",
        default=None,
        help="Local LeRobot dataset root. May be passed multiple times.",
    )
    parser.add_argument("--suite", default="libero_10", help="Fallback suite name.")
    parser.add_argument("--action-key", default="action", help="Dataset column to replay.")
    parser.add_argument(
        "--action-transform",
        choices=("auto", "none", "chunk-relative-to-absolute"),
        default="none",
        help=(
            "Transform stored actions before replay. LIBERO dataset actions should usually "
            "already be absolute controller goals, so the default is 'none'."
        ),
    )
    parser.add_argument("--control-mode", choices=("relative", "absolute"), default="absolute")
    parser.add_argument(
        "--backend",
        choices=("direct", "lerobot"),
        default="direct",
        help="Use direct LIBERO env without camera observations, or the full LeRobot env wrapper.",
    )
    parser.add_argument("--episodes", default=None, help="Comma-separated episode ids to replay.")
    parser.add_argument(
        "--source-episodes",
        default=None,
        help="Comma-separated original source episode ids to replay across one or more splits.",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--num-steps-wait", type=int, default=0)
    parser.add_argument("--post-noop-steps", type=int, default=50)
    parser.add_argument("--observation-width", type=int, default=128)
    parser.add_argument("--observation-height", type=int, default=128)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/audit/libero10_v3_replay"),
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument(
        "--source-hdf5-dir",
        type=Path,
        default=None,
        help="For direct backend, reset each demo from its original LIBERO model XML and init_state.",
    )
    return parser.parse_args()


def _default_roots() -> list[Path]:
    train = Path("outputs/datasets/libero10_mam_v3_sample_train")
    eval_ = Path("outputs/datasets/libero10_mam_v3_sample_eval")
    if train.exists() and eval_.exists():
        return [train, eval_]
    return [Path("outputs/datasets/libero10_absolute_v3_sample")]


def _scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot JSON encode {type(value)}")


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _task_text(row: dict[str, Any]) -> str:
    value = _first_present(row, ("tasks", "task", "libero/task_name", "task_name"))
    value = _scalar(value)
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return "" if value is None else str(value)


def _source_episode_id(row: dict[str, Any]) -> int | None:
    value = _first_present(row, ("libero/source_episode_id", "source_episode_id"))
    if value is None:
        return None
    return int(_scalar(value))


def _parse_episode_ids(text: str | None) -> set[int] | None:
    if not text:
        return None
    return {int(part.strip()) for part in text.split(",") if part.strip()}


def _read_demo_refs(root: Path, root_index: int, args: argparse.Namespace) -> list[DemoRef]:
    paths = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata parquet found under {root / 'meta' / 'episodes'}")

    refs: list[DemoRef] = []
    for path in paths:
        for source_row in pq.read_table(path).to_pylist():
            row = dict(source_row)
            episode_index = int(row["episode_index"])
            task_id_value = _first_present(row, ("libero/task_id", "task_id"))
            if task_id_value is None:
                raise ValueError(f"{root}: episode {episode_index} has no libero/task_id.")
            init_state = _first_present(row, ("libero/init_state", "init_state"))
            init_state_id = _first_present(row, ("libero/init_state_id", "init_state_id"))
            if init_state is None and init_state_id is None:
                raise ValueError(f"{root}: episode {episode_index} has no init_state or init_state_id.")
            source_episode_id = _source_episode_id(row)
            dataset = root.name
            key = f"{root_index}:{episode_index}"
            refs.append(
                DemoRef(
                    key=key,
                    root=root,
                    root_index=root_index,
                    dataset=dataset,
                    episode_index=episode_index,
                    source_episode_id=source_episode_id,
                    suite=str(_scalar(_first_present(row, ("libero/suite", "suite"))) or args.suite),
                    task_id=int(_scalar(task_id_value)),
                    task=_task_text(row),
                    length=None if row.get("length") is None else int(row["length"]),
                    init_state=init_state,
                    init_state_id=init_state_id,
                    source_file=_scalar(_first_present(row, ("libero/source_file", "source_file"))),
                    source_demo=_scalar(_first_present(row, ("libero/source_demo", "source_demo"))),
                )
            )
    return refs


def _read_all_demo_refs(args: argparse.Namespace) -> list[DemoRef]:
    roots = args.dataset_root if args.dataset_root else _default_roots()
    refs: list[DemoRef] = []
    for root_index, root in enumerate(roots):
        refs.extend(_read_demo_refs(root, root_index, args))
    selected_episode_ids = _parse_episode_ids(args.episodes)
    if selected_episode_ids is not None:
        refs = [ref for ref in refs if ref.episode_index in selected_episode_ids]
    selected_source_episode_ids = _parse_episode_ids(args.source_episodes)
    if selected_source_episode_ids is not None:
        refs = [ref for ref in refs if ref.source_episode_id in selected_source_episode_ids]
    refs.sort(
        key=lambda ref: (
            ref.source_episode_id is None,
            ref.source_episode_id if ref.source_episode_id is not None else ref.root_index,
            ref.root_index,
            ref.episode_index,
        )
    )
    if args.max_episodes is not None:
        refs = refs[: args.max_episodes]
    if not refs:
        raise ValueError("No demos selected.")
    return refs


def _read_actions_for_root(
    root: Path,
    root_index: int,
    episode_ids: set[int],
    action_key: str,
    action_transform: str,
) -> dict[str, np.ndarray]:
    paths = sorted((root / "data").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No data parquet found under {root / 'data'}")

    schema_names = set(pq.read_schema(paths[0]).names)
    frames: dict[int, list[tuple[int, np.ndarray, np.ndarray | None]]] = defaultdict(list)
    resolved_transform = action_transform
    if resolved_transform == "auto":
        # An auxiliary MAM column does not describe the selected action column.
        # Converted MAM datasets retain already-absolute actions in `action`.
        resolved_transform = "none"
    if resolved_transform == "chunk-relative-to-absolute" and "observation.state" not in schema_names:
        raise ValueError(f"{root}: action transform requires observation.state column.")
    print(f"{root.name}: action_transform={resolved_transform}")

    columns = ["episode_index", "frame_index", action_key]
    if resolved_transform == "chunk-relative-to-absolute":
        columns.append("observation.state")
    for path in tqdm(paths, desc=f"read {root.name} parquet"):
        table = pq.read_table(path, columns=columns)
        for row in table.to_pylist():
            episode_index = int(row["episode_index"])
            if episode_index not in episode_ids:
                continue
            action = np.asarray(row[action_key], dtype=np.float32)
            state = None
            if resolved_transform == "chunk-relative-to-absolute":
                state = np.asarray(row["observation.state"], dtype=np.float32)
            frames[episode_index].append((int(row["frame_index"]), action, state))

    actions: dict[str, np.ndarray] = {}
    for episode_index, items in frames.items():
        items.sort(key=lambda item: item[0])
        frame_indices = [item[0] for item in items]
        expected = list(range(len(items)))
        if frame_indices != expected:
            raise ValueError(f"{root}: episode {episode_index} has non-contiguous frame_index.")
        episode_actions = np.stack([item[1] for item in items], axis=0)
        if resolved_transform == "chunk-relative-to-absolute":
            episode_states = np.stack([item[2] for item in items], axis=0)
            episode_actions = np.asarray(
                chunk_relative_to_absolute(episode_actions, episode_states),
                dtype=np.float32,
            )
        actions[f"{root_index}:{episode_index}"] = episode_actions

    missing = sorted(episode_ids - set(frames))
    if missing:
        raise ValueError(f"{root}: missing action frames for episodes {missing[:20]}.")
    return actions


def _read_actions(refs: list[DemoRef], args: argparse.Namespace) -> dict[str, np.ndarray]:
    by_root: dict[tuple[int, Path], set[int]] = defaultdict(set)
    for ref in refs:
        by_root[(ref.root_index, ref.root)].add(ref.episode_index)

    actions: dict[str, np.ndarray] = {}
    for (root_index, root), episode_ids in sorted(by_root.items(), key=lambda item: item[0][0]):
        actions.update(
            _read_actions_for_root(
                root,
                root_index,
                episode_ids,
                args.action_key,
                args.action_transform,
            )
        )
    return actions


def _manifest(args: argparse.Namespace, refs: list[DemoRef]) -> dict[str, Any]:
    return {
        "version": 1,
        "dataset_roots": [str(root.resolve()) for root in (args.dataset_root or _default_roots())],
        "demo_keys": [ref.key for ref in refs],
        "action_key": args.action_key,
        "action_transform": args.action_transform,
        "control_mode": args.control_mode,
        "backend": args.backend,
        "episodes": args.episodes,
        "source_episodes": args.source_episodes,
        "seed": args.seed,
        "num_steps_wait": args.num_steps_wait,
        "post_noop_steps": args.post_noop_steps,
        "observation_width": args.observation_width,
        "observation_height": args.observation_height,
        "source_hdf5_dir": None if args.source_hdf5_dir is None else str(args.source_hdf5_dir.resolve()),
    }


def _prepare_output(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    if args.overwrite_output and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError(
                f"Output manifest differs: {manifest_path}. "
                "Use another --output-dir or pass --overwrite-output."
            )
        if not args.resume:
            raise FileExistsError(
                f"{args.output_dir} already contains a run; use --resume or --overwrite-output."
            )
    else:
        manifest_path.write_text(
            json.dumps(manifest, default=_jsonable, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row["demo_key"])] = row
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, default=_jsonable, ensure_ascii=False) + "\n")
        file.flush()


class _DirectReplayEnv:
    def __init__(
        self,
        *,
        suite_name: str,
        task_id: int,
        refs: list[DemoRef],
        control_mode: str,
        num_steps_wait: int,
        source_hdf5_dir: Path | None,
    ) -> None:
        import h5py
        from libero.libero.envs import OffScreenRenderEnv

        from lerobot.envs.libero import (
            _get_suite,
            get_libero_dummy_action,
            get_task_init_states,
            sync_libero_controllers,
        )
        from lerobot.envs.libero_assets import (
            get_libero_resource_path,
            rewrite_libero_demo_xml_paths,
            validate_libero_assets,
        )

        validate_libero_assets()
        task_suite = _get_suite(suite_name)
        task = task_suite.get_task(task_id)
        bddl_file = get_libero_resource_path("bddl_files") / task.problem_folder / task.bddl_file
        self._env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            use_camera_obs=False,
            camera_heights=1,
            camera_widths=1,
        )
        self._env.reset()
        self.task_id = task_id
        self.task = task.name
        self.control_mode = control_mode
        self.num_steps_wait = num_steps_wait
        self._dummy_action = np.asarray(get_libero_dummy_action(), dtype=np.float32)
        self._sync_controllers = sync_libero_controllers
        self._refs = refs
        self._source_hdf5_dir = source_hdf5_dir
        self._h5py = h5py
        self._rewrite_libero_demo_xml_paths = rewrite_libero_demo_xml_paths
        self._cursor = 0

        use_raw_init_state = refs[0].init_state is not None
        if any((ref.init_state is not None) != use_raw_init_state for ref in refs):
            raise ValueError(f"Task {suite_name}/{task_id} mixes raw init states and init state ids.")
        self._init_state_values: list[np.ndarray] | None
        if use_raw_init_state:
            self._init_state_values = [
                np.asarray(ref.init_state, dtype=np.float64).reshape(-1) for ref in refs
            ]
            self._init_states = None
            self._init_state_ids = None
        else:
            self._init_state_values = None
            self._init_states = get_task_init_states(task_suite, task_id)
            self._init_state_ids = [int(_scalar(ref.init_state_id)) for ref in refs]

    def reset(self, seed: int | None = None):
        ref = self._refs[self._cursor % len(self._refs)]
        self._cursor += 1
        self._env.seed(seed)
        raw_obs = self._env.reset()
        if self._source_hdf5_dir is not None:
            if ref.source_file is None or ref.source_demo is None:
                raise ValueError(f"{ref.root}: episode {ref.episode_index} lacks source HDF5 metadata.")
            with self._h5py.File(self._source_hdf5_dir / ref.source_file, "r") as source_h5:
                source_group = source_h5[f"data/{ref.source_demo}"]
                model_xml = self._rewrite_libero_demo_xml_paths(str(source_group.attrs["model_file"]))
                init_state = np.asarray(source_group.attrs["init_state"], dtype=np.float64)
            raw_obs = self._env.reset_from_xml_string(model_xml)
            self._env.sim.reset()
            raw_obs = self._env.set_init_state(init_state)
        elif self._init_state_values is not None:
            init_state = self._init_state_values[(self._cursor - 1) % len(self._init_state_values)]
            raw_obs = self._env.set_init_state(init_state)
        else:
            assert self._init_states is not None
            assert self._init_state_ids is not None
            init_state_id = self._init_state_ids[(self._cursor - 1) % len(self._init_state_ids)]
            raw_obs = self._env.set_init_state(self._init_states[init_state_id % len(self._init_states)])

        self._sync_controllers(self._env)

        for _ in range(self.num_steps_wait):
            raw_obs, _, _, _ = self._env.step(self._dummy_action)
        self._sync_controllers(self._env)

        if self.control_mode == "absolute":
            for robot in self._env.robots:
                robot.controller.use_delta = False
        elif self.control_mode == "relative":
            for robot in self._env.robots:
                robot.controller.use_delta = True
        else:
            raise ValueError(f"Invalid control mode: {self.control_mode}")
        return raw_obs, {"is_success": False}

    def step(self, action: np.ndarray):
        raw_obs, reward, done, info = self._env.step(action)
        is_success = self._env.check_success()
        info.update(
            {
                "task": self.task,
                "task_id": self.task_id,
                "done": done,
                "is_success": is_success,
            }
        )
        return raw_obs, reward, bool(done or is_success), False, info

    def close(self) -> None:
        self._env.close()


def _make_lerobot_env(suite_name: str, task_id: int, refs: list[DemoRef], args: argparse.Namespace):
    from lerobot.envs.libero import LiberoEnv, _get_suite

    use_raw_init_state = refs[0].init_state is not None
    if any((ref.init_state is not None) != use_raw_init_state for ref in refs):
        raise ValueError(f"Task {suite_name}/{task_id} mixes raw init states and init state ids.")

    init_state_values = None
    init_state_ids = None
    if use_raw_init_state:
        init_state_values = [
            np.asarray(ref.init_state, dtype=np.float64).reshape(-1).tolist() for ref in refs
        ]
    else:
        init_state_ids = [int(_scalar(ref.init_state_id)) for ref in refs]

    return LiberoEnv(
        task_suite=_get_suite(suite_name),
        task_id=task_id,
        task_suite_name=suite_name,
        camera_name="agentview_image,robot0_eye_in_hand_image",
        obs_type="pixels_agent_pos",
        observation_width=args.observation_width,
        observation_height=args.observation_height,
        init_states=True,
        episode_index=0,
        init_state_values=init_state_values,
        init_state_ids=init_state_ids,
        n_envs=1,
        num_steps_wait=args.num_steps_wait,
        control_mode=args.control_mode,
    )


def _make_env(suite_name: str, task_id: int, refs: list[DemoRef], args: argparse.Namespace):
    if args.backend == "direct":
        return _DirectReplayEnv(
            suite_name=suite_name,
            task_id=task_id,
            refs=refs,
            control_mode=args.control_mode,
            num_steps_wait=args.num_steps_wait,
            source_hdf5_dir=args.source_hdf5_dir,
        )
    return _make_lerobot_env(suite_name, task_id, refs, args)


def _relative_noop() -> np.ndarray:
    from lerobot.envs.libero import get_libero_dummy_action

    return np.asarray(get_libero_dummy_action(), dtype=np.float32)


def _absolute_hold_action(env: Any) -> np.ndarray:
    assert env._env is not None
    robot = env._env.robots[0]
    hold = _relative_noop()
    hold[:3] = np.asarray(robot.controller.ee_pos, dtype=np.float32)
    hold[3:6] = np.asarray(
        matrix_to_axis_angle(np.asarray(robot.controller.ee_ori_mat, dtype=np.float32)),
        dtype=np.float32,
    )
    return hold


def _replay_one(env: Any, actions: np.ndarray, args: argparse.Namespace) -> tuple[bool, int | None]:
    env.reset(seed=args.seed)
    for step, action in enumerate(actions):
        _, _, _, _, info = env.step(action.astype(np.float32))
        if bool(info.get("is_success", False)):
            return True, step

    for offset in range(args.post_noop_steps):
        action = _absolute_hold_action(env) if args.control_mode == "absolute" else _relative_noop()
        _, _, _, _, info = env.step(action.astype(np.float32))
        if bool(info.get("is_success", False)):
            return True, len(actions) + offset
    return False, None


def _result_row(ref: DemoRef, actions: np.ndarray, success: bool, success_step: int | None) -> dict[str, Any]:
    return {
        "demo_key": ref.key,
        "dataset": ref.dataset,
        "dataset_root": str(ref.root),
        "episode_index": ref.episode_index,
        "source_episode_id": ref.source_episode_id,
        "suite": ref.suite,
        "task_id": ref.task_id,
        "task": ref.task,
        "length": ref.length if ref.length is not None else len(actions),
        "replayed_steps": len(actions),
        "success": success,
        "success_step": success_step,
    }


def _run_replay(
    refs: list[DemoRef],
    actions: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    result_path = args.output_dir / "episodes.jsonl"
    results = _read_jsonl(result_path) if args.resume else {}

    groups: dict[tuple[str, int], list[DemoRef]] = defaultdict(list)
    for ref in refs:
        if ref.key not in results:
            groups[(ref.suite, ref.task_id)].append(ref)

    for (suite_name, task_id), group in sorted(groups.items()):
        group.sort(
            key=lambda ref: (
                ref.source_episode_id is None,
                ref.source_episode_id if ref.source_episode_id is not None else ref.root_index,
                ref.root_index,
                ref.episode_index,
            )
        )
        env = _make_env(suite_name, task_id, group, args)
        try:
            for ref in tqdm(group, desc=f"replay {suite_name}/{task_id}"):
                success, success_step = _replay_one(env, actions[ref.key], args)
                row = _result_row(ref, actions[ref.key], success, success_step)
                _append_jsonl(result_path, row)
                results[ref.key] = row
        finally:
            env.close()
    return results


def _write_summary(refs: list[DemoRef], results: dict[str, dict[str, Any]], args: argparse.Namespace) -> None:
    grouped: dict[tuple[str, int, str], list[DemoRef]] = defaultdict(list)
    for ref in refs:
        grouped[(ref.suite, ref.task_id, ref.task)].append(ref)

    task_rows: list[dict[str, Any]] = []
    for (suite, task_id, task), group in sorted(grouped.items()):
        success = sum(bool(results[ref.key]["success"]) for ref in group)
        total = len(group)
        task_rows.append(
            {
                "suite": suite,
                "task_id": task_id,
                "task": task,
                "total": total,
                "success": success,
                "failure": total - success,
                "success_rate": success / total if total else 0.0,
            }
        )

    csv_path = args.output_dir / "task_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(task_rows)

    total = len(refs)
    success = sum(bool(results[ref.key]["success"]) for ref in refs)
    summary = {
        "dataset_roots": [str(root) for root in (args.dataset_root or _default_roots())],
        "action_key": args.action_key,
        "action_transform": args.action_transform,
        "control_mode": args.control_mode,
        "backend": args.backend,
        "total": total,
        "success": success,
        "failure": total - success,
        "success_rate": success / total if total else 0.0,
        "task_summary": task_rows,
        "episodes_jsonl": str(args.output_dir / "episodes.jsonl"),
        "task_summary_csv": str(csv_path),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, default=_jsonable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n| task_id | task | total | success | failure | success_rate |")
    print("|---:|---|---:|---:|---:|---:|")
    for row in task_rows:
        print(
            f"| {row['task_id']} | {row['task']} | {row['total']} | "
            f"{row['success']} | {row['failure']} | {row['success_rate']:.2%} |"
        )
    print(f"\nTotal: {success}/{total} success, {total - success}/{total} failure.")
    print(f"JSONL: {args.output_dir / 'episodes.jsonl'}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {summary_path}")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("LIBERO_ASSETS_PATH", str(_repo_root() / ".cache/libero/assets"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(_repo_root() / ".cache/numba"))
    os.environ.setdefault("MPLCONFIGDIR", str(_repo_root() / ".cache/matplotlib"))

    for root in args.dataset_root or _default_roots():
        require_libero_v3_action_dataset(root, action_representation=LIBERO_ABSOLUTE_ACTION)

    refs = _read_all_demo_refs(args)
    manifest = _manifest(args, refs)
    _prepare_output(args, manifest)
    actions = _read_actions(refs, args)
    print(
        f"Selected {len(refs)} demos from "
        f"{', '.join(str(root) for root in (args.dataset_root or _default_roots()))}; "
        f"action_key={args.action_key}; action_transform={args.action_transform}; "
        f"control_mode={args.control_mode}; backend={args.backend}"
    )
    results = _run_replay(refs, actions, args)
    _write_summary(refs, results, args)


if __name__ == "__main__":
    main()
