#!/usr/bin/env python

"""Convert raw LIBERO-10 HDF5 demos to a LeRobot dataset.

Expected official LIBERO demo layout:
  data/demo_*/actions
  data/demo_*/obs/agentview_rgb
  data/demo_*/obs/eye_in_hand_rgb
  data/demo_*/obs/ee_states
  data/demo_*/obs/joint_states

The script validates keys per file and reports available datasets when a key is
missing, so local variants can be fixed by passing explicit key arguments.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.libero_pipeline import (
    LIBERO_DELTA_ACTION,
    LIBERO_PIPELINE_VERSION,
    LIBERO_STATE_14D,
    write_libero_pipeline_manifest,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DEFAULT_AGENTVIEW_KEYS = (
    "obs/agentview_rgb",
    "obs/agentview_image",
    "obs/agentview",
    "agentview_rgb",
    "agentview_image",
)
DEFAULT_WRIST_KEYS = (
    "obs/eye_in_hand_rgb",
    "obs/eye_in_hand_image",
    "obs/robot0_eye_in_hand_rgb",
    "obs/robot0_eye_in_hand_image",
    "eye_in_hand_rgb",
    "robot0_eye_in_hand_image",
)
DEFAULT_EE_KEYS = ("obs/ee_states", "ee_states")
DEFAULT_GRIPPER_KEYS = ("obs/gripper_states", "gripper_states")
DEFAULT_JOINT_KEYS = (
    "obs/joint_states",
    "obs/robot0_joint_pos",
    "obs/joint_pos",
    "joint_states",
    "robot0_joint_pos",
    "joint_pos",
)
DEFAULT_ACTION_KEYS = ("actions", "action")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/libero_10"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/datasets/libero10_full_v3")
    )
    parser.add_argument("--output-repo-id", default="local/libero10_full_v3")
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    parser.add_argument(
        "--demo-index-map-json",
        type=Path,
        default=None,
        help=(
            "Optional mapping from task id/filename to explicit demo indices for audited subsets. "
            "An explicit list replaces prefix selection for that task and may not exceed "
            "--max-episodes-per-task."
        ),
    )
    parser.add_argument("--task-id-map-json", type=Path, default=None)
    parser.add_argument("--agentview-key", default=None)
    parser.add_argument("--wrist-key", default=None)
    parser.add_argument("--ee-key", default=None)
    parser.add_argument("--gripper-key", default=None)
    parser.add_argument("--joint-key", default=None)
    parser.add_argument("--action-key", default=None)
    parser.add_argument("--use-videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _task_table(suite_name: str) -> list[dict[str, Any]]:
    try:
        from lerobot.envs.libero import _get_suite
    except Exception as exc:  # pragma: no cover - depends on optional LIBERO install
        logging.warning("Could not import LIBERO suite metadata: %s", exc)
        return []

    suite = _get_suite(suite_name)
    rows = []
    for task_id, task in enumerate(suite.tasks):
        rows.append(
            {
                "task_id": task_id,
                "task_name": str(task.name),
                "language": str(task.language),
                "bddl_file": str(task.bddl_file),
            }
        )
    return rows


def _hdf5_files(input_dir: Path) -> list[Path]:
    files = sorted([*input_dir.glob("*.hdf5"), *input_dir.glob("*.h5")])
    if not files:
        raise FileNotFoundError(f"No .hdf5/.h5 files found under {input_dir}")
    return files


def _load_task_id_map(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): int(value) for key, value in data.items()}


def _load_demo_index_map(path: Path | None) -> dict[str, list[int]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping = {str(key): [int(index) for index in value] for key, value in data.items()}
    for key, indices in mapping.items():
        if not indices or len(indices) != len(set(indices)) or any(index < 0 for index in indices):
            raise ValueError(f"Invalid demo indices for {key!r}: {indices}.")
    return mapping


def _assign_demo_index_map(
    files: list[Path],
    file_tasks: dict[Path, dict[str, Any]],
    demo_index_map: dict[str, list[int]],
    max_episodes_per_task: int | None,
) -> dict[Path, list[int]]:
    assigned: dict[Path, list[int]] = {}
    consumed: set[str] = set()
    for hdf5_path in files:
        task = file_tasks[hdf5_path]
        matching_keys = [
            key
            for key in (str(task["task_id"]), hdf5_path.name, hdf5_path.stem)
            if key in demo_index_map
        ]
        if len(matching_keys) > 1:
            raise ValueError(
                f"Demo index map has ambiguous aliases for {hdf5_path.name}: "
                f"{matching_keys}. Use exactly one key per task."
            )
        if not matching_keys:
            continue
        key = matching_keys[0]
        indices = demo_index_map[key]
        if max_episodes_per_task is not None and len(indices) > max_episodes_per_task:
            raise ValueError(
                f"Demo index map selects {len(indices)} demos for {hdf5_path.name}, "
                f"exceeding --max-episodes-per-task={max_episodes_per_task}."
            )
        assigned[hdf5_path] = indices
        consumed.add(key)

    unused_keys = sorted(set(demo_index_map) - consumed)
    if unused_keys:
        raise ValueError(
            "Demo index map contains keys that did not match any resolved task/file: "
            f"{unused_keys}."
        )
    return assigned


def _resolve_file_tasks(
    files: list[Path],
    suite_name: str,
    task_id_map: dict[str, int],
) -> dict[Path, dict[str, Any]]:
    table = _task_table(suite_name)
    if not table and len(task_id_map) < len(files):
        raise RuntimeError(
            "Could not load LIBERO suite metadata, and --task-id-map-json does not cover all input files. "
            "Refusing to infer LIBERO task ids from sorted filenames because that silently corrupts "
            "multi-task eval metadata."
        )
    by_id = {int(row["task_id"]): row for row in table}
    resolved = {}
    for path in files:
        if path.name in task_id_map:
            task_id = task_id_map[path.name]
        elif path.stem in task_id_map:
            task_id = task_id_map[path.stem]
        else:
            stem_norm = _normalize_text(path.stem)
            matches = [
                row
                for row in table
                if _normalize_text(row["language"]) in stem_norm
                or _normalize_text(Path(row["bddl_file"]).stem) in stem_norm
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Could not resolve exactly one LIBERO task for {path.name}; got {len(matches)} "
                    "matches. Pass --task-id-map-json to make the mapping explicit."
                )
            task_id = int(matches[0]["task_id"])
        task_row = by_id.get(
            task_id,
            {
                "task_id": task_id,
                "task_name": path.stem,
                "language": path.stem.replace("_", " "),
                "bddl_file": "",
            },
        )
        resolved[path] = task_row
    return resolved


def _demo_groups(h5: h5py.File) -> list[tuple[str, h5py.Group]]:
    root = h5.get("data", h5)
    groups = [(name, obj) for name, obj in root.items() if isinstance(obj, h5py.Group)]
    groups.sort(
        key=lambda item: (
            (
                0,
                int(re.search(r"\d+", item[0]).group(0)),
            )
            if re.search(r"\d+", item[0])
            else (1, item[0])
        )
    )
    if not groups:
        raise ValueError("No demo groups found. Expected groups under 'data/demo_*'.")
    return groups


def _dataset_paths(group: h5py.Group) -> list[str]:
    paths: list[str] = []

    def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset):
            paths.append(name)

    group.visititems(visitor)
    return sorted(paths)


def _find_dataset(group: h5py.Group, requested: str | None, aliases: tuple[str, ...]) -> h5py.Dataset:
    candidates = (requested,) if requested else aliases
    for key in candidates:
        if key and key in group and isinstance(group[key], h5py.Dataset):
            return group[key]
    paths = _dataset_paths(group)
    basenames = {Path(path).name: path for path in paths}
    for key in candidates:
        if key and Path(key).name in basenames:
            return group[basenames[Path(key).name]]
    raise KeyError(f"Missing HDF5 dataset. Tried {list(candidates)}. Available datasets: {paths}")


def _as_hwc_uint8(image: np.ndarray, height: int, width: int) -> np.ndarray:
    import cv2

    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape={image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        image = (
            np.clip(image, 0, 1) * 255 if np.issubdtype(image.dtype, np.floating) else np.clip(image, 0, 255)
        )
        image = image.astype(np.uint8)
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def _quat_xyzw_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / np.clip(norm, 1e-8, None)
    xyz = quat[..., :3]
    w = np.clip(quat[..., 3], -1.0, 1.0)
    den = np.sqrt(np.clip(1.0 - w * w, 0.0, None))
    angle = 2.0 * np.arccos(w)
    axis = np.zeros_like(xyz)
    mask = den > 1e-8
    axis[mask] = xyz[mask] / den[mask, None]
    return axis * angle[..., None]


def _axis_angle_to_quat_xyzw(axis_angle: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(axis_angle, dtype=np.float32)
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    half_angle = angle * 0.5
    scale = np.where(angle > 1e-8, np.sin(half_angle) / angle, 0.5 - angle * angle / 48.0)
    quat = np.concatenate([rotvec * scale, np.cos(half_angle)], axis=-1)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return (quat / np.clip(norm, 1e-8, None)).astype(np.float32)


def _state_from_demo(group: h5py.Group, args: argparse.Namespace) -> np.ndarray:
    ee = np.asarray(_find_dataset(group, args.ee_key, DEFAULT_EE_KEYS), dtype=np.float32)
    joints = np.asarray(_find_dataset(group, args.joint_key, DEFAULT_JOINT_KEYS), dtype=np.float32)
    if ee.ndim != 2 or ee.shape[1] not in {6, 7}:
        raise ValueError(f"Expected ee_states shape (T,6) or (T,7), got {ee.shape}")
    if ee.shape[1] == 7:
        eef_quat = ee[:, 3:7]
        eef_quat = eef_quat / np.clip(np.linalg.norm(eef_quat, axis=-1, keepdims=True), 1e-8, None)
    else:
        eef_quat = _axis_angle_to_quat_xyzw(ee[:, 3:6])
    if joints.ndim == 1:
        joints = joints[:, None]
    if joints.ndim != 2 or joints.shape[1] < 7:
        raise ValueError(f"Expected joint_states shape (T,7+) with arm joints first, got {joints.shape}")
    return np.concatenate([ee[:, :3], eef_quat[:, :4], joints[:, :7]], axis=1).astype(np.float32)


def _init_state_id(demo_name: str, group: h5py.Group) -> int:
    for key in ("init_state_id", "init_state_index"):
        if key in group.attrs:
            return int(group.attrs[key])
    match = re.search(r"\d+", demo_name)
    return int(match.group(0)) if match else 0


def _init_state_value(group: h5py.Group) -> list[float] | None:
    if "init_state" not in group.attrs:
        return None
    # MuJoCo states are float64. Preserve them exactly for contact-rich replay.
    return np.asarray(group.attrs["init_state"], dtype=np.float64).reshape(-1).tolist()


def _patch_episode_metadata(root: Path, rows: dict[int, dict[str, Any]]) -> None:
    for parquet_path in sorted((root / "meta" / "episodes").glob("**/*.parquet")):
        df = pd.read_parquet(parquet_path)
        for key in (
            "libero/suite",
            "libero/task_id",
            "libero/task_name",
            "libero/init_state_id",
            "libero/init_state",
            "libero/source_file",
            "libero/source_demo",
        ):
            df[key] = [rows[int(ep)].get(key) for ep in df["episode_index"].astype(int).tolist()]
        df.to_parquet(parquet_path)


def convert(args: argparse.Namespace) -> None:
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} exists. Use --overwrite to replace it.")
        shutil.rmtree(args.output_root)

    files = _hdf5_files(args.input_dir)
    file_tasks = _resolve_file_tasks(files, args.suite, _load_task_id_map(args.task_id_map_json))
    demo_index_map = _load_demo_index_map(args.demo_index_map_json)
    demo_indices_by_file = _assign_demo_index_map(
        files,
        file_tasks,
        demo_index_map,
        args.max_episodes_per_task,
    )
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
        use_videos=args.use_videos,
    )

    episode_rows: dict[int, dict[str, Any]] = {}
    local_episode_index = 0
    for hdf5_path in tqdm(files, desc="LIBERO-10 task files"):
        task = file_tasks[hdf5_path]
        with h5py.File(hdf5_path, "r") as h5:
            demos = _demo_groups(h5)
            explicit_indices = demo_indices_by_file.get(hdf5_path)
            if explicit_indices is not None:
                if max(explicit_indices) >= len(demos):
                    raise ValueError(
                        f"Demo index map for {hdf5_path.name} exceeds available range 0..{len(demos) - 1}."
                    )
                demos = [demos[index] for index in explicit_indices]
            elif args.max_episodes_per_task is not None:
                demos = demos[: args.max_episodes_per_task]
            for demo_name, group in tqdm(demos, desc=hdf5_path.stem, leave=False):
                actions = np.asarray(
                    _find_dataset(group, args.action_key, DEFAULT_ACTION_KEYS),
                    dtype=np.float32,
                )
                state = _state_from_demo(group, args)
                agentview = np.asarray(_find_dataset(group, args.agentview_key, DEFAULT_AGENTVIEW_KEYS))
                wrist = np.asarray(_find_dataset(group, args.wrist_key, DEFAULT_WRIST_KEYS))
                length = min(len(actions), len(state), len(agentview), len(wrist))
                if length <= 0:
                    raise ValueError(f"Empty demo {hdf5_path}:{demo_name}")
                if actions.shape[1] != 7:
                    raise ValueError(f"Expected actions shape (T,7), got {actions.shape}")
                if len({len(actions), len(state), len(agentview), len(wrist)}) != 1:
                    logging.warning(
                        "Truncating %s:%s to %d frames due to length mismatch.",
                        hdf5_path,
                        demo_name,
                        length,
                    )

                for frame_idx in range(length):
                    dataset.add_frame(
                        {
                            ACTION: actions[frame_idx].astype(np.float32),
                            OBS_STATE: state[frame_idx].astype(np.float32),
                            f"{OBS_IMAGES}.image": _as_hwc_uint8(
                                agentview[frame_idx],
                                args.height,
                                args.width,
                            ),
                            f"{OBS_IMAGES}.image2": _as_hwc_uint8(wrist[frame_idx], args.height, args.width),
                            "task": str(task["language"]),
                        }
                    )
                dataset.save_episode()
                episode_rows[local_episode_index] = {
                    "libero/suite": args.suite,
                    "libero/task_id": int(task["task_id"]),
                    "libero/task_name": str(task["task_name"]),
                    "libero/init_state_id": _init_state_id(demo_name, group),
                    "libero/init_state": _init_state_value(group),
                    "libero/source_file": hdf5_path.name,
                    "libero/source_demo": demo_name,
                }
                local_episode_index += 1

    dataset.finalize()
    _patch_episode_metadata(args.output_root, episode_rows)
    write_libero_pipeline_manifest(
        args.output_root,
        {
            "pipeline_version": LIBERO_PIPELINE_VERSION,
            "stage": "hdf5_to_lerobot_delta",
            "conversion_complete": True,
            "suite": args.suite,
            "action_representation": LIBERO_DELTA_ACTION,
            "state_representation": LIBERO_STATE_14D,
            "image_size": [args.height, args.width],
            "source_hdf5_dir": str(args.input_dir.resolve()),
            "source_file_count": len(files),
            "episode_count": local_episode_index,
            "max_episodes_per_task": args.max_episodes_per_task,
            "demo_index_map": demo_index_map,
        },
    )
    logging.info("Wrote %d episodes to %s", local_episode_index, args.output_root)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    convert(parse_args())


if __name__ == "__main__":
    main()
