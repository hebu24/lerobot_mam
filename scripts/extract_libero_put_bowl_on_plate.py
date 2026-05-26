#!/usr/bin/env python

"""Extract the LIBERO single-task dataset: "put the bowl on the plate".

The script reads only metadata first, finds episodes whose task text matches
the target task, then downloads/copies only those episodes into a new local
LeRobot dataset.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
from pathlib import Path


TARGET_TASK = "put the bowl on the plate"
SOURCE_REPO_ID = "HuggingFaceVLA/libero"
OUTPUT_REPO_ID = "local/libero_put_bowl_on_plate"
OUTPUT_ROOT = Path("outputs/datasets/libero_put_bowl_on_plate")
SOURCE_ROOT = Path("outputs/cache/libero_source")
LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_90", "libero_10")
DATA_FILE_RE = re.compile(r"^data/chunk-(\d{3})/file-(\d{3})\.parquet$")


def normalize_task(text: str) -> str:
    """Normalize task text and ignore a leading numeric task id."""
    text = re.sub(r"^\s*\d+\s+", "", text)
    text = text.lower().strip()
    text = re.sub(r"[.。]+$", "", text)
    return re.sub(r"\s+", " ", text)


def task_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def find_matching_task_names(tasks_index, target: str, allow_substring: bool) -> list[str]:
    target_norm = normalize_task(target)
    exact_matches = [str(task) for task in tasks_index if normalize_task(str(task)) == target_norm]
    if exact_matches or not allow_substring:
        return exact_matches
    return [str(task) for task in tasks_index if target_norm in normalize_task(str(task))]


def find_episode_indices(meta, matched_task_names: set[str]) -> list[int]:
    matched_norms = {normalize_task(task) for task in matched_task_names}
    episode_indices: list[int] = []
    for episode in meta.episodes:
        episode_tasks = task_list(episode.get("tasks"))
        if any(normalize_task(task) in matched_norms for task in episode_tasks):
            episode_indices.append(int(episode["episode_index"]))
    return episode_indices


def log_required_source_files(meta, episode_indices: list[int]) -> None:
    data_files = {str(meta.get_data_file_path(ep_idx)) for ep_idx in episode_indices}
    video_files = {
        str(meta.get_video_file_path(ep_idx, video_key))
        for video_key in meta.video_keys
        for ep_idx in episode_indices
    }
    logging.info("Required source data shards: %d", len(data_files))
    logging.info("Required source video shards: %d", len(video_files))


def get_task_index(meta, task_name: str) -> int:
    row = meta.tasks.loc[task_name]
    return int(row["task_index"])


def list_remote_data_files(repo_id: str, revision: str | None) -> list[tuple[int, str]]:
    from huggingface_hub import HfApi

    paths = []
    for path in HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision):
        match = DATA_FILE_RE.match(path)
        if match:
            paths.append((int(match.group(2)), path))
    if not paths:
        raise FileNotFoundError(f"No data parquet files found in {repo_id}@{revision}")
    return sorted(paths)


def download_data_file(
    repo_id: str,
    revision: str | None,
    filename: str,
    source_root: Path | None,
    force_download: bool,
) -> Path:
    from huggingface_hub import hf_hub_download

    if source_root is not None:
        source_root.mkdir(parents=True, exist_ok=True)
        return Path(
            hf_hub_download(
                repo_id,
                filename=filename,
                repo_type="dataset",
                revision=revision,
                local_dir=source_root,
                force_download=force_download,
            )
        )
    return Path(
        hf_hub_download(
            repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            force_download=force_download,
        )
    )


def locate_actual_data_files(
    repo_id: str,
    revision: str | None,
    source_root: Path | None,
    episode_indices: list[int],
    force_download: bool,
) -> dict[int, Path]:
    import pyarrow.parquet as pq

    remote_files = list_remote_data_files(repo_id, revision)
    ranges: dict[int, tuple[int, int, Path]] = {}

    def episode_range(pos: int) -> tuple[int, int, Path]:
        file_idx, filename = remote_files[pos]
        if file_idx not in ranges:
            path = download_data_file(repo_id, revision, filename, source_root, force_download)
            table = pq.read_table(path, columns=["episode_index"])
            values = table.column("episode_index").to_pylist()
            ranges[file_idx] = (int(min(values)), int(max(values)), path)
        return ranges[file_idx]

    found: dict[int, Path] = {}
    for episode_index in episode_indices:
        lo, hi = 0, len(remote_files) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            min_ep, max_ep, path = episode_range(mid)
            if episode_index < min_ep:
                hi = mid - 1
            elif episode_index > max_ep:
                lo = mid + 1
            else:
                found[remote_files[mid][0]] = path
                break
        else:
            raise FileNotFoundError(f"Could not locate data file containing episode {episode_index}")
    return dict(sorted(found.items()))


def episode_stats_from_row(row: dict, features: dict) -> dict:
    import numpy as np

    episode_stats = {}
    for key, value in row.items():
        if not key.startswith("stats/"):
            continue
        stat_key = key.replace("stats/", "")
        parts = stat_key.split("/")
        if len(parts) != 2:
            continue
        feature_name, stat_name = parts
        episode_stats.setdefault(feature_name, {})

        if feature_name in features:
            feature_dtype = features[feature_name]["dtype"]
            if feature_dtype in ["image", "video"] and stat_name != "count":
                if isinstance(value, np.ndarray) and value.dtype == object:
                    flat_values = []
                    for item in value:
                        while isinstance(item, np.ndarray):
                            item = item.flatten()[0]
                        flat_values.append(item)
                    value = np.array(flat_values, dtype=np.float64).reshape(3, 1, 1)
                elif isinstance(value, np.ndarray) and value.shape == (3,):
                    value = value.reshape(3, 1, 1)
        episode_stats[feature_name][stat_name] = value
    return episode_stats


def write_single_task_dataset(
    meta,
    data_files: dict[int, Path],
    episode_indices: list[int],
    matched_task: str,
    output_root: Path,
    output_repo_id: str,
) -> None:
    import pandas as pd
    from tqdm import tqdm

    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.dataset_tools import _write_parquet
    from lerobot.datasets.io_utils import write_info, write_stats
    from lerobot.datasets.utils import DEFAULT_DATA_PATH
    from lerobot.utils.constants import IMAGENET_STATS
    from lerobot.utils.utils import flatten_dict

    episode_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(episode_indices))}
    keep = set(episode_mapping)

    new_meta = LeRobotDatasetMetadata.create(
        repo_id=output_repo_id,
        fps=meta.fps,
        features=meta.features,
        robot_type=meta.robot_type,
        root=output_root,
        use_videos=False,
    )
    new_meta.save_episode_tasks([matched_task])

    global_index = 0
    data_metadata: dict[int, dict] = {}

    for out_file_idx, src_path in enumerate(tqdm(data_files.values(), desc="Processing data files")):
        df = pd.read_parquet(src_path)
        df = df[df["episode_index"].isin(keep)].copy().reset_index(drop=True)
        if df.empty:
            continue

        df["episode_index"] = df["episode_index"].replace(episode_mapping)
        df["task_index"] = 0
        df["index"] = range(global_index, global_index + len(df))

        dst_path = output_root / DEFAULT_DATA_PATH.format(chunk_index=0, file_index=out_file_idx)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        _write_parquet(df, dst_path, new_meta)

        for new_idx in sorted(df["episode_index"].unique()):
            ep_df = df[df["episode_index"] == new_idx]
            data_metadata[int(new_idx)] = {
                "data/chunk_index": 0,
                "data/file_index": out_file_idx,
                "dataset_from_index": int(ep_df["index"].min()),
                "dataset_to_index": int(ep_df["index"].max() + 1),
            }
        global_index += len(df)

    episodes_df = meta.episodes.to_pandas()
    episodes_df = episodes_df.set_index("episode_index", drop=False)
    all_stats = []

    for old_idx, new_idx in tqdm(sorted(episode_mapping.items(), key=lambda item: item[1]), desc="Writing metadata"):
        if new_idx not in data_metadata:
            raise ValueError(f"No frames were written for source episode {old_idx}")
        source_row = episodes_df.loc[old_idx].to_dict()
        episode_stats = episode_stats_from_row(source_row, meta.features)
        all_stats.append(episode_stats)

        episode_dict = {
            "episode_index": new_idx,
            "tasks": [matched_task],
            "length": data_metadata[new_idx]["dataset_to_index"] - data_metadata[new_idx]["dataset_from_index"],
            "libero/init_state_id": new_idx,
            "libero/source_episode_index": old_idx,
        }
        episode_dict.update(data_metadata[new_idx])
        episode_dict.update(flatten_dict({"stats": episode_stats}))
        new_meta._save_episode_metadata(episode_dict)

    new_meta.finalize()
    new_meta.info.total_episodes = len(episode_mapping)
    new_meta.info.total_frames = global_index
    new_meta.info.total_tasks = 1
    new_meta.info.splits = {"train": f"0:{len(episode_mapping)}"}
    write_info(new_meta.info, new_meta.root)

    aggregated_stats = aggregate_stats(all_stats)
    filtered_stats = {key: value for key, value in aggregated_stats.items() if key in new_meta.features}
    for key, feature in new_meta.features.items():
        if feature["dtype"] in ["image", "video"]:
            filtered_stats.setdefault(key, IMAGENET_STATS)
    write_stats(filtered_stats, new_meta.root)

    # Validate that the result can be opened and contains the expected episodes.
    dataset = LeRobotDataset(output_repo_id, root=output_root)
    if dataset.num_episodes != len(episode_mapping):
        raise RuntimeError(f"Expected {len(episode_mapping)} episodes, got {dataset.num_episodes}")


def find_libero_task_locations(target: str, allow_substring: bool) -> list[tuple[str, int, str]]:
    """Find matching task ids inside installed LIBERO benchmark suites."""
    try:
        from libero.libero import benchmark
    except ImportError:
        logging.warning("LIBERO is not installed; skip suite/task_id lookup.")
        return []

    target_norm = normalize_task(target)
    locations: list[tuple[str, int, str]] = []
    for suite_name in LIBERO_SUITES:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        for task_id in range(len(suite.tasks)):
            language = str(suite.get_task(task_id).language)
            language_norm = normalize_task(language)
            if language_norm == target_norm or (allow_substring and target_norm in language_norm):
                locations.append((suite_name, task_id, language))
    return locations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract LIBERO episodes for the task 'put the bowl on the plate'."
    )
    parser.add_argument("--source-repo-id", default=SOURCE_REPO_ID)
    parser.add_argument("--output-repo-id", default=OUTPUT_REPO_ID)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=SOURCE_ROOT,
        help="Optional local cache/materialization root for the source dataset.",
    )
    parser.add_argument(
        "--target-task",
        default=TARGET_TASK,
        help="Target task text. A leading number like '10 ' is ignored.",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--force-cache-sync", action="store_true")
    parser.add_argument(
        "--allow-substring",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fall back to substring matching if exact task text is not found.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print matched task names and episodes; do not create the dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output_root first if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.output_root.exists():
        if not args.overwrite and not args.dry_run:
            raise FileExistsError(
                f"{args.output_root} already exists. Pass --overwrite to replace it."
            )
        if args.overwrite and not args.dry_run:
            shutil.rmtree(args.output_root)

    from lerobot.datasets import LeRobotDatasetMetadata

    logging.info("Loading metadata from %s", args.source_repo_id)
    meta = LeRobotDatasetMetadata(
        args.source_repo_id,
        root=args.source_root,
        revision=args.revision,
        force_cache_sync=args.force_cache_sync,
    )

    matched_task_names = find_matching_task_names(
        meta.tasks.index,
        target=args.target_task,
        allow_substring=args.allow_substring,
    )
    if not matched_task_names:
        candidates = [str(task) for task in meta.tasks.index if "bowl" in normalize_task(str(task))]
        candidate_text = "\n".join(f"  - {task}" for task in candidates[:30])
        raise ValueError(
            f"No task matched {args.target_task!r}. Bowl-related candidates:\n{candidate_text}"
        )
    if len(matched_task_names) > 1:
        match_text = "\n".join(f"  - {task}" for task in matched_task_names)
        raise ValueError(
            "More than one task matched. Re-run with a more specific --target-task:\n"
            f"{match_text}"
        )

    matched_task = matched_task_names[0]
    episode_indices = find_episode_indices(meta, {matched_task})
    if not episode_indices:
        raise ValueError(f"Task matched but no episodes were found: {matched_task!r}")

    logging.info("Matched task: %s", matched_task)
    logging.info("Matched task_index: %d", get_task_index(meta, matched_task))
    logging.info("Matched episodes: %d", len(episode_indices))
    logging.info("First episodes: %s", episode_indices[:20])
    log_required_source_files(meta, episode_indices)
    locations = find_libero_task_locations(matched_task, args.allow_substring)
    for suite_name, task_id, language in locations:
        logging.info(
            "LIBERO eval target: --env.task=%s --env.task_ids='[%d]' (%s)",
            suite_name,
            task_id,
            language,
        )

    if args.dry_run:
        return

    logging.info("Locating actual source data shards by parquet episode ranges")
    data_files = locate_actual_data_files(
        args.source_repo_id,
        meta.revision,
        args.source_root,
        episode_indices,
        args.force_cache_sync,
    )
    logging.info("Actual source data shards: %d", len(data_files))
    logging.info("First actual shards: %s", list(data_files)[:20])

    logging.info("Writing filtered dataset to %s", args.output_root)
    write_single_task_dataset(
        meta,
        data_files,
        episode_indices,
        matched_task,
        args.output_root,
        args.output_repo_id,
    )
    logging.info("Done.")
    logging.info("New dataset root: %s", args.output_root)
    logging.info("New dataset repo_id: %s", args.output_repo_id)
    logging.info("New dataset episodes: %d", len(episode_indices))


if __name__ == "__main__":
    main()
