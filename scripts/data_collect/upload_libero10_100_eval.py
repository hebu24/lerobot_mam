#!/usr/bin/env python
"""Upload the generated eval folder into hebu2024/libero10_mam."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/datasets/libero10_100_eval"))
    parser.add_argument("--repo-id", default="hebu2024/libero10_mam")
    parser.add_argument("--path-in-repo", default="libero10_100_eval")
    parser.add_argument(
        "--commit-message",
        default="Add 100 seeded successful DP rollouts for fixed MAM evaluation",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="LFS upload concurrency. Keep at 1 for reliable large uploads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Invalid LeRobot dataset folder: {args.root}")
    api = HfApi()
    user = api.whoami()
    print(f"Authenticated as {user['name']}; uploading {args.root} ...")
    operations = [
        CommitOperationAdd(
            path_in_repo=f"{args.path_in_repo}/{path.relative_to(args.root).as_posix()}",
            path_or_fileobj=path,
        )
        for path in sorted(args.root.rglob("*"))
        if path.is_file()
    ]
    result = api.create_commit(
        repo_id=args.repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=args.commit_message,
        num_threads=args.num_threads,
    )
    print(result)


if __name__ == "__main__":
    main()
