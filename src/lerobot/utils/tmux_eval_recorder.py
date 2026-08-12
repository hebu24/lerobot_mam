#!/usr/bin/env python

"""Persist compact evaluation results from tmux training runs on shared storage."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


_FIELD_NAMES = [
    "recorded_at",
    "eval_time",
    "record_origin",
    "host",
    "tmux_session",
    "job_name",
    "output_dir",
    "source_eval_metrics",
    "step",
    "n_episodes",
    "pc_success",
    "avg_sum_reward",
    "avg_max_reward",
    *[f"task_{task_id}_pc_success" for task_id in range(10)],
    "per_mask_type_success",
    "per_mask_slot_success",
]


def default_record_path() -> Path:
    """Return the shared ledger path, with an environment override for other deployments."""
    configured = os.environ.get("LEROBOT_TMUX_EVAL_RECORD_PATH")
    if configured:
        return Path(configured).expanduser().resolve()

    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "outputs" / "logs" / "tmux_eval_results.csv"


def current_tmux_session() -> str:
    """Resolve the current tmux session without making recording depend on tmux CLI success."""
    configured = os.environ.get("LEROBOT_TMUX_SESSION_NAME")
    if configured:
        return configured

    pane = os.environ.get("TMUX_PANE")
    if pane:
        try:
            result = subprocess.run(
                ["tmux", "display-message", "-p", "-t", pane, "#S"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            session = result.stdout.strip()
            if session:
                return session
        except (OSError, subprocess.SubprocessError):
            pass
    return pane or "unknown"


def _compact_record(
    *,
    eval_info: dict[str, Any],
    step: int,
    eval_time: float,
    job_name: str,
    output_dir: str,
    tmux_session: str,
    source_eval_metrics: str,
    record_origin: str,
    host: str | None = None,
) -> dict[str, Any]:
    overall = eval_info.get("overall", {})
    task_success = {
        int(item["task_id"]): item.get("metrics", {}).get("pc_success", "")
        for item in eval_info.get("per_task", [])
        if "task_id" in item
    }
    recorded_at = datetime.now().astimezone().isoformat(timespec="seconds")
    evaluated_at = datetime.fromtimestamp(eval_time).astimezone().isoformat(timespec="seconds")
    record: dict[str, Any] = {
        "recorded_at": recorded_at,
        "eval_time": evaluated_at,
        "record_origin": record_origin,
        "host": host or socket.gethostname(),
        "tmux_session": tmux_session,
        "job_name": job_name,
        "output_dir": output_dir,
        "source_eval_metrics": source_eval_metrics,
        "step": int(step),
        "n_episodes": overall.get("n_episodes", ""),
        "pc_success": overall.get("pc_success", ""),
        "avg_sum_reward": overall.get("avg_sum_reward", ""),
        "avg_max_reward": overall.get("avg_max_reward", ""),
        "per_mask_type_success": json.dumps(
            eval_info.get("per_mask_type_success", {}), ensure_ascii=False, sort_keys=True
        ),
        "per_mask_slot_success": json.dumps(
            eval_info.get("per_mask_slot_success", {}), ensure_ascii=False, sort_keys=True
        ),
    }
    record.update(
        {f"task_{task_id}_pc_success": task_success.get(task_id, "") for task_id in range(10)}
    )
    return record


def append_eval_result(
    *,
    eval_info: dict[str, Any],
    step: int,
    eval_time: float,
    job_name: str,
    output_dir: str,
    tmux_session: str,
    source_eval_metrics: str,
    record_origin: str,
    record_path: Path | None = None,
    host: str | None = None,
    deduplicate: bool = False,
) -> Path:
    """Append one result under a file lock so concurrent tmux runs cannot corrupt the ledger."""
    path = record_path or default_record_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = _compact_record(
        eval_info=eval_info,
        step=step,
        eval_time=eval_time,
        job_name=job_name,
        output_dir=output_dir,
        tmux_session=tmux_session,
        source_eval_metrics=source_eval_metrics,
        record_origin=record_origin,
        host=host,
    )

    with path.open("a+", encoding="utf-8", newline="") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        existing_keys: set[tuple[str, str, str]] = set()
        if deduplicate:
            existing_keys = {
                (row.get("job_name", ""), row.get("step", ""), row.get("eval_time", ""))
                for row in csv.DictReader(stream)
            }
        key = (record["job_name"], str(record["step"]), record["eval_time"])
        stream.seek(0, os.SEEK_END)
        writer = csv.DictWriter(stream, fieldnames=_FIELD_NAMES)
        if stream.tell() == 0:
            writer.writeheader()
        if key not in existing_keys:
            writer.writerow(record)
            stream.flush()
            os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return path


def append_tmux_eval_result(
    *, cfg: Any, step: int, eval_time: float, eval_info: dict[str, Any]
) -> Path | None:
    """Record an evaluation only when the training process is running inside tmux."""
    if not os.environ.get("TMUX"):
        return None
    output_dir = str(cfg.output_dir)
    return append_eval_result(
        eval_info=eval_info,
        step=step,
        eval_time=eval_time,
        job_name=str(cfg.job_name),
        output_dir=output_dir,
        tmux_session=current_tmux_session(),
        source_eval_metrics=str(Path(output_dir) / "logs" / "eval_metrics.jsonl"),
        record_origin="live",
    )


def _backfill(args: argparse.Namespace) -> tuple[int, int | None]:
    metrics_path = args.eval_metrics.expanduser().resolve()
    added = 0
    max_step: int | None = None
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("mode") != "eval" or "metrics" not in record:
            continue
        step = int(record["step"])
        max_step = step if max_step is None else max(max_step, step)
        before = args.record_path.stat().st_size if args.record_path.exists() else 0
        append_eval_result(
            eval_info=record["metrics"],
            step=step,
            eval_time=float(record["time"]),
            job_name=args.job_name,
            output_dir=args.output_dir,
            tmux_session=args.tmux_session,
            source_eval_metrics=str(metrics_path),
            record_origin="backfill",
            record_path=args.record_path,
            host=args.host,
            deduplicate=True,
        )
        after = args.record_path.stat().st_size
        added += int(after > before)
    return added, max_step


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-metrics", type=Path, required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tmux-session", required=True)
    parser.add_argument("--host")
    parser.add_argument("--record-path", type=Path, default=default_record_path())
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--until-step", type=int)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")

    while True:
        try:
            added, max_step = _backfill(args)
        except FileNotFoundError:
            added, max_step = 0, None
        if added or not args.watch:
            print(f"record={args.record_path} added={added} max_step={max_step}", flush=True)
        if not args.watch or (args.until_step is not None and max_step == args.until_step):
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
