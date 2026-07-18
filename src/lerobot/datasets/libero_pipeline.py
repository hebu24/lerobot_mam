"""Manifest helpers for the LIBERO-10 v3 action pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LIBERO_PIPELINE_VERSION = "v3"
LIBERO_PIPELINE_MANIFEST = Path("meta/libero_pipeline.json")
LIBERO_DELTA_ACTION = "osc_pose_delta"
LIBERO_ABSOLUTE_ACTION = "osc_pose_absolute_goal"
LIBERO_CHUNK_RELATIVE_ACTION = "chunk_relative_se3"
LIBERO_STATE_14D = "eef_pos_quat_xyzw_joint_pos_14d"
LIBERO_CLOSED_LOOP_ABSOLUTE_MATERIALIZATION = "closed_loop_absolute_controller"


def read_libero_pipeline_manifest(root: str | Path) -> dict[str, Any]:
    path = Path(root) / LIBERO_PIPELINE_MANIFEST
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. This dataset is not certified for the LIBERO-10 v3 pipeline."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid LIBERO pipeline manifest at {path}: expected a JSON object.")
    return payload


def write_libero_pipeline_manifest(root: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(root) / LIBERO_PIPELINE_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)
    return path


def require_libero_v3_action_dataset(
    root: str | Path,
    *,
    action_representation: str,
) -> dict[str, Any]:
    manifest = read_libero_pipeline_manifest(root)
    errors: list[str] = []
    if manifest.get("pipeline_version") != LIBERO_PIPELINE_VERSION:
        errors.append(f"pipeline_version={manifest.get('pipeline_version')!r}")
    if manifest.get("action_representation") != action_representation:
        errors.append(f"action_representation={manifest.get('action_representation')!r}")
    if manifest.get("state_representation") != LIBERO_STATE_14D:
        errors.append(f"state_representation={manifest.get('state_representation')!r}")
    if manifest.get("conversion_complete") is not True:
        errors.append(f"conversion_complete={manifest.get('conversion_complete')!r}")
    if errors:
        raise ValueError(
            f"{Path(root)} is not a complete LIBERO-10 {LIBERO_PIPELINE_VERSION} "
            f"{action_representation} dataset: {', '.join(errors)}"
        )
    return manifest


def require_libero_v3_relative_ready_dataset(root: str | Path) -> dict[str, Any]:
    """Require observations and absolute actions from the same closed-loop rollout.

    A legacy v3 ``delta_to_absolute`` artifact only replaced the action column while
    retaining observations recorded under the delta controller.  Such a dataset is
    valid for action replay diagnostics, but its state/action pairs are not valid
    anchors for chunk-relative policy training.  The two manifest capabilities
    checked here are written only after closed-loop absolute-controller
    rematerialization has completed.
    """
    manifest = require_libero_v3_action_dataset(
        root,
        action_representation=LIBERO_ABSOLUTE_ACTION,
    )
    errors: list[str] = []
    if (
        manifest.get("observation_materialization")
        != LIBERO_CLOSED_LOOP_ABSOLUTE_MATERIALIZATION
    ):
        errors.append(
            "observation_materialization="
            f"{manifest.get('observation_materialization')!r}"
        )
    if manifest.get("relative_action_ready") is not True:
        errors.append(f"relative_action_ready={manifest.get('relative_action_ready')!r}")
    if errors:
        raise ValueError(
            f"{Path(root)} is not certified for LIBERO v3 chunk-relative training: "
            f"{', '.join(errors)}. Rebuild it by closed-loop absolute-controller "
            "rematerialization; action-only v3 conversion is insufficient."
        )
    return manifest
