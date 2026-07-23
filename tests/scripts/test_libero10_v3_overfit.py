import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.datasets.libero_pipeline import (
    LIBERO_ABSOLUTE_ACTION,
    LIBERO_CHUNK_RELATIVE_ACTION,
    LIBERO_CLOSED_LOOP_ABSOLUTE_MATERIALIZATION,
    LIBERO_PIPELINE_VERSION,
    LIBERO_STATE_14D,
    require_libero_v3_relative_ready_dataset,
    write_libero_pipeline_manifest,
)
from lerobot.datasets.libero_training import validate_libero_v3_training_dataset
from lerobot.envs.libero_eval import validate_libero_action_semantics
from scripts import convert_libero_absolute_to_mam
from scripts.convert_libero10_hdf5_to_lerobot import _assign_demo_index_map
from scripts.convert_libero_absolute_to_mam import _as_float_list
from scripts.prepare_libero10_v3_overfit import select_overfit_episodes


def _row(task: int, episode: int, source: int, slot: int = 0, mask_type: str = "random_mask"):
    return {
        "episode_index": episode,
        "libero/task_id": task,
        "libero/source_episode_id": source,
        "libero/suite": "libero_10",
        "libero/init_state": [0.1, float(source)],
        "libero/source_file": f"task{task}.hdf5",
        "libero/source_demo": f"demo_{source}",
        "mask_type": mask_type,
        "mask_type_slot": slot,
    }


def test_v3_selection_is_unique_and_stable_as_k_grows():
    rows = [
        _row(0, 4, 11, 1),
        _row(0, 3, 11, 0),
        _row(0, 7, 20, 0),
        _row(1, 8, 5, 0),
        _row(2, 9, 2, 0),
    ]

    one = select_overfit_episodes(rows, task_ids=[0], demo_rank=0)
    three = select_overfit_episodes(rows, task_ids=[0, 1, 2], demo_rank=0)

    assert one == three[:1]
    assert [item["episode_index"] for item in three] == [3, 8, 9]
    assert [item["source_episode_id"] for item in three] == [11, 5, 2]


def test_v3_selection_requires_exact_trajectory_metadata():
    row = _row(0, 0, 0)
    row["libero/init_state"] = None

    with pytest.raises(ValueError, match="lacks exact v3 trajectory metadata"):
        select_overfit_episodes([row], task_ids=[0], demo_rank=0)


def test_v3_selection_supports_multiple_demos_per_task():
    rows = [
        _row(0, 4, 11, 1),
        _row(0, 3, 11, 0),
        _row(0, 7, 20, 0),
        _row(1, 8, 5, 0),
        _row(1, 10, 6, 0),
    ]

    selected = select_overfit_episodes(
        rows,
        task_ids=[0, 1],
        demo_rank=0,
        demos_per_task=2,
    )

    assert [item["episode_index"] for item in selected] == [3, 7, 8, 10]
    assert [item["source_episode_id"] for item in selected] == [11, 20, 5, 6]


def test_v3_selection_filters_requested_mask_type():
    rows = [
        _row(0, 3, 11, slot=0, mask_type="random_mask"),
        _row(0, 4, 11, slot=1, mask_type="3D_points"),
    ]

    selected = select_overfit_episodes(
        rows,
        task_ids=[0],
        demo_rank=0,
        mask_type="3D_points",
    )

    assert selected[0]["episode_index"] == 4
    assert selected[0]["mask_type"] == "3D_points"


def test_mam_conversion_preserves_float64_init_state():
    value = np.asarray([1.0000000001, -0.1234567890123], dtype=np.float64)

    converted = np.asarray(_as_float_list(value), dtype=np.float64)

    assert np.array_equal(converted, value)


def test_chunk_relative_libero_requires_absolute_controller():
    cfg = SimpleNamespace(
        policy=SimpleNamespace(type="diffusion", use_relative_actions=True),
        env=SimpleNamespace(type="libero", control_mode="relative"),
    )

    with pytest.raises(ValueError, match="requires env.control_mode='absolute'"):
        validate_libero_action_semantics(cfg.env, cfg.policy)

    cfg.env.control_mode = "absolute"
    validate_libero_action_semantics(cfg.env, cfg.policy)


def _write_mam_manifest(root, split):
    write_libero_pipeline_manifest(
        root,
        {
            "pipeline_version": LIBERO_PIPELINE_VERSION,
            "stage": "absolute_to_mam",
            "conversion_complete": True,
            "dataset_split": split,
            "action_representation": LIBERO_ABSOLUTE_ACTION,
            "policy_action_representation": LIBERO_CHUNK_RELATIVE_ACTION,
            "observation_materialization": LIBERO_CLOSED_LOOP_ABSOLUTE_MATERIALIZATION,
            "relative_action_ready": True,
            "state_representation": LIBERO_STATE_14D,
            "source_root": str(root.parent / "absolute"),
            "source_repo_id": "local/absolute",
            "source_episode_ids": [1, 2] if split == "train" else [3, 4],
        },
    )


def _write_action_only_manifest(root):
    write_libero_pipeline_manifest(
        root,
        {
            "pipeline_version": LIBERO_PIPELINE_VERSION,
            "stage": "delta_to_absolute",
            "conversion_complete": True,
            "action_representation": LIBERO_ABSOLUTE_ACTION,
            "state_representation": LIBERO_STATE_14D,
        },
    )


def test_relative_ready_certificate_rejects_legacy_action_only_v3(tmp_path):
    root = tmp_path / "absolute_action_only"
    _write_action_only_manifest(root)

    with pytest.raises(ValueError, match="action-only v3 conversion is insufficient"):
        require_libero_v3_relative_ready_dataset(root)


def test_relative_ready_certificate_rejects_incomplete_rematerialization(tmp_path):
    root = tmp_path / "absolute_partial"
    _write_action_only_manifest(root)
    manifest_path = root / "meta" / "libero_pipeline.json"
    payload = json.loads(manifest_path.read_text())
    payload["observation_materialization"] = LIBERO_CLOSED_LOOP_ABSOLUTE_MATERIALIZATION
    payload["relative_action_ready"] = False
    write_libero_pipeline_manifest(root, payload)

    with pytest.raises(ValueError, match="relative_action_ready=False"):
        require_libero_v3_relative_ready_dataset(root)


def _training_cfg(train_root, eval_root, *, overfit):
    episodes = [3, 8]
    return SimpleNamespace(
        trainable_config=SimpleNamespace(type="diffusion", use_relative_actions=True),
        dataset=SimpleNamespace(repo_id="local/train", root=str(train_root), episodes=episodes),
        eval=SimpleNamespace(
            dataset_repo_id="local/train" if overfit else "local/eval",
            dataset_root=str(train_root if overfit else eval_root),
            dataset_episodes=episodes if overfit else None,
        ),
        env=SimpleNamespace(type="libero"),
        eval_freq=100,
        overfit_test=overfit,
    )


def test_training_preflight_certifies_normal_and_exact_overfit_splits(tmp_path):
    train_root = tmp_path / "train"
    eval_root = tmp_path / "eval"
    _write_mam_manifest(train_root, "train")
    _write_mam_manifest(eval_root, "eval")
    dataset = SimpleNamespace(root=train_root, meta=SimpleNamespace(robot_type="libero"))

    validate_libero_v3_training_dataset(_training_cfg(train_root, eval_root, overfit=False), dataset)
    validate_libero_v3_training_dataset(_training_cfg(train_root, eval_root, overfit=True), dataset)

    mam_cfg = _training_cfg(train_root, eval_root, overfit=False)
    mam_cfg.trainable_config.type = "mam"
    mam_cfg.trainable_config.mam_eval_dataset_repo_id = "local/eval"
    mam_cfg.trainable_config.mam_eval_dataset_root = str(eval_root)
    mam_cfg.trainable_config.mam_eval_episodes = None
    validate_libero_v3_training_dataset(mam_cfg, dataset)


def test_training_preflight_rejects_overfit_eval_trajectory_mismatch(tmp_path):
    train_root = tmp_path / "train"
    eval_root = tmp_path / "eval"
    _write_mam_manifest(train_root, "train")
    cfg = _training_cfg(train_root, eval_root, overfit=True)
    cfg.eval.dataset_episodes = [3]
    dataset = SimpleNamespace(root=train_root, meta=SimpleNamespace(robot_type="libero"))

    with pytest.raises(ValueError, match="exactly identical"):
        validate_libero_v3_training_dataset(cfg, dataset)


def test_training_preflight_rejects_normal_source_trajectory_leakage(tmp_path):
    train_root = tmp_path / "train"
    eval_root = tmp_path / "eval"
    _write_mam_manifest(train_root, "train")
    _write_mam_manifest(eval_root, "eval")
    eval_manifest_path = eval_root / "meta" / "libero_pipeline.json"
    payload = json.loads(eval_manifest_path.read_text())
    payload["source_episode_ids"] = [2, 3]
    write_libero_pipeline_manifest(eval_root, payload)
    dataset = SimpleNamespace(root=train_root, meta=SimpleNamespace(robot_type="libero"))

    with pytest.raises(ValueError, match="source trajectory leakage"):
        validate_libero_v3_training_dataset(_training_cfg(train_root, eval_root, overfit=False), dataset)


def test_training_preflight_fails_closed_for_libero_env_with_wrong_robot_type(tmp_path):
    cfg = _training_cfg(tmp_path / "train", tmp_path / "eval", overfit=False)
    dataset = SimpleNamespace(root=tmp_path / "train", meta=SimpleNamespace(robot_type="unknown"))

    with pytest.raises(ValueError, match="robot_type='libero'"):
        validate_libero_v3_training_dataset(cfg, dataset)


def test_training_preflight_rejects_legacy_action_only_v3(tmp_path):
    train_root = tmp_path / "train"
    eval_root = tmp_path / "eval"
    _write_action_only_manifest(train_root)
    cfg = _training_cfg(train_root, eval_root, overfit=True)
    dataset = SimpleNamespace(root=train_root, meta=SimpleNamespace(robot_type="libero"))

    with pytest.raises(ValueError, match="action-only v3 conversion is insufficient"):
        validate_libero_v3_training_dataset(cfg, dataset)

    cfg.trainable_config.type = "mam"
    with pytest.raises(ValueError, match="action-only v3 conversion is insufficient"):
        validate_libero_v3_training_dataset(cfg, dataset)


def test_mam_training_preflight_rejects_legacy_eval_dataset(tmp_path):
    train_root = tmp_path / "train"
    eval_root = tmp_path / "eval"
    _write_mam_manifest(train_root, "train")
    _write_action_only_manifest(eval_root)
    cfg = _training_cfg(train_root, eval_root, overfit=False)
    cfg.trainable_config.type = "mam"
    cfg.trainable_config.mam_eval_dataset_repo_id = "local/eval"
    cfg.trainable_config.mam_eval_dataset_root = str(eval_root)
    cfg.trainable_config.mam_eval_episodes = None
    dataset = SimpleNamespace(root=train_root, meta=SimpleNamespace(robot_type="libero"))

    with pytest.raises(ValueError, match="action-only v3 conversion is insufficient"):
        validate_libero_v3_training_dataset(cfg, dataset)


def test_mam_conversion_rejects_an_already_materialized_split(tmp_path, monkeypatch):
    source_root = tmp_path / "mam_train"
    _write_mam_manifest(source_root, "train")
    monkeypatch.setattr(
        convert_libero_absolute_to_mam,
        "parse_args",
        lambda: SimpleNamespace(input_root=source_root),
    )

    with pytest.raises(ValueError, match="delta_to_absolute"):
        convert_libero_absolute_to_mam.main()


def test_mam_conversion_rejects_legacy_action_only_absolute(tmp_path, monkeypatch):
    source_root = tmp_path / "absolute_action_only"
    _write_action_only_manifest(source_root)
    monkeypatch.setattr(
        convert_libero_absolute_to_mam,
        "parse_args",
        lambda: SimpleNamespace(input_root=source_root),
    )

    with pytest.raises(ValueError, match="action-only v3 conversion is insufficient"):
        convert_libero_absolute_to_mam.main()


def test_launcher_rejects_uncontrolled_extra_cli_arguments():
    result = subprocess.run(
        ["bash", "scripts/run_diffusion_libero10_v3_overfit.sh", "3", "--resume=true"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Only one optional positional argument" in result.stderr


def test_demo_index_map_rejects_unused_ambiguous_and_oversized_keys(tmp_path):
    source = tmp_path / "task.hdf5"
    tasks = {source: {"task_id": 8}}

    assert _assign_demo_index_map([source], tasks, {"8": [0, 2]}, 2) == {source: [0, 2]}
    with pytest.raises(ValueError, match="did not match"):
        _assign_demo_index_map([source], tasks, {"typo": [0]}, 2)
    with pytest.raises(ValueError, match="ambiguous aliases"):
        _assign_demo_index_map([source], tasks, {"8": [0], "task.hdf5": [0]}, 2)
    with pytest.raises(ValueError, match="exceeding"):
        _assign_demo_index_map([source], tasks, {"8": [0, 1, 2]}, 2)
