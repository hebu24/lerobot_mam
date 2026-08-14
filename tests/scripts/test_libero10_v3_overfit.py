import json
import os
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
from scripts.libero.data import convert_libero_absolute_to_mam
from scripts.libero.data.convert_libero10_hdf5_to_lerobot import _assign_demo_index_map
from scripts.libero.data.convert_libero_absolute_to_mam import _as_float_list
from scripts.libero.data.prepare_libero10_v3_overfit import select_overfit_episodes


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


def test_zero_eval_ratio_keeps_every_episode_in_train():
    train_ids, eval_ids = convert_libero_absolute_to_mam._selected_episode_ids(500, 0, 0)

    assert train_ids == list(range(500))
    assert eval_ids == []


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


def test_mam_remask_accepts_complete_embedded_source_manifests(tmp_path):
    source_root = tmp_path / "merged_mam_train"
    write_libero_pipeline_manifest(
        source_root,
        {
            "stage": "absolute_to_mam",
            "official_source_manifest": {
                "source_root": "/datasets/official",
                "source_repo_id": "local/official",
            },
            "rollout_source_manifest": {
                "source_root": "/datasets/rollout",
                "source_repo_id": "local/rollout",
            },
        },
    )
    args = SimpleNamespace(input_root=source_root, input_repo_id="local/merged")

    source_path, source_repo_id = convert_libero_absolute_to_mam._absolute_source_provenance(args)

    assert source_path == str(source_root.resolve())
    assert source_repo_id == "local/merged"


def test_mam_remask_rejects_incomplete_embedded_source_manifest(tmp_path):
    source_root = tmp_path / "merged_mam_train"
    write_libero_pipeline_manifest(
        source_root,
        {
            "stage": "absolute_to_mam",
            "official_source_manifest": {
                "source_root": "/datasets/official",
                "source_repo_id": "local/official",
            },
            "rollout_source_manifest": {"source_root": "/datasets/rollout"},
        },
    )
    args = SimpleNamespace(input_root=source_root, input_repo_id=None)

    with pytest.raises(ValueError, match=r"complete embedded \*_source_manifest"):
        convert_libero_absolute_to_mam._absolute_source_provenance(args)


def test_mam_pose_mask_retains_complete_actions_at_sampled_timesteps():
    actions = np.arange(70, dtype=np.float32).reshape(10, 7)

    masked, mask = convert_libero_absolute_to_mam._apply_mask(
        actions,
        mask_type="pose",
        retain_ratio=0.3,
        mask_value=-1.0,
        rng=np.random.default_rng(0),
    )

    retained_timesteps = np.flatnonzero(mask[:, 0])
    assert retained_timesteps.size == 3
    assert np.all(mask[retained_timesteps] == 1.0)
    assert np.all(mask[np.setdiff1d(np.arange(10), retained_timesteps)] == 0.0)
    assert np.array_equal(masked[retained_timesteps], actions[retained_timesteps])
    assert np.all(masked[mask == 0.0] == -1.0)


def test_mam_pose_motion_planning_matches_pose_alias():
    actions = np.arange(70, dtype=np.float32).reshape(10, 7)

    _, pose_mask = convert_libero_absolute_to_mam._apply_mask(
        actions, "pose", 0.3, 0.0, np.random.default_rng(7)
    )
    _, planning_mask = convert_libero_absolute_to_mam._apply_mask(
        actions, "pose_motion_planning", 0.3, 0.0, np.random.default_rng(7)
    )

    assert np.array_equal(pose_mask, planning_mask)


@pytest.mark.parametrize(
    ("mask_type", "expected_known"),
    [("points", 6), ("3D_points", 9), ("random_mask", 21)],
)
def test_mam_ratio_masks_match_maniskill_retained_entry_counts(mask_type, expected_known):
    _, mask = convert_libero_absolute_to_mam._apply_mask(
        np.zeros((10, 7), dtype=np.float32),
        mask_type,
        retain_ratio=0.3,
        mask_value=0.0,
        rng=np.random.default_rng(0),
    )

    assert np.count_nonzero(mask) == expected_known


@pytest.mark.parametrize("mask_type", ["2D_video_trajectory", "2D_image_trajectory"])
def test_mam_2d_trajectory_masks_retain_xy_at_every_timestep(mask_type):
    _, mask = convert_libero_absolute_to_mam._apply_mask(
        np.zeros((6, 7), dtype=np.float32),
        mask_type,
        retain_ratio=None,
        mask_value=0.0,
        rng=np.random.default_rng(0),
    )

    assert np.all(mask[:, :2] == 1.0)
    assert np.all(mask[:, 2:] == 0.0)


@pytest.mark.parametrize("mask_type", ["mix", "mix0"])
def test_mam_mix_mask_matches_maniskill_mix0(mask_type):
    _, mask = convert_libero_absolute_to_mam._apply_mask(
        np.zeros((10, 7), dtype=np.float32),
        mask_type,
        retain_ratio=None,
        mask_value=0.0,
        rng=np.random.default_rng(0),
    )

    assert np.all(mask[:, :2] == 1.0)
    assert np.count_nonzero(mask[:, 2]) == 4
    assert np.count_nonzero(np.all(mask == 1.0, axis=1)) == 1


def test_mam_partial_and_local_planner_masks_use_contiguous_windows():
    actions = np.zeros((10, 7), dtype=np.float32)

    _, partial = convert_libero_absolute_to_mam._apply_mask(
        actions,
        "2D_partial_trajectory",
        retain_ratio=None,
        mask_value=0.0,
        rng=np.random.default_rng(0),
        mask_seq_len=3,
    )
    partial_rows = np.flatnonzero(partial[:, 0])
    assert partial_rows.size == 3
    assert np.all(np.diff(partial_rows) == 1)
    assert np.all(partial[partial_rows, :2] == 1.0)
    assert np.all(partial[:, 2:] == 0.0)

    _, local = convert_libero_absolute_to_mam._apply_mask(
        actions,
        "local_planner",
        retain_ratio=None,
        mask_value=0.0,
        rng=np.random.default_rng(0),
        mask_seq_len=3,
    )
    hidden_rows = np.flatnonzero(np.all(local == 0.0, axis=1))
    assert hidden_rows.size == 3
    assert np.all(np.diff(hidden_rows) == 1)
    assert np.all(local[np.setdiff1d(np.arange(10), hidden_rows)] == 1.0)


def test_mam_mixed_mask_specs_and_composition_assignment():
    args = SimpleNamespace(
        mask_type="random_mask",
        mask_types="pose,points,mix0",
        retain_ratio=0.2,
        retain_ratios="0.3,0.4,0.9",
        mask_seq_len=20,
        mask_seq_lens=None,
        mask_composition="0.5,0.3,0.2",
    )

    specs = convert_libero_absolute_to_mam._resolve_mask_specs(args)
    assigned = convert_libero_absolute_to_mam._assign_mask_specs(
        list(range(10)),
        specs,
        assign_mode="composition",
        seed=0,
        task_ids_by_episode=dict.fromkeys(range(10), 0),
    )

    counts = {
        mask_type: sum(rows[0]["mask_type"] == mask_type for rows in assigned.values())
        for mask_type in ("pose", "points", "mix0")
    }
    assert counts == {"pose": 5, "points": 3, "mix0": 2}
    assert [spec["retain_ratio"] for spec in specs] == [0.3, 0.4, None]


def test_mam_composition_assignment_maintains_proportions_per_task():
    specs = [
        {"mask_type": f"mask_{slot}", "mask_type_slot": slot, "composition": weight}
        for slot, weight in enumerate((0.5, 0.3, 0.2))
    ]
    episode_ids = list(range(20))
    task_ids_by_episode = {episode_id: 0 if episode_id < 10 else 1 for episode_id in episode_ids}

    assigned = convert_libero_absolute_to_mam._assign_mask_specs(
        episode_ids,
        specs,
        assign_mode="composition",
        seed=7,
        task_ids_by_episode=task_ids_by_episode,
    )

    for task_id in (0, 1):
        counts = [
            sum(
                assigned[episode_id][0]["mask_type_slot"] == slot
                for episode_id in episode_ids
                if task_ids_by_episode[episode_id] == task_id
            )
            for slot in range(3)
        ]
        assert counts == [5, 3, 2]


def test_mam_composition_assignment_rounds_within_each_task():
    specs = [{"mask_type": f"mask_{slot}", "mask_type_slot": slot, "composition": 0.25} for slot in range(4)]
    episode_ids = list(range(18))
    task_ids_by_episode = {episode_id: 0 if episode_id < 9 else 1 for episode_id in episode_ids}

    assigned = convert_libero_absolute_to_mam._assign_mask_specs(
        episode_ids,
        specs,
        assign_mode="composition",
        seed=3,
        task_ids_by_episode=task_ids_by_episode,
    )

    for task_id in (0, 1):
        counts = [
            sum(
                assigned[episode_id][0]["mask_type_slot"] == slot
                for episode_id in episode_ids
                if task_ids_by_episode[episode_id] == task_id
            )
            for slot in range(4)
        ]
        assert counts == [3, 2, 2, 2]


def test_mam_composition_assignment_requires_task_metadata():
    specs = [{"mask_type": "pose", "mask_type_slot": 0, "composition": 1.0}]

    with pytest.raises(ValueError, match="requires a task id for every episode"):
        convert_libero_absolute_to_mam._assign_mask_specs(
            [0, 1],
            specs,
            assign_mode="composition",
            seed=0,
        )


def test_mam_resolves_reference_experiment_train_and_eval_masks_independently():
    args = SimpleNamespace(
        mask_type="random_mask",
        mask_types=None,
        mask_assign_mode="one_demo_multi_mask",
        mask_composition=None,
        retain_ratio=0.2,
        retain_ratios=None,
        mask_seq_len=20,
        mask_seq_lens=None,
        train_mask_types="points,3D_points,3D_points,pose_motion_planning",
        train_mask_assign_mode="composition",
        train_mask_composition="0.25,0.25,0.25,0.25",
        train_retain_ratios="1,1,0.2,0.2",
        train_mask_seq_lens=None,
        eval_mask_types="points,3D_points,3D_points,pose_motion_planning,mix0",
        eval_mask_assign_mode="composition",
        eval_mask_composition="0.2,0.2,0.2,0.2,0.2",
        eval_retain_ratios="1,1,0.2,0.2,1",
        eval_mask_seq_lens=None,
    )

    train_specs = convert_libero_absolute_to_mam._resolve_mask_specs(args, split="train")
    eval_specs = convert_libero_absolute_to_mam._resolve_mask_specs(args, split="eval")

    assert [spec["mask_type"] for spec in train_specs] == [
        "points",
        "3D_points",
        "3D_points",
        "pose_motion_planning",
    ]
    assert [spec["retain_ratio"] for spec in train_specs] == [1.0, 1.0, 0.2, 0.2]
    assert [spec["mask_type"] for spec in eval_specs] == [
        "points",
        "3D_points",
        "3D_points",
        "pose_motion_planning",
        "mix0",
    ]
    assert [spec["retain_ratio"] for spec in eval_specs] == [1.0, 1.0, 0.2, 0.2, None]
    assert convert_libero_absolute_to_mam._resolve_mask_assign_mode(args, "train") == "composition"
    assert convert_libero_absolute_to_mam._resolve_mask_assign_mode(args, "eval") == "composition"


def test_mam_full_mask_type_is_no_longer_supported():
    with pytest.raises(ValueError, match="Unsupported mask_type='full'"):
        convert_libero_absolute_to_mam._apply_mask(
            np.zeros((5, 7), dtype=np.float32),
            mask_type="full",
            retain_ratio=0.2,
            mask_value=0.0,
            rng=np.random.default_rng(0),
        )


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


def test_mam_training_preflight_allows_opt_in_independent_eval_source(tmp_path):
    train_root = tmp_path / "train"
    eval_root = tmp_path / "eval"
    _write_mam_manifest(train_root, "train")
    _write_mam_manifest(eval_root, "eval")
    eval_manifest_path = eval_root / "meta" / "libero_pipeline.json"
    payload = json.loads(eval_manifest_path.read_text())
    payload["source_root"] = str(tmp_path / "independent_absolute")
    payload["source_repo_id"] = "local/independent_absolute"
    payload["source_episode_ids"] = [1, 2]
    write_libero_pipeline_manifest(eval_root, payload)

    cfg = _training_cfg(train_root, eval_root, overfit=False)
    cfg.trainable_config.type = "mam"
    cfg.trainable_config.mam_eval_dataset_repo_id = "local/eval"
    cfg.trainable_config.mam_eval_dataset_root = str(eval_root)
    cfg.trainable_config.mam_eval_episodes = None
    cfg.trainable_config.allow_independent_eval_source = True
    dataset = SimpleNamespace(root=train_root, meta=SimpleNamespace(robot_type="libero"))

    validate_libero_v3_training_dataset(cfg, dataset)


def test_mam_training_preflight_rejects_independent_override_for_normal_split(tmp_path):
    train_root = tmp_path / "train"
    eval_root = tmp_path / "eval"
    _write_mam_manifest(train_root, "train")
    _write_mam_manifest(eval_root, "eval")
    cfg = _training_cfg(train_root, eval_root, overfit=False)
    cfg.trainable_config.type = "mam"
    cfg.trainable_config.mam_eval_dataset_repo_id = "local/eval"
    cfg.trainable_config.mam_eval_dataset_root = str(eval_root)
    cfg.trainable_config.mam_eval_episodes = None
    cfg.trainable_config.allow_independent_eval_source = True
    dataset = SimpleNamespace(root=train_root, meta=SimpleNamespace(robot_type="libero"))

    with pytest.raises(ValueError, match="requires train and eval to identify different source datasets"):
        validate_libero_v3_training_dataset(cfg, dataset)


def test_training_preflight_allows_random_env_evaluation_without_eval_dataset(tmp_path):
    train_root = tmp_path / "train"
    _write_mam_manifest(train_root, "train")
    cfg = _training_cfg(train_root, tmp_path / "unused_eval", overfit=False)
    cfg.eval.dataset_repo_id = None
    cfg.eval.dataset_root = None
    dataset = SimpleNamespace(root=train_root, meta=SimpleNamespace(robot_type="libero"))

    validate_libero_v3_training_dataset(cfg, dataset)


def test_full_dp_launcher_rejects_invalid_eval_env_mode():
    result = subprocess.run(
        ["bash", "scripts/libero/train/run_diffusion_libero10.sh"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "EVAL_ENV_MODE": "invalid"},
    )

    assert result.returncode == 2
    assert "EVAL_ENV_MODE must be fixed or random" in result.stderr


def test_full_dp_launcher_rejects_random_eval_seed_overlap():
    result = subprocess.run(
        ["bash", "scripts/libero/train/run_diffusion_libero10.sh"],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "EVAL_ENV_MODE": "random",
            "EVAL_START_SEED": "49",
            "EVAL_N_EPISODES": "2",
        },
    )

    assert result.returncode == 2
    assert "Random eval seeds must not overlap 0..49" in result.stderr


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
        ["bash", "scripts/libero/train/run_diffusion_libero10_v3_overfit.sh", "3", "--resume=true"],
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
