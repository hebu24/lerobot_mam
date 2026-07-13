from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.scripts.lerobot_eval import validate_libero_action_semantics
from scripts.convert_libero_absolute_to_mam import _as_float_list
from scripts.prepare_libero10_v3_overfit import select_overfit_episodes


def _row(task: int, episode: int, source: int, slot: int = 0):
    return {
        "episode_index": episode,
        "libero/task_id": task,
        "libero/source_episode_id": source,
        "libero/suite": "libero_10",
        "libero/init_state": [0.1, float(source)],
        "libero/source_file": f"task{task}.hdf5",
        "libero/source_demo": f"demo_{source}",
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
        validate_libero_action_semantics(cfg)

    cfg.env.control_mode = "absolute"
    validate_libero_action_semantics(cfg)
