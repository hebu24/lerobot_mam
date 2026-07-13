from types import SimpleNamespace

import pytest

import lerobot.scripts.lerobot_eval as eval_module


class EpisodeRows(list):
    column_names = [
        "episode_index",
        "libero/task_id",
        "libero/suite",
        "libero/init_state",
    ]


def _cfg(n_episodes: int = 2):
    return SimpleNamespace(
        env=SimpleNamespace(
            type="libero",
            task="libero_10",
            task_ids=None,
            init_state_ids=None,
            init_state_ids_by_task=None,
            init_state_values=None,
            init_state_values_by_task=None,
            num_steps_wait=10,
        ),
        eval=SimpleNamespace(
            dataset_repo_id="local/libero10_mam_v3_eval",
            dataset_root="/tmp/eval",
            dataset_episodes=None,
            n_episodes=n_episodes,
            batch_size=4,
        ),
    )


def test_fixed_libero_eval_uses_recorded_init_states_per_task(monkeypatch):
    rows = EpisodeRows(
        [
            {
                "episode_index": episode,
                "libero/task_id": task,
                "libero/suite": "libero_10",
                "libero/init_state": [float(episode), float(task)],
            }
            for episode, task in [(4, 0), (1, 0), (3, 0), (2, 1), (0, 1)]
        ]
    )
    monkeypatch.setattr(
        eval_module,
        "LeRobotDatasetMetadata",
        lambda *args, **kwargs: SimpleNamespace(episodes=rows),
    )
    cfg = _cfg()

    eval_module.configure_fixed_libero_eval_from_dataset(cfg)

    assert cfg.env.task_ids == [0, 1]
    assert cfg.env.init_state_values_by_task == {
        "libero_10/0": [[1.0, 0.0], [3.0, 0.0]],
        "libero_10/1": [[0.0, 1.0], [2.0, 1.0]],
    }
    assert cfg.env.init_state_ids_by_task is None
    assert cfg.env.num_steps_wait == 0
    assert cfg.eval.batch_size == 2


def test_fixed_libero_eval_requires_n_episodes_for_every_task(monkeypatch):
    rows = EpisodeRows(
        [
            {
                "episode_index": 0,
                "libero/task_id": 0,
                "libero/suite": "libero_10",
                "libero/init_state": [0.0],
            }
        ]
    )
    monkeypatch.setattr(
        eval_module,
        "LeRobotDatasetMetadata",
        lambda *args, **kwargs: SimpleNamespace(episodes=rows),
    )

    with pytest.raises(ValueError, match="interpreted per task"):
        eval_module.configure_fixed_libero_eval_from_dataset(_cfg(n_episodes=2))
