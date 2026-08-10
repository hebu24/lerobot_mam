from types import SimpleNamespace

import pytest

import lerobot.envs.libero_eval as libero_eval
from lerobot.envs.configs import LiberoEnv


class EpisodeRows(list):
    column_names = [
        "episode_index",
        "libero/task_id",
        "libero/suite",
        "libero/init_state",
    ]


def _cfg(n_episodes: int = 2):
    return SimpleNamespace(
        env=LiberoEnv(
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
        libero_eval,
        "LeRobotDatasetMetadata",
        lambda *args, **kwargs: SimpleNamespace(episodes=rows),
    )
    cfg = _cfg()

    cfg.env.prepare_evaluation(cfg)

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
        libero_eval,
        "LeRobotDatasetMetadata",
        lambda *args, **kwargs: SimpleNamespace(episodes=rows),
    )

    with pytest.raises(ValueError, match="interpreted per task"):
        cfg = _cfg(n_episodes=2)
        cfg.env.prepare_evaluation(cfg)


def _random_cfg(*, start_seed=None, n_episodes=50):
    return SimpleNamespace(
        env=LiberoEnv(task="libero_10", init_states=False),
        eval=SimpleNamespace(
            dataset_repo_id=None,
            start_seed=start_seed,
            n_episodes=n_episodes,
        ),
    )


def test_random_libero_eval_defaults_to_lpb_seed_range():
    cfg = _random_cfg()

    cfg.env.prepare_evaluation(cfg)

    assert cfg.eval.start_seed == 100_000


def test_random_libero_eval_preserves_non_overlapping_explicit_seed():
    cfg = _random_cfg(start_seed=120_000, n_episodes=7)

    cfg.env.prepare_evaluation(cfg)

    assert cfg.eval.start_seed == 120_000


@pytest.mark.parametrize(("start_seed", "n_episodes"), [(0, 1), (49, 2)])
def test_random_libero_eval_rejects_demo_seed_overlap(start_seed, n_episodes):
    cfg = _random_cfg(start_seed=start_seed, n_episodes=n_episodes)

    with pytest.raises(ValueError, match=r"must not overlap demonstration seeds 0\.\.49"):
        cfg.env.prepare_evaluation(cfg)
