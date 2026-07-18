from collections import deque
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.mam.configuration_mam import MamConfig
from lerobot.policies.mam.eval_mam import (
    MamEvalEpisode,
    _resolve_stpm_paths,
    _slice_episode_window,
    _stack_history,
    configure_mam_eval_init_state_ids,
    eval_mam_policy,
    eval_mam_policy_all,
    load_mam_eval_episodes,
    make_stpm_encoder,
)
from lerobot.scripts.lerobot_eval import _prepare_mam_eval_episodes
from lerobot.utils.constants import ACTION, OBS_STATE


def _config(**overrides) -> MamConfig:
    kwargs = {
        "device": "cpu",
        "input_features": {OBS_STATE: PolicyFeature(FeatureType.STATE, (6,))},
        "output_features": {ACTION: PolicyFeature(FeatureType.ACTION, (7,))},
        "horizon": 8,
        "n_action_steps": 4,
        "n_obs_steps": 2,
        "down_dims": (32, 64),
        "diffusion_step_embed_dim": 16,
        "n_groups": 8,
        "num_train_timesteps": 4,
        "mas_long_feature_dim": 8,
        "mas_long_forward_length": 8,
        "mas_short_window_horizon": 4,
    }
    kwargs.update(overrides)
    return MamConfig(**kwargs)


def _episode(index: int, task_id: int, length: int = 20) -> MamEvalEpisode:
    values = torch.arange(length, dtype=torch.float32).view(length, 1).repeat(1, 7)
    return MamEvalEpisode(
        episode_index=index,
        source_episode_id=100 + index,
        init_state_id=index,
        init_state=None,
        suite="libero_10",
        task_id=task_id,
        mask_type="random_mask",
        mask_type_slot=0,
        task=f"task {task_id}",
        mas_action_absolute=values,
        mas_action_mask=torch.ones_like(values),
        progress=torch.linspace(0, 1, length).unsqueeze(-1),
    )


def test_stack_history_uses_training_frame_gap_and_left_padding():
    history = deque(torch.tensor([[float(index)]]) for index in range(7))
    assert _stack_history(history, target_len=3, frame_gap=2).flatten().tolist() == [2.0, 4.0, 6.0]

    short = deque([torch.tensor([[3.0]]), torch.tensor([[4.0]])])
    assert _stack_history(short, target_len=3, frame_gap=2).flatten().tolist() == [3.0, 3.0, 4.0]


def test_eval_mam_window_clamps_each_relative_offset_without_shifting():
    config = _config(n_obs_steps=3, mas_long_backward_length=2)
    episode = _episode(0, 0, length=10)

    actions, _, _ = _slice_episode_window(episode, progress=0.0, config=config)

    expected = torch.as_tensor(config.mam_delta_indices).clamp(0, 9).float()
    assert torch.equal(actions[:, 0], expected)


def test_mam_eval_requires_absolute_libero_control_mode():
    cfg = SimpleNamespace(
        env=SimpleNamespace(type="libero", control_mode="relative", init_state_ids=None),
        eval=SimpleNamespace(batch_size=1),
    )

    with pytest.raises(ValueError, match="control_mode='absolute'"):
        configure_mam_eval_init_state_ids(cfg, [_episode(0, 0)], n_episodes=1)


def test_standalone_eval_prepares_mam_before_environment_creation(monkeypatch):
    episode = _episode(0, 0)
    policy_config = _config(
        mam_eval_dataset_repo_id="local/eval",
        mam_eval_dataset_root="/tmp/eval",
        mam_eval_episodes=[9],
    )
    cfg = SimpleNamespace(
        policy=policy_config,
        env=SimpleNamespace(
            type="libero",
            task="libero_10",
            task_ids=None,
            control_mode="absolute",
            init_state_ids=None,
            init_state_ids_by_task=None,
            init_state_values=None,
            init_state_values_by_task=None,
            num_steps_wait=0,
        ),
        eval=SimpleNamespace(batch_size=1, n_episodes=1),
    )
    calls = []

    def fake_load(**kwargs):
        calls.append(kwargs)
        return [episode]

    monkeypatch.setattr("lerobot.policies.mam.eval_mam.load_mam_eval_episodes", fake_load)

    loaded = _prepare_mam_eval_episodes(cfg)

    assert loaded == [episode]
    assert calls == [{"repo_id": "local/eval", "root": "/tmp/eval", "episodes": [9]}]
    assert cfg.env.task_ids == [0]
    assert cfg.env.init_state_ids_by_task == {"libero_10/0": [0]}


def test_mam_eval_loader_rejects_uncertified_libero_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "lerobot.policies.mam.eval_mam.LeRobotDatasetMetadata",
        lambda *args, **kwargs: SimpleNamespace(robot_type="libero", root=tmp_path),
    )

    with pytest.raises(FileNotFoundError, match="not certified"):
        load_mam_eval_episodes("local/legacy", root=tmp_path)


def test_mam_eval_loader_rejects_non_libero_robot_type(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "lerobot.policies.mam.eval_mam.LeRobotDatasetMetadata",
        lambda *args, **kwargs: SimpleNamespace(robot_type="unknown", root=tmp_path),
    )

    with pytest.raises(ValueError, match="robot_type='libero'"):
        load_mam_eval_episodes("local/mislabeled", root=tmp_path)


def test_eval_all_consumes_only_the_same_selected_episode_prefix(monkeypatch):
    config = _config()
    policy = SimpleNamespace(config=config)
    episodes = [_episode(0, 0), _episode(1, 1), _episode(2, 0), _episode(3, 1), _episode(4, 0)]
    calls = []

    class FakeEnv:
        def close(self):
            pass

    def fake_eval_mam_policy(**kwargs):
        task_episodes = kwargs["episodes"]
        calls.append([episode.episode_index for episode in task_episodes])
        per_episode = [
            {
                "episode_ix": local_index,
                "sum_reward": 0.0,
                "max_reward": 0.0,
                "success": False,
                "mask_type": episode.mask_type,
                "mask_type_slot": episode.mask_type_slot,
            }
            for local_index, episode in enumerate(task_episodes)
        ]
        return {"per_episode": per_episode, "aggregated": {"pc_success": 0.0}}

    monkeypatch.setattr("lerobot.policies.mam.eval_mam.make_stpm_encoder", lambda *args, **kwargs: None)
    monkeypatch.setattr("lerobot.policies.mam.eval_mam.eval_mam_policy", fake_eval_mam_policy)

    result = eval_mam_policy_all(
        envs={"libero_10": {0: FakeEnv(), 1: FakeEnv()}},
        policy=policy,
        env_preprocessor=None,
        env_postprocessor=None,
        preprocessor=None,
        postprocessor=None,
        episodes=episodes,
        n_episodes=3,
        start_seed=11,
    )

    assert calls == [[0, 2], [1]]
    assert result["overall"]["n_episodes"] == 3
    assert [episode["episode_ix"] for episode in result["per_episode"]] == [0, 1, 2]


def test_task_scoped_stpm_rejects_checkpoint_from_another_task(monkeypatch, tmp_path):
    root = tmp_path / "stpm"
    (root / "checkpoints").mkdir(parents=True)
    (root / "checkpoints" / "reward_best.pt").touch()
    (root / "config.yaml").touch()
    config = _config(stpm_paths={"libero_10/2": str(root)})

    class FakeEncoder:
        def __init__(self, **kwargs):
            self.cfg = {"split_identity": {"tasks": ["8"]}}

    monkeypatch.setattr("lerobot.policies.mam.eval_mam.STPMEncoder", FakeEncoder)

    with pytest.raises(ValueError, match="task mismatch"):
        make_stpm_encoder(config, task_group="libero_10", task_id=2)


def test_task_stpm_root_takes_precedence_over_global_files(tmp_path):
    task_root = tmp_path / "task2"
    config = _config(
        stpm_path=str(tmp_path / "global"),
        stpm_checkpoint_path=str(tmp_path / "global.pt"),
        stpm_config_path=str(tmp_path / "global.json"),
        stpm_paths={"libero_10/2": str(task_root)},
    )

    checkpoint, stpm_config = _resolve_stpm_paths(config, "libero_10", 2)

    assert checkpoint == task_root / "checkpoints" / "reward_best.pt"
    assert stpm_config == task_root / "config.yaml"


def test_eval_mam_rng_isolated_and_terminal_transition_counted_once(monkeypatch):
    episode = _episode(0, 0)
    draws = []

    class FakePolicy:
        def eval(self):
            return self

    class FakeEnv:
        num_envs = 1

    def fake_rollout(**kwargs):
        draws.append(float(torch.rand(())))
        assert kwargs["seeds"] == [7]
        return {
            ACTION: torch.zeros(1, 3, 7),
            "reward": torch.tensor([[1.0, 2.0, 100.0]]),
            "success": torch.tensor([[False, True, True]]),
            "done": torch.tensor([[False, True, True]]),
        }

    monkeypatch.setattr("lerobot.policies.mam.eval_mam.rollout_mam", fake_rollout)
    rng_before = torch.random.get_rng_state().clone()
    kwargs = {
        "env": FakeEnv(),
        "policy": FakePolicy(),
        "env_preprocessor": None,
        "env_postprocessor": None,
        "preprocessor": None,
        "postprocessor": None,
        "episodes": [episode],
        "n_episodes": 1,
        "start_seed": 7,
    }
    first = eval_mam_policy(**kwargs)
    second = eval_mam_policy(**kwargs)

    assert draws[0] == draws[1]
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert first["per_episode"][0]["sum_reward"] == 3.0
    assert second["per_episode"][0]["source_episode_id"] == 100
