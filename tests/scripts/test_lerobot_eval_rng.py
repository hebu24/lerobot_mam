#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from types import SimpleNamespace

import numpy as np
import torch
from torch import Tensor, nn

import lerobot.scripts.lerobot_eval as eval_module
from lerobot.policies import PreTrainedPolicy
from lerobot.processor.libero_relative_action_processor import (
    absolute_to_chunk_relative,
    axis_angle_to_matrix,
)
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.random_utils import set_seed


class _StubPolicy(PreTrainedPolicy):
    config_class = object
    name = "stub"

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.config = SimpleNamespace()

    def get_optim_params(self) -> dict:
        return {}

    def reset(self) -> None:
        pass

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        raise NotImplementedError

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        raise NotImplementedError


class _RelativeChunkPolicy(_StubPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(use_relative_actions=True, n_obs_steps=1, n_action_steps=2)
        self.prediction_calls = 0

    def reset(self) -> None:
        self.prediction_calls = 0

    def update_observation_queue(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return batch

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.prediction_calls += 1
        return torch.tensor(
            [
                [
                    [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                ]
            ],
            dtype=torch.float32,
        )


class _TwoStepEnv:
    num_envs = 1

    def __init__(self) -> None:
        self.steps = 0
        self.actions: list[np.ndarray] = []

    @staticmethod
    def _observation(position: tuple[float, float, float]) -> dict[str, np.ndarray]:
        state = np.zeros((1, 14), dtype=np.float32)
        state[0, :3] = position
        state[0, 6] = 1.0
        return {"agent_pos": state}

    def reset(self, seed=None):
        self.steps = 0
        return self._observation((0.4, -0.2, 0.7)), {}

    def call(self, name):
        if name == "_max_episode_steps":
            return [2]
        if name in {"task_description", "task"}:
            return ["test task"]
        raise AttributeError(name)

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self.steps += 1
        terminated = np.asarray([self.steps >= 2])
        return (
            self._observation((9.0, 9.0, 9.0)),
            np.asarray([0.0], dtype=np.float32),
            terminated,
            np.asarray([False]),
            {"is_success": np.asarray([False])},
        )


class _ClosedLoopFourStepEnv(_TwoStepEnv):
    def reset(self, seed=None):
        self.steps = 0
        self.actions.clear()
        self.position = np.asarray((0.4, -0.2, 0.7), dtype=np.float32)
        return self._observation(tuple(self.position)), {}

    def call(self, name):
        if name == "_max_episode_steps":
            return [4]
        return super().call(name)

    def step(self, action):
        action = np.asarray(action).copy()
        self.actions.append(action)
        self.steps += 1
        self.position = action[0, :3].astype(np.float32, copy=True)
        terminated = np.asarray([self.steps >= 4])
        return (
            self._observation(tuple(self.position)),
            np.asarray([0.0], dtype=np.float32),
            terminated,
            np.asarray([False]),
            {"is_success": np.asarray([False])},
        )


class _MaterializedRelativeChunkPolicy(_StubPolicy):
    def __init__(self, relative_chunks: list[Tensor], *, n_obs_steps: int = 1, n_action_steps: int = 2) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            use_relative_actions=True, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps
        )
        self.relative_chunks = relative_chunks
        self.prediction_calls = 0

    def reset(self) -> None:
        self.prediction_calls = 0

    def update_observation_queue(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return batch

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        chunk = self.relative_chunks[self.prediction_calls]
        self.prediction_calls += 1
        return chunk


def test_eval_policy_seeds_policy_rng_and_restores_caller_state(monkeypatch):
    policy_draws: list[float] = []
    extra_draws = 0

    def fake_rollout(**kwargs):
        policy_draws.append(torch.rand(1).item())
        _ = torch.rand(extra_draws)
        return {
            ACTION: torch.zeros(1, 1, 1),
            "reward": torch.zeros(1, 1),
            "success": torch.zeros(1, 1, dtype=torch.bool),
            "done": torch.ones(1, 1, dtype=torch.bool),
        }

    monkeypatch.setattr(eval_module, "rollout", fake_rollout)
    kwargs = {
        "env": SimpleNamespace(num_envs=1),
        "policy": _StubPolicy(),
        "env_preprocessor": None,
        "env_postprocessor": None,
        "preprocessor": None,
        "postprocessor": None,
        "n_episodes": 1,
        "start_seed": 123,
    }

    set_seed(999)
    expected_next_draw = torch.rand(1).item()
    set_seed(999)

    eval_module.eval_policy(**kwargs)
    first_policy_draw = policy_draws[-1]
    assert torch.rand(1).item() == expected_next_draw

    extra_draws = 100
    eval_module.eval_policy(**kwargs)
    assert policy_draws[-1] == first_policy_draw


def test_eval_policy_metrics_exclude_steps_after_first_done(monkeypatch):
    def fake_rollout(**kwargs):
        return {
            ACTION: torch.zeros(1, 2, 1),
            "reward": torch.tensor([[0.0, 1.0]]),
            "success": torch.tensor([[False, True]]),
            "done": torch.tensor([[True, True]]),
        }

    monkeypatch.setattr(eval_module, "rollout", fake_rollout)
    result = eval_module.eval_policy(
        env=SimpleNamespace(num_envs=1),
        policy=_StubPolicy(),
        env_preprocessor=None,
        env_postprocessor=None,
        preprocessor=None,
        postprocessor=None,
        n_episodes=1,
        start_seed=123,
    )

    assert result["per_episode"][0]["sum_reward"] == 0.0
    assert result["per_episode"][0]["success"] is False


def test_relative_rollout_converts_one_chunk_to_absolute_once(monkeypatch):
    monkeypatch.setattr(eval_module, "check_env_attributes_and_types", lambda env: None)
    env = _TwoStepEnv()
    policy = _RelativeChunkPolicy()

    def identity(value):
        return value

    result = eval_module.rollout(
        env=env,
        policy=policy,
        env_preprocessor=identity,
        env_postprocessor=identity,
        preprocessor=identity,
        postprocessor=identity,
    )

    executed = result[ACTION][0]
    assert policy.prediction_calls == 1
    torch.testing.assert_close(executed[:, :3], torch.tensor([[0.5, -0.2, 0.7], [0.6, -0.2, 0.7]]))
    torch.testing.assert_close(executed[:, 6], torch.tensor([1.0, -1.0]))
    assert not torch.allclose(executed[0, 3:6], torch.zeros(3))
    assert OBS_STATE not in result


def test_relative_rollout_matches_closed_loop_materialized_actions_across_chunks(monkeypatch):
    """One live anchor is used per queue fill, then refreshed for the next chunk."""
    monkeypatch.setattr(eval_module, "check_env_attributes_and_types", lambda env: None)
    env = _ClosedLoopFourStepEnv()

    recorded_states = torch.zeros(4, 14, dtype=torch.float32)
    recorded_states[:, :3] = torch.tensor(
        [[0.4, -0.2, 0.7], [0.5, -0.2, 0.7], [0.6, -0.2, 0.7], [0.7, -0.2, 0.7]]
    )
    recorded_states[:, 6] = 1.0
    absolute_actions = torch.zeros(4, 7, dtype=torch.float32)
    absolute_actions[:, :3] = torch.tensor(
        [[0.5, -0.2, 0.7], [0.6, -0.2, 0.7], [0.7, -0.2, 0.7], [0.8, -0.2, 0.7]]
    )
    absolute_actions[:, 3:6] = torch.tensor((math.pi / 2, 0.0, 0.0))
    absolute_actions[:, 6] = torch.tensor([1.0, -1.0, 1.0, -1.0])
    relative_chunks = [
        absolute_to_chunk_relative(absolute_actions[start : start + 2], recorded_states[start]).unsqueeze(0)
        for start in (0, 2)
    ]
    policy = _MaterializedRelativeChunkPolicy(relative_chunks)

    result = eval_module.rollout(
        env=env,
        policy=policy,
        env_preprocessor=lambda value: value,
        env_postprocessor=lambda value: value,
        preprocessor=lambda value: value,
        postprocessor=lambda value: value,
    )

    executed = result[ACTION][0]
    assert policy.prediction_calls == 2
    torch.testing.assert_close(executed[:, :3], absolute_actions[:, :3], atol=1e-6, rtol=0)
    torch.testing.assert_close(executed[:, 6], absolute_actions[:, 6], atol=0, rtol=0)
    torch.testing.assert_close(
        axis_angle_to_matrix(executed[:, 3:6]),
        axis_angle_to_matrix(absolute_actions[:, 3:6]),
        atol=1e-6,
        rtol=0,
    )


def test_relative_rollout_executes_current_window_not_past_action(monkeypatch):
    monkeypatch.setattr(eval_module, "check_env_attributes_and_types", lambda env: None)
    env = _TwoStepEnv()

    recorded_state = torch.zeros(14, dtype=torch.float32)
    recorded_state[:3] = torch.tensor((0.4, -0.2, 0.7))
    recorded_state[6] = 1.0
    past_current_future = torch.zeros(3, 7, dtype=torch.float32)
    past_current_future[:, :3] = torch.tensor(
        [[-9.0, -9.0, -9.0], [0.5, -0.2, 0.7], [0.6, -0.2, 0.7]]
    )
    past_current_future[:, 6] = torch.tensor([-1.0, 1.0, -1.0])
    relative_chunk = absolute_to_chunk_relative(past_current_future, recorded_state).unsqueeze(0)
    policy = _MaterializedRelativeChunkPolicy([relative_chunk], n_obs_steps=2, n_action_steps=2)

    result = eval_module.rollout(
        env=env,
        policy=policy,
        env_preprocessor=lambda value: value,
        env_postprocessor=lambda value: value,
        preprocessor=lambda value: value,
        postprocessor=lambda value: value,
    )

    executed = result[ACTION][0]
    torch.testing.assert_close(executed[:, :3], past_current_future[1:, :3], atol=1e-6, rtol=0)
    torch.testing.assert_close(executed[:, 6], past_current_future[1:, 6], atol=0, rtol=0)
