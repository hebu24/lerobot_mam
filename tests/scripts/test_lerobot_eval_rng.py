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

from types import SimpleNamespace

import torch
from torch import Tensor, nn

import lerobot.scripts.lerobot_eval as eval_module
from lerobot.policies import PreTrainedPolicy
from lerobot.utils.constants import ACTION
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
