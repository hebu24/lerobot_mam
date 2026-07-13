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

import json
from types import SimpleNamespace

import torch

from lerobot.scripts.lerobot_train import (
    _restore_resume_eval_state,
    _trim_jsonl_after_step,
    apply_diffusion_relative_action_stats,
    apply_overfit_subset_stats,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def test_apply_overfit_subset_stats_updates_numeric_policy_features_only():
    frames = {
        OBS_STATE: [torch.tensor([1.0, 10.0]), torch.tensor([3.0, 14.0])],
        ACTION: [torch.tensor([-1.0]), torch.tensor([1.0])],
        "observation.images.image": [torch.zeros(3, 2, 2), torch.ones(3, 2, 2)],
    }
    old_image_stats = {"mean": torch.tensor([0.5])}
    dataset = SimpleNamespace(
        reader=SimpleNamespace(hf_dataset=frames),
        meta=SimpleNamespace(
            features={
                OBS_STATE: {"dtype": "float32"},
                ACTION: {"dtype": "float32"},
                "observation.images.image": {"dtype": "image"},
            },
            stats={
                OBS_STATE: {"min": torch.tensor([-100.0, -100.0])},
                ACTION: {"min": torch.tensor([-100.0])},
                "observation.images.image": old_image_stats,
            },
        ),
    )
    cfg = SimpleNamespace(
        overfit_test=True,
        dataset=SimpleNamespace(episodes=[1, 10], streaming=False),
        trainable_config=SimpleNamespace(
            input_features={},
            output_features={},
        ),
    )

    apply_overfit_subset_stats(cfg, dataset)

    assert torch.equal(dataset.meta.stats[OBS_STATE]["min"], torch.tensor([1.0, 10.0]))
    assert torch.equal(dataset.meta.stats[OBS_STATE]["max"], torch.tensor([3.0, 14.0]))
    assert torch.equal(dataset.meta.stats[OBS_STATE]["mean"], torch.tensor([2.0, 12.0]))
    assert torch.equal(dataset.meta.stats[OBS_STATE]["count"], torch.tensor([2]))
    assert torch.equal(dataset.meta.stats[ACTION]["min"], torch.tensor([-1.0]))
    assert dataset.meta.stats["observation.images.image"] is old_image_stats


def test_mam_relative_stats_replace_raw_overfit_action_stats(monkeypatch):
    relative_stats = {"min": torch.full((7,), -0.25), "max": torch.full((7,), 0.25)}
    calls = []

    def fake_compute(**kwargs):
        calls.append(kwargs)
        return relative_stats

    monkeypatch.setattr(
        "lerobot.datasets.compute_stats.compute_libero_relative_action_stats",
        fake_compute,
    )
    dataset = SimpleNamespace(
        hf_dataset=object(),
        meta=SimpleNamespace(
            features={ACTION: {"dtype": "float32"}, OBS_STATE: {"dtype": "float32"}},
            stats={ACTION: {"min": torch.full((7,), 10.0)}},
        ),
    )
    cfg = SimpleNamespace(
        num_workers=3,
        trainable_config=SimpleNamespace(
            type="mam",
            action_delta_indices=[-1, 0, 1],
            use_relative_actions=True,
        ),
    )

    apply_diffusion_relative_action_stats(cfg, dataset)

    assert dataset.meta.stats[ACTION] is relative_stats
    assert calls == [
        {
            "hf_dataset": dataset.hf_dataset,
            "action_delta_indices": [-1, 0, 1],
            "num_workers": 3,
        }
    ]


def test_resume_trims_future_logs_and_restores_best_eval(tmp_path):
    output_dir = tmp_path / "run"
    logs_dir = output_dir / "logs"
    checkpoints_dir = output_dir / "checkpoints"
    logs_dir.mkdir(parents=True)
    (checkpoints_dir / "000100").mkdir(parents=True)
    eval_path = logs_dir / "eval_metrics.jsonl"
    records = [
        {"step": 100, "metrics": {"overall": {"pc_success": 50.0, "avg_sum_reward": 0.5}}},
        {"step": 200, "metrics": {"overall": {"pc_success": 0.0, "avg_sum_reward": 0.0}}},
    ]
    eval_path.write_text("".join(f"{json.dumps(record)}\n" for record in records))

    kept = _trim_jsonl_after_step(eval_path, max_step=100)
    score, checkpoint_dir = _restore_resume_eval_state(
        kept,
        output_dir=output_dir,
        total_steps=1000,
        checkpoint_path=None,
    )

    assert [record["step"] for record in kept] == [100]
    assert '"step": 200' not in eval_path.read_text()
    assert score == (50.0, 0.5)
    assert checkpoint_dir == checkpoints_dir / "000100"
