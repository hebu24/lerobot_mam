#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import numpy as np
import torch

from lerobot.envs.utils import preprocess_observation
from lerobot.processor.env_processor import LiberoProcessorStep
from lerobot.processor.pipeline import PolicyProcessorPipeline


def test_libero_processor_state_and_image_orientation():
    seed = 42
    np.random.seed(seed)

    batch_size = 5
    obs1 = {
        "pixels": {
            "image": (np.random.rand(batch_size, 256, 256, 3) * 255).astype(np.uint8),
            "image2": (np.random.rand(batch_size, 256, 256, 3) * 255).astype(np.uint8),
        },
        "robot_state": {
            "eef": {
                "pos": np.random.randn(batch_size, 3),
                "quat": np.random.randn(batch_size, 4),
                "mat": np.random.randn(batch_size, 3, 3),
            },
            "gripper": {
                "qpos": np.random.randn(batch_size, 2),
                "qvel": np.random.randn(batch_size, 2),
            },
            "joints": {
                "pos": np.random.randn(batch_size, 7),
                "vel": np.random.randn(batch_size, 7),
            },
        },
    }

    observation = preprocess_observation(obs1)
    libero_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
    processed_obs = libero_preprocessor(observation)
    assert "observation.state" in processed_obs
    state = processed_obs["observation.state"]
    assert isinstance(state, torch.Tensor)
    assert state.dtype == torch.float32

    assert state.shape[0] == batch_size
    assert state.shape[1] == 14

    assert "observation.images.image" in processed_obs
    assert "observation.images.image2" in processed_obs

    assert isinstance(processed_obs["observation.images.image"], torch.Tensor)
    assert isinstance(processed_obs["observation.images.image2"], torch.Tensor)

    assert processed_obs["observation.images.image"].shape == (batch_size, 3, 256, 256)
    assert processed_obs["observation.images.image2"].shape == (batch_size, 3, 256, 256)
    assert torch.equal(processed_obs["observation.images.image"], observation["observation.images.image"])

    flipped_obs = PolicyProcessorPipeline(steps=[LiberoProcessorStep(flip_images=True)])(observation)
    assert torch.equal(
        flipped_obs["observation.images.image"],
        torch.flip(observation["observation.images.image"], dims=[2, 3]),
    )
