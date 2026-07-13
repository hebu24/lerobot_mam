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

import math

import numpy as np
import torch

from lerobot.processor.libero_relative_action_processor import (
    LIBERO_EEF_BODY_TO_CONTROLLER_ROTATION,
    absolute_to_chunk_relative,
    axis_angle_to_matrix,
    chunk_relative_to_absolute,
    eef_body_quaternion_to_controller_matrix,
    matrix_to_axis_angle,
)


def _quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)
    x, y, z, w = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), -1),
            torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), -1),
            torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), -1),
        ),
        dim=-2,
    )


def test_matrix_axis_angle_roundtrip_at_pi() -> None:
    axes = torch.tensor(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 2.0, -3.0)),
        dtype=torch.float32,
    )
    axes = axes / axes.norm(dim=-1, keepdim=True)
    angles = torch.tensor((math.pi - 1e-6, math.pi, math.pi + 1e-6), dtype=torch.float32)
    rotations = (axes[:, None] * angles[None, :, None]).reshape(-1, 3)
    rotations = torch.cat((rotations, torch.tensor(((-3.084729, -0.324334, -0.498857),))))

    matrices = axis_angle_to_matrix(rotations)
    reconstructed = axis_angle_to_matrix(matrix_to_axis_angle(matrices))

    torch.testing.assert_close(reconstructed, matrices, atol=1e-6, rtol=1e-6)


def test_matrix_axis_angle_roundtrip_random_so3() -> None:
    generator = torch.Generator().manual_seed(2718)
    quaternions = torch.randn(10_000, 4, generator=generator, dtype=torch.float32)
    matrices = _quaternion_to_matrix(quaternions)

    reconstructed = axis_angle_to_matrix(matrix_to_axis_angle(matrices))

    torch.testing.assert_close(reconstructed, matrices, atol=1e-6, rtol=1e-6)


def test_eef_body_quaternion_is_right_multiplied_into_controller_frame() -> None:
    body_quaternion = torch.tensor((0.2, -0.3, 0.1, 0.92), dtype=torch.float64)
    body_quaternion /= body_quaternion.norm()
    body_matrix = _quaternion_to_matrix(body_quaternion)
    body_to_controller = torch.tensor(LIBERO_EEF_BODY_TO_CONTROLLER_ROTATION, dtype=torch.float64)

    controller_matrix = eef_body_quaternion_to_controller_matrix(body_quaternion)

    torch.testing.assert_close(controller_matrix, body_matrix @ body_to_controller)
    assert not torch.allclose(controller_matrix, body_to_controller @ body_matrix)


def test_14d_state_anchor_uses_controller_frame_for_relative_actions() -> None:
    body_quaternion = torch.tensor((0.2, -0.3, 0.1, 0.92), dtype=torch.float64)
    body_quaternion /= body_quaternion.norm()
    controller_matrix = eef_body_quaternion_to_controller_matrix(body_quaternion)
    controller_axis_angle = matrix_to_axis_angle(controller_matrix)

    anchor = torch.zeros(14, dtype=torch.float64)
    anchor[:3] = torch.tensor((0.4, -0.2, 0.7), dtype=torch.float64)
    anchor[3:7] = body_quaternion
    actions = torch.zeros(5, 7, dtype=torch.float64)
    actions[:, :3] = anchor[:3]
    actions[:, 3:6] = controller_axis_angle
    actions[:, 6] = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64)

    relative = absolute_to_chunk_relative(actions, anchor)
    reconstructed = chunk_relative_to_absolute(relative, anchor)

    torch.testing.assert_close(relative[:, :6], torch.zeros_like(relative[:, :6]), atol=1e-12, rtol=0)
    torch.testing.assert_close(reconstructed[:, :3], actions[:, :3], atol=1e-12, rtol=0)
    torch.testing.assert_close(reconstructed[:, 6], actions[:, 6], atol=1e-12, rtol=0)
    torch.testing.assert_close(
        axis_angle_to_matrix(reconstructed[:, 3:6]),
        axis_angle_to_matrix(actions[:, 3:6]),
        atol=1e-12,
        rtol=0,
    )


def test_eef_body_controller_conversion_preserves_numpy_type() -> None:
    body_quaternion = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)

    controller_matrix = eef_body_quaternion_to_controller_matrix(body_quaternion)

    assert isinstance(controller_matrix, np.ndarray)
    np.testing.assert_allclose(
        controller_matrix,
        np.asarray(LIBERO_EEF_BODY_TO_CONTROLLER_ROTATION, dtype=np.float32),
        atol=0,
        rtol=0,
    )
