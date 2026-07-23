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

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE

from .pipeline import PolicyProcessorPipeline, ProcessorStep, ProcessorStepRegistry

LIBERO_DELTA_POS_SCALE = 0.05
LIBERO_DELTA_ROT_SCALE = 0.5

# The LIBERO state quaternion describes the Panda ``right_hand`` body, while
# OSC_POSE controls the gripper ``grip_site``. The latter is rotated -90 degrees
# around the body's z axis in the Panda gripper MJCF.
LIBERO_EEF_BODY_TO_CONTROLLER_ROTATION = (
    (0.0, 1.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
)


def _to_tensor(value: Tensor | np.ndarray, *, like: Tensor | None = None) -> Tensor:
    if isinstance(value, Tensor):
        tensor = value
    else:
        if not value.flags.writeable:
            value = value.copy()
        tensor = torch.as_tensor(value)
    if like is not None:
        tensor = tensor.to(device=like.device, dtype=like.dtype)
    elif not torch.is_floating_point(tensor):
        tensor = tensor.to(dtype=torch.float32)
    return tensor


def _maybe_numpy(tensor: Tensor, template: Tensor | np.ndarray) -> Tensor | np.ndarray:
    if isinstance(template, np.ndarray):
        return tensor.detach().cpu().numpy()
    return tensor


def _quaternion_to_matrix(quat: Tensor) -> Tensor:
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = quat.unbind(dim=-1)
    two_s = 2.0

    m00 = 1 - two_s * (y * y + z * z)
    m01 = two_s * (x * y - z * w)
    m02 = two_s * (x * z + y * w)
    m10 = two_s * (x * y + z * w)
    m11 = 1 - two_s * (x * x + z * z)
    m12 = two_s * (y * z - x * w)
    m20 = two_s * (x * z - y * w)
    m21 = two_s * (y * z + x * w)
    m22 = 1 - two_s * (x * x + y * y)

    return torch.stack(
        (
            torch.stack((m00, m01, m02), dim=-1),
            torch.stack((m10, m11, m12), dim=-1),
            torch.stack((m20, m21, m22), dim=-1),
        ),
        dim=-2,
    )


def axis_angle_to_matrix(axis_angle: Tensor | np.ndarray) -> Tensor | np.ndarray:
    """Convert axis-angle rotation vectors to rotation matrices."""
    rotvec = _to_tensor(axis_angle)
    angle = rotvec.norm(dim=-1, keepdim=True)
    half_angle = angle * 0.5
    scale = torch.where(
        angle > 1e-8,
        torch.sin(half_angle) / angle,
        0.5 - angle * angle / 48.0,
    )
    quat = torch.cat((rotvec * scale, torch.cos(half_angle)), dim=-1)
    return _maybe_numpy(_quaternion_to_matrix(quat), axis_angle)


def _matrix_to_quaternion(matrix: Tensor) -> Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices with shape (..., 3, 3), got {tuple(matrix.shape)}.")

    m = matrix
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]

    # Build one quaternion candidate for each possible largest component, then
    # select the best-conditioned candidate. Independent copysign recovery is
    # ambiguous when the angle is near pi because all antisymmetric terms can
    # be close to zero.
    q_abs = torch.sqrt(
        torch.stack(
            (
                1 + m00 + m11 + m22,
                1 + m00 - m11 - m22,
                1 - m00 + m11 - m22,
                1 - m00 - m11 + m22,
            ),
            dim=-1,
        ).clamp_min(0)
    )
    quat_by_wxyz = torch.stack(
        (
            torch.stack((q_abs[..., 0].square(), m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1].square(), m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2].square(), m12 + m21), dim=-1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3].square()), dim=-1),
        ),
        dim=-2,
    )
    candidates = quat_by_wxyz / (2.0 * q_abs.clamp_min(0.1).unsqueeze(-1))
    best = q_abs.argmax(dim=-1)
    best = best[..., None, None].expand(*best.shape, 1, 4)
    quat_wxyz = candidates.gather(dim=-2, index=best).squeeze(-2)
    quat = quat_wxyz[..., (1, 2, 3, 0)]
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def eef_body_quaternion_to_controller_matrix(quat: Tensor | np.ndarray) -> Tensor | np.ndarray:
    """Convert a LIBERO EEF body quaternion (xyzw) to the OSC controller frame."""
    quaternion = _to_tensor(quat)
    body_matrix = _quaternion_to_matrix(quaternion)
    body_to_controller = body_matrix.new_tensor(LIBERO_EEF_BODY_TO_CONTROLLER_ROTATION)
    return _maybe_numpy(body_matrix @ body_to_controller, quat)


def matrix_to_axis_angle(matrix: Tensor | np.ndarray) -> Tensor | np.ndarray:
    """Convert rotation matrices to axis-angle rotation vectors."""
    mat = _to_tensor(matrix)
    quat = _matrix_to_quaternion(mat)
    quat = torch.where(quat[..., 3:4] < 0, -quat, quat)
    xyz = quat[..., :3]
    w = quat[..., 3:4].clamp(-1.0, 1.0)
    sin_half = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w)
    scale = torch.where(sin_half > 1e-8, angle / sin_half, 2.0 * torch.ones_like(sin_half))
    return _maybe_numpy(xyz * scale, matrix)


def _broadcast_anchor_matrix(actions: Tensor, anchor_matrix: Tensor) -> Tensor:
    if actions.ndim == anchor_matrix.ndim:
        return anchor_matrix
    if actions.ndim == anchor_matrix.ndim + 1:
        return anchor_matrix.unsqueeze(-3)
    raise ValueError(
        f"Cannot broadcast anchor matrix with shape {tuple(anchor_matrix.shape)} "
        f"to actions with shape {tuple(actions.shape)}"
    )


def _broadcast_anchor_vector(actions: Tensor, anchor: Tensor) -> Tensor:
    if actions.ndim == anchor.ndim:
        return anchor
    if actions.ndim == anchor.ndim + 1:
        return anchor.unsqueeze(-2)
    raise ValueError(
        f"Cannot broadcast anchor with shape {tuple(anchor.shape)} to actions with shape {tuple(actions.shape)}"
    )


def _anchor_rotation_matrix(anchor: Tensor) -> Tensor:
    if anchor.shape[-1] >= 14:
        controller_matrix = eef_body_quaternion_to_controller_matrix(anchor[..., 3:7])
        if not isinstance(controller_matrix, Tensor):
            raise TypeError("Tensor anchor unexpectedly produced a NumPy controller matrix.")
        return controller_matrix
    start_mat = axis_angle_to_matrix(anchor[..., 3:6])
    if not isinstance(start_mat, Tensor):
        start_mat = torch.as_tensor(start_mat, device=anchor.device, dtype=anchor.dtype)
    return start_mat


def delta_to_absolute_action(
    delta_action: Tensor | np.ndarray,
    eef_pos: Tensor | np.ndarray,
    eef_mat: Tensor | np.ndarray,
) -> Tensor | np.ndarray:
    """Convert LIBERO/robosuite OSC_POSE delta action to absolute OSC_POSE action."""
    delta = _to_tensor(delta_action)
    pos = _to_tensor(eef_pos, like=delta)
    mat = _to_tensor(eef_mat, like=delta)

    target_pos = delta[..., :3] * LIBERO_DELTA_POS_SCALE + pos
    delta_rot = axis_angle_to_matrix(delta[..., 3:6] * LIBERO_DELTA_ROT_SCALE)
    if not isinstance(delta_rot, Tensor):
        delta_rot = torch.as_tensor(delta_rot, device=delta.device, dtype=delta.dtype)
    target_mat = delta_rot @ mat
    target_axis_angle = matrix_to_axis_angle(target_mat)
    if not isinstance(target_axis_angle, Tensor):
        target_axis_angle = torch.as_tensor(target_axis_angle, device=delta.device, dtype=delta.dtype)
    absolute = torch.cat((target_pos, target_axis_angle, delta[..., 6:7]), dim=-1)
    return _maybe_numpy(absolute, delta_action)


def absolute_to_chunk_relative(
    absolute_actions: Tensor | np.ndarray,
    anchor_state: Tensor | np.ndarray,
) -> Tensor | np.ndarray:
    """Convert absolute LIBERO action chunk to one chunk-relative action sequence."""
    actions = _to_tensor(absolute_actions)
    anchor = _to_tensor(anchor_state, like=actions)
    if actions.shape[-1] < 7 or anchor.shape[-1] < 6:
        raise ValueError(
            f"Expected action_dim>=7 and state_dim>=6, got {actions.shape[-1]=}, {anchor.shape[-1]=}."
        )

    anchor_pos = _broadcast_anchor_vector(actions[..., :3], anchor[..., :3])
    rel_pos = actions[..., :3] - anchor_pos

    abs_mat = axis_angle_to_matrix(actions[..., 3:6])
    start_mat = _anchor_rotation_matrix(anchor)
    if not isinstance(abs_mat, Tensor):
        abs_mat = torch.as_tensor(abs_mat, device=actions.device, dtype=actions.dtype)
    start_mat = _broadcast_anchor_matrix(abs_mat, start_mat)
    rel_axis_angle = matrix_to_axis_angle(abs_mat @ start_mat.transpose(-1, -2))
    if not isinstance(rel_axis_angle, Tensor):
        rel_axis_angle = torch.as_tensor(rel_axis_angle, device=actions.device, dtype=actions.dtype)

    rel = torch.cat((rel_pos, rel_axis_angle, actions[..., 6:7]), dim=-1)
    return _maybe_numpy(rel, absolute_actions)


def chunk_relative_to_absolute(
    relative_actions: Tensor | np.ndarray,
    anchor_state: Tensor | np.ndarray,
) -> Tensor | np.ndarray:
    """Convert one chunk-relative LIBERO action sequence back to absolute actions."""
    actions = _to_tensor(relative_actions)
    anchor = _to_tensor(anchor_state, like=actions)
    if actions.shape[-1] < 7 or anchor.shape[-1] < 6:
        raise ValueError(
            f"Expected action_dim>=7 and state_dim>=6, got {actions.shape[-1]=}, {anchor.shape[-1]=}."
        )

    anchor_pos = _broadcast_anchor_vector(actions[..., :3], anchor[..., :3])
    abs_pos = actions[..., :3] + anchor_pos

    rel_mat = axis_angle_to_matrix(actions[..., 3:6])
    start_mat = _anchor_rotation_matrix(anchor)
    if not isinstance(rel_mat, Tensor):
        rel_mat = torch.as_tensor(rel_mat, device=actions.device, dtype=actions.dtype)
    start_mat = _broadcast_anchor_matrix(rel_mat, start_mat)
    abs_axis_angle = matrix_to_axis_angle(rel_mat @ start_mat)
    if not isinstance(abs_axis_angle, Tensor):
        abs_axis_angle = torch.as_tensor(abs_axis_angle, device=actions.device, dtype=actions.dtype)

    absolute = torch.cat((abs_pos, abs_axis_angle, actions[..., 6:7]), dim=-1)
    return _maybe_numpy(absolute, relative_actions)


def slice_current_action_window(
    action_chunk: Tensor,
    *,
    n_obs_steps: int,
    n_action_steps: int,
) -> Tensor:
    """Return the executable window from a model-predicted action chunk.

    Diffusion-style chunks are indexed from the oldest observation. When
    ``n_obs_steps > 1``, the first generated action(s) correspond to past
    observation frames and must not be sent to the environment. The current
    environment step starts at ``n_obs_steps - 1``.

    Some policy implementations already apply this slice inside
    ``predict_action_chunk`` / ``generate_actions`` and therefore return exactly
    ``n_action_steps`` actions. In that case, the chunk is already executable.
    """
    if action_chunk.ndim < 3:
        raise ValueError(
            f"Expected action chunk with shape (batch, horizon, action_dim), got {action_chunk.shape}."
        )
    if n_obs_steps < 1:
        raise ValueError(f"n_obs_steps must be >= 1, got {n_obs_steps}.")
    if n_action_steps < 1:
        raise ValueError(f"n_action_steps must be >= 1, got {n_action_steps}.")
    if action_chunk.shape[1] == n_action_steps:
        return action_chunk

    start = n_obs_steps - 1
    end = start + n_action_steps
    if end > action_chunk.shape[1]:
        raise ValueError(
            "Executable action window exceeds predicted chunk horizon: "
            f"{start=}, {end=}, horizon={action_chunk.shape[1]}."
        )
    return action_chunk[:, start:end]


@ProcessorStepRegistry.register("libero_chunk_relative_actions_processor")
@dataclass
class LiberoChunkRelativeActionsProcessorStep(ProcessorStep):
    """Convert absolute LIBERO action chunks to chunk-relative actions before normalization."""

    enabled: bool = False
    anchor_index: int = -1

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        action = transition.get(TransitionKey.ACTION)
        state = observation.get(OBS_STATE) if observation else None

        if not self.enabled or action is None or state is None:
            return transition

        anchor_state = state[:, self.anchor_index] if state.ndim >= 3 else state
        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = absolute_to_chunk_relative(action, anchor_state)
        return new_transition

    def get_config(self) -> dict[str, object]:
        return {"enabled": self.enabled, "anchor_index": self.anchor_index}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("libero_chunk_relative_actions_postprocessor")
@dataclass
class LiberoChunkRelativeActionsPostprocessorStep(ProcessorStep):
    """Decode a LIBERO chunk-relative policy output into executable absolute goals."""

    enabled: bool = False
    n_obs_steps: int = 1
    n_action_steps: int = 1

    @property
    def requires_transition_context(self) -> bool:
        return self.enabled

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if not self.enabled or action is None:
            return transition

        observation = transition.get(TransitionKey.OBSERVATION)
        state = observation.get(OBS_STATE) if observation else None
        if state is None:
            raise ValueError(
                "LIBERO chunk-relative action postprocessing requires observation.state "
                "from the chunk generation step."
            )

        anchor_state = state[:, -1] if state.ndim >= 3 else state
        absolute_action = chunk_relative_to_absolute(action, anchor_state)
        if not isinstance(absolute_action, Tensor):
            absolute_action = torch.as_tensor(absolute_action, device=action.device, dtype=action.dtype)
        if absolute_action.ndim >= 3:
            absolute_action = slice_current_action_window(
                absolute_action,
                n_obs_steps=self.n_obs_steps,
                n_action_steps=self.n_action_steps,
            )
        elif absolute_action.ndim != 2:
            raise ValueError(
                "Expected LIBERO actions with shape (batch, action_dim) or "
                f"(batch, horizon, action_dim), got {tuple(absolute_action.shape)}."
            )

        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = absolute_action
        return new_transition

    def get_config(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "n_obs_steps": self.n_obs_steps,
            "n_action_steps": self.n_action_steps,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def ensure_libero_chunk_relative_actions_postprocessor(
    postprocessor: PolicyProcessorPipeline,
    *,
    enabled: bool,
    n_obs_steps: int,
    n_action_steps: int,
) -> None:
    """Add the context-aware decoder when loading a legacy processor checkpoint."""
    if not enabled or any(
        isinstance(step, LiberoChunkRelativeActionsPostprocessorStep) for step in postprocessor.steps
    ):
        return

    decoder = LiberoChunkRelativeActionsPostprocessorStep(
        enabled=True,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
    )
    steps = list(postprocessor.steps)
    insert_at = next(
        (
            index
            for index, step in enumerate(steps)
            if getattr(step.__class__, "_registry_name", None) == "device_processor"
        ),
        len(steps),
    )
    steps.insert(insert_at, decoder)
    postprocessor.steps = steps
