from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.converters import create_transition, transition_to_batch
from lerobot.processor.libero_relative_action_processor import absolute_to_chunk_relative
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import ACTION, OBS_PREFIX, OBS_STATE
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_mam import MamConfig

MAM_MAS_ACTION_ABSOLUTE = "mam.mas_action_absolute"
MAM_MAS_ACTION_MASK = "mam.mas_action_mask"
MAM_PROGRESS = "mam.progress"
MAM_LONG_WINDOW = "mam.mas_long_window"
MAM_LONG_WINDOW_MASK = "mam.mas_long_window_mask"
MAM_SHORT_WINDOW = "mam.mas_short_window"
MAM_SHORT_WINDOW_MASK = "mam.mas_short_window_mask"
MAM_ACTION_MASK = "mam.action_mask"


def mam_batch_to_transition(batch: dict[str, Any]) -> EnvTransition:
    action = batch.get(ACTION)
    observation = {
        key: value
        for key, value in batch.items()
        if key.startswith(OBS_PREFIX) or key.startswith("mam.")
    }
    complementary_data = {key: value for key, value in batch.items() if "_is_pad" in key}
    for key in ("task", "index", "task_index", "episode_index", "timestamp"):
        if key in batch:
            complementary_data[key] = batch[key]
    return create_transition(
        observation=observation if observation else None,
        action=action,
        complementary_data=complementary_data if complementary_data else None,
    )


def _normalize_with_action_stats(value: Tensor, stats: dict[str, Tensor], eps: float = 1e-8) -> Tensor:
    min_val = stats["min"].to(device=value.device, dtype=value.dtype)
    max_val = stats["max"].to(device=value.device, dtype=value.dtype)
    denom = torch.where(max_val == min_val, torch.full_like(max_val, eps), max_val - min_val)
    return 2.0 * (value - min_val) / denom - 1.0


def _slice_window(values: Tensor, start: int, length: int) -> Tensor:
    if length <= 0:
        return values.new_empty((*values.shape[:-2], 0, values.shape[-1]))
    start = max(0, int(start))
    end = min(values.shape[-2], start + int(length))
    out = values[..., start:end, :]
    if out.shape[-2] < length:
        pad = out[..., -1:, :].repeat_interleave(length - out.shape[-2], dim=-2)
        out = torch.cat((out, pad), dim=-2)
    return out


@dataclass
@ProcessorStepRegistry.register(name="mam_features_processor")
class MamFeaturesProcessorStep(ProcessorStep):
    """Build MAM relative/normalized conditioning windows before model forward."""

    action_stats: dict[str, Any] | None = None
    mas_long_backward_length: int = 0
    mas_long_forward_length: int = 16
    mas_short_window_horizon: int = 8
    eps: float = 1e-8

    def __post_init__(self):
        self._tensor_action_stats = None
        if self.action_stats is not None:
            self._tensor_action_stats = {
                key: torch.as_tensor(value, dtype=torch.float32)
                for key, value in self.action_stats.items()
                if key in {"min", "max"}
            }

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        action = transition.get(TransitionKey.ACTION)
        if observation is None:
            return transition

        new_transition = transition.copy()
        new_observation = dict(observation)
        state = new_observation.get(OBS_STATE)

        if action is not None and state is not None:
            anchor_state = state[:, -1] if state.ndim >= 3 else state
            new_transition[TransitionKey.ACTION] = absolute_to_chunk_relative(action, anchor_state)

        mas_abs = new_observation.get(MAM_MAS_ACTION_ABSOLUTE)
        mas_mask = new_observation.get(MAM_MAS_ACTION_MASK)
        progress = new_observation.get(MAM_PROGRESS)
        if mas_abs is None or mas_mask is None or progress is None or state is None:
            new_transition[TransitionKey.OBSERVATION] = new_observation
            return new_transition

        if self._tensor_action_stats is None:
            raise ValueError("MAM features require action stats with min/max.")

        anchor_state = state[:, -1] if state.ndim >= 3 else state
        rel_mas = absolute_to_chunk_relative(mas_abs, anchor_state)
        rel_mas = _normalize_with_action_stats(rel_mas, self._tensor_action_stats, eps=self.eps)
        mas_mask = mas_mask.to(device=rel_mas.device, dtype=rel_mas.dtype)
        progress = progress.to(device=rel_mas.device, dtype=rel_mas.dtype)
        if progress.ndim == rel_mas.ndim - 1:
            progress = progress.unsqueeze(-1)

        long_len = self.mas_long_backward_length + self.mas_long_forward_length
        long_window = _slice_window(rel_mas, 0, long_len)
        long_mask = _slice_window(mas_mask, 0, long_len)
        long_progress = _slice_window(progress, 0, long_len)
        short_window = _slice_window(rel_mas, self.mas_long_backward_length, self.mas_short_window_horizon)
        short_mask = _slice_window(mas_mask, self.mas_long_backward_length, self.mas_short_window_horizon)
        short_progress = _slice_window(progress, self.mas_long_backward_length, self.mas_short_window_horizon)

        new_observation[MAM_LONG_WINDOW] = torch.cat((long_window * long_mask, long_progress), dim=-1)
        new_observation[MAM_LONG_WINDOW_MASK] = torch.cat((long_mask, long_progress), dim=-1)
        new_observation[MAM_SHORT_WINDOW] = torch.cat((short_window * short_mask, short_progress), dim=-1)
        new_observation[MAM_SHORT_WINDOW_MASK] = torch.cat((short_mask, short_progress), dim=-1)
        if MAM_MAS_ACTION_MASK in new_observation:
            new_observation[MAM_ACTION_MASK] = (
                mas_mask[:, : action.shape[1]] if action is not None else mas_mask
            )

        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def get_config(self) -> dict[str, object]:
        return {
            "mas_long_backward_length": self.mas_long_backward_length,
            "mas_long_forward_length": self.mas_long_forward_length,
            "mas_short_window_horizon": self.mas_short_window_horizon,
            "eps": self.eps,
        }

    def state_dict(self) -> dict[str, Tensor]:
        if self._tensor_action_stats is None:
            return {}
        return {f"action.{key}": value for key, value in self._tensor_action_stats.items()}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        loaded = {
            key.removeprefix("action."): value.to(dtype=torch.float32)
            for key, value in state.items()
            if key.startswith("action.")
        }
        self._tensor_action_stats = loaded or None
        if loaded:
            self.action_stats = loaded

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_mam_pre_post_processors(
    config: MamConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    action_stats = None if dataset_stats is None else dataset_stats.get(ACTION)
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        MamFeaturesProcessorStep(
            action_stats=action_stats,
            mas_long_backward_length=config.mas_long_backward_length,
            mas_long_forward_length=config.mas_long_forward_length,
            mas_short_window_horizon=config.mas_short_window_horizon,
        ),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
            to_transition=mam_batch_to_transition,
            to_output=transition_to_batch,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
