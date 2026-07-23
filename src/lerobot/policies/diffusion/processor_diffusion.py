#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
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
from typing import Any

import torch

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    LiberoChunkRelativeActionsPostprocessorStep,
    LiberoChunkRelativeActionsProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    ensure_libero_chunk_relative_actions_postprocessor,
    policy_action_with_context_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_diffusion import DiffusionConfig


def make_diffusion_pre_post_processors(
    config: DiffusionConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction | dict[str, Any], PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for a diffusion policy.

    The pre-processing pipeline prepares the input data for the model by:
    1. Renaming features.
    2. Normalizing the input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Moving the data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Unnormalizing the output features to their original scale.
    2. Decoding chunk-relative actions and selecting the executable window.
    3. Moving the result to the CPU.

    Args:
        config: The configuration object for the diffusion policy,
            containing feature definitions, normalization mappings, and device information.
        dataset_stats: A dictionary of statistics used for normalization.
            Defaults to None.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        LiberoChunkRelativeActionsProcessorStep(enabled=config.use_relative_actions),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        LiberoChunkRelativeActionsPostprocessorStep(
            enabled=config.use_relative_actions,
            n_obs_steps=config.n_obs_steps,
            n_action_steps=config.n_action_steps,
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction | dict[str, Any], PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_with_context_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def upgrade_loaded_diffusion_postprocessor(
    config: DiffusionConfig,
    postprocessor: PolicyProcessorPipeline[PolicyAction | dict[str, Any], PolicyAction],
) -> None:
    """Upgrade a legacy serialized DP postprocessor to the current contract."""
    ensure_libero_chunk_relative_actions_postprocessor(
        postprocessor,
        enabled=config.use_relative_actions,
        n_obs_steps=config.n_obs_steps,
        n_action_steps=config.n_action_steps,
    )
