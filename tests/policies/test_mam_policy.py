import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.mam.configuration_mam import MamConfig
from lerobot.policies.mam.modeling_mam import MamPolicy
from lerobot.policies.mam.processor_mam import (
    MAM_ACTION_MASK,
    MAM_MAS_ACTION_ABSOLUTE,
    MAM_MAS_ACTION_MASK,
    MAM_PROGRESS,
    make_mam_pre_post_processors,
)
from lerobot.processor.libero_relative_action_processor import (
    absolute_to_chunk_relative,
    chunk_relative_to_absolute,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


def _make_config() -> MamConfig:
    return MamConfig(
        device="cpu",
        input_features={
            OBS_STATE: PolicyFeature(FeatureType.STATE, (6,)),
            OBS_ENV_STATE: PolicyFeature(FeatureType.ENV, (1,)),
        },
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (7,))},
        horizon=16,
        n_action_steps=8,
        n_obs_steps=2,
        down_dims=(32, 64),
        diffusion_step_embed_dim=16,
        n_groups=8,
        num_train_timesteps=4,
        mas_long_feature_dim=8,
    )


def _make_stats() -> dict:
    return {
        ACTION: {"min": torch.full((7,), -1.0), "max": torch.full((7,), 1.0)},
        OBS_STATE: {"min": torch.full((6,), -1.0), "max": torch.full((6,), 1.0)},
        OBS_ENV_STATE: {"min": torch.full((1,), -1.0), "max": torch.full((1,), 1.0)},
    }


def test_libero_chunk_relative_roundtrip():
    anchor = torch.randn(3, 6, dtype=torch.float64) * 0.01
    actions = torch.randn(3, 16, 7, dtype=torch.float64) * 0.01
    actions[..., 6] = torch.tanh(actions[..., 6])

    relative = absolute_to_chunk_relative(actions, anchor)
    reconstructed = chunk_relative_to_absolute(relative, anchor)

    assert torch.max(torch.abs(reconstructed - actions)) < 1e-5


def test_mam_forward_and_action_mask_affects_loss():
    torch.manual_seed(0)
    cfg = _make_config()
    preprocessor, _ = make_mam_pre_post_processors(cfg, _make_stats())
    batch = {
        OBS_STATE: torch.zeros(2, 2, 6),
        OBS_ENV_STATE: torch.zeros(2, 2, 1),
        ACTION: torch.zeros(2, 16, 7),
        MAM_MAS_ACTION_ABSOLUTE: torch.zeros(2, 16, 7),
        MAM_MAS_ACTION_MASK: torch.ones(2, 16, 7),
        MAM_PROGRESS: torch.linspace(0, 1, 16).view(1, 16, 1).repeat(2, 1, 1),
        "action_is_pad": torch.zeros(2, 16, dtype=torch.bool),
    }
    processed = preprocessor(batch)

    policy = MamPolicy(cfg)
    loss, _ = policy(processed)
    assert torch.isfinite(loss)
    assert processed["mam.mas_long_window"].shape == (2, 16, 8)
    assert processed["mam.mas_short_window"].shape == (2, 8, 8)

    zero_mask_batch = dict(processed)
    zero_mask_batch[MAM_ACTION_MASK] = torch.zeros_like(processed[MAM_ACTION_MASK])
    zero_mask_loss = policy.diffusion.compute_loss(zero_mask_batch)
    assert zero_mask_loss.item() == 0.0
