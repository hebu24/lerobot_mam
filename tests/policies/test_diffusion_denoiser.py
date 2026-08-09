import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    DiffusionPolicy,
    DiffusionTransformerDenoiser,
    make_diffusion_denoiser,
)
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import DiffusionTransformer
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


def _make_config(denoiser_type: str) -> DiffusionConfig:
    return DiffusionConfig(
        device="cpu",
        input_features={
            OBS_STATE: PolicyFeature(FeatureType.STATE, (6,)),
            OBS_ENV_STATE: PolicyFeature(FeatureType.ENV, (1,)),
        },
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (7,))},
        denoiser_type=denoiser_type,
        horizon=16,
        down_dims=(32, 64),
        kernel_size=3,
        n_groups=8,
        diffusion_step_embed_dim=16,
        dit_hidden_dim=32,
        dit_num_layers=2,
        dit_num_heads=4,
        dit_dropout=0.0,
        dit_timestep_embed_dim=16,
        num_train_timesteps=4,
    )


@pytest.mark.parametrize(
    ("denoiser_type", "expected_type"),
    [("unet", DiffusionConditionalUnet1d), ("dit", DiffusionTransformerDenoiser)],
)
def test_diffusion_denoiser_shape_and_backward(denoiser_type: str, expected_type: type[torch.nn.Module]):
    config = _make_config(denoiser_type)
    denoiser = make_diffusion_denoiser(config, global_cond_dim=14)
    noisy_actions = torch.randn(2, config.horizon, config.action_feature.shape[0], requires_grad=True)

    prediction = denoiser(
        noisy_actions,
        torch.tensor([1, 2]),
        global_cond=torch.randn(2, 14),
    )
    prediction.square().mean().backward()

    assert isinstance(denoiser, expected_type)
    assert prediction.shape == noisy_actions.shape
    assert noisy_actions.grad is not None
    if denoiser_type == "dit":
        assert isinstance(denoiser.model, DiffusionTransformer)


def test_diffusion_policy_uses_dit_for_training_loss():
    config = _make_config("dit")
    policy = DiffusionPolicy(config)
    loss, output_dict = policy(
        {
            OBS_STATE: torch.randn(2, config.n_obs_steps, 6),
            OBS_ENV_STATE: torch.randn(2, config.n_obs_steps, 1),
            ACTION: torch.randn(2, config.horizon, 7),
            "action_is_pad": torch.zeros(2, config.horizon, dtype=torch.bool),
        }
    )
    loss.backward()

    assert isinstance(policy.diffusion.unet, DiffusionTransformerDenoiser)
    assert torch.isfinite(loss)
    assert output_dict is None


def test_diffusion_dit_save_and_load(tmp_path):
    policy = DiffusionPolicy(_make_config("dit"))
    policy.save_pretrained(tmp_path)

    loaded_policy = DiffusionPolicy.from_pretrained(tmp_path)

    assert loaded_policy.config.denoiser_type == "dit"
    assert isinstance(loaded_policy.diffusion.unet, DiffusionTransformerDenoiser)
    for expected, actual in zip(policy.parameters(), loaded_policy.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_dit_does_not_require_unet_horizon_divisibility():
    config = DiffusionConfig(denoiser_type="dit", horizon=15)
    assert config.horizon == 15

    with pytest.raises(ValueError, match="integer multiple"):
        DiffusionConfig(denoiser_type="unet", horizon=15)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"denoiser_type": "invalid"}, "denoiser_type"),
        ({"denoiser_type": "dit", "dit_hidden_dim": 30, "dit_num_heads": 8}, "divisible"),
        ({"denoiser_type": "dit", "dit_dropout": 1.1}, "dit_dropout"),
        ({"denoiser_type": "dit", "dit_timestep_embed_dim": 3}, "dit_timestep_embed_dim"),
    ],
)
def test_diffusion_denoiser_config_validation(kwargs: dict, message: str):
    with pytest.raises(ValueError, match=message):
        DiffusionConfig(**kwargs)
