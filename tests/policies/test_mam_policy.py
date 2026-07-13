from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.mam.configuration_mam import MamConfig
from lerobot.policies.mam.eval_mam import MamEvalEpisode, configure_mam_eval_init_state_ids
from lerobot.policies.mam.modeling_mam import (
    MamLongWindowEncoder,
    MamPolicy,
    _compute_mam_diffusion_loss,
)
from lerobot.policies.mam.processor_mam import (
    MAM_ACTION_MASK,
    MAM_LONG_WINDOW,
    MAM_LONG_WINDOW_MASK,
    MAM_MAS_ACTION_ABSOLUTE,
    MAM_MAS_ACTION_MASK,
    MAM_PROGRESS,
    MAM_SHORT_WINDOW,
    MAM_SHORT_WINDOW_MASK,
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
        mas_long_forward_length=16,
        mas_short_window_horizon=8,
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


def test_mam_config_covers_every_observation_window_and_requires_relative_actions():
    cfg = _make_config()

    assert cfg.use_relative_actions is True
    assert cfg.mam_delta_indices == list(range(-1, 16))

    cfg.mas_long_backward_length = 2
    assert cfg.mam_delta_indices == list(range(-3, 16))

    with pytest.raises(ValueError, match="use_relative_actions=True"):
        MamConfig(use_relative_actions=False)


def test_mam_processor_normalizes_relative_actions_before_masking():
    cfg = _make_config()
    preprocessor, _ = make_mam_pre_post_processors(cfg, _make_stats())
    time = torch.linspace(-0.4, 0.4, 17)
    mas_absolute = torch.zeros(1, 17, 7)
    mas_absolute[0, :, 0] = 0.3 + time * 0.1
    mas_absolute[0, :, 1] = -0.2 + time * 0.2
    mas_absolute[0, :, 2] = 0.4 - time * 0.1
    mas_absolute[0, :, 3] = 0.7 + time * 0.2
    mas_absolute[0, :, 4] = -0.5 + time * 0.1
    mas_absolute[0, :, 5] = 0.4 - time * 0.3
    mas_absolute[0, :, 6] = 1.0
    mas_mask = torch.zeros_like(mas_absolute)
    mas_mask[..., 3] = 1.0
    mas_mask[..., 6] = 1.0
    state = torch.tensor([[[-0.6, 0.2, -0.3, -0.5, 0.4, -0.2], [0.2, -0.1, 0.3, 0.4, -0.3, 0.2]]])
    progress = torch.linspace(0, 1, 17).view(1, 17, 1)
    batch = {
        OBS_STATE: state,
        OBS_ENV_STATE: torch.zeros(1, 2, 1),
        ACTION: mas_absolute[:, :16].clone(),
        MAM_MAS_ACTION_ABSOLUTE: mas_absolute,
        MAM_MAS_ACTION_MASK: mas_mask,
        MAM_PROGRESS: progress,
        "action_is_pad": torch.zeros(1, 16, dtype=torch.bool),
    }

    processed = preprocessor(batch)

    relative = absolute_to_chunk_relative(mas_absolute, state[:, -1])
    masked_relative = relative * mas_mask
    expected_long = torch.stack((masked_relative[:, :16], masked_relative[:, 1:17]), dim=1)
    expected_progress = torch.stack((progress[:, :16], progress[:, 1:17]), dim=1)
    torch.testing.assert_close(processed[MAM_LONG_WINDOW][..., :7], expected_long)
    torch.testing.assert_close(processed[MAM_LONG_WINDOW][..., 7:], expected_progress)
    torch.testing.assert_close(processed[ACTION], relative[:, :16])

    premasked_relative = absolute_to_chunk_relative(mas_absolute * mas_mask, state[:, -1])
    assert not torch.allclose(masked_relative, premasked_relative * mas_mask)


def test_mam_long_encoder_preserves_observation_axis_and_pools_time_axis():
    encoder = MamLongWindowEncoder(step_dim=8, out_dim=5)
    pool_input_shapes = []
    handle = encoder.net[2].register_forward_pre_hook(
        lambda _module, inputs: pool_input_shapes.append(tuple(inputs[0].shape))
    )

    output = encoder(torch.randn(2, 3, 16, 8), torch.ones(2, 3, 16, 8))
    handle.remove()

    assert output.shape == (2, 3, 5)
    assert pool_input_shapes == [(6, 16, 8, 16)]
    assert sum(isinstance(module, nn.Conv2d) for module in encoder.net) == 3
    assert sum(isinstance(module, nn.MaxPool2d) for module in encoder.net) == 2


def test_mam_forward_and_action_mask_builds_features():
    torch.manual_seed(0)
    cfg = _make_config()
    preprocessor, _ = make_mam_pre_post_processors(cfg, _make_stats())
    batch = {
        OBS_STATE: torch.zeros(2, 2, 6),
        OBS_ENV_STATE: torch.zeros(2, 2, 1),
        ACTION: torch.zeros(2, 16, 7),
        MAM_MAS_ACTION_ABSOLUTE: torch.zeros(2, 17, 7),
        MAM_MAS_ACTION_MASK: torch.ones(2, 17, 7),
        MAM_PROGRESS: torch.linspace(0, 1, 17).view(1, 17, 1).repeat(2, 1, 1),
        "action_is_pad": torch.zeros(2, 16, dtype=torch.bool),
    }
    processed = preprocessor(batch)

    policy = MamPolicy(cfg)
    loss, _ = policy(processed)
    assert torch.isfinite(loss)
    assert processed[MAM_LONG_WINDOW].shape == (2, 2, 16, 8)
    assert processed[MAM_SHORT_WINDOW].shape == (2, 2, 8, 8)

    zero_mask_batch = dict(processed)
    zero_mask_batch[MAM_ACTION_MASK] = torch.zeros_like(processed[MAM_ACTION_MASK])
    zero_mask_loss = policy.diffusion.compute_loss(zero_mask_batch)
    assert torch.isfinite(zero_mask_loss)


def test_mam_action_mask_aligns_to_action_delta_indices():
    cfg = _make_config()
    preprocessor, _ = make_mam_pre_post_processors(cfg, _make_stats())
    mas_mask = torch.arange(1, 18, dtype=torch.float32).view(1, 17, 1).repeat(1, 1, 7)
    batch = {
        OBS_STATE: torch.zeros(1, 2, 6),
        OBS_ENV_STATE: torch.zeros(1, 2, 1),
        ACTION: torch.zeros(1, 16, 7),
        MAM_MAS_ACTION_ABSOLUTE: torch.zeros(1, 17, 7),
        MAM_MAS_ACTION_MASK: mas_mask,
        MAM_PROGRESS: torch.linspace(0, 1, 17).view(1, 17, 1),
        "action_is_pad": torch.zeros(1, 16, dtype=torch.bool),
    }

    processed = preprocessor(batch)

    assert processed[MAM_ACTION_MASK][0, :, 0].tolist() == [float(i) for i in range(1, 17)]


def test_mam_padding_mask_requires_action_is_pad():
    cfg = _make_config()
    cfg.do_mask_loss_for_padding = True
    policy = MamPolicy(cfg)
    batch = {
        OBS_STATE: torch.zeros(1, 2, 6),
        OBS_ENV_STATE: torch.zeros(1, 2, 1),
        ACTION: torch.zeros(1, 16, 7),
        MAM_ACTION_MASK: torch.zeros(1, 16, 7),
        MAM_LONG_WINDOW: torch.zeros(1, 2, 16, 8),
        MAM_LONG_WINDOW_MASK: torch.zeros(1, 2, 16, 8),
        MAM_SHORT_WINDOW: torch.zeros(1, 2, 8, 8),
    }

    with pytest.raises(ValueError, match="action_is_pad"):
        policy.diffusion.compute_loss(batch)


def test_mam_region_balanced_loss_excludes_padding_per_action_dimension():
    per_element_loss = torch.tensor([[[1.0, 3.0], [100.0, 200.0]]])
    action_mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    valid_mask = torch.tensor([[[True], [False]]])

    weighted = _compute_mam_diffusion_loss(
        per_element_loss,
        action_mask=action_mask,
        valid_mask=valid_mask,
        loss_mode="weighted",
        known_region_weight=0.2,
    )
    average = _compute_mam_diffusion_loss(
        per_element_loss,
        action_mask=None,
        valid_mask=valid_mask,
        loss_mode="average",
        known_region_weight=0.2,
    )

    torch.testing.assert_close(weighted, torch.tensor(2.6))
    torch.testing.assert_close(average, torch.tensor(2.0))


@pytest.mark.parametrize("known", [False, True])
def test_mam_region_balanced_loss_handles_single_present_region(known: bool):
    per_element_loss = torch.tensor([[[1.0, 3.0]]])
    action_mask = torch.full_like(per_element_loss, float(known))

    loss = _compute_mam_diffusion_loss(
        per_element_loss,
        action_mask=action_mask,
        valid_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        loss_mode="weighted",
        known_region_weight=0.2,
    )

    torch.testing.assert_close(loss, torch.tensor(2.0))


def test_mam_region_balanced_loss_returns_zero_when_every_action_is_padding():
    per_element_loss = torch.tensor([[[1.0, 3.0]]], requires_grad=True)

    loss = _compute_mam_diffusion_loss(
        per_element_loss,
        action_mask=torch.ones_like(per_element_loss),
        valid_mask=torch.zeros(1, 1, 1, dtype=torch.bool),
        loss_mode="weighted",
        known_region_weight=0.2,
    )
    loss.backward()

    assert loss.item() == 0.0
    torch.testing.assert_close(per_element_loss.grad, torch.zeros_like(per_element_loss))


class _ZeroDenoiser(nn.Module):
    def forward(self, sample, timestep, global_cond=None):
        del timestep, global_cond
        return torch.zeros_like(sample)


def test_mam_inpainting_overwrites_known_region_at_every_reverse_step(monkeypatch):
    cfg = _make_config()
    cfg.inpainting = True
    policy = MamPolicy(cfg)
    model = policy.diffusion
    model.unet = _ZeroDenoiser()
    model.num_inference_steps = 4
    known = torch.linspace(-0.8, 0.8, 16 * 7).reshape(1, 16, 7)
    mask = torch.zeros_like(known)
    mask[:, 1:9, 0::2] = 1.0
    add_noise_timesteps = []
    original_add_noise = model.noise_scheduler.add_noise

    def tracked_add_noise(original_samples, noise, timesteps):
        add_noise_timesteps.append(timesteps.detach().cpu().clone())
        return original_add_noise(original_samples, noise, timesteps)

    monkeypatch.setattr(model.noise_scheduler, "add_noise", tracked_add_noise)
    sample = model.conditional_sample(
        batch_size=1,
        global_cond=torch.zeros(1, 1),
        noise=torch.zeros_like(known),
        action_known_0=known,
        action_mask=mask,
    )

    assert len(add_noise_timesteps) == model.num_inference_steps - 1
    torch.testing.assert_close(sample[mask.bool()], known[mask.bool()])


def test_mam_predict_action_chunk_forwards_short_window_mask(monkeypatch):
    cfg = _make_config()
    cfg.inpainting = True
    policy = MamPolicy(cfg)
    short_mask = torch.zeros(1, 2, 8, 8)
    short_mask[:, -1, :, :7] = 1.0
    batch = {
        OBS_STATE: torch.zeros(1, 6),
        OBS_ENV_STATE: torch.zeros(1, 1),
        MAM_LONG_WINDOW: torch.zeros(1, 2, 16, 8),
        MAM_LONG_WINDOW_MASK: torch.zeros(1, 2, 16, 8),
        MAM_SHORT_WINDOW: torch.zeros(1, 2, 8, 8),
        MAM_SHORT_WINDOW_MASK: short_mask,
    }
    batch = policy.update_observation_queue(batch)
    captured_batch = {}

    def fake_generate_actions(model_batch, noise=None):
        del noise
        captured_batch.update(model_batch)
        return torch.zeros(1, cfg.n_action_steps, 7)

    monkeypatch.setattr(policy.diffusion, "generate_actions", fake_generate_actions)
    action_chunk = policy.predict_action_chunk(batch)

    assert action_chunk.shape == (1, cfg.n_action_steps, 7)
    torch.testing.assert_close(captured_batch[MAM_SHORT_WINDOW_MASK], short_mask)


def test_mam_inpainting_uses_latest_short_window_at_action_chunk_offset():
    cfg = _make_config()
    cfg.inpainting = True
    model = MamPolicy(cfg).diffusion
    short_window = torch.zeros(1, 2, 8, 8)
    short_mask = torch.zeros_like(short_window)
    short_window[:, 0, :, :7] = -0.5
    short_window[:, 1, :, :7] = 0.25
    short_mask[:, 1, :, :7] = 1.0

    known, mask = model._build_inpainting_inputs(
        {
            MAM_SHORT_WINDOW: short_window,
            MAM_SHORT_WINDOW_MASK: short_mask,
        }
    )

    start = cfg.n_obs_steps - 1
    torch.testing.assert_close(known[:, start : start + 8], torch.full((1, 8, 7), 0.25))
    assert torch.count_nonzero(mask[:, start : start + 8]) == 8 * 7
    assert torch.count_nonzero(mask[:, :start]) == 0


def test_configure_mam_eval_prefers_raw_init_state_values():
    cfg = SimpleNamespace(
        env=SimpleNamespace(
            type="libero",
            task="libero_10",
            task_ids=None,
            init_state_ids=None,
            init_state_ids_by_task=None,
            init_state_values=None,
            init_state_values_by_task=None,
            num_steps_wait=10,
        ),
        eval=SimpleNamespace(batch_size=2),
    )
    episodes = [
        MamEvalEpisode(
            episode_index=0,
            init_state_id=3,
            init_state=[0.1, 0.2],
            suite="libero_10",
            task_id=0,
            mask_type="random_mask",
            mask_type_slot=0,
            task="task 0",
            mas_action_absolute=torch.zeros(2, 7),
            mas_action_mask=torch.zeros(2, 7),
            progress=torch.zeros(2, 1),
        ),
        MamEvalEpisode(
            episode_index=1,
            init_state_id=4,
            init_state=[0.3, 0.4],
            suite="libero_10",
            task_id=1,
            mask_type="random_mask",
            mask_type_slot=0,
            task="task 1",
            mas_action_absolute=torch.zeros(2, 7),
            mas_action_mask=torch.zeros(2, 7),
            progress=torch.zeros(2, 1),
        ),
    ]

    configure_mam_eval_init_state_ids(cfg, episodes, n_episodes=2)

    assert cfg.eval.batch_size == 1
    assert cfg.env.task_ids == [0, 1]
    assert cfg.env.init_state_ids_by_task is None
    assert cfg.env.init_state_values_by_task == {"libero_10/0": [[0.1, 0.2]], "libero_10/1": [[0.3, 0.4]]}
    assert cfg.env.num_steps_wait == 0


def test_configure_mam_eval_falls_back_to_ids_when_raw_init_state_missing():
    cfg = SimpleNamespace(
        env=SimpleNamespace(
            type="libero",
            task="libero_10",
            task_ids=None,
            init_state_ids=[99],
            init_state_ids_by_task=None,
            init_state_values=[[9.9]],
            init_state_values_by_task={"old/0": [[9.9]]},
            num_steps_wait=10,
        ),
        eval=SimpleNamespace(batch_size=2),
    )
    episodes = [
        MamEvalEpisode(
            episode_index=0,
            init_state_id=3,
            init_state=[0.1, 0.2],
            suite="libero_10",
            task_id=0,
            mask_type="random_mask",
            mask_type_slot=0,
            task="task 0",
            mas_action_absolute=torch.zeros(2, 7),
            mas_action_mask=torch.zeros(2, 7),
            progress=torch.zeros(2, 1),
        ),
        MamEvalEpisode(
            episode_index=1,
            init_state_id=4,
            init_state=None,
            suite="libero_10",
            task_id=1,
            mask_type="random_mask",
            mask_type_slot=0,
            task="task 1",
            mas_action_absolute=torch.zeros(2, 7),
            mas_action_mask=torch.zeros(2, 7),
            progress=torch.zeros(2, 1),
        ),
    ]

    configure_mam_eval_init_state_ids(cfg, episodes, n_episodes=2)

    assert cfg.eval.batch_size == 1
    assert cfg.env.init_state_ids_by_task == {"libero_10/0": [3], "libero_10/1": [4]}
    assert cfg.env.init_state_values_by_task is None
    assert cfg.env.init_state_values is None
    assert cfg.env.init_state_ids is None
    assert cfg.env.num_steps_wait == 10


def test_configure_mam_eval_rejects_mixed_task_metadata():
    cfg = SimpleNamespace(
        env=SimpleNamespace(type="libero", init_state_ids=None),
        eval=SimpleNamespace(batch_size=1),
    )
    episodes = [
        MamEvalEpisode(
            episode_index=0,
            init_state_id=3,
            init_state=None,
            suite="libero_10",
            task_id=0,
            mask_type="random_mask",
            mask_type_slot=0,
            task="task 0",
            mas_action_absolute=torch.zeros(2, 7),
            mas_action_mask=torch.zeros(2, 7),
            progress=torch.zeros(2, 1),
        ),
        MamEvalEpisode(
            episode_index=1,
            init_state_id=4,
            init_state=None,
            suite="",
            task_id=None,
            mask_type="random_mask",
            mask_type_slot=0,
            task="task 1",
            mas_action_absolute=torch.zeros(2, 7),
            mas_action_mask=torch.zeros(2, 7),
            progress=torch.zeros(2, 1),
        ),
    ]

    with pytest.raises(ValueError, match="task_id"):
        configure_mam_eval_init_state_ids(cfg, episodes, n_episodes=2)
