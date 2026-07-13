import pytest
import torch

from lerobot.stpm import RewardTransformer


def _make_model() -> RewardTransformer:
    torch.manual_seed(0)
    model = RewardTransformer(
        d_model=16,
        vis_emb_dim=8,
        text_emb_dim=6,
        state_dim=5,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        num_cameras=2,
    )
    return model.eval()


def test_reward_transformer_outputs_each_timestep_and_masks_padding():
    model = _make_model()
    images = torch.randn(2, 2, 4, 8)
    text = torch.randn(2, 6)
    state = torch.randn(2, 4, 5)

    progress = model(images, text, state, torch.tensor([3, 4]))

    assert progress.shape == (2, 4)
    assert torch.all((progress >= 0.0) & (progress <= 1.0))
    assert progress[0, 3].item() == 0.0
    assert model.visual_proj.in_features == 8
    assert model.modality_pos.shape == (1, 1, 4, 16)


def test_reward_transformer_does_not_observe_future_timesteps():
    model = _make_model()
    images = torch.randn(1, 2, 4, 8)
    text = torch.randn(1, 6)
    state = torch.randn(1, 4, 5)
    changed_images = images.clone()
    changed_state = state.clone()
    changed_images[:, :, 2:] += 100.0
    changed_state[:, 2:] -= 100.0

    baseline = model(images, text, state, torch.tensor([4]))
    changed = model(changed_images, text, changed_state, torch.tensor([4]))

    torch.testing.assert_close(changed[:, :2], baseline[:, :2], atol=1e-6, rtol=1e-6)


def test_reward_transformer_padding_hides_padded_tokens():
    model = _make_model()
    images = torch.randn(1, 2, 4, 8)
    text = torch.randn(1, 6)
    state = torch.randn(1, 4, 5)
    changed_images = images.clone()
    changed_state = state.clone()
    changed_images[:, :, 2:] = torch.randn_like(changed_images[:, :, 2:]) * 100.0
    changed_state[:, 2:] = torch.randn_like(changed_state[:, 2:]) * 100.0

    baseline = model(images, text, state, torch.tensor([2]))
    changed = model(changed_images, text, changed_state, torch.tensor([2]))

    torch.testing.assert_close(changed, baseline, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(changed[:, 2:], torch.zeros(1, 2))


def test_reward_transformer_rejects_legacy_checkpoint():
    model = _make_model()
    legacy_state_dict = {
        "vis_proj.weight": torch.randn(16, 16),
        "text_proj.weight": torch.randn(16, 6),
    }

    with pytest.raises(RuntimeError, match="legacy STPM checkpoint.*architecture v1"):
        model.load_state_dict(legacy_state_dict)


@pytest.mark.parametrize("lengths", [torch.tensor([0]), torch.tensor([5])])
def test_reward_transformer_validates_lengths(lengths):
    model = _make_model()

    with pytest.raises(ValueError, match="lengths values"):
        model(
            torch.randn(1, 2, 4, 8),
            torch.randn(1, 6),
            torch.randn(1, 4, 5),
            lengths,
        )
