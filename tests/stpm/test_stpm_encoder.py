import json

import torch

from lerobot.stpm import FrameLeRobotDataset, RewardTransformer, STPMEncoder
from lerobot.stpm.normalizer import save_state_norm
from lerobot.utils.constants import OBS_STATE


def test_frame_dataset_uses_episode_index_metadata_lookup():
    dataset = FrameLeRobotDataset.__new__(FrameLeRobotDataset)
    dataset.camera_keys = ["observation.images.image"]
    dataset.relative_indices = [-1, 0]
    dataset.sequence_length = 2
    dataset.episode_rows = {5: {"episode_index": 5, "length": 10}}

    class FakeDataset:
        def __getitem__(self, index):
            return {
                OBS_STATE: torch.zeros(2, 8),
                "observation.images.image": torch.zeros(2, 3, 16, 16),
                "episode_index": torch.tensor(5),
                "frame_index": torch.tensor(7),
                "task": "put bowl on plate",
            }

    dataset.dataset = FakeDataset()

    item = dataset[0]

    assert item["episode_index"] == 5
    assert torch.allclose(item["targets"], torch.tensor([6 / 9, 7 / 9], dtype=torch.float32))


def test_stpm_encoder_predicts_batched_progress(tmp_path):
    state_dim = 8
    output_dir = tmp_path / "stpm"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    save_state_norm(output_dir / "state_norm.json", torch.randn(4, 3, state_dim), meta={})
    model = RewardTransformer(
        d_model=16,
        state_dim=state_dim,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        num_cameras=2,
    )
    torch.save({"model": model.state_dict()}, checkpoint_dir / "reward_best.pt")
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        json.dump(
            {
                "device": "cpu",
                "camera_names": ["observation.images.image", "observation.images.image2"],
                "image_shape": [[3, 16, 16], [3, 16, 16]],
                "state_dim": state_dim,
                "n_obs_steps": 2,
                "frame_gap": 1,
                "task_description": "put bowl on plate",
                "state_norm_path": str(output_dir / "state_norm.json"),
                "d_model": 16,
                "n_layers": 1,
                "n_heads": 4,
                "dropout": 0.0,
                "vision_ckpt": "",
            },
            f,
        )

    encoder = STPMEncoder(checkpoint_dir / "reward_best.pt", output_dir / "config.yaml", device="cpu")
    progress = encoder.predict_progress(
        torch.rand(2, 3, 2, 3, 16, 16),
        torch.randn(2, 3, state_dim),
        ["put bowl on plate", "put bowl on plate"],
    )

    assert progress.shape == (2,)
    assert torch.all((progress >= 0) & (progress <= 1))
