import torch

from lerobot.scripts.lerobot_train_stpm import (
    _endpoint_mse_loss,
    _load_all_states,
    _masked_mse_loss,
    _split_dataset_by_episode,
)
from lerobot.utils.constants import OBS_STATE


class _FakeLeRobotDataset:
    def __init__(self, episode_ids: list[int]):
        self.episode_ids = episode_ids
        self.states = [torch.full((2,), float(episode_id)) for episode_id in episode_ids]

    def select_columns(self, column_name: str):
        if column_name == "episode_index":
            return {column_name: self.episode_ids}
        if column_name == OBS_STATE:
            return {column_name: self.states}
        raise KeyError(column_name)


class _FakeFrameDataset:
    def __init__(self, episode_ids: list[int], episode_rows: dict[int, dict] | None = None):
        self.dataset = _FakeLeRobotDataset(episode_ids)
        self.episode_rows = episode_rows or {}

    def __len__(self):
        return len(self.dataset.episode_ids)

    def __getitem__(self, index: int):
        return self.dataset.episode_ids[index]


def test_split_dataset_by_episode_has_no_frame_leakage():
    dataset = _FakeFrameDataset([10, 10, 20, 20, 20, 30, 30, 40])

    train, val, train_episodes, val_episodes, _ = _split_dataset_by_episode(dataset, 0.25)
    repeated = _split_dataset_by_episode(dataset, 0.25)

    assert set(train_episodes).isdisjoint(val_episodes)
    assert set(train_episodes) | set(val_episodes) == {10, 20, 30, 40}
    assert set(train.indices).isdisjoint(val.indices)
    assert sorted(train.indices + val.indices) == list(range(len(dataset)))
    assert val_episodes == repeated[3]
    assert all(dataset.dataset.episode_ids[index] in train_episodes for index in train.indices)
    assert all(dataset.dataset.episode_ids[index] in val_episodes for index in val.indices)


def test_state_normalization_uses_only_train_episode_frames():
    dataset = _FakeFrameDataset([1, 1, 2, 2, 3, 3, 4, 4])
    train, _, train_episodes, _, _ = _split_dataset_by_episode(dataset, 0.25)

    states = _load_all_states(dataset, list(train.indices))

    assert set(states[:, 0].tolist()) == {float(episode_id) for episode_id in train_episodes}


def test_split_groups_multi_mask_episodes_by_source_trajectory():
    dataset = _FakeFrameDataset(
        [0, 0, 1, 1, 2, 2, 3, 3],
        episode_rows={
            0: {"libero/task_id": 2, "libero/source_episode_id": 7},
            1: {"libero/task_id": 2, "libero/source_episode_id": 7},
            2: {"libero/task_id": 2, "libero/source_episode_id": 8},
            3: {"libero/task_id": 3, "libero/source_episode_id": 7},
        },
    )

    _, _, train_episodes, val_episodes, identity = _split_dataset_by_episode(dataset, 0.34)

    assert (0 in train_episodes) == (1 in train_episodes)
    assert (0 in val_episodes) == (1 in val_episodes)
    assert identity["source_field"] == "libero/source_episode_id"
    assert identity["task_field"] == "libero/task_id"


def test_masked_mse_loss_ignores_padded_timesteps():
    prediction = torch.tensor([[0.0, 1.0, 100.0], [1.0, 2.0, 3.0]])
    target = torch.tensor([[0.0, 0.0, -100.0], [1.0, 1.0, 1.0]])

    loss = _masked_mse_loss(prediction, target, torch.tensor([2, 3]))

    torch.testing.assert_close(loss, torch.tensor((0.0 + 1.0 + 0.0 + 1.0 + 4.0) / 5.0))


def test_endpoint_mse_loss_uses_last_valid_timestep():
    prediction = torch.tensor([[0.0, 2.0, 100.0], [1.0, 2.0, 4.0]])
    target = torch.tensor([[0.0, 1.0, -100.0], [1.0, 1.0, 2.0]])

    loss = _endpoint_mse_loss(prediction, target, torch.tensor([2, 3]))

    torch.testing.assert_close(loss, torch.tensor((1.0 + 4.0) / 2.0))
