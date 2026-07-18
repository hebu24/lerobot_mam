from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from lerobot.utils.constants import ACTION, OBS_STATE
from scripts.convert_libero_delta_to_absolute import (
    AGENTVIEW_IMAGE,
    WRIST_IMAGE,
    _apply_materialized_values,
    _compute_materialized_episode_stats,
    _record_closed_loop_episode,
    _validate_closed_loop_observation_schema,
    _validate_episode_frame_layout,
)


def _observation(step: int) -> dict:
    return {
        "robot_state": {
            "eef": {
                "pos": np.asarray([step, step + 0.1, step + 0.2], dtype=np.float64),
                "quat": np.asarray([0.0, 0.0, 0.0, 2.0], dtype=np.float64),
            },
            "joints": {"pos": np.arange(7, dtype=np.float64) + step},
        },
        "pixels": {
            "image": np.full((2, 3, 3), step, dtype=np.uint8),
            "image2": np.full((2, 3, 3), step + 10, dtype=np.uint8),
        },
    }


class _FakeRuntimeCore:
    def __init__(self, *, success_step: int | None):
        self.step_index = 0
        self.success_step = success_step
        self.actions: list[np.ndarray] = []

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self.step_index += 1
        # Deliberately report done on early success. The materializer must not
        # auto-reset or shorten the episode.
        done = self.success_step is not None and self.step_index == self.success_step
        return _observation(self.step_index), 0.0, done, {}

    def check_success(self):
        return self.success_step is not None and self.step_index >= self.success_step


class _FakeRuntimeEnv:
    def __init__(self, *, success_step: int | None):
        self._env = _FakeRuntimeCore(success_step=success_step)
        self.seed = None

    def reset(self, seed=None):
        self.seed = seed
        self._env.step_index = 0
        return _observation(0), {}

    def _format_raw_obs(self, raw_observation):
        return raw_observation


def _episode_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            ACTION: [np.zeros(7, dtype=np.float32) for _ in range(3)],
            OBS_STATE: [np.full(14, -1, dtype=np.float32) for _ in range(3)],
            AGENTVIEW_IMAGE: [np.full((2, 3, 3), 255, dtype=np.uint8) for _ in range(3)],
            WRIST_IMAGE: [np.full((2, 3, 3), 255, dtype=np.uint8) for _ in range(3)],
            "frame_index": [0, 1, 2],
            "episode_index": [4, 4, 4],
            "index": [90, 91, 92],
            "task_index": [7, 7, 7],
        },
        index=[10, 11, 12],
    )


def _absolute_actions() -> dict[int, np.ndarray]:
    return {index: np.full(7, index, dtype=np.float32) for index in (10, 11, 12)}


def test_closed_loop_replay_rematerializes_every_frame_after_early_success():
    source = _episode_df()
    ordered = _validate_episode_frame_layout(source, expected_length=3)
    env = _FakeRuntimeEnv(success_step=1)

    observations, success_step = _record_closed_loop_episode(
        ordered,
        absolute_actions=_absolute_actions(),
        runtime_env=env,
        seed=1000,
        observation_height=2,
        observation_width=3,
    )
    converted = _apply_materialized_values(
        source,
        absolute_actions=_absolute_actions(),
        observations=observations,
        target_indices={10, 11, 12},
    )

    assert success_step == 0
    assert env.seed == 1000
    assert len(env._env.actions) == len(source)
    assert [float(value[0]) for value in converted[OBS_STATE]] == [0.0, 1.0, 2.0]
    assert [int(value[0, 0, 0]) for value in converted[AGENTVIEW_IMAGE]] == [0, 1, 2]
    assert [float(value[0]) for value in converted[ACTION]] == [10.0, 11.0, 12.0]
    pd.testing.assert_frame_equal(
        converted[["frame_index", "episode_index", "index", "task_index"]],
        source[["frame_index", "episode_index", "index", "task_index"]],
    )


def test_closed_loop_replay_rejects_unsuccessful_trajectory():
    with pytest.raises(RuntimeError, match="did not succeed"):
        _record_closed_loop_episode(
            _episode_df(),
            absolute_actions=_absolute_actions(),
            runtime_env=_FakeRuntimeEnv(success_step=None),
            seed=1000,
            observation_height=2,
            observation_width=3,
        )


def test_closed_loop_replay_requires_complete_episode_layout():
    partial = _episode_df().iloc[[0, 2]]

    with pytest.raises(ValueError, match="wholly contained"):
        _validate_episode_frame_layout(partial, expected_length=3)


def _meta(*, video_keys=(), extra_features=()):
    features = {
        OBS_STATE: {"dtype": "float32", "shape": (14,)},
        AGENTVIEW_IMAGE: {"dtype": "image", "shape": (2, 3, 3)},
        WRIST_IMAGE: {"dtype": "image", "shape": (2, 3, 3)},
        **{key: {"dtype": "float32", "shape": (1,)} for key in extra_features},
    }
    return SimpleNamespace(features=features, video_keys=list(video_keys))


def test_closed_loop_schema_rejects_video_and_unknown_observation_fields():
    _validate_closed_loop_observation_schema(_meta())

    with pytest.raises(ValueError, match="video-backed"):
        _validate_closed_loop_observation_schema(_meta(video_keys=[AGENTVIEW_IMAGE]))
    with pytest.raises(ValueError, match="unsupported"):
        _validate_closed_loop_observation_schema(
            _meta(extra_features=["observation.environment_state"])
        )


def _embedded_png(value: int) -> dict[str, bytes | None]:
    buffer = BytesIO()
    Image.fromarray(np.full((2, 3, 3), value, dtype=np.uint8)).save(buffer, format="PNG")
    return {"bytes": buffer.getvalue(), "path": None}


def test_materialized_episode_stats_are_recomputed_from_numeric_and_embedded_images():
    episode = _episode_df().iloc[:2].copy()
    episode[ACTION] = [np.zeros(7, dtype=np.float32), np.full(7, 2.0, dtype=np.float32)]
    episode[OBS_STATE] = [np.zeros(14, dtype=np.float32), np.full(14, 4.0, dtype=np.float32)]
    episode[AGENTVIEW_IMAGE] = [_embedded_png(0), _embedded_png(255)]
    episode[WRIST_IMAGE] = [_embedded_png(64), _embedded_png(128)]
    features = {
        ACTION: {"dtype": "float32", "shape": (7,)},
        OBS_STATE: {"dtype": "float32", "shape": (14,)},
        AGENTVIEW_IMAGE: {"dtype": "image", "shape": (2, 3, 3)},
        WRIST_IMAGE: {"dtype": "image", "shape": (2, 3, 3)},
        "frame_index": {"dtype": "int64", "shape": (1,)},
        "episode_index": {"dtype": "int64", "shape": (1,)},
        "index": {"dtype": "int64", "shape": (1,)},
        "task_index": {"dtype": "int64", "shape": (1,)},
    }

    stats = _compute_materialized_episode_stats(episode, features)

    np.testing.assert_allclose(stats[ACTION]["mean"], np.ones(7), atol=1e-6)
    np.testing.assert_allclose(stats[OBS_STATE]["mean"], np.full(14, 2.0), atol=1e-6)
    np.testing.assert_allclose(stats[AGENTVIEW_IMAGE]["mean"], 0.5, atol=1e-6)
    np.testing.assert_allclose(stats[WRIST_IMAGE]["mean"], 96.0 / 255.0, atol=1e-6)
