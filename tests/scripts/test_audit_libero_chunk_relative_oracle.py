from types import SimpleNamespace

import numpy as np

from lerobot.processor.libero_relative_action_processor import (
    eef_body_quaternion_to_controller_matrix,
    matrix_to_axis_angle,
)
from scripts.audit_libero_chunk_relative_oracle import EpisodeSpec, _run_oracle, parse_args


class _ClosedLoopOracleEnv:
    def __init__(self) -> None:
        self.step_index = 0
        self.position = np.zeros(3, dtype=np.float32)

    def _observation(self) -> dict:
        return {
            "robot_state": {
                "eef": {
                    "pos": self.position.copy(),
                    "quat": np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32),
                },
                "joints": {"pos": np.zeros(7, dtype=np.float32)},
            }
        }

    def reset(self, seed=None):
        self.step_index = 0
        self.position = np.asarray((0.4, -0.2, 0.7), dtype=np.float32)
        return self._observation(), {}

    def step(self, action):
        self.position = np.asarray(action[:3], dtype=np.float32).copy()
        self.step_index += 1
        return self._observation(), 0.0, self.step_index == 4, False, {
            "is_success": self.step_index == 4
        }


def test_oracle_cli_defaults_to_strict_no_post_hold(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["audit_libero_chunk_relative_oracle.py"])

    assert parse_args().post_hold_steps == 0


def test_oracle_replays_closed_loop_materialized_episode_for_multiple_chunk_sizes() -> None:
    states = np.zeros((4, 14), dtype=np.float32)
    states[:, :3] = np.asarray(
        [[0.4, -0.2, 0.7], [0.5, -0.2, 0.7], [0.6, -0.2, 0.7], [0.7, -0.2, 0.7]],
        dtype=np.float32,
    )
    states[:, 6] = 1.0
    actions = np.zeros((4, 7), dtype=np.float32)
    actions[:, :3] = np.asarray(
        [[0.5, -0.2, 0.7], [0.6, -0.2, 0.7], [0.7, -0.2, 0.7], [0.8, -0.2, 0.7]],
        dtype=np.float32,
    )
    controller_matrix = eef_body_quaternion_to_controller_matrix(states[0, 3:7])
    actions[:, 3:6] = matrix_to_axis_angle(controller_matrix)
    actions[:, 6] = np.asarray((1.0, -1.0, 1.0, -1.0), dtype=np.float32)
    spec = EpisodeSpec(
        episode_index=5,
        suite="libero_10",
        task_id=0,
        init_state=[0.0],
        source_episode_id=10,
        source_file="demo.hdf5",
        source_demo="demo_0",
    )
    args = SimpleNamespace(seed=1000, post_hold_steps=0)
    env = _ClosedLoopOracleEnv()

    for chunk_size in (1, 2, 4):
        result = _run_oracle(env, spec, actions, states, chunk_size, args)

        assert result.success is True
        assert result.success_step == 3
        assert result.max_anchor_position_error_m < 1e-7
        assert result.max_anchor_rotation_error_rad < 1e-7
        assert result.max_goal_position_error_m < 1e-7
        assert result.max_goal_rotation_error_rad < 1e-7
