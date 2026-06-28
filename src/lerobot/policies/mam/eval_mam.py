from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor
from tqdm import trange

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.envs import preprocess_observation
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.libero_relative_action_processor import chunk_relative_to_absolute
from lerobot.scripts.lerobot_eval import _compile_episode_data
from lerobot.stpm import STPMEncoder
from lerobot.types import PolicyAction
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE, OBS_STR
from lerobot.utils.utils import inside_slurm

from .configuration_mam import MamConfig
from .processor_mam import (
    MAM_MAS_ACTION_ABSOLUTE,
    MAM_MAS_ACTION_MASK,
    MAM_PROGRESS,
    MAM_SHORT_WINDOW,
    MAM_SHORT_WINDOW_MASK,
)


LIBERO_INIT_STATE_ID_KEYS = ("libero/init_state_id", "init_state_id")


@dataclass
class MamEvalEpisode:
    episode_index: int
    init_state_id: int
    mask_type: str
    mask_type_slot: int
    task: str
    mas_action_absolute: Tensor
    mas_action_mask: Tensor
    progress: Tensor


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (KeyError, TypeError):
        return default


def _episode_rows(meta: LeRobotDatasetMetadata) -> dict[int, Any]:
    return {int(row["episode_index"]): row for row in meta.episodes}


def _column_names(obj: Any) -> set[str]:
    return set(getattr(obj, "column_names", []) or [])


def _stack_float32(values: list[Any], *, keep_feature_dim: bool = False) -> torch.Tensor:
    stacked = torch.stack([torch.as_tensor(value, dtype=torch.float32) for value in values], dim=0)
    if keep_feature_dim and stacked.ndim == 1:
        stacked = stacked.unsqueeze(-1)
    return stacked


def load_mam_eval_episodes(
    repo_id: str,
    root: str | Path | None = None,
    episodes: list[int] | None = None,
) -> list[MamEvalEpisode]:
    """Load only the MAM control columns needed by online eval."""

    meta = LeRobotDatasetMetadata(repo_id, root=root)
    missing = [
        key
        for key in (MAM_MAS_ACTION_ABSOLUTE, MAM_MAS_ACTION_MASK, MAM_PROGRESS)
        if key not in meta.features
    ]
    if missing:
        raise ValueError(f"MAM eval dataset is missing required features: {missing}")

    dataset = LeRobotDataset(repo_id, root=root, episodes=episodes, return_uint8=False)
    columns = [
        "episode_index",
        "frame_index",
        MAM_MAS_ACTION_ABSOLUTE,
        MAM_MAS_ACTION_MASK,
        MAM_PROGRESS,
    ]
    if "task" in dataset.hf_dataset.column_names:
        columns.append("task")
    table = dataset.select_columns(columns)

    grouped: dict[int, dict[str, list[Any]]] = defaultdict(
        lambda: {
            MAM_MAS_ACTION_ABSOLUTE: [],
            MAM_MAS_ACTION_MASK: [],
            MAM_PROGRESS: [],
            "task": [],
        }
    )
    for row in table:
        ep_idx = int(row["episode_index"])
        grouped[ep_idx][MAM_MAS_ACTION_ABSOLUTE].append(row[MAM_MAS_ACTION_ABSOLUTE])
        grouped[ep_idx][MAM_MAS_ACTION_MASK].append(row[MAM_MAS_ACTION_MASK])
        grouped[ep_idx][MAM_PROGRESS].append(row[MAM_PROGRESS])
        grouped[ep_idx]["task"].append(row.get("task", ""))

    rows = _episode_rows(meta)
    episode_columns = _column_names(meta.episodes)
    init_key = next((key for key in LIBERO_INIT_STATE_ID_KEYS if key in episode_columns), None)
    out: list[MamEvalEpisode] = []
    for ep_idx in sorted(grouped):
        row = rows.get(ep_idx, {})
        init_state_id = int(_row_get(row, init_key, ep_idx)) if init_key is not None else ep_idx
        mask_type = str(_row_get(row, "mask_type", "unknown"))
        raw_mask_slot = _row_get(row, "mask_type_slot", -1)
        try:
            mask_type_slot = int(raw_mask_slot)
        except (TypeError, ValueError):
            mask_type_slot = -1
        task_values = grouped[ep_idx]["task"]
        task = str(task_values[0]) if task_values else ""
        out.append(
            MamEvalEpisode(
                episode_index=ep_idx,
                init_state_id=init_state_id,
                mask_type=mask_type,
                mask_type_slot=mask_type_slot,
                task=task,
                mas_action_absolute=_stack_float32(grouped[ep_idx][MAM_MAS_ACTION_ABSOLUTE]),
                mas_action_mask=_stack_float32(grouped[ep_idx][MAM_MAS_ACTION_MASK]),
                progress=_stack_float32(grouped[ep_idx][MAM_PROGRESS], keep_feature_dim=True),
            )
        )
    return out


def configure_mam_eval_init_state_ids(cfg: Any, episodes: list[MamEvalEpisode], n_episodes: int) -> None:
    if cfg.env is None or getattr(cfg.env, "type", None) not in {"libero", "libero_plus"}:
        raise ValueError("MAM online eval currently requires env.type=libero or libero_plus.")
    if not hasattr(cfg.env, "init_state_ids"):
        raise ValueError("MAM online eval requires env config with init_state_ids support.")
    if len(episodes) < n_episodes:
        raise ValueError(f"MAM eval requested {n_episodes} episodes but dataset only has {len(episodes)}.")
    if getattr(cfg.eval, "batch_size", 1) > n_episodes:
        cfg.eval.batch_size = n_episodes
    cfg.env.init_state_ids = [int(ep.init_state_id) for ep in episodes[:n_episodes]]


def _resolve_stpm_paths(config: MamConfig) -> tuple[Path | None, Path | None]:
    ckpt = Path(config.stpm_checkpoint_path) if config.stpm_checkpoint_path else None
    cfg = Path(config.stpm_config_path) if config.stpm_config_path else None
    if config.stpm_path:
        root = Path(config.stpm_path)
        ckpt = ckpt or root / "checkpoints" / "reward_best.pt"
        cfg = cfg or root / "config.yaml"
    return ckpt, cfg


def make_stpm_encoder(config: MamConfig) -> STPMEncoder | None:
    ckpt, cfg = _resolve_stpm_paths(config)
    if ckpt is None or cfg is None:
        return None
    if not ckpt.exists() or not cfg.exists():
        raise FileNotFoundError(f"STPM artifacts not found: checkpoint={ckpt}, config={cfg}")
    return STPMEncoder(ckpt_path=ckpt, config_path=cfg, device=config.device)


def _stack_history(history: deque[Tensor], target_len: int) -> Tensor:
    if not history:
        raise RuntimeError("Cannot stack empty history.")
    frames = list(history)
    while len(frames) < target_len:
        frames.insert(0, frames[0])
    return torch.stack(frames[-target_len:], dim=1)


def _predict_progress(
    stpm: STPMEncoder | None,
    image_history: dict[str, deque[Tensor]],
    state_history: deque[Tensor],
    tasks: list[str],
    step: int,
    max_steps: int,
) -> Tensor:
    batch_size = state_history[-1].shape[0]
    if stpm is None:
        return torch.full((batch_size,), step / float(max(max_steps - 1, 1)), dtype=torch.float32)

    camera_tensors = []
    for key in stpm.camera_names:
        if key not in image_history:
            raise ValueError(
                f"STPM expected camera '{key}' but current observation has {list(image_history)}"
            )
        camera_tensors.append(_stack_history(image_history[key], stpm.n_obs_steps + 1))
    rgb = torch.stack(camera_tensors, dim=2)
    state = _stack_history(state_history, stpm.n_obs_steps + 1)
    return stpm.predict_progress(rgb, state, tasks=tasks).detach().cpu().float().clamp(0, 1)


def _slice_episode_window(
    episode: MamEvalEpisode,
    progress: float,
    config: MamConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    ep_len = int(episode.mas_action_absolute.shape[0])
    center = int(round(float(np.clip(progress, 0.0, 1.0)) * max(ep_len - 1, 0)))
    start = max(0, center - config.mas_long_backward_length)
    length = max(
        config.mas_long_window_horizon,
        config.mas_long_backward_length + config.mas_short_window_horizon,
        config.n_action_steps,
    )
    indices = torch.arange(start, start + length).clamp(max=max(ep_len - 1, 0))
    return (
        episode.mas_action_absolute[indices],
        episode.mas_action_mask[indices],
        episode.progress[indices],
    )


def _build_mam_fields(
    episodes: list[MamEvalEpisode],
    progress: Tensor,
    config: MamConfig,
    offset: int,
    batch_size: int,
) -> dict[str, Tensor]:
    mas_actions = []
    mas_masks = []
    progresses = []
    for env_i in range(batch_size):
        ep = episodes[offset + env_i]
        action, mask, prog = _slice_episode_window(ep, float(progress[env_i]), config)
        mas_actions.append(action)
        mas_masks.append(mask)
        progresses.append(prog)
    return {
        MAM_MAS_ACTION_ABSOLUTE: torch.stack(mas_actions, dim=0),
        MAM_MAS_ACTION_MASK: torch.stack(mas_masks, dim=0),
        MAM_PROGRESS: torch.stack(progresses, dim=0),
    }


def _apply_inpainting(config: MamConfig, action: Tensor, batch: dict[str, Tensor]) -> Tensor:
    if not config.inpainting or MAM_SHORT_WINDOW not in batch or MAM_SHORT_WINDOW_MASK not in batch:
        return action
    action_dim = action.shape[-1]
    known = batch[MAM_SHORT_WINDOW][:, : action.shape[1], :action_dim].to(action.device, action.dtype)
    mask = batch[MAM_SHORT_WINDOW_MASK][:, : action.shape[1], :action_dim].to(action.device, action.dtype)
    return torch.where(mask > 0.5, known, action)


def rollout_mam(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    episodes: list[MamEvalEpisode],
    episode_offset: int,
    stpm: STPMEncoder | None,
    return_observations: bool = False,
) -> dict[str, Tensor | dict[str, Tensor]]:
    assert isinstance(policy.config, MamConfig)
    config = policy.config
    policy.reset()
    observation, _ = env.reset(seed=None)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []
    absolute_action_queue: deque[Tensor] = deque()
    history_len = stpm.n_obs_steps + 1 if stpm else 1
    image_history: dict[str, deque[Tensor]] = defaultdict(lambda: deque(maxlen=history_len))
    state_history: deque[Tensor] = deque(maxlen=(stpm.n_obs_steps + 1 if stpm else 1))

    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running MAM rollout with at most {max_steps} steps",
        disable=inside_slurm(),
        leave=False,
    )
    step = 0
    while not np.all(done) and step < max_steps:
        observation = preprocess_observation(observation)
        if return_observations:
            all_observations.append(
                {key: value.clone() for key, value in observation.items() if isinstance(value, Tensor)}
            )

        try:
            observation["task"] = list(env.call("task_description"))
        except (AttributeError, NotImplementedError):
            observation["task"] = [episodes[episode_offset + i].task for i in range(env.num_envs)]

        observation = env_preprocessor(observation)
        anchor_state = observation[OBS_STATE].clone()
        state_history.append(anchor_state)
        for key, value in observation.items():
            if key.startswith(f"{OBS_IMAGES}."):
                image_history[key].append(value)

        progress = _predict_progress(
            stpm=stpm,
            image_history=image_history,
            state_history=state_history,
            tasks=list(observation.get("task", [""] * env.num_envs)),
            step=step,
            max_steps=max_steps,
        )
        observation.update(_build_mam_fields(episodes, progress, config, episode_offset, env.num_envs))
        processed = preprocessor(observation)

        with torch.inference_mode():
            processed = policy.update_observation_queue(processed)
            if len(absolute_action_queue) == 0:
                relative_chunk = policy.predict_action_chunk(processed)
                relative_chunk = _apply_inpainting(config, relative_chunk, processed)
                relative_chunk = postprocessor(relative_chunk)
                absolute_chunk = chunk_relative_to_absolute(
                    relative_chunk, anchor_state.to(relative_chunk.device)
                )
                if not isinstance(absolute_chunk, Tensor):
                    absolute_chunk = torch.as_tensor(absolute_chunk)
                absolute_action_queue.extend(absolute_chunk.transpose(0, 1))
            action = absolute_action_queue.popleft()

        action_transition = env_postprocessor({ACTION: action})
        action_numpy = action_transition[ACTION].detach().cpu().numpy()
        observation, reward, terminated, truncated, info = env.step(action_numpy)

        if "final_info" in info:
            successes = info["final_info"]["is_success"].tolist()
        elif "is_success" in info:
            successes = (
                info["is_success"].tolist()
                if hasattr(info["is_success"], "tolist")
                else [bool(info["is_success"])] * env.num_envs
            )
        else:
            successes = [False] * env.num_envs

        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)
        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))
        step += 1
        progbar.update()

    if return_observations:
        observation = preprocess_observation(observation)
        all_observations.append(
            {key: value.clone() for key, value in observation.items() if isinstance(value, Tensor)}
        )

    ret: dict[str, Any] = {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    if return_observations:
        ret[OBS_STR] = {
            key: torch.stack([obs[key] for obs in all_observations], dim=1)
            for key in all_observations[0]
        }
    return ret


def eval_mam_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    episodes: list[MamEvalEpisode],
    n_episodes: int,
    stpm: STPMEncoder | None = None,
    return_episode_data: bool = False,
) -> dict[str, Any]:
    start = time.time()
    policy.eval()
    if not episodes:
        raise ValueError("MAM eval requires at least one episode.")
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)
    padded_len = n_batches * env.num_envs
    rollout_episodes = episodes + [episodes[-1]] * max(0, padded_len - len(episodes))
    sum_rewards = []
    max_rewards = []
    successes = []
    episode_data: dict | None = None
    per_episode = []

    for batch_ix in trange(n_batches, desc="Stepping through MAM eval batches", disable=inside_slurm()):
        offset = batch_ix * env.num_envs
        rollout_data = rollout_mam(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            episodes=rollout_episodes,
            episode_offset=offset,
            stpm=stpm,
            return_observations=return_episode_data,
        )
        n_steps = rollout_data["done"].shape[1]
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)
        mask = (torch.arange(n_steps) <= (done_indices + 1).unsqueeze(1)).int()
        batch_sum_rewards = ((rollout_data["reward"] * mask).sum(dim=1)).tolist()
        batch_max_rewards = ((rollout_data["reward"] * mask).max(dim=1).values).tolist()
        batch_successes = ((rollout_data["success"] * mask).any(dim=1)).tolist()

        for env_i, (sum_reward, max_reward, success) in enumerate(
            zip(batch_sum_rewards, batch_max_rewards, batch_successes, strict=False)
        ):
            if offset + env_i >= n_episodes:
                continue
            ep = episodes[offset + env_i]
            sum_rewards.append(sum_reward)
            max_rewards.append(max_reward)
            successes.append(success)
            per_episode.append(
                {
                    "episode_ix": offset + env_i,
                    "source_episode_id": ep.episode_index,
                    "init_state_id": ep.init_state_id,
                    "mask_type": ep.mask_type,
                    "mask_type_slot": ep.mask_type_slot,
                    "sum_reward": sum_reward,
                    "max_reward": max_reward,
                    "success": success,
                }
            )

        if return_episode_data:
            this_episode_data = _compile_episode_data(
                rollout_data,
                done_indices,
                start_episode_index=offset,
                start_data_index=(0 if episode_data is None else (episode_data["index"][-1].item() + 1)),
                fps=env.unwrapped.metadata["render_fps"],
            )
            episode_data = this_episode_data if episode_data is None else {
                key: torch.cat([episode_data[key], this_episode_data[key]]) for key in episode_data
            }

    info: dict[str, Any] = {
        "per_episode": per_episode[:n_episodes],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
            "video_paths": [],
        },
    }
    if return_episode_data:
        info["episodes"] = episode_data
    return info


def _aggregate_success(per_episode: list[dict[str, Any]], key: str) -> dict[str, float]:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for ep in per_episode:
        buckets[str(ep[key])].append(bool(ep["success"]))
    return {bucket: float(np.nanmean(values) * 100) for bucket, values in sorted(buckets.items())}


def eval_mam_policy_all(
    envs: dict[str, dict[int, gym.vector.VectorEnv]],
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    episodes: list[MamEvalEpisode],
    n_episodes: int,
) -> dict[str, Any]:
    start = time.time()
    if not isinstance(policy.config, MamConfig):
        raise TypeError("eval_mam_policy_all requires a MamPolicy/MamConfig.")
    stpm = make_stpm_encoder(policy.config)
    if stpm is None:
        logging.warning("MAM eval running without STPM; using rollout step ratio as progress fallback.")

    per_task_infos = []
    all_per_episode = []
    all_sum_rewards = []
    all_max_rewards = []
    all_successes = []
    offset = 0
    for task_group, task_map in envs.items():
        for task_id, env in task_map.items():
            remaining = n_episodes - offset
            if remaining <= 0:
                break
            task_n = min(remaining, len(episodes) - offset)
            task_result = eval_mam_policy(
                env=env,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                episodes=episodes[offset : offset + task_n],
                n_episodes=task_n,
                stpm=stpm,
            )
            per_episode = task_result["per_episode"]
            all_per_episode.extend(per_episode)
            all_sum_rewards.extend([ep["sum_reward"] for ep in per_episode])
            all_max_rewards.extend([ep["max_reward"] for ep in per_episode])
            all_successes.extend([ep["success"] for ep in per_episode])
            per_task_infos.append(
                {
                    "task_group": task_group,
                    "task_id": task_id,
                    "metrics": task_result["aggregated"],
                }
            )
            offset += task_n
            env.close()

    return {
        "per_task": per_task_infos,
        "per_episode": all_per_episode,
        "per_mask_type_success": _aggregate_success(all_per_episode, "mask_type"),
        "per_mask_slot_success": _aggregate_success(all_per_episode, "mask_type_slot"),
        "overall": {
            "avg_sum_reward": float(np.nanmean(all_sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(all_max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "n_episodes": len(all_sum_rewards[:n_episodes]),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / max(1, len(all_sum_rewards[:n_episodes])),
            "video_paths": [],
        },
    }
