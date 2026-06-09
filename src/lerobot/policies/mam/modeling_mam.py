from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import einops
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    DiffusionRgbEncoder,
    _make_noise_scheduler,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.utils.import_utils import require_package

from .configuration_mam import MamConfig
from .processor_mam import (
    MAM_ACTION_MASK,
    MAM_LONG_WINDOW,
    MAM_LONG_WINDOW_MASK,
    MAM_SHORT_WINDOW,
)

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDatasetMetadata


class MamLongWindowEncoder(nn.Module):
    def __init__(self, step_dim: int, out_dim: int):
        super().__init__()
        self.step_dim = int(step_dim)
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, values: Tensor, masks: Tensor) -> Tensor:
        if values.ndim != 3 or masks.shape != values.shape:
            raise ValueError(
                f"Expected MAM long values/masks shape (B,T,D), got {values.shape}, {masks.shape}"
            )
        x = torch.stack((values, masks), dim=1)
        return self.net(x)


class MamDiffusionModel(nn.Module):
    def __init__(self, config: MamConfig):
        super().__init__()
        self.config = config

        global_cond_dim = config.robot_state_feature.shape[0]
        if config.image_features:
            num_images = len(config.image_features)
            if config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if config.env_state_feature:
            global_cond_dim += config.env_state_feature.shape[0]

        action_dim = config.action_feature.shape[0]
        self.mas_step_dim = action_dim + 1
        self.long_window_encoder = None
        if config.mas_long_feature_dim > 0 and config.mas_long_window_horizon > 0:
            self.long_window_encoder = MamLongWindowEncoder(
                step_dim=self.mas_step_dim,
                out_dim=config.mas_long_feature_dim,
            )
            global_cond_dim += config.mas_long_feature_dim
        global_cond_dim += config.mas_short_window_horizon * self.mas_step_dim

        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)
        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )
        self.num_inference_steps = (
            self.noise_scheduler.config.num_train_timesteps
            if config.num_inference_steps is None
            else config.num_inference_steps
        )

    def _encode_images(self, batch: dict[str, Tensor], batch_size: int, n_obs_steps: int) -> list[Tensor]:
        if not self.config.image_features:
            return []
        if self.config.use_separate_rgb_encoder_per_camera:
            images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
            img_features_list = torch.cat(
                [
                    encoder(images)
                    for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                ]
            )
            img_features = einops.rearrange(
                img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
            )
        else:
            img_features = self.rgb_encoder(
                einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
            )
            img_features = einops.rearrange(
                img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
            )
        return [img_features]

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        features = [batch[OBS_STATE]]
        features.extend(self._encode_images(batch, batch_size, n_obs_steps))
        if self.config.env_state_feature:
            features.append(batch[OBS_ENV_STATE])

        state = batch[OBS_STATE]
        if self.long_window_encoder is not None:
            long_values = batch.get(MAM_LONG_WINDOW)
            long_masks = batch.get(MAM_LONG_WINDOW_MASK)
            if long_values is None or long_masks is None:
                long_feature = state.new_zeros((batch_size, n_obs_steps, self.config.mas_long_feature_dim))
            else:
                encoded = self.long_window_encoder(long_values, long_masks)
                long_feature = encoded[:, None].expand(-1, n_obs_steps, -1)
            features.append(long_feature)

        short_window = batch.get(MAM_SHORT_WINDOW)
        if short_window is None:
            short_dim = self.config.mas_short_window_horizon * self.mas_step_dim
            short_feature = state.new_zeros((batch_size, n_obs_steps, short_dim))
        else:
            short_feature = short_window.reshape(batch_size, -1)[:, None].expand(-1, n_obs_steps, -1)
        features.append(short_feature)
        return torch.cat(features, dim=-1).flatten(start_dim=1)

    def conditional_sample(self, batch_size: int, global_cond: Tensor, noise: Tensor | None = None) -> Tensor:
        device = global_cond.device
        dtype = global_cond.dtype
        sample = (
            noise
            if noise is not None
            else torch.randn(
                size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
                dtype=dtype,
                device=device,
            )
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            sample = self.noise_scheduler.step(model_output, t, sample).prev_sample
        return sample

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond = self._prepare_global_conditioning(batch)
        actions = self.conditional_sample(batch_size, global_cond=global_cond, noise=noise)
        start = n_obs_steps - 1
        return actions[:, start : start + self.config.n_action_steps]

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        trajectory = batch[ACTION]
        batch_size = trajectory.shape[0]
        global_cond = self._prepare_global_conditioning(batch)
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(batch_size,),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)
        target = eps if self.config.prediction_type == "epsilon" else trajectory
        loss = F.mse_loss(pred, target, reduction="none")
        action_mask = batch.get(MAM_ACTION_MASK)
        if action_mask is None:
            return loss.mean()
        action_mask = action_mask.to(device=loss.device, dtype=loss.dtype)
        if self.config.loss_mode == "weighted":
            weights = torch.where(
                action_mask > 0.5,
                torch.full_like(action_mask, self.config.loss_mask_area_weight),
                torch.full_like(action_mask, 1.0 - self.config.loss_mask_area_weight),
            )
            return (loss * weights).sum() / weights.sum().clamp_min(1.0)
        return (loss * action_mask).sum() / action_mask.sum().clamp_min(1.0)


class MamPolicy(PreTrainedPolicy):
    config_class = MamConfig
    name = "mam"

    def __init__(
        self,
        config: MamConfig,
        dataset_stats: dict | None = None,
        dataset_meta: "LeRobotDatasetMetadata | None" = None,
        **kwargs,
    ):
        require_package("diffusers", extra="diffusion")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.diffusion = MamDiffusionModel(config)
        self._queues = None
        self.reset()

    def get_optim_params(self) -> dict:
        return self.diffusion.parameters()

    def reset(self):
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def update_observation_queue(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)
        batch.pop(ACTION, None)
        if self.config.image_features:
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        self._queues = populate_queues(self._queues, batch)
        return batch

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        queued = {key: torch.stack(list(self._queues[key]), dim=1) for key in batch if key in self._queues}
        for key in (MAM_LONG_WINDOW, MAM_LONG_WINDOW_MASK, MAM_SHORT_WINDOW):
            if key in batch:
                queued[key] = batch[key]
        return self.diffusion.generate_actions(queued, noise=noise)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch = self.update_observation_queue(batch)
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))
        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        if self.config.image_features:
            batch = dict(batch)
            for key in self.config.image_features:
                if self.config.n_obs_steps == 1 and batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        loss = self.diffusion.compute_loss(batch)
        return loss, None
