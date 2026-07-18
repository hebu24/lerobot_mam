from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import einops
import torch
from torch import Tensor, nn
from torch.nn import functional

from lerobot.policies.diffusion.modeling_diffusion import (
    TASK_KEY,
    DiffusionConditionalUnet1d,
    DiffusionLanguageEncoder,
    DiffusionRgbEncoder,
    _make_noise_scheduler,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.processor.libero_relative_action_processor import slice_current_action_window
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.utils.import_utils import require_package

from .configuration_mam import MamConfig
from .processor_mam import (
    MAM_ACTION_MASK,
    MAM_LONG_WINDOW,
    MAM_LONG_WINDOW_MASK,
    MAM_SHORT_WINDOW,
    MAM_SHORT_WINDOW_MASK,
)

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDatasetMetadata


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor | None:
    if mask.shape != values.shape:
        raise ValueError(f"Mask shape {tuple(mask.shape)} must match values shape {tuple(values.shape)}.")
    if not torch.any(mask):
        return None
    return values[mask].mean()


def _compute_mam_diffusion_loss(
    per_element_loss: Tensor,
    *,
    action_mask: Tensor | None,
    valid_mask: Tensor | None,
    loss_mode: str,
    known_region_weight: float,
) -> Tensor:
    if valid_mask is None:
        valid = torch.ones_like(per_element_loss, dtype=torch.bool)
    else:
        try:
            valid = torch.broadcast_to(
                valid_mask.to(device=per_element_loss.device, dtype=torch.bool),
                per_element_loss.shape,
            )
        except RuntimeError as exc:
            raise ValueError(
                f"Padding mask shape {tuple(valid_mask.shape)} cannot cover loss shape "
                f"{tuple(per_element_loss.shape)}."
            ) from exc

    if loss_mode == "average":
        valid_loss = _masked_mean(per_element_loss, valid)
        return per_element_loss.sum() * 0.0 if valid_loss is None else valid_loss
    if loss_mode != "weighted":
        raise ValueError(f"Unsupported loss_mode={loss_mode!r}.")
    if action_mask is None:
        raise ValueError("Weighted MAM loss requires mam.action_mask.")
    if action_mask.shape != per_element_loss.shape:
        raise ValueError(
            f"Action mask shape {tuple(action_mask.shape)} must match loss shape "
            f"{tuple(per_element_loss.shape)}."
        )

    known = action_mask.to(device=per_element_loss.device) > 0.5
    known_loss = _masked_mean(per_element_loss, known & valid)
    unknown_loss = _masked_mean(per_element_loss, (~known) & valid)
    known_weight = float(known_region_weight)
    unknown_weight = 1.0 - known_weight

    weighted_loss = None
    total_weight = 0.0
    if known_loss is not None and known_weight > 0.0:
        weighted_loss = known_loss * known_weight
        total_weight += known_weight
    if unknown_loss is not None and unknown_weight > 0.0:
        unknown_term = unknown_loss * unknown_weight
        weighted_loss = unknown_term if weighted_loss is None else weighted_loss + unknown_term
        total_weight += unknown_weight
    if weighted_loss is not None and total_weight > 0.0:
        return weighted_loss / total_weight

    present_losses = [loss for loss in (known_loss, unknown_loss) if loss is not None]
    if present_losses:
        return torch.stack(present_losses).mean()
    return per_element_loss.sum() * 0.0


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
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, out_dim),
            nn.ReLU(inplace=True),
        )
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, values: Tensor, masks: Tensor) -> Tensor:
        if values.ndim != 4 or masks.shape != values.shape:
            raise ValueError(
                f"Expected MAM long values/masks shape (B,S,T,D), got {values.shape}, {masks.shape}."
            )
        if values.shape[-1] != self.step_dim:
            raise ValueError(f"Expected MAM step dimension {self.step_dim}, got {values.shape[-1]}.")
        batch_size, n_obs_steps = values.shape[:2]
        mas_values = values.permute(0, 1, 3, 2)
        mas_masks = masks.permute(0, 1, 3, 2)
        x = torch.stack((mas_values, mas_masks), dim=2).reshape(
            batch_size * n_obs_steps,
            2,
            self.step_dim,
            values.shape[2],
        )
        return self.net(x).reshape(batch_size, n_obs_steps, self.out_dim)


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

        self.language_encoder = DiffusionLanguageEncoder(config) if config.use_language_conditioning else None
        unet_global_cond_dim = global_cond_dim * config.n_obs_steps
        if self.language_encoder is not None:
            unet_global_cond_dim += config.language_output_dim

        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=unet_global_cond_dim)
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
                [encoder(images) for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)]
            )
            img_features = einops.rearrange(
                img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
            )
        else:
            img_features = self.rgb_encoder(einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ..."))
            img_features = einops.rearrange(
                img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
            )
        return [img_features]

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        if n_obs_steps != self.config.n_obs_steps:
            raise ValueError(f"Expected {self.config.n_obs_steps} observation steps, got {n_obs_steps}.")
        features = [batch[OBS_STATE]]
        features.extend(self._encode_images(batch, batch_size, n_obs_steps))
        if self.config.env_state_feature:
            features.append(batch[OBS_ENV_STATE])

        state = batch[OBS_STATE]
        if self.long_window_encoder is not None:
            long_values = batch.get(MAM_LONG_WINDOW)
            long_masks = batch.get(MAM_LONG_WINDOW_MASK)
            if (long_values is None) != (long_masks is None):
                raise ValueError("MAM long window values and masks must be provided together.")
            if long_values is None or long_masks is None:
                long_feature = state.new_zeros((batch_size, n_obs_steps, self.config.mas_long_feature_dim))
            else:
                if long_values.shape[:2] != (batch_size, n_obs_steps):
                    raise ValueError(
                        "MAM long window must align with observation history: "
                        f"expected prefix {(batch_size, n_obs_steps)}, got {tuple(long_values.shape)}."
                    )
                long_feature = self.long_window_encoder(long_values, long_masks)
            features.append(long_feature)

        short_window = batch.get(MAM_SHORT_WINDOW)
        if short_window is None:
            short_dim = self.config.mas_short_window_horizon * self.mas_step_dim
            short_feature = state.new_zeros((batch_size, n_obs_steps, short_dim))
        else:
            expected_shape = (
                batch_size,
                n_obs_steps,
                self.config.mas_short_window_horizon,
                self.mas_step_dim,
            )
            if short_window.shape != expected_shape:
                raise ValueError(
                    f"Expected MAM short window shape {expected_shape}, got {tuple(short_window.shape)}."
                )
            short_feature = short_window.reshape(batch_size, n_obs_steps, -1)
        features.append(short_feature)
        global_cond = torch.cat(features, dim=-1).flatten(start_dim=1)
        if self.language_encoder is not None:
            language_cond = self.language_encoder(
                batch.get(TASK_KEY),
                batch_size,
                device=global_cond.device,
                dtype=global_cond.dtype,
            )
            global_cond = torch.cat((global_cond, language_cond), dim=-1)
        return global_cond

    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor,
        noise: Tensor | None = None,
        action_known_0: Tensor | None = None,
        action_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        device = global_cond.device
        dtype = global_cond.dtype
        sample = (
            noise.to(device=device, dtype=dtype)
            if noise is not None
            else torch.randn(
                size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )
        if (action_known_0 is None) != (action_mask is None):
            raise ValueError("action_known_0 and action_mask must be provided together.")
        known_mask = None
        if action_known_0 is not None and action_mask is not None:
            if action_known_0.shape != sample.shape or action_mask.shape != sample.shape:
                raise ValueError(
                    "Inpainting tensors must match the diffusion sample shape: "
                    f"sample={tuple(sample.shape)}, known={tuple(action_known_0.shape)}, "
                    f"mask={tuple(action_mask.shape)}."
                )
            action_known_0 = action_known_0.to(device=device, dtype=dtype)
            known_mask = action_mask.to(device=device) > 0.5
        has_known_action = known_mask is not None and bool(torch.any(known_mask).item())

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        timesteps = [int(t) for t in self.noise_scheduler.timesteps]
        for step_index, timestep in enumerate(timesteps):
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], timestep, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            sample = self.noise_scheduler.step(
                model_output=model_output,
                timestep=timestep,
                sample=sample,
                generator=generator,
            ).prev_sample
            if has_known_action:
                if step_index + 1 < len(timesteps):
                    next_timestep = timesteps[step_index + 1]
                    known_noise = torch.randn(
                        action_known_0.shape,
                        dtype=dtype,
                        device=device,
                        generator=generator,
                    )
                    timestep_batch = torch.full((batch_size,), next_timestep, dtype=torch.long, device=device)
                    known_at_timestep = self.noise_scheduler.add_noise(
                        action_known_0, known_noise, timestep_batch
                    )
                else:
                    known_at_timestep = action_known_0
                sample = torch.where(known_mask, known_at_timestep, sample)
        return sample

    def _build_inpainting_inputs(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        short_window = batch.get(MAM_SHORT_WINDOW)
        short_mask = batch.get(MAM_SHORT_WINDOW_MASK)
        if short_window is None or short_mask is None:
            raise ValueError("MAM inpainting requires mam.mas_short_window and mam.mas_short_window_mask.")
        if short_window.shape != short_mask.shape:
            raise ValueError(
                f"MAM inpainting value/mask shapes differ: {tuple(short_window.shape)} and "
                f"{tuple(short_mask.shape)}."
            )
        if short_window.ndim == 4:
            current_window = short_window[:, -1]
            current_mask = short_mask[:, -1]
        elif short_window.ndim == 3:
            current_window = short_window
            current_mask = short_mask
        else:
            raise ValueError(f"Expected MAM short window shape (B,S,T,D), got {tuple(short_window.shape)}.")

        batch_size = current_window.shape[0]
        action_dim = self.config.action_feature.shape[0]
        action_known_0 = current_window.new_zeros((batch_size, self.config.horizon, action_dim))
        action_mask = current_mask.new_zeros((batch_size, self.config.horizon, action_dim))
        start = self.config.n_obs_steps - 1
        copy_length = min(current_window.shape[1], self.config.horizon - start)
        action_known_0[:, start : start + copy_length] = current_window[:, :copy_length, :action_dim]
        action_mask[:, start : start + copy_length] = current_mask[:, :copy_length, :action_dim]
        return action_known_0, action_mask

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond = self._prepare_global_conditioning(batch)
        action_known_0 = None
        action_mask = None
        if self.config.inpainting:
            action_known_0, action_mask = self._build_inpainting_inputs(batch)
        actions = self.conditional_sample(
            batch_size,
            global_cond=global_cond,
            noise=noise,
            action_known_0=action_known_0,
            action_mask=action_mask,
        )
        start = n_obs_steps - 1
        return actions[:, start : start + self.config.n_action_steps]

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        trajectory = batch[ACTION]
        batch_size = trajectory.shape[0]
        global_cond = self._prepare_global_conditioning(batch)
        eps = torch.randn_like(trajectory)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(batch_size,),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)
        target = eps if self.config.prediction_type == "epsilon" else trajectory
        per_element_loss = functional.mse_loss(pred, target, reduction="none")
        valid_mask = None
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            valid_mask = (~batch["action_is_pad"].to(device=per_element_loss.device).bool()).unsqueeze(-1)

        return _compute_mam_diffusion_loss(
            per_element_loss,
            action_mask=batch.get(MAM_ACTION_MASK),
            valid_mask=valid_mask,
            loss_mode=self.config.loss_mode,
            known_region_weight=self.config.loss_mask_area_weight,
        )


class MamPolicy(PreTrainedPolicy):
    config_class = MamConfig
    name = "mam"

    def __init__(
        self,
        config: MamConfig,
        dataset_stats: dict | None = None,
        dataset_meta: LeRobotDatasetMetadata | None = None,
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
        for key in (
            MAM_LONG_WINDOW,
            MAM_LONG_WINDOW_MASK,
            MAM_SHORT_WINDOW,
            MAM_SHORT_WINDOW_MASK,
        ):
            if key in batch:
                queued[key] = batch[key]
        if self.config.use_language_conditioning and TASK_KEY in batch:
            queued[TASK_KEY] = batch[TASK_KEY]
        return self.diffusion.generate_actions(queued, noise=noise)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch = self.update_observation_queue(batch)
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            actions = slice_current_action_window(
                actions,
                n_obs_steps=self.config.n_obs_steps,
                n_action_steps=self.config.n_action_steps,
            )
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
