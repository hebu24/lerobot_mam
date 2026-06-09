from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


@PreTrainedConfig.register_subclass("mam")
@dataclass
class MamConfig(DiffusionConfig):
    """Masked Action Model policy configuration.

    MAM reuses the Diffusion Policy U-Net backbone, but adds masked-action-space
    conditioning and a mask-aware diffusion loss.
    """

    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8
    drop_n_last_frames: int = 7

    mas_short_window_horizon: int = 8
    mas_long_backward_length: int = 0
    mas_long_forward_length: int = 16
    mas_long_feature_dim: int = 64
    loss_mode: str = "average"
    loss_mask_area_weight: float = 0.2
    inpainting: bool = False

    mam_eval_dataset_repo_id: str | None = None
    mam_eval_dataset_root: str | None = None
    mam_eval_episodes: list[int] | None = None
    stpm_path: str | None = None
    stpm_checkpoint_path: str | None = None
    stpm_config_path: str | None = None

    @property
    def mam_delta_indices(self) -> list[int]:
        end = max(self.mas_long_forward_length, self.mas_short_window_horizon)
        return list(range(-self.mas_long_backward_length, end))

    @property
    def mas_long_window_horizon(self) -> int:
        return self.mas_long_backward_length + self.mas_long_forward_length

    def __post_init__(self):
        super().__post_init__()
        if self.mas_long_backward_length < 0:
            raise ValueError("mas_long_backward_length must be non-negative")
        if self.mas_long_forward_length <= 0:
            raise ValueError("mas_long_forward_length must be positive")
        if self.mas_short_window_horizon < 0:
            raise ValueError("mas_short_window_horizon must be non-negative")
        if self.mas_long_feature_dim < 0:
            raise ValueError("mas_long_feature_dim must be non-negative")
        if self.loss_mode not in {"average", "weighted"}:
            raise ValueError(f"Unsupported loss_mode={self.loss_mode!r}")
        if not 0.0 <= float(self.loss_mask_area_weight) <= 1.0:
            raise ValueError("loss_mask_area_weight must be in [0, 1]")
