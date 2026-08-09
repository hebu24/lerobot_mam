from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


@PreTrainedConfig.register_subclass("mam")
@dataclass
class MamConfig(DiffusionConfig):
    """Masked Action Model policy configuration.

    MAM reuses the Diffusion Policy denoiser, but adds masked-action-space
    conditioning and a mask-aware diffusion loss.
    """

    n_obs_steps: int = 2
    horizon: int = 32
    n_action_steps: int = 15
    drop_n_last_frames: int = 7
    use_relative_actions: bool = True

    mas_short_window_horizon: int = 15
    mas_long_backward_length: int = 0
    mas_long_forward_length: int = 32
    mas_long_feature_dim: int = 64
    loss_mode: str = "weighted"
    loss_mask_area_weight: float = 0.2
    do_mask_loss_for_padding: bool = True
    inpainting: bool = False

    mam_eval_dataset_repo_id: str | None = None
    mam_eval_dataset_root: str | None = None
    mam_eval_episodes: list[int] | None = None
    stpm_path: str | None = None
    stpm_checkpoint_path: str | None = None
    stpm_config_path: str | None = None
    stpm_paths: dict[str, str] | None = None
    stpm_checkpoint_paths: dict[str, str] | None = None
    stpm_config_paths: dict[str, str] | None = None

    @property
    def mam_delta_indices(self) -> list[int]:
        start = 1 - self.n_obs_steps - self.mas_long_backward_length
        action_end = 1 - self.n_obs_steps + self.horizon
        end = max(self.mas_long_forward_length, self.mas_short_window_horizon, action_end)
        return list(range(start, end))

    @property
    def mas_long_window_horizon(self) -> int:
        return self.mas_long_backward_length + self.mas_long_forward_length

    def __post_init__(self):
        super().__post_init__()
        if not self.use_relative_actions:
            raise ValueError(
                "MAM requires use_relative_actions=True so targets, MAS, and eval share one space."
            )
        if self.mas_long_backward_length < 0:
            raise ValueError("mas_long_backward_length must be non-negative")
        if self.mas_long_forward_length <= 0:
            raise ValueError("mas_long_forward_length must be positive")
        if self.mas_short_window_horizon < 0:
            raise ValueError("mas_short_window_horizon must be non-negative")
        if self.mas_long_feature_dim < 0:
            raise ValueError("mas_long_feature_dim must be non-negative")
        if self.mas_long_feature_dim > 0 and self.mas_long_window_horizon < 4:
            raise ValueError("mas_long_window_horizon must be at least 4 for the two temporal pooling stages")
        if self.loss_mode not in {"average", "weighted"}:
            raise ValueError(f"Unsupported loss_mode={self.loss_mode!r}")
        if not 0.0 <= float(self.loss_mask_area_weight) <= 1.0:
            raise ValueError("loss_mask_area_weight must be in [0, 1]")
