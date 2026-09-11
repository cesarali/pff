import os
from dataclasses import dataclass
from typing import Optional, Union

try:  # pragma: no cover - exercised indirectly via configuration loading
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    from pff.config_classes import yaml_fallback as yaml


@dataclass
class SourceProcessConfig:
    """
    Configuration for source processes used by flow and diffusion PK models.

    Supported source_type values (case-insensitive):
      - "gaussian_process" / "gp"
      - "ornstein_uhlenbeck" / "ou"
      - "wiener"
      - "normal" / "gaussian"
    """

    source_type: str = "gaussian_process"

    # Gaussian process hyper-parameter for RBF or OU.
    gp_length_scale: float = 0.1
    gp_variance: float = 1.0
    gp_eps: float = 1e-8
    gp_transform: str = 'softplus'  # transformation to apply to the sampled noise, e.g. 'softplus', 'exp'

    # Flow matching additive noise scale (used only in FlowPK).
    flow_sigma: float = 1e-4
    flow_num_steps: int = 100
    use_OT_coupling: bool = False

    @classmethod
    def from_yaml(cls, file_path: Union[str, os.PathLike]) -> "SourceProcessConfig":
        """Instantiate the source-process configuration from a YAML file."""

        with open(file_path, "r", encoding="utf-8") as handle:
            config_dict = yaml.safe_load(handle) or {}

        if isinstance(config_dict, dict):
            for key in ("source_process", "source", "noise_model"):
                if key in config_dict:
                    config_dict = config_dict.get(key) or {}
                    break

        if not isinstance(config_dict, dict):
            raise TypeError("Expected source process configuration to be a mapping.")

        return cls(**config_dict)
