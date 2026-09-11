"""Shared helpers for source-process configuration in flow and diffusion models."""

from __future__ import annotations

from typing import Callable, Tuple

import torch

from pff.config_classes.source_process_config import SourceProcessConfig
from pff.models.diffusion.noise import GaussianProcess, GaussianProcessRegression, Normal, OrnsteinUhlenbeck, Wiener

def normalize_source_type(source_type: str) -> str:
    """Normalize a source-process name to a canonical lowercase form."""
    return source_type.strip().lower().replace("-", "_").replace(" ", "_")

def build_source_process(
    source_cfg: SourceProcessConfig | None,
    *,
    dim: int = 1,
) -> Tuple[Callable, bool]:
    """
    Build the source process for flow/diffusion models.

    Returns
    -------
    process:
        Callable noise/source process instance.
    is_time_series:
        Whether the process induces temporal covariance and requires `t`.
    """
    if source_cfg is None:
        source_cfg = SourceProcessConfig()

    source_type = normalize_source_type(str(source_cfg.source_type))
    gp_variance = float(source_cfg.gp_variance)
    gp_length_scale = float(source_cfg.gp_length_scale)
    gp_eps = float(source_cfg.gp_eps)
    gp_transform=source_cfg.gp_transform

    if source_type in ("gaussian_process", "gp"):
        return GaussianProcess(dim=dim, variance=gp_variance, length_scale=gp_length_scale, epsilon=gp_eps, transform=gp_transform), True

    if source_type in ("gaussian_process_regression", "gp_regression"):
        return GaussianProcessRegression(dim=dim, variance=gp_variance, length_scale=gp_length_scale, epsilon=gp_eps, transform=gp_transform), True

    if source_type in ("ornstein_uhlenbeck", "ou"):
        return OrnsteinUhlenbeck(dim=dim, variance=gp_variance, length_scale=gp_length_scale, epsilon=gp_eps), True

    if source_type in ("wiener", "brownian"):
        return Wiener(dim=dim), False
         
    if source_type in ("normal", "gaussian", "white_noise"):
        return Normal(dim=dim), False

    raise ValueError(f"Unsupported source_type: {source_type!r}")

def sample_source_process(
    source_process: Callable,
    t: torch.Tensor,
    *,
    device: torch.device | None = None,
    is_time_series: bool = True,
) -> torch.Tensor:
    """Sample a source process given the target time grid."""
    if is_time_series:
        sample = source_process(t=t, device=device)
    else:
        sample = source_process(*t.shape[:-1])
    if torch.is_tensor(sample) and device is not None:
        sample = sample.to(device)
    return sample
