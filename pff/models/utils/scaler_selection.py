"""Helpers to map mix-data flags to scaler methods."""

from __future__ import annotations

from typing import Tuple


def resolve_scaler_methods(mix_cfg) -> Tuple[str, str]:
    """Resolve ``(value_method, time_method)`` from mix-data settings.

    Precedence:
    1. ``log_and_z=True`` -> ``"log_and_z"``
    2. ``log_and_max=True`` -> ``"log_and_max"``
    3. ``log_transform=True`` -> ``"log"``
    4. ``z_score_normalization=True`` -> ``"zscore"``
    5. ``normalize_by_max=True`` -> ``"max"``
    6. otherwise -> ``"none"``
    """
    if mix_cfg is None:
        return "none", "none"

    if getattr(mix_cfg, "log_and_z", False):
        value_method = "log_and_z"
    elif getattr(mix_cfg, "log_and_max", False):
        value_method = "log_and_max"
    elif getattr(mix_cfg, "log_transform", False):
        value_method = "log"
    elif getattr(mix_cfg, "z_score_normalization", False):
        value_method = "zscore"
    elif getattr(mix_cfg, "normalize_by_max", False):
        value_method = "max"
    else:
        value_method = "none"

    time_method = "max" if getattr(mix_cfg, "normalize_time", False) else "none"
    return value_method, time_method
