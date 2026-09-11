"""Utilities for reshaping target-individual axes in FlowPK tensors.

These helpers centralize the repeated reshape patterns used when decoding
multiple target individuals independently.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torchtyping import TensorType


def flatten_target_axis_4d(
    x: TensorType["B", "It", "T", "C"]
) -> TensorType["BIt", 1, "T", "C"]:
    """Flatten target axis: ``[B, It, T, C] -> [B*It, 1, T, C]``."""
    if x.ndim != 4:
        raise ValueError(f"Expected a 4D tensor shaped [B, It, T, C], got ndim={x.ndim}.")
    bsz, num_targets, time_steps, channels = x.shape
    return x.reshape(bsz * num_targets, 1, time_steps, channels)


def flatten_target_axis_3d(x: TensorType["B", "It", "T"]) -> TensorType["BIt", 1, "T"]:
    """Flatten target axis: ``[B, It, T] -> [B*It, 1, T]``."""
    if x.ndim != 3:
        raise ValueError(f"Expected a 3D tensor shaped [B, It, T], got ndim={x.ndim}.")
    bsz, num_targets, time_steps = x.shape
    return x.reshape(bsz * num_targets, 1, time_steps)


def unflatten_target_axis_4d(
    x: TensorType["BIt", 1, "T", "C"], batch_size: int, num_targets: int
) -> TensorType["B", "It", "T", "C"]:
    """Unflatten target axis: ``[B*It, 1, T, C] -> [B, It, T, C]``."""
    if x.ndim != 4:
        raise ValueError(f"Expected a 4D tensor shaped [B*It, 1, T, C], got ndim={x.ndim}.")
    flat_batch, singleton, time_steps, channels = x.shape
    expected_flat_batch = batch_size * num_targets
    if singleton != 1:
        raise ValueError(f"Expected singleton target axis of size 1, got {singleton}.")
    if flat_batch != expected_flat_batch:
        raise ValueError(
            f"Expected leading axis {expected_flat_batch} (= {batch_size} * {num_targets}), "
            f"got {flat_batch}."
        )
    return x.reshape(batch_size, num_targets, time_steps, channels)


def flatten_many_target_axis_4d(*tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Flatten target axis for many 4D tensors in one call."""
    return tuple(flatten_target_axis_4d(x) for x in tensors)


def flatten_many_target_axis_3d(*tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Flatten target axis for many 3D tensors in one call."""
    return tuple(flatten_target_axis_3d(x) for x in tensors)
