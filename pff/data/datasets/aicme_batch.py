"""Batch structures shared between synthetic and empirical pipelines."""

from __future__ import annotations

from collections import namedtuple
from typing import List, NamedTuple

import torch
from torchtyping import TensorType

ShapeConfig = namedtuple(
    "ShapeConfig",
    [
        "batch_size",
        "c_individuals",
        "num_obs_c",
        "remaining_obs_c",
        "t_individuals",
        "num_obs_t",
        "remaining_obs_t",
    ],
)


class AICMECompartmentsDataBatch(NamedTuple):
    """Container aggregating context and target trajectories.

    The tuple carries tensors describing observed measurements, simulated
    remainders, dosing metadata and masking utilities used across both the
    synthetic simulation pipeline and the empirical JSON tooling.
    """

    # max_num_individuals-max_n_new_individuals = n_c_individuals
    target_obs: TensorType["B", "t_ind", "num_obs_t", 1]
    target_obs_time: TensorType["B", "t_ind", "num_obs_t", 1]
    target_obs_mask: TensorType["B", "t_ind", "num_obs_t"]

    target_rem_sim: TensorType["B", "t_ind", "rem_obs_t", 1]
    target_rem_sim_time: TensorType["B", "t_ind", "rem_obs_t", 1]
    target_rem_sim_mask: TensorType["B", "t_ind", "rem_obs_t"]

    context_obs: TensorType["B", "c_ind", "num_obs_c", 1]
    context_obs_time: TensorType["B", "c_ind", "num_obs_c", 1]
    context_obs_mask: TensorType["B", "c_ind", "num_obs_c"]

    context_rem_sim: TensorType["B", "c_ind", "rem_obs_c", 1]
    context_rem_sim_time: TensorType["B", "c_ind", "rem_obs_c", 1]
    context_rem_sim_mask: TensorType["B", "c_ind", "rem_obs_c"]

    # Dosing information
    target_dosing_amounts: TensorType["B", "t_ind"]
    target_dosing_route_types: TensorType["B", "t_ind"]
    context_dosing_amounts: TensorType["B", "c_ind"]
    context_dosing_route_types: TensorType["B", "c_ind"]

    # Masks over padded individuals
    mask_context_individuals: TensorType["B", "c_ind"]
    mask_target_individuals: TensorType["B", "t_ind"]

    # 🆕 NEW: tracking metadata
    study_name: List[str]
    """Study identifier for each element in the batch (length ``B``)."""
    context_subject_name: List[List[str]]
    """Names of context individuals: shape ``[B][c_ind]``."""
    target_subject_name: List[List[str]]
    """Names of target individuals: shape ``[B][t_ind]``."""
    substance_name: List[str]
    """Drug or compound names corresponding to each study (length ``B``)."""

    # Meta information
    time_scales: TensorType["B", 2]  # shape : [B,2]
    is_empirical: bool = False  # NEW: True ⇢ empirical CSV, False ⇢ simulation

    @property
    def mask_individuals(self) -> TensorType["B", "c_ind"]:
        """Alias for backward compatibility; returns ``mask_context_individuals``."""

        return self.mask_context_individuals

    def detach_all(self) -> "AICMECompartmentsDataBatch":
        """Detaches all tensor fields from the computation graph."""

        return AICMECompartmentsDataBatch(
            *(t.detach() if isinstance(t, torch.Tensor) else t for t in self)
        )

    def log_transform(self) -> "AICMECompartmentsDataBatch":
        """Applies log transformation to observation and remainder tensors.

        Deprecated for training: log scaling is now expected to be handled by
        ``PKScaler`` (for example via ``value_method="log"`` or
        ``value_method="log_and_max"``).
        Kept for backward compatibility with older utilities.
        """

        transformed_tensors = []
        for name, tensor in zip(self._fields, self):
            if name in [
                "target_obs",
                "target_rem_sim",
                "context_obs",
                "context_rem_sim",
            ]:
                transformed_tensors.append(torch.log(tensor + 1e-6))
            else:
                transformed_tensors.append(tensor)
        return AICMECompartmentsDataBatch(*transformed_tensors)

    def to_device(self, device: torch.device) -> "AICMECompartmentsDataBatch":
        """Moves all tensor fields to the specified device (leaves strings untouched)."""

        return AICMECompartmentsDataBatch(
            *(t.to(device) if isinstance(t, torch.Tensor) else t for t in self)
        )

    def to(self, device: torch.device | str) -> "AICMECompartmentsDataBatch":
        """PyTorch-style alias delegating to :meth:`to_device`.

        Several generic utilities expect batch-like objects to implement
        ``.to(device)``. Exposing this alias keeps the explicit
        ``to_device(...)`` API while allowing those utilities to move the full
        databatch onto the target device safely.
        """

        return self.to_device(torch.device(device))

    def to_reconstruct_type(self) -> "AICMECompartmentsDataBatch":
        """
        Return a new databatch where the target trajectories are reconstructed
        by concatenating observed and remainder segments, then right-padding
        so that the target has the same time dimension as the context.
        The context is left untouched.
        """

        B, Ic, Tc, _ = self.context_obs.shape  # context time dimension is reference
        _, It, _, _ = self.target_obs.shape

        T_max = Tc  # max length for padding

        # allocate reconstructed tensors
        Xt_full = torch.zeros(
            B, It, T_max, 1, dtype=self.target_obs.dtype, device=self.target_obs.device
        )
        Tt_full = torch.zeros(
            B, It, T_max, 1, dtype=self.target_obs_time.dtype, device=self.target_obs_time.device
        )
        Mt_full = torch.zeros(B, It, T_max, dtype=torch.bool, device=self.target_obs_mask.device)

        # fill with observed + remainder segments
        for b in range(B):
            for i in range(It):
                o_len = int(self.target_obs_mask[b, i].sum().item())
                r_len = int(self.target_rem_sim_mask[b, i].sum().item())
                total = o_len + r_len
                if total == 0:
                    continue
                Xt_full[b, i, :o_len] = self.target_obs[b, i, :o_len]
                Xt_full[b, i, o_len:total] = self.target_rem_sim[b, i, :r_len]
                Tt_full[b, i, :o_len] = self.target_obs_time[b, i, :o_len]
                Tt_full[b, i, o_len:total] = self.target_rem_sim_time[b, i, :r_len]
                Mt_full[b, i, :total] = True

        return self._replace(
            target_obs=Xt_full,
            target_obs_time=Tt_full,
            target_obs_mask=Mt_full,
        )
