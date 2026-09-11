"""Utility for initializing :class:`AICMECompartmentsDataBatch` objects.

This small helper is primarily used in older preprocessing scripts. It takes
precomputed observation tensors and wraps them into a minimal
``AICMECompartmentsDataBatch`` where only the context fields are populated.
All other entries are set to empty tensors or placeholders so that the
resulting object conforms to the new metadata interface.
"""

from __future__ import annotations

import torch

from pff.data.datasets.aicme_batch import AICMECompartmentsDataBatch


def initialize_aicme_batch(
    observations: torch.Tensor,
    observations_times: torch.Tensor,
    observations_mask: torch.Tensor,
) -> AICMECompartmentsDataBatch:
    """Wrap raw tensors into an :class:`AICMECompartmentsDataBatch`.

    Parameters
    ----------
    observations:
        Tensor of shape ``[I, T]`` containing concentration values.
    observations_times:
        Tensor of shape ``[I, T]`` with the corresponding time points.
    observations_mask:
        Boolean tensor of shape ``[I, T]`` indicating valid entries.

    Returns
    -------
    AICMECompartmentsDataBatch
        Batch with ``B=1`` where all context fields are populated and the
        remaining fields are placeholders (zeros or empty strings).
    """

    # Add batch dimension (B=1) and feature dimension for observations and times
    context_obs = observations.unsqueeze(0).unsqueeze(-1)  # [1, I, T, 1]
    context_obs_time = observations_times.unsqueeze(0).unsqueeze(-1)  # [1, I, T, 1]
    # Add batch dimension for mask
    context_obs_mask = observations_mask.unsqueeze(0)  # [1, I, T]

    num_individuals = observations.shape[0]

    return AICMECompartmentsDataBatch(
        target_obs=None,
        target_obs_time=None,
        target_obs_mask=None,
        target_rem_sim=None,
        target_rem_sim_time=None,
        target_rem_sim_mask=None,
        target_dosing_amounts=torch.zeros(1, 0),
        target_dosing_route_types=torch.zeros(1, 0, dtype=torch.long),
        context_obs=context_obs,
        context_obs_time=context_obs_time,
        context_obs_mask=context_obs_mask,
        context_rem_sim=None,
        context_rem_sim_time=None,
        context_rem_sim_mask=None,
        context_dosing_amounts=torch.zeros(1, num_individuals),
        context_dosing_route_types=torch.zeros(1, num_individuals, dtype=torch.long),
        study_name=[""],
        context_subject_name=[["" for _ in range(num_individuals)]],
        target_subject_name=[["" for _ in range(0)]],
        substance_name=[""],
        time_scales=None,
        is_empirical=False,
    )

