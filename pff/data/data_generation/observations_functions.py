"""
This file contains the observation functions that create the separation 
between observations and  remainders, the reminder can be either future 
or selected from random in betweens, or None

"""
import torch
from typing import Callable, Optional, Tuple
from torchtyping import TensorType

def fix_past_time_random_selection(
    full_simulation: TensorType["N", "S"],
    full_simulation_times: TensorType["N", "S"],
    *,
    boundary_ratio: float = 0.1,
    fixed_M_max: int,
    num_obs_sampler: Optional[Callable[[int], torch.Tensor]] = None,
    generator: Optional[torch.Generator] = None,
    **kwargs,
) -> Tuple[
    TensorType["N", "M"],
    TensorType["N", "M"],
    TensorType["N", "M"],
    None,
    None,
    None,
]:
    """Select observation time-points uniformly without replacement.

    Each row samples indices from the simulation grid independently and
    uniformly (no replacement), then sorts the selected points by sampled
    timestamps to keep chronological ordering in the output tensors.
    """
    if full_simulation is None:
        return (None,) * 6

    device = full_simulation.device
    N, S = full_simulation.shape
    M = int(max(0, fixed_M_max))

    gen = generator if generator is not None else torch.default_generator
    observations = torch.zeros(N, M, device=device, dtype=full_simulation.dtype)
    observation_times = torch.zeros(N, M, device=device, dtype=full_simulation_times.dtype)
    obs_mask = torch.zeros(N, M, dtype=torch.bool, device=device)

    sample_cap = min(M, S)
    if sample_cap == 0:
        return observations, observation_times, obs_mask, None, None, None

    if num_obs_sampler is None:
        num_obs = torch.full((N,), sample_cap, dtype=torch.long, device=device)
    else:
        num_obs = num_obs_sampler(N).to(device=device, dtype=torch.long).clamp(1, sample_cap)

    # Per-row sampling keeps selection uniform without replacement.
    for row in range(N):
        row_count = int(num_obs[row].item())
        if row_count <= 0:
            continue
        selected = torch.randperm(S, generator=gen, device=device)[:row_count]
        if row_count > 1:
            # Order chosen simulation indices by sampled time for stable packing.
            order = torch.argsort(full_simulation_times[row, selected])
            selected = selected[order]
        observations[row, :row_count] = full_simulation[row, selected]
        observation_times[row, :row_count] = full_simulation_times[row, selected]
        obs_mask[row, :row_count] = True

    return observations, observation_times, obs_mask, None, None, None
