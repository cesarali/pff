import random
from typing import Optional, Sequence, Tuple

import torch
from torchtyping import TensorType


def gather_distinct_times_per_substance(
    db,
    time_sources: Sequence[str] | None = None,
) -> Tuple[
    TensorType["B", "Tdistinct_max", 1],
    TensorType["B", "Tdistinct_max"],
]:
    """
    Gather and deduplicate available times per substance.

    This utility collects times from selected context/target streams, merges
    them per batch element (substance), removes duplicates, and produces a
    unified time grid suitable for decoding new trajectories. Invalid entries
    are ignored via time masks and individual masks.

    Parameters
    ----------
    db : AICMECompartmentsDataBatch
        Input databatch containing fields:
        - ``context_obs_time``      : [B, Ic, Tc, 1]
        - ``target_obs_time``       : [B, It, Tt, 1]
        - ``context_rem_sim_time``  : [B, Ic, Trc, 1]
        - ``target_rem_sim_time``   : [B, It, Trt, 1]
        - ``context_obs_mask``      : [B, Ic, Tc]
        - ``target_obs_mask``       : [B, It, Tt]
        - ``context_rem_sim_mask``  : [B, Ic, Trc]
        - ``target_rem_sim_mask``   : [B, It, Trt]
        - ``mask_context_individuals`` : [B, Ic]
        - ``mask_target_individuals``  : [B, It]
    time_sources : sequence[str], optional
        Names of the time tensors to aggregate. Supported values are
        ``"context_obs_time"``, ``"target_obs_time"``,
        ``"context_rem_sim_time"``, and ``"target_rem_sim_time"``.
        Defaults to all sources, preserving the historical behavior.

    Returns
    -------
    times : TensorType["B", "Tdistinct_max", 1]
        Sorted unique time grid per batch (substance). Padded with zeros for
        shorter sequences.
    mask : TensorType["B", "Tdistinct_max"]
        Boolean mask indicating which entries in ``times`` are valid for each
        batch element.

    Examples
    --------
    >>> times, mask = gather_distinct_times_per_substance(db)
    >>> times.shape
    torch.Size([B, Tdistinct_max, 1])
    >>> mask.shape
    torch.Size([B, Tdistinct_max])
    """
    B = db.context_obs_time.size(0)

    allowed_sources = (
        "context_obs_time",
        "target_obs_time",
        "context_rem_sim_time",
        "target_rem_sim_time",
    )
    allowed_sources_set = set(allowed_sources)
    if time_sources is None:
        resolved_sources = allowed_sources
    else:
        resolved_sources = tuple(time_sources)
        invalid_sources = sorted(set(resolved_sources) - allowed_sources_set)
        if invalid_sources:
            raise ValueError(
                "`time_sources` contains unsupported entries: "
                f"{invalid_sources}. Supported values are {list(allowed_sources)}."
            )
        if len(resolved_sources) == 0:
            raise ValueError("`time_sources` must contain at least one source.")

    # Flatten per-substance times from all available streams.
    # Shapes:
    #   times_*: [B, I, T]
    #   mask_* : [B, I, T]
    #   ind_*  : [B, I] -> broadcast to [B, I, T]
    ctx_ind_mask = db.mask_context_individuals.bool()
    tgt_ind_mask = db.mask_target_individuals.bool()

    ctx_obs_times = db.context_obs_time.squeeze(-1)
    tgt_obs_times = db.target_obs_time.squeeze(-1)
    ctx_rem_times = db.context_rem_sim_time.squeeze(-1)
    tgt_rem_times = db.target_rem_sim_time.squeeze(-1)

    # Fill value used to push invalid entries to the right after sorting.
    big_val = torch.finfo(ctx_obs_times.dtype).max

    def _flatten_valid_times(
        times_3d: torch.Tensor, time_mask_3d: torch.Tensor, ind_mask_2d: torch.Tensor
    ) -> torch.Tensor:
        valid_mask = time_mask_3d.bool() & ind_mask_2d.unsqueeze(-1)
        return times_3d.masked_fill(~valid_mask, big_val).reshape(B, -1)

    all_chunks = {
        "context_obs_time": _flatten_valid_times(
            ctx_obs_times, db.context_obs_mask, ctx_ind_mask
        ),  # [B, Ic*Tc]
        "target_obs_time": _flatten_valid_times(
            tgt_obs_times, db.target_obs_mask, tgt_ind_mask
        ),  # [B, It*Tt]
        "context_rem_sim_time": _flatten_valid_times(
            ctx_rem_times, db.context_rem_sim_mask, ctx_ind_mask
        ),  # [B, Ic*Trc]
        "target_rem_sim_time": _flatten_valid_times(
            tgt_rem_times, db.target_rem_sim_mask, tgt_ind_mask
        ),  # [B, It*Trt]
    }
    chunks = [all_chunks[source] for source in resolved_sources]
    all_times = torch.cat(chunks, dim=1)  # [B, total_T]

    # Sort within each batch
    all_times, _ = torch.sort(all_times, dim=1)  # [B, total_T]

    # Mark unique entries (first always kept, then keep if diff != 0)
    diffs = torch.diff(all_times, dim=1)  # [B, total_T-1]
    is_new = torch.cat(
        [torch.ones(B, 1, dtype=torch.bool, device=all_times.device), diffs.ne(0)],
        dim=1,
    )  # [B, total_T]

    # Mask out +inf (from padding)
    valid = all_times < big_val
    is_new = is_new & valid

    # Gather unique times (already sorted, so we just need to compact)
    Tdistinct = is_new.sum(dim=1)  # [B]
    Tdistinct_max = int(Tdistinct.max().item())

    # Replace duplicates/padding with +inf, then sort
    masked_times = all_times.masked_fill(~is_new, big_val)
    masked_times, _ = torch.sort(masked_times, dim=1)

    # Take the first Tdistinct_max entries
    times = masked_times[:, :Tdistinct_max].unsqueeze(-1)  # [B, Tdistinct_max, 1]
    mask = times.squeeze(-1) < big_val

    # Ensure padded entries are explicitly zero
    times = times.masked_fill(~mask.unsqueeze(-1), 0.0)

    return times, mask


def ensure_tensor_or_empty(tensor: Optional[torch.Tensor], shape: Tuple[int, ...]) -> torch.Tensor:
    """Ensure a tensor has a valid value.

    Parameters
    ----------
    tensor:
        Optional tensor to validate.
    shape:
        Expected shape of the output tensor.

    Returns
    -------
    torch.Tensor
        ``tensor`` if provided, otherwise a zeros tensor of ``shape``.
    """

    # If there was no data, return a zero tensor of the correct shape
    if tensor is None:
        return torch.zeros(shape, dtype=torch.float32)  # all zeros
    else:
        return tensor


def ensure_mask_or_empty(mask: Optional[torch.Tensor], shape: Tuple[int, ...]) -> torch.Tensor:
    """Ensure a boolean mask has a valid value.

    Parameters
    ----------
    mask:
        Optional boolean mask.
    shape:
        Expected shape of the output mask.

    Returns
    -------
    torch.Tensor
        ``mask`` if provided, otherwise a zeros boolean tensor of ``shape``.
    """

    if mask is None:
        return torch.zeros(shape, dtype=torch.bool)  # all “invalid”
    else:
        return mask


def weighted_average(
    x: torch.Tensor, weights: Optional[torch.Tensor] = None, dim=None
) -> torch.Tensor:
    """
    Computes the weighted average of a given tensor across a given dim, masking
    values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.

    Parameters
    ----------
    x
        Input tensor, of which the average must be computed.
    weights
        Weights tensor, of the same shape as `x`.
    dim
        The dim along which to average `x`

    Returns
    -------
    Tensor:
        The tensor with values averaged along the specified `dim`.
    """
    if weights is not None:
        weighted_tensor = torch.where(weights != 0, x * weights, torch.zeros_like(x))
        sum_weights = torch.clamp(weights.sum(dim=dim) if dim else weights.sum(), min=1.0)
        return (weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()) / sum_weights
    else:
        return x.mean(dim=dim)


def split_individuals_tensor_batch(
    full_tensor_a: torch.Tensor,  # shape [I, ...]
    full_tensor_b: torch.Tensor,  # shape [I, ...]
    full_tensor_c: Optional[torch.Tensor] = None,  # optional [I, ...]
    n_of_target_individuals: int = 0,
    seed: Optional[int] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],  # context
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],  # target
]:
    """
    Splits input tensors along the first dimension [I, ...] into context and target sets.

    Returns:
        context_a, context_b, context_c, target_a, target_b, target_c
    """
    num_individuals = full_tensor_a.shape[0]
    assert full_tensor_b.shape[0] == num_individuals
    if full_tensor_c is not None:
        assert full_tensor_c.shape[0] == num_individuals

    if seed is not None:
        random.seed(seed)

    if n_of_target_individuals == 0:
        return full_tensor_a, full_tensor_b, full_tensor_c, None, None, None

    # select random target indices
    all_indices = list(range(num_individuals))
    target_indices = random.sample(all_indices, n_of_target_individuals)
    context_indices = [i for i in all_indices if i not in target_indices]

    context_a = full_tensor_a[context_indices]
    context_b = full_tensor_b[context_indices]
    context_c = full_tensor_c[context_indices] if full_tensor_c is not None else None

    target_a = full_tensor_a[target_indices]
    target_b = full_tensor_b[target_indices]
    target_c = full_tensor_c[target_indices] if full_tensor_c is not None else None

    return context_a, context_b, context_c, target_a, target_b, target_c
