from typing import Tuple

import torch
from torchtyping import TensorType


def compute_predictive_quantiles(
    pred_values: TensorType["B", "S", "T", 1],
    pred_mask: TensorType["B", "S", "T"] | TensorType["B", "T"],
    alpha: float,
) -> Tuple[
    TensorType["B", "T", 1],
    TensorType["B", "T", 1],
]:
    """
    Compute lower and upper predictive quantiles (α/2, 1−α/2)
    across stochastic samples, supporting both shared and
    per-sample (per-individual) masks.

    Parameters
    ----------
    pred_values : [B, S, T, 1]
        Predicted sample trajectories.
    pred_mask : [B, T] or [B, S, T]
        Boolean mask marking valid time points.
        - If [B, T]: same mask for all samples.
        - If [B, S, T]: individual-specific masks.
    alpha : float
        Significance level (e.g. 0.05 for 90% interval).

    Returns
    -------
    q_low, q_high : [B, T, 1]
        Predictive lower and upper quantile envelopes.
    """
    B, S, T, _ = pred_values.shape
    device = pred_values.device

    # --- normalize mask shape ---
    if pred_mask.ndim == 2:
        pred_mask = pred_mask.unsqueeze(1).repeat(1, S, 1)  # [B,S,T]

    q_low_list = []
    q_high_list = []

    for b in range(B):
        q_low_b = torch.zeros(T, device=device)
        q_high_b = torch.zeros(T, device=device)

        # for each time index, only include valid samples
        for t_idx in range(T):
            valid_s = pred_mask[b, :, t_idx]
            if valid_s.any():
                vals = pred_values[b, valid_s, t_idx, 0]
                q_low_b[t_idx] = vals.quantile(alpha / 2)
                q_high_b[t_idx] = vals.quantile(1 - alpha / 2)
            else:
                # leave zeros (or NaN if preferred)
                q_low_b[t_idx] = 0.0
                q_high_b[t_idx] = 0.0

        q_low_list.append(q_low_b.unsqueeze(-1))
        q_high_list.append(q_high_b.unsqueeze(-1))

    q_low = torch.stack(q_low_list, dim=0)  # [B,T,1]
    q_high = torch.stack(q_high_list, dim=0)  # [B,T,1]

    return q_low, q_high


def interpolate_quantiles_to_obs_times(
    q_low: TensorType["B", "Tpred", 1],
    q_high: TensorType["B", "Tpred", 1],
    pred_times: TensorType["B", "Tpred", 1],
    pred_mask: TensorType["B", "Tpred"],
    real_times: TensorType["B", "I", "Treal", 1],
    real_mask: TensorType["B", "I", "Treal"],
) -> Tuple[
    TensorType["B", "I", "Treal", 1],
    TensorType["B", "I", "Treal", 1],
]:
    """
    Interpolate predictive quantile bands (q_low, q_high) to the irregular
    observation times of real data.

    Parameters
    ----------
    q_low, q_high : TensorType["B", "Tpred", 1]
        Predictive lower and upper quantile curves at distinct time grid points.
    pred_times : TensorType["B", "Tpred", 1]
        Time grid corresponding to the quantile curves.
    pred_mask : TensorType["B", "Tpred"]
        Boolean mask marking valid predictive times per batch.
    real_times : TensorType["B", "I", "Treal", 1]
        Observation times for each individual and batch.
    real_mask : TensorType["B", "I", "Treal"]
        Mask indicating valid observed time points.

    Returns
    -------
    q_low_interp, q_high_interp : Tuple[
        TensorType["B", "I", "Treal", 1],
        TensorType["B", "I", "Treal", 1],
    ]
        Interpolated quantile band values at each observed time, padded
        where invalid.

    Notes
    -----
    - Uses linear interpolation between nearest predictive time knots.
    - Out-of-range times are clamped to the nearest boundary quantile.
    - Invalid (masked) observations are returned as zeros.
    """

    B, I, Treal, _ = real_times.shape
    device = real_times.device

    q_low_interp_list, q_high_interp_list = [], []

    for b in range(B):
        # Extract valid predictive points for this batch
        valid_mask_b = pred_mask[b]  # [Tpred]
        valid_T = valid_mask_b.sum().item()
        if valid_T < 2:
            # Degenerate case: not enough points for interpolation
            q_low_interp_list.append(torch.zeros(I, Treal, 1, device=device))
            q_high_interp_list.append(torch.zeros(I, Treal, 1, device=device))
            continue

        t_pred = pred_times[b, valid_mask_b, 0]  # [T_b]
        ql = q_low[b, valid_mask_b, 0]  # [T_b]
        qh = q_high[b, valid_mask_b, 0]  # [T_b]

        # For each individual, interpolate its observation times
        q_low_i, q_high_i = [], []
        for i in range(I):
            t_obs = real_times[b, i, :, 0]  # [Treal]
            valid_obs = real_mask[b, i]  # [Treal]

            # Clamp obs times into predictive range
            t_clamped = torch.clamp(t_obs, t_pred.min(), t_pred.max())

            # Use searchsorted to find bracketing indices
            idx_right = torch.searchsorted(t_pred, t_clamped)
            idx_left = (idx_right - 1).clamp(min=0)
            idx_right = idx_right.clamp(max=valid_T - 1)

            # Gather times and values for interpolation
            t_L, t_R = t_pred[idx_left], t_pred[idx_right]
            ql_L, ql_R = ql[idx_left], ql[idx_right]
            qh_L, qh_R = qh[idx_left], qh[idx_right]

            denom = (t_R - t_L).clamp(min=1e-8)
            w_R = (t_clamped - t_L) / denom
            w_L = 1.0 - w_R

            ql_interp = w_L * ql_L + w_R * ql_R
            qh_interp = w_L * qh_L + w_R * qh_R

            # Zero out invalid times
            ql_interp = ql_interp.masked_fill(~valid_obs, 0.0)
            qh_interp = qh_interp.masked_fill(~valid_obs, 0.0)

            q_low_i.append(ql_interp.unsqueeze(-1))
            q_high_i.append(qh_interp.unsqueeze(-1))

        q_low_interp_list.append(torch.stack(q_low_i, dim=0))  # [I, Treal, 1]
        q_high_interp_list.append(torch.stack(q_high_i, dim=0))  # [I, Treal, 1]

    q_low_interp = torch.stack(q_low_interp_list, dim=0)  # [B, I, Treal, 1]
    q_high_interp = torch.stack(q_high_interp_list, dim=0)  # [B, I, Treal, 1]

    return q_low_interp, q_high_interp


def compute_time_weighted_coverage(
    real_values: TensorType["B", "I", "Treal", 1],
    real_times: TensorType["B", "I", "Treal", 1],
    real_mask: TensorType["B", "I", "Treal"],
    q_low_interp: TensorType["B", "I", "Treal", 1],
    q_high_interp: TensorType["B", "I", "Treal", 1],
    reduce: bool = True,
) -> TensorType["B"]:
    """
    Compute time-weighted coverage fraction of observations within predictive bands.
    """
    # [B, I, Treal, 1]
    covered = (real_values >= q_low_interp) & (real_values <= q_high_interp)
    covered = covered.squeeze(-1) & real_mask  # [B, I, Treal]

    # Compute Δt (difference along time)
    dt = torch.diff(real_times, dim=2, prepend=real_times[:, :, :1])
    dt = dt.squeeze(-1) * real_mask  # [B, I, Treal]
    dt_sum = dt.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)
    weights = dt / dt_sum  # normalized time weights

    coverage = (covered.float() * weights).sum(dim=(1, 2))  # [B]
    return coverage


def compute_interval_score(
    real_values: TensorType["B", "I", "Treal", 1],
    real_times: TensorType["B", "I", "Treal", 1],
    real_mask: TensorType["B", "I", "Treal"],
    q_low_interp: TensorType["B", "I", "Treal", 1],
    q_high_interp: TensorType["B", "I", "Treal", 1],
    alpha: float,
) -> TensorType["B"]:
    """
    Compute the time-weighted interval score (Gneiting & Raftery, 2007).
    """
    width = (q_high_interp - q_low_interp).abs()
    below = (q_low_interp - real_values).clamp(min=0)
    above = (real_values - q_high_interp).clamp(min=0)

    interval_score = width + (2 / alpha) * (below + above)
    interval_score = interval_score.squeeze(-1) * real_mask  # [B, I, Treal]

    # Δt weighting
    dt = torch.diff(real_times, dim=2, prepend=real_times[:, :, :1]).squeeze(-1)
    dt = dt * real_mask
    dt_sum = dt.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)
    weights = dt / dt_sum

    # Weighted mean per batch
    score_weighted = (interval_score * weights).sum(dim=(1, 2))
    return score_weighted


def compute_percentile_coverage(
    pred_values,
    pred_times,
    pred_mask,
    real_values,
    real_times,
    real_mask,
    alpha: float = 0.05,
):
    """
    Compute predictive interval coverage and interval score between predicted and observed trajectories.

    This function evaluates how well a stochastic predictive model captures
    the true (real) observations within its predictive uncertainty bands.

    It combines three subroutines:
        1. :func:`compute_predictive_quantiles` — compute lower and upper predictive quantiles.
        2. :func:`interpolate_quantiles_to_obs_times` — align quantile predictions to observation times.
        3. :func:`compute_time_weighted_coverage` and :func:`compute_interval_score` —
           compute Δt-weighted coverage fraction and proper scoring rule.

    Parameters
    ----------
    pred_values : TensorType["B", "S", "T_pred", 1]
        Stochastic predictions for each batch element `B` and stochastic sample `S`.
        Typically obtained by sampling the model multiple times.

    pred_times : TensorType["B", "T_pred", 1]
        Distinct prediction time grid per batch (shared across stochastic samples).

    pred_mask : TensorType["B", "T_pred"]
        Boolean mask indicating valid prediction time steps.

    real_values : TensorType["B", "I", "T_real", 1]
        Ground-truth or observed values for each batch and individual.

    real_times : TensorType["B", "I", "T_real", 1]
        Observation times corresponding to `real_values`.

    real_mask : TensorType["B", "I", "T_real"]
        Boolean mask indicating valid observed time points.

    alpha : float, optional (default = 0.05)
        Significance level defining the predictive interval width.
        For example:
            * α = 0.05 → 90% central interval (quantiles 0.025 and 0.975)
            * α = 0.10 → 80% central interval (quantiles 0.05 and 0.95)
        Smaller α yields wider intervals (more conservative coverage).

    Returns
    -------
    dict[str, TensorType["B"]]
        Dictionary containing:
            - ``"coverage"`` : Δt-weighted fraction of observations inside the predictive interval.
            - ``"interval_score"`` : Proper interval score (Gneiting & Raftery, 2007),
              penalizing both interval width and miscoverage.

    Notes
    -----
    - High coverage (≈1.0) indicates all real points lie inside the predictive band.
      In well-calibrated models, expected coverage ≈ 1−α.
    - Lower interval scores correspond to sharper and better-calibrated predictions.

    References
    ----------
    Gneiting, T. & Raftery, A. E. (2007). *Strictly Proper Scoring Rules, Prediction, and Estimation*.
    Journal of the American Statistical Association, 102(477), 359-378.
    """
    q_low, q_high = compute_predictive_quantiles(pred_values, pred_mask, alpha)
    q_low_interp, q_high_interp = interpolate_quantiles_to_obs_times(
        q_low, q_high, pred_times, pred_mask, real_times, real_mask
    )

    coverage = compute_time_weighted_coverage(
        real_values, real_times, real_mask, q_low_interp, q_high_interp
    )
    interval_score = compute_interval_score(
        real_values, real_times, real_mask, q_low_interp, q_high_interp, alpha
    )

    return {"coverage": coverage, "interval_score": interval_score}
