from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torchtyping import TensorType

# ============================================================
# 🧩 1. Synthetic Gaussian Band Sampler (regular or irregular)
# ============================================================


def sample_dummy_band(
    n_samples: int = 100,
    n_points: int = 80,
    t_min: float = 0.0,
    t_max: float = 24.0,
    noise_scale: float = 0.05,
    band_scale: float = 0.2,
    baseline: float = 0.05,
    irregular: bool = False,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Generate PK-like decaying trajectories with a baseline offset and
    per-individual level variability.

    Each trajectory:
        y_i(t) = (1 + δ_i) * (f(t) + b) + ε_i

    Parameters
    ----------
    n_samples : int
        Number of trajectories (individuals).
    n_points : int
        Number of time points per trajectory.
    t_min, t_max : float
        Range of the time grid (e.g., hours).
    noise_scale : float
        Standard deviation of additive noise ε_i.
    band_scale : float
        Std. dev. of multiplicative inter-individual variability δ_i.
    baseline : float
        Constant baseline offset b added to f(t) before scaling.
    irregular : bool
        If True, randomly remove ~25% of points to mimic irregular sampling.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, torch.Tensor]
        {
            "t": [T, 1],       # time grid (possibly irregular)
            "ref": [T, 1],     # reference curve f(t)
            "samples": [S, T, 1],
            "mask": [T]        # valid time mask
        }
    """
    torch.manual_seed(seed)

    # Time grid
    t = torch.linspace(t_min, t_max, n_points).unsqueeze(-1)
    mask = torch.ones(n_points, dtype=torch.bool)
    if irregular:
        keep = torch.rand(n_points) > 0.25
        t = t[keep]
        mask = keep
        n_points = t.size(0)

    # Bi-exponential reference decay
    A, B = 5.0, 1.0
    k1, k2 = 1.0, 0.1
    f_t = A * torch.exp(-k1 * t) + B * torch.exp(-k2 * t)  # [T,1]

    # Random effects
    delta = band_scale * torch.randn(n_samples, 1, 1)  # multiplicative
    eps = noise_scale * torch.randn(n_samples, 1, 1)  # additive (constant)

    # Generate samples
    samples = (1 + delta) * (f_t.unsqueeze(0) + baseline) + eps

    return {"t": t, "ref": f_t, "samples": samples, "mask": mask}


def sample_dummy_band_simple(
    n_samples: int = 100,
    n_points: int = 20,
    t_min: float = 0.0,
    t_max: float = 24.0,
    band_scale: float = 0.2,
    baseline: float = 0.05,
    irregular: bool = False,
    mask_diff_per_ind: bool = False,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Generate smooth PK-like decaying trajectories with optional irregular masking.

    If ``mask_diff_per_ind=True`` and ``irregular=True``,
    each individual receives a mask with a *different sequence length*
    drawn uniformly from [T/2, T], right-padded with zeros.

    Example
    -------
    If n_points = 20, individuals may have 10–20 valid observations.
    All tensors remain size [S, T, 1] or [S, T].
    """
    torch.manual_seed(seed)

    # --- shared time grid ---
    t: TensorType["T", 1] = torch.linspace(t_min, t_max, n_points).unsqueeze(-1)  # [T,1]

    # --- reference function ---
    A, B, k1, k2 = 5.0, 1.0, 1.0, 0.1
    f_t: TensorType["T", 1] = A * torch.exp(-k1 * t) + B * torch.exp(-k2 * t)
    f_t = f_t / f_t.max()  # normalize to [0,1]
    ref: TensorType["T", 1] = f_t + baseline

    # --- multiplicative & additive random effects ---
    delta: TensorType["S", 1, 1] = torch.randn(n_samples, 1, 1) * band_scale
    bias: TensorType["S", 1, 1] = torch.randn(n_samples, 1, 1) * 0.05 * band_scale
    samples: TensorType["S", "T", 1] = (1 + delta) * f_t.unsqueeze(0) + baseline + bias

    # --- mask logic ---
    if not irregular:
        mask: TensorType["S", "T"] = torch.ones(n_samples, n_points, dtype=torch.bool)

    elif irregular and not mask_diff_per_ind:
        # Same irregular pattern for everyone (≈50% kept)
        keep_prob = 0.5
        mask = (torch.rand(n_points) < keep_prob).unsqueeze(0).repeat(n_samples, 1)

    else:
        # --- Per-individual variable-length masks (right-padded) ---
        lengths = torch.randint(
            low=n_points // 2, high=n_points + 1, size=(n_samples,)
        )  # [S] random valid lengths
        mask = torch.zeros(n_samples, n_points, dtype=torch.bool)
        for i, L in enumerate(lengths):
            mask[i, :L] = True  # right padding: valid first, zeros at end

        # Optional: small random irregular dropouts within valid region
        dropout_prob = 0.5
        drop_mask = (torch.rand_like(mask.float()) < dropout_prob) & mask
        mask = mask & ~drop_mask

    return {"t": t, "ref": ref, "samples": samples, "mask": mask}


def _plot_band_on_axis(
    ax: plt.Axes,
    t: torch.Tensor,  # [T,1]
    samples: torch.Tensor,  # [S,T,1]
    ref: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,  # [S,T] or [T]
    alpha: float = 0.05,
    label: str = "Sample",
    color: str = "tab:blue",
    log_y: bool = False,
    show_samples: bool = True,
    sample_alpha: float = 0.2,
    show_points: bool = True,
) -> None:
    """
    ✅ Simplest possible version:
    - Each individual drawn only at its valid points.
    - Quantile band computed only from valid points.
    """
    from pff.metrics.quantiles_coverage import compute_predictive_quantiles

    S, T, _ = samples.shape
    device = samples.device

    # Default: everything valid
    if mask is None:
        mask = torch.ones(S, T, dtype=torch.bool, device=device)
    elif mask.ndim == 1:
        mask = mask.unsqueeze(0).repeat(S, 1)

    # Compute quantiles (your updated function handles per-sample masks)
    q_low, q_high = compute_predictive_quantiles(
        samples.unsqueeze(0),  # [1,S,T,1]
        pred_mask=mask.unsqueeze(0),  # [1,S,T]
        alpha=alpha,
    )

    # --- Draw each trajectory only at its valid times ---
    if show_samples:
        for s in range(S):
            valid_t = t[mask[s]].squeeze()
            valid_y = samples[s, mask[s], 0]
            ax.plot(valid_t, valid_y, color=color, lw=0.6, alpha=sample_alpha)
            if show_points:
                ax.scatter(valid_t, valid_y, s=8, color=color, alpha=0.3, edgecolors="none")

    # --- Compute mean over valid region (rough approximation) ---
    valid_any = mask.any(dim=0)
    mean_valid = samples[:, valid_any, 0].mean(dim=0)
    ql_valid = q_low[0, valid_any, 0]
    qh_valid = q_high[0, valid_any, 0]
    valid_t = t[valid_any, 0]

    # --- Overlay quantile band and mean ---
    ax.plot(valid_t, mean_valid, color=color, lw=1.5, label=f"{label} mean")
    ax.fill_between(
        valid_t,
        ql_valid,
        qh_valid,
        color=color,
        alpha=0.25,
        label=f"{int((1 - alpha) * 100)}% band",
    )

    if log_y:
        ax.set_yscale("log")
        ax.set_ylabel("Value (log)")
    else:
        ax.set_ylabel("Value")
    ax.grid(alpha=0.3)


def plot_dummy_band(
    data: Dict[str, torch.Tensor],
    alpha: float = 0.05,
    title: str = "Dummy Band",
    save_dir: Optional[Path] = None,
    show: bool = True,
    log_y: bool = False,
    show_samples: bool = True,
) -> Optional[Path]:
    """Plot a single simulated band with mean, quantile shading, and raw samples."""
    t, ref, samples, mask = data["t"], data["ref"], data["samples"], data.get("mask", None)

    fig, ax = plt.subplots(figsize=(6, 4))
    _plot_band_on_axis(
        ax,
        t,
        samples,
        ref,
        mask=mask,  # ✅ direct mask (can be [S,T])
        alpha=alpha,
        label="Sample",
        log_y=log_y,
        show_samples=show_samples,
    )
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.legend(frameon=False)

    save_path = None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{title.replace(' ', '_')}.png"
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"✅ Saved plot at: {save_path}")

    if show:
        plt.show()
    plt.close(fig)
    return save_path


def compare_bands_plot(
    data_left: Dict[str, torch.Tensor],
    data_right: Dict[str, torch.Tensor],
    alpha: float = 0.05,
    labels: Tuple[str, str] = ("Sample", "Reference"),
    save_dir: Optional[Path] = None,
    show: bool = True,
    log_y: bool = True,
) -> Optional[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, data, label in zip(axes, [data_left, data_right], labels):
        t, ref, samples, mask = data["t"], data["ref"], data["samples"], data.get("mask", None)

        # ❌ remove this collapsing logic
        # if mask is not None and mask.ndim == 2:
        #     mask_combined = mask.any(dim=0)
        # else:
        #     mask_combined = mask

        # ✅ pass full per-individual mask directly
        _plot_band_on_axis(
            ax,
            t,
            samples,
            ref,
            mask=mask,  # keep shape [S, T]
            alpha=alpha,
            label=label,
            log_y=log_y,
        )
        ax.set_title(label)
        ax.set_xlabel("Time")
        ax.legend(frameon=False)

    plt.tight_layout()

    save_path = None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = f"comparison_{labels[0]}_{labels[1]}{'_log' if log_y else ''}.png"
        save_path = save_dir / fname
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"✅ Saved comparison plot at: {save_path}")

    if show:
        plt.show()
    plt.close(fig)
    return save_path
