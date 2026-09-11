import matplotlib.pyplot as plt
import numpy as np
import torch
from torchtyping import TensorType

import torch
from torchtyping import TensorType
from typing import List,Tuple,Optional
import numpy as np

from pff.data.data_preprocessing.raw_to_tensors_bundles import substance_cvs_to_tensors_bundle,substances_csv_to_tensors

from typing import NamedTuple
import torch
from torchtyping import TensorType

class SubstanceTensorGroup(NamedTuple):
    observations: TensorType[1, "I", "T"]
    times: TensorType[1, "I", "T"]
    mask: TensorType[1, "I", "T"]
    subject_mask: TensorType[1, "I"]

def apply_timescale_filter(
    observations:    TensorType["S", "I", "T"],
    times:           TensorType["S", "I", "T"],
    masks:           TensorType["S", "I", "T"],
    subject_mask:    TensorType["S", "I"],
    *,
    strategy: str = "log_zscore",   # "log_zscore" | "median_fraction" | "none"
    max_abs_z: float = 2.0,         # for "log_zscore"
    tau: float = 0.4,               # for "median_fraction"  (≈ ln 1.5)
) -> Tuple[
    TensorType["S", "I", "T"],  # filtered observations
    TensorType["S", "I", "T"],  # filtered times
    TensorType["S", "I", "T"],  # filtered masks
    TensorType["S", "I"],       # filtered subject_mask
]:
    """
    Zeroes‑out and un‑masks subjects whose time‑span is an outlier
    w.r.t. other subjects in the *same* substance.

    •  strategy="log_zscore": keep subjects with |z| ≤ max_abs_z in log‑span  
    •  strategy="median_fraction": keep subjects within ±tau of median(log‑span)  
    •  strategy="none": return inputs unchanged
    """
    if strategy == "none":
        return observations, times, masks, subject_mask

    # combine padding + subject mask to know valid time points
    valid = masks.bool() & subject_mask.unsqueeze(-1)

    # --- compute log‑spans ----------------------------------------------------
    t_max = times.masked_fill(~valid, float("-inf")).max(dim=2).values  # [S, I]
    t_min = times.masked_fill(~valid, float("inf")).min(dim=2).values   # [S, I]
    span  = (t_max - t_min).clamp(min=1e-12)
    log_span = span.log()                                               # [S, I]

    # --- decide which subjects to keep ---------------------------------------
    if strategy == "log_zscore":
        z      = (log_span - log_span.mean(dim=1, keepdim=True)) / \
                 (log_span.std(dim=1, keepdim=True).clamp(min=1e-6))
        keep = torch.abs(z) <= max_abs_z                                # [S, I]

    elif strategy == "median_fraction":
        med   = log_span.median(dim=1, keepdim=True).values            # [S,1]
        keep  = (log_span >= med - tau) & (log_span <= med + tau)      # [S,I]

    else:
        # No filtering applied — return inputs unchanged
        return observations, times, masks, subject_mask

    # --- apply filter: zero & un‑mask ----------------------------------------
    # clone so we don't mutate original tensors accidentally
    obs_f   = observations.clone()
    times_f = times.clone()
    masks_f = masks.clone()
    subj_f  = subject_mask.clone()

    # indices where we drop subjects
    drop = ~keep & subj_f.bool()
    subj_f[drop] = False
    masks_f[drop] = False
    obs_f[drop] = 0.0
    times_f[drop] = 0.0

    return obs_f, times_f, masks_f, subj_f

def plot_subjects_for_substance(
    drug_data_frame,
    substance_label: str,
    *,
    z_score_normalization: bool = False,
    normalize_by_max:bool = False,
    time_strategy:str="log_zscore",   # "log_zscore" | "median_fraction" | "none"
    max_abs_z:float=2.,
    x_scale: str = "linear",          # "linear" ▸ default  ·  "log"
    y_scale: str = "linear",          # "linear" ▸ default  ·  "log"
    alpha: float = 1.0,               # 0 ≤ alpha ≤ 1
    legend_outside: bool = True,      # park legend to the right
    figsize: Tuple[float, float] = (10, 5),  # default width × height
    save_dir: Optional[str] = None,  # if set, saves the figure here

) -> None:
    """
    Draw every subject‑trajectory (points + line) for *one* substance.

    Parameters
    ----------
    drug_data_frame : pandas.DataFrame
    substance_label : str
    z_score_normalization : bool, optional
    x_scale, y_scale : {"linear", "log"}, optional
        Axis scaling.  If you pick "log", make sure data are strictly > 0
        on that axis or Matplotlib will complain.
    alpha : float in [0, 1], optional
        Transparency applied to both the line and the markers.
    legend_outside : bool, optional
        True ⇢ legend in a separate column to the right;
        False ⇢ legend inside plot.
    """
    # ── 1.  Pull tensors ────────────────────────────────────────────
    data_bundle = substance_cvs_to_tensors_bundle(drug_data_frame,normalize_by_max=True)

    all_obs      = data_bundle.observations         # [S, I, T]
    all_times   = data_bundle.times                # [S, I, T]
    all_masks    = data_bundle.masks                # [S, I, T]
    all_subj_mask = data_bundle.individuals_mask
    substance_labels      = data_bundle.substance_names     # [S]
    mapping               = data_bundle.mapping
    study_names           = data_bundle.study_names          # [S]
    subject_names         = data_bundle.individuals_names        # [S][I]
    empirical_loaded      = True

    # ── 2.  Find substance row ──────────────────────────────────────
    try:
        s_idx: int = int(np.where(substance_labels == substance_label)[0][0])
    except IndexError:
        raise ValueError(f"Substance '{substance_label}' not found.")

    # ("I", "T")
    obs:        TensorType["I", "T"] = all_obs[s_idx]
    times:      TensorType["I", "T"] = all_times[s_idx]
    step_mask:  TensorType["I", "T"] = all_masks[s_idx].bool()
    subj_mask:  TensorType["I"]      = all_subj_mask[s_idx].bool()

    # ── 3.  Filter Time Series  ──────────────────────────────────────
    # Add batch dimension to match expected input [S, I, T], [S, I]
    obs_b       = obs.unsqueeze(0)         # [1, I, T]
    times_b     = times.unsqueeze(0)       # [1, I, T]
    step_mask_b = step_mask.unsqueeze(0)   # [1, I]
    subj_mask_b = subj_mask.unsqueeze(0)   # [1, I]

    # Apply timescale filter (choose one strategy)
    obs_b, times_b, step_mask_b, subj_mask_b = apply_timescale_filter(
        observations=obs_b,
        times=times_b,
        masks=step_mask_b,
        subject_mask=subj_mask_b,
        strategy=time_strategy,   # or "median_fraction"
        max_abs_z=max_abs_z,
        tau=0.4,
    )

    # Remove batch dim again
    obs = obs_b[0]
    times = times_b[0]
    step_mask = step_mask_b[0]
    subj_mask = subj_mask_b[0]


    # ── 4.  Plot one line per *real* subject ────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(obs.shape[0]):                # iterate subjects (I)
        if not subj_mask[i]:
            continue                             # skip padded rows

        valid: TensorType["T"] = step_mask[i]    # True ⇢ real sample
        t: TensorType["T"] = times[i][valid].cpu()
        y: TensorType["T"] = obs[i][valid].cpu()

        ax.plot(t, y, marker="o", alpha=alpha, label=f"subject {i}")

    # ── 5.  Styling ────────────────────────────────────────────────
    ax.set_title(f"All subjects – {substance_label}")
    ax.set_xlabel("Time (normalised per substance)")
    ax.set_ylabel("Observation")

    # Axis scales
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)

    # Legend placement
    if legend_outside:
        # ncol=1 ▸ vertical list; bbox_to_anchor shifts legend fully outside
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            frameon=False,
        )
        plt.tight_layout(rect=[0, 0, 0.82, 1])   # leave room on the right
    else:
        ax.legend(frameon=False)
        plt.tight_layout()

    # Save figure if path is given
    if save_dir is not None:
        from pathlib import Path
        study_name = mapping[substance_label]["study_name"]
        index = mapping[substance_label]["index"]
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        filename = f"{study_name}_{substance_label}_{index}.png"
        filepath = Path(save_dir) / filename
        fig.savefig(filepath, bbox_inches="tight", dpi=300)

    plt.show()

def substances_with_min_timesteps(
    drug_data_frame,
    min_timesteps: int = 140,
    *,
    z_score_normalization: bool = False,
    normalize_by_max:bool = False,
) -> List[str]:
    """
    Return the list of substance labels whose **best** subject has
    ≥ `min_timesteps` valid observations.

    Parameters
    ----------
    drug_data_frame : pandas.DataFrame
        Same dataframe you already pass to `substance_cvs_to_tensors_from_list`.
    min_timesteps : int, default = 140
        Threshold on the number of valid (unpadded) time‑points.
    z_score_normalization : bool, default = False
        Passed straight through to `substance_cvs_to_tensors_from_list`.

    Returns
    -------
    List[str]
        Substance strings that satisfy the criterion.
    """
    (
        all_observations,            # TensorType["S", "I", "T"]   – concentration values
        all_times,                   # TensorType["S", "I", "T"]   – time grid (0‥1)
        all_masks,                   # TensorType["S", "I", "T"]   – bool, 1 = real step
        all_subjects_mask,           # TensorType["S", "I"]        – bool, 1 = real subject
        substance_labels,            # np.ndarray, shape ["S"]
        mapping
    ) = substance_cvs_to_tensors_bundle(
        drug_data_frame,
        z_score_normalization=z_score_normalization,
        normalize_by_max=normalize_by_max
    )

    # --- Shapes -------------------------------------------------------
    # S = number of substances, I = max subjects per substance,
    # T = max time‑steps per subject.
    # all_masks       : (S, I, T) – True at valid positions
    # all_subjects_mask: (S, I)   – True for *existing* subjects only
    # -----------------------------------------------------------------

    # Convert to bool & mask out padded subjects
    valid_masks: TensorType["S", "I", "T"] = all_masks.bool()
    subj_mask:   TensorType["S", "I", 1]   = all_subjects_mask.bool().unsqueeze(-1)
    valid_masks = valid_masks & subj_mask          # shape keeps (S,I,T)

    # Count valid steps per subject ───────────────────────────────────
    # counts[s, i] = #valid time‑points of subject i in substance s
    counts: TensorType["S", "I"] = valid_masks.sum(dim=2)  # (S, I)

    # Max over subjects (per substance) -------------------------------
    max_counts: TensorType["S"] = counts.max(dim=1).values  # (S,)

    # Pick substances that meet / beat the threshold ------------------
    qualifying: TensorType["S"] = max_counts >= min_timesteps  # (S,)

    # Build the output list -------------------------------------------
    return [label for label, keep in zip(substance_labels.tolist(), qualifying.tolist()) if keep]

def get_substance_tensors_by_label(
    drug_data_frame,
    substance_label: str,
    *,
    z_score_normalization: bool = False,
    normalize_by_max: bool = False,
) -> SubstanceTensorGroup:
    """
    Returns tensors for a selected substance, preserving S=1 batch shape.

    Shapes:
        observations : [1, I, T]
        times        : [1, I, T]
        mask         : [1, I, T]
        subject_mask : [1, I]
    """
    data_bundle = substance_cvs_to_tensors_bundle(drug_data_frame,
                                                     z_score_normalization=z_score_normalization,
                                                     normalize_by_max=normalize_by_max)

    all_observations      = data_bundle.observations         # [S, I, T]
    all_empirical_times   = data_bundle.times                # [S, I, T]
    all_empirical_mask    = data_bundle.masks                # [S, I, T]
    all_subjects_mask     = data_bundle.individuals_mask
    substance_labels      = data_bundle.substance_names     # [S]
    mapping               = data_bundle.mapping

    # Lookup index
    label_to_index = {label: idx for idx, label in enumerate(substance_labels)}
    if substance_label not in label_to_index:
        raise ValueError(f"Substance label '{substance_label}' not found.")
    s_idx = label_to_index[substance_label]

    # Add batch dim: [1, I, T] or [1, I]
    return SubstanceTensorGroup(
        observations=all_observations[s_idx].unsqueeze(0),     # [1, I, T]
        times=all_empirical_times[s_idx].unsqueeze(0),                   # [1, I, T]
        mask=all_empirical_mask[s_idx].unsqueeze(0).bool(),             # [1, I, T]
        subject_mask=all_subjects_mask[s_idx].unsqueeze(0).bool()  # [1, I]
    )

