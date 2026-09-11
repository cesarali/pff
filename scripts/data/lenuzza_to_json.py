"""Utilities for converting the Lenuzza CSV to canonical JSON.

The resulting JSON conforms to :mod:`pff.data.data_empirical.json_schema`
and can be fed directly into :class:`EmpiricalJSONBuilder`.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Literal, Tuple
import json
import numpy as np
import pandas as pd
from pathlib import Path

from pff.data.data_empirical.json_schema import canonicalize_study

# --- Dose dictionary (mg per g) ---
lenuzza_doses_mg_per_g: Dict[str, float] = {
    "memantine": 0.005,
    "omeprazole": 0.010,
    "repaglinide": 0.00025,
    "rosuvastatin": 0.005,
    "tolbutamide": 0.010,
    "dextromethorphan": 0.018,
    "digoxin": 0.00025,
    "paracetamol": 0.060,
    "caffeine": 0.073,
    "midazolam": 0.004,
    "paraxanthine": 0.073,
    "dextrorphan": 0.018,
}

def _infer_dose_for_substance(substance_name: str) -> float:
    s = (substance_name or "").lower()
    for k, v in lenuzza_doses_mg_per_g.items():
        if k in s:
            return v
    # fallback if not matched
    return 0.5

def _build_individual_json(
    subject_name: str,
    subdf: pd.DataFrame,
    *,
    normalize_time: bool,
    dosing_route_default: str,
) -> Dict:
    """
    subdf: rows for a single subject within a single (study_name, substance_label).
           Must contain columns: 'time', 'value'.
           Optional: 'dosing_time', 'dosing_value', 'dosing_type', 'substance_name'
    """
    # Sort and sanitize observations
    g = subdf.sort_values("time")
    t = g["time"].astype(float).to_numpy()
    y = g["value"].astype(float).to_numpy()

    # Drop NaNs consistently (on either, mask both)
    valid = np.isfinite(t) & np.isfinite(y)
    t, y = t[valid], y[valid]

    # Optional per-subject normalization [0,1] (no time shift if degenerate)
    if normalize_time and len(t) > 0:
        tmin, tmax = float(np.min(t)), float(np.max(t))
        if tmax > tmin:
            t_norm = (t - tmin) / (tmax - tmin)
        else:
            t_norm = np.zeros_like(t)
        obs_times_out = t_norm.tolist()
    else:
        obs_times_out = t.tolist()

    # Dosing info: prefer explicit columns if present, otherwise infer single dose @ t=0
    if {"dosing_times", "dosing", "dosing_type", "dosing_name"}.issubset(set(g.columns)):
        # Already denormalized fields present (rare). Expect list-likes per row; collapse.
        # If not present as lists, we’ll reconstruct from scalar columns below.
        raise ValueError(
            "Found dosing_* wide columns unexpectedly; this loader expects tidy rows. "
            "Provide columns 'dosing_time', 'dosing_value', 'dosing_type', 'substance_name' per dose event, "
            "or omit them and we’ll infer a single dose."
        )

    # If tidy dosing columns exist, collect; otherwise infer one oral dose at t=0
    dosing_times, dosing_vals, dosing_types, dosing_names = [], [], [], []
    has_tidy_dose_cols = all(c in g.columns for c in ["dosing_time", "dosing_value"])
    if has_tidy_dose_cols:
        # Filter finite entries
        dmask = np.isfinite(g["dosing_time"]) & np.isfinite(g["dosing_value"])
        if dmask.any():
            d_times = g.loc[dmask, "dosing_time"].astype(float).to_numpy()
            d_vals = g.loc[dmask, "dosing_value"].astype(float).to_numpy()
            # Optional normalization of dosing times must match observation normalization
            if normalize_time and len(t) > 0:
                # Use same (tmin,tmax) to maintain temporal consistency
                tmin, tmax = float(np.min(t)), float(np.max(t))
                if tmax > tmin:
                    d_times = (d_times - tmin) / (tmax - tmin)
                else:
                    d_times = np.zeros_like(d_times)
            dosing_times = d_times.tolist()
            dosing_vals = d_vals.tolist()
            # fallbacks for names/types
            d_type = g["dosing_type"].iloc[0] if "dosing_type" in g.columns else dosing_route_default
            s_name = g["substance_name"].iloc[0] if "substance_name" in g.columns else g["substance_label"].iloc[0]
            dosing_types = [str(d_type)] * len(dosing_times)
            dosing_names = [str(s_name)] * len(dosing_times)
    if not dosing_times:
        # Infer a single oral dose event
        s_name = (
            str(g["substance_name"].iloc[0]) if "substance_name" in g.columns
            else str(g["substance_label"].iloc[0])
        )
        inferred_dose = _infer_dose_for_substance(s_name)
        dosing_times = [0.0 if not normalize_time else 0.0]
        dosing_vals = [float(inferred_dose)]
        dosing_types = [dosing_route_default]
        dosing_names = [s_name]

    # Fields required by your schema but not present from CSV (use empty lists)
    remaining: List[float] = []
    remaining_times: List[float] = []
    covariates: Dict[str, object] = {}

    return {
        "name_id": str(subject_name),
        "observations": [float(v) for v in y.tolist()],
        "observation_times": [float(tt) for tt in obs_times_out],
        "remaining": remaining,
        "remaining_times": remaining_times,
        "dosing": [float(d) for d in dosing_vals],
        "dosing_type": [str(dt) for dt in dosing_types],
        "dosing_times": [float(dt) for dt in dosing_times],
        "dosing_name": [str(dn) for dn in dosing_names],
        "covariates": covariates,
    }

def _split_context_target(
    individuals: List[Dict],
    strategy: Literal["leave_one_out", "random_fraction", "all_in_context"] = "all_in_context",
    *,
    rng: Optional[np.random.Generator] = None,
    context_fraction: float = 0.5,
) -> List[Tuple[List[Dict], List[Dict]]]:
    """
    Returns a list of (context, target) pairs.
    - leave_one_out: yields N study_jsons by holding out each subject once.
    - random_fraction: yields a single split with ~context_fraction in context.
    - all_in_context: places all individuals in context and leaves target empty.
    """
    N = len(individuals)
    if N == 0:
        return [([], [])]
    if strategy == "leave_one_out":
        pairs = []
        for i in range(N):
            target = [individuals[i]]
            context = [ind for j, ind in enumerate(individuals) if j != i]
            pairs.append((context, target))
        return pairs
    elif strategy == "random_fraction":
        rng = rng or np.random.default_rng(0)
        idx = np.arange(N)
        rng.shuffle(idx)
        k = max(1, int(round(context_fraction * N)))
        context_idx = set(idx[:k])
        context = [individuals[i] for i in range(N) if i in context_idx]
        target = [individuals[i] for i in range(N) if i not in context_idx]
        if len(target) == 0:
            # ensure at least one target
            context, target = context[:-1], context[-1:]
        return [(context, target)]
    elif strategy == "all_in_context":
        return [(individuals, [])]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def dataframe_to_study_jsons(
    df: pd.DataFrame,
    *,
    normalize_time: bool = False,
    dosing_route_default: str = "oral",
    split_strategy: Literal["leave_one_out", "random_fraction", "all_in_context"] = "all_in_context",
    context_fraction: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict]:
    """
    Convert the tidy CSV into a list of study_json dicts:
        {
          "context": [individual_json, ...],
          "target":  [individual_json, ...],
          "meta_data": {"study_name": str, "substance_name": str}
        }

    Expected minimum columns in df:
      - 'study_name', 'substance_label', 'subject_name', 'time', 'value'
    Optional dosing columns (tidy per event):
      - 'dosing_time', 'dosing_value', ['dosing_type'], ['substance_name']

    Parameters
    ----------
    split_strategy:
        Strategy used to divide subjects into context and target sets. ``"all_in_context"``
        (default) places all individuals in the context and leaves the target set
        empty. ``"leave_one_out"`` yields a study per held-out subject, and
        ``"random_fraction"`` allocates approximately ``context_fraction`` of the
        individuals to the context.
    """
    required = {"study_name", "substance_label", "subject_name", "time", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {sorted(missing)}")

    study_jsons: List[Dict] = []

    # Group by (study_name, substance_label)
    for (study_name, substance_label), g_study in df.groupby(["study_name", "substance_label"], sort=False):
        # Build per-subject individuals
        individuals: List[Dict] = []
        for subject_name, g_subj in g_study.groupby("subject_name", sort=False):
            ind = _build_individual_json(
                subject_name,
                g_subj,
                normalize_time=normalize_time,
                dosing_route_default=dosing_route_default,
            )
            individuals.append(ind)

        # Produce (context, target) pairs
        pairs = _split_context_target(
            individuals,
            strategy=split_strategy,
            rng=rng,
            context_fraction=context_fraction,
        )

        # One study_json per split
        for context, target in pairs:
            raw = {
                "context": context,
                "target": target,
                "meta_data": {
                    "study_name": str(study_name),
                    "substance_name": str(substance_label),
                },
            }
            study_jsons.append(canonicalize_study(raw))

    return study_jsons


def save_study_jsons(study_jsons: List[Dict], path: Path) -> None:
    """Serialize ``study_jsons`` to ``path`` as JSON."""
    with Path(path).open("w") as f:
        json.dump(study_jsons, f, indent=2)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert Lenuzza CSV to canonical JSON")
    parser.add_argument("csv", type=Path, help="Input CSV path")
    parser.add_argument("out", type=Path, help="Output JSON path")
    parser.add_argument(
        "--normalize-time",
        action="store_true",
        help="Normalize observation times to [0,1]",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    study_jsons = dataframe_to_study_jsons(df, normalize_time=args.normalize_time)
    save_study_jsons(study_jsons, args.out)


if __name__ == "__main__":
    main()