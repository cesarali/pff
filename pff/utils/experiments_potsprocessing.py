from __future__ import annotations

import math
from typing import Any, Collection, Dict, List, Tuple, Optional

import pandas as pd  # if you have it in this module already

# 1. Standardization dictionary
_DRUG_NAME_MAP = {
    "1-hydroxymidazolam": "1-hydroxy-midazolam",
    "4-hydroxytolbutamide": "4-hydroxy-tolbutamide",
    "5-hydroxyomeprazole": "5-hydroxy-omeprazole",
    "caffeine (137X)": "caffeine",
    "dextromethorphan": "dextromethorphan",
    "dextrorphan": "dextrorphan",
    "digoxin": "digoxin",
    "hydroxy repaglinide": "hydroxy-repaglinide",
    "memantine": "memantine",
    "midazolam": "midazolam",
    "omeprazole": "omeprazole",
    "omeprazole sulfone": "omeprazole sulfone",
    "paracetamol": "paracetamol",
    "paracetamol glucuronide": "paracetamol glucuronide",
    "paraxanthine (17X)": "paraxanthine",
    "repaglinide": "repaglinide",
    "rosuvastatin": "rosuvastatin",
    "tolbutamide": "tolbutamide",
    "Indometacin": "indometacin",
    "Theophylline": "theophylline",
}


reference_results = {
    "drug": [
        "caffeine",
        "dextromethorphan",
        "digoxin",
        "memantine",
        "midazolam",
        "omeprazole",
        "paracetamol",
        "repaglinide",
        "rosuvastatin",
        "tolbutamide",
        "1-hydroxy-midazolam",
        "4-hydroxy-tolbutamide",
        "5-hydroxy-omeprazole",
        "dextrorphan",
        "hydroxy-repaglinide",
        "omeprazole sulfone",
        "paracetamol glucuronide",
        "paraxanthine",
    ],
    "NLME": [
        0.356,
        0.796,
        0.315,
        0.411,
        0.674,
        1.470,
        0.319,
        0.632,
        0.470,
        0.766,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
    ],
    "NODE-PK": [
        0.914,
        0.668,
        1.403,
        0.549,
        0.456,
        1.940,
        1.094,
        0.879,
        0.471,
        0.683,
        0.741,
        0.871,
        2.014,
        0.723,
        0.340,
        1.992,
        0.509,
        1.648,
    ],
    "T-PK": [
        0.575,
        0.630,
        0.717,
        0.799,
        0.735,
        1.864,
        0.825,
        0.846,
        0.748,
        0.816,
        0.678,
        0.898,
        1.683,
        1.001,
        0.532,
        1.620,
        0.423,
        0.646,
    ],
    "SNODE-PK": [
        0.780,
        1.702,
        0.501,
        0.580,
        0.874,
        1.267,
        1.115,
        1.514,
        0.624,
        0.949,
        1.395,
        0.524,
        1.811,
        0.904,
        0.059,
        1.529,
        0.823,
        0.653,
    ],
    "ST-PK": [
        0.984,
        1.412,
        0.421,
        0.869,
        0.817,
        1.078,
        1.050,
        1.246,
        0.604,
        0.998,
        1.216,
        0.742,
        1.600,
        0.860,
        0.336,
        1.294,
        1.057,
        0.858,
    ],
    "AICME-RNN": [
        0.646,
        0.640,
        0.569,
        0.534,
        0.548,
        1.395,
        0.691,
        0.562,
        0.578,
        0.854,
        0.935,
        0.274,
        1.575,
        0.614,
        0.095,
        1.438,
        0.365,
        0.409,
    ],
    "AICMET": [
        0.477,
        0.437,
        0.457,
        0.362,
        0.366,
        1.139,
        0.406,
        0.583,
        0.396,
        0.691,
        0.729,
        0.265,
        1.615,
        0.374,
        0.113,
        1.366,
        0.295,
        0.266,
    ],
}

reference_data_nme = {
    "drug": [
        "caffeine (137X)",
        "dextromethorphan",
        "digoxin",
        "memantine",
        "midazolam",
        "omeprazole",
        "paracetamol",
        "repaglinide",
        "rosuvastatin",
        "tolbutamide",
        "indometacin",
        "theophylline",
    ],
    "log-rmse": [0.356, 0.796, 0.315, 0.411, 0.674, 1.47, 0.319, 0.632, 0.470, 0.766, 0.604, 0.754],
    "log-r2": [0.820, 0.556, 0.482, 0.740, 0.344, -0.75, 0.905, 0.561, 0.557, 0.506, 100.0, 100.0],
}

reference_df = pd.DataFrame(reference_results)


def normalize_drug_name(raw: str) -> str:
    """
    Normalize drug names from comet logs to match reference table names.
    Falls back to the raw name if no mapping exists.
    """
    return _DRUG_NAME_MAP.get(raw, raw)


def _extract_drug_from_metric_name(
    metric_name_full: str,
    metric_name: str,
    top_level: str | None = None,
) -> str | None:
    """
    Extract drug name from metricName.

    Handles patterns like:
        "Empirical/Synthetic/paracetamol glucuronide/r2"
        "Synthetic/Synthetic/substance_16/rmse"
        (and ignores things like "Empirical/epoch_399/r2")

    top_level:
        If given, require metricName to start with this first segment, e.g. "Empirical".
    """
    parts = metric_name_full.split("/")
    if not parts:
        return None

    # Require that the last segment matches the metric_name we're interested in
    if parts[-1] != metric_name:
        return None

    # Optional filter on the very first segment: "Empirical", "Synthetic", etc.
    if top_level is not None and parts[0] != top_level:
        return None

    # Drop the metric name at the end
    core = parts[:-1]

    # Old-style names might have a trailing "epoch_399" segment; drop it if present
    if core and core[-1].startswith("epoch_"):
        core = core[:-1]

    # We expect at least [prefix, drug] -> length >= 2
    if len(core) < 2:
        return None

    raw_drug = core[-1]
    if not raw_drug:
        return None

    # Don't treat these prefixes as drugs
    if raw_drug.lower() in {"empirical", "synthetic", "train", "val", "test"}:
        return None

    return normalize_drug_name(raw_drug)


def _parse_metric_name_new(metric_name_full: str) -> dict[str, str | bool | None]:
    """
    Parse one newer scheduler metric path into structured components.

    Expected shapes include:
        "{log_prefix}/{task_name}__{checkpoint_name}/{drug}/{metric_name}"
        "{log_prefix}/{task_name}__{checkpoint_name}/{metric_name}"
    """
    parts = [part for part in metric_name_full.split("/") if part]
    if not parts:
        return {
            "full_metric_name": metric_name_full,
            "log_prefix": None,
            "task_path": None,
            "task_name": None,
            "checkpoint_name": None,
            "drug": None,
            "metric_name": None,
            "is_per_drug": False,
        }

    log_prefix = parts[0]
    metric_name = parts[-1]

    if len(parts) >= 4:
        task_parts = parts[1:-2]
        raw_drug = parts[-2]
    elif len(parts) >= 3:
        task_parts = parts[1:-1]
        raw_drug = None
    else:
        task_parts = []
        raw_drug = None

    task_path = "/".join(task_parts) if task_parts else None
    task_name = task_path
    checkpoint_name = None
    if task_parts:
        last_task_segment = task_parts[-1]
        if "__" in last_task_segment:
            task_stem, checkpoint_name = last_task_segment.rsplit("__", 1)
            task_name = "/".join(task_parts[:-1] + [task_stem])

    drug: str | None = None
    if raw_drug:
        raw_drug_lower = raw_drug.lower()
        if raw_drug_lower not in {
            "empirical",
            "synthetic",
            "train",
            "val",
            "test",
            "mean",
            "std",
            "predictive_metrics",
            "generative_metrics",
            "vpc_npde_pvalues",
        } and not raw_drug_lower.startswith("summary__") and not raw_drug_lower.startswith("epoch_"):
            drug = normalize_drug_name(raw_drug)

    return {
        "full_metric_name": metric_name_full,
        "log_prefix": log_prefix,
        "task_path": task_path,
        "task_name": task_name,
        "checkpoint_name": checkpoint_name,
        "drug": drug,
        "metric_name": metric_name,
        "is_per_drug": drug is not None,
    }


def _resolve_target_epoch(
    metrics_list: List[Dict[str, Any]],
    epoch: int | str,
) -> int | None:
    """
    Resolve an epoch selector into one concrete numeric epoch.

    Supported values
    ----------------
    - integer: use that epoch directly
    - numeric string: same as integer
    - ``"last"``: latest numeric epoch found in ``metrics_list``
    - ``"end"``: alias for ``"last"``
    """
    if isinstance(epoch, str):
        epoch_text = epoch.strip()
        if epoch_text in {"last", "end"}:
            epochs: List[int] = []
            for m in metrics_list:
                e = m.get("epoch")
                try:
                    if e is not None:
                        epochs.append(int(e))
                except (TypeError, ValueError):
                    continue
            return max(epochs) if epochs else None

        try:
            return int(epoch_text)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported epoch selector '{epoch}'. "
                "Use an integer, a numeric string, 'last', or 'end'."
            ) from exc

    return int(epoch)


def metrics_list_to_pandas(
    metrics_list: List[Dict[str, Any]],
    model_name: str,
    metric_name: str,
    epoch: int | str,
    top_level: str | None = None,
) -> pd.DataFrame:
    """
    Convert comet_ml metrics to a per-drug DataFrame for a given metric and epoch.

    metrics_list entries look like:
        {
            "metricName": "Empirical/Synthetic/paracetamol glucuronide/r2",
            "metricValue": "-0.09778215289115906",
            "timestamp": 1764093835814,
            "step": 2,
            "epoch": 0,
            ...
        }

    top_level:
        Optional filter on the first path segment in metricName, e.g. "Empirical" or "Synthetic".
    """

    # -----------------------
    # 1) Resolve target epoch
    # -----------------------
    target_epoch = _resolve_target_epoch(metrics_list, epoch)
    if target_epoch is None:
        return pd.DataFrame(columns=["drug", model_name])

    # -----------------------
    # 2) Collect rows
    # -----------------------
    rows: list[tuple[str, float, int]] = []

    for m in metrics_list:
        name = m.get("metricName") or ""

        drug = _extract_drug_from_metric_name(
            metric_name_full=name,
            metric_name=metric_name,
            top_level=top_level,
        )
        if not drug:
            continue

        # Filter by epoch field (new comet format)
        e_raw = m.get("epoch")
        try:
            e_val = int(e_raw)
        except (TypeError, ValueError):
            continue

        if e_val != target_epoch:
            continue

        # Metric value
        try:
            value = float(m.get("metricValue"))
        except (TypeError, ValueError):
            continue

        ts = int(m.get("timestamp", 0))
        rows.append((drug, value, ts))

    if not rows:
        return pd.DataFrame(columns=["drug", model_name])

    # -----------------------
    # 3) Keep latest per drug
    # -----------------------
    latest: dict[str, tuple[float, int]] = {}
    for drug, value, ts in rows:
        cur = latest.get(drug)
        if cur is None or ts > cur[1]:
            latest[drug] = (value, ts)

    data = [{"drug": d, model_name: vts[0]} for d, vts in latest.items()]
    df = pd.DataFrame(data).sort_values("drug").reset_index(drop=True)
    return df


def metrics_list_to_pandas_new(
    metrics_list: List[Dict[str, Any]],
    model_name: str,
    metric_name: str,
    epoch: int | str,
    log_prefix: str | None = None,
    task_name: str | None = None,
    checkpoint_name: str | None = None,
    top_level: str | None = None,
) -> pd.DataFrame:
    """
    Convert newer scheduler-based comet metrics to a per-drug DataFrame.

    This parser targets namespaced metric names logged as:
        {log_prefix}/{task_name}/{optional_context...}/{drug}/{metric_name}

    Examples
    --------
    Per-drug predictive metric:
        "Empirical/empirical/predictive_metrics/lenuzza-2016__log_rmse/repaglinide/log_r2_std"

    Scalar summary metric that is intentionally ignored:
        "Empirical/empirical/summary__end/log_rmse"

    Parameters
    ----------
    metrics_list : List[Dict[str, Any]]
        Raw comet metric entries.
    model_name : str
        Name of the output metric column.
    metric_name : str
        Metric key to extract from the last path segment.
    epoch : int | str
        Either an explicit epoch integer or ``"last"``.
    log_prefix : str | None, optional
        Optional filter on the first path segment in ``metricName``.
        In your examples this is ``"Empirical"``.
    task_name : str | None, optional
        Optional exact scheduler task path between ``log_prefix`` and the
        ``drug/metric`` suffix.

        Example:
            ``"empirical/predictive_metrics/lenuzza-2016"``

        If ``checkpoint_name`` is also given, the function matches:
            ``{task_name}__{checkpoint_name}``
    checkpoint_name : str | None, optional
        Optional checkpoint selector appended by the scheduler to the last task
        segment, for example ``"last"``, ``"end"``, or ``"log_rmse"``.
    top_level : str | None, optional
        Backward-compatible alias for ``log_prefix``.

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame with ``["drug", model_name]`` sorted by drug.
    """
    resolved_log_prefix = log_prefix if log_prefix is not None else top_level

    target_epoch = _resolve_target_epoch(metrics_list, epoch)
    if target_epoch is None:
        return pd.DataFrame(columns=["drug", model_name])

    rows: list[tuple[str, float, int]] = []

    for m in metrics_list:
        name = m.get("metricName") or ""
        parsed = _parse_metric_name_new(name)

        if parsed["metric_name"] != metric_name:
            continue
        if resolved_log_prefix is not None and parsed["log_prefix"] != resolved_log_prefix:
            continue
        if task_name is not None and parsed["task_name"] != task_name:
            continue
        if checkpoint_name is not None and parsed["checkpoint_name"] != checkpoint_name:
            continue

        drug = parsed["drug"]
        if drug is None:
            continue

        e_raw = m.get("epoch")
        try:
            e_val = int(e_raw)
        except (TypeError, ValueError):
            continue

        if e_val != target_epoch:
            continue

        try:
            value = float(m.get("metricValue"))
        except (TypeError, ValueError):
            continue

        ts = int(m.get("timestamp", 0))
        rows.append((drug, value, ts))

    if not rows:
        return pd.DataFrame(columns=["drug", model_name])

    latest: dict[str, tuple[float, int]] = {}
    for drug, value, ts in rows:
        cur = latest.get(drug)
        if cur is None or ts > cur[1]:
            latest[drug] = (value, ts)

    data = [{"drug": d, model_name: vts[0]} for d, vts in latest.items()]
    df = pd.DataFrame(data).sort_values("drug").reset_index(drop=True)
    return df


def available_tasks_and_checkpoints_new(
    metrics_list: List[Dict[str, Any]],
    epoch: int | str | None = None,
    log_prefix: str | None = None,
) -> pd.DataFrame:
    """
    Summarize available task/checkpoint/metric combinations in new-style metrics.

    This is intended as a discovery helper so you can inspect which values to pass
    to ``task_name`` and ``checkpoint_name`` in ``metrics_list_to_pandas_new``.
    """
    target_epoch: int | None = None
    if epoch is not None:
        target_epoch = _resolve_target_epoch(metrics_list, epoch)

    rows: list[dict[str, Any]] = []
    for m in metrics_list:
        e_raw = m.get("epoch")
        if target_epoch is not None:
            try:
                e_val = int(e_raw)
            except (TypeError, ValueError):
                continue
            if e_val != target_epoch:
                continue

        parsed = _parse_metric_name_new(m.get("metricName") or "")
        if parsed["metric_name"] is None:
            continue
        if log_prefix is not None and parsed["log_prefix"] != log_prefix:
            continue

        rows.append(
            {
                "log_prefix": parsed["log_prefix"],
                "task_name": parsed["task_name"],
                "checkpoint_name": parsed["checkpoint_name"],
                "metric_name": parsed["metric_name"],
                "drug": parsed["drug"],
                "is_per_drug": parsed["is_per_drug"],
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "log_prefix",
                "task_name",
                "checkpoint_name",
                "metric_name",
                "is_per_drug",
                "n_drugs",
            ]
        )

    raw_df = pd.DataFrame(rows)
    summary = (
        raw_df.groupby(
            ["log_prefix", "task_name", "checkpoint_name", "metric_name", "is_per_drug"],
            dropna=False,
        )["drug"]
        .nunique(dropna=True)
        .reset_index(name="n_drugs")
        .sort_values(
            ["log_prefix", "task_name", "checkpoint_name", "metric_name"],
            na_position="last",
        )
        .reset_index(drop=True)
    )
    return summary


def empirical_batches_to_pandas(
    all_empirical_batches: Dict[str, List["AICMECompartmentsDataBatch"]],
    model: Any,
    model_name: str,
    metric_name: str,
    repo_filter: Optional[Collection[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate per-drug metrics computed from all_empirical_batches into a
    DataFrame with columns ["drug", model_name], analogous to metrics_list_to_pandas.

    Parameters
    ----------
    all_empirical_batches : Dict[str, List[AICMECompartmentsDataBatch]]
        Mapping repo_id -> list of batches.
    model : Any
        Model instance exposing `_compute_metrics_from_batch_list(batch_list, repo_id)`.
    model_name : str
        Name of the model; becomes the metric column name in the DataFrame.
    metric_name : str
        Which metric to extract ("rmse", "log_rmse", "r2", "log_r2", ...).
    repo_filter : Optional[Collection[str]]
        If given, only these repo_ids are processed.

    Returns
    -------
    pd.DataFrame
        Columns: ["drug", model_name], sorted by drug.
    """
    rows: list[tuple[str, float, str]] = []

    for repo_id, batch_list in all_empirical_batches.items():
        if repo_filter is not None and repo_id not in repo_filter:
            continue

        # metrics: dict[raw_drug -> dict[metric_name -> value, ...]]
        metrics, _prediction_cache = model._compute_metrics_from_batch_list(batch_list, repo_id)

        for raw_drug, metric_dict in metrics.items():
            if metric_dict is None:
                continue

            if metric_name not in metric_dict:
                continue

            value = metric_dict[metric_name]
            if value is None:
                continue

            try:
                v = float(value)
            except (TypeError, ValueError):
                continue

            drug = normalize_drug_name(raw_drug)
            rows.append((drug, v, repo_id))

    if not rows:
        return pd.DataFrame(columns=["drug", model_name])

    # If a drug appears multiple times (e.g. in several repos), keep the last one.
    latest_by_drug: Dict[str, float] = {}
    for drug, value, _repo_id in rows:
        latest_by_drug[drug] = value

    data = [{"drug": d, model_name: v} for d, v in latest_by_drug.items()]
    df = pd.DataFrame(data).sort_values("drug").reset_index(drop=True)
    return df


def reference_dict_to_pandas(
    reference_data: Dict[str, list],
    model_name: str,
    metric_name: str,
) -> pd.DataFrame:
    """
    Convert a reference dictionary with drug-level metrics into a pandas DataFrame.

    The dictionary must have at least the keys:
      - "drug": list[str]
      - <metric_name>: list[float]

    Applies normalization of drug names to ensure consistency.

    Parameters
    ----------
    reference_data : dict
        Dictionary with keys "drug" and metric names (e.g., "log-rmse", "log-r2").
    model_name : str
        Name for the output value column (like "NodePK" or "GP").
    metric_name : str
        Which metric to extract (must be in the dict).

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame with:
          - "drug": standardized drug names
          - model_name: metric values
        Sorted by drug name.
    """
    if metric_name not in reference_data:
        raise ValueError(
            f"Metric '{metric_name}' not in reference_data keys {list(reference_data.keys())}"
        )

    drugs = [normalize_drug_name(d) for d in reference_data["drug"]]
    values = reference_data[metric_name]
    df = pd.DataFrame({"drug": drugs, model_name: values})
    return df.sort_values("drug").reset_index(drop=True)


def available_epochs_and_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, list[str]]:
    """
    Summarize which epochs, metrics and top-level prefixes are available in
    a comet_ml metrics list.

    This handles both:
      - New-style: epoch is in the 'epoch' field and metricName is something like
            "Empirical/Synthetic/paracetamol glucuronide/r2"
      - Old-style: epoch encoded in metricName, e.g.
            "Empirical/epoch_399/r2"

    Returns
    -------
    Dict[str, list[str]]
        {
            "epochs_available":  list of unique epoch identifiers (strings),
            "metrics_available": list of unique metric names (last path segment),
            "top_levels_available": list of unique top-level prefixes (first path segment)
        }
    """
    epochs: set[str] = set()
    metrics: set[str] = set()
    top_levels: set[str] = set()

    for m in metrics_list:
        name = m.get("metricName") or ""
        if not name:
            continue

        parts = name.split("/")
        if not parts:
            continue

        # top-level, e.g. "Empirical" or "Synthetic"
        top_levels.add(parts[0])

        # metric name is always the last segment, e.g. "rmse", "r2"
        metric = parts[-1]
        metrics.add(metric)

        # --- New-style: epoch field present ---
        e_field = m.get("epoch", None)
        if e_field is not None:
            try:
                epochs.add(str(int(e_field)))
            except (TypeError, ValueError):
                pass
        else:
            # --- Fallback: old-style epoch encoded in the parent segment ---
            if len(parts) >= 2:
                parent = parts[-2]
                if parent.startswith("epoch_"):
                    epochs.add(parent.replace("epoch_", ""))

    return {
        "epochs_available": sorted(epochs, key=lambda x: (x != "last", x)),
        "metrics_available": sorted(metrics),
        "top_levels_available": sorted(top_levels),
    }


def count_model_wins(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    *,
    smaller_is_better: bool = True,
) -> Tuple[int, int, int]:
    """
    Compare two models column-by-column in a merged DataFrame and count wins.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the two columns `model_a` and `model_b` with numeric values.
    model_a : str
        Name of the first model column in df.
    model_b : str
        Name of the second model column in df.
    smaller_is_better : bool, default=True
        If True, smaller values are considered better (e.g. RMSE).
        If False, larger values are considered better (e.g. R^2).

    Returns
    -------
    wins_a : int
        Number of rows where model_a outperforms model_b.
    wins_b : int
        Number of rows where model_b outperforms model_a.
    ties : int
        Number of rows where they are equal (after dropping NaNs).
    """
    # Select valid rows only
    valid = df[[model_a, model_b]].dropna()

    if smaller_is_better:
        wins_a = (valid[model_a] < valid[model_b]).sum()
        wins_b = (valid[model_b] < valid[model_a]).sum()
    else:
        wins_a = (valid[model_a] > valid[model_b]).sum()
        wins_b = (valid[model_b] > valid[model_a]).sum()

    ties = (valid[model_a] == valid[model_b]).sum()

    return wins_a, wins_b, ties
