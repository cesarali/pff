#!/usr/bin/env python3
"""Compute signature-kernel MMD from an ``.npz`` payload.

This runner intentionally stays isolated from ``pff`` so it can be
executed with a dedicated Python environment that only needs ``numpy`` and
``ksig`` installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ksig
import ksig.algorithms as ksig_algorithms
import ksig.kernels as ksig_kernels
import ksig.static.kernels as ksig_static_kernels
import ksig.utils as ksig_utils
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--signature-levels", required=True, type=int)
    parser.add_argument("--estimator", choices=("biased", "unbiased"), default="unbiased")
    parser.add_argument("--include-time-channel", choices=("0", "1"), default="1")
    return parser.parse_args()


def _force_cpu_backend_if_needed() -> None:
    """Patch ``ksig`` onto a NumPy backend when CuPy/CUDA is unavailable."""

    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
        return
    except Exception:
        pass

    if not hasattr(np, "asnumpy"):
        np.asnumpy = lambda x: x  # type: ignore[attr-defined]

    ksig_utils.cp = np
    ksig_static_kernels.cp = np
    ksig_algorithms.cp = np
    ksig_kernels.cp = np


def _mmd2_biased(k_xx: np.ndarray, k_yy: np.ndarray, k_xy: np.ndarray) -> float:
    """Return the biased ``MMD^2`` estimator."""

    return float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())


def _mmd2_unbiased(k_xx: np.ndarray, k_yy: np.ndarray, k_xy: np.ndarray) -> float:
    """Return the unbiased ``MMD^2`` estimator."""

    m = int(k_xx.shape[0])
    n = int(k_yy.shape[0])
    if m < 2 or n < 2:
        raise ValueError("Unbiased MMD requires at least two series in each group.")

    sum_xx = (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
    sum_yy = (k_yy.sum() - np.trace(k_yy)) / (n * (n - 1))
    sum_xy = k_xy.mean()
    return float(sum_xx + sum_yy - 2.0 * sum_xy)


def _compact_series_group(
    values_3d: np.ndarray,
    times_3d: np.ndarray,
    mask_2d: np.ndarray,
    *,
    include_time_channel: bool,
) -> list[np.ndarray]:
    """Compact one ``[It, T, F]`` group into a list of valid target series."""

    series_list: list[np.ndarray] = []
    for series_idx in range(values_3d.shape[0]):
        valid = mask_2d[series_idx].astype(bool)
        if not np.any(valid):
            continue

        series_values = values_3d[series_idx, valid, :]
        if include_time_channel:
            series_times = times_3d[series_idx, valid, :]
            series_values = np.concatenate([series_values, series_times], axis=-1)
        series_list.append(series_values.astype(np.float64, copy=False))
    return series_list


def _pad_series_list(series_list: list[np.ndarray]) -> np.ndarray:
    """Pad variable-length series to a common length for ``ksig``."""

    if not series_list:
        raise ValueError("Cannot pad an empty list of time series.")

    max_len = max(series.shape[0] for series in series_list)
    n_feat = int(series_list[0].shape[-1])
    padded = np.zeros((len(series_list), max_len, n_feat), dtype=np.float64)
    for idx, series in enumerate(series_list):
        seq_len = int(series.shape[0])
        padded[idx, :seq_len, :] = series
    return padded


def _compute_per_batch_mmd2(
    observed_values: np.ndarray,
    generated_values: np.ndarray,
    times: np.ndarray,
    mask: np.ndarray,
    *,
    signature_levels: int,
    estimator: str,
    include_time_channel: bool,
) -> list[float]:
    """Compute one ``MMD^2`` value per batch element."""

    static_kernel = ksig.static.kernels.RBFKernel()
    signature_kernel = ksig.kernels.SignatureKernel(
        n_levels=signature_levels,
        static_kernel=static_kernel,
    )

    per_batch_mmd2: list[float] = []
    batch_size = int(observed_values.shape[0])
    for batch_idx in range(batch_size):
        observed_series = _compact_series_group(
            observed_values[batch_idx],
            times[batch_idx],
            mask[batch_idx],
            include_time_channel=include_time_channel,
        )
        generated_series = _compact_series_group(
            generated_values[batch_idx],
            times[batch_idx],
            mask[batch_idx],
            include_time_channel=include_time_channel,
        )
        if not observed_series or not generated_series:
            raise ValueError(
                f"Batch element {batch_idx} does not contain valid observed/generated target series."
            )

        x = _pad_series_list(observed_series)
        y = _pad_series_list(generated_series)
        k_xx = np.asarray(signature_kernel(x))
        k_yy = np.asarray(signature_kernel(y))
        k_xy = np.asarray(signature_kernel(x, y))
        if estimator == "biased":
            mmd2_value = _mmd2_biased(k_xx, k_yy, k_xy)
        else:
            mmd2_value = _mmd2_unbiased(k_xx, k_yy, k_xy)
        per_batch_mmd2.append(float(mmd2_value))
    return per_batch_mmd2


def main() -> None:
    args = _parse_args()
    include_time_channel = args.include_time_channel == "1"
    _force_cpu_backend_if_needed()

    with np.load(args.payload, allow_pickle=False) as payload:
        observed_values = np.asarray(payload["observed_values"], dtype=np.float64)
        generated_values = np.asarray(payload["generated_values"], dtype=np.float64)
        times = np.asarray(payload["times"], dtype=np.float64)
        mask = np.asarray(payload["mask"], dtype=np.bool_)

    per_batch_mmd2 = _compute_per_batch_mmd2(
        observed_values,
        generated_values,
        times,
        mask,
        signature_levels=int(args.signature_levels),
        estimator=str(args.estimator),
        include_time_channel=include_time_channel,
    )
    result = {
        "mean_mmd2": float(np.mean(per_batch_mmd2)),
        "per_batch_mmd2": [float(value) for value in per_batch_mmd2],
        "num_batch_elements": int(len(per_batch_mmd2)),
        "include_time_channel": bool(include_time_channel),
        "signature_levels": int(args.signature_levels),
        "estimator": str(args.estimator),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
