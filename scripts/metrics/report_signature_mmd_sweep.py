#!/usr/bin/env python3
"""Generate a signature-MMD sweep report for progressively dissimilar time series.

This script builds one reference cohort of synthetic univariate time series and a
sequence of progressively perturbed cohorts. For each perturbation level it writes
an ``.npz`` payload, runs the standalone signature-MMD runner, and stores:

1. a JSON summary with the measured ``MMD^2`` values,
2. a PNG figure showing representative trajectory shifts and the MMD curve,
3. a Markdown report in ``reports/metrics/signature_mmd_sweep`` by default.

Usage
-----
python scripts/metrics/report_signature_mmd_sweep.py
python scripts/metrics/report_signature_mmd_sweep.py --signature-levels 5 --seed 13
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pff import reports_dir


DEFAULT_KSIG_PYTHON = Path("/home/cesarali/miniconda3/envs/ksig/bin/python")


@dataclass(frozen=True)
class SweepArtifacts:
    """Resolved output locations for the sweep report."""

    output_dir: Path
    figure_path: Path
    report_path: Path
    metrics_path: Path


@dataclass(frozen=True)
class SeriesFamily:
    """Latent parameters used to generate one coherent family of trajectories."""

    amplitude: np.ndarray
    decay: np.ndarray
    phase: np.ndarray
    frequency: np.ndarray
    shoulder: np.ndarray
    amp_shift: np.ndarray
    phase_shift: np.ndarray
    trend_shift: np.ndarray
    warp_shift: np.ndarray


def _parse_scales(raw: str) -> list[float]:
    """Parse a comma-separated perturbation scale list."""

    scales = [float(piece.strip()) for piece in raw.split(",") if piece.strip()]
    if not scales:
        raise ValueError("At least one dissimilarity scale is required.")
    if any(scale < 0.0 for scale in scales):
        raise ValueError("Dissimilarity scales must be non-negative.")
    return scales


def _resolve_default_python() -> Path:
    """Prefer the dedicated ksig environment when it exists locally."""

    if DEFAULT_KSIG_PYTHON.exists():
        return DEFAULT_KSIG_PYTHON
    return Path(sys.executable)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the sweep report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=reports_dir / "metrics" / "signature_mmd_sweep",
        help="Directory where the figure, report, and JSON summary are written.",
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=_resolve_default_python(),
        help="Python executable that can run scripts/metrics/run_signature_mmd.py.",
    )
    parser.add_argument("--num-series", type=int, default=24)
    parser.add_argument("--num-timepoints", type=int, default=48)
    parser.add_argument("--signature-levels", type=int, default=4)
    parser.add_argument("--estimator", choices=("biased", "unbiased"), default="unbiased")
    parser.add_argument("--include-time-channel", choices=("0", "1"), default="1")
    parser.add_argument(
        "--dissimilarity-scales",
        type=str,
        default="0.0,0.1,0.2,0.35,0.5,0.75,1.0",
        help="Comma-separated perturbation scales used to create increasingly different cohorts.",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _resolve_runner_path() -> Path:
    """Return the standalone signature-MMD runner path."""

    return Path(__file__).resolve().parent / "run_signature_mmd.py"


def _resolve_artifacts(output_dir: Path) -> SweepArtifacts:
    """Resolve the sweep output locations and create the parent directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    return SweepArtifacts(
        output_dir=output_dir,
        figure_path=output_dir / "signature_mmd_sweep.png",
        report_path=output_dir / "signature_mmd_sweep.md",
        metrics_path=output_dir / "signature_mmd_sweep.json",
    )


def _build_time_grid(num_timepoints: int) -> np.ndarray:
    """Return a shared regular time grid with shape ``[T]``."""

    if num_timepoints < 4:
        raise ValueError("num_timepoints must be at least 4.")
    return np.linspace(0.0, 1.0, num_timepoints, dtype=np.float64)


def _sample_series_family(num_series: int, rng: np.random.Generator) -> SeriesFamily:
    """Sample latent coefficients for one family of related trajectories."""

    if num_series < 2:
        raise ValueError("num_series must be at least 2 for unbiased MMD.")

    return SeriesFamily(
        amplitude=rng.uniform(0.8, 1.4, size=num_series),
        decay=rng.uniform(0.8, 1.6, size=num_series),
        phase=rng.uniform(0.0, 2.0 * np.pi, size=num_series),
        frequency=rng.uniform(0.8, 1.4, size=num_series),
        shoulder=rng.uniform(0.08, 0.18, size=num_series),
        amp_shift=rng.uniform(0.2, 0.55, size=num_series),
        phase_shift=rng.uniform(0.6, 1.3, size=num_series),
        trend_shift=rng.uniform(-0.25, 0.35, size=num_series),
        warp_shift=rng.uniform(0.08, 0.24, size=num_series),
    )


def _generate_series_values(
    family: SeriesFamily,
    times_1d: np.ndarray,
    *,
    dissimilarity_scale: float,
) -> np.ndarray:
    """Generate values with shape ``[It, T, 1]`` for one perturbation level."""

    values_list: list[np.ndarray] = []
    for series_idx in range(len(family.amplitude)):
        time_warp = np.clip(
            times_1d
            + dissimilarity_scale * family.warp_shift[series_idx] * times_1d * (1.0 - times_1d),
            0.0,
            1.0,
        )
        envelope = (
            family.amplitude[series_idx]
            * (1.0 + dissimilarity_scale * family.amp_shift[series_idx])
            * np.exp(-(family.decay[series_idx] + 0.75 * dissimilarity_scale) * 3.0 * time_warp)
        )
        oscillation = 1.0 + 0.18 * np.sin(
            2.0 * np.pi * (family.frequency[series_idx] + 0.25 * dissimilarity_scale) * time_warp
            + family.phase[series_idx]
            + dissimilarity_scale * family.phase_shift[series_idx]
        )
        shoulder = (
            family.shoulder[series_idx]
            * (1.0 + 0.6 * dissimilarity_scale)
            * np.exp(-(((time_warp - (0.28 + 0.18 * dissimilarity_scale)) / 0.11) ** 2))
        )
        trend = dissimilarity_scale * family.trend_shift[series_idx] * times_1d
        series_values = envelope * oscillation + shoulder + trend + 0.02 * dissimilarity_scale**2
        values_list.append(series_values[:, None].astype(np.float64, copy=False))
    return np.stack(values_list, axis=0)


def _build_runner_inputs(
    observed_values_3d: np.ndarray,
    generated_values_3d: np.ndarray,
    times_1d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build runner arrays with batch dimension ``B=1``."""

    num_series, num_timepoints, _ = observed_values_3d.shape
    times_4d = np.broadcast_to(times_1d[None, None, :, None], (1, num_series, num_timepoints, 1))
    mask_3d = np.ones((1, num_series, num_timepoints), dtype=np.bool_)
    observed_values_4d = observed_values_3d[None, ...]
    generated_values_4d = generated_values_3d[None, ...]
    return observed_values_4d, generated_values_4d, times_4d, mask_3d


def _write_payload(
    payload_path: Path,
    *,
    observed_values: np.ndarray,
    generated_values: np.ndarray,
    times: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Serialize one synthetic sweep comparison to an ``.npz`` payload."""

    np.savez_compressed(
        payload_path,
        observed_values=observed_values,
        generated_values=generated_values,
        times=times,
        mask=mask.astype(np.bool_, copy=False),
    )


def _run_signature_mmd_runner(
    *,
    python_executable: Path,
    payload_path: Path,
    output_path: Path,
    signature_levels: int,
    estimator: str,
    include_time_channel: bool,
) -> dict[str, Any]:
    """Execute the standalone runner and parse its JSON output."""

    runner_path = _resolve_runner_path()
    if not runner_path.exists():
        raise FileNotFoundError(f"Signature MMD runner not found at '{runner_path}'.")
    if not python_executable.exists():
        raise FileNotFoundError(f"Python executable not found at '{python_executable}'.")

    command = [
        str(python_executable),
        str(runner_path),
        "--payload",
        str(payload_path),
        "--output",
        str(output_path),
        "--signature-levels",
        str(int(signature_levels)),
        "--estimator",
        estimator,
        "--include-time-channel",
        "1" if include_time_channel else "0",
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Signature MMD runner failed with exit code "
            f"{exc.returncode}. stdout={exc.stdout!r} stderr={exc.stderr!r}"
        ) from exc

    with output_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise TypeError("Signature MMD runner output must be a JSON object.")
    return result


def _plot_sweep(
    *,
    artifacts: SweepArtifacts,
    times_1d: np.ndarray,
    reference_values_3d: np.ndarray,
    example_series_by_scale: dict[float, np.ndarray],
    scales: list[float],
    mmd2_values: list[float],
) -> None:
    """Render a two-panel figure with trajectory examples and the MMD curve."""

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4.5))

    reference_mean = reference_values_3d[:, :, 0].mean(axis=0)
    ax_left.plot(times_1d, reference_mean, color="black", linewidth=2.2, label="reference mean")
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(example_series_by_scale)))
    for color, (scale, values_3d) in zip(colors, example_series_by_scale.items(), strict=True):
        candidate_mean = values_3d[:, :, 0].mean(axis=0)
        ax_left.plot(
            times_1d,
            candidate_mean,
            color=color,
            linewidth=1.8,
            label=f"scale={scale:.2f}",
        )
    ax_left.set_title("Mean Time Series Shift")
    ax_left.set_xlabel("normalized time")
    ax_left.set_ylabel("value")
    ax_left.legend(frameon=False)

    ax_right.plot(scales, mmd2_values, marker="o", linewidth=2.0, color="tab:blue")
    ax_right.set_title("Signature MMD Sweep")
    ax_right.set_xlabel("dissimilarity scale")
    ax_right.set_ylabel("MMD$^2$")
    ax_right.grid(alpha=0.25)

    fig.suptitle("Progressively Dissimilar Synthetic Cohorts")
    fig.tight_layout()
    fig.savefig(artifacts.figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_markdown_report(
    *,
    artifacts: SweepArtifacts,
    num_series: int,
    num_timepoints: int,
    signature_levels: int,
    estimator: str,
    include_time_channel: bool,
    seed: int,
    scales: list[float],
    mmd2_values: list[float],
) -> None:
    """Write a short Markdown report summarizing the sweep."""

    monotonic_steps = int(sum(diff > 0.0 for diff in np.diff(mmd2_values)))
    table_rows = "\n".join(
        f"| {scale:.2f} | {mmd2:.6f} |" for scale, mmd2 in zip(scales, mmd2_values, strict=True)
    )
    report = f"""# Signature MMD Sweep Report

This report was generated from synthetic cohorts with progressively larger trajectory perturbations.

## Configuration

- num_series: {num_series}
- num_timepoints: {num_timepoints}
- signature_levels: {signature_levels}
- estimator: {estimator}
- include_time_channel: {include_time_channel}
- seed: {seed}
- output_figure: `{artifacts.figure_path.name}`
- output_metrics: `{artifacts.metrics_path.name}`

## Result

The measured `MMD^2` values increase across {monotonic_steps} of {max(len(scales) - 1, 0)} adjacent sweep steps.

| dissimilarity_scale | mmd2 |
| --- | ---: |
{table_rows}
"""
    artifacts.report_path.write_text(report, encoding="utf-8")


def run_sweep_report(
    *,
    output_dir: Path,
    python_executable: Path,
    num_series: int,
    num_timepoints: int,
    signature_levels: int,
    estimator: str,
    include_time_channel: bool,
    dissimilarity_scales: list[float],
    seed: int,
) -> dict[str, Any]:
    """Run the full synthetic MMD sweep and persist the report artifacts."""

    artifacts = _resolve_artifacts(output_dir)
    rng = np.random.default_rng(seed)
    times_1d = _build_time_grid(num_timepoints)
    family = _sample_series_family(num_series, rng)
    reference_values_3d = _generate_series_values(
        family,
        times_1d,
        dissimilarity_scale=0.0,
    )  # reference_values_3d: [It, T, 1]

    mmd2_values: list[float] = []
    example_series_by_scale: dict[float, np.ndarray] = {}
    example_indices = {0, len(dissimilarity_scales) // 2, len(dissimilarity_scales) - 1}

    with tempfile.TemporaryDirectory(prefix="signature_mmd_sweep_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for scale_idx, scale in enumerate(dissimilarity_scales):
            candidate_values_3d = _generate_series_values(
                family,
                times_1d,
                dissimilarity_scale=scale,
            )  # candidate_values_3d: [It, T, 1]
            observed_values, generated_values, times, mask = _build_runner_inputs(
                reference_values_3d,
                candidate_values_3d,
                times_1d,
            )
            payload_path = tmp_root / f"signature_mmd_scale_{scale_idx:02d}.npz"
            result_path = tmp_root / f"signature_mmd_scale_{scale_idx:02d}.json"
            _write_payload(
                payload_path,
                observed_values=observed_values,
                generated_values=generated_values,
                times=times,
                mask=mask,
            )
            result = _run_signature_mmd_runner(
                python_executable=python_executable,
                payload_path=payload_path,
                output_path=result_path,
                signature_levels=signature_levels,
                estimator=estimator,
                include_time_channel=include_time_channel,
            )
            mmd2_values.append(float(result["mean_mmd2"]))
            if scale_idx in example_indices:
                example_series_by_scale[scale] = candidate_values_3d

    summary: dict[str, Any] = {
        "seed": int(seed),
        "num_series": int(num_series),
        "num_timepoints": int(num_timepoints),
        "signature_levels": int(signature_levels),
        "estimator": estimator,
        "include_time_channel": bool(include_time_channel),
        "dissimilarity_scales": [float(scale) for scale in dissimilarity_scales],
        "mmd2_values": [float(value) for value in mmd2_values],
        "figure_path": str(artifacts.figure_path),
        "report_path": str(artifacts.report_path),
    }
    artifacts.metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _plot_sweep(
        artifacts=artifacts,
        times_1d=times_1d,
        reference_values_3d=reference_values_3d,
        example_series_by_scale=example_series_by_scale,
        scales=dissimilarity_scales,
        mmd2_values=mmd2_values,
    )
    _write_markdown_report(
        artifacts=artifacts,
        num_series=num_series,
        num_timepoints=num_timepoints,
        signature_levels=signature_levels,
        estimator=estimator,
        include_time_channel=include_time_channel,
        seed=seed,
        scales=dissimilarity_scales,
        mmd2_values=mmd2_values,
    )
    return summary


def main() -> None:
    """CLI entry point."""

    args = _parse_args()
    run_sweep_report(
        output_dir=Path(args.output_dir),
        python_executable=Path(args.python_executable),
        num_series=int(args.num_series),
        num_timepoints=int(args.num_timepoints),
        signature_levels=int(args.signature_levels),
        estimator=str(args.estimator),
        include_time_channel=args.include_time_channel == "1",
        dissimilarity_scales=_parse_scales(str(args.dissimilarity_scales)),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
