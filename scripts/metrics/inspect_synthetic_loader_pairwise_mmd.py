#!/usr/bin/env python3
"""Inspect pairwise signature-MMD distances across synthetic loader studies.

This script mirrors the synthetic-loader setup used by
``pk.diverse_experiment.distances`` and computes all-to-all pairwise
``MMD^2`` distances between independent synthetic studies sampled from that
loader. It saves:

1. a histogram of the pairwise ``MMD^2`` values,
2. a CSV file with one row per study pair,
3. the raw runner JSON emitted by ``scripts/metrics/run_signature_mmd.py``.

Example
-------
python scripts/metrics/inspect_synthetic_loader_pairwise_mmd.py \
    --n-targets 500 \
    --dataset-size 10 \
    --python-executable /home/cesarali/miniconda3/envs/ksig/bin/python
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torchtyping import TensorType

from pff import config_dir, reports_dir
from pff.config_classes.data_config import ObservationsConfig
from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.data.datasets.aicme_datasets import AICMECompartmentsDataModule
from pff.metrics.sample_distance_metrics import (
    _run_signature_mmd_runner,
    _write_synthetic_mmd_payload,
)

DEFAULT_CONFIG_PATH = (
    Path(config_dir) / "experiment_configs" / "UAI" / "Submission" / "aicme-t-pk" / "base.yaml"
)
DEFAULT_KSIG_PYTHON = Path("/home/cesarali/miniconda3/envs/ksig/bin/python")


@dataclass(frozen=True)
class SyntheticStudySeries:
    """One synthetic study packaged for pairwise MMD comparison."""

    study_label: str
    values: TensorType["It", "Tobs", 1]
    times: TensorType["It", "Tobs", 1]
    mask: TensorType["It", "Tobs"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="NodePK experiment YAML used to construct the datamodule.",
    )
    parser.add_argument("--n-targets", type=int, default=500)
    parser.add_argument("--dataset-size", type=int, default=10)
    parser.add_argument("--signature-levels", type=int, default=4)
    parser.add_argument(
        "--estimator",
        choices=("biased", "unbiased"),
        default="unbiased",
    )
    parser.add_argument(
        "--include-time-channel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to append the time channel before the signature kernel.",
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=DEFAULT_KSIG_PYTHON,
        help="Python executable with numpy and ksig installed.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--max-num-obs",
        type=int,
        default=20,
        help="Number of target observations on the fixed regular grid.",
    )
    parser.add_argument(
        "--fixed-grid-start-index",
        type=int,
        default=3,
        help="Skip this many solver-grid steps before selecting the fixed target grid.",
    )
    parser.add_argument(
        "--time-num-steps",
        type=int,
        default=24,
        help="Synthetic solver-grid length before the fixed target grid is sampled.",
    )
    parser.add_argument("--hist-bins", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(reports_dir) / "metrics" / "synthetic_loader_pairwise_mmd",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducible synthetic sampling."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_datamodule(
    args: argparse.Namespace,
) -> tuple[AICMECompartmentsDataModule, ObservationsConfig]:
    """Build a datamodule configured only for synthetic-loader inspection."""

    cfg = NodePKExperimentConfig.from_yaml(str(args.config))
    cfg.upload_to_hf_hub = False
    cfg.train.batch_size = 1
    cfg.train.num_workers = 0
    cfg.train.persistent_workers = False

    # Keep datamodule preparation cheap because this script only needs the
    # synthetic loader, not full train/val/test datasets.
    cfg.mix_data.train_size = 2
    cfg.mix_data.val_size = 1
    cfg.mix_data.test_size = 1
    cfg.mix_data.n_of_permutations = 1
    cfg.mix_data.test_empirical_datasets = []

    # Ensure the fixed-grid sampler has enough solver points to choose from.
    cfg.meta_study.num_individuals_range = (3, 3)
    cfg.meta_study.time_num_steps = max(
        int(args.time_num_steps),
        int(args.max_num_obs) + int(args.fixed_grid_start_index) + 1,
    )

    synthetic_target_observation_config = ObservationsConfig(
        type="fixed_regular_grid",
        add_rem=False,
        split_past_future=False,
        max_num_obs=int(args.max_num_obs),
        fixed_grid_start_index=int(args.fixed_grid_start_index),
    )

    datamodule = AICMECompartmentsDataModule(cfg)
    datamodule.prepare_data()
    return datamodule, synthetic_target_observation_config


def _validate_python_executable(python_executable: Path) -> Path:
    """Validate the external Python executable used for the standalone runner."""

    resolved = python_executable.expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Configured --python-executable was not found: '{resolved}'.")
    if not resolved.is_file():
        raise FileNotFoundError(f"Configured --python-executable is not a file: '{resolved}'.")
    return resolved


def _extract_single_study_series(
    study_index: int,
    batch: object,
) -> SyntheticStudySeries:
    """Extract one study from a synthetic loader batch.

    The script enforces ``batch_size=1`` so each loader item contains exactly
    one study. Target tensors are converted to the MMD runner layout:

    - ``values``: ``[It, Tobs, 1]``
    - ``times``: ``[It, Tobs, 1]``
    - ``mask``: ``[It, Tobs]``
    """

    batch_size = int(batch.target_obs.shape[0])
    if batch_size != 1:
        raise ValueError(
            "inspect_synthetic_loader_pairwise_mmd expects synthetic loader batches "
            f"with batch_size=1, got batch_size={batch_size}."
        )

    # `target_obs`: [1, It, Tobs, 1]
    target_values = batch.target_obs.detach().cpu()[0]
    # `target_obs_time`: [1, It, Tobs, 1]
    target_times = batch.target_obs_time.detach().cpu()[0]
    # `target_obs_mask`: [1, It, Tobs]
    target_mask = batch.target_obs_mask.bool().detach().cpu()[0]
    # `mask_target_individuals`: [1, It]
    target_individual_mask = batch.mask_target_individuals.bool().detach().cpu()[0].unsqueeze(-1)

    valid_mask = target_mask & target_individual_mask
    values = target_values * valid_mask.unsqueeze(-1).to(dtype=target_values.dtype)
    times = target_times * valid_mask.unsqueeze(-1).to(dtype=target_times.dtype)

    raw_study_name = batch.study_name[0] if getattr(batch, "study_name", None) else None
    study_label = str(raw_study_name or f"study_{study_index:03d}")
    return SyntheticStudySeries(
        study_label=f"{study_index:03d}_{study_label}",
        values=values,
        times=times,
        mask=valid_mask,
    )


def _collect_synthetic_studies(
    datamodule: AICMECompartmentsDataModule,
    synthetic_target_observation_config: ObservationsConfig,
    *,
    n_targets: int,
    dataset_size: int,
) -> list[SyntheticStudySeries]:
    """Collect synthetic studies from the same loader surface used by the task."""

    loader = datamodule.get_synthetic_experiment_dataloader(
        n_targets=n_targets,
        n_dosings=1,
        dosing_mode="diverse_dosing",
        dataset_size=dataset_size,
        synthetic_target_observation_config=synthetic_target_observation_config,
        shuffle=False,
    )

    studies: list[SyntheticStudySeries] = []
    for batch_idx, batch_list in enumerate(loader):
        if not isinstance(batch_list, (list, tuple)):
            raise TypeError(
                "Synthetic experiment dataloader items must be lists or tuples of databatches."
            )
        if len(batch_list) != 1:
            raise ValueError(
                "inspect_synthetic_loader_pairwise_mmd expects n_dosings=1 and therefore "
                f"one databatch per loader item, got {len(batch_list)}."
            )
        studies.append(_extract_single_study_series(batch_idx, batch_list[0]))

    if len(studies) != dataset_size:
        raise RuntimeError(
            "Collected synthetic study count does not match the requested dataset size: "
            f"requested={dataset_size}, collected={len(studies)}."
        )
    return studies


def _validate_shared_study_grid(studies: Sequence[SyntheticStudySeries]) -> None:
    """Ensure every study uses the same deterministic target schedule."""

    if not studies:
        raise ValueError("At least one synthetic study is required.")

    reference = studies[0]
    for study in studies[1:]:
        if study.values.shape != reference.values.shape:
            raise ValueError(
                "All synthetic studies must share the same target tensor shape for pairwise MMD. "
                f"Reference shape={tuple(reference.values.shape)}, "
                f"study '{study.study_label}' shape={tuple(study.values.shape)}."
            )
        if not torch.equal(study.mask, reference.mask):
            raise ValueError(
                "All synthetic studies must share the same target mask for pairwise MMD. "
                f"Study '{study.study_label}' differs from '{reference.study_label}'."
            )
        if not torch.allclose(study.times, reference.times, atol=1.0e-6, rtol=0.0):
            raise ValueError(
                "All synthetic studies must share the same target times for pairwise MMD. "
                f"Study '{study.study_label}' differs from '{reference.study_label}'."
            )


def _build_pairwise_payload(
    studies: Sequence[SyntheticStudySeries],
) -> tuple[
    TensorType["P", "It", "Tobs", 1],
    TensorType["P", "It", "Tobs", 1],
    TensorType["P", "It", "Tobs", 1],
    TensorType["P", "It", "Tobs"],
    list[tuple[str, str]],
]:
    """Stack all study pairs into one runner payload."""

    pair_labels = [(left.study_label, right.study_label) for left, right in combinations(studies, 2)]
    if not pair_labels:
        raise ValueError("At least two synthetic studies are required for pairwise MMD.")

    # `observed_stack`: [P, It, Tobs, 1]
    observed_stack = torch.stack(
        [left.values for left, _ in combinations(studies, 2)],
        dim=0,
    )
    # `generated_stack`: [P, It, Tobs, 1]
    generated_stack = torch.stack(
        [right.values for _, right in combinations(studies, 2)],
        dim=0,
    )
    # `times_stack`: [P, It, Tobs, 1]
    times_stack = torch.stack(
        [left.times for left, _ in combinations(studies, 2)],
        dim=0,
    )
    # `mask_stack`: [P, It, Tobs]
    mask_stack = torch.stack(
        [left.mask for left, _ in combinations(studies, 2)],
        dim=0,
    )
    return observed_stack, generated_stack, times_stack, mask_stack, pair_labels


def _write_pairwise_csv(
    output_path: Path,
    *,
    pair_labels: Sequence[tuple[str, str]],
    pairwise_mmd2: Sequence[float],
) -> None:
    """Write one CSV row per study pair."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["study_i", "study_j", "mmd2"])
        writer.writeheader()
        for (study_i, study_j), mmd2 in zip(pair_labels, pairwise_mmd2, strict=True):
            writer.writerow(
                {
                    "study_i": study_i,
                    "study_j": study_j,
                    "mmd2": float(mmd2),
                }
            )


def _plot_histogram(
    output_path: Path,
    *,
    pairwise_mmd2: np.ndarray,
    hist_bins: int,
    n_targets: int,
    dataset_size: int,
    signature_levels: int,
    estimator: str,
) -> None:
    """Render and save a histogram of pairwise ``MMD^2`` values."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.hist(pairwise_mmd2, bins=hist_bins, color="tab:blue", alpha=0.85, edgecolor="black")
    ax.set_xlabel("MMD$^2$")
    ax.set_ylabel("Count")
    ax.set_title("Pairwise Synthetic-Loader Signature MMD")
    summary_text = "\n".join(
        [
            f"n_targets={n_targets}",
            f"dataset_size={dataset_size}",
            f"pairs={pairwise_mmd2.size}",
            f"signature_levels={signature_levels}",
            f"estimator={estimator}",
            f"mean={pairwise_mmd2.mean():.6f}",
            f"std={pairwise_mmd2.std(ddof=0):.6f}",
            f"min={pairwise_mmd2.min():.6f}",
            f"max={pairwise_mmd2.max():.6f}",
        ]
    )
    ax.text(
        0.98,
        0.98,
        summary_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    if args.n_targets <= 0:
        raise ValueError("--n-targets must be positive.")
    if args.estimator == "unbiased" and args.n_targets <= 1:
        raise ValueError("--n-targets must be greater than 1 when estimator='unbiased'.")
    if args.dataset_size <= 1:
        raise ValueError("--dataset-size must be greater than 1 to form study pairs.")
    if args.signature_levels <= 0:
        raise ValueError("--signature-levels must be positive.")
    if args.max_num_obs <= 0:
        raise ValueError("--max-num-obs must be positive.")
    if args.hist_bins <= 0:
        raise ValueError("--hist-bins must be positive.")

    _set_seed(int(args.seed))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    python_executable = _validate_python_executable(Path(args.python_executable))

    datamodule, synthetic_target_observation_config = _build_datamodule(args)
    studies = _collect_synthetic_studies(
        datamodule,
        synthetic_target_observation_config,
        n_targets=int(args.n_targets),
        dataset_size=int(args.dataset_size),
    )
    _validate_shared_study_grid(studies)

    observed_stack, generated_stack, times_stack, mask_stack, pair_labels = _build_pairwise_payload(
        studies
    )

    payload_path = output_dir / "pairwise_mmd_payload.npz"
    result_path = output_dir / "pairwise_mmd_runner_result.json"
    csv_path = output_dir / "pairwise_mmd_values.csv"
    summary_path = output_dir / "pairwise_mmd_summary.json"
    histogram_path = output_dir / "pairwise_mmd_histogram.png"

    _write_synthetic_mmd_payload(
        payload_path,
        observed_values=observed_stack,
        generated_values=generated_stack,
        times=times_stack,
        mask=mask_stack,
    )
    result = _run_signature_mmd_runner(
        python_executable=python_executable,
        payload_path=payload_path,
        output_path=result_path,
        signature_levels=int(args.signature_levels),
        estimator=str(args.estimator),
        include_time_channel=bool(args.include_time_channel),
    )

    pairwise_mmd2 = np.asarray(result["per_batch_mmd2"], dtype=np.float64)
    if pairwise_mmd2.shape[0] != len(pair_labels):
        raise RuntimeError(
            "Runner returned an unexpected number of pairwise MMD values: "
            f"expected={len(pair_labels)}, got={pairwise_mmd2.shape[0]}."
        )

    _write_pairwise_csv(
        csv_path,
        pair_labels=pair_labels,
        pairwise_mmd2=pairwise_mmd2.tolist(),
    )
    _plot_histogram(
        histogram_path,
        pairwise_mmd2=pairwise_mmd2,
        hist_bins=int(args.hist_bins),
        n_targets=int(args.n_targets),
        dataset_size=int(args.dataset_size),
        signature_levels=int(args.signature_levels),
        estimator=str(args.estimator),
    )

    summary = {
        "config_path": str(args.config.resolve()),
        "python_executable": str(python_executable.resolve()),
        "n_targets": int(args.n_targets),
        "dataset_size": int(args.dataset_size),
        "num_pairs": int(len(pair_labels)),
        "signature_levels": int(args.signature_levels),
        "estimator": str(args.estimator),
        "include_time_channel": bool(args.include_time_channel),
        "max_num_obs": int(args.max_num_obs),
        "fixed_grid_start_index": int(args.fixed_grid_start_index),
        "time_num_steps": int(datamodule.meta_config.time_num_steps),
        "mean_mmd2": float(pairwise_mmd2.mean()),
        "std_mmd2": float(pairwise_mmd2.std(ddof=0)),
        "min_mmd2": float(pairwise_mmd2.min()),
        "max_mmd2": float(pairwise_mmd2.max()),
        "histogram_path": str(histogram_path),
        "csv_path": str(csv_path),
        "runner_result_path": str(result_path),
        "payload_path": str(payload_path),
        "pairs": [
            {
                "study_i": study_i,
                "study_j": study_j,
                "mmd2": float(mmd2),
            }
            for (study_i, study_j), mmd2 in zip(pair_labels, pairwise_mmd2.tolist(), strict=True)
        ],
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Collected {len(studies)} synthetic studies with {args.n_targets} targets each.")
    print(f"Computed {len(pair_labels)} pairwise MMD^2 values.")
    print(f"Mean MMD^2: {pairwise_mmd2.mean():.6f}")
    print(f"Std  MMD^2: {pairwise_mmd2.std(ddof=0):.6f}")
    print(f"Min  MMD^2: {pairwise_mmd2.min():.6f}")
    print(f"Max  MMD^2: {pairwise_mmd2.max():.6f}")
    print(f"Histogram: {histogram_path}")
    print(f"CSV: {csv_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
