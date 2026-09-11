"""Distance utilities for comparing sampled PK trajectories.

These helpers are shared by synthetic-experiment scheduler tasks that compare
observed versus generated target series using either a signature-kernel MMD or
an in-process classifier AUC.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from tqdm.auto import tqdm


def _resolve_mmd_runner_path() -> Path:
    """Return the standalone signature-kernel MMD runner script path."""

    return Path(__file__).resolve().parents[2] / "scripts" / "metrics" / "run_signature_mmd.py"


def _resolve_python_executable(task_cfg: Mapping[str, Any]) -> Path:
    """Resolve and validate the configured Python interpreter for external MMD runs."""

    raw_python = str(task_cfg.get("python_executable", "")).strip()
    if not raw_python:
        raise ValueError(
            "task_cfg.python_executable is required for "
            "task_diverse_synthetic_experiment_sample_distances when computing mmd2."
        )

    python_path = Path(raw_python).expanduser()
    if not python_path.exists():
        raise FileNotFoundError(f"Configured python_executable does not exist: '{python_path}'.")
    if not python_path.is_file():
        raise FileNotFoundError(f"Configured python_executable is not a file: '{python_path}'.")
    if not python_path.stat().st_mode & 0o111:
        raise PermissionError(f"Configured python_executable is not executable: '{python_path}'.")
    return python_path


def _write_synthetic_mmd_payload(
    output_path: Path,
    *,
    observed_values: torch.Tensor,
    generated_values: torch.Tensor,
    times: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Serialize aligned MMD tensors to a compact ``.npz`` payload."""

    np.savez_compressed(
        output_path,
        observed_values=observed_values.detach().cpu().numpy(),
        generated_values=generated_values.detach().cpu().numpy(),
        times=times.detach().cpu().numpy(),
        mask=mask.detach().cpu().numpy().astype(np.bool_),
    )


def _resolve_distance_metric_names(task_cfg: Mapping[str, Any]) -> list[str]:
    """Resolve the requested diverse-synthetic sample-distance metrics."""

    raw_metrics = task_cfg.get("distance_metrics", ["mmd2"])
    if isinstance(raw_metrics, str):
        metrics = [raw_metrics]
    else:
        metrics = [str(metric) for metric in list(raw_metrics or [])]

    resolved: list[str] = []
    valid = {"mmd2", "classifier_auc"}
    for metric in metrics:
        metric_name = str(metric).strip().lower()
        if not metric_name:
            continue
        if metric_name not in valid:
            raise ValueError(
                "task_diverse_synthetic_experiment_sample_distances supports only "
                f"{sorted(valid)}, got '{metric_name}'."
            )
        if metric_name not in resolved:
            resolved.append(metric_name)

    if not resolved:
        raise ValueError(
            "task_diverse_synthetic_experiment_sample_distances requires at least one distance metric."
        )
    return resolved


def _resolve_mmd_task_cfg(task_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve nested MMD config with flat-key fallback from the legacy task surface."""

    raw_nested = task_cfg.get("mmd")
    nested_cfg = dict(raw_nested) if isinstance(raw_nested, Mapping) else {}
    return {
        "python_executable": nested_cfg.get(
            "python_executable",
            task_cfg.get("python_executable"),
        ),
        "signature_levels": nested_cfg.get(
            "signature_levels",
            task_cfg.get("signature_levels", 4),
        ),
        "estimator": nested_cfg.get(
            "estimator",
            task_cfg.get("estimator", "unbiased"),
        ),
        "include_time_channel": nested_cfg.get(
            "include_time_channel",
            task_cfg.get("include_time_channel", True),
        ),
    }


def _resolve_classifier_auc_cfg(task_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve classifier-AUC configuration with deterministic defaults."""

    raw_nested = task_cfg.get("classifier_auc")
    nested_cfg = dict(raw_nested) if isinstance(raw_nested, Mapping) else {}
    mode = str(nested_cfg.get("mode", "joint")).strip().lower() or "joint"
    if mode not in {"joint", "per_bundle_mean"}:
        raise ValueError(
            f"classifier_auc.mode must be either 'joint' or 'per_bundle_mean', got '{mode}'."
        )

    hidden_dim = int(nested_cfg.get("hidden_dim", 64))
    num_hidden_layers = int(nested_cfg.get("num_hidden_layers", 2))
    learning_rate = float(nested_cfg.get("learning_rate", 1.0e-3))
    weight_decay = float(nested_cfg.get("weight_decay", 1.0e-4))
    epochs = int(nested_cfg.get("epochs", 100))
    batch_size = int(nested_cfg.get("batch_size", 128))
    seed = int(nested_cfg.get("seed", 0))
    show_progress = bool(nested_cfg.get("show_progress", True))
    include_time_channel = bool(
        nested_cfg.get(
            "include_time_channel",
            task_cfg.get("include_time_channel", True),
        )
    )

    if hidden_dim <= 0:
        raise ValueError("classifier_auc.hidden_dim must be > 0.")
    if num_hidden_layers < 0:
        raise ValueError("classifier_auc.num_hidden_layers must be >= 0.")
    if learning_rate <= 0.0:
        raise ValueError("classifier_auc.learning_rate must be > 0.")
    if weight_decay < 0.0:
        raise ValueError("classifier_auc.weight_decay must be >= 0.")
    if epochs <= 0:
        raise ValueError("classifier_auc.epochs must be > 0.")
    if batch_size <= 0:
        raise ValueError("classifier_auc.batch_size must be > 0.")

    return {
        "mode": mode,
        "include_time_channel": include_time_channel,
        "hidden_dim": hidden_dim,
        "num_hidden_layers": num_hidden_layers,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "show_progress": show_progress,
    }


def _average_rankdata(scores: np.ndarray) -> np.ndarray:
    """Return 1-indexed average ranks for a 1D score vector."""

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_ranks = np.empty(scores.shape[0], dtype=np.float64)

    start = 0
    while start < sorted_scores.shape[0]:
        end = start + 1
        while end < sorted_scores.shape[0] and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        sorted_ranks[start:end] = average_rank
        start = end

    ranks = np.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


def _binary_roc_auc_from_scores(
    labels: torch.Tensor,
    scores: torch.Tensor,
) -> float:
    """Compute binary ROC AUC from logits using rank statistics."""

    labels_cpu = labels.detach().cpu().to(dtype=torch.int64).view(-1)
    scores_cpu = scores.detach().cpu().to(dtype=torch.float64).view(-1)
    if labels_cpu.numel() != scores_cpu.numel():
        raise ValueError("AUC labels and scores must have the same number of elements.")

    positive_mask = labels_cpu == 1
    negative_mask = labels_cpu == 0
    num_positive = int(positive_mask.sum().item())
    num_negative = int(negative_mask.sum().item())
    if num_positive == 0 or num_negative == 0:
        raise ValueError("ROC AUC requires at least one positive and one negative example.")

    ranks = _average_rankdata(scores_cpu.numpy())
    sum_positive_ranks = float(ranks[positive_mask.numpy()].sum())
    auc = (sum_positive_ranks - (num_positive * (num_positive + 1) / 2.0)) / float(
        num_positive * num_negative
    )
    return float(auc)


def _build_classifier_feature_matrix(
    bundle: Any,
    *,
    include_time_channel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten aligned target series into classifier-ready examples."""

    valid_mask = bundle.mask.detach().cpu().bool()  # [B, It, Tobs]
    valid_series_mask = valid_mask.any(dim=-1).reshape(-1)  # [B * It]
    if not valid_series_mask.any():
        return (
            torch.zeros((0, 0), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
        )

    mask_channel = valid_mask.unsqueeze(-1).to(dtype=torch.float32)  # [B, It, Tobs, 1]
    observed_values = bundle.observed_values.detach().cpu().float() * mask_channel
    generated_values = bundle.generated_values.detach().cpu().float() * mask_channel
    times = bundle.times.detach().cpu().float() * mask_channel

    # observed_values/generated_values/times: [B, It, Tobs, 1]
    # mask_channel: [B, It, Tobs, 1]
    if include_time_channel:
        observed_channels = torch.cat([times, observed_values, mask_channel], dim=-1)
        generated_channels = torch.cat([times, generated_values, mask_channel], dim=-1)
    else:
        observed_channels = torch.cat([observed_values, mask_channel], dim=-1)
        generated_channels = torch.cat([generated_values, mask_channel], dim=-1)

    observed_features = observed_channels.reshape(
        observed_channels.shape[0] * observed_channels.shape[1], -1
    )
    generated_features = generated_channels.reshape(
        generated_channels.shape[0] * generated_channels.shape[1],
        -1,
    )
    observed_features = observed_features[valid_series_mask]
    generated_features = generated_features[valid_series_mask]

    num_series = int(observed_features.shape[0])
    labels = torch.cat(
        [
            torch.ones(num_series, dtype=torch.float32),
            torch.zeros(num_series, dtype=torch.float32),
        ],
        dim=0,
    )
    features = torch.cat([observed_features, generated_features], dim=0).to(dtype=torch.float32)
    return features, labels


def _build_classifier_mlp(
    *,
    input_dim: int,
    hidden_dim: int,
    num_hidden_layers: int,
) -> torch.nn.Module:
    """Build the in-process classifier used for classifier AUC."""

    layers: list[torch.nn.Module] = []
    current_dim = int(input_dim)
    for _ in range(int(num_hidden_layers)):
        layers.append(torch.nn.Linear(current_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
        current_dim = hidden_dim
    layers.append(torch.nn.Linear(current_dim, 1))
    return torch.nn.Sequential(*layers)


def _train_classifier_auc_model(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    hidden_dim: int,
    num_hidden_layers: int,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    seed: int,
    show_progress: bool,
    progress_desc: str = "Training classifier_auc",
) -> dict[str, Any]:
    """Train one small CPU MLP and return the train-set ROC AUC."""

    num_positive = int((labels == 1).sum().item())
    num_negative = int((labels == 0).sum().item())
    if num_positive < 2 or num_negative < 2:
        raise ValueError(
            "classifier_auc requires at least 2 examples per class, got "
            f"{num_positive} positive and {num_negative} negative."
        )

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    classifier = _build_classifier_mlp(
        input_dim=int(features.shape[1]),
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
    ).to(torch.device("cpu"))
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    num_examples = int(features.shape[0])
    actual_batch_size = min(int(batch_size), num_examples)
    loss_history: list[float] = []

    epoch_iterator = range(int(epochs))
    if show_progress:
        epoch_iterator = tqdm(
            epoch_iterator,
            desc=progress_desc,
            total=int(epochs),
            leave=False,
            ncols=100,
        )

    for _ in epoch_iterator:
        permutation = torch.randperm(num_examples, generator=generator)
        epoch_loss = 0.0
        seen_examples = 0

        for batch_start in range(0, num_examples, actual_batch_size):
            batch_indices = permutation[batch_start : batch_start + actual_batch_size]
            batch_features = features[batch_indices]  # [Bs, D]
            batch_labels = labels[batch_indices].unsqueeze(-1)  # [Bs, 1]

            optimizer.zero_grad(set_to_none=True)
            logits = classifier(batch_features)  # [Bs, 1]
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()

            batch_count = int(batch_indices.numel())
            epoch_loss += float(loss.item()) * batch_count
            seen_examples += batch_count

        epoch_mean_loss = epoch_loss / float(max(seen_examples, 1))
        loss_history.append(epoch_mean_loss)
        if show_progress:
            epoch_iterator.set_postfix(loss=f"{epoch_mean_loss:.4f}")

    with torch.inference_mode():
        logits_full = classifier(features).squeeze(-1)  # [N]

    train_auc = _binary_roc_auc_from_scores(labels, logits_full)
    loss_summary = {
        "num_epochs": int(epochs),
        "initial_loss": float(loss_history[0]),
        "final_loss": float(loss_history[-1]),
        "best_loss": float(min(loss_history)),
    }
    return {
        "classifier_auc": float(train_auc),
        "loss_summary": loss_summary,
        "num_examples": num_examples,
        "num_positive": num_positive,
        "num_negative": num_negative,
    }


def _run_signature_mmd_runner(
    *,
    python_executable: Path,
    payload_path: Path,
    output_path: Path,
    signature_levels: int,
    estimator: str,
    include_time_channel: bool,
) -> dict[str, Any]:
    """Run the standalone signature-kernel MMD script and parse its JSON output."""

    runner_path = _resolve_mmd_runner_path()
    if not runner_path.exists():
        raise FileNotFoundError(f"MMD runner script not found: '{runner_path}'.")

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
        str(estimator),
        "--include-time-channel",
        "1" if include_time_channel else "0",
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Synthetic MMD runner failed with exit code "
            f"{exc.returncode}. stdout={exc.stdout!r} stderr={exc.stderr!r}"
        ) from exc

    with output_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise TypeError("Synthetic MMD runner output must be a JSON object.")
    return result


def _run_mmd2_distance(
    *,
    bundle: Any,
    mmd_cfg: Mapping[str, Any],
    output_root: Path,
    save_details: bool,
) -> dict[str, Any]:
    """Run the external signature-kernel MMD distance on the collected dataset bundle."""

    signature_levels = int(mmd_cfg.get("signature_levels", 4))
    estimator = str(mmd_cfg.get("estimator", "unbiased")).strip().lower() or "unbiased"
    include_time_channel = bool(mmd_cfg.get("include_time_channel", True))
    if signature_levels <= 0:
        raise ValueError(
            "task_diverse_synthetic_experiment_sample_distances requires signature_levels > 0."
        )
    if estimator not in {"biased", "unbiased"}:
        raise ValueError(
            "task_diverse_synthetic_experiment_sample_distances estimator must be either "
            "'biased' or 'unbiased'."
        )

    python_executable = _resolve_python_executable(mmd_cfg)
    payload_path = output_root / "mmd_payload.npz"
    result_path = output_root / "mmd_result.json"
    _write_synthetic_mmd_payload(
        payload_path,
        observed_values=bundle.observed_values,
        generated_values=bundle.generated_values,
        times=bundle.times,
        mask=bundle.mask,
    )
    result = _run_signature_mmd_runner(
        python_executable=python_executable,
        payload_path=payload_path,
        output_path=result_path,
        signature_levels=signature_levels,
        estimator=estimator,
        include_time_channel=include_time_channel,
    )
    outputs: dict[str, Any] = {"mmd2": float(result["mean_mmd2"])}
    if save_details:
        outputs["mmd_details"] = result_path
        outputs["mmd_payload"] = payload_path
    return outputs


def _run_classifier_auc_distance(
    *,
    collection: Any,
    classifier_cfg: Mapping[str, Any],
    output_root: Path,
    save_details: bool,
) -> dict[str, Any]:
    """Train the in-process classifier and report train-set AUC."""

    mode = str(classifier_cfg["mode"])
    include_time_channel = bool(classifier_cfg["include_time_channel"])
    train_kwargs = {
        key: classifier_cfg[key]
        for key in (
            "hidden_dim",
            "num_hidden_layers",
            "learning_rate",
            "weight_decay",
            "epochs",
            "batch_size",
            "seed",
            "show_progress",
        )
    }
    detail_report: dict[str, Any] = {
        "mode": mode,
        "include_time_channel": include_time_channel,
        "hidden_dim": int(classifier_cfg["hidden_dim"]),
        "num_hidden_layers": int(classifier_cfg["num_hidden_layers"]),
        "learning_rate": float(classifier_cfg["learning_rate"]),
        "weight_decay": float(classifier_cfg["weight_decay"]),
        "epochs": int(classifier_cfg["epochs"]),
        "batch_size": int(classifier_cfg["batch_size"]),
        "seed": int(classifier_cfg["seed"]),
        "show_progress": bool(classifier_cfg["show_progress"]),
    }

    if mode == "joint":
        features, labels = _build_classifier_feature_matrix(
            collection.dataset_bundle,
            include_time_channel=include_time_channel,
        )
        result = _train_classifier_auc_model(
            features,
            labels,
            progress_desc="Training classifier_auc",
            **train_kwargs,
        )
        detail_report["num_bundles"] = int(len(collection.aligned_bundles))
        detail_report["classifier_auc"] = float(result["classifier_auc"])
        detail_report["train_result"] = result
        outputs: dict[str, Any] = {"classifier_auc": float(result["classifier_auc"])}
    else:
        per_bundle_results: list[dict[str, Any]] = []
        per_bundle_auc_values: list[float] = []
        for bundle_idx, bundle in enumerate(collection.aligned_bundles):
            features, labels = _build_classifier_feature_matrix(
                bundle,
                include_time_channel=include_time_channel,
            )
            positives = int((labels == 1).sum().item())
            negatives = int((labels == 0).sum().item())
            if positives < 2 or negatives < 2:
                per_bundle_results.append(
                    {
                        "bundle_idx": bundle_idx,
                        "skipped": True,
                        "num_positive": positives,
                        "num_negative": negatives,
                    }
                )
                continue

            result = _train_classifier_auc_model(
                features,
                labels,
                progress_desc=(
                    "Training classifier_auc "
                    f"bundle {bundle_idx + 1}/{len(collection.aligned_bundles)}"
                ),
                **train_kwargs,
            )
            per_bundle_results.append(
                {
                    "bundle_idx": bundle_idx,
                    "skipped": False,
                    **result,
                }
            )
            per_bundle_auc_values.append(float(result["classifier_auc"]))

        if not per_bundle_auc_values:
            raise ValueError(
                "classifier_auc.mode='per_bundle_mean' could not train on any bundle: "
                "all bundles had fewer than 2 examples per class."
            )

        mean_auc = float(sum(per_bundle_auc_values) / len(per_bundle_auc_values))
        detail_report["classifier_auc"] = mean_auc
        detail_report["per_bundle_results"] = per_bundle_results
        outputs = {"classifier_auc": mean_auc}

    if save_details:
        details_path = output_root / "classifier_auc_details.json"
        with details_path.open("w", encoding="utf-8") as handle:
            json.dump(detail_report, handle, indent=2)
        outputs["classifier_auc_details"] = details_path
    return outputs


__all__ = [
    "_binary_roc_auc_from_scores",
    "_resolve_classifier_auc_cfg",
    "_resolve_distance_metric_names",
    "_resolve_mmd_task_cfg",
    "_run_classifier_auc_distance",
    "_run_mmd2_distance",
    "_run_signature_mmd_runner",
    "_train_classifier_auc_model",
    "_write_synthetic_mmd_payload",
]
