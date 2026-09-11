"""Canonical empirical PK scheduler task entrypoints.

This module owns scheduler task implementations that operate on empirical
datasets. Shared helpers and backward-compatible aliases remain in
:mod:`pk_tasks`.
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

from pff.data.datasets.aicme_datasets import AICMECompartmentsDataBatch
from pff.metrics.sample_distance_metrics import (
    _resolve_classifier_auc_cfg,
    _resolve_distance_metric_names,
    _resolve_mmd_task_cfg,
    _run_classifier_auc_distance,
    _run_mmd2_distance,
)
from pff.training.callbacks import pk_tasks as _shared


def _resolve_empirical_repo_ids(
    *,
    task_cfg: Mapping[str, Any],
    pl_module: Any,
) -> list[str]:
    """Resolve one explicit repo override or all configured empirical repos."""

    empirical_name = str(task_cfg.get("empirical_name", "")).strip()
    if empirical_name:
        return [empirical_name]

    mix_cfg = getattr(getattr(pl_module, "model_config", None), "mix_data", None)
    resolved: list[str] = []
    seen: set[str] = set()
    for raw_repo_id in list(getattr(mix_cfg, "test_empirical_datasets", []) or []):
        repo_id = str(raw_repo_id).strip()
        if not repo_id or repo_id in seen:
            continue
        seen.add(repo_id)
        resolved.append(repo_id)
    return resolved


def _get_empirical_batches_for_predictive_eval(
    *,
    datamodule: Any,
    split: str,
    empirical_name: str,
    pl_module: Any,
) -> list[AICMECompartmentsDataBatch]:
    """Fetch empirical held-out batches with fixed past selection when configured."""

    if str(split).strip().lower() != "empirical_heldout":
        return datamodule.get_empirical_batches(
            split=split,
            empirical_name=empirical_name,
        )

    mix_cfg = getattr(getattr(pl_module, "model_config", None), "mix_data", None)
    raw_fix_past_value = getattr(mix_cfg, "evaluate_prediction_steps_past", None)
    if raw_fix_past_value is None or not callable(getattr(datamodule, "fix_past_selection", None)):
        return datamodule.get_empirical_batches(
            split=split,
            empirical_name=empirical_name,
        )

    datamodule.fix_past_selection(int(raw_fix_past_value), who="target")
    try:
        return datamodule.get_empirical_batches(
            split=split,
            empirical_name=empirical_name,
        )
    finally:
        releaser = getattr(datamodule, "release_past_selection", None)
        if callable(releaser):
            releaser(who="target")


def _resolve_empirical_distance_metric_names(task_cfg: Mapping[str, Any]) -> list[str]:
    """Resolve held-out empirical distance metrics with a classifier-only fallback."""

    if task_cfg.get("distance_metrics") is None:
        return ["classifier_auc"]
    return _resolve_distance_metric_names(task_cfg)


def _pool_empirical_mmd_bundle(
    bundle: _shared.SyntheticMMDSeriesBundle,
) -> _shared.SyntheticMMDSeriesBundle:
    """Collapse all empirical series into one batch element for pooled MMD."""

    observed_values = bundle.observed_values.detach().cpu()  # [B, It, Tobs, 1]
    generated_values = bundle.generated_values.detach().cpu()  # [B, It, Tobs, 1]
    times = bundle.times.detach().cpu()  # [B, It, Tobs, 1]
    mask = bundle.mask.detach().cpu()  # [B, It, Tobs]

    if observed_values.ndim != 4 or generated_values.ndim != 4 or times.ndim != 4 or mask.ndim != 3:
        raise ValueError(
            "Empirical held-out MMD pooling expects shapes "
            "[B, It, Tobs, 1] for values/times and [B, It, Tobs] for the mask."
        )

    batch_size, num_targets, num_times, num_features = observed_values.shape
    pooled_num_targets = int(batch_size * num_targets)
    return _shared.SyntheticMMDSeriesBundle(
        observed_values=observed_values.reshape(1, pooled_num_targets, num_times, num_features),
        generated_values=generated_values.reshape(1, pooled_num_targets, num_times, num_features),
        times=times.reshape(1, pooled_num_targets, num_times, num_features),
        mask=mask.reshape(1, pooled_num_targets, num_times),
    )


def task_empirical_predictive_metrics(
    *,
    samples: Any | None,
    batches: Sequence[AICMECompartmentsDataBatch],
    task_cfg: Mapping[str, Any],
    trainer: Any,
    pl_module: Any,
) -> Mapping[str, Any]:
    """Run held-out empirical predictive metrics pooled across empirical repos.

    By default this canonical task reads the configured empirical repo list from
    ``pl_module.model_config.mix_data.test_empirical_datasets`` and pools raw
    held-out target observations across all repos before aggregating one final
    mean/std per substance.

    ``task_cfg.empirical_name`` is still accepted as a temporary one-repo
    override for compatibility, but the preferred task meaning is cross-repo
    per-substance aggregation.
    """

    del samples
    del batches

    datamodule = getattr(trainer, "datamodule", None)
    if datamodule is None:
        raise ValueError("task_empirical_predictive_metrics requires trainer.datamodule.")

    split = str(task_cfg.get("split", "empirical_heldout")).strip() or "empirical_heldout"
    sample_size = int(task_cfg.get("sample_size", 0))
    empirical_repos = _resolve_empirical_repo_ids(task_cfg=task_cfg, pl_module=pl_module)
    if not empirical_repos:
        return {}

    aggregated = _shared._compute_empirical_predictive_metrics_across_repos(
        empirical_repos=empirical_repos,
        datamodule=datamodule,
        split=split,
        model=pl_module,
        sample_size=sample_size,
        batch_fetcher=lambda *, split, empirical_name: _get_empirical_batches_for_predictive_eval(
            datamodule=datamodule,
            split=split,
            empirical_name=empirical_name,
            pl_module=pl_module,
        ),
    )
    return _shared._flatten_aggregated_prediction_metrics(aggregated)


def task_empirical_heldout_generated_classifier(
    *,
    samples: Any | None,
    batches: Sequence[AICMECompartmentsDataBatch],
    task_cfg: Mapping[str, Any],
    trainer: Any,
    pl_module: Any,
) -> Mapping[str, Any]:
    """Compute held-out empirical sampled distances on pooled held-out targets."""

    del samples
    del batches

    datamodule = getattr(trainer, "datamodule", None)
    if datamodule is None:
        raise ValueError(
            "task_empirical_heldout_generated_classifier requires trainer.datamodule."
        )

    split = str(task_cfg.get("split", "empirical_heldout")).strip() or "empirical_heldout"
    empirical_repos = _resolve_empirical_repo_ids(task_cfg=task_cfg, pl_module=pl_module)
    if not empirical_repos:
        return {}

    requested_metrics = _resolve_empirical_distance_metric_names(task_cfg)
    save_details = bool(task_cfg.get("save_details", True))
    default_repo_id = empirical_repos[0] if len(empirical_repos) == 1 else "all_empirical_repos"
    repo_id = _shared._resolve_repo_id(task_cfg, default=default_repo_id) or default_repo_id
    num_steps_raw = task_cfg.get("num_steps")
    num_steps = int(num_steps_raw) if num_steps_raw is not None else None
    batch_list: list[AICMECompartmentsDataBatch] = []
    for empirical_name in empirical_repos:
        batch_list.extend(
            _shared._as_batch_list(
                datamodule.get_empirical_batches(
                    split=split,
                    empirical_name=empirical_name,
                )
            )
        )

    collection, output_substances = _shared._collect_empirical_heldout_generated_classifier_collection(
        batch_list,
        model=pl_module,
        num_steps=num_steps,
    )
    if not output_substances:
        return {}
    num_valid_series = int(collection.dataset_bundle.mask.any(dim=-1).sum().item())

    task_root = _shared._resolve_task_root(trainer, "empirical_heldout_generated_classifier")
    repo_suffix = str(repo_id).split("/")[-1] or "empirical"
    run_name = (
        "epoch_"
        f"{_shared._resolve_epoch(trainer):03d}_step_{int(getattr(trainer, 'global_step', 0)):07d}"
    )
    persistent_run_dir = task_root / f"{run_name}_{repo_suffix}"
    if save_details:
        if persistent_run_dir.exists():
            shutil.rmtree(persistent_run_dir)
        persistent_run_dir.mkdir(parents=True, exist_ok=True)

    context_manager: Any
    if save_details:
        context_manager = contextlib.nullcontext(persistent_run_dir)
    else:
        context_manager = tempfile.TemporaryDirectory(
            prefix="empirical_heldout_generated_classifier_"
        )

    outputs: dict[str, Any] = {}
    with context_manager as temp_root:
        run_root = Path(temp_root)
        if "classifier_auc" in requested_metrics:
            if num_valid_series < 2:
                warnings.warn(
                    "Skipping empirical held-out generated classifier because fewer than two valid "
                    f"held-out target series remain after filtering; got {num_valid_series}.",
                    stacklevel=2,
                )
            else:
                classifier_cfg = _resolve_classifier_auc_cfg(task_cfg)
                if str(classifier_cfg["mode"]) != "joint":
                    raise ValueError(
                        "task_empirical_heldout_generated_classifier supports only "
                        "classifier_auc.mode='joint'."
                    )
                outputs.update(
                    _run_classifier_auc_distance(
                        collection=collection,
                        classifier_cfg=classifier_cfg,
                        output_root=run_root,
                        save_details=save_details,
                    )
                )
        if "mmd2" in requested_metrics:
            mmd_cfg = _resolve_mmd_task_cfg(task_cfg)
            estimator = str(mmd_cfg.get("estimator", "unbiased")).strip().lower() or "unbiased"
            min_valid_series = 2 if estimator == "unbiased" else 1
            if num_valid_series < min_valid_series:
                warnings.warn(
                    "Skipping empirical held-out generated mmd2 because too few valid held-out "
                    f"target series remain after filtering for estimator='{estimator}'; "
                    f"got {num_valid_series}, need at least {min_valid_series}.",
                    stacklevel=2,
                )
            else:
                pooled_mmd_bundle = _pool_empirical_mmd_bundle(collection.dataset_bundle)
                outputs.update(
                    _run_mmd2_distance(
                        bundle=pooled_mmd_bundle,
                        mmd_cfg=mmd_cfg,
                        output_root=run_root,
                        save_details=save_details,
                    )
                )
    return outputs


def task_empirical_summary(
    *,
    samples: _shared.PKTaskSamples | _shared.PredictiveTaskSamples | list[Any] | None,
    batches: Sequence[AICMECompartmentsDataBatch],
    task_cfg: Mapping[str, Any],
    trainer: Any,
    pl_module: Any,
) -> Mapping[str, Any]:
    """Compute one explicit cross-repo empirical summary scalar."""

    del samples
    del batches

    metric_name = str(task_cfg.get("summary_metric", "")).strip()
    selected_drugs = list(task_cfg.get("selected_summary_drugs", []) or [])
    summary_scope = str(task_cfg.get("summary_scope", "full")).strip().lower() or "full"
    if not metric_name or not selected_drugs:
        return {}

    datamodule = getattr(trainer, "datamodule", None)
    if datamodule is None:
        return {}

    mix_cfg = getattr(getattr(pl_module, "model_config", None), "mix_data", None)
    empirical_repos = list(getattr(mix_cfg, "test_empirical_datasets", []) or [])
    metrics_by_repo: dict[str, dict[str, dict[str, float]]] = {}

    for repo_id in empirical_repos:
        per_repo: dict[str, dict[str, float]] = {}
        if summary_scope in {"predictive", "full"}:
            predictive_batches = _get_empirical_batches_for_predictive_eval(
                datamodule=datamodule,
                split="empirical_heldout",
                empirical_name=repo_id,
                pl_module=pl_module,
            )
            for substance, metric_dict in _shared._compute_empirical_predictive_metrics_from_batch_list(
                predictive_batches,
                model=pl_module,
                sample_size=1,
                repo_id=repo_id,
            ).items():
                per_repo[substance] = metric_dict
        elif summary_scope == "heldout_classifier":
            classifier_cfg = _shared._resolve_classifier_auc_cfg(task_cfg)
            if str(classifier_cfg["mode"]) != "joint":
                raise ValueError(
                    "task_empirical_summary with summary_scope='heldout_classifier' supports "
                    "only classifier_auc.mode='joint'."
                )
            classifier_batches = datamodule.get_empirical_batches(
                split="empirical_heldout",
                empirical_name=repo_id,
            )
            classifier_metrics, _ = (
                _shared._compute_empirical_heldout_generated_classifier_metrics_from_batch_list(
                    classifier_batches,
                    model=pl_module,
                    repo_id=repo_id,
                    classifier_cfg=classifier_cfg,
                    num_steps=(
                        int(task_cfg["num_steps"]) if task_cfg.get("num_steps") is not None else None
                    ),
                )
            )
            per_repo.update(classifier_metrics)

        if summary_scope == "full":
            generative_batches = datamodule.get_empirical_batches(
                split="empirical_no_heldout",
                empirical_name=repo_id,
            )

            batch_list = _shared._as_batch_list(generative_batches)
            if batch_list:
                batch = batch_list[0]
                batch_device = batch.to(pl_module.device) if hasattr(batch, "to") else batch
                payload = _shared.build_pk_task_samples(pl_module, batch_device, num_samples=1)
                generative_metrics = _shared._generative_metrics_per_batch(
                    payload.generative,
                    batch_device,
                )
                vpc_metrics = _shared._vpc_npde_pvalues_per_batch(payload.vpc, batch_device)
                merged_tensors = dict(generative_metrics)
                merged_tensors.update(vpc_metrics)
                for substance, metric_dict in _shared._aggregate_tensor_metrics(
                    metrics=merged_tensors,
                    batch=batch_device,
                    repo_id=repo_id,
                ).items():
                    repo_metrics = per_repo.setdefault(substance, {})
                    repo_metrics.update(metric_dict)

        if per_repo:
            metrics_by_repo[str(repo_id)] = per_repo

    summary_value = _shared.summarize_across_repos(
        metrics_by_repo,
        selected_drugs=selected_drugs,
        metric_name=metric_name,
    )
    if summary_value is None:
        return {}
    return {metric_name: float(summary_value)}


__all__ = [
    "task_empirical_heldout_generated_classifier",
    "task_empirical_predictive_metrics",
    "task_empirical_summary",
]
