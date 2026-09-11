"""Canonical synthetic PK scheduler task entrypoints.

This module owns scheduler task implementations that operate on synthetic
validation batches or synthetic experiment datasets. Shared helpers and
backward-compatible aliases remain in :mod:`pk_tasks`.
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import torch

from pff.data.data_empirical.json_schema import StudyJSON
from pff.data.datasets.aicme_datasets import AICMECompartmentsDataBatch
from pff.metrics.sampling_quality import compute_vpc_data, vpc_plot
from pff.metrics.sample_distance_metrics import (
    _resolve_classifier_auc_cfg,
    _resolve_distance_metric_names,
    _resolve_mmd_task_cfg,
    _run_classifier_auc_distance,
    _run_mmd2_distance,
)
from pff.training.callbacks import pk_tasks as _shared


def task_predictive_images(
    *,
    samples: _shared.PKTaskSamples | _shared.PredictiveTaskSamples | list[Any] | None,
    batches: Sequence[AICMECompartmentsDataBatch],
    task_cfg: Mapping[str, Any],
    trainer: Any,
    pl_module: Any,
) -> Mapping[str, Any]:
    """Render predictive plots for a scheduler task group."""

    label = _shared._resolve_task_label(task_cfg)
    epoch = _shared._resolve_epoch(trainer)
    output_root = _shared._resolve_plot_root(trainer)
    model_label = str(task_cfg.get("model_label") or _shared._normalize_model_label(pl_module))
    plot_kwargs = dict(task_cfg.get("plot_kwargs", {}) or {})
    raw_plot_limit = task_cfg.get(
        "number_of_predictions_plot_per_drug", 1 if label == "Empirical" else None
    )
    number_of_predictions_plot_per_drug = (
        int(raw_plot_limit) if raw_plot_limit is not None else None
    )
    if number_of_predictions_plot_per_drug is not None:
        plot_kwargs["number_of_predictions_plot_per_drug"] = number_of_predictions_plot_per_drug
    empirical_plots_per_substance: dict[str, int] | None = {} if label == "Empirical" else None
    image_paths: list[str] = []
    for perm_index, (payload, batch) in enumerate(
        zip(_shared._as_sample_list(samples), _shared._as_batch_list(batches))
    ):
        image_paths.extend(
            _shared._predictive_images_per_batch(
                _shared._predictive_payload(payload),
                batch,
                model=pl_module,
                label=label,
                epoch=epoch,
                perm_index=perm_index,
                output_root=output_root,
                model_label=model_label,
                plot_kwargs=plot_kwargs or None,
                empirical_plots_per_substance=empirical_plots_per_substance,
                number_of_predictions_plot_per_drug=number_of_predictions_plot_per_drug,
            )
        )
    return _shared._image_outputs(image_paths)


def task_generative_metrics(
    *,
    samples: _shared.PKTaskSamples | list[Any] | None,
    batches: Sequence[AICMECompartmentsDataBatch],
    task_cfg: Mapping[str, Any],
    trainer: Any,
    pl_module: Any,
) -> Mapping[str, Any]:
    """Run generative metrics over one scheduler task group."""

    del trainer
    del pl_module
    repo_id = _shared._resolve_repo_id(task_cfg)
    metric_tensors: dict[str, list[torch.Tensor]] = {}
    for payload, batch in zip(_shared._as_sample_list(samples), _shared._as_batch_list(batches)):
        per_batch = _shared._generative_metrics_per_batch(payload.generative, batch)
        for metric_name, metric_tensor in per_batch.items():
            metric_tensors.setdefault(metric_name, []).append(metric_tensor)
    return _shared._flatten_tensor_metrics(
        _shared._aggregate_tensor_metrics(
            metrics=metric_tensors,
            batch=_shared._concatenate_aicme_plot_batches(_shared._as_batch_list(batches)),
            repo_id=repo_id,
        )
    )


def task_generative_images(
    *,
    samples: _shared.PKTaskSamples | list[Any] | None,
    batches: Sequence[AICMECompartmentsDataBatch],
    task_cfg: Mapping[str, Any],
    trainer: Any,
    pl_module: Any,
) -> Mapping[str, Any]:
    """Render generative plots for a scheduler task group."""

    label = _shared._resolve_task_label(task_cfg)
    epoch = _shared._resolve_epoch(trainer)
    output_root = _shared._resolve_plot_root(trainer)
    model_label = str(task_cfg.get("model_label") or _shared._normalize_model_label(pl_module))
    plot_kwargs = task_cfg.get("plot_kwargs")
    image_paths: list[str] = []
    for perm_index, (payload, batch) in enumerate(
        zip(_shared._as_sample_list(samples), _shared._as_batch_list(batches))
    ):
        image_paths.extend(
            _shared._generative_images_per_batch(
                payload.generative,
                batch,
                model=pl_module,
                label=label,
                epoch=epoch,
                perm_index=perm_index,
                output_root=output_root,
                model_label=model_label,
                plot_kwargs=dict(plot_kwargs) if isinstance(plot_kwargs, Mapping) else None,
            )
        )
    return _shared._image_outputs(image_paths)


def task_vpc_npde_pvalues(
    *,
    samples: _shared.PKTaskSamples | list[Any] | None,
    batches: Sequence[AICMECompartmentsDataBatch],
    task_cfg: Mapping[str, Any],
    trainer: Any,
    pl_module: Any,
) -> Mapping[str, Any]:
    """Compute VPC/NPDE p-values for a scheduler task group."""

    del trainer
    del pl_module
    repo_id = _shared._resolve_repo_id(task_cfg)
    metric_tensors: dict[str, list[torch.Tensor]] = {}
    for payload, batch in zip(_shared._as_sample_list(samples), _shared._as_batch_list(batches)):
        per_batch = _shared._vpc_npde_pvalues_per_batch(payload.vpc, batch)
        for metric_name, metric_tensor in per_batch.items():
            metric_tensors.setdefault(metric_name, []).append(metric_tensor)
    return _shared._flatten_tensor_metrics(
        _shared._aggregate_tensor_metrics(
            metrics=metric_tensors,
            batch=_shared._concatenate_aicme_plot_batches(_shared._as_batch_list(batches)),
            repo_id=repo_id,
        )
    )


def task_vpc_images(
    *,
    samples: _shared.PKTaskSamples | list[Any] | None,
    batches: Sequence[AICMECompartmentsDataBatch],
    task_cfg: Mapping[str, Any],
    trainer: Any,
    pl_module: Any,
) -> Mapping[str, Any]:
    """Render VPC plots for a scheduler task group."""

    label = _shared._resolve_task_label(task_cfg)
    epoch = _shared._resolve_epoch(trainer)
    output_root = _shared._resolve_plot_root(trainer)
    model_label = str(task_cfg.get("model_label") or _shared._normalize_model_label(pl_module))
    plot_kwargs = task_cfg.get("plot_kwargs")
    image_paths: list[str] = []
    for perm_index, (payload, batch) in enumerate(
        zip(_shared._as_sample_list(samples), _shared._as_batch_list(batches))
    ):
        image_paths.extend(
            _shared._vpc_images_per_batch(
                payload.vpc,
                batch,
                label=label,
                epoch=epoch,
                perm_index=perm_index,
                output_root=output_root,
                model_label=model_label,
                plot_kwargs=dict(plot_kwargs) if isinstance(plot_kwargs, Mapping) else None,
            )
        )
    return _shared._image_outputs(image_paths)


def _resolve_synthetic_vpc_task_sizes(task_cfg: Mapping[str, Any]) -> tuple[int, int, int]:
    """Resolve the synthetic paired-VPC generation sizes with task defaults."""

    n_cases = int(task_cfg.get("n_cases", 10))
    sample_size = int(task_cfg.get("sample_size", 500))
    n_observed_individuals = int(task_cfg.get("n_observed_individuals", 10))

    if n_cases <= 0:
        raise ValueError("task_synthetic_vpc_paired_images requires n_cases > 0.")
    if sample_size <= 0:
        raise ValueError("task_synthetic_vpc_paired_images requires sample_size > 0.")
    if n_observed_individuals <= 0:
        raise ValueError(
            "task_synthetic_vpc_paired_images requires n_observed_individuals > 0."
        )
    return n_cases, sample_size, n_observed_individuals


def _resolve_synthetic_vpc_datamodule(trainer: Any) -> Any:
    """Return the datamodule required by the synthetic paired-VPC task."""

    datamodule = getattr(trainer, "datamodule", None)
    if datamodule is None:
        raise ValueError(
            "task_synthetic_vpc_paired_images requires trainer.datamodule."
        )

    generate_cases = getattr(datamodule, "generate_synthetic_vpc_data_list", None)
    if not callable(generate_cases):
        raise ValueError(
            "task_synthetic_vpc_paired_images requires "
            "trainer.datamodule.generate_synthetic_vpc_data_list(...)."
        )

    build_batch = getattr(datamodule, "_build_synthetic_vpc_evaluation_batch", None)
    if not callable(build_batch):
        raise ValueError(
            "task_synthetic_vpc_paired_images requires the datamodule private helper "
            "_build_synthetic_vpc_evaluation_batch(...)."
        )
    return datamodule


def _resolve_model_vpc_sampler_owner(pl_module: Any) -> Any:
    """Resolve the object exposing ``sample_new_individuals_to_vpc_format``."""

    sampler = getattr(pl_module, "sample_new_individuals_to_vpc_format", None)
    if callable(sampler):
        return pl_module

    model = getattr(pl_module, "model", None)
    sampler = getattr(model, "sample_new_individuals_to_vpc_format", None)
    if callable(sampler):
        return model

    raise ValueError(
        "task_synthetic_vpc_paired_images requires "
        "sample_new_individuals_to_vpc_format(...) on pl_module or pl_module.model."
    )


def _extract_single_case_model_replicates(
    *,
    returned_studies: Any,
    case_index: int,
) -> list[StudyJSON]:
    """Validate and unpack one-case model VPC samples from the native ``[B][S]`` format."""

    if not isinstance(returned_studies, list):
        raise ValueError(
            "task_synthetic_vpc_paired_images expected model VPC samples as a list of "
            f"per-case study lists, got {type(returned_studies)!r}."
        )
    if len(returned_studies) != 1:
        raise ValueError(
            "task_synthetic_vpc_paired_images expected one replicate-study list for the "
            f"single-case evaluation batch at case_index={case_index}, got "
            f"{len(returned_studies)}."
        )

    case_replicates = returned_studies[0]
    if not isinstance(case_replicates, list):
        raise ValueError(
            "task_synthetic_vpc_paired_images expected the model VPC payload for one case "
            f"to be a list of StudyJSON objects, got {type(case_replicates)!r}."
        )
    return case_replicates


def task_synthetic_vpc_paired_images(
    *,
    samples: Any | None,
    batches: Sequence[AICMECompartmentsDataBatch],
    task_cfg: Mapping[str, Any],
    trainer: Any,
    pl_module: Any,
) -> Mapping[str, Any]:
    """Render one paired synthetic truth-vs-model VPC image for each generated case."""

    del samples
    del batches

    datamodule = _resolve_synthetic_vpc_datamodule(trainer)
    sampler_owner = _resolve_model_vpc_sampler_owner(pl_module)

    n_cases, sample_size, n_observed_individuals = _resolve_synthetic_vpc_task_sizes(task_cfg)
    epoch = _shared._resolve_epoch(trainer)
    output_root = _shared._resolve_plot_root(trainer)
    model_label = str(task_cfg.get("model_label") or _shared._normalize_model_label(pl_module))
    output_dir = output_root / model_label / "synthetic_vpc_paired"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_bins = int(task_cfg.get("n_bins", 10))
    binning = str(task_cfg.get("binning", "equal_count"))
    log_y = bool(task_cfg.get("log_y", False))
    num_steps = task_cfg.get("num_steps")

    synthetic_cases = datamodule.generate_synthetic_vpc_data_list(
        n_cases=n_cases,
        n_observed_individuals=n_observed_individuals,
        sample_size=sample_size,
    )

    image_paths: list[str] = []
    model_device = getattr(pl_module, "device", torch.device("cpu"))
    for case_index, (observed_study, truth_replicates) in enumerate(synthetic_cases):
        truth_vpc = compute_vpc_data(
            observed_study,
            truth_replicates,
            n_bins=n_bins,
            binning=binning,
        )

        batch = datamodule._build_synthetic_vpc_evaluation_batch(observed_study)
        batch_device = _shared._move_batch_to_device(batch, model_device)
        sample_kwargs: dict[str, Any] = {"sample_size": sample_size}
        if num_steps is not None:
            sample_kwargs["num_steps"] = num_steps
        with torch.inference_mode():
            returned_studies = sampler_owner.sample_new_individuals_to_vpc_format(
                batch_device,
                **sample_kwargs,
            )
        model_replicates = _extract_single_case_model_replicates(
            returned_studies=returned_studies,
            case_index=case_index,
        )
        model_vpc = compute_vpc_data(
            observed_study,
            model_replicates,
            n_bins=n_bins,
            binning=binning,
        )

        raw_substance_name = observed_study.get("meta_data", {}).get("substance_name")
        rendered_substance_name = _shared.display_substance_name(
            raw_substance_name,
            fallback=f"substance_{case_index}",
        )
        safe_name = _shared.safe_substance_name(
            raw_substance_name,
            fallback=f"substance_{case_index}",
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=log_y)
        vpc_plot(truth_vpc, ax=axes[0], log_y=log_y)
        vpc_plot(model_vpc, ax=axes[1], log_y=log_y)
        axes[0].set_title("Synthetic Truth")
        axes[1].set_title("Model Samples")
        fig.suptitle(f"{rendered_substance_name} | case {case_index:03d}")
        fig.tight_layout()

        image_path = output_dir / f"epoch_{epoch:03d}_case_{case_index:03d}_{safe_name}.png"
        fig.savefig(image_path, bbox_inches="tight")
        plt.close(fig)
        image_paths.append(str(image_path))

    return _shared._image_outputs(image_paths)


def _collect_diverse_synthetic_experiment_sample_distances(
    *,
    task_cfg: Mapping[str, Any],
    trainer: Any,
    pl_module: Any,
) -> _shared.DiverseExperimentDistanceCollection:
    """Collect aligned synthetic diverse-experiment bundles once for all metrics."""

    datamodule = getattr(trainer, "datamodule", None)
    if datamodule is None:
        raise ValueError(
            "task_diverse_synthetic_experiment_sample_distances requires trainer.datamodule."
        )

    synthetic_loader_kwargs = _shared._resolve_synthetic_loader_kwargs(task_cfg)
    n_targets = int(synthetic_loader_kwargs.get("n_targets", 0))
    dataset_size = int(synthetic_loader_kwargs.get("dataset_size", 0))
    n_dosings = int(synthetic_loader_kwargs.get("n_dosings", 1))
    dosing_mode = (
        str(synthetic_loader_kwargs.get("dosing_mode", "diverse_dosing")).strip()
        or "diverse_dosing"
    )
    plot_num_studies = int(task_cfg.get("plot_num_studies", 0))
    if n_targets <= 0:
        raise ValueError(
            "task_diverse_synthetic_experiment_sample_distances requires task_cfg.n_targets > 0."
        )
    if dataset_size <= 0:
        raise ValueError(
            "task_diverse_synthetic_experiment_sample_distances requires task_cfg.dataset_size > 0."
        )
    if n_dosings != 1:
        raise ValueError(
            "task_diverse_synthetic_experiment_sample_distances currently supports only "
            "n_dosings=1."
        )
    if dosing_mode != "diverse_dosing":
        raise ValueError(
            "task_diverse_synthetic_experiment_sample_distances currently supports only "
            "dosing_mode='diverse_dosing'."
        )
    if plot_num_studies < 0:
        raise ValueError(
            "task_diverse_synthetic_experiment_sample_distances requires plot_num_studies >= 0."
        )

    synthetic_loader_kwargs["n_targets"] = n_targets
    synthetic_loader_kwargs["dataset_size"] = dataset_size
    synthetic_loader_kwargs["n_dosings"] = n_dosings
    synthetic_loader_kwargs["dosing_mode"] = dosing_mode
    synthetic_loader = datamodule.get_synthetic_experiment_dataloader(**synthetic_loader_kwargs)

    model_device = getattr(pl_module, "device", torch.device("cpu"))
    num_steps = task_cfg.get("num_steps")

    aligned_bundles: list[_shared.SyntheticMMDSeriesBundle] = []
    plot_batches: list[AICMECompartmentsDataBatch] = []
    plot_aligned_bundles: list[_shared.SyntheticMMDSeriesBundle] = []
    cached_plot_studies = 0

    for batch_list in synthetic_loader:
        if not isinstance(batch_list, (list, tuple)):
            raise TypeError(
                "Synthetic experiment dataloader items must be lists or tuples of databatches."
            )
        if len(batch_list) != 1:
            raise ValueError(
                "task_diverse_synthetic_experiment_sample_distances expects synthetic "
                f"dataloader items of length 1 when n_dosings=1, got {len(batch_list)}."
            )

        batch = batch_list[0]
        batch_device = _shared._move_batch_to_device(batch, model_device)
        sample_kwargs: dict[str, Any] = {
            "sample_size": 1,
            "resolve_sampling_from_target": True,
            "include_rem": False,
        }
        if num_steps is not None:
            sample_kwargs["num_steps"] = int(num_steps)

        with torch.inference_mode():
            generated_samples, generated_times, generated_mask = pl_module.sample_new_individual(
                batch_device,
                **sample_kwargs,
            )

        aligned = _shared.validate_generated_samples_match_target_schedule(
            batch_device,
            generated_samples,
            generated_times,
            generated_mask,
        )
        aligned_cpu = _shared.SyntheticMMDSeriesBundle(
            observed_values=aligned.observed_values.detach().cpu(),
            generated_values=aligned.generated_values.detach().cpu(),
            times=aligned.times.detach().cpu(),
            mask=aligned.mask.detach().cpu(),
        )
        aligned_bundles.append(aligned_cpu)

        if plot_num_studies > 0 and cached_plot_studies < plot_num_studies:
            plot_batches.append(batch.detach_all().to_device(torch.device("cpu")))
            plot_aligned_bundles.append(aligned_cpu)
            cached_plot_studies += int(aligned_cpu.observed_values.shape[0])

    if not aligned_bundles:
        raise RuntimeError(
            "Diverse synthetic experiment sample-distances task collected no synthetic batches."
        )

    dataset_bundle = _shared._concatenate_synthetic_mmd_bundles(aligned_bundles)
    return _shared.DiverseExperimentDistanceCollection(
        dataset_bundle=dataset_bundle,
        aligned_bundles=aligned_bundles,
        plot_batches=plot_batches,
        plot_aligned_bundles=plot_aligned_bundles,
    )


def task_diverse_synthetic_experiment_sample_distances(
    *,
    samples: Any | None,
    batches: Sequence[AICMECompartmentsDataBatch],
    task_cfg: Mapping[str, Any],
    trainer: Any,
    pl_module: Any,
) -> Mapping[str, Any]:
    """Compute one or more distances on synthetic diverse-dosing experiments."""

    del samples
    del batches

    requested_metrics = _resolve_distance_metric_names(task_cfg)
    save_details = bool(task_cfg.get("save_details", True))
    plot_num_studies = int(task_cfg.get("plot_num_studies", 0))
    plot_kwargs = task_cfg.get("plot_kwargs")
    model_label = str(task_cfg.get("model_label") or _shared._normalize_model_label(pl_module))

    collection = _collect_diverse_synthetic_experiment_sample_distances(
        task_cfg=task_cfg,
        trainer=trainer,
        pl_module=pl_module,
    )

    task_root = _shared._resolve_task_root(trainer, "diverse_synthetic_experiment_sample_distances")
    run_name = f"epoch_{_shared._resolve_epoch(trainer):03d}_step_{int(getattr(trainer, 'global_step', 0)):07d}"
    persistent_run_dir = task_root / run_name
    if save_details:
        if persistent_run_dir.exists():
            shutil.rmtree(persistent_run_dir)
        persistent_run_dir.mkdir(parents=True, exist_ok=True)

    context_manager: Any
    if save_details:
        context_manager = contextlib.nullcontext(persistent_run_dir)
    else:
        context_manager = tempfile.TemporaryDirectory(
            prefix="diverse_synthetic_experiment_sample_distances_"
        )

    outputs: dict[str, Any] = {}
    with context_manager as temp_root:
        run_root = Path(temp_root)
        if "mmd2" in requested_metrics:
            outputs.update(
                _run_mmd2_distance(
                    bundle=collection.dataset_bundle,
                    mmd_cfg=_resolve_mmd_task_cfg(task_cfg),
                    output_root=run_root,
                    save_details=save_details,
                )
            )
        if "classifier_auc" in requested_metrics:
            outputs.update(
                _run_classifier_auc_distance(
                    collection=collection,
                    classifier_cfg=_resolve_classifier_auc_cfg(task_cfg),
                    output_root=run_root,
                    save_details=save_details,
                )
            )

    if plot_num_studies > 0:
        image_path = _shared.sub_task_plotting_mmd(
            plot_batches=collection.plot_batches,
            aligned_bundles=collection.plot_aligned_bundles,
            num_studies=plot_num_studies,
            trainer=trainer,
            model_label=model_label,
            route_options=getattr(getattr(pl_module, "meta_dosing", None), "route_options", None),
            plot_kwargs=dict(plot_kwargs) if isinstance(plot_kwargs, Mapping) else None,
        )
        if image_path is not None:
            outputs.update(_shared._image_outputs([str(image_path)]))
    return outputs


def task_diverse_experiment_distances(
    *,
    samples: Any | None,
    batches: Sequence[AICMECompartmentsDataBatch],
    task_cfg: Mapping[str, Any],
    trainer: Any,
    pl_module: Any,
) -> Mapping[str, Any]:
    """Backward-compatible alias for the synthetic sample-distance task."""

    return task_diverse_synthetic_experiment_sample_distances(
        samples=samples,
        batches=batches,
        task_cfg=task_cfg,
        trainer=trainer,
        pl_module=pl_module,
    )


__all__ = [
    "task_diverse_synthetic_experiment_sample_distances",
    "task_diverse_experiment_distances",
    "task_generative_images",
    "task_generative_metrics",
    "task_predictive_images",
    "task_synthetic_vpc_paired_images",
    "task_vpc_images",
    "task_vpc_npde_pvalues",
]
