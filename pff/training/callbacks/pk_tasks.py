"""PK-specific scheduler payloads, shared helpers, and compatibility aliases.

These helpers keep PK evaluation logic inside the scheduler stack. The generic
callback in :mod:`pff.training.callbacks.scheduler` remains
responsible for trigger timing, grouping, caching, and logger dispatch; this
module only defines:

- typed sample bundles shared across PK tasks,
- batch-level PK metric/image helpers,
- compatibility exports forwarding to canonical task modules.
"""

from __future__ import annotations

import contextlib
import json
import math
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchtyping import TensorType
from tqdm.auto import tqdm

from pff import reports_dir
from pff.config_classes.data_config import ObservationsConfig
from pff.data.data_empirical.builder import (
    databatch_to_study_jsons,
    prediction_to_study_jsons,
)
from pff.data.data_empirical.json_schema import StudyJSON, studies_from_sampled_targets
from pff.data.datasets.aicme_datasets import AICMECompartmentsDataBatch
from pff.metrics.quantiles_coverage import compute_percentile_coverage
from pff.metrics.sample_distance_metrics import (
    _binary_roc_auc_from_scores,
    _build_classifier_feature_matrix,
    _build_classifier_mlp,
    _resolve_classifier_auc_cfg,
    _resolve_distance_metric_names,
    _resolve_mmd_task_cfg,
    _run_classifier_auc_distance,
    _run_mmd2_distance,
    _run_signature_mmd_runner,
    _train_classifier_auc_model,
    _write_synthetic_mmd_payload,
)
from pff.metrics.sampling_quality import (
    compute_npde_data,
    compute_vpc_data,
    npde_pvalues,
    vpc_plot,
)
from pff.models.utils.callbacks_naming import (
    display_substance_name,
    safe_substance_name,
)
from pff.utils.plots.databatch_plot import (
    plot_list_list_study_json,
    plot_synthetic_mmd_overlay,
)


@dataclass(frozen=True)
class PredictiveBundle:
    """Sampled predictive outputs for one batch permutation."""

    samples_S: TensorType["S", "B", "It", "Tr", 1]
    times_S: TensorType["S", "B", "It", "Tr", 1]
    target_raw: TensorType["B", "It", "Tr", 1]
    target_mask: TensorType["B", "It", "Tr"]


@dataclass(frozen=True)
class GenerativeBundle:
    """Sampled new-individual outputs for one batch permutation."""

    samples: TensorType["S", "B", "Tdistinct", 1]
    times: TensorType["B", "Tdistinct", 1]
    mask: TensorType["B", "Tdistinct"]
    pred_values: TensorType["B", "S", "Tdistinct", 1]


@dataclass(frozen=True)
class VPCBundle:
    """Observed/simulated StudyJSONs used for empirical VPC rendering."""

    observed_studies: list[StudyJSON]
    simulated_studies_by_substance: list[list[StudyJSON]]


@dataclass(frozen=True)
class SyntheticMMDSeriesBundle:
    """Observed/generated target series validated on one shared synthetic grid."""

    observed_values: TensorType["B", "It", "Tobs", 1]
    generated_values: TensorType["B", "It", "Tobs", 1]
    times: TensorType["B", "It", "Tobs", 1]
    mask: TensorType["B", "It", "Tobs"]


@dataclass(frozen=True)
class DiverseExperimentDistanceCollection:
    """Aligned synthetic diverse-experiment payload shared across distance metrics."""

    dataset_bundle: SyntheticMMDSeriesBundle
    aligned_bundles: list[SyntheticMMDSeriesBundle]
    plot_batches: list[AICMECompartmentsDataBatch]
    plot_aligned_bundles: list[SyntheticMMDSeriesBundle]


def sample_predictive_bundle(
    model: Any,
    batch: AICMECompartmentsDataBatch,
    *,
    sample_size: int = 1,
) -> PredictiveBundle:
    """Sample predictive trajectories for one batch."""

    with torch.inference_mode():
        (
            samples_S,
            times_S,
            target_raw,
            target_mask,
        ) = model.sample_individual_prediction(batch, sample_size=sample_size)

    # samples_S: [S, B, It, Tr, 1]
    # times_S: [S, B, It, Tr, 1]
    # target_raw: [B, It, Tr, 1]
    # target_mask: [B, It, Tr]
    return PredictiveBundle(
        samples_S=samples_S,
        times_S=times_S,
        target_raw=target_raw,
        target_mask=target_mask,
    )


def sample_generative_bundle(
    model: Any,
    batch: AICMECompartmentsDataBatch,
    *,
    sample_size: int = 10,
) -> GenerativeBundle:
    """Sample new-individual trajectories for one batch."""

    with torch.inference_mode():
        samples, times, mask = model.sample_new_individual(batch, sample_size=sample_size)

    # samples: [S, B, Tdistinct, 1]
    # times: [B, Tdistinct, 1]
    # mask: [B, Tdistinct]
    pred_values = samples.transpose(0, 1)  # [B, S, Tdistinct, 1]
    return GenerativeBundle(
        samples=samples,
        times=times,
        mask=mask,
        pred_values=pred_values,
    )


def sample_vpc_bundle(
    model: Any,
    batch: AICMECompartmentsDataBatch,
    *,
    sample_size: int = 4,
) -> VPCBundle:
    """Sample empirical VPC inputs for one batch."""

    meta_dosing = getattr(model, "meta_dosing", None)
    if meta_dosing is None:
        raise AttributeError("`meta_dosing` must be available to render VPC images.")

    with torch.inference_mode():
        observed_studies = databatch_to_study_jsons(batch, meta_dosing)
        simulated_studies_by_substance = model.sample_new_individuals_to_vpc_format(
            batch,
            sample_size=sample_size,
        )

    return VPCBundle(
        observed_studies=observed_studies,
        simulated_studies_by_substance=simulated_studies_by_substance,
    )


def _slice_tensor_prefix(tensor: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Slice the leading sample axis while tolerating short tensors."""

    if tensor.ndim == 0:
        return tensor
    take = min(int(n_samples), int(tensor.shape[0]))
    return tensor[:take]


@dataclass(frozen=True)
class PKTaskSamples:
    """Composite sample payload shared across PK scheduler tasks."""

    predictive: PredictiveBundle
    generative: GenerativeBundle
    vpc: VPCBundle

    def scheduler_slice(self, n_samples: int) -> "PKTaskSamples":
        """Return a task-local view trimmed to ``n_samples`` when applicable."""

        predictive = PredictiveBundle(
            samples_S=_slice_tensor_prefix(self.predictive.samples_S, n_samples),
            times_S=_slice_tensor_prefix(self.predictive.times_S, n_samples),
            target_raw=self.predictive.target_raw,
            target_mask=self.predictive.target_mask,
        )
        generative_samples = _slice_tensor_prefix(self.generative.samples, n_samples)
        generative = GenerativeBundle(
            samples=generative_samples,
            times=self.generative.times,
            mask=self.generative.mask,
            pred_values=generative_samples.transpose(0, 1),
        )
        vpc = VPCBundle(
            observed_studies=self.vpc.observed_studies,
            simulated_studies_by_substance=[
                list(studies[: min(int(n_samples), len(studies))])
                for studies in self.vpc.simulated_studies_by_substance
            ],
        )
        return PKTaskSamples(predictive=predictive, generative=generative, vpc=vpc)


@dataclass(frozen=True)
class GenerativeTaskSamples:
    """Composite sample payload for models that only expose generative sampling."""

    generative: GenerativeBundle
    vpc: VPCBundle

    def scheduler_slice(self, n_samples: int) -> "GenerativeTaskSamples":
        """Return a task-local generative/VPC view trimmed to ``n_samples``."""

        generative_samples = _slice_tensor_prefix(self.generative.samples, n_samples)
        generative = GenerativeBundle(
            samples=generative_samples,
            times=self.generative.times,
            mask=self.generative.mask,
            pred_values=generative_samples.transpose(0, 1),
        )
        return GenerativeTaskSamples(
            generative=generative,
            vpc=self.vpc,
        )


@dataclass(frozen=True)
class PredictiveTaskSamples:
    """Predictive-only scheduler payload used by ``PredictionPK``."""

    predictive: PredictiveBundle

    def scheduler_slice(self, n_samples: int) -> "PredictiveTaskSamples":
        """Return a task-local predictive view trimmed to ``n_samples``."""

        return PredictiveTaskSamples(
            predictive=PredictiveBundle(
                samples_S=_slice_tensor_prefix(self.predictive.samples_S, n_samples),
                times_S=_slice_tensor_prefix(self.predictive.times_S, n_samples),
                target_raw=self.predictive.target_raw,
                target_mask=self.predictive.target_mask,
            )
        )


def build_pk_task_samples(
    model: Any,
    batch: AICMECompartmentsDataBatch,
    *,
    num_samples: int,
) -> PKTaskSamples | GenerativeTaskSamples:
    """Build the composite payload consumed by PK scheduler tasks.

    Predictive sampling is included only when the model exposes a callable
    ``sample_individual_prediction`` method. This keeps generative-only models
    usable with generative image, VPC and distance tasks while predictive tasks
    still fail explicitly when they request a missing predictive payload.
    """

    generative = sample_generative_bundle(model, batch, sample_size=num_samples)
    vpc = sample_vpc_bundle(model, batch, sample_size=num_samples)
    if not callable(getattr(model, "sample_individual_prediction", None)):
        return GenerativeTaskSamples(
            generative=generative,
            vpc=vpc,
        )

    predictive = sample_predictive_bundle(model, batch, sample_size=num_samples)
    return PKTaskSamples(
        predictive=predictive,
        generative=generative,
        vpc=vpc,
    )


def build_predictive_task_samples(
    model: Any,
    batch: AICMECompartmentsDataBatch,
    *,
    num_samples: int,
) -> PredictiveTaskSamples:
    """Build the predictive-only payload consumed by ``PredictionPK`` tasks.

    This helper performs predictive sampling for exactly one atomic databatch.
    It does not concatenate across permutations or across empirical repos.

    Important shape convention
    --------------------------
    ``batch`` already carries one fixed batch axis ``B`` indexing the studies /
    drugs inside that databatch. The returned predictive payload therefore keeps
    the layout

    - ``samples_S``: ``[S, B, It, Tr, 1]``
    - ``times_S``: ``[S, B, It, Tr, 1]``

    where:
    - ``S`` is the stochastic sample axis requested by ``num_samples``,
    - ``B`` is the original databatch axis and is preserved exactly,
    - ``It`` is the target-individual capacity inside each batch slot.

    No averaging happens here. This function is only responsible for "sample one
    databatch and preserve its native slot order".
    """

    predictive = sample_predictive_bundle(model, batch, sample_size=num_samples)
    return PredictiveTaskSamples(predictive=predictive)


def _as_batch_list(batches: Any) -> list[AICMECompartmentsDataBatch]:
    if isinstance(batches, AICMECompartmentsDataBatch):
        return [batches]
    return list(batches or [])


def _as_sample_list(samples: Any) -> list[Any]:
    if samples is None:
        return []
    if isinstance(samples, list):
        return list(samples)
    return [samples]


def _predictive_payload(samples: Any) -> PredictiveBundle:
    """Resolve a predictive bundle from either composite or predictive-only payloads."""

    predictive = getattr(samples, "predictive", None)
    if predictive is None:
        raise TypeError("Scheduler payload does not expose a predictive bundle.")
    return predictive


def _move_batch_to_device(batch: Any, device: Any) -> Any:
    """Move one databatch to ``device`` when the batch object supports it."""

    to_device = getattr(batch, "to_device", None)
    if callable(to_device):
        return to_device(device)

    to = getattr(batch, "to", None)
    if callable(to):
        return to(device)
    return batch


def _resolve_task_root(trainer: Any, task_name: str) -> Path:
    """Return a persistent artifact directory for standalone callback tasks."""

    root_dir = getattr(trainer, "default_root_dir", None)
    base_root = Path(str(root_dir)) if root_dir is not None else Path(".")
    task_root = base_root / "scheduler_tasks" / task_name
    task_root.mkdir(parents=True, exist_ok=True)
    return task_root


def _coerce_synthetic_target_observation_config(
    raw_config: Any,
) -> ObservationsConfig | None:
    """Coerce one optional synthetic-target observation config into its dataclass."""

    if raw_config is None:
        return None
    if isinstance(raw_config, ObservationsConfig):
        return raw_config
    if not isinstance(raw_config, Mapping):
        raise TypeError(
            "task_cfg.synthetic_loader.synthetic_target_observation_config must be a mapping "
            "or ObservationsConfig when provided."
        )
    return ObservationsConfig(**dict(raw_config))


def _resolve_synthetic_loader_kwargs(task_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the preferred nested synthetic-loader config with flat-key fallback."""

    raw_nested_loader = task_cfg.get("synthetic_loader")
    if raw_nested_loader is None:
        loader_kwargs: dict[str, Any] = {
            "n_targets": task_cfg.get("n_targets", 0),
            "n_dosings": task_cfg.get("n_dosings", 1),
            "dosing_mode": task_cfg.get("dosing_mode", "diverse_dosing"),
            "dataset_size": task_cfg.get("dataset_size", 0),
        }
        if "dosing_list_generation" in task_cfg:
            loader_kwargs["dosing_list_generation"] = task_cfg.get("dosing_list_generation")
        if "logdose_range" in task_cfg:
            loader_kwargs["logdose_range"] = task_cfg.get("logdose_range")
        if "synthetic_target_observation_config" in task_cfg:
            loader_kwargs["synthetic_target_observation_config"] = task_cfg.get(
                "synthetic_target_observation_config"
            )
        if "shuffle" in task_cfg:
            loader_kwargs["shuffle"] = task_cfg.get("shuffle")
    else:
        if not isinstance(raw_nested_loader, Mapping):
            raise TypeError("task_cfg.synthetic_loader must be a mapping when provided.")
        loader_kwargs = dict(raw_nested_loader)

    if "logdose_range" in loader_kwargs and loader_kwargs["logdose_range"] is not None:
        loader_kwargs["logdose_range"] = tuple(loader_kwargs["logdose_range"])
    if "shuffle" in loader_kwargs:
        loader_kwargs["shuffle"] = bool(loader_kwargs["shuffle"])
    if "synthetic_target_observation_config" in loader_kwargs:
        loader_kwargs["synthetic_target_observation_config"] = (
            _coerce_synthetic_target_observation_config(
                loader_kwargs["synthetic_target_observation_config"]
            )
        )

    return loader_kwargs


def validate_generated_samples_match_target_schedule(
    batch: AICMECompartmentsDataBatch,
    generated_samples: TensorType["It", "B", "Tdistinct", 1],
    generated_times: TensorType["B", "Tdistinct", 1],
    generated_mask: TensorType["B", "Tdistinct"],
    *,
    atol: float = 1.0e-5,
) -> SyntheticMMDSeriesBundle:
    """Validate that synthetic targets and generated samples already share one grid.

    Parameters
    ----------
    batch:
        Synthetic databatch carrying observed target values and target schedules.
    generated_samples:
        Target-resolved sampled values with shape ``[It, B, Tdistinct, 1]``.
    generated_times:
        Shared decoded time grid per batch element with shape ``[B, Tdistinct, 1]``.
    generated_mask:
        Valid-time mask for ``generated_times`` with shape ``[B, Tdistinct]``.

    Returns
    -------
    SyntheticMMDSeriesBundle
        Observed and generated values packaged on the shared target-observation
        grid, with shapes:
        - observed_values: ``[B, It, Tobs, 1]``
        - generated_values: ``[B, It, Tobs, 1]``
        - times: ``[B, It, Tobs, 1]``
        - mask: ``[B, It, Tobs]``
    """

    target_values = batch.target_obs.detach()
    target_times = batch.target_obs_time.detach()
    target_mask = batch.target_obs_mask.bool().detach()
    target_individual_mask = batch.mask_target_individuals.bool().detach().unsqueeze(-1)
    valid_target_mask = target_mask & target_individual_mask

    B, It, Tobs, _ = target_values.shape
    Tdistinct = generated_times.shape[1]
    if generated_samples.shape[:2] != (It, B):
        raise ValueError(
            "Expected generated_samples to have leading shape "
            f"(It={It}, B={B}), got {tuple(generated_samples.shape[:2])}."
        )
    if generated_samples.ndim != 4 or generated_samples.shape[-1] != 1:
        raise ValueError(
            "Expected generated_samples to have shape [It, B, Tdistinct, 1], got "
            f"{tuple(generated_samples.shape)}."
        )
    if generated_times.shape[0] != B:
        raise ValueError(
            f"Expected generated_times to have B={B} batch elements, got {generated_times.shape[0]}."
        )
    if generated_times.ndim != 3 or generated_times.shape[-1] != 1:
        raise ValueError(
            "Expected generated_times to have shape [B, Tdistinct, 1], got "
            f"{tuple(generated_times.shape)}."
        )
    if generated_mask.shape[0] != B:
        raise ValueError(
            f"Expected generated_mask to have B={B} batch elements, got {generated_mask.shape[0]}."
        )
    if generated_mask.ndim != 2:
        raise ValueError(
            "Expected generated_mask to have shape [B, Tdistinct], got "
            f"{tuple(generated_mask.shape)}."
        )
    if generated_samples.shape[2] != Tdistinct or generated_mask.shape[1] != Tdistinct:
        raise ValueError(
            "Generated synthetic MMD tensors must share one time dimension, got "
            f"samples T={generated_samples.shape[2]}, times T={Tdistinct}, "
            f"mask T={generated_mask.shape[1]}."
        )
    valid_generated_mask = generated_mask.bool().detach()  # [B, Tdistinct]
    shared_times = generated_times.detach()  # [B, Tdistinct, 1]
    generated_value_grid = generated_samples.detach().permute(1, 0, 2, 3).contiguous()
    aligned_generated = torch.zeros_like(target_values)
    for batch_idx in range(B):
        valid_generated_mask_b = valid_generated_mask[batch_idx]  # [Tdistinct]
        valid_generated_count = int(valid_generated_mask_b.sum().item())
        shared_times_b = shared_times[batch_idx, valid_generated_mask_b, 0]  # [Tvalid]
        for target_idx in range(It):
            valid_mask_bt = valid_target_mask[batch_idx, target_idx]  # [Tobs]
            if not valid_mask_bt.any():
                continue
            target_valid_count = int(valid_mask_bt.sum().item())
            if target_valid_count != valid_generated_count:
                raise ValueError(
                    "Synthetic target observations must already use the generated shared grid "
                    f"length for batch index {batch_idx}, target index {target_idx}. "
                    f"Target valid count={target_valid_count}, generated valid count="
                    f"{valid_generated_count}."
                )
            target_times_bt = target_times[batch_idx, target_idx, valid_mask_bt, 0]
            if not torch.allclose(target_times_bt, shared_times_b, atol=atol, rtol=0.0):
                time_abs_diff = (target_times_bt - shared_times_b).abs()
                max_abs_diff = float(time_abs_diff.max().item())
                preview_count = min(5, int(target_times_bt.numel()))
                target_preview = [float(x) for x in target_times_bt[:preview_count].tolist()]
                generated_preview = [float(x) for x in shared_times_b[:preview_count].tolist()]
                raise ValueError(
                    "Synthetic target observation times must already match the generated "
                    f"shared grid for batch index {batch_idx}, target index {target_idx}. "
                    f"Max abs diff={max_abs_diff:.6g} with atol={atol:.6g}. "
                    f"Target preview={target_preview}; generated preview={generated_preview}."
                )
            aligned_generated[batch_idx, target_idx, valid_mask_bt] = generated_value_grid[
                batch_idx,
                target_idx,
                valid_generated_mask_b,
            ]

    observed_values = target_values * valid_target_mask.unsqueeze(-1).to(target_values.dtype)
    aligned_generated = aligned_generated * valid_target_mask.unsqueeze(-1).to(
        aligned_generated.dtype
    )
    aligned_times = target_times * valid_target_mask.unsqueeze(-1).to(target_times.dtype)
    return SyntheticMMDSeriesBundle(
        observed_values=observed_values,
        generated_values=aligned_generated,
        times=aligned_times,
        mask=valid_target_mask,
    )


def _reconstruct_full_target_databatch(
    batch: AICMECompartmentsDataBatch,
) -> AICMECompartmentsDataBatch:
    """Rebuild full held-out target series by concatenating past and future blocks.

    The returned batch keeps the original context and metadata untouched, while
    replacing the target block with one contiguous full schedule:

    - ``target_obs``      -> full target values ``[B, It, Tfull, 1]``
    - ``target_obs_time`` -> full target times ``[B, It, Tfull, 1]``
    - ``target_obs_mask`` -> full target mask ``[B, It, Tfull]``
    - ``target_rem_*``    -> empty remainder tensors with ``T=0``
    """

    B, It, To, _ = batch.target_obs.shape
    Tr = int(batch.target_rem_sim.shape[2])
    Tfull = int(To + Tr)

    full_target = torch.zeros(
        B,
        It,
        Tfull,
        1,
        dtype=batch.target_obs.dtype,
        device=batch.target_obs.device,
    )  # [B, It, Tfull, 1]
    full_times = torch.zeros(
        B,
        It,
        Tfull,
        1,
        dtype=batch.target_obs_time.dtype,
        device=batch.target_obs_time.device,
    )  # [B, It, Tfull, 1]
    full_mask = torch.zeros(
        B,
        It,
        Tfull,
        dtype=torch.bool,
        device=batch.target_obs_mask.device,
    )  # [B, It, Tfull]

    for batch_idx in range(B):
        for target_idx in range(It):
            obs_mask_bt = batch.target_obs_mask[batch_idx, target_idx].bool()  # [To]
            rem_mask_bt = batch.target_rem_sim_mask[batch_idx, target_idx].bool()  # [Tr]
            obs_count = int(obs_mask_bt.sum().item())
            rem_count = int(rem_mask_bt.sum().item())
            total_count = int(obs_count + rem_count)
            if total_count == 0:
                continue

            if obs_count > 0:
                full_target[batch_idx, target_idx, :obs_count] = batch.target_obs[
                    batch_idx,
                    target_idx,
                    obs_mask_bt,
                ]
                full_times[batch_idx, target_idx, :obs_count] = batch.target_obs_time[
                    batch_idx,
                    target_idx,
                    obs_mask_bt,
                ]
            if rem_count > 0:
                full_target[batch_idx, target_idx, obs_count:total_count] = batch.target_rem_sim[
                    batch_idx,
                    target_idx,
                    rem_mask_bt,
                ]
                full_times[batch_idx, target_idx, obs_count:total_count] = (
                    batch.target_rem_sim_time[batch_idx, target_idx, rem_mask_bt]
                )
            full_mask[batch_idx, target_idx, :total_count] = True

    empty_rem = torch.zeros(
        B,
        It,
        0,
        1,
        dtype=batch.target_rem_sim.dtype,
        device=batch.target_rem_sim.device,
    )  # [B, It, 0, 1]
    empty_rem_time = torch.zeros(
        B,
        It,
        0,
        1,
        dtype=batch.target_rem_sim_time.dtype,
        device=batch.target_rem_sim_time.device,
    )  # [B, It, 0, 1]
    empty_rem_mask = torch.zeros(
        B,
        It,
        0,
        dtype=torch.bool,
        device=batch.target_rem_sim_mask.device,
    )  # [B, It, 0]

    return batch._replace(
        target_obs=full_target,
        target_obs_time=full_times,
        target_obs_mask=full_mask,
        target_rem_sim=empty_rem,
        target_rem_sim_time=empty_rem_time,
        target_rem_sim_mask=empty_rem_mask,
    )


def _slice_empirical_batch_rows(
    batch: AICMECompartmentsDataBatch,
    keep_indices: Sequence[int],
) -> AICMECompartmentsDataBatch:
    """Return an empirical batch restricted to the selected study/drug slots.

    Important note for empirical held-out evaluation
    -----------------------------------------------
    The leading ``B`` axis of an empirical batch indexes the independent
    study/drug slots stacked together for that permutation. Restricting the
    batch therefore means "keep only the slots that can actually be compared",
    not "modify the target-individual axis inside one study".
    """

    batch_size = int(batch.target_obs.shape[0])
    index_tensor = torch.as_tensor(
        list(keep_indices),
        dtype=torch.long,
        device=batch.target_obs.device,
    )
    sliced_fields: dict[str, Any] = {}
    for field_name in batch._fields:
        value = getattr(batch, field_name)
        if isinstance(value, torch.Tensor):
            if value.ndim > 0 and int(value.shape[0]) == batch_size:
                sliced_fields[field_name] = value.index_select(0, index_tensor)
            else:
                sliced_fields[field_name] = value
        elif isinstance(value, list) and len(value) == batch_size:
            sliced_fields[field_name] = [value[idx] for idx in index_tensor.tolist()]
        else:
            sliced_fields[field_name] = value
    return batch._replace(**sliced_fields)


def _select_empirical_classifier_comparable_batches(
    batch_list: Sequence[AICMECompartmentsDataBatch],
) -> tuple[list[AICMECompartmentsDataBatch], list[str]]:
    """Keep only empirical slots that have a real held-out target to compare.

    Plain-language contract
    -----------------------
    The held-out classifier should count only comparisons that are actually
    defined:

    - if a permutation contains a study/drug slot with one valid held-out
      target observation schedule, we keep that slot and generate against it;
    - if a slot has no valid held-out target for that permutation, that slot is
      simply not counted for the classifier;
    - if a whole permutation contains no valid held-out targets at all, that
      permutation is skipped with a warning.

    This keeps the metric aligned with the intuitive rule:
    "one generated sample series for each valid held-out target series".

    Why this is needed
    ------------------
    Empirical loaders may include structural padding permutations so that all
    studies expose the same outer permutation count. Those padded permutations
    can legitimately contain empty target blocks. The classifier task should
    ignore them rather than forcing the model sampler to invent a target
    schedule that does not exist.
    """

    comparable_batches: list[AICMECompartmentsDataBatch] = []
    output_substances: list[str] = []
    seen_substances: set[str] = set()

    for permutation_idx, batch in enumerate(_as_batch_list(batch_list)):
        valid_slot_mask = (
            batch.target_obs_mask.bool() & batch.mask_target_individuals.bool().unsqueeze(-1)
        ).any(dim=(1, 2))  # [B]
        keep_indices = torch.nonzero(valid_slot_mask, as_tuple=False).view(-1).tolist()
        if not keep_indices:
            warnings.warn(
                "Skipping empirical held-out classifier permutation "
                f"{permutation_idx} because it contains no valid held-out target schedule.",
                stacklevel=3,
            )
            continue

        comparable_batch = _slice_empirical_batch_rows(batch, keep_indices)
        comparable_batches.append(comparable_batch)
        for _, substance in _resolve_empirical_slot_labels(comparable_batch):
            if substance not in seen_substances:
                seen_substances.add(substance)
                output_substances.append(substance)

    return comparable_batches, output_substances


def _empty_synthetic_mmd_series_bundle() -> SyntheticMMDSeriesBundle:
    """Return an empty aligned-series bundle for graceful task-level skipping."""

    return SyntheticMMDSeriesBundle(
        observed_values=torch.zeros(0, 0, 0, 1, dtype=torch.float32),
        generated_values=torch.zeros(0, 0, 0, 1, dtype=torch.float32),
        times=torch.zeros(0, 0, 0, 1, dtype=torch.float32),
        mask=torch.zeros(0, 0, 0, dtype=torch.bool),
    )


def _collect_empirical_heldout_generated_classifier_collection(
    batch_list: Sequence[AICMECompartmentsDataBatch],
    *,
    model: Any,
    num_steps: int | None = None,
) -> tuple[DiverseExperimentDistanceCollection, list[str]]:
    """Collect pooled empirical comparisons for the held-out-vs-generated classifier.

    Each counted example corresponds to one valid held-out target series and
    one generated series sampled on that same target schedule. Permutations or
    batch slots without a real held-out target are skipped before sampling.
    """

    if not callable(getattr(model, "sample_new_individual", None)):
        raise ValueError(
            "Empirical held-out generated classifier requires a model implementing "
            "`sample_new_individual`."
        )

    batches, output_substances = _select_empirical_classifier_comparable_batches(batch_list)
    if not batches:
        return (
            DiverseExperimentDistanceCollection(
                dataset_bundle=_empty_synthetic_mmd_series_bundle(),
                aligned_bundles=[],
                plot_batches=[],
                plot_aligned_bundles=[],
            ),
            output_substances,
        )

    model_device = getattr(model, "device", torch.device("cpu"))
    aligned_bundles: list[SyntheticMMDSeriesBundle] = []

    for batch in batches:
        batch_device = _move_batch_to_device(batch, model_device)
        sample_kwargs: dict[str, Any] = {
            "sample_size": 1,
            "resolve_sampling_from_target": True,
            "include_rem": True,
        }
        if num_steps is not None:
            sample_kwargs["num_steps"] = int(num_steps)

        with torch.inference_mode():
            generated_samples, generated_times, generated_mask = model.sample_new_individual(
                batch_device,
                **sample_kwargs,
            )

        full_target_batch = _reconstruct_full_target_databatch(batch_device)
        aligned = validate_generated_samples_match_target_schedule(
            full_target_batch,
            generated_samples,
            generated_times,
            generated_mask,
        )
        aligned_bundles.append(
            SyntheticMMDSeriesBundle(
                observed_values=aligned.observed_values.detach().cpu(),
                generated_values=aligned.generated_values.detach().cpu(),
                times=aligned.times.detach().cpu(),
                mask=aligned.mask.detach().cpu(),
            )
        )

    dataset_bundle = _concatenate_synthetic_mmd_bundles(aligned_bundles)
    collection = DiverseExperimentDistanceCollection(
        dataset_bundle=dataset_bundle,
        aligned_bundles=aligned_bundles,
        plot_batches=[],
        plot_aligned_bundles=[],
    )
    return collection, output_substances


def _compute_empirical_heldout_generated_classifier_metrics_from_batch_list(
    batch_list: Sequence[AICMECompartmentsDataBatch],
    *,
    model: Any,
    repo_id: str,
    classifier_cfg: Mapping[str, Any],
    num_steps: int | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Train one joint classifier on pooled held-out vs generated empirical series."""

    if str(classifier_cfg.get("mode", "joint")) != "joint":
        raise ValueError(
            "Empirical held-out generated classifier only supports classifier_auc.mode='joint'."
        )

    collection, output_substances = _collect_empirical_heldout_generated_classifier_collection(
        batch_list,
        model=model,
        num_steps=num_steps,
    )

    include_time_channel = bool(classifier_cfg["include_time_channel"])
    features, labels = _build_classifier_feature_matrix(
        collection.dataset_bundle,
        include_time_channel=include_time_channel,
    )
    num_valid_series = int(collection.dataset_bundle.mask.any(dim=-1).sum().item())
    detail_report: dict[str, Any] = {
        "repo_id": str(repo_id),
        "mode": "joint",
        "include_time_channel": include_time_channel,
        "hidden_dim": int(classifier_cfg["hidden_dim"]),
        "num_hidden_layers": int(classifier_cfg["num_hidden_layers"]),
        "learning_rate": float(classifier_cfg["learning_rate"]),
        "weight_decay": float(classifier_cfg["weight_decay"]),
        "epochs": int(classifier_cfg["epochs"]),
        "batch_size": int(classifier_cfg["batch_size"]),
        "seed": int(classifier_cfg["seed"]),
        "show_progress": bool(classifier_cfg["show_progress"]),
        "num_bundles": int(len(collection.aligned_bundles)),
        "num_valid_target_series": num_valid_series,
        "output_substances": list(output_substances),
    }
    if num_valid_series < 2:
        warnings.warn(
            "Skipping empirical held-out generated classifier because fewer than two valid "
            f"held-out target series remain after filtering; got {num_valid_series}.",
            stacklevel=2,
        )
        detail_report["skipped_reason"] = "insufficient_valid_target_series"
        return {}, detail_report

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
    train_result = _train_classifier_auc_model(
        features,
        labels,
        progress_desc="Training empirical heldout_generated classifier_auc",
        **train_kwargs,
    )
    classifier_auc = float(train_result["classifier_auc"])
    metrics_by_substance = {
        substance: {
            "classifier_auc": classifier_auc,
            "repo_id": str(repo_id),
        }
        for substance in output_substances
    }
    detail_report["classifier_auc"] = classifier_auc
    detail_report["train_result"] = train_result
    return metrics_by_substance, detail_report


def _resolve_task_label(task_cfg: Mapping[str, Any], *, default: str = "Synthetic") -> str:
    raw = task_cfg.get("label", default)
    text = str(raw).strip()
    return text or default


def _resolve_repo_id(task_cfg: Mapping[str, Any], *, default: str | None = None) -> str | None:
    raw = task_cfg.get("repo_id", default)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or default


def _resolve_epoch(trainer: Any) -> int:
    return int(getattr(trainer, "current_epoch", 0))


def _resolve_plot_root(trainer: Any) -> Path:
    root_dir = getattr(trainer, "default_root_dir", None)
    root = Path(str(root_dir)) if root_dir is not None else Path(".")
    output_root = root / "training_images"
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def _resolve_metrics_report_root(trainer: Any) -> Path:
    """Return the persistent metrics-report directory used by standalone tasks."""

    del trainer
    output_root = Path(reports_dir) / "metrics"
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def _normalize_model_label(pl_module: Any) -> str:
    model_cfg = getattr(pl_module, "model_config", None)
    return str(getattr(model_cfg, "name_str", None) or pl_module.__class__.__name__)


def _prediction_metrics_per_batch(
    bundle: PredictiveBundle,
    batch: AICMECompartmentsDataBatch,
    *,
    model: Any,
    **_: Any,
) -> dict[str, dict[str, float]]:
    """Compute per-substance predictive metrics for a single batch permutation."""

    with torch.inference_mode():
        samples_S = bundle.samples_S  # [S, B, It, Tr, 1]
        target_raw = bundle.target_raw  # [B, It, Tr, 1]
        target_mask = bundle.target_mask  # [B, It, Tr]

        pred_mean = samples_S.mean(dim=0)  # [B, It, Tr, 1]
        batch_size = int(pred_mean.shape[0])
        indiv_mask = batch.mask_target_individuals  # [B, It]

        raw_names = list(batch.substance_name)
        substance_names: list[str] = []
        for batch_idx, name in enumerate(raw_names):
            if name is None or str(name).strip() == "":
                substance_names.append(f"substance_{batch_idx}")
            else:
                substance_names.append(str(name))

        metrics_by_substance: dict[str, dict[str, float]] = {}
        for batch_idx in range(batch_size):
            substance = substance_names[batch_idx]

            if not indiv_mask[batch_idx, 0]:
                metrics_b = {"rmse": 0.0, "log_rmse": 0.0, "r2": 0.0, "log_r2": 0.0}
            else:
                pred_mean_b = pred_mean[batch_idx, 0].unsqueeze(0).unsqueeze(0)  # [1, 1, Tr, 1]
                target_b = target_raw[batch_idx, 0].unsqueeze(0).unsqueeze(0)  # [1, 1, Tr, 1]
                target_mask_b = target_mask[batch_idx, 0].unsqueeze(0).unsqueeze(0)  # [1, 1, Tr]
                metrics_b = {
                    "rmse": model.masked_rmse_loss(pred_mean_b, target_b, target_mask_b)[
                        "rmse"
                    ].item(),
                    "log_rmse": model.masked_log_rmse_loss(pred_mean_b, target_b, target_mask_b)[
                        "rmse"
                    ].item(),
                    "r2": model.masked_r2_score(pred_mean_b, target_b, target_mask_b).item(),
                    "log_r2": model.masked_log_r2_score(
                        pred_mean_b, target_b, target_mask_b
                    ).item(),
                }

            metrics_by_substance[substance] = metrics_b

        return metrics_by_substance


def _compute_predictive_target_metrics(
    *,
    pred_mean: TensorType[1, 1, "Tr", 1],
    target_raw: TensorType[1, 1, "Tr", 1],
    target_mask: TensorType[1, 1, "Tr"],
    model: Any,
) -> dict[str, float]:
    """Compute predictive metrics for one target individual."""

    return {
        "rmse": model.masked_rmse_loss(pred_mean, target_raw, target_mask)["rmse"].item(),
        "log_rmse": model.masked_log_rmse_loss(pred_mean, target_raw, target_mask)["rmse"].item(),
        "r2": model.masked_r2_score(pred_mean, target_raw, target_mask).item(),
        "log_r2": model.masked_log_r2_score(pred_mean, target_raw, target_mask).item(),
    }


def _resolve_substance_names(batch: AICMECompartmentsDataBatch) -> list[str]:
    """Resolve one stable output substance name for each batch slot."""

    batch_size = int(batch.target_obs.shape[0])
    raw_names = list(batch.substance_name)
    substance_names: list[str] = []
    for batch_idx in range(batch_size):
        raw_name = raw_names[batch_idx] if batch_idx < len(raw_names) else None
        if raw_name is None or str(raw_name).strip() == "":
            substance_names.append(f"substance_{batch_idx}")
        else:
            substance_names.append(str(raw_name))
    return substance_names


def _normalize_empirical_substance_group_key(name: object) -> str:
    """Normalize empirical substance names before cross-repo pooling."""

    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _resolve_study_names(batch: AICMECompartmentsDataBatch) -> list[str]:
    """Resolve one stable study label for each batch slot."""

    batch_size = int(batch.target_obs.shape[0])
    raw_names = list(batch.study_name)
    study_names: list[str] = []
    for batch_idx in range(batch_size):
        raw_name = raw_names[batch_idx] if batch_idx < len(raw_names) else None
        if raw_name is None or str(raw_name).strip() == "":
            study_names.append(f"study_{batch_idx}")
        else:
            study_names.append(str(raw_name))
    return study_names


def _resolve_empirical_slot_labels(
    batch: AICMECompartmentsDataBatch,
) -> list[tuple[str, str]]:
    """Return ``(study_name, substance_name)`` pairs for empirical batch slots."""

    study_names = _resolve_study_names(batch)
    substance_names = _resolve_substance_names(batch)
    labels = list(zip(study_names, substance_names))

    unique_substances = {substance for _, substance in labels}
    if len(unique_substances) != len(labels):
        raise ValueError(
            "Empirical predictive metrics require unique substance names per repo because "
            "logged metric namespaces are keyed by substance."
        )
    return labels


def _collect_empirical_prediction_metric_observations(
    bundle: PredictiveBundle,
    batch: AICMECompartmentsDataBatch,
    *,
    model: Any,
) -> dict[int, list[dict[str, float]]]:
    """Collect per-target predictive metric observations for each batch slot."""

    with torch.inference_mode():
        samples_S = bundle.samples_S  # [S, B, It, Tr, 1]
        target_raw = bundle.target_raw  # [B, It, Tr, 1]
        target_mask = bundle.target_mask  # [B, It, Tr]
        pred_mean = samples_S.mean(dim=0)  # [B, It, Tr, 1]
        indiv_mask = batch.mask_target_individuals.bool()  # [B, It]

        batch_size = int(pred_mean.shape[0])
        target_capacity = int(pred_mean.shape[1])
        per_slot_metrics: dict[int, list[dict[str, float]]] = {}
        for batch_idx in range(batch_size):
            slot_metrics: list[dict[str, float]] = []
            for target_idx in range(target_capacity):
                if not bool(indiv_mask[batch_idx, target_idx]):
                    continue

                pred_mean_bt = pred_mean[batch_idx, target_idx].unsqueeze(0).unsqueeze(0)
                target_bt = target_raw[batch_idx, target_idx].unsqueeze(0).unsqueeze(0)
                target_mask_bt = target_mask[batch_idx, target_idx].unsqueeze(0).unsqueeze(0)
                slot_metrics.append(
                    _compute_predictive_target_metrics(
                        pred_mean=pred_mean_bt,
                        target_raw=target_bt,
                        target_mask=target_mask_bt,
                        model=model,
                    )
                )

            per_slot_metrics[batch_idx] = slot_metrics

        return per_slot_metrics


def _compute_empirical_predictive_metrics_from_batch_list(
    batch_list: Sequence[AICMECompartmentsDataBatch],
    *,
    model: Any,
    sample_size: int,
    repo_id: str | None,
) -> dict[str, dict[str, float]]:
    """Aggregate held-out empirical predictive metrics across permutation batches.

    This helper is specific to the empirical leave-one-out path where
    ``batch_list`` represents a list of permutation batches from one repo.

    Mental model
    ------------
    ``batch_list[p]`` is the ``p``-th leave-one-out permutation. Within each
    permutation batch, axis ``B`` indexes the stacked studies / drugs loaded
    from that empirical repo.

    The critical assumption is that the batch slots are aligned across the
    permutation list:

    - ``batch_list[0][b]`` and ``batch_list[1][b]`` refer to the same study /
      drug slot ``b``,
    - only the held-out target individual inside that slot changes from one
      permutation batch to the next.

    Therefore the averages performed below are *not* over raw dataloader
    batches. They are over the collection of valid held-out target individuals
    observed for the same batch slot ``b`` across the full permutation list.

    Concretely, for each batch slot ``b``:
    1. sample predictive trajectories for every permutation batch separately,
    2. reduce the stochastic sample axis ``S`` to a predictive mean inside that
       permutation,
    3. compute one metric dictionary for each valid held-out target individual
       in that slot,
    4. pool those metric dictionaries across permutations,
    5. report mean/std over that pooled set.

    So the final metric for one substance is a per-study/per-drug aggregate over
    held-out target realizations, not a mean over heterogeneous batches.
    """

    batches = _as_batch_list(batch_list)
    if not batches:
        return {}
    if int(sample_size) <= 0:
        raise ValueError("Empirical predictive metrics require sample_size > 0.")

    expected_labels: list[tuple[str, str]] | None = None
    per_slot_metric_observations: dict[int, list[dict[str, float]]] = {}
    slot_substances: dict[int, str] = {}
    model_device = getattr(model, "device", torch.device("cpu"))

    for permutation_idx, batch in enumerate(batches):
        batch_device = _move_batch_to_device(batch, model_device)
        slot_labels = _resolve_empirical_slot_labels(batch_device)
        if expected_labels is None:
            # Freeze the reference slot ordering from the first permutation.
            # Every later permutation must preserve the same ``B``-axis identity.
            expected_labels = slot_labels
            slot_substances = {
                batch_idx: substance for batch_idx, (_, substance) in enumerate(slot_labels)
            }
        elif slot_labels != expected_labels:
            raise ValueError(
                "Empirical predictive metrics require permutation-aligned batch slots. "
                f"Permutation 0 labels={expected_labels}, permutation {permutation_idx} "
                f"labels={slot_labels}."
            )

        predictive_payload = build_predictive_task_samples(
            model,
            batch_device,
            num_samples=int(sample_size),
        )
        # ``per_slot_metrics[batch_idx]`` collects one metric dict per valid
        # held-out target individual inside this permutation batch slot.
        per_slot_metrics = _collect_empirical_prediction_metric_observations(
            predictive_payload.predictive,
            batch_device,
            model=model,
        )
        for batch_idx, metric_list in per_slot_metrics.items():
            # We append, rather than average here, because the semantic unit is
            # "one held-out target observation set for slot ``batch_idx``".
            # The actual mean/std are computed only after all permutations have
            # contributed their observations for that slot.
            per_slot_metric_observations.setdefault(batch_idx, []).extend(metric_list)

    aggregated: dict[str, dict[str, float]] = {}
    for batch_idx, metric_list in per_slot_metric_observations.items():
        if not metric_list:
            continue
        substance = slot_substances[batch_idx]
        # At this point ``metric_list`` contains all held-out observations for
        # one aligned batch slot / substance across the permutation list.
        aggregated[substance] = _aggregate_prediction_metrics(
            {substance: metric_list},
            repo_id=repo_id,
        )[substance]
    return aggregated


def _collect_empirical_predictive_metric_observations_from_batch_list(
    batch_list: Sequence[AICMECompartmentsDataBatch],
    *,
    model: Any,
    sample_size: int,
) -> dict[str, list[dict[str, float]]]:
    """Collect one repo's held-out predictive observations grouped by substance.

    Unlike ``_compute_empirical_predictive_metrics_from_batch_list``, this helper
    intentionally keeps the raw per-target metric observations so that callers
    can pool them across repo boundaries before computing one final mean/std per
    substance.
    """

    batches = _as_batch_list(batch_list)
    if not batches:
        return {}
    if int(sample_size) <= 0:
        raise ValueError("Empirical predictive metrics require sample_size > 0.")

    model_device = getattr(model, "device", torch.device("cpu"))
    pooled_by_substance_key: dict[str, list[dict[str, float]]] = {}
    output_name_by_substance_key: dict[str, str] = {}

    for batch in batches:
        batch_device = _move_batch_to_device(batch, model_device)
        predictive_payload = build_predictive_task_samples(
            model,
            batch_device,
            num_samples=int(sample_size),
        )
        per_slot_metrics = _collect_empirical_prediction_metric_observations(
            predictive_payload.predictive,
            batch_device,
            model=model,
        )
        substance_names = _resolve_substance_names(batch_device)

        for batch_idx, metric_list in per_slot_metrics.items():
            if not metric_list:
                continue

            substance_name = substance_names[batch_idx]
            substance_key = _normalize_empirical_substance_group_key(substance_name)
            output_name_by_substance_key.setdefault(substance_key, substance_name)
            pooled_by_substance_key.setdefault(substance_key, []).extend(metric_list)

    return {
        output_name_by_substance_key[substance_key]: metric_list
        for substance_key, metric_list in pooled_by_substance_key.items()
        if metric_list
    }


def _compute_empirical_predictive_metrics_across_repos(
    *,
    empirical_repos: Sequence[str],
    datamodule: Any,
    split: str,
    model: Any,
    sample_size: int,
    batch_fetcher: Any | None = None,
) -> dict[str, dict[str, float]]:
    """Pool held-out predictive observations across repos and aggregate by substance.

    Each valid held-out target individual contributes exactly one observation to
    its normalized substance bucket. Repo boundaries disappear after this
    collection step; the final output reports mean/std per substance across the
    full pooled observation set.
    """

    pooled_by_substance_key: dict[str, list[dict[str, float]]] = {}
    output_name_by_substance_key: dict[str, str] = {}
    resolved_batch_fetcher = batch_fetcher or datamodule.get_empirical_batches

    for repo_id in empirical_repos:
        repo_batch_list = resolved_batch_fetcher(
            split=split,
            empirical_name=repo_id,
        )
        repo_metric_observations = _collect_empirical_predictive_metric_observations_from_batch_list(
            repo_batch_list,
            model=model,
            sample_size=sample_size,
        )
        for substance_name, metric_list in repo_metric_observations.items():
            if not metric_list:
                continue

            substance_key = _normalize_empirical_substance_group_key(substance_name)
            output_name_by_substance_key.setdefault(substance_key, substance_name)
            pooled_by_substance_key.setdefault(substance_key, []).extend(metric_list)

    if not pooled_by_substance_key:
        return {}

    pooled_by_substance_name = {
        output_name_by_substance_key[substance_key]: metric_list
        for substance_key, metric_list in pooled_by_substance_key.items()
        if metric_list
    }
    return _aggregate_prediction_metrics(
        pooled_by_substance_name,
        repo_id=None,
    )


def _flatten_aggregated_prediction_metrics(
    metrics_by_substance: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Flatten aggregated per-substance predictive metrics for scheduler logging."""

    out: dict[str, float] = {}
    for substance, metric_dict in metrics_by_substance.items():
        for metric_name, metric_value in metric_dict.items():
            if metric_name == "repo_id":
                continue
            out[f"{substance}/{metric_name}"] = float(metric_value)
    return out


def _predictive_images_per_batch(
    bundle: PredictiveBundle,
    batch: AICMECompartmentsDataBatch,
    *,
    model: Any,
    label: str,
    epoch: int,
    perm_index: int,
    output_root: Path,
    model_label: str,
    plot_kwargs: Optional[dict[str, Any]] = None,
    empirical_plots_per_substance: Optional[dict[str, int]] = None,
    number_of_predictions_plot_per_drug: Optional[int] = None,
    **_: Any,
) -> list[str]:
    """Render predictive images for a single batch permutation."""

    if getattr(model, "meta_dosing", None) is None:
        return []

    samples_S = bundle.samples_S  # [S, B, It, Tr, 1]
    times_S = bundle.times_S  # [S, B, It, Tr, 1]
    studies_for_batch = prediction_to_study_jsons(samples_S, times_S, batch, model.meta_dosing)
    if not studies_for_batch:
        return []

    plot_all = label == "Empirical"
    if plot_all and number_of_predictions_plot_per_drug is not None:
        if int(number_of_predictions_plot_per_drug) <= 0:
            return []
        if empirical_plots_per_substance is None:
            empirical_plots_per_substance = {}

        filtered_studies: list[Any] = []
        for study in studies_for_batch:
            substance_name = str(study.get("meta_data", {}).get("substance_name") or "").strip()
            if not substance_name:
                substance_name = f"substance_{len(filtered_studies)}"
            current_count = int(empirical_plots_per_substance.get(substance_name, 0))
            if current_count >= int(number_of_predictions_plot_per_drug):
                continue
            empirical_plots_per_substance[substance_name] = current_count + 1
            filtered_studies.append(study)
        studies_for_batch = filtered_studies

    if not studies_for_batch:
        return []

    studies = [studies_for_batch]
    output_dir = output_root / model_label / "predictions" / label.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"epoch_{epoch:03d}_perm_{perm_index:03d}.png"

    resolved_plot_kwargs = dict(plot_kwargs) if plot_kwargs else {}
    if plot_all and "number_of_predictions_plot_per_drug" not in resolved_plot_kwargs:
        resolved_plot_kwargs["number_of_predictions_plot_per_drug"] = 1

    render_kwargs: dict[str, Any] = {
        "studies": studies,
        "file_name": str(image_path),
        "plot_all_separately": plot_all,
    }
    if plot_all:
        render_kwargs["number_of_columns"] = 1
        render_kwargs["number_of_rows"] = None
    if resolved_plot_kwargs:
        render_kwargs["plot_kwargs"] = resolved_plot_kwargs

    img = plot_list_list_study_json(**render_kwargs)
    if not img:
        return []
    if isinstance(img, list):
        return [str(path) for path in img]
    return [str(img)]


def _generative_metrics_per_batch(
    bundle: GenerativeBundle,
    batch: AICMECompartmentsDataBatch,
    **_: Any,
) -> dict[str, torch.Tensor]:
    """Compute coverage metrics for new individuals on one batch permutation."""

    pred_values = bundle.pred_values  # [B, S, Tdistinct, 1]
    times = bundle.times  # [B, Tdistinct, 1]
    mask = bundle.mask  # [B, Tdistinct]
    return compute_percentile_coverage(
        pred_values,
        times,
        mask,
        batch.context_obs,
        batch.context_obs_time,
        batch.context_obs_mask,
    )


def _generative_images_per_batch(
    bundle: GenerativeBundle,
    batch: AICMECompartmentsDataBatch,
    *,
    model: Any,
    label: str,
    epoch: int,
    output_root: Path,
    model_label: str,
    **_: Any,
) -> list[str]:
    """Render new-individual images for a single batch permutation."""

    samples = bundle.samples  # [S, B, Tdistinct, 1]
    times = bundle.times  # [B, Tdistinct, 1]
    mask = bundle.mask  # [B, Tdistinct]

    studies = [
        studies_from_sampled_targets(
            db=batch,
            samples=samples,
            times=times,
            mask=mask,
            route_options=model.meta_dosing.route_options,
            dosing_time=float(model.meta_dosing.time),
        )
    ]

    plot_all = label == "Empirical"
    output_dir = output_root / model_label / "new_individuals" / label.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"epoch_{epoch:03d}.png"

    render_kwargs: dict[str, Any] = {
        "studies": studies,
        "file_name": str(image_path),
        "plot_all_separately": plot_all,
    }
    if plot_all:
        render_kwargs["number_of_columns"] = 1
        render_kwargs["number_of_rows"] = None

    img = plot_list_list_study_json(**render_kwargs)
    if not img:
        return []
    if isinstance(img, list):
        return [str(path) for path in img]
    return [str(img)]


def _vpc_images_per_batch(
    bundle: VPCBundle,
    batch: AICMECompartmentsDataBatch,
    *,
    label: str,
    epoch: int,
    output_root: Path,
    model_label: str,
    n_bins: int = 10,
    binning: str = "equal_count",
    log_y: bool = False,
    **_: Any,
) -> list[str]:
    """Render per-substance VPC images from observed and simulated empirical studies."""

    _ = batch
    if not bundle.observed_studies or not bundle.simulated_studies_by_substance:
        return []

    output_dir = output_root / model_label / "vpc" / label.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[str] = []
    for batch_idx, observed in enumerate(bundle.observed_studies):
        if batch_idx >= len(bundle.simulated_studies_by_substance):
            break
        simulated = bundle.simulated_studies_by_substance[batch_idx]
        if not simulated:
            continue

        try:
            vpc_results = compute_vpc_data(observed, simulated, n_bins=n_bins, binning=binning)
        except ValueError as exc:
            raw_substance_name = observed.get("meta_data", {}).get("substance_name")
            rendered_substance_name = display_substance_name(
                raw_substance_name,
                fallback=f"substance_{batch_idx}",
            )
            warnings.warn(
                (
                    f"Skipping VPC image for '{rendered_substance_name}' due to "
                    f"VPC input mismatch: {exc}"
                ),
                stacklevel=2,
            )
            continue
        if vpc_results.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        vpc_plot(vpc_results, ax=ax, log_y=log_y)
        raw_substance_name = observed.get("meta_data", {}).get("substance_name")
        ax.set_title(
            display_substance_name(
                raw_substance_name,
                fallback=f"substance_{batch_idx}",
            )
        )
        substance_name = safe_substance_name(
            raw_substance_name,
            fallback=f"substance_{batch_idx}",
        )
        image_path = output_dir / f"epoch_{epoch:03d}_{substance_name}_{batch_idx:03d}.png"
        fig.savefig(image_path, bbox_inches="tight")
        plt.close(fig)
        image_paths.append(str(image_path))

    return image_paths


def _compute_vpc_npde_pvalues_per_substance(bundle: VPCBundle) -> list[dict[str, float]]:
    """Compute NPDE p-values per substance from a cached VPC bundle."""

    max_substances = min(
        len(bundle.observed_studies),
        len(bundle.simulated_studies_by_substance),
    )
    pvalues_by_substance: list[dict[str, float]] = []
    for batch_idx in range(max_substances):
        observed = bundle.observed_studies[batch_idx]
        simulated = bundle.simulated_studies_by_substance[batch_idx]
        fallback = {"mean": float("nan"), "variance": float("nan"), "normality": float("nan")}

        if not simulated:
            pvalues_by_substance.append(fallback)
            continue

        try:
            npde_values = compute_npde_data(observed, simulated)
            raw_pvalues = npde_pvalues(npde_values)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            pvalues_by_substance.append(fallback)
            continue

        pvalues_by_substance.append(
            {
                "mean": float(raw_pvalues.get("mean", float("nan"))),
                "variance": float(raw_pvalues.get("variance", float("nan"))),
                "normality": float(raw_pvalues.get("normality", float("nan"))),
            }
        )

    return pvalues_by_substance


def _vpc_npde_pvalues_per_batch(
    bundle: VPCBundle,
    batch: AICMECompartmentsDataBatch,
    **_: Any,
) -> dict[str, torch.Tensor]:
    """Compute NPDE p-value metrics for each substance in one VPC bundle."""

    batch_size = len(batch.substance_name)
    metric_names = ("mean", "variance", "normality")
    metric_tensors = {
        f"npde_pvalue_{name}": torch.full((batch_size,), float("nan"), dtype=torch.float32)
        for name in metric_names
    }

    pvalues_by_substance = _compute_vpc_npde_pvalues_per_substance(bundle)
    for batch_idx, pvalues in enumerate(pvalues_by_substance):
        if batch_idx >= batch_size:
            break
        for name in metric_names:
            value = float(pvalues.get(name, float("nan")))
            if math.isfinite(value):
                metric_tensors[f"npde_pvalue_{name}"][batch_idx] = value

    return metric_tensors


def _aggregate_prediction_metrics(
    per_substance_perm_metrics: dict[str, list[dict[str, float]]],
    *,
    repo_id: str | None,
) -> dict[str, dict[str, float]]:
    """Aggregate predictive metrics across observations assigned to a substance.

    The input contract is intentionally generic:

    - keys are output substance names,
    - values are lists of metric dictionaries already collected for that
      substance.

    This helper does not know whether those metric dictionaries came from:
    - one metric per permutation,
    - one metric per held-out target individual across permutations,
    - or some other upstream grouping.

    It simply computes, for each substance and metric name:
    - the arithmetic mean across the list,
    - the sample standard deviation across the same list.

    In the synthetic scheduler path this usually means "average over
    permutations". In the empirical held-out path introduced by
    ``_compute_empirical_predictive_metrics_from_batch_list`` it means
    "average over all valid held-out target observations pooled for one aligned
    batch slot / substance across the permutation list".
    """

    final: dict[str, dict[str, float]] = {}
    for substance, metrics_list in per_substance_perm_metrics.items():
        n_perm = len(metrics_list)
        if n_perm == 0:
            continue

        metric_keys = list(metrics_list[0].keys())
        agg: dict[str, float] = {}
        for metric_name in metric_keys:
            values = [metric_dict[metric_name] for metric_dict in metrics_list]
            # ``values`` is the full observation set already assigned to this
            # substance by the caller. We intentionally do not re-weight by
            # batch size here because the caller has already decided what one
            # observation means in its own context.
            mean_val = sum(values) / n_perm
            if n_perm > 1:
                var = sum((value - mean_val) ** 2 for value in values) / (n_perm - 1)
            else:
                var = 0.0

            agg[metric_name] = float(mean_val)
            agg[f"{metric_name}_std"] = float(var**0.5)

        if repo_id is not None:
            agg["repo_id"] = str(repo_id)
        final[substance] = agg
    return final


def _aggregate_tensor_metrics(
    *,
    metrics: dict[str, torch.Tensor],
    batch: AICMECompartmentsDataBatch,
    repo_id: str | None,
) -> dict[str, dict[str, float]]:
    """Aggregate per-substance tensor metrics into scalar dictionaries."""

    resolved_repo = repo_id or "Synthetic"
    final: dict[str, dict[str, float]] = {}
    raw_substances = list(batch.substance_name)

    for batch_idx, raw_substance in enumerate(raw_substances):
        substance = str(raw_substance).strip() if raw_substance is not None else ""
        if not substance:
            substance = f"substance_{batch_idx}"

        per_substance: dict[str, float] = {}
        for metric_name, metric_tensor in metrics.items():
            metric_values = metric_tensor.detach().float().cpu()
            if metric_values.ndim == 0:
                if batch_idx != 0:
                    continue
                value = float(metric_values.item())
            else:
                if batch_idx >= metric_values.shape[0]:
                    continue
                metric_value = metric_values[batch_idx]
                if metric_value.numel() != 1:
                    continue
                value = float(metric_value.item())

            if not math.isfinite(value):
                continue
            per_substance[metric_name] = value

        if not per_substance:
            continue

        per_substance["repo_id"] = str(resolved_repo)
        final[substance] = per_substance

    return final


def summarize_across_repos(
    metrics_by_repo: dict[str, dict[str, dict[str, float]]],
    *,
    selected_drugs: Sequence[str],
    metric_name: str,
) -> float | None:
    """Average one metric across selected substances from all repos."""

    metric_key = str(metric_name).strip()
    if not metric_key:
        return None

    selected = {
        str(drug).strip().lower()
        for drug in selected_drugs
        if isinstance(drug, str) and str(drug).strip()
    }
    if not selected:
        return None

    values: list[float] = []
    for substances in metrics_by_repo.values():
        for substance, metric_dict in substances.items():
            if str(substance).strip().lower() not in selected:
                continue
            raw_value = metric_dict.get(metric_key)
            if raw_value is None:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)

    if not values:
        return None
    return float(sum(values) / len(values))


def _flatten_prediction_metrics(
    per_substance_perm_metrics: dict[str, list[dict[str, float]]],
    *,
    repo_id: str | None,
) -> dict[str, float]:
    final = _aggregate_prediction_metrics(
        per_substance_perm_metrics,
        repo_id=repo_id,
    )
    out: dict[str, float] = {}
    for substance, metric_dict in final.items():
        for metric_name, metric_value in metric_dict.items():
            if metric_name == "repo_id":
                continue
            out[f"{substance}/{metric_name}"] = float(metric_value)
    return out


def _flatten_tensor_metrics(
    *,
    metrics: dict[str, torch.Tensor],
    batch: AICMECompartmentsDataBatch,
    repo_id: str | None,
) -> dict[str, float]:
    final = _aggregate_tensor_metrics(
        metrics=metrics,
        batch=batch,
        repo_id=repo_id,
    )
    out: dict[str, float] = {}
    metric_tensors = {name: tensor.detach().float().cpu() for name, tensor in metrics.items()}
    for substance, metric_dict in final.items():
        for metric_name, metric_value in metric_dict.items():
            if metric_name == "repo_id":
                continue
            out[f"{substance}/{metric_name}"] = float(metric_value)

    for metric_name, metric_values in metric_tensors.items():
        if metric_values.ndim == 0:
            value = float(metric_values.item())
            if math.isfinite(value):
                out[f"mean/{metric_name}"] = value
                out[f"std/{metric_name}"] = 0.0
            continue

        finite_mask = torch.isfinite(metric_values)
        valid_values = metric_values[finite_mask]
        if valid_values.numel() == 0:
            continue
        out[f"mean/{metric_name}"] = float(valid_values.mean().item())
        out[f"std/{metric_name}"] = float(valid_values.std(unbiased=False).item())
    return out


def _image_outputs(image_paths: Sequence[str]) -> dict[str, Path]:
    return {f"image_{idx:03d}": Path(str(path)) for idx, path in enumerate(image_paths)}


def _concatenate_aicme_plot_batches(
    batches: Sequence[AICMECompartmentsDataBatch],
) -> AICMECompartmentsDataBatch:
    """Concatenate builder-style databatches along the leading study axis."""

    if not batches:
        raise ValueError("At least one databatch is required for synthetic MMD plotting.")

    concatenated_fields: list[Any] = []
    for field_name in AICMECompartmentsDataBatch._fields:
        values = [getattr(batch, field_name) for batch in batches]
        first_value = values[0]
        if isinstance(first_value, torch.Tensor):
            concatenated_fields.append(torch.cat(values, dim=0))
            continue
        if isinstance(first_value, list):
            merged_list: list[Any] = []
            for value in values:
                merged_list.extend(list(value))
            concatenated_fields.append(merged_list)
            continue
        concatenated_fields.append(first_value)

    return AICMECompartmentsDataBatch(*concatenated_fields)


def _concatenate_synthetic_mmd_bundles(
    bundles: Sequence[SyntheticMMDSeriesBundle],
) -> SyntheticMMDSeriesBundle:
    """Concatenate aligned synthetic-MMD bundles along the study axis."""

    if not bundles:
        raise ValueError("At least one aligned synthetic MMD bundle is required for plotting.")

    return SyntheticMMDSeriesBundle(
        observed_values=torch.cat([bundle.observed_values for bundle in bundles], dim=0),
        generated_values=torch.cat([bundle.generated_values for bundle in bundles], dim=0),
        times=torch.cat([bundle.times for bundle in bundles], dim=0),
        mask=torch.cat([bundle.mask for bundle in bundles], dim=0),
    )


def sub_task_plotting_mmd(
    *,
    plot_batches: Sequence[AICMECompartmentsDataBatch],
    aligned_bundles: Sequence[SyntheticMMDSeriesBundle],
    num_studies: int,
    trainer: Any,
    model_label: str,
    route_options: Optional[Sequence[str]] = None,
    plot_kwargs: Optional[dict[str, Any]] = None,
) -> Path | None:
    """Render and persist one overlay figure for diverse-experiment inspection."""

    if num_studies <= 0 or not plot_batches or not aligned_bundles:
        return None

    plot_batch = _concatenate_aicme_plot_batches(plot_batches)
    plot_bundle = _concatenate_synthetic_mmd_bundles(aligned_bundles)

    output_dir = (
        _resolve_metrics_report_root(trainer) / model_label / "diverse_experiment_distances"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / (
        f"epoch_{_resolve_epoch(trainer):03d}_"
        f"step_{int(getattr(trainer, 'global_step', 0)):07d}_samples.png"
    )

    plot_synthetic_mmd_overlay(
        plot_batch,
        observed_values=plot_bundle.observed_values,
        generated_values=plot_bundle.generated_values,
        times=plot_bundle.times,
        mask=plot_bundle.mask,
        num_studies=int(num_studies),
        route_options=route_options,
        file_name=str(image_path),
        plot_kwargs=plot_kwargs,
    )
    return image_path


def task_empirical_predictive_metrics(
    *,
    samples: Any | None,
    batches: Sequence[AICMECompartmentsDataBatch],
    task_cfg: Mapping[str, Any],
    trainer: Any,
    pl_module: Any,
) -> Mapping[str, Any]:
    """Backward-compatible alias for the canonical empirical predictive task."""

    from pff.training.callbacks.pk_task_empirical import (
        task_empirical_predictive_metrics as _task_impl,
    )

    return _task_impl(
        samples=samples,
        batches=batches,
        task_cfg=task_cfg,
        trainer=trainer,
        pl_module=pl_module,
    )


def task_empirical_heldout_generated_classifier(
    *,
    samples: Any | None,
    batches: Sequence[AICMECompartmentsDataBatch],
    task_cfg: Mapping[str, Any],
    trainer: Any,
    pl_module: Any,
) -> Mapping[str, Any]:
    """Backward-compatible alias for the canonical empirical classifier task."""

    from pff.training.callbacks.pk_task_empirical import (
        task_empirical_heldout_generated_classifier as _task_impl,
    )

    return _task_impl(
        samples=samples,
        batches=batches,
        task_cfg=task_cfg,
        trainer=trainer,
        pl_module=pl_module,
    )


def task_predictive_images(
    *,
    samples: PKTaskSamples | PredictiveTaskSamples | list[Any] | None,
    batches: Sequence[AICMECompartmentsDataBatch],
    task_cfg: Mapping[str, Any],
    trainer: Any,
    pl_module: Any,
) -> Mapping[str, Any]:
    """Backward-compatible alias for the canonical synthetic predictive image task."""

    from pff.training.callbacks.pk_task_synthetic import (
        task_predictive_images as _task_impl,
    )

    return _task_impl(
        samples=samples,
        batches=batches,
        task_cfg=task_cfg,
        trainer=trainer,
        pl_module=pl_module,
    )


def task_generative_metrics(
    *,
    samples: PKTaskSamples | list[PKTaskSamples] | None,
    batches: Sequence[AICMECompartmentsDataBatch],
    task_cfg: Mapping[str, Any],
    trainer: Any,
    pl_module: Any,
) -> Mapping[str, Any]:
    """Backward-compatible alias for the canonical synthetic generative metrics task."""

    from pff.training.callbacks.pk_task_synthetic import (
        task_generative_metrics as _task_impl,
    )

    return _task_impl(
        samples=samples,
        batches=batches,
        task_cfg=task_cfg,
        trainer=trainer,
        pl_module=pl_module,
    )


def task_generative_images(
    *,
    samples: PKTaskSamples | list[PKTaskSamples] | None,
    batches: Sequence[AICMECompartmentsDataBatch],
    task_cfg: Mapping[str, Any],
    trainer: Any,
    pl_module: Any,
) -> Mapping[str, Any]:
    """Backward-compatible alias for the canonical synthetic generative image task."""

    from pff.training.callbacks.pk_task_synthetic import (
        task_generative_images as _task_impl,
    )

    return _task_impl(
        samples=samples,
        batches=batches,
        task_cfg=task_cfg,
        trainer=trainer,
        pl_module=pl_module,
    )


def task_vpc_npde_pvalues(
    *,
    samples: PKTaskSamples | list[PKTaskSamples] | None,
    batches: Sequence[AICMECompartmentsDataBatch],
    task_cfg: Mapping[str, Any],
    trainer: Any,
    pl_module: Any,
) -> Mapping[str, Any]:
    """Backward-compatible alias for the canonical synthetic VPC metrics task."""

    from pff.training.callbacks.pk_task_synthetic import (
        task_vpc_npde_pvalues as _task_impl,
    )

    return _task_impl(
        samples=samples,
        batches=batches,
        task_cfg=task_cfg,
        trainer=trainer,
        pl_module=pl_module,
    )


def task_vpc_images(
    *,
    samples: PKTaskSamples | list[PKTaskSamples] | None,
    batches: Sequence[AICMECompartmentsDataBatch],
    task_cfg: Mapping[str, Any],
    trainer: Any,
    pl_module: Any,
) -> Mapping[str, Any]:
    """Backward-compatible alias for the canonical synthetic VPC image task."""

    from pff.training.callbacks.pk_task_synthetic import (
        task_vpc_images as _task_impl,
    )

    return _task_impl(
        samples=samples,
        batches=batches,
        task_cfg=task_cfg,
        trainer=trainer,
        pl_module=pl_module,
    )


def task_diverse_experiment_distances(
    *,
    samples: Any | None,
    batches: Sequence[AICMECompartmentsDataBatch],
    task_cfg: Mapping[str, Any],
    trainer: Any,
    pl_module: Any,
) -> Mapping[str, Any]:
    """Backward-compatible alias for the synthetic distance task entrypoint."""

    from pff.training.callbacks.pk_task_synthetic import (
        task_diverse_synthetic_experiment_sample_distances,
    )

    return task_diverse_synthetic_experiment_sample_distances(
        samples=samples,
        batches=batches,
        task_cfg=task_cfg,
        trainer=trainer,
        pl_module=pl_module,
    )


def task_diverse_synthetic_experiment_sample_distances(
    *,
    samples: Any | None,
    batches: Sequence[AICMECompartmentsDataBatch],
    task_cfg: Mapping[str, Any],
    trainer: Any,
    pl_module: Any,
) -> Mapping[str, Any]:
    """Backward-compatible export for the synthetic distance task entrypoint."""

    from pff.training.callbacks.pk_task_synthetic import (
        task_diverse_synthetic_experiment_sample_distances as _task_impl,
    )

    return _task_impl(
        samples=samples,
        batches=batches,
        task_cfg=task_cfg,
        trainer=trainer,
        pl_module=pl_module,
    )


def task_empirical_summary(
    *,
    samples: PKTaskSamples | PredictiveTaskSamples | list[Any] | None,
    batches: Sequence[AICMECompartmentsDataBatch],
    task_cfg: Mapping[str, Any],
    trainer: Any,
    pl_module: Any,
) -> Mapping[str, Any]:
    """Backward-compatible alias for the canonical empirical summary task."""

    from pff.training.callbacks.pk_task_empirical import (
        task_empirical_summary as _task_impl,
    )

    return _task_impl(
        samples=samples,
        batches=batches,
        task_cfg=task_cfg,
        trainer=trainer,
        pl_module=pl_module,
    )


__all__ = [
    "DiverseExperimentDistanceCollection",
    "GenerativeBundle",
    "GenerativeTaskSamples",
    "PKTaskSamples",
    "PredictiveTaskSamples",
    "PredictiveBundle",
    "SyntheticMMDSeriesBundle",
    "VPCBundle",
    "_compute_vpc_npde_pvalues_per_substance",
    "_generative_images_per_batch",
    "_generative_metrics_per_batch",
    "_prediction_metrics_per_batch",
    "_predictive_images_per_batch",
    "_vpc_images_per_batch",
    "_vpc_npde_pvalues_per_batch",
    "build_predictive_task_samples",
    "build_pk_task_samples",
    "sample_generative_bundle",
    "sample_predictive_bundle",
    "sample_vpc_bundle",
    "sub_task_plotting_mmd",
    "summarize_across_repos",
    "task_empirical_heldout_generated_classifier",
    "task_diverse_experiment_distances",
    "task_diverse_synthetic_experiment_sample_distances",
    "task_empirical_predictive_metrics",
    "task_empirical_summary",
    "task_generative_images",
    "task_generative_metrics",
    "task_predictive_images",
    "task_vpc_images",
    "task_vpc_npde_pvalues",
    "_reconstruct_full_target_databatch",
    "validate_generated_samples_match_target_schedule",
]
