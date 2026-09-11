"""Base pharmacokinetic model utilities and mixins.

This module provides :class:`BasePKModel`, which centralises utilities shared
across predictive and generative PK models, along with the
:class:`PredictiveMixin` and :class:`GenerativeMixin` helper classes.
"""

from __future__ import annotations

import inspect
import math
import os
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lightning.pytorch as pl
import torch
from huggingface_hub import PyTorchModelHubMixin
from torchtyping import TensorType

from pff.config_classes.node_pk_config import HFNodePKConfig, NodePKExperimentConfig
from pff.data.data_empirical.builder import prediction_to_study_jsons
from pff.data.data_empirical.json_schema import StudyJSON, studies_from_sampled_targets
from pff.data.datasets.aicme_datasets import (
    AICMECompartmentsDataBatch,
    list_of_databath_to_device,
)
from pff.metrics.quantiles_coverage import compute_percentile_coverage
from pff.models.amortized_inference.generative_pk import AbstractForwardOutputs
from pff.models.architectures import get_decoder
from pff.models.architectures.encoders_pk import get_individual_encoder
from pff.models.utils.scaler_selection import resolve_scaler_methods
from pff.models.utils.scalers import PKScaler
from pff.training.utils import metrics_are_finite
from pff.utils.plots.databatch_plot import plot_list_list_study_json
from pff.utils.tensors_operations import gather_distinct_times_per_substance


class PredictiveMixin(ABC):
    """Mixin supplying batch-level predictive sampling helpers."""

    @abstractmethod
    def sample_individual_prediction(
        self,
        databatch: AICMECompartmentsDataBatch,
        sample_size: int = 1,
    ) -> Tuple[
        TensorType["S", "B", "It", "Tr", 1],
        TensorType["S", "B", "It", "Tr", 1],
        TensorType["B", "It", "Tr", 1],
        TensorType["B", "It", "Tr"],
    ]:
        """Sample predictive trajectories for known individuals."""

    def _deprecated_log_predictions_from_batches(
        self,
        batch_list: Sequence[AICMECompartmentsDataBatch],
        label: str,
        epoch: int,
        batch_idx: int,
    ) -> None:
        """Log prediction plots if predictive sampling utilities are available."""

        if self.meta_dosing is None:
            raise AttributeError(
                "`meta_dosing` must be configured on BasePKModel before logging predictions."
            )

        logger = getattr(self, "logger", None)
        experiment = getattr(logger, "experiment", None) if logger else None
        studies = self.sample_individual_prediction_from_batch_list_to_studyjson(
            batch_list,
        )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            plot_kwargs = {
                "studies": studies,
                "file_name": tmp.name,
                "plot_all_separately": label == "Empirical",
            }
            if label == "Empirical":
                plot_kwargs["number_of_columns"] = 3
                plot_kwargs["number_of_rows"] = None
            img = plot_list_list_study_json(**plot_kwargs)

        if img:
            step = epoch if batch_idx < 0 else epoch * 1000 + batch_idx
            if isinstance(img, list):
                for idx, image_path in enumerate(img):
                    experiment.log_image(
                        image_path,
                        name=f"{label}/Predictions_E{epoch:03d}_{idx:02d}",
                        step=step,
                    )
                self._generated_images.extend(img)
            else:
                experiment.log_image(
                    img,
                    name=f"{label}/Predictions_E{epoch:03d}",
                    step=step,
                )
                self._generated_images.append(img)

    @torch.inference_mode()
    def sample_individual_prediction_from_batch_list_to_tensors(
        self,
        list_of_batches: Sequence[AICMECompartmentsDataBatch],
        sample_size: int = 8,
    ) -> Tuple[
        TensorType["S", "B", "P*It", "Tr", 1],
        TensorType["S", "B", "P*It", "Tr", 1],
        TensorType["B", "P*It", "Tr", 1],
        TensorType["B", "P*It", "Tr"],
    ]:
        """Concatenate predictions for a list of permutations along individuals."""

        all_times: List[torch.Tensor] = []
        all_real: List[torch.Tensor] = []
        all_samples: List[torch.Tensor] = []
        all_masks: List[torch.Tensor] = []

        for batch in list_of_batches:
            (
                prediction_sample,
                prediction_time,
                real_observations,
                real_observations_mask,
            ) = self.sample_individual_prediction(batch, sample_size=sample_size)
            all_samples.append(prediction_sample)
            all_times.append(prediction_time)
            all_real.append(real_observations)
            all_masks.append(real_observations_mask)

        all_samples = torch.cat(all_samples, dim=2) if all_samples else torch.empty(0)
        all_times = torch.cat(all_times, dim=2) if all_times else torch.empty(0)
        all_real = torch.cat(all_real, dim=1) if all_real else torch.empty(0)
        all_masks = torch.cat(all_masks, dim=1) if all_masks else torch.empty(0)
        return all_samples, all_times, all_real, all_masks

    @torch.inference_mode()
    def sample_individual_prediction_from_batch_list(
        self,
        list_of_batches: Sequence[AICMECompartmentsDataBatch],
        sample_size: int = 8,
    ) -> Tuple[
        TensorType["S", "B", "P*It", "Tr", 1],
        TensorType["S", "B", "P*It", "Tr", 1],
        TensorType["B", "P*It", "Tr", 1],
        TensorType["B", "P*It", "Tr"],
    ]:
        """Alias kept for backward compatibility with legacy callers."""

        return self.sample_individual_prediction_from_batch_list_to_tensors(
            list_of_batches, sample_size=sample_size
        )

    @torch.inference_mode()
    def sample_individual_prediction_from_batch_list_to_studyjson(
        self,
        list_of_batches: Sequence[AICMECompartmentsDataBatch],
        sample_size: int = 8,
    ) -> List[List[StudyJSON]]:
        """Convert predictive samples to nested ``StudyJSON`` structures."""

        if self.meta_dosing is None:
            raise AttributeError(
                "`meta_dosing` must be configured on BasePKModel before building StudyJSONs."
            )

        studies_per_perm: List[List[StudyJSON]] = []
        for batch in list_of_batches:
            (
                prediction_sample,
                prediction_time,
                _,
                _,
            ) = self.sample_individual_prediction(batch, sample_size=sample_size)
            studies = prediction_to_study_jsons(
                prediction_sample,
                prediction_time,
                batch,
                self.meta_dosing,
            )
            studies_per_perm.append(studies)
        return studies_per_perm

    def forward_report(
        self,
        databatch_list: Sequence[AICMECompartmentsDataBatch],
        outputs_collector: "AbstractForwardOutputs",
    ) -> Tuple[List[Dict[str, Any]] | None, Dict[str, Dict[str, float]]]:
        """Compute evaluation metrics from aggregated forward outputs."""

        per_substance = self.forward_report_per_substance(
            databatch_list,
            outputs_collector,
        )
        return None, per_substance

    def forward_report_per_individual(
        self,
        databatch_list: Sequence[AICMECompartmentsDataBatch],
        mean_predictions: torch.Tensor,
        logvar_predictions: torch.Tensor,
        remainder_targets: torch.Tensor,
        remainder_masks: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """Compute per-individual metrics averaged across permutations."""

        meta_batch = databatch_list[0]

        per_individual: List[Dict[str, Any]] = []
        B, I, _, _ = mean_predictions.shape

        for b in range(B):
            study = meta_batch.study_name[b]
            substance = meta_batch.substance_name[b]
            for j in range(I):
                mean_subject = mean_predictions[b, j].unsqueeze(0).unsqueeze(0)
                logvar_subject = logvar_predictions[b, j].unsqueeze(0).unsqueeze(0)
                target_subject = remainder_targets[b, j].unsqueeze(0).unsqueeze(0)
                mask_subject = remainder_masks[b, j].unsqueeze(0).unsqueeze(0)

                _ = logvar_subject

                rmse_val = self.masked_rmse_loss(
                    mean_subject,
                    target_subject,
                    mask_subject,
                )["rmse"].item()
                log_rmse_val = self.masked_log_rmse_loss(
                    mean_subject,
                    target_subject,
                    mask_subject,
                )["rmse"].item()
                r2_val = self.masked_r2_score(
                    mean_subject,
                    target_subject,
                    mask_subject,
                ).item()
                log_r2_val = self.masked_log_r2_score(
                    mean_subject,
                    target_subject,
                    mask_subject,
                ).item()

                metrics = {
                    "rmse": rmse_val,
                    "log_rmse": log_rmse_val,
                    "r2": r2_val,
                    "log_r2": log_r2_val,
                }

                subject_name = meta_batch.target_subject_name[b][j]
                per_individual.append(
                    {
                        "study": study,
                        "substance": substance,
                        "subject": subject_name,
                        "metrics": metrics,
                    }
                )

        return per_individual

    def forward_report_per_substance(
        self,
        databatch_list: Sequence[AICMECompartmentsDataBatch],
        outputs_collector: "AbstractForwardOutputs",
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics across permutations at the substance level."""

        meta_batch = databatch_list[0]

        predictions = outputs_collector.prepare_predictions()
        if not predictions:
            return {}

        aggregated_sums: Dict[str, Dict[str, float]] = {}
        aggregated_counts: Dict[str, int] = {}

        for perm in predictions:
            mean_pred_perm = perm["mean"]
            target_perm = perm["target"]
            mask_perm = perm["mask"]

            perm_sums: Dict[str, Dict[str, float]] = {}
            perm_counts: Dict[str, int] = {}
            B, I, _, _ = mean_pred_perm.shape
            for b in range(B):
                if len(meta_batch.substance_name) == B:
                    substance = meta_batch.substance_name[b]
                else:
                    substance = meta_batch.substance_name[0][b]

                for j in range(I):
                    mean_subject = mean_pred_perm[b, j].unsqueeze(0).unsqueeze(0)
                    target_subject = target_perm[b, j].unsqueeze(0).unsqueeze(0)
                    mask_subject = mask_perm[b, j].unsqueeze(0).unsqueeze(0)

                    rmse_val = self.masked_rmse_loss(
                        mean_subject,
                        target_subject,
                        mask_subject,
                    )["rmse"].item()
                    log_rmse_val = self.masked_log_rmse_loss(
                        mean_subject,
                        target_subject,
                        mask_subject,
                    )["rmse"].item()
                    r2_val = self.masked_r2_score(
                        mean_subject,
                        target_subject,
                        mask_subject,
                    ).item()
                    log_r2_val = self.masked_log_r2_score(
                        mean_subject,
                        target_subject,
                        mask_subject,
                    ).item()

                    metrics = {
                        "rmse": rmse_val,
                        "log_rmse": log_rmse_val,
                        "r2": r2_val,
                        "log_r2": log_r2_val,
                    }

                    if substance not in perm_sums:
                        perm_sums[substance] = dict.fromkeys(metrics, 0.0)
                        perm_counts[substance] = 0
                    for key, value in metrics.items():
                        perm_sums[substance][key] += value
                    perm_counts[substance] += 1

            for substance, sums in perm_sums.items():
                if substance not in aggregated_sums:
                    aggregated_sums[substance] = dict.fromkeys(sums, 0.0)
                    aggregated_counts[substance] = 0
                denom = max(perm_counts[substance], 1)
                for metric, value in sums.items():
                    aggregated_sums[substance][metric] += value / denom
                aggregated_counts[substance] += 1

        per_substance = {
            substance: {
                metric: aggregated_sums[substance][metric] / max(aggregated_counts[substance], 1)
                for metric in aggregated_sums[substance]
            }
            for substance in aggregated_sums
        }

        return per_substance

    def _compute_metrics_from_batch_list(
        self,
        batch_list: Sequence[AICMECompartmentsDataBatch],
        repo_id: str | None = None,
    ):
        per_substance_perm_metrics: dict[str, list[dict[str, float]]] = {}
        prediction_cache = {}  # <-- NEW

        for p_idx, batch in enumerate(batch_list):
            # run prediction ONCE per batch_list entry
            samples_S, times_S, target_raw, target_mask = self.sample_individual_prediction(batch)

            # store in cache so we reuse later
            prediction_cache[p_idx] = {
                "samples_S": samples_S,
                "times_S": times_S,
                "target_raw": target_raw,
                "target_mask": target_mask,
                "batch": batch,
            }

            pred_mean = samples_S.mean(dim=0)  # [B,1,Tr,1]

            B, It, Tr, _ = pred_mean.shape
            indiv_mask = batch.mask_target_individuals

            # resolve names
            raw_names = list(batch.substance_name)
            substance_names: list[str] = []
            for b, name in enumerate(raw_names):
                if name is None or name == "" or str(name).strip() == "":
                    substance_names.append(f"substance_{b}")
                else:
                    substance_names.append(str(name))

            # compute metrics per substance / permutation
            for b in range(B):
                substance = substance_names[b]

                if not indiv_mask[b, 0]:
                    metrics_b = {"rmse": 0.0, "log_rmse": 0.0, "r2": 0.0, "log_r2": 0.0}
                else:
                    pm = pred_mean[b, 0].unsqueeze(0).unsqueeze(0)
                    tg = target_raw[b, 0].unsqueeze(0).unsqueeze(0)
                    mk = target_mask[b, 0].unsqueeze(0).unsqueeze(0)

                    metrics_b = {
                        "rmse": self.masked_rmse_loss(pm, tg, mk)["rmse"].item(),
                        "log_rmse": self.masked_log_rmse_loss(pm, tg, mk)["rmse"].item(),
                        "r2": self.masked_r2_score(pm, tg, mk).item(),
                        "log_r2": self.masked_log_r2_score(pm, tg, mk).item(),
                    }

                per_substance_perm_metrics.setdefault(substance, []).append(metrics_b)

        # ------------------------------------------------------------------
        # average across permutations + compute std across permutations
        # ------------------------------------------------------------------
        final: dict[str, dict[str, float]] = {}
        for substance, metrics_list in per_substance_perm_metrics.items():
            n_perm = len(metrics_list)
            if n_perm == 0:
                continue

            # all metrics have same keys by construction
            metric_keys = list(metrics_list[0].keys())
            agg: dict[str, float] = {}

            for m in metric_keys:
                vals = [md[m] for md in metrics_list]
                mean_val = sum(vals) / n_perm

                if n_perm > 1:
                    var = sum((v - mean_val) ** 2 for v in vals) / (n_perm - 1)
                else:
                    var = 0.0

                std_val = var**0.5

                agg[m] = float(mean_val)
                agg[m + "_std"] = float(std_val)

            agg["repo_id"] = repo_id
            final[substance] = agg

        return final, prediction_cache

    def _log_prediction_images_from_batch_list_with_cache(
        self,
        prediction_cache,  # NEW
        label: str,
        epoch: int,
        batch_idx: int,
        experiment,
    ):
        step = epoch if batch_idx < 0 else epoch * 1000 + batch_idx

        for p_idx, entry in prediction_cache.items():
            samples_S = entry["samples_S"]
            times_S = entry["times_S"]
            batch = entry["batch"]

            studies = [prediction_to_study_jsons(samples_S, times_S, batch, self.meta_dosing)]
            if not studies:
                continue

            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                plot_kwargs = {
                    "studies": studies,
                    "file_name": tmp.name,
                    "plot_all_separately": label == "Empirical",
                }
                if label == "Empirical":
                    plot_kwargs["number_of_columns"] = 1
                    plot_kwargs["number_of_rows"] = None

                img = plot_list_list_study_json(**plot_kwargs)

            if not img:
                continue

            if isinstance(img, list):
                for idx, image_path in enumerate(img):
                    experiment.log_image(
                        image_path,
                        name=f"{label}/Predictions_E{epoch:03d}_P{p_idx:03d}_{idx:02d}",
                        step=step,
                    )
                    self._generated_images.append(image_path)
            else:
                experiment.log_image(
                    img,
                    name=f"{label}/Predictions_E{epoch:03d}_P{p_idx:03d}",
                    step=step,
                )
                self._generated_images.append(img)
            break

    def log_predictions_from_batches(
        self,
        batch_list,
        label: str,
        epoch: int,
        batch_idx: int,
        repo_id: str | None = None,
    ):
        if not batch_list:
            return

        # resolve experiment logger
        trainer = getattr(self, "_trainer", None)
        logger = getattr(trainer, "logger", None) if trainer else None
        experiment = getattr(logger, "experiment", None) if logger else None
        if experiment is None:
            experiment = getattr(self, "_experiment_override", None)

        step = epoch if batch_idx < 0 else epoch * 1000 + batch_idx

        # ---- 1) METRICS (computed once, also produces prediction cache) ----
        final_metrics, pred_cache = self._compute_metrics_from_batch_list(
            batch_list, repo_id=repo_id
        )

        # ---- 2) IMAGE LOGGING (reuses pred_cache, no recomputation) --------
        self._log_prediction_images_from_batch_list_with_cache(
            prediction_cache=pred_cache,
            label=label,
            epoch=epoch,
            batch_idx=batch_idx,
            experiment=experiment,
        )

        # ---- 3) WRITE METRICS TO EXPERIMENT --------------------------------
        if experiment is not None:
            for substance, d in final_metrics.items():
                rid = d.get("repo_id") or (repo_id or "Synthetic")

                for metric_name, metric_value in d.items():
                    # skip metadata fields
                    if metric_name == "repo_id":
                        continue

                    experiment.log_metric(
                        name=f"{label}/{rid}/{substance}/{metric_name}",
                        value=float(metric_value),
                        step=step,
                    )


class GenerativeMixin(ABC):
    """Mixin supplying utilities for sampling brand new individuals."""

    @abstractmethod
    def sample_new_individual(
        self,
        db: AICMECompartmentsDataBatch,
        sample_size: int = 10,
        decode_times: Tuple[
            TensorType["B", "Tdistinct_max", 1],
            TensorType["B", "Tdistinct_max"],
        ]
        | None = None,
        ignore_logvar: bool = True,
        num_steps: int = None,
    ) -> Tuple[
        TensorType["S", "B", "Tdistinct_max", 1],
        TensorType["B", "Tdistinct_max", 1],
        TensorType["B", "Tdistinct_max"],
    ]:
        """Sample trajectories for new individuals conditioned on context."""

    @torch.inference_mode()
    def sample_new_individuals_to_studyjson(
        self,
        db: AICMECompartmentsDataBatch,
        sample_size: int = 8,
        decode_times: Tuple[
            TensorType["B", "Tdistinct_max", 1],
            TensorType["B", "Tdistinct_max"],
        ]
        | None = None,
        num_steps: int = None,
    ) -> List[StudyJSON]:
        """Generate new individuals and convert them into ``StudyJSON`` objects."""

        samples, times, mask = self.sample_new_individual(
            db, sample_size=sample_size, decode_times=decode_times, num_steps=num_steps
        )
        if self.meta_dosing is None:
            raise AttributeError(
                "model_config must define a `dosing` section for StudyJSON conversion."
            )

        return studies_from_sampled_targets(
            db=db,
            samples=samples,
            times=times,
            mask=mask,
            route_options=self.meta_dosing.route_options,
            dosing_time=float(self.meta_dosing.time),
        )

    @torch.inference_mode()
    def sample_new_individuals_from_batchlist_to_study_json(
        self,
        list_of_batches: Sequence[AICMECompartmentsDataBatch],
        sample_size: int = 8,
        decode_times: Tuple[
            TensorType["B", "Tdistinct_max", 1],
            TensorType["B", "Tdistinct_max"],
        ]
        | None = None,
        num_steps: int = None,
        max_permutations: int = 3,
    ) -> List[List[StudyJSON]]:
        """Return nested ``StudyJSON`` objects for a list of permutations."""

        studies_per_perm: List[List[StudyJSON]] = []
        for perm_idx, batch in enumerate(list_of_batches):
            if perm_idx >= max_permutations:
                break
            studies_per_perm.append(
                self.sample_new_individuals_to_studyjson(
                    batch,
                    sample_size=sample_size,
                    decode_times=decode_times,
                    num_steps=num_steps,
                )
            )
        return studies_per_perm

    def log_new_individuals_from_batches(
        self,
        batch_list: Sequence[AICMECompartmentsDataBatch],
        label: str,
        epoch: int,
        batch_idx: int,
        repo_id: str,
    ) -> None:
        """Log sampled new individuals as images to the active experiment.

        The function also reports coverage metrics emitted by
        :func:`compute_percentile_coverage` under the experiment scopes
        ``"{label}/{repo_id}/{substance}/{metric_name}"`` for per-substance
        values and ``"{label}/{repo_id}/mean|std/{metric_name}"`` for the
        repository-level aggregates.
        """

        if self.meta_dosing is None:
            raise AttributeError(
                "`meta_dosing` must be configured on BasePKModel before logging reconstructions."
            )

        trainer = getattr(self, "_trainer", None)
        logger = getattr(trainer, "logger", None) if trainer is not None else None
        experiment = getattr(logger, "experiment", None) if logger else None

        studies: List[List[StudyJSON]] = []
        batch = batch_list[0]
        samples, times, mask = self.sample_new_individual(batch)

        # ``sample_new_individual`` returns [S, B, T, 1]; swap to [B, S, T, 1]
        pred_values = samples.transpose(0, 1)
        metrics = compute_percentile_coverage(
            pred_values,
            times,
            mask,
            batch.context_obs,
            batch.context_obs_time,
            batch.context_obs_mask,
        )

        step = epoch if batch_idx < 0 else epoch * 1000 + batch_idx

        self._log_new_individuals_metrics(
            experiment=experiment,
            metrics=metrics,
            batch=batch,
            label=label,
            repo_id=repo_id,
            step=step,
        )

        studies.append(
            studies_from_sampled_targets(
                db=batch,
                samples=samples,
                times=times,
                mask=mask,
                route_options=self.meta_dosing.route_options,
                dosing_time=float(self.meta_dosing.time),
            )
        )

        self._log_new_individuals_images(
            experiment=experiment,
            studies=studies,
            label=label,
            epoch=epoch,
            step=step,
        )

    def _log_new_individuals_metrics(
        self,
        *,
        experiment: Optional[Any],
        metrics: Dict[str, torch.Tensor],
        batch: AICMECompartmentsDataBatch,
        label: str,
        repo_id: str,
        step: int,
    ) -> None:
        """Log coverage metrics for sampled individuals.

        Parameters
        ----------
        experiment:
            Optional logger experiment used to persist the metrics.
        metrics:
            Mapping produced by :func:`compute_percentile_coverage`.
        batch:
            Batch that provided context observations and metadata.
        label:
            High-level scope that prefixes all metric names (``Train``/``Val``/``Empirical``).
        repo_id:
            Repository identifier, typically the Hugging Face repo name.
        step:
            Global logging step used for all metric entries.
        """

        if experiment is None:
            return

        for metric_name, metric_tensor in metrics.items():
            if metric_tensor.numel() == 0:
                continue
            metric_values = metric_tensor.detach().float().cpu()
            finite_mask = torch.isfinite(metric_values)
            for substance, value_tensor, is_finite in zip(
                batch.substance_name, metric_values, finite_mask
            ):
                value = value_tensor.item()
                if not is_finite.item():
                    continue
                metric_full_name = f"{label}/{repo_id}/{substance}/{metric_name}"
                experiment.log_metric(
                    name=metric_full_name,
                    value=value,
                    step=step,
                )
            valid_values = metric_values[finite_mask]
            if valid_values.numel() == 0:
                continue
            experiment.log_metric(
                name=f"{label}/{repo_id}/mean/{metric_name}",
                value=valid_values.mean().item(),
                step=step,
            )
            experiment.log_metric(
                name=f"{label}/{repo_id}/std/{metric_name}",
                value=valid_values.std(unbiased=False).item(),
                step=step,
            )

    def _log_new_individuals_images(
        self,
        *,
        experiment: Optional[Any],
        studies: List[List[StudyJSON]],
        label: str,
        epoch: int,
        step: int,
    ) -> None:
        """Log plots of the sampled individuals to the active experiment."""

        if not studies or experiment is None:
            return

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            plot_kwargs = {
                "studies": studies,
                "file_name": tmp.name,
                "plot_all_separately": label == "Empirical",
            }
            if label == "Empirical":
                plot_kwargs["number_of_columns"] = 1
                plot_kwargs["number_of_rows"] = None
            img = plot_list_list_study_json(**plot_kwargs)

        if img:
            if isinstance(img, list):
                for idx, image_path in enumerate(img):
                    experiment.log_image(
                        image_path,
                        name=f"{label}/NewIndividuals_E{epoch:03d}_{idx:02d}",
                        step=step,
                    )
                self._generated_images.extend(img)
            else:
                experiment.log_image(
                    img,
                    name=f"{label}/NewIndividuals_E{epoch:03d}",
                    step=step,
                )
                self._generated_images.append(img)


class BasePKModel(pl.LightningModule, PyTorchModelHubMixin, ABC):
    """Base class exposing utilities shared by PK models."""

    config_class = HFNodePKConfig

    def __init__(self, model_config: NodePKExperimentConfig):
        super().__init__()
        self.model_config = model_config
        self.config = HFNodePKConfig.from_nodepk(model_config)

        network_cfg = getattr(model_config, "network", None)
        self.loss_name = getattr(network_cfg, "loss_name", None)
        train_cfg = getattr(model_config, "train", None)
        self.meta_dosing = getattr(self.model_config, "dosing", None)
        mix_cfg = getattr(model_config, "mix_data", None)

        self.encoder = get_individual_encoder(model_config)
        self.decoder = get_decoder(model_config)

        if train_cfg is not None:
            self.learning_rate = train_cfg.learning_rate
            self.weight_decay = train_cfg.weight_decay
        else:
            self.learning_rate = 0.0
            self.weight_decay = 0.0

        value_method, time_method = resolve_scaler_methods(mix_cfg)
        self.scaler = PKScaler(value_method=value_method, time_method=time_method)

        self._generated_images: List[str] = []

    def __del__(self):
        self._delete_images()

    def _should_log_predictions(self) -> bool:
        """Return ``True`` when predictive visualisations are enabled."""
        network_cfg = getattr(self.model_config, "network", None)
        training_cfg = getattr(self.model_config, "train", None)
        reconstruction_only = bool(getattr(network_cfg, "reconstruction_only", False))
        log_prediction_in_val = bool(getattr(training_cfg, "log_prediction_in_val", False))
        return (not reconstruction_only) and log_prediction_in_val

    def _should_log_reconstructions(self) -> bool:
        """Return ``True`` when predictive visualisations are enabled."""
        network_cfg = getattr(self.model_config, "network", None)
        training_cfg = getattr(self.model_config, "train", None)
        prediction_only = bool(getattr(network_cfg, "prediction_only", False))
        log_prediction_in_val = bool(getattr(training_cfg, "log_reconstruction_in_val", False))
        return (not prediction_only) and log_prediction_in_val

    # ------------------------------------------------------------------
    # LOSSES
    # ------------------------------------------------------------------
    def compute_loss(
        self,
        mean: TensorType["B", "C", "T", 1],
        logvar: TensorType["B", "C", "T", 1],
        target: TensorType["B", "C", "T", 1],
        mask: TensorType["B", "C", "T"],
        H: TensorType["B", "C", "T", "p"] = None,
        *,
        mask_individuals: TensorType["B", "C"] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the reconstruction loss dictated by the configured objective.

        Parameters
        ----------
        mean:
            Mean predictions in the (potentially scaled) space.
        logvar:
            Log-variance predictions matching ``mean``.
        target:
            Target trajectories against which ``mean`` is evaluated.
        mask:
            Boolean mask indicating valid target entries.
        H:
            Optional linear transform used for multivariate Gaussian losses.
        mask_individuals:
            Optional mask specifying which individuals should contribute to the
            loss.
        """

        if mask_individuals is not None:
            mask = mask & mask_individuals.unsqueeze(-1)
        if self.loss_name == "mse":
            loss_dict = self.masked_mse_loss(mean, target, mask, mask_individuals=mask_individuals)
        elif self.loss_name == "rmse":
            loss_dict = self.masked_rmse_loss(mean, target, mask, mask_individuals=mask_individuals)
        elif self.loss_name == "nll":
            loss_dict = self.masked_gaussian_nll_loss(
                mean, logvar, target, mask, mask_individuals=mask_individuals
            )
        elif self.loss_name == "log_nll":
            loss_dict = self.masked_gaussian_nll_log_loss(
                mean, logvar, target, mask, mask_individuals=mask_individuals
            )
        elif self.loss_name == "mv_nll":
            loss_dict = self.masked_gaussian_nll_loss_mv(
                mean, H, target, mask, mask_individuals=mask_individuals
            )
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_name}")

        # supplementary reporting metrics to preserve historical logging
        loss_dict["log_rmse"] = self.masked_log_rmse_loss(
            mean, target, mask, mask_individuals=mask_individuals
        )["rmse"]
        loss_dict["r2"] = self.masked_r2_score(
            mean, target, mask, mask_individuals=mask_individuals
        )
        loss_dict["log_r2"] = self.masked_log_r2_score(
            mean, target, mask, mask_individuals=mask_individuals
        )

        return loss_dict

    def masked_gaussian_nll_loss_mv(
        self,
        mean: TensorType["B", "C", "T", 1],
        H: TensorType["B", "C", "T", "p"],
        target: TensorType["B", "C", "T", 1],
        mask: TensorType["B", "C", "T"],
        mask_individuals: TensorType["B", "C"] | None = None,
        jitter: float = 1e-5,
    ) -> dict[str, torch.Tensor]:
        """
        Multivariate NLL with *per-series* masking.

        Observations without valid entries are removed from the covariance
        structure, yielding a smaller system.  The implementation follows
        equation (10) from the original paper where ``L = lower(H Hᵀ)`` and
        ``Σ = L Lᵀ``.
        """

        if H is None:  # fallback to diagonal path
            return self.masked_gaussian_nll_loss(mean, mean.new_zeros(()), target, mask)

        if mask_individuals is not None:
            mask = mask & mask_individuals.unsqueeze(-1)

        B, C, T, _ = mean.shape
        device = mean.device
        logs2pi = torch.log(torch.tensor(2.0 * torch.pi, device=device))

        mean = mean.view(-1, T)
        target = target.view(-1, T)
        mask = mask.view(-1, T)
        H = H.view(-1, T, H.shape[-1])

        total_ll = total_cnt = mean.new_tensor(0.0)
        for m, t, h, msk in zip(mean, target, H, mask):
            idx = msk.nonzero(as_tuple=False).squeeze(-1)
            k = idx.numel()
            if k == 0:
                continue

            mu = m[idx]
            y = t[idx]
            Hk = h[idx]

            # L = torch.linalg.cholesky((Hk @ Hk.T) + jitter * torch.eye(k, device=device))
            L = torch.tril(Hk @ Hk.T)
            alpha = torch.cholesky_solve((y - mu).unsqueeze(-1), L).squeeze(-1)
            nll_i = 0.5 * (
                alpha @ (y - mu) + 2.0 * torch.log(torch.diagonal(L)).sum() + k * logs2pi
            )

            total_ll += nll_i
            total_cnt += k

        loss = total_ll / total_cnt.clamp(min=1)
        rmse = torch.sqrt(((mean - target) ** 2 * mask).sum() / total_cnt.clamp(min=1))
        return {"loss": loss, "rmse": rmse, "count": total_cnt}

    def masked_gaussian_nll_loss(
        self,
        mean: TensorType["B", "C", "T", 1],
        logvar: TensorType["B", "C", "T", 1],
        target: TensorType["B", "C", "T", 1],
        mask: TensorType["B", "C", "T"],
        *,
        mask_individuals: TensorType["B", "C"] | None = None,
        strict: bool = False,
        min_valid_fraction: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """
        NaN-safe Gaussian NLL.

        When ``strict`` is true, samples with fewer than
        ``min_valid_fraction`` valid time-steps are excluded from the loss.
        """

        finite = (
            torch.isfinite(mean).squeeze(-1)
            & torch.isfinite(logvar).squeeze(-1)
            & torch.isfinite(target).squeeze(-1)
        )
        if mask_individuals is not None:
            mask = mask & mask_individuals.unsqueeze(-1)
        valid_mask = mask & finite

        if strict:
            B, _, _ = valid_mask.shape
            frac_valid = valid_mask.float().view(B, -1).mean(dim=1)
            keep = frac_valid >= min_valid_fraction
            if not keep.any():
                dummy = torch.zeros([], device=mean.device, requires_grad=True)
                return {"loss": dummy, "rmse": dummy}

            valid_mask = valid_mask[keep]
            mean = mean[keep]
            logvar = logvar[keep]
            target = target[keep]

        mean = torch.where(valid_mask[..., None], mean, 0.0)
        logvar = torch.where(valid_mask[..., None], logvar, 0.0)
        target = torch.where(valid_mask[..., None], target, 0.0)

        var = logvar.exp()
        sq_error = (mean - target) ** 2
        nll = 0.5 * (logvar + sq_error / var)
        nll += 0.5 * torch.log(torch.tensor(2.0 * torch.pi, device=mean.device))
        nll = nll.squeeze(-1) * valid_mask

        total_nll = nll.sum()
        total_count = valid_mask.sum().clamp(min=1)
        loss = total_nll / total_count
        rmse = torch.sqrt((sq_error.squeeze(-1) * valid_mask).sum() / total_count)

        return {"loss": loss, "rmse": rmse, "count": total_count}

    def masked_gaussian_nll_log_loss(
        self,
        mean: TensorType["B", "C", "T", 1],
        logvar: TensorType["B", "C", "T", 1],
        target: TensorType["B", "C", "T", 1],
        mask: TensorType["B", "C", "T"],
        *,
        mask_individuals: TensorType["B", "C"] | None = None,
        eps: float = 1e-8,
    ) -> dict[str, torch.Tensor]:
        """Gaussian NLL on the log scale with masking."""

        finite = (
            torch.isfinite(mean).squeeze(-1)
            & torch.isfinite(logvar).squeeze(-1)
            & torch.isfinite(target).squeeze(-1)
        )
        positive = (mean.squeeze(-1) > 0) & (target.squeeze(-1) > 0)
        if mask_individuals is not None:
            mask = mask & mask_individuals.unsqueeze(-1)
        valid_mask = mask & finite & positive

        mean = torch.where(valid_mask[..., None], mean, 0.0)
        logvar = torch.where(valid_mask[..., None], logvar, 0.0)
        target = torch.where(valid_mask[..., None], target, 0.0)

        log_mean = torch.log(torch.clamp(mean, min=eps))
        log_target = torch.log(torch.clamp(target, min=eps))

        var = logvar.exp()
        sq_error = (log_mean - log_target) ** 2

        nll = 0.5 * (logvar + sq_error / var)
        nll += 0.5 * torch.log(torch.tensor(2.0 * torch.pi, device=mean.device))
        nll = nll.squeeze(-1) * valid_mask

        total_nll = nll.sum()
        total_count = valid_mask.sum().clamp(min=1)
        loss = total_nll / total_count
        rmse = torch.sqrt(sq_error.squeeze(-1).sum() / total_count)

        return {"loss": loss, "rmse": rmse, "count": total_count}

    def masked_rmse_loss(
        self,
        mean: TensorType["B", "C", "T", 1],
        target: TensorType["B", "C", "T", 1],
        mask: TensorType["B", "C", "T"],
        *,
        mask_individuals: TensorType["B", "C"] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        NaN-safe RMSE.

        Any time-step where ``mean`` or ``target`` is non-finite is removed
        before applying the mask.
        """

        finite_mask = torch.isfinite(mean).squeeze(-1) & torch.isfinite(target).squeeze(-1)
        if mask_individuals is not None:
            mask = mask & mask_individuals.unsqueeze(-1)
        valid_mask = mask & finite_mask

        mean = torch.where(valid_mask.unsqueeze(-1), mean, torch.zeros_like(mean))
        target = torch.where(valid_mask.unsqueeze(-1), target, torch.zeros_like(target))

        sq_error = (mean - target).squeeze(-1).square() * valid_mask
        rmse = torch.sqrt(sq_error.sum() / valid_mask.sum().clamp(min=1))

        return {"loss": rmse, "rmse": rmse}

    def masked_mse_loss(
        self,
        mean: TensorType["B", "C", "T", 1],
        target: TensorType["B", "C", "T", 1],
        mask: TensorType["B", "C", "T"],
        *,
        mask_individuals: TensorType["B", "C"] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        NaN-safe RMSE.

        Any time-step where ``mean`` or ``target`` is non-finite is removed
        before applying the mask.
        """

        finite_mask = torch.isfinite(mean).squeeze(-1) & torch.isfinite(target).squeeze(-1)
        if mask_individuals is not None:
            mask = mask & mask_individuals.unsqueeze(-1)
        valid_mask = mask & finite_mask

        mean = torch.where(valid_mask.unsqueeze(-1), mean, torch.zeros_like(mean))
        target = torch.where(valid_mask.unsqueeze(-1), target, torch.zeros_like(target))

        sq_error = (mean - target).squeeze(-1).square() * valid_mask
        mse = sq_error.sum() / valid_mask.sum().clamp(min=1)

        return {"loss": mse, "mse": mse}

    def masked_log_rmse_loss(
        self,
        mean: TensorType["B", "C", "T", 1],
        target: TensorType["B", "C", "T", 1],
        mask: TensorType["B", "C", "T"],
        *,
        mask_individuals: TensorType["B", "C"] | None = None,
        eps: float = 1e-8,
    ) -> dict[str, torch.Tensor]:
        """RMSE on the log scale with masking."""

        finite_mask = torch.isfinite(mean).squeeze(-1) & torch.isfinite(target).squeeze(-1)
        positive = (mean.squeeze(-1) > 0) & (target.squeeze(-1) > 0)
        if mask_individuals is not None:
            mask = mask & mask_individuals.unsqueeze(-1)
        valid_mask = mask & finite_mask & positive

        log_mean = torch.log(torch.clamp(mean, min=eps))
        log_target = torch.log(torch.clamp(target, min=eps))

        sq_error = (log_mean - log_target).squeeze(-1).square() * valid_mask
        rmse = torch.sqrt(sq_error.sum() / valid_mask.sum().clamp(min=1))

        return {"loss": rmse, "rmse": rmse}

    def masked_r2_score(
        self,
        mean: TensorType["B", "C", "T", 1],
        target: TensorType["B", "C", "T", 1],
        mask: TensorType["B", "C", "T"],
        *,
        mask_individuals: TensorType["B", "C"] | None = None,
    ) -> torch.Tensor:
        """Coefficient of determination :math:`R^2` computed with masking."""

        finite_mask = torch.isfinite(mean).squeeze(-1) & torch.isfinite(target).squeeze(-1)
        if mask_individuals is not None:
            mask = mask & mask_individuals.unsqueeze(-1)
        valid_mask = mask & finite_mask

        mean = torch.where(valid_mask.unsqueeze(-1), mean, torch.zeros_like(mean))
        target = torch.where(valid_mask.unsqueeze(-1), target, torch.zeros_like(target))

        y_true = target.squeeze(-1)
        y_pred = mean.squeeze(-1)

        count = valid_mask.sum().clamp(min=1)
        y_mean = (y_true * valid_mask).sum() / count

        ss_tot = ((y_true - y_mean) ** 2 * valid_mask).sum()
        ss_res = ((y_true - y_pred) ** 2 * valid_mask).sum()

        return 1.0 - ss_res / ss_tot.clamp(min=1e-8)

    def masked_log_r2_score(
        self,
        mean: TensorType["B", "C", "T", 1],
        target: TensorType["B", "C", "T", 1],
        mask: TensorType["B", "C", "T"],
        *,
        mask_individuals: TensorType["B", "C"] | None = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Coefficient of determination :math:`R^2` on the log scale."""

        finite_mask = torch.isfinite(mean).squeeze(-1) & torch.isfinite(target).squeeze(-1)
        positive = (mean.squeeze(-1) > 0) & (target.squeeze(-1) > 0)
        if mask_individuals is not None:
            mask = mask & mask_individuals.unsqueeze(-1)
        valid_mask = mask & finite_mask & positive

        log_mean = torch.log(torch.clamp(mean, min=eps))
        log_target = torch.log(torch.clamp(target, min=eps))

        y_true = log_target.squeeze(-1)
        y_pred = log_mean.squeeze(-1)

        count = valid_mask.sum().clamp(min=1)
        y_mean = (y_true * valid_mask).sum() / count

        ss_tot = ((y_true - y_mean) ** 2 * valid_mask).sum()
        ss_res = ((y_true - y_pred) ** 2 * valid_mask).sum()

        return 1.0 - ss_res / ss_tot.clamp(min=1e-8)

    def _materialize_losses(self, outputs: Any) -> Dict[str, torch.Tensor]:
        """Extract a dictionary of losses from ``outputs`` returned by ``forward``."""

        if hasattr(outputs, "to_dict"):
            return outputs.to_dict()  # type: ignore[attr-defined]
        if isinstance(outputs, dict):
            return outputs
        if isinstance(outputs, tuple) and outputs:
            return self._materialize_losses(outputs[0])
        return {}

    def _resolve_epoch_label(self, *, explicit: str | None = None) -> str:
        """Return a human-readable label for the current epoch."""

        if explicit is not None:
            return explicit

        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            if bool(getattr(trainer, "should_stop", False)):
                return "last"

            max_epochs = getattr(trainer, "max_epochs", None)
            try:
                max_epochs_int = int(max_epochs) if max_epochs is not None else None
            except (TypeError, ValueError):
                max_epochs_int = None

            if max_epochs_int is not None and max_epochs_int > 0:
                if self.current_epoch >= max_epochs_int - 1:
                    return "last"

        return str(self.current_epoch)

    # ------------------------------------------------------------------
    # Lightning interface
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):  # type: ignore[override]
        """Execute a single optimisation step shared by all PK models."""

        try:
            outputs = self(batch)
        except AssertionError as exc:  # encoder caught NaNs or infs
            self.print(f"⚠️  {exc} at batch {batch_idx}; skipping.")
            zero = torch.zeros((), device=self.device, requires_grad=True)
            return {"loss": zero}

        losses = self._materialize_losses(outputs)
        if not losses:
            zero = torch.zeros((), device=self.device, requires_grad=True)
            return {"loss": zero}

        if not metrics_are_finite(losses):
            self.print(f"⚠️  Non-finite loss values at batch {batch_idx}; skipping.")
            zero = torch.zeros((), device=self.device, requires_grad=True)
            return {"loss": zero}

        for key, value in losses.items():
            self.log(
                f"train_{key}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=(key == "loss"),
            )

        return losses

    def validation_step(self, batch_list, batch_idx):  # type: ignore[override]
        outputs = self(batch_list)
        losses_dict = self._materialize_losses(outputs)

        for key, value in losses_dict.items():
            self.log(
                f"val_{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=(key == "loss"),
                sync_dist=True,
            )

        train_cfg = getattr(self.model_config, "train", None)
        eval_pct = getattr(train_cfg, "log_image_every_epoch_pct", None)
        total_epochs = getattr(train_cfg, "epochs", 0)

        should_plot = False
        if eval_pct is not None and eval_pct > 0:
            interval = max(1, math.ceil(total_epochs * float(eval_pct)))
            should_plot = (self.current_epoch + 1) % interval == 0 and batch_idx == 0

        if should_plot and self.meta_dosing is not None:
            if isinstance(self, PredictiveMixin) and self._should_log_predictions():
                self.log_predictions_from_batches(
                    batch_list,
                    "Synthetic",
                    self.current_epoch,
                    batch_idx,
                )

            if isinstance(self, GenerativeMixin) and self._should_log_reconstructions():
                self.log_new_individuals_from_batches(
                    batch_list,
                    "Synthetic",
                    self.current_epoch,
                    batch_idx,
                    "Synthetic",
                )

        return outputs

    def on_train_end(self) -> None:
        if getattr(self, "_last_empirical_logging_epoch", None) == self.current_epoch:
            return
        mix_cfg = getattr(self.model_config, "mix_data", None)
        fix_past_value = int(getattr(mix_cfg, "evaluate_prediction_steps_past", 4))
        self._log_empirical_evaluation(fix_past_value=fix_past_value, epoch_label="last")

    def on_train_epoch_end(self) -> None:
        train_cfg = getattr(self.model_config, "train", None)
        eval_pct = getattr(train_cfg, "log_empirical_evaluation_pct", None)
        if eval_pct is None or eval_pct <= 0:
            return
        total_epochs = getattr(train_cfg, "epochs", 0)
        mix_cfg = getattr(self.model_config, "mix_data", None)
        fix_past_value = int(getattr(mix_cfg, "evaluate_prediction_steps_past", 4))
        interval = max(1, math.ceil(total_epochs * float(eval_pct)))

        if (self.current_epoch + 1) % interval == 0:
            self._log_empirical_evaluation(fix_past_value=fix_past_value)

    def configure_optimizers(self):
        """Instantiate the optimiser used by all PK models."""
        return torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    # ------------------------------------------------------------------
    # LOG OPERATIONS
    # ------------------------------------------------------------------
    def _log_per_substance_metrics(
        self,
        outputs,
        repo_id: str,
        *,
        epoch_label: str | None = None,
        experiment=None,
    ) -> None:
        """Log per-substance metrics emitted by ``forward`` reports."""

        per_substance = getattr(outputs, "per_substance", None)
        if not per_substance:
            return

        resolved_epoch_label = self._resolve_epoch_label(explicit=epoch_label)

        for substance, metrics in per_substance.items():
            for metric_name, metric_value in metrics.items():
                metric_full_name = (
                    f"empirical/{repo_id}/{substance}/epoch_{resolved_epoch_label}/{metric_name}"
                )
                value = float(metric_value)
                if experiment is not None:
                    experiment.log_metric(name=metric_full_name, value=value)
                else:
                    self.log(
                        metric_full_name,
                        value,
                        on_step=False,
                        on_epoch=False,
                    )

    def _log_empirical_evaluation(
        self,
        *,
        label: str = "Empirical",
        fix_past_value: int | None = None,
        epoch_label: str | None = None,
    ) -> None:
        """Run empirical evaluation batches and log their outputs."""

        if not hasattr(self, "_last_empirical_logging_epoch"):
            self._last_empirical_logging_epoch = None

        trainer = getattr(self, "trainer", None)
        datamodule = getattr(trainer, "datamodule", None)
        logger = getattr(self, "logger", None)
        experiment = getattr(logger, "experiment", None)
        meta_dosing = getattr(self.model_config, "dosing", None)
        was_training = self.training
        self.eval()
        device = self.device
        datamodule.fix_past_selection(fix_past_value, who="target")
        empirical_batches = datamodule.get_empirical_test_batches()
        fix_applied = True
        logged_any = False

        try:
            with torch.inference_mode():
                for repo_id, batch_list in empirical_batches.items():
                    logged_any = True
                    batch_list_on_device = list_of_databath_to_device(batch_list, device)

                    if isinstance(self, PredictiveMixin) and self._should_log_predictions():
                        self.log_predictions_from_batches(
                            batch_list_on_device,
                            label,
                            self.current_epoch,
                            -1,
                        )

                    if isinstance(self, GenerativeMixin) and self._should_log_reconstructions():
                        self.log_new_individuals_from_batches(
                            batch_list_on_device,
                            label,
                            self.current_epoch,
                            -1,
                            repo_id,
                        )

        finally:
            if was_training:
                self.train()

            if fix_applied and hasattr(datamodule, "release_past_selection"):
                datamodule.release_past_selection(who="target")

        if logged_any:
            self._last_empirical_logging_epoch = self.current_epoch

    # ------------------------------------------------------------------
    # UTILS
    # ------------------------------------------------------------------
    def _delete_images(self) -> None:
        for path in getattr(self, "_generated_images", []):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:  # pragma: no cover - best effort cleanup
                    self.print(f"⚠️ Could not delete image {path}: {e}")

    def get_last_valid_observation(
        self,
        X_c: TensorType["B", "I", "T", 1],
        M_c: TensorType["B", "I", "T"],
        T_c: TensorType["B", "I", "T", 1] | None = None,
    ) -> (
        Tuple[TensorType["B", "I", 1, 1]]
        | Tuple[TensorType["B", "I", 1, 1], TensorType["B", "I", 1, 1]]
    ):
        """Extract the last valid observation (and time) for every individual."""

        B, I, T, _ = X_c.shape
        device = X_c.device

        time_idx = torch.arange(T, device=device).view(1, 1, T).expand(B, I, T)
        fallback = T - 1
        valid_time_idx = torch.where(M_c, time_idx, torch.full_like(time_idx, fallback))
        last_valid_idx = valid_time_idx.max(dim=2).values

        batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, I)
        indiv_idx = torch.arange(I, device=device).view(1, I).expand(B, I)

        last_obs = X_c[batch_idx, indiv_idx, last_valid_idx, :]
        if T_c is not None:
            last_t = T_c[batch_idx, indiv_idx, last_valid_idx, :]
            return last_obs.unsqueeze(2), last_t.unsqueeze(2)

        return last_obs.unsqueeze(2)

    def _call_forward_for_logging(self, batch_on_device):
        """Execute ``forward`` with optional ``return_forward_report`` flag."""

        try:
            signature = inspect.signature(self.forward)
        except (TypeError, ValueError):
            signature = None

        if signature is not None and "return_forward_report" in signature.parameters:
            try:
                return self(batch_on_device, return_forward_report=True)
            except TypeError:
                pass
        return self(batch_on_device)

    @staticmethod
    def get_first_valid_observation(
        X: TensorType["B", "I", "T", 1],
        M: TensorType["B", "I", "T"],
        T_tensor: TensorType["B", "I", "T", 1] | None = None,
    ) -> (
        Tuple[TensorType["B", "I", 1, 1], TensorType["B", "I", 1]]
        | Tuple[TensorType["B", "I", 1, 1], TensorType["B", "I", 1], TensorType["B", "I", 1, 1]]
    ):
        """Return the first valid observation, its mask, and optionally its time.

        Works for both context and target blocks. Used by ContextVAEPK and AICMEPK.
        """
        B, I, T, _ = X.shape
        device = X.device
        time_idx = torch.arange(T, device=device).view(1, 1, T).expand(B, I, T)
        fallback = T - 1
        valid_time_idx = torch.where(M, time_idx, torch.full_like(time_idx, fallback))
        first_valid_idx = valid_time_idx.min(dim=2).values  # [B, I]

        batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, I)
        indiv_idx = torch.arange(I, device=device).view(1, I).expand(B, I)

        first_obs = X[batch_idx, indiv_idx, first_valid_idx, :]  # [B, I, 1]
        first_mask = M[batch_idx, indiv_idx, first_valid_idx]  # [B, I]

        if T_tensor is not None:
            first_t = T_tensor[batch_idx, indiv_idx, first_valid_idx, :]
            return first_obs.unsqueeze(2), first_mask.unsqueeze(2), first_t.unsqueeze(2)

        return first_obs.unsqueeze(2), first_mask.unsqueeze(2)


__all__ = ["BasePKModel", "PredictiveMixin", "GenerativeMixin"]
