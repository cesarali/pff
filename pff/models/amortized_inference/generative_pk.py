"""Logging-free base utilities and mixins for PK models.

This module mirrors the legacy PK base/mixin utilities while removing all
logging and evaluation orchestration. Models remain responsible for forward
logic and loss computation only; callbacks own visualization and empirical
reporting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import lightning.pytorch as pl
import torch
from huggingface_hub import PyTorchModelHubMixin
from torchtyping import TensorType

from pff.config_classes.flow_pk_config import FlowPKExperimentConfig, HFFlowPKConfig
from pff.data.data_empirical.builder import (
    EmpiricalBatchConfig,
    JSON2AICMEBuilder,
    prediction_to_study_jsons,
)
from pff.data.data_empirical.json_schema import (
    IndividualJSON,
    StudyJSON,
    canonicalize_study,
    studies_from_sampled_targets,
)
from pff.data.datasets.aicme_datasets import AICMECompartmentsDataBatch

from pff.models.architectures import get_decoder
from pff.models.architectures.encoders_pk import get_individual_encoder
from pff.models.utils.scaler_selection import resolve_scaler_methods
from pff.models.utils.scalers import PKScaler
from pff.training.utils import metrics_are_finite


def _has_predictive_sampling(model: object) -> bool:
    """Return ``True`` when ``model`` exposes predictive StudyJSON sampling."""

    return callable(getattr(model, "sample_individual_prediction", None))


def _has_generative_sampling(model: object) -> bool:
    """Return ``True`` when ``model`` exposes generative StudyJSON sampling."""

    return callable(getattr(model, "sample_new_individual", None))


def _scheduler_label_for_split(split: str) -> str:
    """Return the human-readable scheduler label for a data split."""

    return "Empirical" if str(split).strip().lower().startswith("empirical") else "Synthetic"


def _expand_empirical_scheduler_tasks(
    raw_scheduler: Dict[str, object],
    *,
    empirical_datasets: Sequence[str],
    model_label: str,
) -> Dict[str, object]:
    """Normalize empirical scheduler task templates before callback construction.

    The empirical dataset collection remains part of the API because callback
    configurations provide it, although task-internal sampling now resolves
    repositories at execution time.
    """

    del empirical_datasets

    expanded = deepcopy(raw_scheduler)
    for section_name in ("tasks_validation", "task_during", "tasks_end"):
        raw_tasks = list(expanded.get(section_name, []) or [])
        expanded_tasks: list[Dict[str, object]] = []
        for raw_task in raw_tasks:
            if not isinstance(raw_task, dict):
                expanded_tasks.append(raw_task)
                continue

            task = deepcopy(raw_task)
            task_cfg = dict(task.get("task_cfg", {}) or {})
            task_cfg.setdefault("label", _scheduler_label_for_split(str(task.get("split", "val"))))
            task_cfg.setdefault("model_label", model_label)
            task["task_cfg"] = task_cfg

            sample_source = str(task.get("sample_source", "")).strip().lower()
            if sample_source != "empirical_set":
                expanded_tasks.append(task)
                continue

            fn_key = str(task.get("fn_key", "")).strip()
            if fn_key not in {
                "pk.empirical.predictive.metrics",
                "pk.empirical.heldout_generated_classifier",
            }:
                expanded_tasks.append(task)
                continue

            empirical_name = task.get("empirical_name")
            repo_id = str(empirical_name).strip() if empirical_name else None
            task_cfg["split"] = str(task.get("split", "empirical_heldout"))
            if fn_key == "pk.empirical.predictive.metrics":
                task_cfg.setdefault("sample_size", int(task.get("n_samples", 0)))
            if repo_id:
                task_cfg["empirical_name"] = repo_id
                task_cfg["repo_id"] = repo_id
            task["sample_source"] = "task_internal"
            task["n_samples"] = 0
            task["empirical_name"] = repo_id
            expanded_tasks.append(task)

        expanded[section_name] = expanded_tasks
    return expanded


def _resolve_model_device(model: object) -> torch.device:
    """Return the most appropriate device for auxiliary StudyJSON batches."""

    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    if isinstance(model, torch.nn.Module):
        parameter = next(model.parameters(), None)
        if parameter is not None:
            return parameter.device
    return torch.device("cpu")


def _normalize_study_json_input(
    studies: StudyJSON | Sequence[StudyJSON],
) -> tuple[list[StudyJSON], bool]:
    """Normalize single-or-sequence StudyJSON input into a mutable list."""

    if isinstance(studies, dict):
        return [deepcopy(studies)], True
    return [deepcopy(study) for study in studies], False


def _empirical_batch_config_for_study(study: StudyJSON) -> EmpiricalBatchConfig:
    """Build an exact-fit empirical batch config for one StudyJSON record."""

    context = list(study.get("context", []))
    target = list(study.get("target", []))

    def _max_len(inds: Sequence[IndividualJSON], key: str) -> int:
        return max((len(ind.get(key, [])) for ind in inds), default=0)

    ctx_obs = _max_len(context, "observations")
    tgt_obs = _max_len(target, "observations")
    ctx_rem = _max_len(context, "remaining_times")
    tgt_rem = _max_len(target, "remaining_times")

    return EmpiricalBatchConfig(
        max_databatch_size=1,
        max_individuals=max(1, len(context), len(target)),
        max_observations=max(ctx_obs, tgt_obs),
        max_remaining=max(ctx_rem, tgt_rem),
        max_context_individuals=max(1, len(context)),
        max_target_individuals=max(1, len(target)),
        max_context_observations=ctx_obs,
        max_target_observations=tgt_obs,
        max_context_remaining=ctx_rem,
        max_target_remaining=tgt_rem,
    )


def _build_single_study_batch(
    *,
    study: StudyJSON,
    meta_dosing: object,
    device: torch.device,
) -> AICMECompartmentsDataBatch:
    """Convert one StudyJSON record into a one-study empirical batch."""

    builder = JSON2AICMEBuilder(_empirical_batch_config_for_study(study))
    return builder.build_study_batch(study, meta_dosing).to(device)


def _require_target_dosing(individual: IndividualJSON, *, study_idx: int, target_idx: int) -> None:
    """Require explicit target dosing fields for mixed StudyJSON sampling."""

    dosing_keys = ("dosing", "dosing_type", "dosing_times", "dosing_name")
    missing = [key for key in dosing_keys if not individual.get(key)]
    if missing:
        raise ValueError(
            f"Study {study_idx} target[{target_idx}] must define dosing, dosing_type, "
            f"dosing_times, and dosing_name. Missing: {missing}."
        )


def _resolve_target_sampling_mode(
    individual: IndividualJSON,
    *,
    study_idx: int,
    target_idx: int,
) -> str:
    """Resolve whether a target individual requires predictive or generative sampling."""

    observations = list(individual.get("observations", []))
    observation_times = list(individual.get("observation_times", []))
    remaining_times = list(individual.get("remaining_times", []))

    obs_present = len(observations) > 0
    time_present = len(observation_times) > 0
    if obs_present != time_present:
        raise ValueError(
            f"Study {study_idx} target[{target_idx}] must provide observations and "
            "observation_times together, or leave both empty."
        )
    if not remaining_times:
        raise ValueError(
            f"Study {study_idx} target[{target_idx}] must provide non-empty remaining_times."
        )
    return "predictive" if obs_present else "generative"


def _normalize_generated_target_samples(
    samples: torch.Tensor,
) -> TensorType["S", "B", "T", 1]:
    """Normalize generated target samples to ``[S, B, T, 1]``."""

    if samples.ndim == 4:
        normalized = samples
    elif samples.ndim == 5 and samples.shape[2] == 1:
        normalized = samples.squeeze(2)
    else:
        raise ValueError(
            "Unsupported generated sample layout for StudyJSON sampling: expected "
            f"[S,B,T,1] or [S,B,1,T,1], got {tuple(samples.shape)}."
        )

    if normalized.ndim != 4 or normalized.shape[-1] != 1:
        raise ValueError(
            "Normalized generated samples must have shape [S, B, T, 1], "
            f"got {tuple(normalized.shape)}."
        )
    return normalized


def _sample_from_study_json_impl(
    model: object,
    studies: StudyJSON | Sequence[StudyJSON],
    *,
    sample_size: int = 1,
    num_steps: int | None = None,
) -> StudyJSON | list[StudyJSON]:
    """Shared implementation behind mixin-level ``sample_from_study_json`` methods."""

    if int(sample_size) < 1:
        raise ValueError("sample_size must be >= 1.")

    meta_dosing = getattr(model, "meta_dosing", None)
    if meta_dosing is None:
        raise AttributeError("model_config must define a `dosing` section for StudyJSON sampling.")

    input_studies, is_single_input = _normalize_study_json_input(studies)
    output_studies = deepcopy(input_studies)
    canonical_studies = [
        canonicalize_study(deepcopy(study), drop_tgt_too_few=False) for study in input_studies
    ]

    supports_predictive = _has_predictive_sampling(model)
    supports_generative = _has_generative_sampling(model)
    device = _resolve_model_device(model)

    for study_idx, canonical_study in enumerate(canonical_studies):
        context = list(canonical_study.get("context", []))
        targets = list(canonical_study.get("target", []))
        if not context:
            raise ValueError("sample_from_study_json requires at least one context individual.")
        if not targets:
            raise ValueError("sample_from_study_json requires at least one target individual.")

        for target_idx, target_individual in enumerate(targets):
            mode = _resolve_target_sampling_mode(
                target_individual,
                study_idx=study_idx,
                target_idx=target_idx,
            )
            _require_target_dosing(target_individual, study_idx=study_idx, target_idx=target_idx)

            if mode == "predictive" and not supports_predictive:
                raise ValueError(
                    "Predictive target individuals require a model implementing "
                    "`sample_individual_prediction`."
                )
            if mode == "generative" and not supports_generative:
                raise ValueError(
                    "Generative target individuals require a model implementing "
                    "`sample_new_individual`."
                )

            one_target_study: StudyJSON = {
                "context": deepcopy(context),
                "target": [deepcopy(target_individual)],
                "meta_data": dict(canonical_study.get("meta_data", {})),
            }
            batch = _build_single_study_batch(
                study=one_target_study,
                meta_dosing=meta_dosing,
                device=device,
            )

            if mode == "predictive":
                prediction_sample, _, _, _ = model.sample_individual_prediction(  # type: ignore[attr-defined]
                    batch,
                    sample_size=int(sample_size),
                )
                output_studies[study_idx]["target"][target_idx]["prediction_samples"] = (
                    prediction_sample[:, 0, 0, :, 0].tolist()
                )
                output_studies[study_idx]["target"][target_idx]["prediction_times"] = list(
                    target_individual.get("remaining_times", [])
                )
                continue

            decode_times = (
                batch.target_rem_sim_time[:, 0, :, :],
                batch.target_rem_sim_mask[:, 0, :].bool(),
            )
            dosing = (
                batch.target_dosing_amounts[:, 0],
                batch.target_dosing_route_types[:, 0],
            )
            generated_samples, _, generated_mask = model.sample_new_individual(  # type: ignore[attr-defined]
                batch,
                sample_size=int(sample_size),
                decode_times=decode_times,
                num_steps=num_steps,
                dosing=dosing,
            )
            generated_samples = _normalize_generated_target_samples(generated_samples)
            valid_generated_mask = generated_mask[0].bool()
            output_studies[study_idx]["target"][target_idx]["prediction_samples"] = (
                generated_samples[:, 0, valid_generated_mask, 0].tolist()
            )
            output_studies[study_idx]["target"][target_idx]["prediction_times"] = list(
                target_individual.get("remaining_times", [])
            )

    if is_single_input:
        return output_studies[0]
    return output_studies


class AbstractForwardOutputs:
    """Declarative forward-output container shared across PK models.

    The container acts as the glue between the per-permutation forward passes
    executed by :class:`BasePKModel` subclasses and the permutation-level
    aggregation that happens once all permutations for a study have been
    processed.  ``Permutation`` here refers to one particular context/target
    split of the same underlying study.  Data modules typically emit several of
    these permutations during training so that the model observes different
    combinations of context and target individuals for the same study.  Each
    forward call produces an :class:`AbstractForwardOutputs` (or subclass)
    instance representing a single permutation.  Calling :meth:`add` registers
    that instance for aggregation and :meth:`reduce` combines the registered
    items into a single object by averaging heads and losses across
    permutations.

    Subclasses declare which tensors they need to keep track of via the
    class-level :attr:`HEAD_SCHEMAS` and :attr:`LOSS_SCHEMAS` dictionaries.  The
    keys of those dictionaries are *scopes* (for example ``"reconstruction"`` or
    ``"prediction"``) and the values are the tensor names expected inside each
    scope.  During a permutation forward pass the model uses :meth:`update_head`
    and :meth:`update_losses` to populate those scopes.  When
    :meth:`aggregate_losses` or :meth:`aggregate_heads` run they only average the
    keys that were explicitly marked as active, thereby supporting models that
    enable or disable particular losses on the fly.

    Subclasses **must** define :attr:`HEAD_SCHEMAS` and :attr:`LOSS_SCHEMAS`.
    Providing additional attributes (for example cached scalers or latent
    samples) is optional and can be implemented in ``__init__``.  Methods such as
    :meth:`initialize_heads`, :meth:`initialize_losses`, :meth:`aggregate_heads`
    and :meth:`aggregate_losses` can be overridden when bespoke behaviour is
    required, but the default implementation already covers the most common
    cases.  ``AICMEForwardOutputs`` for instance tracks two head scopes
    (``reconstruction`` and ``prediction``) and augments ``reduce`` to attach
    invariance statistics, whereas :class:`ContextVAEForwardOutputs` only needs a
    single ``reconstruction`` scope that stores the tensors
    ``["mean", "logvar", "target", "mask"]``.

    The aggregation contract can be summarised as::

        outputs = ContextVAEForwardOutputs()
        for permutation in permutations:
            outputs.add(model_forward(permutation))
        aggregated = outputs.reduce()  # averages heads/losses across permutations

    ``aggregated`` now contains mean heads and scalar losses averaged over the
    registered permutations.  Its :attr:`total_loss` is computed automatically by
    summing the flattened losses unless the subclass provided a custom
    ``loss_multihead`` callable.

    The helper :meth:`prepare_predictions` extracts prediction dictionaries for
    every stored permutation.  This replaces ad-hoc list comprehensions in model
    ``forward`` methods and ensures that permutation order is preserved.  The
    returned dictionaries match the schema declared in ``HEAD_SCHEMAS``; for
    example ``AICMEForwardOutputs`` yields ``[{"mean": ..., "mask": ...}, ...]``
    for each permutation with a prediction head.  The same method is used by
    :class:`AICMEPK` and :class:`ContextVAEPK` when compiling validation reports.

    ``update_head`` and ``update_losses`` simply record tensors for the active
    permutation and mark which keys were set.  During :meth:`aggregate_losses`
    those keys are averaged across permutations and the resulting scalars are fed
    into :meth:`reduce`, which in turn computes :attr:`total_loss`.  Subclasses
    may add convenience properties—``AICMEForwardOutputs`` keeps a
    ``per_substance`` dictionary derived from :meth:`forward_report`—but the
    permutation aggregation semantics are entirely governed by the trio
    ``add`` → ``aggregate`` → ``reduce`` documented above.
    """

    #: Mapping from head scope to the tensor keys stored within it.
    HEAD_SCHEMAS: Dict[str, List[str]] = {}
    #: Mapping from loss scope to the scalar loss keys stored within it.
    LOSS_SCHEMAS: Dict[str, List[str]] = {}

    def __init__(
        self,
        *,
        loss_multihead: Optional[Callable[[List[torch.Tensor]], Tuple[torch.Tensor, Any]]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.loss_multihead = loss_multihead
        self.device = device
        resolved_device = device if device is not None else torch.device("cpu")
        self.heads: Dict[str, Dict[str, torch.Tensor]] = self.initialize_heads(resolved_device)
        self.losses: Dict[str, Dict[str, torch.Tensor]] = self.initialize_losses(resolved_device)
        self.total_loss: Optional[torch.Tensor] = None
        self.items: List["AbstractForwardOutputs"] = []
        self._active_heads: Dict[str, Set[str]] = {scope: set() for scope in self.HEAD_SCHEMAS}
        self._active_losses: Dict[str, Set[str]] = {scope: set() for scope in self.LOSS_SCHEMAS}
        self.prediction_permutations: List[Dict[str, torch.Tensor]] = []

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def initialize_heads(self, device: torch.device) -> Dict[str, Dict[str, torch.Tensor]]:
        """Initialise declared heads with zero tensors on ``device``."""

        heads: Dict[str, Dict[str, torch.Tensor]] = {}
        for scope, keys in self.HEAD_SCHEMAS.items():
            heads[scope] = {key: torch.zeros((), device=device) for key in keys}
        return heads

    def initialize_losses(self, device: torch.device) -> Dict[str, Dict[str, torch.Tensor]]:
        """Initialise declared losses with zero scalars on ``device``."""

        losses: Dict[str, Dict[str, torch.Tensor]] = {}
        for scope, keys in self.LOSS_SCHEMAS.items():
            losses[scope] = {key: torch.zeros((), device=device) for key in keys}
        return losses

    # ------------------------------------------------------------------
    # Update helpers
    # ------------------------------------------------------------------
    def update_head(self, scope: str, values: Dict[str, torch.Tensor]) -> None:
        """Update a head scope with concrete tensors."""

        if scope not in self.heads:
            raise KeyError(f"Unknown head scope '{scope}'.")
        for key, tensor in values.items():
            if key not in self.heads[scope]:
                raise KeyError(f"Unknown head key '{scope}.{key}'.")
            self.heads[scope][key] = tensor
            self._active_heads.setdefault(scope, set()).add(key)

    def update_losses(self, scope: str, values: Dict[str, torch.Tensor]) -> None:
        """Update a loss scope with concrete scalars."""

        if scope not in self.losses:
            raise KeyError(f"Unknown loss scope '{scope}'.")
        for key, tensor in values.items():
            if key not in self.losses[scope]:
                raise KeyError(f"Unknown loss key '{scope}.{key}'.")
            self.losses[scope][key] = tensor
            self._active_losses.setdefault(scope, set()).add(key)

    def add(self, output: "AbstractForwardOutputs") -> None:
        """Store permutation-specific outputs for later aggregation."""

        self.items.append(output)

    # ------------------------------------------------------------------
    # Aggregation utilities
    # ------------------------------------------------------------------
    def prepare_predictions(self) -> List[Dict[str, torch.Tensor]]:
        """Collect prediction heads for all stored permutations.

        The method inspects the ``"prediction"`` scope for each registered item
        (or the current instance when already reduced) and returns the tensors as
        dictionaries.  It mirrors the structure previously assembled manually in
        :class:`AICMEPK.forward` and provides a uniform representation for report
        generation.  Only permutations that expose the full prediction schema are
        returned, ensuring callers no longer need to filter lists manually.
        """

        predictions: List[Dict[str, torch.Tensor]] = []
        required_keys: Set[str] = set(self.HEAD_SCHEMAS.get("prediction", ()))
        if not required_keys:
            # Backwards compatibility: fall back to the common trio of tensors if
            # the schema does not explicitly list the prediction keys.
            required_keys = {"mean", "target", "mask"}

        def _extract(item: "AbstractForwardOutputs") -> Optional[Dict[str, torch.Tensor]]:
            if "prediction" not in item.heads:
                return None
            active_keys = item._active_heads.get("prediction", set())
            if required_keys and not required_keys.issubset(active_keys):
                return None
            scope = item.heads["prediction"]
            return {key: scope[key] for key in required_keys if key in scope}

        if self.items:
            for item in self.items:
                extracted = _extract(item)
                if extracted is not None:
                    predictions.append(extracted)
        elif getattr(self, "prediction_permutations", None):
            predictions.extend(
                {key: pred[key] for key in required_keys if key in pred}
                for pred in self.prediction_permutations  # type: ignore[attr-defined]
                if pred and required_keys.issubset(pred.keys())
            )
        else:
            extracted = _extract(self)
            if extracted is not None:
                predictions.append(extracted)

        return predictions

    def _infer_device(self) -> torch.device:
        if self.device is not None:
            return self.device

        # Prefer concrete tensors from aggregated child outputs before falling
        # back to placeholder zero scalars created during container
        # initialisation. Those placeholders live on CPU by default and can
        # incorrectly force DDP-logged metrics onto CPU.
        for item in self.items:
            try:
                return item._infer_device()
            except RuntimeError:
                continue

        for scope, active_keys in self._active_heads.items():
            scope_values = self.heads.get(scope, {})
            for key in active_keys:
                tensor = scope_values.get(key)
                if tensor is not None and tensor.numel() > 0:
                    return tensor.device

        for scope, active_keys in self._active_losses.items():
            scope_values = self.losses.get(scope, {})
            for key in active_keys:
                tensor = scope_values.get(key)
                if tensor is not None and tensor.numel() > 0:
                    return tensor.device

        if self.total_loss is not None:
            return self.total_loss.device

        for scopes in (self.heads, self.losses):
            for scope_values in scopes.values():
                for tensor in scope_values.values():
                    if tensor is not None and tensor.numel() > 0:
                        return tensor.device
        raise RuntimeError("Unable to infer device for forward outputs")

    @staticmethod
    def _aggregate_dict(dicts: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Element-wise mean aggregation over a sequence of dictionaries."""

        aggregated: Dict[str, torch.Tensor] = {}
        keys = {key for scope_dict in dicts for key in scope_dict}
        for key in keys:
            tensors = [scope_dict[key] for scope_dict in dicts if key in scope_dict]
            if not tensors:
                continue
            stack = torch.stack(tensors, dim=0)
            if stack.dtype.is_floating_point or stack.dtype.is_complex:
                aggregated[key] = stack.mean(dim=0)
            elif stack.dtype == torch.bool:
                aggregated[key] = stack.float().mean(dim=0) >= 0.5
            else:
                aggregated[key] = stack.float().mean(dim=0).round().to(stack.dtype)
        return aggregated

    def aggregate_heads(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Aggregate declared heads across stored permutation outputs."""

        aggregated: Dict[str, Dict[str, torch.Tensor]] = {}
        for scope in self.HEAD_SCHEMAS:
            scope_dicts: List[Dict[str, torch.Tensor]] = []
            for item in self.items:
                active_keys = item._active_heads.get(scope, set())
                if not active_keys:
                    continue
                scope_values = item.heads.get(scope, {})
                scope_dicts.append({key: scope_values[key] for key in active_keys})
            if scope_dicts:
                aggregated[scope] = self._aggregate_dict(scope_dicts)
        return aggregated

    def aggregate_losses(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Aggregate declared losses across stored permutation outputs."""

        aggregated: Dict[str, Dict[str, torch.Tensor]] = {}
        for scope in self.LOSS_SCHEMAS:
            scope_dicts: List[Dict[str, torch.Tensor]] = []
            for item in self.items:
                active_keys = item._active_losses.get(scope, set())
                if not active_keys:
                    continue
                scope_values = item.losses.get(scope, {})
                scope_dicts.append({key: scope_values[key] for key in active_keys})
            if scope_dicts:
                aggregated[scope] = self._aggregate_dict(scope_dicts)
        return aggregated

    # ------------------------------------------------------------------
    # Reduction helpers
    # ------------------------------------------------------------------
    def _compute_total_loss(self, flat_losses: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if not flat_losses:
            return None
        if self.loss_multihead is None:
            return torch.stack(list(flat_losses.values())).sum()
        values = list(flat_losses.values())
        total, _ = self.loss_multihead(values)
        return total

    def _flatten_losses(self, include_total: bool = True) -> Dict[str, torch.Tensor]:
        flat: Dict[str, torch.Tensor] = {}
        for scope_values in self.losses.values():
            flat.update(scope_values)
        if include_total and self.total_loss is not None:
            flat["loss"] = self.total_loss
        return flat

    def _new_like(self) -> "AbstractForwardOutputs":
        return type(self)(loss_multihead=self.loss_multihead, device=self.device)

    def reduce(self) -> "AbstractForwardOutputs":
        """Aggregate stored permutation outputs into a single instance."""

        if not self.items:
            if self.total_loss is None:
                flat_losses = self._flatten_losses(include_total=False)
                self.total_loss = self._compute_total_loss(flat_losses)
            return self

        prediction_cache = self.prepare_predictions()
        reduced = self._new_like()
        reduced.device = self._infer_device()
        reduced.heads = self.aggregate_heads()
        reduced.losses = self.aggregate_losses()
        flat_losses = reduced._flatten_losses(include_total=False)
        reduced.total_loss = reduced._compute_total_loss(flat_losses)
        reduced.items = []
        reduced._active_heads = {
            scope: set(values.keys()) for scope, values in reduced.heads.items()
        }
        reduced._active_losses = {
            scope: set(values.keys()) for scope, values in reduced.losses.items()
        }
        reduced.prediction_permutations = prediction_cache
        return reduced

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Flatten all tracked losses into a logging-friendly dictionary."""

        flat = self._flatten_losses(include_total=False)
        if self.total_loss is not None:
            flat["loss"] = self.total_loss
        return flat


class NewPredictiveMixin(ABC):
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

        if getattr(self, "meta_dosing", None) is None:
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

    @torch.inference_mode()
    def sample_from_study_json(
        self,
        studies: StudyJSON | Sequence[StudyJSON],
        sample_size: int = 1,
        num_steps: int | None = None,
    ) -> StudyJSON | list[StudyJSON]:
        """Sample StudyJSON targets, dispatching predictive/generative modes per target.

        Predictive targets have non-empty ``observations`` and
        ``observation_times``. Generative targets leave both lists empty and
        request a decode schedule through ``remaining_times``. In both cases the
        sampled trajectories are written back into ``prediction_samples`` and
        ``prediction_times`` on the original target records.

        Example target blocks accepted by this helper::

            {"observations": [0.2, 0.4], "observation_times": [0.5, 1.0], "remaining_times": [2.0, 4.0]}
            {"observations": [], "observation_times": [], "remaining_times": [1.5, 3.0]}
        """

        return _sample_from_study_json_impl(
            self,
            studies,
            sample_size=sample_size,
            num_steps=num_steps,
        )


class NewGenerativeMixin(ABC):
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
        num_steps: int | None = None,
        dosing: Tuple[torch.Tensor, torch.Tensor] | None = None,
        resolve_sampling_from_target: bool = False,
        include_rem: bool = False,
    ) -> Tuple[
        TensorType["S", "B", "Tdistinct_max", 1],
        TensorType["B", "Tdistinct_max", 1],
        TensorType["B", "Tdistinct_max"],
    ]:
        """Sample trajectories for new individuals conditioned on context.

        When ``resolve_sampling_from_target`` is enabled and ``decode_times`` is
        not provided, ``include_rem=True`` extends the decode grid with
        ``target_rem_sim_time`` in addition to ``target_obs_time``.
        """

    def _resolve_sampling_num_steps(self, num_steps: int | None) -> int:
        """Resolve ODE integration steps for generative sampling wrappers.

        Preference order:
        1. Explicit ``num_steps`` argument.
        2. Model attribute ``flow_num_steps``.
        3. Hard default ``50``.
        """
        resolved_steps = num_steps
        if resolved_steps is None:
            resolved_steps = int(getattr(self, "flow_num_steps", 50))
        if resolved_steps <= 0:
            raise ValueError(f"`num_steps` must be a positive integer, got {resolved_steps}.")
        return resolved_steps

    @staticmethod
    def _normalize_vpc_samples(
        samples: torch.Tensor,
        *,
        context_index: int,
        context_size: int,
    ) -> TensorType["S", "B", "T", 1]:
        """Normalize model outputs to ``[S, B, T, 1]`` for one context index."""

        # Accepted layouts:
        #   [S, B, T, 1]
        #   [S, B, 1, T, 1] -> squeeze singleton individual axis
        #   [S, B, I, T, 1] -> select individual at ``context_index``
        if samples.ndim == 4:
            normalized = samples
        elif samples.ndim == 5:
            if samples.shape[2] == 1:
                normalized = samples.squeeze(2)
            elif samples.shape[2] == context_size:
                normalized = samples[:, :, context_index, :, :]
            else:
                raise ValueError(
                    "Unsupported sample layout for VPC conversion: expected "
                    "[S,B,1,T,1] or [S,B,I,T,1] with I==c_ind, got "
                    f"{tuple(samples.shape)}."
                )
        else:
            raise ValueError(
                "Unsupported sample layout for VPC conversion: expected 4D or 5D tensor, "
                f"got {samples.ndim}D with shape {tuple(samples.shape)}."
            )

        if normalized.ndim != 4 or normalized.shape[-1] != 1:
            raise ValueError(
                "Normalized VPC samples must have shape [S, B, T, 1], "
                f"got {tuple(normalized.shape)}."
            )
        return normalized

    @staticmethod
    def _normalize_vpc_decode_output(
        times: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[
        TensorType["B", "T", 1],
        TensorType["B", "T"],
    ]:
        """Normalize decode outputs to ``times:[B,T,1]`` and ``mask:[B,T]``."""

        # Accept [B, T, 1] or [B, 1, T, 1] for times.
        if times.ndim == 4 and times.shape[1] == 1:
            times_out = times.squeeze(1)
        elif times.ndim == 3:
            times_out = times
        else:
            raise ValueError(
                "Unsupported decode times layout for VPC conversion: expected [B,T,1] "
                f"or [B,1,T,1], got {tuple(times.shape)}."
            )

        # Accept [B, T] or [B, 1, T] for mask.
        if mask.ndim == 3 and mask.shape[1] == 1:
            mask_out = mask.squeeze(1)
        elif mask.ndim == 2:
            mask_out = mask
        else:
            raise ValueError(
                "Unsupported decode mask layout for VPC conversion: expected [B,T] "
                f"or [B,1,T], got {tuple(mask.shape)}."
            )

        if times_out.shape[0] != mask_out.shape[0] or times_out.shape[1] != mask_out.shape[1]:
            raise ValueError(
                "Decode times and mask must agree on [B, T] dimensions, got "
                f"times={tuple(times_out.shape)} mask={tuple(mask_out.shape)}."
            )

        return times_out, mask_out.bool()

    @torch.inference_mode()
    def sample_new_individuals_to_vpc_format(
        self,
        db: AICMECompartmentsDataBatch,
        sample_size: int = 8,
        num_steps: int | None = None,
    ) -> List[List[StudyJSON]]:
        """Generate empirical-VPC studies as a nested ``[B][S]`` list.

        The method reuses the empirical context design without creating new time
        grids. For each context individual index ``i`` it extracts
        ``context_obs_time[:, i, :, :]`` and ``context_obs_mask[:, i, :]``,
        and forwards those raw schedules directly to
        :meth:`sample_new_individual` via ``decode_times=(times, mask)``.

        Returned samples are normalized to ``[S, B, T, 1]`` (supporting model
        outputs ``[S,B,T,1]``, ``[S,B,1,T,1]`` and ``[S,B,I,T,1]``). The final
        structure is ``List[List[StudyJSON]]`` with conceptual shape ``[B][S]``:
        one list per batch element (substance), and one ``StudyJSON`` per Monte
        Carlo sample.
        """

        if getattr(self, "meta_dosing", None) is None:
            raise AttributeError(
                "model_config must define a `dosing` section for VPC StudyJSON conversion."
            )

        B, c_ind, _, _ = db.context_obs_time.shape  # [B, c_ind, Tc, 1]
        route_options = list(self.meta_dosing.route_options)
        dosing_time = float(self.meta_dosing.time)

        studies_per_substance: List[List[StudyJSON]] = []
        resolved_num_steps = self._resolve_sampling_num_steps(num_steps)
        for b in range(B):
            study_name = (
                db.study_name[b] if b < len(db.study_name) and db.study_name[b] else f"study_{b}"
            )
            substance_name = (
                db.substance_name[b]
                if b < len(db.substance_name) and db.substance_name[b]
                else f"substance_{b}"
            )
            studies_for_b: List[StudyJSON] = []
            for _ in range(sample_size):
                studies_for_b.append(
                    {
                        "context": [],
                        "target": [],
                        "meta_data": {
                            "study_name": study_name,
                            "substance_name": substance_name,
                        },
                    }
                )
            studies_per_substance.append(studies_for_b)

        for i in range(c_ind):
            # context_obs_time/context_obs_mask slices for one individual index:
            #   raw_times_i: [B, Tc, 1]
            #   raw_mask_i : [B, Tc]
            raw_times_i = db.context_obs_time[:, i, :, :]
            raw_mask_i = db.context_obs_mask[:, i, :].bool()

            # samples_i can be [S,B,T,1] or equivalent with explicit individual axis.
            samples_i, times_i, mask_i = self.sample_new_individual(
                db,
                sample_size=sample_size,
                decode_times=(raw_times_i, raw_mask_i),
                num_steps=resolved_num_steps,
            )

            samples_i = self._normalize_vpc_samples(
                samples_i,
                context_index=i,
                context_size=c_ind,
            )  # [S, B, T, 1]
            times_out, mask_out = self._normalize_vpc_decode_output(
                times_i, mask_i
            )  # [B, T, 1], [B, T]

            if samples_i.shape[0] != sample_size:
                raise ValueError(
                    "sample_new_individual returned an unexpected number of samples: "
                    f"expected {sample_size}, got {samples_i.shape[0]}."
                )

            # Prefer the original empirical decode schedule to avoid float drift in
            # model-returned times. VPC/NPDE validation matches on exact
            # (Type, ID, Time) keys, so even tiny time jitter can suppress plots.
            # raw_times_i/raw_mask_i: [B, Tc, 1], [B, Tc]
            # times_for_vpc/mask_for_vpc: [B, T, 1], [B, T]
            if (
                raw_times_i.shape[1] == samples_i.shape[2]
                and raw_mask_i.shape[1] == samples_i.shape[2]
            ):
                times_for_vpc = raw_times_i
                mask_for_vpc = raw_mask_i
            else:
                times_for_vpc = times_out
                mask_for_vpc = mask_out

            for b in range(B):
                if not bool(db.mask_context_individuals[b, i]):
                    continue

                valid_time_mask = mask_for_vpc[b]  # [T]
                observation_times = times_for_vpc[b, valid_time_mask, 0].tolist()

                route_idx = int(db.context_dosing_route_types[b, i].item())
                route_label = (
                    route_options[route_idx]
                    if 0 <= route_idx < len(route_options)
                    else str(route_idx)
                )
                dose_value = float(db.context_dosing_amounts[b, i].item())

                name_id: Optional[str] = None
                if b < len(db.context_subject_name) and i < len(db.context_subject_name[b]):
                    subject_name = db.context_subject_name[b][i]
                    if subject_name:
                        name_id = subject_name

                for s in range(sample_size):
                    observations = samples_i[s, b, valid_time_mask, 0].tolist()
                    individual: IndividualJSON = {
                        "observations": observations,
                        "observation_times": observation_times,
                        "dosing": [dose_value],
                        "dosing_type": [route_label],
                        "dosing_times": [dosing_time],
                        "dosing_name": [route_label],
                    }
                    if name_id is not None:
                        individual["name_id"] = name_id
                    studies_per_substance[b][s]["context"].append(individual)

        return studies_per_substance

    @torch.inference_mode()
    def sample_from_study_json(
        self,
        studies: StudyJSON | Sequence[StudyJSON],
        sample_size: int = 1,
        num_steps: int | None = None,
    ) -> StudyJSON | list[StudyJSON]:
        """Sample StudyJSON targets, dispatching predictive/generative modes per target.

        Predictive targets have non-empty ``observations`` and
        ``observation_times``. Generative targets leave both lists empty and
        request a decode schedule through ``remaining_times``. In both cases the
        sampled trajectories are written back into ``prediction_samples`` and
        ``prediction_times`` on the original target records.

        Example target blocks accepted by this helper::

            {"observations": [0.2, 0.4], "observation_times": [0.5, 1.0], "remaining_times": [2.0, 4.0]}
            {"observations": [], "observation_times": [], "remaining_times": [1.5, 3.0]}
        """

        return _sample_from_study_json_impl(
            self,
            studies,
            sample_size=sample_size,
            num_steps=num_steps,
        )

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
        num_steps: int | None = None,
        resolve_sampling_from_target: bool = False,
    ) -> List[StudyJSON]:
        """Generate new individuals and convert them into ``StudyJSON`` objects."""
        resolved_num_steps = self._resolve_sampling_num_steps(num_steps)
        samples, times, mask = self.sample_new_individual(
            db,
            sample_size=sample_size,
            decode_times=decode_times,
            num_steps=resolved_num_steps,
            resolve_sampling_from_target=resolve_sampling_from_target,
        )
        if getattr(self, "meta_dosing", None) is None:
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
            resolve_sampling_from_target=resolve_sampling_from_target,
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
        num_steps: int | None = None,
        max_permutations: int = 3,
        resolve_sampling_from_target: bool = False,
    ) -> List[List[StudyJSON]]:
        """Return nested ``StudyJSON`` objects for a list of permutations."""

        studies_per_perm: List[List[StudyJSON]] = []
        resolved_num_steps = self._resolve_sampling_num_steps(num_steps)
        for perm_idx, batch in enumerate(list_of_batches):
            if perm_idx >= max_permutations:
                break
            studies_per_perm.append(
                self.sample_new_individuals_to_studyjson(
                    batch,
                    sample_size=sample_size,
                    decode_times=decode_times,
                    num_steps=resolved_num_steps,
                    resolve_sampling_from_target=resolve_sampling_from_target,
                )
            )
        return studies_per_perm


class NewBasePKModel(pl.LightningModule, PyTorchModelHubMixin, ABC):
    """FlowPK base class exposing training, scaling, and sampling utilities."""

    config_class = HFFlowPKConfig

    def __init__(self, model_config: FlowPKExperimentConfig):
        super().__init__()
        self.model_config = model_config
        self.config = HFFlowPKConfig.from_flowpk(model_config)

        network_cfg = getattr(model_config, "network", None)
        vector_field_cfg = getattr(model_config, "vector_field", None)
        model_section = network_cfg if network_cfg is not None else vector_field_cfg
        self.loss_name = getattr(model_section, "loss_name", None)
        train_cfg = getattr(model_config, "train", None)
        self.meta_dosing = getattr(self.model_config, "dosing", None)
        mix_cfg = getattr(model_config, "mix_data", None)

        # Build core encoder/decoder modules (decoder required for all models,
        # encoder only for configs that define a network section).
        self.encoder = None
        if network_cfg is not None:
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

    def build_visualization_callback(self):
        """Build scheduler-driven PK callbacks from the training configuration.

        Empirical scheduler tasks may be declared as templates with
        ``sample_source='empirical_set'`` and no ``empirical_name``. In that
        case the callback keeps one task entry and resolves all configured
        empirical repos at execution time. Task-internal empirical tasks are
        also rewritten here so they can manage their own sampling once.
        """

        train_cfg = getattr(getattr(self, "model_config", None), "train", None)
        callbacks_scheduler = getattr(train_cfg, "callbacks_scheduler", None)
        if not callbacks_scheduler:
            return []

        from pff.training.callbacks.scheduler import BaseSchedulerCallback
        from pff.training.callbacks.task_registry import TASK_REGISTRY

        mix_cfg = getattr(getattr(self, "model_config", None), "mix_data", None)
        empirical_datasets = list(getattr(mix_cfg, "test_empirical_datasets", []) or [])
        model_label = getattr(self.model_config, "name_str", self.__class__.__name__)
        raw_scheduler = (
            asdict(callbacks_scheduler)
            if is_dataclass(callbacks_scheduler)
            else dict(callbacks_scheduler)
        )
        expanded_cfg = _expand_empirical_scheduler_tasks(
            raw_scheduler,
            empirical_datasets=empirical_datasets,
            model_label=str(model_label),
        )
        return [BaseSchedulerCallback.from_config(cfg=expanded_cfg, registry=TASK_REGISTRY)]

    def generate(
        self,
        batch: AICMECompartmentsDataBatch,
        num_samples: int = 1,
    ):
        """Return the composite scheduler payload consumed by PK task functions."""

        from pff.training.callbacks.pk_tasks import build_pk_task_samples

        return build_pk_task_samples(self, batch, num_samples=int(num_samples))

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
        equation (10) from the original paper where ``L = lower(H H^T)`` and
        ``Sigma = L L^T``.
        """

        if H is None:  # fallback to diagonal path
            return self.masked_gaussian_nll_loss(mean, mean.new_zeros(()), target, mask)

        if mask_individuals is not None:
            mask = mask & mask_individuals.unsqueeze(-1)

        _, _, T, _ = mean.shape
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

    @staticmethod
    def _infer_batch_size(batch: Any) -> int | None:
        """Infer batch size ``B`` from AICME batches or batch lists."""
        if isinstance(batch, (list, tuple)) and batch:
            batch = batch[0]
        if hasattr(batch, "context_obs"):
            try:
                return int(batch.context_obs.shape[0])
            except (AttributeError, TypeError, ValueError):
                return None
        return None

    # ------------------------------------------------------------------
    # Lightning interface
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):  # type: ignore[override]
        """Execute a single optimisation step shared by all PK models."""

        try:
            outputs = self(batch)
        except AssertionError as exc:  # encoder caught NaNs or infs
            self.print(f"Warning: {exc} at batch {batch_idx}; skipping.")
            zero = torch.zeros((), device=self.device, requires_grad=True)
            return {"loss": zero}

        losses = self._materialize_losses(outputs)
        if not losses:
            zero = torch.zeros((), device=self.device, requires_grad=True)
            return {"loss": zero}

        if not metrics_are_finite(losses):
            self.print(f"Warning: Non-finite loss values at batch {batch_idx}; skipping.")
            zero = torch.zeros((), device=self.device, requires_grad=True)
            return {"loss": zero}

        batch_size = self._infer_batch_size(batch)
        for key, value in losses.items():
            self.log(
                f"train_{key}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=(key == "loss"),
                batch_size=batch_size,
            )

        return losses

    def validation_step(self, batch_list, batch_idx):  # type: ignore[override]
        outputs = self(batch_list)
        losses_dict = self._materialize_losses(outputs)

        batch_size = self._infer_batch_size(batch_list)
        for key, value in losses_dict.items():
            self.log(
                f"val_{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=(key == "loss"),
                sync_dist=True,
                batch_size=batch_size,
            )

        return outputs

    def configure_optimizers(self):
        """Instantiate the optimiser used by all PK models."""
        return torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    # ------------------------------------------------------------------
    # UTILS
    # ------------------------------------------------------------------
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

    def forward_report(
        self,
        databatch_list: Sequence[AICMECompartmentsDataBatch],
        outputs_collector: Any,
    ) -> Tuple[List[Dict[str, Any]] | None, Dict[str, Dict[str, float]]]:
        """Compute aggregate evaluation metrics for predictive heads."""
        per_substance = self.forward_report_per_substance(databatch_list, outputs_collector)
        try:
            setattr(outputs_collector, "per_substance", per_substance)
        except Exception:
            pass
        return None, per_substance

    def forward_report_per_individual(
        self,
        databatch_list: Sequence[AICMECompartmentsDataBatch],
        mean_predictions: torch.Tensor,
        logvar_predictions: torch.Tensor,
        remainder_targets: torch.Tensor,
        remainder_masks: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """Compute per-individual predictive metrics averaged across permutations."""
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

                mse_val = self.masked_mse_loss(
                    mean_subject,
                    target_subject,
                    mask_subject,
                )["mse"].item()
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
                    "mse": mse_val,
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
        outputs_collector: Any,
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate predictive metrics across permutations per substance."""
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
                    mse_val = self.masked_mse_loss(
                        mean_subject,
                        target_subject,
                        mask_subject,
                    )["mse"].item()
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
                        "mse": mse_val,
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
        prediction_cache: dict[int, dict[str, object]] = {}

        for p_idx, batch in enumerate(batch_list):
            samples_S, times_S, target_raw, target_mask = self.sample_individual_prediction(batch)

            prediction_cache[p_idx] = {
                "samples_S": samples_S,
                "times_S": times_S,
                "target_raw": target_raw,
                "target_mask": target_mask,
                "batch": batch,
            }

            pred_mean = samples_S.mean(dim=0)  # [B,It,Tr,1]

            B, It, Tr, _ = pred_mean.shape
            indiv_mask = batch.mask_target_individuals

            raw_names = list(batch.substance_name)
            substance_names: list[str] = []
            for b, name in enumerate(raw_names):
                if name is None or name == "" or str(name).strip() == "":
                    substance_names.append(f"substance_{b}")
                else:
                    substance_names.append(str(name))

            for b in range(B):
                substance = substance_names[b]

                if not indiv_mask[b, 0]:
                    metrics_b = {"mse": 0.0, "rmse": 0.0, "log_rmse": 0.0, "r2": 0.0, "log_r2": 0.0}
                else:
                    pm = pred_mean[b, 0].unsqueeze(0).unsqueeze(0)
                    tg = target_raw[b, 0].unsqueeze(0).unsqueeze(0)
                    mk = target_mask[b, 0].unsqueeze(0).unsqueeze(0)

                    metrics_b = {
                        "mse": self.masked_mse_loss(pm, tg, mk)["mse"].item(),
                        "rmse": self.masked_rmse_loss(pm, tg, mk)["rmse"].item(),
                        "log_rmse": self.masked_log_rmse_loss(pm, tg, mk)["rmse"].item(),
                        "r2": self.masked_r2_score(pm, tg, mk).item(),
                        "log_r2": self.masked_log_r2_score(pm, tg, mk).item(),
                    }

                per_substance_perm_metrics.setdefault(substance, []).append(metrics_b)

        final: dict[str, dict[str, float]] = {}
        for substance, metrics_list in per_substance_perm_metrics.items():
            n_perm = len(metrics_list)
            if n_perm == 0:
                continue

            metric_keys = list(metrics_list[0].keys())
            agg: dict[str, float] = {}

            for metric_name in metric_keys:
                vals = [md[metric_name] for md in metrics_list]
                mean_val = sum(vals) / n_perm

                if n_perm > 1:
                    var = sum((v - mean_val) ** 2 for v in vals) / (n_perm - 1)
                else:
                    var = 0.0

                std_val = var**0.5

                agg[metric_name] = float(mean_val)
                agg[metric_name + "_std"] = float(std_val)

            agg["repo_id"] = repo_id
            final[substance] = agg

        return final, prediction_cache


__all__ = ["NewBasePKModel", "NewPredictiveMixin", "NewGenerativeMixin"]
