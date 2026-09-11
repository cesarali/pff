"""Tests for mixed predictive/generative StudyJSON sampling helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from pff import config_dir
from pff.config_classes.data_config import MetaDosingConfig
from pff.config_classes.flow_pk_config import FlowPKExperimentConfig
from pff.data.data_empirical.json_schema import StudyJSON
from pff.data.datasets.aicme_batch import AICMECompartmentsDataBatch
from pff.models.amortized_inference.flows_pk import FlowPK
from pff.models.amortized_inference.generative_pk import (
    NewGenerativeMixin,
    NewPredictiveMixin,
)


def _mixed_study_json() -> StudyJSON:
    """Return a compact StudyJSON mixing predictive and generative targets."""

    return {
        "context": [
            {
                "name_id": "ctx_0",
                "observations": [0.1, 0.2, 0.3],
                "observation_times": [0.5, 1.0, 2.0],
                "dosing": [1.0],
                "dosing_type": ["oral"],
                "dosing_times": [0.0],
                "dosing_name": ["oral"],
            }
        ],
        "target": [
            {
                "name_id": "tgt_predict",
                "observations": [0.4, 0.8],
                "observation_times": [0.5, 1.0],
                "remaining_times": [2.0, 4.0],
                "dosing": [2.0],
                "dosing_type": ["oral"],
                "dosing_times": [0.0],
                "dosing_name": ["oral"],
            },
            {
                "name_id": "tgt_generate",
                "observations": [],
                "observation_times": [],
                "remaining_times": [1.5, 3.0],
                "dosing": [7.0],
                "dosing_type": ["iv"],
                "dosing_times": [0.0],
                "dosing_name": ["iv"],
            },
        ],
        "meta_data": {
            "study_name": "mixed_demo",
            "substance_name": "drug_x",
        },
    }


class _DummyMixedStudySampler(NewPredictiveMixin, NewGenerativeMixin):
    """Deterministic stub exercising both sampling paths."""

    def __init__(self) -> None:
        self.meta_dosing = replace(MetaDosingConfig(), route_options=["oral", "iv"], time=0.0)

    def sample_individual_prediction(
        self,
        databatch: AICMECompartmentsDataBatch,
        sample_size: int = 1,
    ):
        times = databatch.target_rem_sim_time
        samples = torch.stack(
            [times + float(100 * (sample_idx + 1)) for sample_idx in range(sample_size)],
            dim=0,
        )
        repeated_times = times.unsqueeze(0).repeat(sample_size, 1, 1, 1, 1)
        return samples, repeated_times, databatch.target_rem_sim, databatch.target_rem_sim_mask

    def sample_new_individual(
        self,
        db: AICMECompartmentsDataBatch,
        sample_size: int = 10,
        decode_times=None,
        ignore_logvar: bool = True,
        num_steps: int | None = None,
        dosing=None,
        resolve_sampling_from_target: bool = False,
    ):
        _ = ignore_logvar
        _ = num_steps
        _ = resolve_sampling_from_target
        if decode_times is None or dosing is None:
            raise ValueError("Dummy generative sampler expects decode_times and dosing.")
        times, mask = decode_times
        dose, route = dosing
        base = times[:, :, 0] + dose.unsqueeze(-1) * 10.0 + route.float().unsqueeze(-1)
        samples = torch.stack(
            [(base + float(sample_idx)).unsqueeze(-1) for sample_idx in range(sample_size)],
            dim=0,
        )
        return samples, times, mask


class _DummyPredictiveOnly(NewPredictiveMixin):
    """Predictive-only stub for capability checks."""

    def __init__(self) -> None:
        self.meta_dosing = replace(MetaDosingConfig(), route_options=["oral", "iv"], time=0.0)

    def sample_individual_prediction(
        self,
        databatch: AICMECompartmentsDataBatch,
        sample_size: int = 1,
    ):
        times = databatch.target_rem_sim_time
        samples = torch.stack(
            [times + float(sample_idx + 1) for sample_idx in range(sample_size)],
            dim=0,
        )
        repeated_times = times.unsqueeze(0).repeat(sample_size, 1, 1, 1, 1)
        return samples, repeated_times, databatch.target_rem_sim, databatch.target_rem_sim_mask


class _DummyGenerativeOnly(NewGenerativeMixin):
    """Generative-only stub for capability checks."""

    def __init__(self) -> None:
        self.meta_dosing = replace(MetaDosingConfig(), route_options=["oral", "iv"], time=0.0)

    def sample_new_individual(
        self,
        db: AICMECompartmentsDataBatch,
        sample_size: int = 10,
        decode_times=None,
        ignore_logvar: bool = True,
        num_steps: int | None = None,
        dosing=None,
        resolve_sampling_from_target: bool = False,
    ):
        _ = db
        _ = ignore_logvar
        _ = num_steps
        _ = resolve_sampling_from_target
        if decode_times is None:
            raise ValueError("Dummy generative sampler expects decode_times.")
        times, mask = decode_times
        samples = torch.stack(
            [times + float(sample_idx + 1) for sample_idx in range(sample_size)],
            dim=0,
        )
        return samples, times, mask


def _flow_pk_config() -> FlowPKExperimentConfig:
    """Load the mixed predictive/generative FlowPK config used for smoke tests."""

    default_yaml = (
        Path(config_dir)
        / "experiment_configs"
        / "UAI"
        / "flow-pk-predict-n-generate-test"
        / "flowPK.yaml"
    )
    return FlowPKExperimentConfig.from_yaml(str(default_yaml))


def test_sample_from_study_json_mixes_predictive_and_generative_targets() -> None:
    study = _mixed_study_json()
    original = deepcopy(study)
    model = _DummyMixedStudySampler()

    sampled = model.sample_from_study_json(study, sample_size=2)

    assert sampled["context"] == original["context"]
    assert sampled["meta_data"] == original["meta_data"]
    assert [target["name_id"] for target in sampled["target"]] == ["tgt_predict", "tgt_generate"]

    predictive_target = sampled["target"][0]
    generative_target = sampled["target"][1]

    assert predictive_target["observations"] == original["target"][0]["observations"]
    assert predictive_target["remaining_times"] == original["target"][0]["remaining_times"]
    assert predictive_target["prediction_times"] == [2.0, 4.0]
    assert predictive_target["prediction_samples"] == [[102.0, 104.0], [202.0, 204.0]]

    assert generative_target["observations"] == []
    assert generative_target["remaining_times"] == original["target"][1]["remaining_times"]
    assert generative_target["prediction_times"] == [1.5, 3.0]
    assert generative_target["prediction_samples"] == [[72.5, 74.0], [73.5, 75.0]]
    assert "remaining" not in generative_target


def test_sample_from_study_json_requires_target_entries() -> None:
    study = _mixed_study_json()
    study["target"] = []
    model = _DummyMixedStudySampler()

    with pytest.raises(ValueError, match="at least one target"):
        _ = model.sample_from_study_json(study, sample_size=1)


def test_sample_from_study_json_rejects_partial_target_observation_blocks() -> None:
    study = _mixed_study_json()
    study["target"][0]["observation_times"] = []
    model = _DummyMixedStudySampler()

    with pytest.raises(ValueError, match="observations and observation_times together"):
        _ = model.sample_from_study_json(study, sample_size=1)


def test_sample_from_study_json_requires_predictive_capability_for_observed_targets() -> None:
    study = _mixed_study_json()
    model = _DummyGenerativeOnly()

    with pytest.raises(ValueError, match="Predictive target individuals require"):
        _ = model.sample_from_study_json(study, sample_size=1)


def test_sample_from_study_json_requires_generative_capability_for_empty_targets() -> None:
    study = _mixed_study_json()
    model = _DummyPredictiveOnly()

    with pytest.raises(ValueError, match="Generative target individuals require"):
        _ = model.sample_from_study_json(study, sample_size=1)


def test_flow_pk_sample_from_study_json_smoke() -> None:
    model = FlowPK(_flow_pk_config())
    model.eval()

    sampled = model.sample_from_study_json(_mixed_study_json(), sample_size=1, num_steps=2)

    assert len(sampled["target"]) == 2
    assert sampled["target"][0]["prediction_times"] == [2.0, 4.0]
    assert sampled["target"][1]["prediction_times"] == [1.5, 3.0]
    assert len(sampled["target"][0]["prediction_samples"]) == 1
    assert len(sampled["target"][1]["prediction_samples"]) == 1
    assert len(sampled["target"][0]["prediction_samples"][0]) == 2
    assert len(sampled["target"][1]["prediction_samples"][0]) == 2


class _TrackingGenerativeOnly(_DummyGenerativeOnly):
    """Track whether helper wrappers forward target-resolved mode."""

    def __init__(self) -> None:
        super().__init__()
        self.resolve_flags: list[bool] = []

    def sample_new_individual(self, *args, resolve_sampling_from_target: bool = False, **kwargs):
        self.resolve_flags.append(bool(resolve_sampling_from_target))
        if kwargs.get("decode_times") is None and len(args) >= 1:
            db = args[0]
            kwargs["decode_times"] = (
                db.target_obs_time[:, 0, :, :],
                db.target_obs_mask[:, 0, :].bool(),
            )
        return super().sample_new_individual(
            *args,
            resolve_sampling_from_target=resolve_sampling_from_target,
            **kwargs,
        )


def _wrapper_test_batch() -> AICMECompartmentsDataBatch:
    """Build a minimal databatch for generative helper wrapper tests."""

    return AICMECompartmentsDataBatch(
        target_obs=torch.zeros(1, 1, 2, 1),
        target_obs_time=torch.tensor([[[[1.0], [2.0]]]], dtype=torch.float32),
        target_obs_mask=torch.tensor([[[True, True]]]),
        target_rem_sim=torch.zeros(1, 1, 0, 1),
        target_rem_sim_time=torch.zeros(1, 1, 0, 1),
        target_rem_sim_mask=torch.zeros(1, 1, 0, dtype=torch.bool),
        context_obs=torch.zeros(1, 1, 2, 1),
        context_obs_time=torch.tensor([[[[0.5], [1.5]]]], dtype=torch.float32),
        context_obs_mask=torch.tensor([[[True, True]]]),
        context_rem_sim=torch.zeros(1, 1, 0, 1),
        context_rem_sim_time=torch.zeros(1, 1, 0, 1),
        context_rem_sim_mask=torch.zeros(1, 1, 0, dtype=torch.bool),
        target_dosing_amounts=torch.tensor([[2.0]], dtype=torch.float32),
        target_dosing_route_types=torch.tensor([[1]], dtype=torch.long),
        context_dosing_amounts=torch.tensor([[1.0]], dtype=torch.float32),
        context_dosing_route_types=torch.tensor([[0]], dtype=torch.long),
        mask_context_individuals=torch.tensor([[True]]),
        mask_target_individuals=torch.tensor([[True]]),
        study_name=["wrapper_study"],
        context_subject_name=[["ctx_0"]],
        target_subject_name=[["tgt_0"]],
        substance_name=["drug_wrapper"],
        time_scales=torch.ones(1, 2),
        is_empirical=True,
    )


def test_sample_new_individual_helper_wrappers_forward_target_resolved_flag() -> None:
    """StudyJSON helper wrappers should forward target-resolved mode."""

    model = _TrackingGenerativeOnly()
    batch = _wrapper_test_batch()

    studies = model.sample_new_individuals_to_studyjson(
        batch,
        sample_size=1,
        resolve_sampling_from_target=True,
    )
    study_batches = model.sample_new_individuals_from_batchlist_to_study_json(
        [batch],
        sample_size=1,
        resolve_sampling_from_target=True,
    )

    assert studies
    assert study_batches
    assert model.resolve_flags == [True, True]
