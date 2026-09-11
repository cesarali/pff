import json
import pathlib

import pytest
import torch

from pff.data.data_empirical.json_schema import (
    ValidationError,
    canonicalize_individual,
    canonicalize_study,
    studies_from_sampled_targets,
)
from pff.data.data_empirical.json_stats import compute_json_stats
from pff.data.datasets.aicme_batch import AICMECompartmentsDataBatch


def test_canonicalize_individual_sorts_and_dedups():
    ind = {
        "name_id": "ID1",
        "observations": [0.2, 0.1, 0.3],
        "observation_times": [0.5, 0.2, 0.5],
    }
    canon = canonicalize_individual(ind, min_obs=1)
    assert canon["observation_times"] == [0.2, 0.5]
    assert canon["observations"] == [0.1, 0.2]
    # ensure original not mutated
    assert ind["observation_times"] == [0.5, 0.2, 0.5]


def test_canonicalize_individual_disjoint_remainder():
    ind = {
        "observations": [1.0, 2.0, 3.0],
        "observation_times": [0.0, 1.0, 2.0],
        "remaining": [10.0, 11.0],
        "remaining_times": [1.0, 3.0],
    }
    canon = canonicalize_individual(ind, min_obs=1)
    assert canon["remaining_times"] == [3.0]
    assert canon["remaining"] == [11.0]


def test_canonicalize_individual_accepts_remaining_times_without_values():
    ind = {
        "observations": [],
        "observation_times": [],
        "remaining_times": [1.0, 3.0],
    }
    canon = canonicalize_individual(ind, min_obs=0)
    assert canon["remaining_times"] == [1.0, 3.0]
    assert "remaining" not in canon


def test_canonicalize_individual_dosing_lengths_match():
    ind = {
        "observations": [1.0, 2.0, 3.0],
        "observation_times": [0.0, 1.0, 2.0],
        "dosing": [50.0, 60.0],
        "dosing_times": [0.0],
        "dosing_type": ["oral"],
        "dosing_name": ["doseA"],
    }
    with pytest.raises(ValidationError):
        canonicalize_individual(ind, min_obs=1)


def test_canonicalize_study_keeps_meta_and_filters_targets():
    study = {
        "context": [
            {
                "name_id": "C1",
                "observations": [1.0, 2.0],
                "observation_times": [0.0, 1.0],
            }
        ],
        "target": [
            {
                "name_id": "T1",
                "observations": [1.0, 2.0],
                "observation_times": [0.0, 1.0],
            }
        ],
        "meta_data": {"study_name": "S", "substance_name": "Drug"},
    }

    canon = canonicalize_study(study)
    assert canon["meta_data"] == study["meta_data"]
    assert canon["meta_data"] is not study["meta_data"]
    assert len(canon["context"]) == 1
    assert len(canon["target"]) == 1


def test_compute_json_stats():
    """Compute statistics from JSON fixtures and ensure values match expectations."""
    fixture_dir = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
    with open(fixture_dir / "studies_long_list.json") as f:
        studies = json.load(f)
    with open(fixture_dir / "study_ctx_tgt.json") as f:
        studies.append(json.load(f))

    stats = compute_json_stats(studies)
    assert stats.min_context_individuals == 1
    assert stats.max_context_individuals == 2
    assert stats.min_target_individuals == 0
    assert stats.max_target_individuals == 1
    assert stats.min_observation == 0.4
    assert stats.max_observation == 5.0
    assert stats.substances == ["DrugX"]
    assert stats.max_total_individuals == 2
    assert stats.max_observations == 10
    assert stats.max_remaining == 10
    summary = stats.get_substance_summary("DrugX")
    assert summary["individual_count"] == 6
    assert summary["min_observations"] == 3
    assert summary["max_observations"] == 10
    assert summary["observation_time_steps"][0] == 0.5
    assert summary["observation_time_steps"][-1] == 15.0
    assert len(stats.studies_for_substance("DrugX")) == 3
    assert stats.get_substance_summary("Unknown") == {}
    assert stats.studies_for_substance("Unknown") == []


def test_studies_from_sampled_targets_supports_target_resolved_layout():
    """Target-resolved outputs should map axis 0 to target individuals."""

    db = AICMECompartmentsDataBatch(
        target_obs=torch.zeros(1, 2, 2, 1),
        target_obs_time=torch.tensor([[[[1.0], [2.0]], [[1.0], [3.0]]]], dtype=torch.float32),
        target_obs_mask=torch.tensor([[[True, True], [True, True]]]),
        target_rem_sim=torch.zeros(1, 2, 0, 1),
        target_rem_sim_time=torch.zeros(1, 2, 0, 1),
        target_rem_sim_mask=torch.zeros(1, 2, 0, dtype=torch.bool),
        context_obs=torch.zeros(1, 1, 1, 1),
        context_obs_time=torch.tensor([[[[0.5]]]], dtype=torch.float32),
        context_obs_mask=torch.tensor([[[True]]]),
        context_rem_sim=torch.zeros(1, 1, 0, 1),
        context_rem_sim_time=torch.zeros(1, 1, 0, 1),
        context_rem_sim_mask=torch.zeros(1, 1, 0, dtype=torch.bool),
        target_dosing_amounts=torch.tensor([[10.0, 20.0]], dtype=torch.float32),
        target_dosing_route_types=torch.tensor([[0, 1]], dtype=torch.long),
        context_dosing_amounts=torch.tensor([[5.0]], dtype=torch.float32),
        context_dosing_route_types=torch.tensor([[1]], dtype=torch.long),
        mask_context_individuals=torch.tensor([[True]]),
        mask_target_individuals=torch.tensor([[True, True]]),
        study_name=["study_json"],
        context_subject_name=[["ctx_0"]],
        target_subject_name=[["tgt_0", "tgt_1"]],
        substance_name=["drug_json"],
        time_scales=torch.ones(1, 2),
        is_empirical=True,
    )
    samples = torch.tensor(
        [
            [[[11.0], [12.0], [13.0]]],
            [[[21.0], [22.0], [23.0]]],
        ],
        dtype=torch.float32,
    )  # [It=2, B=1, T=3, 1]
    times = torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.float32)
    mask = torch.tensor([[True, True, True]])

    studies = studies_from_sampled_targets(
        db=db,
        samples=samples,
        times=times,
        mask=mask,
        route_options=["oral", "iv"],
        dosing_time=0.0,
        resolve_sampling_from_target=True,
    )

    assert len(studies) == 1
    targets = studies[0]["target"]
    assert [target["name_id"] for target in targets] == ["tgt_0", "tgt_1"]
    assert targets[0]["observations"] == [11.0, 12.0, 13.0]
    assert targets[1]["observations"] == [21.0, 22.0, 23.0]
    assert targets[0]["dosing"] == [10.0]
    assert targets[1]["dosing"] == [20.0]
    assert targets[0]["dosing_type"] == ["oral"]
    assert targets[1]["dosing_type"] == ["iv"]


if __name__ == "__main__":
    test_canonicalize_individual_dosing_lengths_match()
