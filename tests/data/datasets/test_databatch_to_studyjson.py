import json
from pathlib import Path

import torch

from pff.data.data_empirical import (
    EmpiricalBatchConfig,
    JSON2AICMEBuilder,
    databatch_to_study_jsons,
)
from pff.config_classes.data_config import MetaDosingConfig
from pff.data.data_empirical import prediction_to_study_jsons


def _load_fixture(name: str) -> dict:
    path = Path(__file__).resolve().parents[1] / "fixtures" / name
    with open(path) as f:
        return json.load(f)


def test_databatch_to_studyjson_roundtrip() -> None:
    study = _load_fixture("study_ctx_tgt.json")

    builder_cfg = EmpiricalBatchConfig(
        max_databatch_size=1,
        max_individuals=1,
        max_observations=3,
        max_remaining=3,
    )
    builder = JSON2AICMEBuilder(builder_cfg)
    meta = MetaDosingConfig()

    batch = builder.build_study_batch(study, meta)
    studies = databatch_to_study_jsons(batch, meta)
    assert len(studies) == 1

    rebuilt = builder.build_one_aicmebatch(studies, meta)
    for name in batch._fields:
        orig = getattr(batch, name)
        new = getattr(rebuilt, name)
        if isinstance(orig, torch.Tensor):
            assert torch.equal(orig, new), name
        else:
            assert orig == new


def test_databatch_to_studyjson_fills_missing_meta() -> None:
    study = _load_fixture("study_ctx_tgt.json")

    builder_cfg = EmpiricalBatchConfig(
        max_databatch_size=1,
        max_individuals=1,
        max_observations=3,
        max_remaining=3,
    )
    builder = JSON2AICMEBuilder(builder_cfg)
    meta = MetaDosingConfig()

    batch = builder.build_study_batch(study, meta)
    batch = batch._replace(study_name=[""], substance_name=[""])
    studies = databatch_to_study_jsons(batch, meta)
    meta_out = studies[0]["meta_data"]
    assert meta_out["study_name"] == "study_0"
    assert meta_out["substance_name"] == "substance_0"


def test_prediction_to_studyjsons() -> None:
    study = _load_fixture("study_ctx_tgt.json")

    builder_cfg = EmpiricalBatchConfig(
        max_databatch_size=1,
        max_individuals=1,
        max_observations=3,
        max_remaining=3,
    )
    builder = JSON2AICMEBuilder(builder_cfg)
    meta = MetaDosingConfig()

    batch = builder.build_study_batch(study, meta)
    pred = batch.target_rem_sim.unsqueeze(0)
    pred_time = batch.target_rem_sim_time.unsqueeze(0)

    studies = prediction_to_study_jsons(pred, pred_time, batch, meta)
    assert len(studies) == 1

    target = studies[0]["target"][0]
    assert "prediction_samples" in target
    assert target["prediction_samples"][0] == pred[0, 0, 0, :, 0].tolist()
    assert "prediction_times" in target
    assert target["prediction_times"] == pred_time[0, 0, 0, :, 0].tolist()


def test_prediction_to_studyjsons_keeps_only_predicted_targets() -> None:
    study = _load_fixture("study_ctx_tgt.json")

    builder_cfg = EmpiricalBatchConfig(
        max_databatch_size=1,
        max_individuals=1,
        max_observations=3,
        max_remaining=3,
    )
    builder = JSON2AICMEBuilder(builder_cfg)
    meta = MetaDosingConfig()

    batch = builder.build_study_batch(study, meta)

    # Build a two-target batch where only the second target is valid.
    multi_target_batch = batch._replace(
        target_obs=torch.cat([batch.target_obs + 100.0, batch.target_obs], dim=1),
        target_obs_time=torch.cat([batch.target_obs_time, batch.target_obs_time], dim=1),
        target_obs_mask=torch.cat([batch.target_obs_mask, batch.target_obs_mask], dim=1),
        target_rem_sim=torch.cat([batch.target_rem_sim + 100.0, batch.target_rem_sim], dim=1),
        target_rem_sim_time=torch.cat([batch.target_rem_sim_time, batch.target_rem_sim_time], dim=1),
        target_rem_sim_mask=torch.cat([batch.target_rem_sim_mask, batch.target_rem_sim_mask], dim=1),
        target_dosing_amounts=torch.cat(
            [batch.target_dosing_amounts + 1.0, batch.target_dosing_amounts], dim=1
        ),
        target_dosing_route_types=torch.cat(
            [batch.target_dosing_route_types, batch.target_dosing_route_types], dim=1
        ),
        mask_target_individuals=torch.tensor([[False, True]], dtype=torch.bool),
        target_subject_name=[["target_invalid", "target_valid"]],
    )

    # Predict only one target individual (It=1), matching FlowPK single-target sampling.
    pred = multi_target_batch.target_rem_sim[:, 1:2].unsqueeze(0)
    pred_time = multi_target_batch.target_rem_sim_time[:, 1:2].unsqueeze(0)

    studies = prediction_to_study_jsons(pred, pred_time, multi_target_batch, meta)
    assert len(studies) == 1
    assert len(studies[0]["target"]) == 1

    target = studies[0]["target"][0]
    assert target["name_id"] == "target_valid"
    assert target["prediction_samples"][0] == pred[0, 0, 0, :, 0].tolist()
    assert target["prediction_times"] == pred_time[0, 0, 0, :, 0].tolist()


def test_builder_materializes_remaining_times_without_remaining_values() -> None:
    study = _load_fixture("study_ctx_tgt.json")
    study["target"][0].pop("remaining")

    builder_cfg = EmpiricalBatchConfig(
        max_databatch_size=1,
        max_individuals=1,
        max_observations=3,
        max_remaining=3,
    )
    builder = JSON2AICMEBuilder(builder_cfg)
    meta = MetaDosingConfig()

    batch = builder.build_study_batch(study, meta)
    assert batch.target_rem_sim_mask[0, 0].tolist() == [True, True, True]
    assert batch.target_rem_sim_time[0, 0, :, 0].tolist() == [2.0, 3.0, 4.0]
    assert batch.target_rem_sim[0, 0, :, 0].tolist() == [0.0, 0.0, 0.0]
