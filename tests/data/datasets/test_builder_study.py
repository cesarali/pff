import json
from pathlib import Path

import torch
from pff.data.data_empirical import EmpiricalBatchConfig, JSON2AICMEBuilder
from pff.data.data_empirical.builder import held_out_ind_json, held_out_list_json
from pff.config_classes.data_config import MetaDosingConfig


import json
from pathlib import Path
from typing import Iterable

import torch
from pff.data.data_empirical import EmpiricalBatchConfig, JSON2AICMEBuilder
from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.config_classes.data_config import MetaDosingConfig
from pff.data.data_generation.observations_classes import ObservationStrategyFactory
from pff.data.datasets.aicme_datasets import (
    AICMECompartmentsDataBatch,
    AICMECompartmentsDataset,
)


def _make_builder(max_individuals: int = 2) -> JSON2AICMEBuilder:
    """Create builder with fixed padding limits."""
    cfg = EmpiricalBatchConfig(
        max_databatch_size=2,
        max_individuals=max_individuals,
        max_observations=10,
        max_remaining=10,
    )
    return JSON2AICMEBuilder(cfg)


def _load_fixture(name: str) -> dict:
    """Load a JSON fixture from the shared ``tests/fixtures`` directory."""
    path = Path(__file__).resolve().parents[1] / "fixtures" / name
    with open(path) as f:
        return json.load(f)


def test_build_study_batch_ctx_only_no_targets():
    builder = _make_builder(max_individuals=5)
    study = _load_fixture("study_ctx_only_long.json")
    batch = builder.build_study_batch(study, MetaDosingConfig())
    assert batch.context_obs.shape == (1, 5, 10, 1)
    assert batch.target_obs_mask.sum() == 0
    expected_vals = torch.tensor(
        [
            [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            [0.4, 0.9, 1.4, 1.9, 2.4, 2.9, 3.4, 3.9, 4.4, 4.9],
            [0.3, 0.8, 1.3, 1.8, 2.3, 2.8, 3.3, 3.8, 4.3, 4.8],
            [0.2, 0.7, 1.2, 1.7, 2.2, 2.7, 3.2, 3.7, 4.2, 4.7],
            [0.1, 0.6, 1.1, 1.6, 2.1, 2.6, 3.1, 3.6, 4.1, 4.6],
        ]
    )
    assert torch.allclose(batch.context_obs.squeeze(0).squeeze(-1), expected_vals)
    expected_times = torch.tensor(
        [
            [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        ]
    )
    assert torch.allclose(batch.context_obs_time.squeeze(0).squeeze(-1), expected_times)
    assert batch.context_obs_mask.all()
    assert batch.mask_context_individuals.squeeze(0).tolist() == [True] * 5
    assert batch.mask_target_individuals.squeeze(0).tolist() == [False] * 5


def test_build_study_batch_ctx_tgt_with_remainder():
    builder = _make_builder()
    study = _load_fixture("study_ctx_tgt_long.json")
    batch = builder.build_study_batch(study, MetaDosingConfig())
    assert batch.target_rem_sim_mask.any()
    obs_times = batch.target_obs_time[0, 0, batch.target_obs_mask[0, 0]].squeeze(-1)
    rem_times = batch.target_rem_sim_time[0, 0, batch.target_rem_sim_mask[0, 0]].squeeze(-1)
    assert set(obs_times.tolist()).isdisjoint(set(rem_times.tolist()))
    assert torch.allclose(rem_times[-3:], torch.tensor([13.0, 14.0, 15.0]))
    assert batch.mask_context_individuals.squeeze(0).tolist() == [True, False]
    assert batch.mask_target_individuals.squeeze(0).tolist() == [True, False]


def test_build_one_aicmebatch_stacks_studies():
    builder = _make_builder(max_individuals=5)
    study_a = _load_fixture("study_ctx_only_long.json")
    study_b = _load_fixture("study_ctx_tgt_long.json")
    meta = MetaDosingConfig()

    stacked = builder.build_one_aicmebatch([study_a, study_b], meta)
    assert stacked.context_obs.shape[0] == 2  # B dimension

    single_a = builder.build_study_batch(study_a, meta)
    single_b = builder.build_study_batch(study_b, meta)

    assert torch.allclose(stacked.context_obs[0], single_a.context_obs[0])
    assert torch.allclose(stacked.target_obs[1], single_b.target_obs[0])


def test_held_out_ind_json_generates_permutations():
    study = _load_fixture("study_ctx_only_long.json")
    perms = held_out_ind_json(study, max_held_out_individuals=7)
    assert len(perms) == 7
    # first five permutations hold out each individual
    for i in range(5):
        assert len(perms[i]["target"]) == 1
        assert perms[i]["target"][0]["name_id"] == study["context"][i]["name_id"]
        assert len(perms[i]["context"]) == 4
    # remaining permutations repeat original study with no target
    for i in range(5, 7):
        assert perms[i]["target"] == []
        assert len(perms[i]["context"]) == 5


def test_held_out_list_json_stacks_batches():
    study_long = _load_fixture("study_ctx_only_long.json")
    study_short = _load_fixture("study_ctx_only.json")
    builder = _make_builder(max_individuals=5)
    meta = MetaDosingConfig()
    batches = held_out_list_json(
        builder, [study_long, study_short], meta, max_held_out_individuals=5
    )
    assert len(batches) == 5
    for b in batches:
        assert b.context_obs.shape[0] == 2
    # study_short has only two individuals -> only first two batches have targets
    assert batches[0].mask_target_individuals[1].any()
    assert batches[1].mask_target_individuals[1].any()
    for idx in range(2, 5):
        assert batches[idx].mask_target_individuals[1].sum() == 0


def _add_batch_dim(
    batch: AICMECompartmentsDataBatch,
) -> AICMECompartmentsDataBatch:
    """Ensure tensor fields carry a leading batch dimension ``B=1``.

    Each field of :class:`AICMECompartmentsDataBatch` represents tensors with a
    batch dimension. ``AICMECompartmentsDataset._generate_item`` omits this
    dimension, so we restore it here before comparing shapes to the empirical
    builder output.
    """

    def maybe_unsqueeze(t: torch.Tensor | Iterable) -> torch.Tensor | Iterable:
        return t.unsqueeze(0) if isinstance(t, torch.Tensor) else t

    return AICMECompartmentsDataBatch(*[maybe_unsqueeze(t) for t in batch])


def test_builder_matches_dataset_shapes() -> None:
    study = _load_fixture("study_ctx_only.json")

    cfg = NodePKExperimentConfig()
    cfg.mix_data.n_of_permutations = 2
    cfg.mix_data.n_of_target_individuals = 1
    cfg.meta_study.num_individuals_range = (2, 2)
    ctx_fn = ObservationStrategyFactory.from_config(cfg.context_observations, cfg.meta_study)
    tgt_fn = ObservationStrategyFactory.from_config(cfg.target_observations, cfg.meta_study)

    ds = AICMECompartmentsDataset(cfg, ctx_fn, tgt_fn, number_of_process=1)
    dataset_batches = [_add_batch_dim(b) for b in ds._generate_item(0)]

    max_obs, max_rem = ctx_fn._get_shapes_raw()
    builder_cfg = EmpiricalBatchConfig(
        max_databatch_size=2,
        max_individuals=1,
        max_observations=max_obs,
        max_remaining=max_rem,
    )
    builder = JSON2AICMEBuilder(builder_cfg)
    built_batches = builder.build_one_aicmebatch_as_dataset(
        [study], ctx_fn, tgt_fn, MetaDosingConfig()
    )

    for b_ds, b_emp in zip(dataset_batches, built_batches):
        for field in [
            "context_obs",
            "target_obs",
            "context_rem_sim",
            "target_rem_sim",
            "context_obs_time",
            "target_obs_time",
            "context_rem_sim_time",
            "target_rem_sim_time",
            "context_obs_mask",
            "target_obs_mask",
            "context_rem_sim_mask",
            "target_rem_sim_mask",
            "context_dosing_amounts",
            "target_dosing_amounts",
            "context_dosing_route_types",
            "target_dosing_route_types",
            "mask_context_individuals",
            "mask_target_individuals",
            "time_scales",
        ]:
            ds_tensor = getattr(b_ds, field)
            emp_tensor = getattr(b_emp, field)
            assert ds_tensor.shape == emp_tensor.shape


def test_build_one_aicmebatch_as_dataset_no_heldout_returns_single_batch() -> None:
    study = _load_fixture("study_ctx_only.json")

    cfg = NodePKExperimentConfig()
    cfg.mix_data.n_of_target_individuals = 1
    cfg.meta_study.num_individuals_range = (2, 2)
    ctx_fn = ObservationStrategyFactory.from_config(cfg.context_observations, cfg.meta_study)
    tgt_fn = ObservationStrategyFactory.from_config(cfg.target_observations, cfg.meta_study)

    max_obs, max_rem = ctx_fn._get_shapes_raw()
    builder_cfg = EmpiricalBatchConfig(
        max_databatch_size=2,
        max_individuals=2,
        max_observations=max_obs,
        max_remaining=max_rem,
    )
    builder = JSON2AICMEBuilder(builder_cfg)

    built_batches = builder.build_one_aicmebatch_as_dataset_no_heldout(
        [study],
        ctx_fn,
        tgt_fn,
        MetaDosingConfig(),
    )

    assert len(built_batches) == 1
    batch = built_batches[0]
    assert isinstance(batch, AICMECompartmentsDataBatch)
    assert int(batch.mask_target_individuals.sum().item()) == 0
    assert int(batch.mask_context_individuals.sum().item()) == len(study["context"])


def test_build_one_aicmebatch_as_dataset_no_heldout_merges_existing_targets() -> None:
    study = _load_fixture("study_ctx_tgt_long.json")

    cfg = NodePKExperimentConfig()
    cfg.mix_data.n_of_target_individuals = 1
    cfg.meta_study.num_individuals_range = (2, 2)
    ctx_fn = ObservationStrategyFactory.from_config(cfg.context_observations, cfg.meta_study)
    tgt_fn = ObservationStrategyFactory.from_config(cfg.target_observations, cfg.meta_study)

    max_obs, max_rem = ctx_fn._get_shapes_raw()
    builder_cfg = EmpiricalBatchConfig(
        max_databatch_size=2,
        max_individuals=5,
        max_observations=max_obs,
        max_remaining=max_rem,
    )
    builder = JSON2AICMEBuilder(builder_cfg)

    built_batches = builder.build_one_aicmebatch_as_dataset_no_heldout(
        [study],
        ctx_fn,
        tgt_fn,
        MetaDosingConfig(),
    )

    batch = built_batches[0]
    assert isinstance(batch, AICMECompartmentsDataBatch)
    assert int(batch.mask_target_individuals.sum().item()) == 0
    expected_context = len(study["context"]) + len(study["target"])
    assert int(batch.mask_context_individuals.sum().item()) == expected_context
