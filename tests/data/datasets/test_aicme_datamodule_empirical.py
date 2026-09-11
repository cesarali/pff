"""Tests for empirical batch caching in ``AICMECompartmentsDataModule``."""

import pytest
import torch

from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.data.datasets.aicme_batch import AICMECompartmentsDataBatch
from pff.data.datasets.aicme_datasets import AICMECompartmentsDataModule


def _build_empirical_batch() -> AICMECompartmentsDataBatch:
    """Build a tiny empirical-like batch with two substances (B=2)."""

    B = 2
    I = 1
    T = 2
    R = 1
    zeros_obs = torch.zeros(B, I, T, 1)
    zeros_rem = torch.zeros(B, I, R, 1)
    full_obs_mask = torch.ones(B, I, T, dtype=torch.bool)
    full_rem_mask = torch.ones(B, I, R, dtype=torch.bool)
    dosing = torch.ones(B, I)
    route = torch.zeros(B, I)
    full_individual_mask = torch.ones(B, I, dtype=torch.bool)
    time_scales = torch.ones(B, 2)

    return AICMECompartmentsDataBatch(
        target_obs=zeros_obs,
        target_obs_time=zeros_obs,
        target_obs_mask=full_obs_mask,
        target_rem_sim=zeros_rem,
        target_rem_sim_time=zeros_rem,
        target_rem_sim_mask=full_rem_mask,
        context_obs=zeros_obs,
        context_obs_time=zeros_obs,
        context_obs_mask=full_obs_mask,
        context_rem_sim=zeros_rem,
        context_rem_sim_time=zeros_rem,
        context_rem_sim_mask=full_rem_mask,
        target_dosing_amounts=dosing,
        target_dosing_route_types=route,
        context_dosing_amounts=dosing,
        context_dosing_route_types=route,
        mask_context_individuals=full_individual_mask,
        mask_target_individuals=full_individual_mask,
        study_name=["study_a", "study_b"],
        context_subject_name=[["ctx_a"], ["ctx_b"]],
        target_subject_name=[["tgt_a"], ["tgt_b"]],
        substance_name=["Drug A", "Drug-B"],
        time_scales=time_scales,
        is_empirical=True,
    )


def test_datamodule_caches_heldout_and_no_heldout_batches(monkeypatch) -> None:
    cfg = NodePKExperimentConfig()
    cfg.mix_data.test_empirical_datasets = ["repo_a"]
    dm = AICMECompartmentsDataModule(cfg)

    calls = []

    def fake_loader(
        repo_id,
        split="train",
        meta_dosing=None,
        stats=None,
        datamodule=None,
        *,
        held_out=True,
    ):
        calls.append((repo_id, held_out))
        if held_out:
            return ["heldout_batch"]
        return ["no_heldout_batch"]

    import pff.data.data_empirical as empirical_pkg

    monkeypatch.setattr(empirical_pkg, "load_empirical_hf_batches_as_dm", fake_loader)
    dm._load_empirical_test_batches()

    assert calls == [("repo_a", True), ("repo_a", False)]
    assert dm.get_empirical_test_batches()["repo_a"] == ["heldout_batch"]
    assert dm.get_empirical_test_batches(no_heldout=True)["repo_a"] == ["no_heldout_batch"]


def test_prepare_data_preloads_empirical_batches_on_rank_zero(monkeypatch) -> None:
    cfg = NodePKExperimentConfig()
    cfg.mix_data.test_empirical_datasets = ["repo_a"]
    cfg.mix_data.train_size = 1
    cfg.mix_data.val_size = 1
    cfg.mix_data.test_size = 1
    cfg.train.num_workers = 0
    cfg.train.persistent_workers = False

    dm = AICMECompartmentsDataModule(cfg)
    calls = []

    def fake_preload() -> None:
        calls.append("load")
        dm.empirical_test_batches = {"repo_a": ["heldout_batch"]}
        dm.empirical_test_batches_no_heldout = {"repo_a": ["no_heldout_batch"]}

    monkeypatch.setattr(dm, "_load_empirical_test_batches", fake_preload)

    dm.prepare_data()

    assert calls == ["load"]
    assert dm.get_empirical_test_batches()["repo_a"] == ["heldout_batch"]
    assert dm.get_empirical_test_batches(no_heldout=True)["repo_a"] == ["no_heldout_batch"]


def test_empirical_target_strategy_uses_fixed_legacy_defaults() -> None:
    """Empirical target strategy is always the fixed legacy PK configuration."""

    cfg = NodePKExperimentConfig()
    cfg.mix_data.test_empirical_datasets = []
    cfg.mix_data.train_size = 1
    cfg.mix_data.val_size = 1
    cfg.mix_data.test_size = 1
    cfg.train.num_workers = 0
    cfg.train.persistent_workers = False
    cfg.target_observations.type = "random"
    cfg.target_observations.max_num_obs = 20
    cfg.target_observations.min_past = 0
    cfg.target_observations.max_past = 5

    dm = AICMECompartmentsDataModule(cfg)
    dm.prepare_data()

    assert dm.empirical_target_config is not None
    assert dm.empirical_target_strategy is not None
    assert dm.empirical_target_config.type is None
    assert dm.empirical_target_config.max_num_obs == 15
    assert dm.empirical_target_config.min_past == 0
    assert dm.empirical_target_config.max_past == 5
    assert dm.empirical_target_config.split_past_future is True

    tgt_obs_cap, _ = dm.empirical_target_strategy.get_shapes()
    assert tgt_obs_cap == 5


def test_fix_past_selection_applies_to_empirical_target_strategy() -> None:
    """Target past override must propagate to empirical target strategy."""

    cfg = NodePKExperimentConfig()
    cfg.mix_data.test_empirical_datasets = []
    cfg.mix_data.train_size = 1
    cfg.mix_data.val_size = 1
    cfg.mix_data.test_size = 1
    cfg.train.num_workers = 0
    cfg.train.persistent_workers = False
    # Keep synthetic target on random to ensure the empirical target strategy
    # is the one that effectively receives the fixed-past override.
    cfg.target_observations.type = "random"

    dm = AICMECompartmentsDataModule(cfg)
    dm.prepare_data()

    assert dm.empirical_target_strategy is not None
    assert getattr(dm.empirical_target_strategy, "_fixed_past_obs_count", None) is None

    dm.fix_past_selection(3, who="target")
    assert getattr(dm.empirical_target_strategy, "_fixed_past_obs_count", None) == 3

    dm.release_past_selection(who="target")
    assert getattr(dm.empirical_target_strategy, "_fixed_past_obs_count", None) is None


def test_get_empirical_test_batches_can_move_batches_to_device_without_mutation() -> None:
    cfg = NodePKExperimentConfig()
    dm = AICMECompartmentsDataModule(cfg)
    batch = _build_empirical_batch()
    dm.empirical_test_batches = {"repo_a": [batch]}
    dm.empirical_test_batches_no_heldout = {}
    dm._empirical_loaded = True

    moved = dm.get_empirical_test_batches(device=torch.device("cpu"))

    assert moved["repo_a"][0] is not batch
    assert moved["repo_a"][0].target_obs.device.type == "cpu"
    assert dm.empirical_test_batches["repo_a"][0] is batch


def test_select_empirical_batch_list_falls_back_to_first_non_empty(monkeypatch) -> None:
    cfg = NodePKExperimentConfig()
    dm = AICMECompartmentsDataModule(cfg)
    expected_batch = object()

    def fake_get_batches(*, no_heldout=False):
        del no_heldout
        return {"repo_empty": [], "repo_nonempty": [expected_batch]}

    monkeypatch.setattr(dm, "get_empirical_test_batches", fake_get_batches)
    selected_key, batch_list = dm.select_empirical_batch_list(dataset_key="unknown_repo")

    assert selected_key == "repo_nonempty"
    assert batch_list == [expected_batch]


def test_describe_empirical_test_batches_lists_studies_and_drugs(monkeypatch) -> None:
    cfg = NodePKExperimentConfig()
    dm = AICMECompartmentsDataModule(cfg)
    batch = _build_empirical_batch()

    def fake_get_batches(*, no_heldout=False):
        del no_heldout
        return {"repo_a": [batch]}

    monkeypatch.setattr(dm, "get_empirical_test_batches", fake_get_batches)
    studies, drugs = dm.describe_empirical_test_batches(
        empirical_batches=dm.get_empirical_test_batches(no_heldout=True),
        no_heldout=True,
        print_available=False,
    )

    assert studies == ["study_a", "study_b"]
    assert drugs == ["Drug A", "Drug-B"]


def test_describe_empirical_test_batches_defaults_to_batch_zero() -> None:
    cfg = NodePKExperimentConfig()
    dm = AICMECompartmentsDataModule(cfg)
    batch_zero = _build_empirical_batch()._replace(
        study_name=["study_zero_a", "study_zero_b"],
        substance_name=["Drug Zero A", "Drug Zero B"],
    )
    batch_one = _build_empirical_batch()._replace(
        study_name=["study_one_a", "study_one_b"],
        substance_name=["Drug One A", "Drug One B"],
    )

    studies, drugs = dm.describe_empirical_test_batches(
        empirical_batches={"repo_a": [batch_zero, batch_one]},
        print_available=False,
    )

    assert studies == ["study_zero_a", "study_zero_b"]
    assert drugs == ["Drug Zero A", "Drug Zero B"]


def test_slice_single_substance_batch_by_name_matches_normalized_name() -> None:
    batch = _build_empirical_batch()

    single = AICMECompartmentsDataModule.slice_single_substance_batch_by_name(batch, "drug b")
    studies, drugs = AICMECompartmentsDataModule.describe_empirical_batch(
        single, print_available=False
    )

    assert single.target_obs.shape[0] == 1
    assert single.context_obs.shape[0] == 1
    assert studies == ["study_b"]
    assert drugs == ["Drug-B"]


def test_slice_single_substance_batch_by_name_raises_when_missing() -> None:
    batch = _build_empirical_batch()

    with pytest.raises(ValueError, match="Choose from"):
        AICMECompartmentsDataModule.slice_single_substance_batch_by_name(
            batch, "missing_drug"
        )


def test_select_empirical_drug_batch_returns_single_drug_and_study() -> None:
    cfg = NodePKExperimentConfig()
    dm = AICMECompartmentsDataModule(cfg)
    batch = _build_empirical_batch()
    empirical_batches = {"repo_a": [batch]}

    single_batch, selected_study, selected_drug = dm.select_empirical_drug_batch(
        empirical_batches=empirical_batches,
        selected_drug="drug-a",
        print_selection=False,
    )

    assert single_batch.target_obs.shape[0] == 1
    assert selected_study == "study_a"
    assert selected_drug == "Drug A"


def test_select_empirical_drug_batch_raises_when_drug_missing() -> None:
    cfg = NodePKExperimentConfig()
    dm = AICMECompartmentsDataModule(cfg)
    batch = _build_empirical_batch()
    empirical_batches = {"repo_a": [batch]}

    with pytest.raises(ValueError, match="Choose from"):
        dm.select_empirical_drug_batch(
            empirical_batches=empirical_batches,
            selected_drug="drug_z",
            print_selection=False,
        )


def test_select_empirical_drug_batch_searches_across_datasets() -> None:
    cfg = NodePKExperimentConfig()
    dm = AICMECompartmentsDataModule(cfg)
    batch_a = _build_empirical_batch()._replace(
        study_name=["study_a_only", "study_b_only"],
        substance_name=["Drug A", "Drug B"],
    )
    batch_c = _build_empirical_batch()._replace(
        study_name=["study_c_only", "study_d_only"],
        substance_name=["Drug C", "Drug D"],
    )

    empirical_batches = {"repo_a": [batch_a], "repo_c": [batch_c]}
    single_batch, selected_study, selected_drug = dm.select_empirical_drug_batch(
        empirical_batches=empirical_batches,
        selected_drug="drug d",
        print_selection=False,
    )

    assert single_batch.target_obs.shape[0] == 1
    assert selected_study == "study_d_only"
    assert selected_drug == "Drug D"


def test_select_empirical_drug_batch_can_select_multiple_permutations() -> None:
    cfg = NodePKExperimentConfig()
    dm = AICMECompartmentsDataModule(cfg)

    batch_perm0 = _build_empirical_batch()
    batch_perm1 = _build_empirical_batch()._replace(
        target_obs=torch.full_like(batch_perm0.target_obs, 1.0),
    )
    empirical_batches = {"repo_a": [batch_perm0, batch_perm1]}

    selected_batches, selected_study, selected_drug = dm.select_empirical_drug_batch(
        empirical_batches=empirical_batches,
        selected_drug="drug-a",
        permutation_indexes=[1, 0],
        print_selection=False,
    )

    assert isinstance(selected_batches, list)
    assert len(selected_batches) == 2
    assert selected_batches[0].target_obs.shape[0] == 1
    assert selected_batches[0].target_obs[0, 0, 0, 0].item() == 1.0
    assert selected_batches[1].target_obs[0, 0, 0, 0].item() == 0.0
    assert selected_study == "study_a"
    assert selected_drug == "Drug A"


def test_select_empirical_drug_batch_can_select_specific_permutation_index() -> None:
    cfg = NodePKExperimentConfig()
    dm = AICMECompartmentsDataModule(cfg)

    batch_perm0 = _build_empirical_batch()
    batch_perm1 = _build_empirical_batch()._replace(
        target_obs=torch.full_like(batch_perm0.target_obs, 2.0),
    )
    empirical_batches = {"repo_a": [batch_perm0, batch_perm1]}

    single_batch, selected_study, selected_drug = dm.select_empirical_drug_batch(
        empirical_batches=empirical_batches,
        selected_drug="drug-a",
        permutation_indexes=1,
        print_selection=False,
    )

    assert single_batch.target_obs.shape[0] == 1
    assert single_batch.target_obs[0, 0, 0, 0].item() == 2.0
    assert selected_study == "study_a"
    assert selected_drug == "Drug A"
