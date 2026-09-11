import pytest
import torch

from pff.config_classes.data_config import ObservationsConfig, SimpleMetaStudyConfig
from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.data.data_generation import compartment_models_management as cmm
from pff.data.data_generation.observations_classes import ObservationStrategyFactory
from pff.data.datasets.aicme_datasets import (
    AICMECompartmentsDataBatch,
    AICMECompartmentsDataModule,
    AICMECompartmentsDataset,
    AICMESyntheticExperimentDataset,
    build_reconstruction_db,
)


def _add_batch_dim(batch: AICMECompartmentsDataBatch) -> AICMECompartmentsDataBatch:
    """Ensure a leading batch dimension ``B=1`` in tensor fields."""

    def maybe_unsqueeze(t):
        return t.unsqueeze(0) if isinstance(t, torch.Tensor) else t

    return AICMECompartmentsDataBatch(*[maybe_unsqueeze(t) for t in batch])


def _build_synthetic_experiment_datamodule(batch_size: int = 2) -> AICMECompartmentsDataModule:
    """Create a small datamodule configured for synthetic-experiment loader tests."""

    cfg = NodePKExperimentConfig()
    cfg.mix_data.test_empirical_datasets = []
    cfg.meta_study.num_individuals_range = (3, 3)
    cfg.meta_study.time_num_steps = 20
    cfg.train.batch_size = batch_size
    cfg.train.num_workers = 0
    cfg.train.persistent_workers = False
    cfg.mix_data.n_of_permutations = 1
    cfg.mix_data.train_size = 4
    cfg.mix_data.val_size = 2
    cfg.mix_data.test_size = 2

    dm = AICMECompartmentsDataModule(cfg)
    dm.prepare_data()
    dm.setup()
    return dm


def _build_simple_synthetic_experiment_datamodule(
    batch_size: int = 2,
) -> AICMECompartmentsDataModule:
    """Create a small simple-mode datamodule for synthetic loader tests."""

    cfg = NodePKExperimentConfig()
    cfg.mix_data.test_empirical_datasets = []
    cfg.meta_study = SimpleMetaStudyConfig(
        num_individuals=3,
        num_individuals_range=(3, 3),
        time_num_steps=20,
        time_stop=24.0,
        p1=0.0,
    )
    cfg.context_observations.max_num_obs = 8
    cfg.train.batch_size = batch_size
    cfg.train.num_workers = 0
    cfg.train.persistent_workers = False
    cfg.mix_data.n_of_permutations = 1
    cfg.mix_data.train_size = 4
    cfg.mix_data.val_size = 2
    cfg.mix_data.test_size = 2

    dm = AICMECompartmentsDataModule(cfg)
    dm.prepare_data()
    dm.setup()
    return dm


def test_generate_item_masks():
    cfg = NodePKExperimentConfig()
    cfg.mix_data.n_of_permutations = 3
    cfg.meta_study.num_individuals_range = (3, 3)
    cfg.meta_study.time_num_steps = 20
    ctx_fn = ObservationStrategyFactory.from_config(cfg.context_observations, cfg.meta_study)
    tgt_fn = ObservationStrategyFactory.from_config(cfg.target_observations, cfg.meta_study)
    ds = AICMECompartmentsDataset(
        cfg,
        ctx_fn,
        tgt_fn,
        number_of_process=1,
    )
    list_of_batches = ds._generate_item(0)
    batch = list_of_batches[0]

    assert batch.mask_context_individuals.shape[0] == ds.max_context_individuals
    assert batch.mask_target_individuals.shape[0] == ds.n_of_target_individuals

    ctx_count = int(batch.context_obs_mask.any(dim=-1).sum().item())
    tgt_count = int(batch.target_obs_mask.any(dim=-1).sum().item())
    assert batch.mask_context_individuals.sum().item() == ctx_count
    assert batch.mask_target_individuals.sum().item() == tgt_count

    # Metadata shapes
    assert len(batch.study_name) == 1
    assert len(batch.substance_name) == 1
    assert len(batch.context_subject_name) == 1
    assert len(batch.context_subject_name[0]) == ds.max_context_individuals
    assert len(batch.target_subject_name) == 1
    assert len(batch.target_subject_name[0]) == ds.n_of_target_individuals


def test_generate_item_sample_target_dosing():
    cfg = NodePKExperimentConfig()
    cfg.mix_data.n_of_permutations = 1  # not relevant here, but keep tidy
    cfg.meta_study.num_individuals_range = (3, 3)
    cfg.meta_study.time_num_steps = 20

    ctx_fn = ObservationStrategyFactory.from_config(cfg.context_observations, cfg.meta_study)
    tgt_fn = ObservationStrategyFactory.from_config(cfg.target_observations, cfg.meta_study)

    ds = AICMECompartmentsDataset(
        cfg,
        ctx_fn,
        tgt_fn,
        number_of_process=1,
    )

    # Call the new generator
    list_of_batches = ds._generate_item_sample_target_dosing(0, n_targets=5)

    # Assertions
    assert isinstance(list_of_batches, list)
    assert len(list_of_batches) == 1
    batch = list_of_batches[0]

    # Check context exists
    assert batch.context_obs is not None
    assert batch.context_obs_time is not None

    # Check target exists and has expected number of individuals
    assert batch.target_obs is not None
    assert batch.target_obs.shape[0] == 5  # 5 target individuals

    # Check dosing metadata is consistent
    assert batch.target_dosing_amounts.shape[0] == 5
    assert batch.target_dosing_route_types.shape[0] == 5

    # All doses should be equal (repeated target dosing)
    assert torch.allclose(
        batch.target_dosing_amounts,
        batch.target_dosing_amounts[0].expand_as(batch.target_dosing_amounts),
    )


def test_build_reconstruction_db_shapes():
    """Ensure reconstruction databatch has matching time dimensions."""
    B, Ic, It = 1, 2, 1  # batch, context individuals, target individuals
    Tc_obs, Trc = 3, 2  # context observed and remainder times
    Tt_obs, Trt = 4, 1  # target observed and remainder times

    # Context tensors
    context_obs = torch.rand(B, Ic, Tc_obs, 1)
    context_obs_time = torch.rand(B, Ic, Tc_obs, 1)
    context_obs_mask = torch.ones(B, Ic, Tc_obs, dtype=torch.bool)
    context_rem_sim = torch.rand(B, Ic, Trc, 1)
    context_rem_sim_time = torch.rand(B, Ic, Trc, 1)
    context_rem_sim_mask = torch.ones(B, Ic, Trc, dtype=torch.bool)

    # Target tensors
    target_obs = torch.rand(B, It, Tt_obs, 1)
    target_obs_time = torch.rand(B, It, Tt_obs, 1)
    target_obs_mask = torch.ones(B, It, Tt_obs, dtype=torch.bool)
    target_rem_sim = torch.rand(B, It, Trt, 1)
    target_rem_sim_time = torch.rand(B, It, Trt, 1)
    target_rem_sim_mask = torch.ones(B, It, Trt, dtype=torch.bool)

    batch = AICMECompartmentsDataBatch(
        target_obs,
        target_obs_time,
        target_obs_mask,
        target_rem_sim,
        target_rem_sim_time,
        target_rem_sim_mask,
        context_obs,
        context_obs_time,
        context_obs_mask,
        context_rem_sim,
        context_rem_sim_time,
        context_rem_sim_mask,
        torch.zeros(B, It),  # target_dosing_amounts
        torch.zeros(B, It),  # target_dosing_route_types
        torch.zeros(B, Ic),  # context_dosing_amounts
        torch.zeros(B, Ic),  # context_dosing_route_types
        torch.ones(B, Ic, dtype=torch.bool),  # mask_context_individuals
        torch.ones(B, It, dtype=torch.bool),  # mask_target_individuals
        ["study"],
        [[f"c{i}" for i in range(Ic)]],
        [[f"t{i}" for i in range(It)]],
        ["drug"],
        torch.ones(B, 2),
        False,
    )

    recon_db = build_reconstruction_db(batch)

    # Time dimensions must match for concatenation along individuals
    assert recon_db.context_obs.shape[2] == recon_db.target_obs.shape[2]
    assert recon_db.context_obs_time.shape[2] == recon_db.target_obs_time.shape[2]
    assert recon_db.context_obs_mask.shape[2] == recon_db.target_obs_mask.shape[2]

    # torch.cat should succeed without raising
    torch.cat([recon_db.context_obs, recon_db.target_obs], dim=1)


def test_build_reconstruction_db_shapes_aicme():
    """Integration test using the ``_aicme_config`` setup from model tests."""
    from tests.models.test_aicme_pk import _aicme_config, _first_batch_list

    cfg = _aicme_config()
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)
    batch = batch_list[0]

    recon_db = build_reconstruction_db(batch)

    # Print shapes for inspection
    print("context_obs.shape:", recon_db.context_obs.shape)
    print("target_obs.shape:", recon_db.target_obs.shape)
    print("context_obs_time.shape:", recon_db.context_obs_time.shape)
    print("target_obs_time.shape:", recon_db.target_obs_time.shape)
    print("context_obs_mask.shape:", recon_db.context_obs_mask.shape)
    print("target_obs_mask.shape:", recon_db.target_obs_mask.shape)

    assert recon_db.context_obs.shape[2] == recon_db.target_obs.shape[2]
    assert recon_db.context_obs_time.shape[2] == recon_db.target_obs_time.shape[2]
    assert recon_db.context_obs_mask.shape[2] == recon_db.target_obs_mask.shape[2]


def test_obtain_shapes_matches_databatch() -> None:
    """Datamodule-obtained shapes match those of emitted databatches."""

    cfg = NodePKExperimentConfig()
    cfg.mix_data.n_of_permutations = 1
    cfg.mix_data.n_of_target_individuals = 1
    cfg.meta_study.num_individuals_range = (2, 2)

    dm = AICMECompartmentsDataModule(cfg)
    max_inds, max_obs, max_rem = dm.obtain_shapes()

    batch = _add_batch_dim(dm.train_dataset._generate_item(0)[0])

    assert batch.context_obs.shape[1] == max_inds == batch.target_obs.shape[1]
    assert batch.context_obs.shape[2] == max_obs == batch.target_obs.shape[2]
    assert batch.context_rem_sim.shape[2] == max_rem == batch.target_rem_sim.shape[2]


def test_aicme_databatch_to_alias_matches_to_device() -> None:
    """The databatch exposes a PyTorch-style ``.to(device)`` alias."""

    B, Ic, It, Tc, Tr = 1, 2, 1, 3, 2
    batch = AICMECompartmentsDataBatch(
        target_obs=torch.zeros(B, It, Tc, 1),
        target_obs_time=torch.zeros(B, It, Tc, 1),
        target_obs_mask=torch.ones(B, It, Tc, dtype=torch.bool),
        target_rem_sim=torch.zeros(B, It, Tr, 1),
        target_rem_sim_time=torch.zeros(B, It, Tr, 1),
        target_rem_sim_mask=torch.ones(B, It, Tr, dtype=torch.bool),
        context_obs=torch.zeros(B, Ic, Tc, 1),
        context_obs_time=torch.zeros(B, Ic, Tc, 1),
        context_obs_mask=torch.ones(B, Ic, Tc, dtype=torch.bool),
        context_rem_sim=torch.zeros(B, Ic, Tr, 1),
        context_rem_sim_time=torch.zeros(B, Ic, Tr, 1),
        context_rem_sim_mask=torch.ones(B, Ic, Tr, dtype=torch.bool),
        target_dosing_amounts=torch.zeros(B, It),
        target_dosing_route_types=torch.zeros(B, It, dtype=torch.long),
        context_dosing_amounts=torch.zeros(B, Ic),
        context_dosing_route_types=torch.zeros(B, Ic, dtype=torch.long),
        mask_context_individuals=torch.ones(B, Ic, dtype=torch.bool),
        mask_target_individuals=torch.ones(B, It, dtype=torch.bool),
        study_name=["study"],
        context_subject_name=[["c0", "c1"]],
        target_subject_name=[["t0"]],
        substance_name=["drug"],
        time_scales=torch.ones(B, 2),
        is_empirical=False,
    )

    moved = batch.to(torch.device("cpu"))

    assert isinstance(moved, AICMECompartmentsDataBatch)
    assert moved.context_obs.device.type == "cpu"
    assert moved.target_obs.device.type == "cpu"


def test_datamodule_generate_sample_experiment_repeated_dosing() -> None:
    """Repeated dosing should return one study per sampled dosing realization."""

    cfg = NodePKExperimentConfig()
    cfg.meta_study.num_individuals_range = (3, 3)
    cfg.meta_study.time_num_steps = 20

    dm = AICMECompartmentsDataModule(cfg)
    studies = dm.generate_synthetic_study_experiment_list(
        n_targets=4,
        n_dosings=5,
        dosing_mode="repeated_dosing",
    )

    assert isinstance(studies, list)
    assert len(studies) == 5

    reference_context = studies[0]["context"]
    for dosing_idx, study in enumerate(studies):
        assert study["context"] == reference_context
        assert len(study["target"]) == 4

        meta = study["meta_data"]
        assert meta["study_name"]
        assert meta["substance_name"]
        assert meta["dosing_mode"] == "repeated_dosing"
        assert meta["dosing_realization_index"] == str(dosing_idx)

        target_doses = [float(ind["dosing"][0]) for ind in study["target"]]
        target_routes = [str(ind["dosing_type"][0]) for ind in study["target"]]
        assert target_doses == [target_doses[0]] * len(target_doses)
        assert target_routes == [target_routes[0]] * len(target_routes)


def test_synthetic_experiment_dataset_item_returns_separate_batches() -> None:
    """Each synthetic-experiment dataset item should return ``P=n_dosings`` batches."""

    dm = _build_synthetic_experiment_datamodule(batch_size=2)
    dataset = AICMESyntheticExperimentDataset(
        dm,
        n_targets=4,
        n_dosings=3,
        dosing_mode="repeated_dosing",
        dataset_size=5,
    )

    assert len(dataset) == 5

    item = dataset[0]

    assert isinstance(item, list)
    assert len(item) == 3
    assert all(isinstance(batch, AICMECompartmentsDataBatch) for batch in item)
    assert all(batch.context_obs.shape[0] == 1 for batch in item)
    assert all(batch.target_obs.shape[0] == 1 for batch in item)
    assert all(batch.target_obs.shape[1] >= 4 for batch in item)
    assert all(len(batch.study_name) == 1 for batch in item)


def test_synthetic_experiment_dataloader_batches_and_metadata() -> None:
    """Collated synthetic-experiment batches should follow the AICME nested format."""

    dm = _build_synthetic_experiment_datamodule(batch_size=2)
    loader = dm.get_synthetic_experiment_dataloader(
        n_targets=4,
        n_dosings=3,
        dosing_mode="repeated_dosing",
        dataset_size=4,
    )

    batch_list = next(iter(loader))

    assert isinstance(batch_list, list)
    assert len(batch_list) == 3

    for batch in batch_list:
        assert isinstance(batch, AICMECompartmentsDataBatch)
        assert batch.context_obs.shape[0] == 2
        assert batch.target_obs.shape[0] == 2
        assert len(batch.study_name) == 2
        assert len(batch.substance_name) == 2
        assert len(batch.context_subject_name) == 2
        assert len(batch.target_subject_name) == 2
        assert all(len(names) == batch.context_obs.shape[1] for names in batch.context_subject_name)
        assert all(len(names) == batch.target_obs.shape[1] for names in batch.target_subject_name)


def test_synthetic_experiment_dataloader_preserves_large_target_capacity() -> None:
    """Requested targets larger than training target capacity must not be truncated."""

    dm = _build_synthetic_experiment_datamodule(batch_size=2)
    n_targets = 6
    loader = dm.get_synthetic_experiment_dataloader(
        n_targets=n_targets,
        n_dosings=2,
        dosing_mode="repeated_dosing",
        dataset_size=2,
    )

    batch_list = next(iter(loader))

    for batch in batch_list:
        assert batch.target_obs.shape[1] >= n_targets
        assert torch.equal(
            batch.mask_target_individuals.sum(dim=1),
            torch.full((batch.target_obs.shape[0],), n_targets, dtype=torch.long),
        )
        assert all(len(names) == batch.target_obs.shape[1] for names in batch.target_subject_name)


def test_synthetic_experiment_dataloader_dataset_size_controls_iteration() -> None:
    """Iterating the loader should cover exactly ``dataset_size`` experiment items."""

    dm = _build_synthetic_experiment_datamodule(batch_size=2)
    dataset_size = 5
    loader = dm.get_synthetic_experiment_dataloader(
        n_targets=3,
        n_dosings=2,
        dosing_mode="repeated_dosing",
        dataset_size=dataset_size,
    )

    total_items = 0
    for batch_list in loader:
        assert len(batch_list) == 2
        total_items += int(batch_list[0].context_obs.shape[0])

    assert total_items == dataset_size


@pytest.mark.parametrize(
    ("dosing_mode", "loader_kwargs"),
    [
        ("repeated_dosing", {}),
        (
            "dosing_list",
            {
                "dosing_list_generation": "dosing_from_samples",
            },
        ),
        ("diverse_dosing", {}),
        ("vpc_context", {}),
    ],
)
def test_synthetic_experiment_dataloader_smoke_by_dosing_mode(
    dosing_mode: str,
    loader_kwargs: dict,
) -> None:
    """All public dosing modes should produce collatable synthetic batches."""

    dm = _build_synthetic_experiment_datamodule(batch_size=2)
    loader = dm.get_synthetic_experiment_dataloader(
        n_targets=4,
        n_dosings=2,
        dosing_mode=dosing_mode,
        dataset_size=2,
        **loader_kwargs,
    )

    batch_list = next(iter(loader))

    assert isinstance(batch_list, list)
    assert len(batch_list) == 2
    assert all(isinstance(batch, AICMECompartmentsDataBatch) for batch in batch_list)
    assert all(batch.context_obs.shape[0] == 2 for batch in batch_list)
    assert all(batch.target_obs.shape[0] == 2 for batch in batch_list)
    if dosing_mode == "vpc_context":
        assert all(not batch.mask_target_individuals.any() for batch in batch_list)
        assert all(batch.mask_context_individuals.any() for batch in batch_list)


def test_simple_mode_synthetic_experiment_dataloader_smoke() -> None:
    """Simple mode should expose the same nested list loader structure."""

    dm = _build_simple_synthetic_experiment_datamodule(batch_size=2)
    loader = dm.get_synthetic_experiment_dataloader(
        n_targets=4,
        n_dosings=3,
        dosing_mode="diverse_dosing",
        dataset_size=2,
    )

    batch_list = next(iter(loader))

    assert isinstance(batch_list, list)
    assert len(batch_list) == 3
    assert all(isinstance(batch, AICMECompartmentsDataBatch) for batch in batch_list)
    assert all(batch.context_obs.shape[0] == 2 for batch in batch_list)
    assert all(batch.target_obs.shape[0] == 2 for batch in batch_list)


def test_simple_mode_synthetic_experiment_dataloader_uses_fixed_grid_targets() -> None:
    """Simple-mode targets should share one deterministic fixed observation grid."""

    dm = _build_simple_synthetic_experiment_datamodule(batch_size=1)
    loader = dm.get_synthetic_experiment_dataloader(
        n_targets=5,
        n_dosings=1,
        dosing_mode="diverse_dosing",
        dataset_size=1,
    )

    batch = next(iter(loader))[0]

    expected_target_capacity = min(
        dm.context_config.max_num_obs,
        dm.meta_config.time_num_steps,
    )
    assert batch.target_obs.shape[2] == expected_target_capacity
    assert not batch.target_rem_sim_mask.any()
    assert batch.context_obs_mask.any()
    assert batch.target_obs_mask.any()

    for batch_idx in range(batch.target_obs.shape[0]):
        valid_targets = batch.mask_target_individuals[batch_idx].nonzero(as_tuple=True)[0]
        assert len(valid_targets) > 0
        ref_idx = int(valid_targets[0].item())
        reference_times = batch.target_obs_time[batch_idx, ref_idx, :, 0]
        reference_mask = batch.target_obs_mask[batch_idx, ref_idx]
        for target_idx in valid_targets.tolist()[1:]:
            assert torch.equal(batch.target_obs_mask[batch_idx, target_idx], reference_mask)
            assert torch.allclose(
                batch.target_obs_time[batch_idx, target_idx, :, 0],
                reference_times,
            )


def test_simple_mode_synthetic_experiment_dataloader_preserves_large_target_capacity() -> None:
    """Simple-mode loaders must preserve requested target counts above train capacity."""

    dm = _build_simple_synthetic_experiment_datamodule(batch_size=2)
    n_targets = 6
    loader = dm.get_synthetic_experiment_dataloader(
        n_targets=n_targets,
        n_dosings=2,
        dosing_mode="diverse_dosing",
        dataset_size=2,
    )

    batch_list = next(iter(loader))

    for batch in batch_list:
        assert batch.target_obs.shape[1] >= n_targets
        assert torch.equal(
            batch.mask_target_individuals.sum(dim=1),
            torch.full((batch.target_obs.shape[0],), n_targets, dtype=torch.long),
        )


def test_simple_mode_synthetic_experiment_dataloader_dataset_size_controls_iteration() -> None:
    """Simple-mode iteration should cover exactly ``dataset_size`` items."""

    dm = _build_simple_synthetic_experiment_datamodule(batch_size=2)
    dataset_size = 5
    loader = dm.get_synthetic_experiment_dataloader(
        n_targets=4,
        n_dosings=2,
        dosing_mode="diverse_dosing",
        dataset_size=dataset_size,
    )

    total_items = 0
    for batch_list in loader:
        assert len(batch_list) == 2
        total_items += int(batch_list[0].context_obs.shape[0])

    assert total_items == dataset_size


@pytest.mark.parametrize("dosing_mode", ["repeated_dosing", "dosing_list", "vpc_context"])
def test_simple_mode_synthetic_experiment_dataloader_rejects_unsupported_dosing_modes(
    dosing_mode: str,
) -> None:
    """Simple-mode loaders should accept only diverse dosing."""

    dm = _build_simple_synthetic_experiment_datamodule(batch_size=1)

    with pytest.raises(ValueError, match="diverse_dosing"):
        dm.get_synthetic_experiment_dataloader(
            n_targets=4,
            n_dosings=1,
            dosing_mode=dosing_mode,
            dataset_size=1,
        )


def test_datamodule_generate_sample_experiment_dosing_list_from_samples() -> None:
    """Sample-experiment list mode should keep context fixed and vary target dosing."""

    cfg = NodePKExperimentConfig()
    cfg.meta_study.num_individuals_range = (3, 3)
    cfg.meta_study.time_num_steps = 20

    dm = AICMECompartmentsDataModule(cfg)
    studies = dm.generate_synthetic_study_experiment_list(
        n_targets=4,
        n_dosings=3,
        dosing_mode="dosing_list",
        dosing_list_generation="dosing_from_samples",
    )

    assert isinstance(studies, list)
    assert len(studies) == 3

    reference_context = studies[0]["context"]
    for dosing_idx, study in enumerate(studies):
        assert study["context"] == reference_context
        assert len(study["target"]) == 4

        meta = study["meta_data"]
        assert meta["dosing_mode"] == "dosing_list"
        assert meta["dosing_realization_index"] == str(dosing_idx)

        target_doses = [float(ind["dosing"][0]) for ind in study["target"]]
        target_routes = [str(ind["dosing_type"][0]) for ind in study["target"]]
        assert target_doses == [target_doses[0]] * len(target_doses)
        assert target_routes == [target_routes[0]] * len(target_routes)


def test_datamodule_generate_sample_experiment_dosing_list_from_range() -> None:
    """Range-based dosing lists should follow a deterministic log-dose grid."""

    cfg = NodePKExperimentConfig()
    cfg.meta_study.num_individuals_range = (3, 3)
    cfg.meta_study.time_num_steps = 20

    dm = AICMECompartmentsDataModule(cfg)
    studies = dm.generate_synthetic_study_experiment_list(
        n_targets=4,
        n_dosings=4,
        dosing_mode="dosing_list",
        dosing_list_generation="dosing_from_range",
        logdose_range=(-1.0, 1.0),
    )

    assert len(studies) == 4

    reference_context = studies[0]["context"]
    expected_doses = torch.exp(torch.linspace(-1.0, 1.0, steps=4)).tolist()
    routes = []
    for dosing_idx, study in enumerate(studies):
        assert study["context"] == reference_context
        assert len(study["target"]) == 4

        meta = study["meta_data"]
        assert meta["dosing_mode"] == "dosing_list"
        assert meta["dosing_realization_index"] == str(dosing_idx)

        target_doses = [float(ind["dosing"][0]) for ind in study["target"]]
        target_routes = [str(ind["dosing_type"][0]) for ind in study["target"]]
        assert target_doses == pytest.approx([expected_doses[dosing_idx]] * len(target_doses))
        assert target_routes == [target_routes[0]] * len(target_routes)
        routes.append(target_routes[0])

    assert len(set(routes)) == 1


def test_datamodule_generate_sample_experiment_diverse_dosing() -> None:
    """Diverse dosing should resample per-target dosing for each returned study."""

    cfg = NodePKExperimentConfig()
    cfg.meta_study.num_individuals_range = (3, 3)
    cfg.meta_study.time_num_steps = 20

    dm = AICMECompartmentsDataModule(cfg)
    studies = dm.generate_synthetic_study_experiment_list(
        n_targets=6,
        n_dosings=4,
        dosing_mode="diverse_dosing",
    )

    assert isinstance(studies, list)
    assert len(studies) == 4

    reference_context = studies[0]["context"]
    for dosing_idx, study in enumerate(studies):
        assert study["context"] == reference_context
        assert len(study["target"]) == 6

        meta = study["meta_data"]
        assert meta["dosing_mode"] == "diverse_dosing"
        assert meta["dosing_realization_index"] == str(dosing_idx)

        target_signatures = [
            (float(ind["dosing"][0]), str(ind["dosing_type"][0])) for ind in study["target"]
        ]
        assert len(set(target_signatures)) > 1


def test_datamodule_generate_sample_experiment_vpc_context() -> None:
    """VPC-context mode should serialize observed synthetic individuals under context only."""

    cfg = NodePKExperimentConfig()
    cfg.meta_study.num_individuals_range = (3, 3)
    cfg.meta_study.time_num_steps = 20

    dm = AICMECompartmentsDataModule(cfg)
    studies = dm.generate_synthetic_study_experiment_list(
        n_targets=5,
        n_dosings=3,
        dosing_mode="vpc_context",
    )

    assert isinstance(studies, list)
    assert len(studies) == 3

    for dosing_idx, study in enumerate(studies):
        assert len(study["context"]) == 5
        assert study["target"] == []

        meta = study["meta_data"]
        assert meta["dosing_mode"] == "vpc_context"
        assert meta["dosing_realization_index"] == str(dosing_idx)

        context_names = [str(ind["name_id"]) for ind in study["context"]]
        assert len(set(context_names)) == 5
        assert all("dosing" in ind for ind in study["context"])
        assert all("dosing_type" in ind for ind in study["context"])


def test_synthetic_experiment_dataloader_retries_invalid_sample_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loader items should recover from a transient invalid sample experiment."""

    dm = _build_synthetic_experiment_datamodule(batch_size=1)
    original = cmm._simulate_and_serialize_sample_experiment_block
    state = {"failed": False}

    def fail_once_then_delegate(**kwargs):
        if kwargs["block_name"] == "context" and not state["failed"]:
            state["failed"] = True
            raise cmm._SampleExperimentInvalidSimulationError(
                block_name="context",
                dosing_signature="forced_loader_retry",
            )
        return original(**kwargs)

    monkeypatch.setattr(cmm, "_simulate_and_serialize_sample_experiment_block", fail_once_then_delegate)

    loader = dm.get_synthetic_experiment_dataloader(
        n_targets=4,
        n_dosings=2,
        dosing_mode="repeated_dosing",
        dataset_size=2,
    )

    batch_list = next(iter(loader))

    assert state["failed"] is True
    assert isinstance(batch_list, list)
    assert len(batch_list) == 2
    assert all(isinstance(batch, AICMECompartmentsDataBatch) for batch in batch_list)
    assert all(batch.context_obs.shape[0] == 1 for batch in batch_list)
    assert all(batch.target_obs.shape[0] == 1 for batch in batch_list)


def test_generate_sample_experiment_diverse_dosing_ignores_dosing_list_generation() -> None:
    """Diverse dosing should not validate dosing-list-only arguments."""

    cfg = NodePKExperimentConfig()
    cfg.meta_study.num_individuals_range = (3, 3)
    cfg.meta_study.time_num_steps = 20

    dm = AICMECompartmentsDataModule(cfg)
    studies = dm.generate_synthetic_study_experiment_list(
        n_targets=4,
        n_dosings=2,
        dosing_mode="diverse_dosing",
        dosing_list_generation="ignored_for_diverse_dosing",
    )

    assert len(studies) == 2
    assert all(study["meta_data"]["dosing_mode"] == "diverse_dosing" for study in studies)


def test_generate_sample_experiment_forces_fixed_regular_grid_serialization() -> None:
    """Sample experiments should keep native context and fixed-grid targets."""

    cfg = NodePKExperimentConfig()
    cfg.meta_study.num_individuals_range = (3, 3)
    cfg.meta_study.time_num_steps = 24
    cfg.context_observations.type = "random"
    cfg.context_observations.max_num_obs = 18
    cfg.context_observations.add_rem = True
    cfg.context_observations.split_past_future = True
    cfg.context_observations.min_past = 2
    cfg.context_observations.max_past = 5
    cfg.target_observations.add_rem = True
    cfg.target_observations.split_past_future = True
    cfg.target_observations.min_past = 1
    cfg.target_observations.max_past = 1

    dm = AICMECompartmentsDataModule(cfg)
    study = dm.generate_synthetic_study_experiment_list(
        n_targets=3,
        n_dosings=1,
        dosing_mode="repeated_dosing",
    )[0]

    expected_target_obs = min(cfg.context_observations.max_num_obs, cfg.meta_study.time_num_steps)

    assert all("remaining" not in ind for ind in study["target"])
    assert all(len(ind["observations"]) <= cfg.context_observations.max_past for ind in study["context"])
    assert all(len(ind["observations"]) == expected_target_obs for ind in study["target"])

    target_times = [tuple(ind["observation_times"]) for ind in study["target"]]
    assert len(set(target_times)) == 1
    assert target_times[0][0] == pytest.approx(3.0)


def test_generate_sample_experiment_accepts_synthetic_target_observation_config() -> None:
    """Sample experiments should honor per-call synthetic target observation overrides."""

    cfg = NodePKExperimentConfig()
    cfg.meta_study.num_individuals_range = (3, 3)
    cfg.meta_study.time_num_steps = 24

    dm = AICMECompartmentsDataModule(cfg)
    study = dm.generate_synthetic_study_experiment_list(
        n_targets=3,
        n_dosings=1,
        dosing_mode="repeated_dosing",
        synthetic_target_observation_config=ObservationsConfig(
            type="pk_peak_half_life",
            max_num_obs=5,
            add_rem=True,
            split_past_future=True,
            min_past=1,
            max_past=2,
            drop_time_zero_observations=False,
            fixed_grid_start_index=1,
        ),
    )[0]

    target_times = [tuple(ind["observation_times"]) for ind in study["target"]]
    assert len(set(target_times)) == 1
    assert len(target_times[0]) == 5
    assert target_times[0][0] == pytest.approx(1.0)


def test_generate_sample_experiment_rejects_invalid_dosing_mode() -> None:
    """Public API should validate dosing mode values."""

    dm = AICMECompartmentsDataModule(NodePKExperimentConfig())

    with pytest.raises(ValueError, match="dosing_mode"):
        dm.generate_synthetic_study_experiment_list(
            n_targets=2,
            n_dosings=2,
            dosing_mode="invalid",
        )


def test_generate_sample_experiment_rejects_legacy_dosing_mode_alias() -> None:
    """Legacy dosing names should no longer be accepted."""

    dm = AICMECompartmentsDataModule(NodePKExperimentConfig())

    with pytest.raises(ValueError, match="dosing_mode"):
        dm.generate_synthetic_study_experiment_list(
            n_targets=2,
            n_dosings=2,
            dosing_mode="same_dosing",
        )


def test_generate_sample_experiment_rejects_missing_logdose_range() -> None:
    """Range-based dosing lists require an explicit log-dose range."""

    dm = AICMECompartmentsDataModule(NodePKExperimentConfig())

    with pytest.raises(ValueError, match="logdose_range"):
        dm.generate_synthetic_study_experiment_list(
            n_targets=2,
            n_dosings=2,
            dosing_mode="dosing_list",
            dosing_list_generation="dosing_from_range",
        )


if __name__ == "__main__":
    test_generate_item_sample_target_dosing()
