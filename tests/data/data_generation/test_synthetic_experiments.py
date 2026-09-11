from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from pff import config_dir
from pff.config_classes.data_config import (
    MetaDosingConfig,
    MetaStudyConfig,
    ObservationsConfig,
)
from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.data.data_generation import compartment_models_management as cmm
from pff.data.datasets.aicme_datasets import (
    AICMECompartmentsDataBatch,
    AICMECompartmentsDataModule,
)

NOTEBOOK_N_TARGETS = 20
NOTEBOOK_N_DOSINGS = 2
NOTEBOOK_DIVERSE_N_TARGETS = 20
NOTEBOOK_LOGDOSE_RANGE = (-1.0, 1.0)


def _build_notebook_sample_experiment_datamodule() -> AICMECompartmentsDataModule:
    """Create the datamodule used by the sample-experiment notebook examples."""

    experiment_dir = config_dir / "experiment_configs" / "UAI" / "flow-pk-generate"
    meta_study_config_path = experiment_dir / "base.meta_study.yaml"
    dosing_config_path = experiment_dir / "base.dosing.yaml"
    observations_config_path = experiment_dir / "base.observations.yaml"

    meta_study_config = MetaStudyConfig.from_yaml(meta_study_config_path)
    meta_dosing_config = MetaDosingConfig.from_yaml(dosing_config_path)
    context_observations = ObservationsConfig.from_yaml(
        observations_config_path,
        section="context_observations",
    )
    target_observations = ObservationsConfig.from_yaml(
        observations_config_path,
        section="target_observations",
    )

    cfg = NodePKExperimentConfig(
        meta_study=meta_study_config,
        dosing=meta_dosing_config,
        context_observations=context_observations,
        target_observations=target_observations,
    )

    # Keep the same notebook-friendly runtime knobs while avoiding dataloader work.
    cfg.train.num_workers = 0
    cfg.train.persistent_workers = False
    cfg.train.batch_size = 2
    cfg.mix_data.n_of_permutations = 1
    cfg.mix_data.train_size = 4
    cfg.mix_data.val_size = 2
    cfg.mix_data.test_size = 2

    dm = AICMECompartmentsDataModule(cfg)
    dm.prepare_data()
    return dm


def _extract_target_dose_and_route(study: dict) -> list[tuple[float, str]]:
    """Return per-target ``(dose, route)`` pairs after asserting dosing keys exist."""

    target_dose_and_route: list[tuple[float, str]] = []
    for ind in study["target"]:
        assert "dosing" in ind
        assert "dosing_type" in ind
        target_dose_and_route.append((float(ind["dosing"][0]), str(ind["dosing_type"][0])))
    return target_dose_and_route


@pytest.fixture(scope="module")
def notebook_dm() -> AICMECompartmentsDataModule:
    """Shared notebook-style datamodule for sample-experiment smoke tests."""

    torch.manual_seed(7)
    return _build_notebook_sample_experiment_datamodule()


def test_notebook_repeated_dosing_call(notebook_dm: AICMECompartmentsDataModule) -> None:
    """Mirror the notebook repeated-dosing call without plotting."""

    repeated_dosing_studies = notebook_dm.generate_synthetic_study_experiment_list(
        n_targets=NOTEBOOK_N_TARGETS,
        n_dosings=NOTEBOOK_N_DOSINGS,
        dosing_mode="repeated_dosing",
    )

    assert len(repeated_dosing_studies) == NOTEBOOK_N_DOSINGS

    reference_context = repeated_dosing_studies[0]["context"]
    for dosing_idx, study in enumerate(repeated_dosing_studies):
        assert study["context"] == reference_context
        assert len(study["target"]) == NOTEBOOK_N_TARGETS
        assert study["meta_data"]["dosing_mode"] == "repeated_dosing"
        assert study["meta_data"]["dosing_realization_index"] == str(dosing_idx)

        target_dose_and_route = _extract_target_dose_and_route(study)
        target_doses = [dose for dose, _ in target_dose_and_route]
        target_routes = [route for _, route in target_dose_and_route]
        assert target_doses == pytest.approx([target_doses[0]] * NOTEBOOK_N_TARGETS)
        assert target_routes == [target_routes[0]] * NOTEBOOK_N_TARGETS


def test_sample_experiment_uses_shared_fixed_observation_times(
    notebook_dm: AICMECompartmentsDataModule,
) -> None:
    """Targets should share one fixed observation grid across studies."""

    studies = notebook_dm.generate_synthetic_study_experiment_list(
        n_targets=4,
        n_dosings=2,
        dosing_mode="repeated_dosing",
    )

    reference_target_times = [
        tuple(ind["observation_times"]) for ind in studies[0]["target"]
    ]

    assert len(set(reference_target_times)) == 1

    for study in studies[1:]:
        assert [tuple(ind["observation_times"]) for ind in study["target"]] == reference_target_times


def test_notebook_dosing_list_from_samples_call(
    notebook_dm: AICMECompartmentsDataModule,
) -> None:
    """Mirror the notebook sample-based dosing-list call without plotting."""

    dosing_list_sampled_studies = notebook_dm.generate_synthetic_study_experiment_list(
        n_targets=NOTEBOOK_N_TARGETS,
        n_dosings=NOTEBOOK_N_DOSINGS,
        dosing_mode="dosing_list",
        dosing_list_generation="dosing_from_samples",
    )

    assert len(dosing_list_sampled_studies) == NOTEBOOK_N_DOSINGS

    reference_context = dosing_list_sampled_studies[0]["context"]
    for dosing_idx, study in enumerate(dosing_list_sampled_studies):
        assert study["context"] == reference_context
        assert len(study["target"]) == NOTEBOOK_N_TARGETS
        assert study["meta_data"]["dosing_mode"] == "dosing_list"
        assert study["meta_data"]["dosing_realization_index"] == str(dosing_idx)

        target_dose_and_route = _extract_target_dose_and_route(study)
        target_doses = [dose for dose, _ in target_dose_and_route]
        target_routes = [route for _, route in target_dose_and_route]
        assert target_doses == pytest.approx([target_doses[0]] * NOTEBOOK_N_TARGETS)
        assert target_routes == [target_routes[0]] * NOTEBOOK_N_TARGETS


def test_notebook_dosing_list_from_range_call(
    notebook_dm: AICMECompartmentsDataModule,
) -> None:
    """Mirror the notebook range-based dosing-list call without plotting."""

    dosing_list_range_studies = notebook_dm.generate_synthetic_study_experiment_list(
        n_targets=NOTEBOOK_N_TARGETS,
        n_dosings=NOTEBOOK_N_DOSINGS,
        dosing_mode="dosing_list",
        dosing_list_generation="dosing_from_range",
        logdose_range=NOTEBOOK_LOGDOSE_RANGE,
    )

    assert len(dosing_list_range_studies) == NOTEBOOK_N_DOSINGS

    reference_context = dosing_list_range_studies[0]["context"]
    expected_doses = torch.exp(
        torch.linspace(
            NOTEBOOK_LOGDOSE_RANGE[0],
            NOTEBOOK_LOGDOSE_RANGE[1],
            steps=NOTEBOOK_N_DOSINGS,
        )
    ).tolist()
    routes = []

    for dosing_idx, study in enumerate(dosing_list_range_studies):
        assert study["context"] == reference_context
        assert len(study["target"]) == NOTEBOOK_N_TARGETS
        assert study["meta_data"]["dosing_mode"] == "dosing_list"
        assert study["meta_data"]["dosing_realization_index"] == str(dosing_idx)

        target_dose_and_route = _extract_target_dose_and_route(study)
        target_doses = [dose for dose, _ in target_dose_and_route]
        target_routes = [route for _, route in target_dose_and_route]
        assert target_doses == pytest.approx([expected_doses[dosing_idx]] * NOTEBOOK_N_TARGETS)
        assert target_routes == [target_routes[0]] * NOTEBOOK_N_TARGETS
        routes.append(target_routes[0])

    assert len(set(routes)) == 1


def test_notebook_diverse_dosing_call(notebook_dm: AICMECompartmentsDataModule) -> None:
    """Mirror the notebook diverse-dosing call without plotting."""

    diverse_dosing_studies = notebook_dm.generate_synthetic_study_experiment_list(
        n_targets=NOTEBOOK_DIVERSE_N_TARGETS,
        n_dosings=NOTEBOOK_N_DOSINGS,
        dosing_mode="diverse_dosing",
    )

    assert len(diverse_dosing_studies) == NOTEBOOK_N_DOSINGS

    reference_context = diverse_dosing_studies[0]["context"]
    for dosing_idx, study in enumerate(diverse_dosing_studies):
        assert study["context"] == reference_context
        assert len(study["target"]) == NOTEBOOK_DIVERSE_N_TARGETS
        assert study["meta_data"]["dosing_mode"] == "diverse_dosing"
        assert study["meta_data"]["dosing_realization_index"] == str(dosing_idx)

        target_signatures = _extract_target_dose_and_route(study)
        assert len(set(target_signatures)) > 1


def test_sample_experiment_retries_invalid_context_block(
    notebook_dm: AICMECompartmentsDataModule,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generator should resample the whole experiment after a context failure."""

    original = cmm._simulate_and_serialize_sample_experiment_block
    state = {"failed": False}

    def fail_once_then_delegate(**kwargs):
        if kwargs["block_name"] == "context" and not state["failed"]:
            state["failed"] = True
            raise cmm._SampleExperimentInvalidSimulationError(
                block_name="context",
                dosing_signature="forced_context_retry",
            )
        return original(**kwargs)

    monkeypatch.setattr(cmm, "_simulate_and_serialize_sample_experiment_block", fail_once_then_delegate)

    studies = notebook_dm.generate_synthetic_study_experiment_list(
        n_targets=NOTEBOOK_N_TARGETS,
        n_dosings=NOTEBOOK_N_DOSINGS,
        dosing_mode="repeated_dosing",
    )

    assert state["failed"] is True
    assert len(studies) == NOTEBOOK_N_DOSINGS
    assert all(len(study["target"]) == NOTEBOOK_N_TARGETS for study in studies)


def test_sample_experiment_retries_invalid_target_block(
    notebook_dm: AICMECompartmentsDataModule,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generator should resample the whole experiment after a target failure."""

    original = cmm._simulate_and_serialize_sample_experiment_block
    state = {"failed": False}

    def fail_once_then_delegate(**kwargs):
        if kwargs["block_name"] == "target" and not state["failed"]:
            state["failed"] = True
            raise cmm._SampleExperimentInvalidSimulationError(
                block_name="target",
                dosing_signature="forced_target_retry",
            )
        return original(**kwargs)

    monkeypatch.setattr(cmm, "_simulate_and_serialize_sample_experiment_block", fail_once_then_delegate)

    studies = notebook_dm.generate_synthetic_study_experiment_list(
        n_targets=NOTEBOOK_N_TARGETS,
        n_dosings=NOTEBOOK_N_DOSINGS,
        dosing_mode="repeated_dosing",
    )

    assert state["failed"] is True
    assert len(studies) == NOTEBOOK_N_DOSINGS
    assert all(len(study["target"]) == NOTEBOOK_N_TARGETS for study in studies)


def test_sample_experiment_retry_exhaustion_reports_last_failure(
    notebook_dm: AICMECompartmentsDataModule,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exhausted retries should surface the last block and dosing signature."""

    def always_fail(**kwargs):
        raise cmm._SampleExperimentInvalidSimulationError(
            block_name=str(kwargs["block_name"]),
            dosing_signature="forced_exhaustion_signature",
        )

    monkeypatch.setattr(cmm, "_SAMPLE_EXPERIMENT_MAX_RETRY_ATTEMPTS", 3)
    monkeypatch.setattr(cmm, "_simulate_and_serialize_sample_experiment_block", always_fail)

    with pytest.raises(
        RuntimeError,
        match=(
            "Unable to build a valid sample experiment after 3 attempts.*"
            "Last failed block='context'.*forced_exhaustion_signature"
        ),
    ):
        notebook_dm.generate_synthetic_study_experiment_list(
            n_targets=NOTEBOOK_N_TARGETS,
            n_dosings=NOTEBOOK_N_DOSINGS,
            dosing_mode="repeated_dosing",
        )


def test_notebook_repeated_dosing_dataloader_structure(
    notebook_dm: AICMECompartmentsDataModule,
) -> None:
    """The sample-experiment loader should collate to ``List[AICMECompartmentsDataBatch]``."""

    loader = notebook_dm.get_synthetic_experiment_dataloader(
        n_targets=NOTEBOOK_N_TARGETS,
        n_dosings=NOTEBOOK_N_DOSINGS,
        dosing_mode="repeated_dosing",
        dataset_size=4,
    )

    batch_list = next(iter(loader))

    assert isinstance(batch_list, list)
    assert len(batch_list) == NOTEBOOK_N_DOSINGS

    for batch in batch_list:
        assert isinstance(batch, AICMECompartmentsDataBatch)
        assert batch.context_obs.shape[0] == notebook_dm.batch_size
        assert batch.target_obs.shape[0] == notebook_dm.batch_size
        assert len(batch.study_name) == notebook_dm.batch_size
        assert len(batch.substance_name) == notebook_dm.batch_size
        assert len(batch.context_subject_name) == notebook_dm.batch_size
        assert len(batch.target_subject_name) == notebook_dm.batch_size
        assert all(len(names) == batch.context_obs.shape[1] for names in batch.context_subject_name)
        assert all(len(names) == batch.target_obs.shape[1] for names in batch.target_subject_name)


def test_notebook_synthetic_experiment_dataloader_preserves_large_target_capacity(
    notebook_dm: AICMECompartmentsDataModule,
) -> None:
    """Loader must preserve requested target counts larger than training capacity."""

    loader = notebook_dm.get_synthetic_experiment_dataloader(
        n_targets=NOTEBOOK_N_TARGETS,
        n_dosings=NOTEBOOK_N_DOSINGS,
        dosing_mode="repeated_dosing",
        dataset_size=2,
    )

    batch_list = next(iter(loader))

    for batch in batch_list:
        assert batch.target_obs.shape[1] >= NOTEBOOK_N_TARGETS
        assert torch.equal(
            batch.mask_target_individuals.sum(dim=1),
            torch.full((batch.target_obs.shape[0],), NOTEBOOK_N_TARGETS, dtype=torch.long),
        )
        assert all(len(names) == batch.target_obs.shape[1] for names in batch.target_subject_name)


def test_notebook_synthetic_experiment_dataloader_dataset_size_controls_iteration(
    notebook_dm: AICMECompartmentsDataModule,
) -> None:
    """The dataloader should expose exactly ``dataset_size`` experiment items."""

    dataset_size = 5
    loader = notebook_dm.get_synthetic_experiment_dataloader(
        n_targets=8,
        n_dosings=NOTEBOOK_N_DOSINGS,
        dosing_mode="repeated_dosing",
        dataset_size=dataset_size,
    )

    total_items = 0
    for batch_list in loader:
        assert len(batch_list) == NOTEBOOK_N_DOSINGS
        total_items += int(batch_list[0].context_obs.shape[0])

    assert total_items == dataset_size


@pytest.mark.parametrize(
    ("dosing_mode", "loader_kwargs"),
    [
        ("repeated_dosing", {}),
        ("dosing_list", {"dosing_list_generation": "dosing_from_samples"}),
        ("diverse_dosing", {}),
    ],
)
def test_notebook_synthetic_experiment_dataloader_smoke_by_dosing_mode(
    notebook_dm: AICMECompartmentsDataModule,
    dosing_mode: str,
    loader_kwargs: dict,
) -> None:
    """All public sample-experiment dosing modes should work through the loader."""

    loader = notebook_dm.get_synthetic_experiment_dataloader(
        n_targets=12,
        n_dosings=NOTEBOOK_N_DOSINGS,
        dosing_mode=dosing_mode,
        dataset_size=2,
        **loader_kwargs,
    )

    batch_list = next(iter(loader))

    assert isinstance(batch_list, list)
    assert len(batch_list) == NOTEBOOK_N_DOSINGS
    assert all(isinstance(batch, AICMECompartmentsDataBatch) for batch in batch_list)
    assert all(batch.context_obs.shape[0] == notebook_dm.batch_size for batch in batch_list)
    assert all(batch.target_obs.shape[0] == notebook_dm.batch_size for batch in batch_list)
