# Unit testing of NPDE calculation and VPC plotting

from dataclasses import replace
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from pff import config_dir, reports_dir
from pff.config_classes.data_config import MetaDosingConfig
from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.config_classes.flow_pk_config import FlowPKExperimentConfig
from pff.data.data_empirical.builder import databatch_to_study_jsons
from pff.data.data_empirical.json_schema import IndividualJSON, StudyJSON
from pff.data.datasets.aicme_batch import AICMECompartmentsDataBatch
from pff.data.datasets.aicme_datasets import AICMECompartmentsDataModule
from pff.metrics.sampling_quality import (
    compute_npde_data,
    compute_vpc_data,
    json_list_to_dataframe,
    json_to_dataframe,
    npde_pvalues,
    validate_npde_vpc_inputs,
    vpc_plot,
)
from pff.models.amortized_inference.aicme import AICMEPK
from pff.models.amortized_inference.flows_pk import FlowPK
from pff.training.callbacks.pk_tasks import (
    VPCBundle,
    _compute_vpc_npde_pvalues_per_substance,
    _vpc_images_per_batch,
    _vpc_npde_pvalues_per_batch,
    sample_vpc_bundle,
)


def _make_small_empirical_batch() -> AICMECompartmentsDataBatch:
    """Build a tiny empirical batch compatible with VPC conversion helpers."""

    B, c_ind, t_ind, T = 1, 2, 1, 3
    context_obs = torch.tensor([[[[10.0], [20.0], [30.0]], [[12.0], [24.0], [36.0]]]])
    context_time = torch.tensor([[[[0.0], [1.0], [2.0]], [[0.0], [1.0], [2.0]]]])
    context_mask = torch.ones(B, c_ind, T, dtype=torch.bool)

    return AICMECompartmentsDataBatch(
        target_obs=torch.zeros(B, t_ind, 1, 1),
        target_obs_time=torch.zeros(B, t_ind, 1, 1),
        target_obs_mask=torch.zeros(B, t_ind, 1, dtype=torch.bool),
        target_rem_sim=torch.zeros(B, t_ind, 0, 1),
        target_rem_sim_time=torch.zeros(B, t_ind, 0, 1),
        target_rem_sim_mask=torch.zeros(B, t_ind, 0, dtype=torch.bool),
        context_obs=context_obs,
        context_obs_time=context_time,
        context_obs_mask=context_mask,
        context_rem_sim=torch.zeros(B, c_ind, 0, 1),
        context_rem_sim_time=torch.zeros(B, c_ind, 0, 1),
        context_rem_sim_mask=torch.zeros(B, c_ind, 0, dtype=torch.bool),
        target_dosing_amounts=torch.zeros(B, t_ind),
        target_dosing_route_types=torch.zeros(B, t_ind, dtype=torch.long),
        context_dosing_amounts=torch.tensor([[100.0, 120.0]]),
        context_dosing_route_types=torch.tensor([[0, 1]], dtype=torch.long),
        mask_context_individuals=torch.ones(B, c_ind, dtype=torch.bool),
        mask_target_individuals=torch.zeros(B, t_ind, dtype=torch.bool),
        study_name=["study_small"],
        context_subject_name=[["ctx_0", "ctx_1"]],
        target_subject_name=[["tgt_0"]],
        substance_name=["warfarin"],
        time_scales=torch.zeros(B, 2),
        is_empirical=True,
    )


def _replicate_for_vpc(
    observed_studies: List[StudyJSON], n_replicates: int
) -> List[List[StudyJSON]]:
    """Create simulation replicates preserving VPC structure keys and schedules."""

    by_substance: List[List[StudyJSON]] = []
    for observed in observed_studies:
        replicas: List[StudyJSON] = []
        for r in range(n_replicates):
            context: List[IndividualJSON] = []
            for ind in observed.get("context", []):
                shifted = [float(v) + 0.5 * float(r) for v in ind["observations"]]
                ind_sim: IndividualJSON = {
                    "observations": shifted,
                    "observation_times": list(ind["observation_times"]),
                }
                for optional_key in (
                    "name_id",
                    "dosing",
                    "dosing_type",
                    "dosing_times",
                    "dosing_name",
                ):
                    if optional_key in ind:
                        ind_sim[optional_key] = ind[optional_key]
                context.append(ind_sim)

            replicas.append(
                {
                    "context": context,
                    "target": [],
                    "meta_data": dict(observed.get("meta_data", {})),
                }
            )
        by_substance.append(replicas)
    return by_substance


class _DummyVPCSampler:
    """Stub model exposing only the API required by ``sample_vpc_bundle``."""

    def __init__(
        self,
        *,
        meta_dosing: MetaDosingConfig,
        studies_by_substance: List[List[StudyJSON]],
    ) -> None:
        self.meta_dosing = meta_dosing
        self._studies_by_substance = studies_by_substance
        self.last_batch: AICMECompartmentsDataBatch | None = None
        self.last_sample_size: int | None = None

    def sample_new_individuals_to_vpc_format(
        self, db: AICMECompartmentsDataBatch, sample_size: int = 8, num_steps: int = None
    ) -> List[List[StudyJSON]]:
        _ = num_steps
        self.last_batch = db
        self.last_sample_size = sample_size
        return self._studies_by_substance


def _tiny_aicme_config() -> NodePKExperimentConfig:
    """Build a compact config for fast local VPC smoke tests."""

    cfg = NodePKExperimentConfig()
    cfg.train = replace(
        cfg.train,
        batch_size=1,
        num_workers=0,
        persistent_workers=False,
    )
    cfg.mix_data = replace(
        cfg.mix_data,
        train_size=2,
        val_size=1,
        test_size=1,
        n_of_permutations=1,
        n_of_target_individuals=1,
        test_empirical_datasets=[],
    )
    cfg.meta_study = replace(cfg.meta_study, num_individuals_range=(3, 3))
    cfg.network = replace(cfg.network, aggregator_type="mean")
    cfg.target_observations = replace(
        cfg.target_observations,
        split_past_future=True,
        min_past=2,
        max_past=4,
        max_num_obs=8,
    )
    cfg.context_observations = replace(
        cfg.target_observations,
        split_past_future=False,
        max_num_obs=10,
    )
    return cfg


def _aicme_aistats_config_for_empirical_vpc() -> NodePKExperimentConfig:
    """Load AISTATS AICME config and shrink it for empirical VPC tests."""

    base_yaml = Path(config_dir) / "experiment_configs" / "AISTATS" / "aicme-t-pk" / "base.yaml"
    cfg = NodePKExperimentConfig.from_yaml(str(base_yaml))

    cfg.train = replace(
        cfg.train,
        batch_size=1,
        num_workers=0,
        persistent_workers=False,
        epochs=1,
    )
    cfg.mix_data = replace(
        cfg.mix_data,
        train_size=2,
        val_size=1,
        test_size=1,
        n_of_permutations=1,
        n_of_target_individuals=1,
        test_empirical_datasets=["cesarali/lenuzza-2016"],
        store_in_tempfile=False,
        keep_tempfile=False,
        recreate_tempfile=False,
    )
    cfg.meta_study = replace(cfg.meta_study, num_individuals_range=(3, 3))
    cfg.network = replace(
        cfg.network,
        aggregator_type="mean",
        zi_latent_dim=64,
        encoder_rnn_hidden_dim=64,
        decoder_hidden_dim=64,
        decoder_rnn_hidden_dim=64,
        time_obs_encoder_hidden_dim=64,
        time_obs_encoder_output_dim=64,
        input_encoding_hidden_dim=64,
        cov_proj_dim=8,
    )
    return cfg


def _flowpk_uai_config_for_empirical_vpc() -> FlowPKExperimentConfig:
    """Load AISTATS AICME config and shrink it for empirical VPC tests."""

    base_yaml = (
        Path(config_dir)
        / "experiment_configs"
        / "UAI"
        / "flow-pk-predict-n-generate-test"
        / "flowPK.yaml"
    )
    cfg = FlowPKExperimentConfig.from_yaml(str(base_yaml))
    cfg.train = replace(
        cfg.train,
        batch_size=1,
        num_workers=0,
        persistent_workers=False,
        epochs=1,
    )
    cfg.mix_data = replace(
        cfg.mix_data,
        train_size=2,
        val_size=1,
        test_size=1,
        n_of_permutations=1,
        n_of_target_individuals=1,
        store_in_tempfile=False,
        keep_tempfile=False,
        recreate_tempfile=False,
    )
    cfg.meta_study = replace(cfg.meta_study, num_individuals_range=(3, 3))

    return cfg


def _aicme_potsdam_config_for_synthetic_vpc() -> NodePKExperimentConfig:
    """Load the native Potsdam AICME config used by the synthetic VPC tests."""

    base_yaml = (
        Path(config_dir)
        / "experiment_configs"
        / "UAI"
        / "Rebuttal"
        / "Potsdam"
        / "aicme-t-pk"
        / "base.yaml"
    )
    return NodePKExperimentConfig.from_yaml(str(base_yaml))


def _aicme_potsdam_datamodule_for_synthetic_vpc() -> AICMECompartmentsDataModule:
    """Build the canonical Potsdam datamodule fixture for native synthetic VPC tests."""

    cfg = _aicme_potsdam_config_for_synthetic_vpc()
    dm = AICMECompartmentsDataModule(cfg)
    dm.prepare_data()
    dm.setup()
    return dm


def test_compute_npde_data():
    # Create a simple StudyJSON object for testing
    study_json = StudyJSON(
        context=[
            IndividualJSON(name_id="ID1", observation_times=[0, 1, 2], observations=[10, 20, 30])
        ],
        target=[
            IndividualJSON(name_id="ID1", observation_times=[0, 1, 2], observations=[12, 18, 28])
        ],
    )  # type: ignore

    # Calculate NPDE
    npde_results = compute_npde_data(study_json, [study_json])

    assert len(npde_results) == 6, "NPDE results should have the same length as observations"
    assert all(npde == 0 for npde in npde_results), "NPDE results should be 0 for this simple case"


def test_validate_npde_vpc_inputs():
    # Create a simple StudyJSON object for testing
    study_json = StudyJSON(
        context=[
            IndividualJSON(
                name_id="ID1", observation_times=[0, 1, 2], observations=[10, 20, 30], dosing=[100]
            ),
            IndividualJSON(
                name_id="ID2", observation_times=[0, 1, 3], observations=[10, 20, 30], dosing=[200]
            ),
        ]
    )  # type: ignore

    study_df1 = json_to_dataframe(study_json)
    study_df2 = json_list_to_dataframe([study_json])

    # Test NPDE error handling for different doses and times
    assert validate_npde_vpc_inputs(study_df1, study_df2, differentTimesError=False) is None, (  # type: ignore
        "Should not raise error when differentTimesError is False"
    )

    try:
        validate_npde_vpc_inputs(study_df1, study_df2, differentTimesError=True)  # type: ignore
    except ValueError as e:
        assert str(e) == "Observation times differ between individuals.", (
            "Should raise error for different observation times"
        )


def test_compute_vpc_data():
    # Create observed StudyJSON
    observed_data = StudyJSON(
        context=[
            IndividualJSON(name_id="1", observation_times=[0, 1, 2], observations=[10, 20, 30]),
            IndividualJSON(name_id="2", observation_times=[0, 1, 2], observations=[11, 21, 31]),
        ]
    )  # type: ignore

    # Create simulated StudyJSONs
    simulated_data = [
        StudyJSON(
            context=[
                IndividualJSON(name_id="1", observation_times=[0, 1, 2], observations=[12, 22, 32]),
                IndividualJSON(name_id="2", observation_times=[0, 1, 2], observations=[13, 21, 30]),
            ]
        ),  # type: ignore
        StudyJSON(
            context=[
                IndividualJSON(name_id="1", observation_times=[0, 1, 2], observations=[8, 18, 28]),
                IndividualJSON(name_id="2", observation_times=[0, 1, 2], observations=[11, 19, 27]),
            ]
        ),  # type: ignore
    ]

    # Calculate VPC data
    vpc_data = compute_vpc_data(observed_data, simulated_data)

    assert not vpc_data.empty, "VPC data should not be empty"
    assert all(
        col in vpc_data.columns for col in ["Time", "Quantile", "Obs", "LowerPred", "UpperPred"]
    ), "VPC data should contain required columns"


def test_sample_vpc_bundle_builds_expected_inputs():
    """`sample_vpc_bundle` should compose observed and simulated VPC studies."""

    batch = _make_small_empirical_batch()
    meta_dosing = replace(MetaDosingConfig(), route_options=["oral", "iv"], time=12.0)
    observed_studies = databatch_to_study_jsons(batch, meta_dosing)
    simulated_studies = _replicate_for_vpc(observed_studies, n_replicates=4)

    model = _DummyVPCSampler(meta_dosing=meta_dosing, studies_by_substance=simulated_studies)
    bundle = sample_vpc_bundle(model, batch)

    assert isinstance(bundle, VPCBundle)
    assert bundle.observed_studies == observed_studies
    assert bundle.simulated_studies_by_substance == simulated_studies
    assert model.last_batch is batch
    assert model.last_sample_size == 4


def test_generate_synthetic_vpc_data_list_return_shape() -> None:
    """The native synthetic VPC API should return ``(StudyJSON, list[StudyJSON])`` cases."""

    torch.manual_seed(7)
    dm = _aicme_potsdam_datamodule_for_synthetic_vpc()
    output_dir = Path(reports_dir) / "test" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_cases = 10
    sample_size = 50
    n_observed_individuals = 10
    vpc_cases = dm.generate_synthetic_vpc_data_list(
        n_cases=n_cases,
        n_observed_individuals=n_observed_individuals,
        sample_size=sample_size,
    )

    assert len(vpc_cases) == n_cases
    for case_index, (observed_study, simulated_replicates) in enumerate(vpc_cases):
        assert isinstance(observed_study, dict)
        assert observed_study["target"] == []
        assert len(observed_study["context"]) == n_observed_individuals
        assert isinstance(simulated_replicates, list)
        assert len(simulated_replicates) == sample_size
        assert all(study["target"] == [] for study in simulated_replicates)

        vpc_data = compute_vpc_data(observed_study, simulated_replicates)
        fig, ax = plt.subplots(figsize=(6, 4))
        vpc_plot(vpc_data, ax=ax, log_y=False)
        image_path = output_dir / f"synthetic_vpc_return_shape_case_{case_index:03d}.png"
        fig.savefig(image_path, bbox_inches="tight")
        plt.close(fig)

        assert image_path.exists()
        assert image_path.stat().st_size > 0


def test_generate_synthetic_vpc_data_list_supports_native_vpc_computation() -> None:
    """Generated synthetic VPC cases should run through ``compute_vpc_data(...)``."""

    torch.manual_seed(11)
    dm = _aicme_potsdam_datamodule_for_synthetic_vpc()

    vpc_cases = dm.generate_synthetic_vpc_data_list(
        n_cases=2,
        n_observed_individuals=3,
        sample_size=4,
    )

    for observed_study, simulated_replicates in vpc_cases:
        vpc_data = compute_vpc_data(observed_study, simulated_replicates)
        assert not vpc_data.empty


def test_generate_synthetic_vpc_data_list_supports_native_vpc_plotting() -> None:
    """Generated synthetic VPC cases should run through ``vpc_plot(...)``."""

    torch.manual_seed(13)
    dm = _aicme_potsdam_datamodule_for_synthetic_vpc()

    vpc_cases = dm.generate_synthetic_vpc_data_list(
        n_cases=2,
        n_observed_individuals=3,
        sample_size=4,
    )

    for observed_study, simulated_replicates in vpc_cases:
        vpc_data = compute_vpc_data(observed_study, simulated_replicates)
        fig, ax = plt.subplots(figsize=(6, 4))
        vpc_plot(vpc_data, ax=ax, log_y=False)
        assert len(ax.lines) > 0
        plt.close(fig)


def test_vpc_images_per_batch_uses_substance_name_in_filename(tmp_path: Path):
    """VPC image filenames should include observed-study substance names."""

    batch = _make_small_empirical_batch()
    meta_dosing = replace(MetaDosingConfig(), route_options=["oral", "iv"], time=12.0)
    observed_studies = databatch_to_study_jsons(batch, meta_dosing)
    simulated_studies = _replicate_for_vpc(observed_studies, n_replicates=4)
    bundle = VPCBundle(
        observed_studies=observed_studies,
        simulated_studies_by_substance=simulated_studies,
    )

    image_paths = _vpc_images_per_batch(
        bundle,
        batch,
        label="Empirical",
        epoch=3,
        output_root=tmp_path,
        model_label="dummy_model",
        n_bins=3,
        binning="equal_count",
        log_y=False,
    )

    assert len(image_paths) == 1
    assert "warfarin" in Path(image_paths[0]).name.lower()


def test_vpc_npde_pvalues_per_batch_returns_finite_metrics():
    """NPDE helper should return per-substance p-values as callback metric tensors."""

    batch = _make_small_empirical_batch()
    meta_dosing = replace(MetaDosingConfig(), route_options=["oral", "iv"], time=12.0)
    observed_studies = databatch_to_study_jsons(batch, meta_dosing)
    simulated_studies = _replicate_for_vpc(observed_studies, n_replicates=6)
    bundle = VPCBundle(
        observed_studies=observed_studies,
        simulated_studies_by_substance=simulated_studies,
    )

    metrics = _vpc_npde_pvalues_per_batch(bundle, batch)
    assert set(metrics) == {"npde_pvalue_mean", "npde_pvalue_variance", "npde_pvalue_normality"}
    for metric_tensor in metrics.values():
        assert tuple(metric_tensor.shape) == (1,)
        assert torch.isfinite(metric_tensor).all()


@pytest.mark.skip("Heavy loading of data")
def test_vpc_images_per_batch_renders_image():
    """`_vpc_images_per_batch` should render from a real empirical no-heldout batch."""
    from pff import reports_dir

    cfg = _aicme_aistats_config_for_empirical_vpc()
    dm = AICMECompartmentsDataModule(cfg)
    dm.prepare_data()
    dm.setup()

    empirical_batches = dm.get_empirical_test_batches(no_heldout=True)
    if not empirical_batches:
        pytest.skip("No empirical no-heldout batches were loaded by the datamodule.")

    batch_list = next((batches for batches in empirical_batches.values() if batches), None)
    if not batch_list:
        pytest.skip("Empirical no-heldout datasets were configured but no batches were available.")

    batch = batch_list[0].to_device("cpu")
    model = AICMEPK(cfg)
    bundle = sample_vpc_bundle(model, batch)

    image_paths = _vpc_images_per_batch(
        bundle,
        batch,
        label="Empirical",
        epoch=3,
        output_root=reports_dir,
        model_label="dummy_model",
        n_bins=10,
        binning="equal_count",
        log_y=False,
    )

    assert len(image_paths) > 0
    for image_path_str in image_paths:
        image_path = Path(image_path_str)
        assert image_path.exists()
        assert image_path.stat().st_size > 0


@pytest.mark.skip("Heavy loading of data")
def test_compute_vpc_npde_pvalues_per_substance_real_empirical():
    """`_compute_vpc_npde_pvalues_per_substance` should run on a real empirical no-heldout batch."""

    cfg = _aicme_aistats_config_for_empirical_vpc()
    dm = AICMECompartmentsDataModule(cfg)
    dm.prepare_data()
    dm.setup()

    empirical_batches = dm.get_empirical_test_batches(no_heldout=True)
    if not empirical_batches:
        pytest.skip("No empirical no-heldout batches were loaded by the datamodule.")

    batch_list = next((batches for batches in empirical_batches.values() if batches), None)
    if not batch_list:
        pytest.skip("Empirical no-heldout datasets were configured but no batches were available.")

    batch = batch_list[0].to_device("cpu")
    model = AICMEPK(cfg)
    bundle = sample_vpc_bundle(model, batch)

    npde_stats_by_substance = _compute_vpc_npde_pvalues_per_substance(bundle)

    assert len(npde_stats_by_substance) > 0
    assert len(npde_stats_by_substance) == min(
        len(bundle.observed_studies), len(bundle.simulated_studies_by_substance)
    )
    assert any(np.isfinite(value) for stats in npde_stats_by_substance for value in stats.values())

    for stats in npde_stats_by_substance:
        assert set(stats.keys()) == {"mean", "variance", "normality"}
        for value in stats.values():
            assert isinstance(value, float)


def test_npde_pvalues():
    # Set seed and draw standard normal samples for NPDE values
    np.random.seed(42)
    npde_samples = np.random.normal(loc=0, scale=1, size=1000)

    # Test p-value calculations
    p_values = npde_pvalues(npde_samples)
    assert p_values["mean"] > 0.05, (
        "Mean test should not reject null hypothesis for standard normal samples"
    )
    assert p_values["variance"] > 0.05, (
        "Variance test should not reject null hypothesis for standard normal samples"
    )
    assert p_values["normality"] > 0.05, (
        "Normality test should not reject null hypothesis for standard normal samples"
    )


@pytest.mark.skip("Heavy loading of data")
def test_aicmepk_sample_to_vpc_plot():
    """Smoke-test VPC plotting from AICMEPK sampled StudyJSON replicates."""
    from pff import reports_dir

    cfg = _tiny_aicme_config()
    dm = AICMECompartmentsDataModule(cfg)
    dm.prepare_data()
    dm.setup()
    batch_list = next(iter(dm.train_dataloader()))
    db0 = batch_list[0].to_device("cpu")

    model = AICMEPK(cfg)
    studies_by_substance = model.sample_new_individuals_to_vpc_format(db0, sample_size=10)

    assert isinstance(studies_by_substance, list) and len(studies_by_substance) > 0
    vpc_data = studies_by_substance[0]
    assert len(vpc_data) >= 2

    vpc_results = compute_vpc_data(vpc_data[0], vpc_data, n_bins=10, binning="equal_count")
    assert not vpc_results.empty

    fig, ax = plt.subplots(figsize=(6, 4))
    vpc_plot(vpc_results, ax=ax, log_y=False)
    assert len(ax.lines) > 0
    plt.savefig(reports_dir / "vpc_sample_from_aicme.png")
    plt.close(fig)


@pytest.mark.skip("Heavy loading of data")
def test_aicmepk_empirical_no_heldout_batch_to_vpc_plot():
    """Smoke-test empirical no-heldout databatch -> StudyJSON -> VPC plot."""
    from pff import reports_dir

    cfg = _aicme_aistats_config_for_empirical_vpc()
    dm = AICMECompartmentsDataModule(cfg)
    dm.prepare_data()
    dm.setup()

    empirical_batches = dm.get_empirical_test_batches(no_heldout=True)
    if not empirical_batches:
        pytest.skip("No empirical no-heldout batches were loaded by the datamodule.")

    batch_list = next((batches for batches in empirical_batches.values() if batches), None)
    if not batch_list:
        pytest.skip("Empirical no-heldout datasets were configured but no batches were available.")

    db0 = batch_list[0].to_device("cpu")
    observed_studies = databatch_to_study_jsons(db0, cfg.dosing)

    model = AICMEPK(cfg)
    simulated_studies_by_substance = model.sample_new_individuals_to_vpc_format(db0, sample_size=4)

    assert len(observed_studies) == len(simulated_studies_by_substance)
    assert len(observed_studies) > 0

    vpc_data = simulated_studies_by_substance[0]
    observed = observed_studies[0]

    vpc_results = compute_vpc_data(observed, vpc_data, n_bins=10, binning="equal_count")
    assert not vpc_results.empty

    fig, ax = plt.subplots(figsize=(6, 4))
    vpc_plot(vpc_results, ax=ax, log_y=False)
    assert len(ax.lines) > 0
    plt.savefig(reports_dir / "vpc_sample_from_aicme_empirical.png")
    plt.close(fig)


if __name__ == "__main__":
    test_generate_synthetic_vpc_data_list_return_shape()
