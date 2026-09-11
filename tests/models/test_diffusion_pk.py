"""Forward pass smoke tests for diffusion-based PK models."""

from pathlib import Path

from pff import config_dir, reports_dir
from pff.config_classes.diffusion_pk_config import DiffusionPKExperimentConfig
from pff.data.datasets.aicme_batch import AICMECompartmentsDataBatch
from pff.data.datasets.aicme_datasets import AICMECompartmentsDataModule
from pff.metrics.quantiles_coverage import compute_percentile_coverage
from pff.models.amortized_inference.diffusion_pk import (
    ContinuousDiffusionPK,
    DiffusionForwardOutputs,
    DiscreteDiffusionPK,
)
from pff.utils.plots.databatch_plot import plot_list_list_study_json
from pff.utils.tensors_operations import gather_distinct_times_per_substance
from tests.models.test_aicme_pk import _first_batch_list


def _diffusion_pk_config_from_file(diffusion_type: str = "continuous") -> DiffusionPKExperimentConfig:
    """
    Load the NJ elephant diffusion config aligned with the FlowPK vector field.

    The diffusion models are generative-only, but use the same point-cloud
    vector-field architecture and data handling as the current FlowPK model.
    """
    default_yaml = (
        Path(config_dir)
        / "experiment_configs"
        / "UAI"
        / "Rebuttal"
        / "NJ"
        / "diffusionPK_elephant"
        / "diffusionPK.yaml"
    )
    cfg: DiffusionPKExperimentConfig = DiffusionPKExperimentConfig.from_yaml(str(default_yaml))
    cfg.diffusion_type = diffusion_type
    if diffusion_type == "discrete":
        cfg.name_str = "DiscreteDiffusionPK"
    return cfg


# ----------------------------------------------------------------------
# Discrete diffusion tests
# ----------------------------------------------------------------------


def test_discrete_diffusion_pk_from_file():
    """Smoke test: instantiate DiscreteDiffusionPK and run a forward pass."""
    cfg = _diffusion_pk_config_from_file("discrete")
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)
    model = DiscreteDiffusionPK(cfg)
    assert not callable(getattr(model, "sample_individual_prediction", None))
    outputs = model(batch_list)

    # Structural checks
    assert isinstance(outputs, DiffusionForwardOutputs)


def test_discrete_diffusion_pk_sample():
    """Sampling smoke test for DiscreteDiffusionPK."""
    sample_size = 2  # number of new individuals to sample
    num_steps = 10  # kept for API compatibility, not used by diffusion

    cfg = _diffusion_pk_config_from_file("discrete")
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)
    batch: AICMECompartmentsDataBatch = batch_list[0]

    model = DiscreteDiffusionPK(cfg)
    pred_values, pred_times, mask = model.sample_new_individual(
        batch,
        sample_size=sample_size,
        num_steps=num_steps,
    )
    # pred_values: [S, B, T, 1] -> [B, S, T, 1]
    pred_values = pred_values.transpose(0, 1)

    metrics = compute_percentile_coverage(
        pred_values,
        pred_times,
        mask,
        batch.context_obs,
        batch.context_obs_time,
        batch.context_obs_mask,
    )

    print(metrics)
    assert pred_values.shape[1] == sample_size


def test_discrete_diffusion_pk_target_resolved_sampling():
    """Discrete diffusion supports target-schedule generative sampling."""

    cfg = _diffusion_pk_config_from_file("discrete")
    dm = AICMECompartmentsDataModule(cfg)
    batch = _first_batch_list(dm)[0]
    model = DiscreteDiffusionPK(cfg)

    samples, times, mask = model.sample_new_individual(
        batch,
        resolve_sampling_from_target=True,
        include_rem=True,
    )
    assert samples.shape[0] == batch.target_obs.shape[1]
    assert samples.shape[1] == batch.target_obs.shape[0]
    assert times.shape[:2] == mask.shape


def test_discrete_diffusion_pk_sample_plot():
    """Generate and save sample plots for DiscreteDiffusionPK for visual inspection."""
    cfg = _diffusion_pk_config_from_file("discrete")
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)
    model = DiscreteDiffusionPK(cfg)

    # gather decode times and generate new individuals
    decode_times = gather_distinct_times_per_substance(batch_list[0]) if batch_list else None
    studies = model.sample_new_individuals_from_batchlist_to_study_json(
        batch_list,
        decode_times=decode_times,
        num_steps=10,
        sample_size=1,
    )
    assert studies, "No studies generated for plotting"

    save_dir = reports_dir / "test" / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    requested_out_path = save_dir / "discrete_diffusion_pk_sample_plot.png"
    saved_out_path = save_dir / "discrete_diffusion_pk_sample_plot.png"

    returned_path = plot_list_list_study_json(studies=studies, file_name=str(requested_out_path))
    assert returned_path == str(saved_out_path)
    assert saved_out_path.exists(), f"Plot file not found: {saved_out_path}"


# ----------------------------------------------------------------------
# Continuous diffusion tests
# ----------------------------------------------------------------------


def test_continuous_diffusion_pk_from_file():
    """Smoke test: instantiate ContinuousDiffusionPK and run a forward pass."""
    cfg = _diffusion_pk_config_from_file()
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)
    model = ContinuousDiffusionPK(cfg)
    assert not callable(getattr(model, "sample_individual_prediction", None))
    outputs = model(batch_list)

    # Structural checks
    assert isinstance(outputs, DiffusionForwardOutputs)


def test_continuous_diffusion_pk_forward_reconstruction():
    """Directly test `_forward_reconstruction` for ContinuousDiffusionPK."""
    cfg = _diffusion_pk_config_from_file()
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)
    batch: AICMECompartmentsDataBatch = batch_list[0]

    model = ContinuousDiffusionPK(cfg)
    outputs = model._forward_reconstruction(batch)

    assert isinstance(outputs, DiffusionForwardOutputs)
    assert "reconstruction" in outputs.heads
    assert "prediction" in outputs.heads["reconstruction"]
    assert "target" in outputs.heads["reconstruction"]
    assert "rmse" in outputs.losses["reconstruction"]
    assert outputs.losses["reconstruction"]["rmse"].isfinite().all()


def test_continuous_diffusion_pk_sample():
    """Sampling smoke test for ContinuousDiffusionPK."""
    sample_size = 2  # number of new individuals to sample
    num_steps = 10  # kept for API compatibility, not used by diffusion

    cfg = _diffusion_pk_config_from_file()
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)
    batch: AICMECompartmentsDataBatch = batch_list[0]

    model = ContinuousDiffusionPK(cfg)
    pred_values, pred_times, mask = model.sample_new_individual(
        batch,
        sample_size=sample_size,
        num_steps=num_steps,
    )
    # pred_values: [S, B, T, 1] -> [B, S, T, 1]
    pred_values = pred_values.transpose(0, 1)

    metrics = compute_percentile_coverage(
        pred_values,
        pred_times,
        mask,
        batch.context_obs,
        batch.context_obs_time,
        batch.context_obs_mask,
    )

    print(metrics)
    assert pred_values.shape[1] == sample_size


def test_continuous_diffusion_pk_target_resolved_sampling():
    """Continuous diffusion supports target-schedule generative sampling."""

    cfg = _diffusion_pk_config_from_file()
    dm = AICMECompartmentsDataModule(cfg)
    batch = _first_batch_list(dm)[0]
    model = ContinuousDiffusionPK(cfg)

    samples, times, mask = model.sample_new_individual(
        batch,
        resolve_sampling_from_target=True,
        include_rem=True,
    )
    assert samples.shape[0] == batch.target_obs.shape[1]
    assert samples.shape[1] == batch.target_obs.shape[0]
    assert times.shape[:2] == mask.shape


def test_continuous_diffusion_pk_sample_plot():
    """Generate and save sample plots for ContinuousDiffusionPK for visual inspection."""
    cfg = _diffusion_pk_config_from_file()
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)
    model = ContinuousDiffusionPK(cfg)

    # gather decode times and generate new individuals
    decode_times = gather_distinct_times_per_substance(batch_list[0]) if batch_list else None
    studies = model.sample_new_individuals_from_batchlist_to_study_json(
        batch_list,
        decode_times=decode_times,
        num_steps=10,
        sample_size=1,
    )
    assert studies, "No studies generated for plotting"

    save_dir = reports_dir / "test" / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    requested_out_path = save_dir / "continuous_diffusion_pk_sample_plot.png"
    saved_out_path = save_dir / "continuous_diffusion_pk_sample_plot.png"

    returned_path = plot_list_list_study_json(studies=studies, file_name=str(requested_out_path))
    assert returned_path == str(saved_out_path)
    assert saved_out_path.exists(), f"Plot file not found: {saved_out_path}"


if __name__ == "__main__":
    # Allow running the tests directly for quick local smoke checks
    # test_discrete_diffusion_pk_from_file()
    # test_discrete_diffusion_pk_sample()
    # test_discrete_diffusion_pk_sample_plot()
    test_continuous_diffusion_pk_from_file()
    # test_continuous_diffusion_pk_sample()
    # test_continuous_diffusion_pk_sample_plot()
    # test_continuous_diffusion_pk_forward_reconstruction()
