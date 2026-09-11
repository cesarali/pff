"""Forward pass smoke test for :class:`NewContextVAEPK`."""

import math
from dataclasses import replace
from pathlib import Path

import pytest

from pff import config_dir
from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.data.datasets.aicme_batch import AICMECompartmentsDataBatch
from pff.data.datasets.aicme_datasets import AICMECompartmentsDataModule
from pff.metrics.quantiles_coverage import compute_percentile_coverage
from pff.models.amortized_inference.context_vae_pk import (
    ContextVAEForwardOutputs,
    ContextVAEPK,
)
from pff.models.utils.new_pk_callbacks import NewPKValidationVisualizationCallback
from pff.utils.plots.databatch_plot import plot_list_list_study_json
from pff.utils.tensors_operations import gather_distinct_times_per_substance
from tests.helpers import DummyExperiment, DummyLogger, DummyTrainer
from tests.models.test_aicme_pk import _first_batch_list


def _cvae_config_from_file() -> NodePKExperimentConfig:
    default_yaml = Path(config_dir) / "experiment_configs" / "node-pk" / "cvaeNoDose.yaml"
    cfg: NodePKExperimentConfig = NodePKExperimentConfig.from_yaml(str(default_yaml))
    return cfg


def _cvae_config() -> NodePKExperimentConfig:
    """Return a minimal but representative configuration for AICMEPK tests.

    - Keeps everything small and deterministic.
    """
    cfg = NodePKExperimentConfig()
    cfg.train = replace(
        cfg.train,
        batch_size=4,
        num_workers=0,
        persistent_workers=False,
        epochs=1,
    )
    cfg.mix_data = replace(
        cfg.mix_data,
        train_size=2,
        val_size=2,
        test_size=2,
        n_of_permutations=3,  # >1 to exercise permutation aggregation
        n_of_target_individuals=1,
        test_empirical_datasets=["cesarali/lenuzza-2016", "cesarali/Indometacin"],
    )
    # cesarali/Indometacin
    cfg.meta_study = replace(cfg.meta_study, num_individuals_range=(3, 3))
    cfg.network = replace(cfg.network, aggregator_type="mean")

    cfg.target_observations = replace(
        cfg.target_observations,
        split_past_future=False,
        min_past=3,
        max_past=5,
        max_num_obs=10,
    )

    cfg.context_observations = replace(
        cfg.target_observations,
        split_past_future=False,
        max_num_obs=10,
    )

    return cfg


def test_cvae_forward_pass() -> None:
    """Context-VAE forward pass yields aggregated losses and reconstruction heads."""

    cfg = _cvae_config_from_file()
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)
    model = ContextVAEPK(cfg)
    outputs = model(batch_list)

    # Structural checks
    assert isinstance(outputs, ContextVAEForwardOutputs)
    losses = outputs.to_dict()
    for key in ["recon_loss", "kl_s", "kl_init", "init_rmse"]:
        assert key in losses

    # Head checks
    head = outputs.heads["reconstruction"]
    for key in ["mean", "logvar", "target", "mask"]:
        assert key in head
        assert head[key] is not None
    assert head["mean"].shape[-1] == 1


@pytest.mark.skip("From config samples are slow")
def test_cvae_from_file():
    experiment_dir = config_dir / "experiment_configs" / "node-pk"
    full_yaml_path = experiment_dir / "cvaeNoDose.yaml"
    cfg = NodePKExperimentConfig.from_yaml(full_yaml_path)
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)
    model = ContextVAEPK(cfg)
    outputs = model(batch_list)
    # Structural checks
    assert isinstance(outputs, ContextVAEForwardOutputs)


def test_cvae_sample():
    sample_size = 20
    cfg = _cvae_config_from_file()
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)
    batch: AICMECompartmentsDataBatch = batch_list[0]
    model = ContextVAEPK(cfg)
    pred_values, pred_times, mask = model.sample_new_individual(batch, sample_size=sample_size)
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


def test_cvae_rejects_target_resolved_sampling():
    """ContextVAE exposes the kwarg but does not implement target-resolved mode."""

    cfg = _cvae_config_from_file()
    dm = AICMECompartmentsDataModule(cfg)
    batch = _first_batch_list(dm)[0]
    model = ContextVAEPK(cfg)

    with pytest.raises(ValueError, match="target-resolved sampling"):
        model.sample_new_individual(batch, resolve_sampling_from_target=True)


def test_cvae_sample_plot():
    """Generate and save sample plots for visual inspection."""
    cfg = _cvae_config_from_file()
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)
    model = ContextVAEPK(cfg)

    # gather decode times and generate new individuals
    decode_times = gather_distinct_times_per_substance(batch_list[0]) if batch_list else None
    studies = model.sample_new_individuals_from_batchlist_to_study_json(
        batch_list,
        decode_times=decode_times,
        sample_size=20,
    )
    assert studies, "No studies generated for plotting"
    from pff import reports_dir

    save_dir = reports_dir / "test" / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    requested_out_path = save_dir / "cvae_sample_plot.png"
    saved_out_path = save_dir / "cvae_sample_plot.png"

    returned_path = plot_list_list_study_json(studies=studies, file_name=str(requested_out_path))
    assert returned_path == str(saved_out_path)
    assert saved_out_path.exists(), f"Plot file not found: {saved_out_path}"


def test_cvae_logs_new_individuals(tmp_path: Path):
    """`_render_new_individuals_images` records coverage metrics and images."""

    cfg = _cvae_config_from_file()
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)
    model = ContextVAEPK(cfg)

    dummy_experiment = DummyExperiment(datamodule=dm)
    dummy_logger = DummyLogger(dummy_experiment)
    dummy_trainer = DummyTrainer(dummy_logger, datamodule=dm)
    torch = pytest.importorskip("torch")

    epoch, batch_idx = 2, 3
    repo_id = "Repo-001"
    torch.manual_seed(0)
    output_root = tmp_path / "training_images"
    output_root.mkdir(parents=True, exist_ok=True)
    callback = NewPKValidationVisualizationCallback(model_label="cvae")
    callback._render_new_individuals_images(
        model,
        batch_list,
        label="Synthetic",
        epoch=epoch,
        step=epoch * 1000 + batch_idx,
        repo_id=repo_id,
        output_root=output_root,
        trainer=dummy_trainer,
    )

    assert dummy_experiment.logged_metrics, "No metrics were logged"
    batch_substances = batch_list[0].substance_name
    metric_names = ("coverage", "interval_score")
    expected_metric_names = {
        f"Synthetic/{repo_id}/{substance}/{metric_name}"
        for substance in batch_substances
        for metric_name in metric_names
    }
    for metric_name in metric_names:
        expected_metric_names.add(f"Synthetic/{repo_id}/mean/{metric_name}")
        expected_metric_names.add(f"Synthetic/{repo_id}/std/{metric_name}")
    logged_metric_names = {metric["name"] for metric in dummy_experiment.logged_metrics}

    assert expected_metric_names.issubset(logged_metric_names)

    for logged_metric in dummy_experiment.logged_metrics:
        parts = logged_metric["name"].split("/")
        assert len(parts) == 4
        assert parts[:2] == ["Synthetic", repo_id]
        assert parts[2] in set(batch_substances) | {"mean", "std"}
        assert parts[3] in set(metric_names)
        assert logged_metric["step"] == epoch * 1000 + batch_idx
    logged_image = dummy_experiment.logged_images[0]
    assert Path(logged_image["path"]).exists()
    assert logged_image["name"].startswith("Synthetic/NewIndividuals_E002")
    assert logged_image["step"] == epoch * 1000 + batch_idx


if __name__ == "__main__":
    test_cvae_sample()
    test_cvae_sample_plot()
