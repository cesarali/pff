"""Smoke tests for synthetic-experiment sampling used by MMD workflows."""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from pff import config_dir, reports_dir
from pff.config_classes.data_config import SimpleMetaStudyConfig
from pff.config_classes.flow_pk_config import FlowPKExperimentConfig
from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.data.datasets.aicme_batch import AICMECompartmentsDataBatch
from pff.training.basic_experiment import BasicLightningExperiment
from pff.training.callbacks.pk_task_synthetic import (
    _collect_diverse_synthetic_experiment_sample_distances,
    task_diverse_synthetic_experiment_sample_distances,
)
from pff.training.callbacks.pk_tasks import (
    validate_generated_samples_match_target_schedule,
)
from pff.utils.plots.databatch_plot import plot_synthetic_mmd_overlay

EXPERIMENT_DIR = Path(config_dir) / "experiment_configs" / "UAI" / "Submission" / "flow-pk-generate"
EXPERIMENT_YAML = EXPERIMENT_DIR / "flowPK.yaml"
SIMPLE_META_STUDY_YAML = EXPERIMENT_DIR / "simple.meta_study.yaml"
AICME_EXPERIMENT_DIR = Path(config_dir) / "experiment_configs" / "UAI" / "Submission" / "aicme-t-pk"
AICME_EXPERIMENT_YAML = AICME_EXPERIMENT_DIR / "base.yaml"
KSIG_PYTHON = Path("/home/cesarali/miniconda3/envs/ksig/bin/python")

# Datamodule helper: synthetic experiment dataloader ---------------------------
N_TARGETS = 40
N_DOSINGS = 2
DATASET_SIZE = 3


def _build_small_flow_generate_config(tmp_path: Path) -> FlowPKExperimentConfig:
    """Load the real FlowPK config and shrink it for a fast local smoke test."""

    cfg = FlowPKExperimentConfig.from_yaml(str(EXPERIMENT_YAML))
    cfg.train = replace(
        cfg.train,
        batch_size=2,
        num_workers=0,
        persistent_workers=False,
        callbacks_scheduler=None,
    )
    cfg.mix_data = replace(
        cfg.mix_data,
        train_size=4,
        val_size=2,
        test_size=2,
        n_of_permutations=1,
        test_empirical_datasets=[],
    )
    cfg.meta_study = replace(
        cfg.meta_study,
        num_individuals_range=(3, 3),
        time_num_steps=20,
    )
    cfg.upload_to_hf_hub = False
    cfg.my_results_path = str(tmp_path)
    return cfg


def _build_small_flow_generate_simple_config(tmp_path: Path) -> FlowPKExperimentConfig:
    """Load the real FlowPK config but swap in the YAML-backed simple meta-study."""

    cfg = _build_small_flow_generate_config(tmp_path)
    cfg.meta_study = replace(
        SimpleMetaStudyConfig.from_yaml(SIMPLE_META_STUDY_YAML),
        num_individuals=3,
        num_individuals_range=(3, 3),
        time_num_steps=20,
    )
    return cfg


def _build_small_aicme_generate_config(tmp_path: Path) -> NodePKExperimentConfig:
    """Load the real AICMEPK config and shrink it for a fast local smoke test."""

    cfg = NodePKExperimentConfig.from_yaml(str(AICME_EXPERIMENT_YAML))
    cfg.train = replace(
        cfg.train,
        batch_size=2,
        num_workers=0,
        persistent_workers=False,
        callbacks_scheduler=None,
    )
    cfg.mix_data = replace(
        cfg.mix_data,
        train_size=4,
        val_size=2,
        test_size=2,
        n_of_permutations=1,
        test_empirical_datasets=[],
    )
    cfg.meta_study = replace(
        cfg.meta_study,
        num_individuals_range=(3, 3),
        time_num_steps=20,
    )
    cfg.context_observations = replace(cfg.context_observations, max_num_obs=8)
    cfg.target_observations = replace(cfg.target_observations, max_num_obs=8)
    cfg.network = replace(
        cfg.network,
        time_obs_encoder_hidden_dim=64,
        time_obs_encoder_output_dim=64,
        encoder_rnn_hidden_dim=64,
        input_encoding_hidden_dim=64,
        zi_latent_dim=32,
        decoder_num_layers=1,
        decoder_attention_layers=1,
        decoder_hidden_dim=64,
        decoder_rnn_hidden_dim=64,
        rnn_decoder_number_of_layers=1,
        rnn_individual_encoder_number_of_layers=1,
        init_hidden_num_layers=1,
        output_head_num_layers=1,
        drift_num_layers=1,
        aggregator_type="mean",
        use_self_attention=False,
    )
    cfg.upload_to_hf_hub = False
    cfg.my_results_path = str(tmp_path)
    return cfg


def _build_small_aicme_generate_simple_config(tmp_path: Path) -> NodePKExperimentConfig:
    """Load the AICMEPK config but replace the meta-study with simple synthetic YAML."""

    cfg = _build_small_aicme_generate_config(tmp_path)
    cfg.meta_study = replace(
        SimpleMetaStudyConfig.from_yaml(SIMPLE_META_STUDY_YAML),
        num_individuals=3,
        num_individuals_range=(3, 3),
        time_num_steps=20,
    )
    return cfg


def _setup_dummy_logger(
    self: BasicLightningExperiment,
    experiment_key: str | None = None,
) -> None:
    """Keep experiment construction local by replacing the Comet logger."""

    del experiment_key
    self.logger_folder = os.path.join(self._resolve_results_root(), "comet")
    self.logger = SimpleNamespace(version="test", experiment=SimpleNamespace())
    self._resolve_experiment_dir()


def _make_synthetic_task_batch() -> AICMECompartmentsDataBatch:
    """Return one compact synthetic batch with non-empty target remainder tensors."""

    return AICMECompartmentsDataBatch(
        target_obs=torch.tensor([[[[1.0], [2.0]]]], dtype=torch.float32),
        target_obs_time=torch.tensor([[[[1.0], [2.0]]]], dtype=torch.float32),
        target_obs_mask=torch.tensor([[[True, True]]], dtype=torch.bool),
        target_rem_sim=torch.tensor([[[[3.0], [4.0]]]], dtype=torch.float32),
        target_rem_sim_time=torch.tensor([[[[3.0], [4.0]]]], dtype=torch.float32),
        target_rem_sim_mask=torch.tensor([[[True, True]]], dtype=torch.bool),
        context_obs=torch.tensor([[[[0.5], [1.0]]]], dtype=torch.float32),
        context_obs_time=torch.tensor([[[[0.0], [1.0]]]], dtype=torch.float32),
        context_obs_mask=torch.tensor([[[True, True]]], dtype=torch.bool),
        context_rem_sim=torch.zeros(1, 1, 0, 1, dtype=torch.float32),
        context_rem_sim_time=torch.zeros(1, 1, 0, 1, dtype=torch.float32),
        context_rem_sim_mask=torch.zeros(1, 1, 0, dtype=torch.bool),
        target_dosing_amounts=torch.ones(1, 1, dtype=torch.float32),
        target_dosing_route_types=torch.zeros(1, 1, dtype=torch.long),
        context_dosing_amounts=torch.ones(1, 1, dtype=torch.float32),
        context_dosing_route_types=torch.zeros(1, 1, dtype=torch.long),
        mask_context_individuals=torch.tensor([[True]], dtype=torch.bool),
        mask_target_individuals=torch.tensor([[True]], dtype=torch.bool),
        study_name=["synthetic_study"],
        context_subject_name=[["context_0"]],
        target_subject_name=[["target_0"]],
        substance_name=["drug_x"],
        time_scales=torch.ones(1, 2, dtype=torch.float32),
        is_empirical=False,
    )


class _DummySyntheticCollectionModel:
    """Minimal model stub recording target-resolved sampling kwargs."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.calls: list[dict[str, object]] = []

    def sample_new_individual(self, batch, **kwargs):
        self.calls.append(dict(kwargs))
        generated_samples = batch.target_obs.permute(1, 0, 2, 3).contiguous() + 0.1
        generated_times = batch.target_obs_time[:, 0].clone()
        generated_mask = batch.target_obs_mask[:, 0].clone()
        return generated_samples, generated_times, generated_mask


class _DummySyntheticDatamodule:
    """Datamodule stub exposing one synthetic experiment batch list."""

    def __init__(self, batch) -> None:
        self._batch = batch

    def get_synthetic_experiment_dataloader(self, **kwargs):
        del kwargs
        return [[self._batch]]


def test_synthetic_distance_collection_uses_target_obs_times_only() -> None:
    """Synthetic distance collection should not request target remainder times."""

    batch = _make_synthetic_task_batch()
    trainer = SimpleNamespace(datamodule=_DummySyntheticDatamodule(batch))
    model = _DummySyntheticCollectionModel()

    _collect_diverse_synthetic_experiment_sample_distances(
        task_cfg={
            "plot_num_studies": 0,
            "synthetic_loader": {
                "n_targets": 1,
                "dataset_size": 1,
                "n_dosings": 1,
                "dosing_mode": "diverse_dosing",
            },
        },
        trainer=trainer,
        pl_module=model,
    )

    assert model.calls
    assert model.calls[0]["resolve_sampling_from_target"] is True
    assert model.calls[0]["include_rem"] is False


def _assert_samples_new_individuals_from_synthetic_loader(
    exp_config: FlowPKExperimentConfig | NodePKExperimentConfig,
) -> None:
    """Exercise one diverse-dosing synthetic loader batch and verify generation."""

    basic_experiment = BasicLightningExperiment.from_config(exp_config, map_location="cpu")

    dm = basic_experiment.get_datamodule()
    loader = dm.get_synthetic_experiment_dataloader(
        n_targets=N_TARGETS,
        n_dosings=N_DOSINGS,
        dosing_mode="diverse_dosing",
        dataset_size=DATASET_SIZE,
    )

    synthetic_batch_list = next(iter(loader))

    assert isinstance(synthetic_batch_list, list)
    assert len(synthetic_batch_list) == N_DOSINGS

    first_batch = synthetic_batch_list[0]
    assert first_batch.target_obs.shape[1] >= N_TARGETS
    assert torch.equal(
        first_batch.mask_target_individuals.sum(dim=1),
        torch.full(
            (first_batch.target_obs.shape[0],),
            N_TARGETS,
            dtype=torch.long,
        ),
    )

    model = basic_experiment.get_module().to(torch.device("cpu"))

    # `samples`: [It, B, T, 1], `times`: [B, T, 1], `mask`: [B, T]
    samples, times, mask = model.sample_new_individual(
        first_batch,
        sample_size=2,
        num_steps=2,
        resolve_sampling_from_target=True,
    )

    assert samples.dim() == 4
    assert samples.shape[0] == first_batch.target_obs.shape[1]
    assert samples.shape[1] == first_batch.context_obs.shape[0]
    assert samples.shape[-1] == 1
    assert times.shape == samples.shape[1:]
    assert mask.shape == samples.shape[1:3]
    assert torch.isfinite(samples).all()
    assert torch.isfinite(times).all()

    aligned = validate_generated_samples_match_target_schedule(
        first_batch,
        samples,
        times,
        mask,
    )
    metrics_reports_dir = Path(reports_dir) / "metrics"
    metrics_reports_dir.mkdir(parents=True, exist_ok=True)
    image_path = metrics_reports_dir / "synthetic_loader_overlay.png"
    returned_path = plot_synthetic_mmd_overlay(
        first_batch,
        observed_values=aligned.observed_values,
        generated_values=aligned.generated_values,
        times=aligned.times,
        mask=aligned.mask,
        num_studies=min(3, int(aligned.observed_values.shape[0])),
        file_name=str(image_path),
    )
    assert returned_path == str(image_path)
    assert image_path.exists()


def test_samples_new_individuals_from_simple_synthetic_loader_aicme() -> None:
    """An AICMEPK experiment should sample from the simple synthetic dataloader path."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch.object(BasicLightningExperiment, "_setup_logger", _setup_dummy_logger):
            _assert_samples_new_individuals_from_synthetic_loader(
                _build_small_aicme_generate_simple_config(Path(tmp_dir))
            )


@pytest.mark.skipif(not KSIG_PYTHON.exists(), reason="ksig environment is not available")
def test_task_mmd_experiment_distances_smoke_from_flow_experiment() -> None:
    """Diverse-experiment distances should run end to end on the shared target grid."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch.object(BasicLightningExperiment, "_setup_logger", _setup_dummy_logger):
            exp_config = _build_small_flow_generate_config(Path(tmp_dir))
            basic_experiment = BasicLightningExperiment.from_config(exp_config, map_location="cpu")
            dm = basic_experiment.get_datamodule()
            model = basic_experiment.get_module().to(torch.device("cpu"))
            trainer = SimpleNamespace(
                datamodule=dm,
                default_root_dir=Path(tmp_dir),
                current_epoch=0,
                global_step=0,
            )

            result = task_diverse_synthetic_experiment_sample_distances(
                samples=None,
                batches=[],
                task_cfg={
                    "distance_metrics": ["mmd2"],
                    "save_details": False,
                    "plot_num_studies": 1,
                    "num_steps": 2,
                    "mmd": {
                        "python_executable": str(KSIG_PYTHON),
                        "signature_levels": 2,
                        "estimator": "unbiased",
                        "include_time_channel": True,
                    },
                    "synthetic_loader": {
                        "n_targets": 6,
                        "dataset_size": 1,
                        "n_dosings": 1,
                        "dosing_mode": "diverse_dosing",
                    },
                },
                trainer=trainer,
                pl_module=model,
            )

            assert "mmd2" in result
            assert torch.isfinite(torch.tensor(result["mmd2"], dtype=torch.float32))
            assert "image_000" in result
            image_path = Path(result["image_000"])
            assert image_path.exists()
            assert (Path(reports_dir) / "metrics") in image_path.parents


def test_task_classifier_experiment_distances_smoke_from_flow_experiment() -> None:
    """Classifier-based diverse-experiment distance should run end to end."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch.object(BasicLightningExperiment, "_setup_logger", _setup_dummy_logger):
            exp_config = _build_small_flow_generate_config(Path(tmp_dir))
            basic_experiment = BasicLightningExperiment.from_config(exp_config, map_location="cpu")
            dm = basic_experiment.get_datamodule()
            model = basic_experiment.get_module().to(torch.device("cpu"))
            trainer = SimpleNamespace(
                datamodule=dm,
                default_root_dir=Path(tmp_dir),
                current_epoch=0,
                global_step=0,
            )

            result = task_diverse_synthetic_experiment_sample_distances(
                samples=None,
                batches=[],
                task_cfg={
                    "distance_metrics": ["classifier_auc"],
                    "save_details": False,
                    "plot_num_studies": 1,
                    "num_steps": 2,
                    "classifier_auc": {
                        "mode": "joint",
                        "include_time_channel": True,
                        "hidden_dim": 32,
                        "num_hidden_layers": 2,
                        "learning_rate": 1.0e-3,
                        "weight_decay": 1.0e-4,
                        "epochs": 10,
                        "batch_size": 32,
                        "seed": 0,
                    },
                    "synthetic_loader": {
                        "n_targets": 6,
                        "dataset_size": 1,
                        "n_dosings": 1,
                        "dosing_mode": "diverse_dosing",
                    },
                },
                trainer=trainer,
                pl_module=model,
            )

            assert "classifier_auc" in result
            assert torch.isfinite(torch.tensor(result["classifier_auc"], dtype=torch.float32))
            assert 0.0 <= float(result["classifier_auc"]) <= 1.0
            assert "image_000" in result
            image_path = Path(result["image_000"])
            assert image_path.exists()
            assert (Path(reports_dir) / "metrics") in image_path.parents


if __name__ == "__main__":
    test_task_mmd_experiment_distances_smoke_from_flow_experiment()
