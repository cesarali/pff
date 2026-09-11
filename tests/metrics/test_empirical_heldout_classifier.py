"""Tests for empirical held-out vs generated classifier evaluation."""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from pff import config_dir
from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.data.datasets.aicme_batch import AICMECompartmentsDataBatch
from pff.models.amortized_inference.aicme import AICMEPK
from pff.models.amortized_inference.flows_pk import FlowPK
from pff.training.basic_experiment import BasicLightningExperiment
from pff.training.callbacks.pk_task_empirical import (
    task_empirical_heldout_generated_classifier,
    task_empirical_summary,
)
from pff.training.callbacks.pk_tasks import (
    DiverseExperimentDistanceCollection,
    SyntheticMMDSeriesBundle,
    _collect_empirical_heldout_generated_classifier_collection,
    _reconstruct_full_target_databatch,
)

AICME_EXPERIMENT_YAML = (
    Path(config_dir) / "experiment_configs" / "UAI" / "Submission" / "aicme-t-pk" / "base.yaml"
)


def _setup_dummy_logger(
    self: BasicLightningExperiment,
    experiment_key: str | None = None,
) -> None:
    """Keep experiment construction local by replacing the Comet logger."""

    del experiment_key
    self.logger_folder = os.path.join(self._resolve_results_root(), "comet")
    self.logger = SimpleNamespace(version="test", experiment=SimpleNamespace())
    self._resolve_experiment_dir()


def _build_small_aicme_empirical_classifier_config(tmp_path: Path) -> NodePKExperimentConfig:
    """Load the real AICME config and shrink it for empirical-classifier smoke tests."""

    cfg = NodePKExperimentConfig.from_yaml(str(AICME_EXPERIMENT_YAML))
    cfg.train = replace(
        cfg.train,
        batch_size=1,
        num_workers=0,
        persistent_workers=False,
        callbacks_scheduler=None,
    )
    cfg.mix_data = replace(
        cfg.mix_data,
        train_size=1,
        val_size=1,
        test_size=1,
        n_of_permutations=2,
        n_of_target_individuals=1,
        test_empirical_datasets=["cesarali/lenuzza-2016"],
        store_in_tempfile=False,
        keep_tempfile=False,
        recreate_tempfile=False,
    )
    cfg.meta_study = replace(
        cfg.meta_study,
        num_individuals_range=(3, 3),
        time_num_steps=20,
    )
    cfg.context_observations = replace(cfg.context_observations, max_num_obs=10)
    cfg.target_observations = replace(cfg.target_observations, max_num_obs=10)
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


def _make_empirical_batch(
    *,
    observed_offset: float,
    study_name: str = "study_a",
    substance_name: str = "Drug_A",
) -> AICMECompartmentsDataBatch:
    """Return one minimal held-out empirical batch with split past/future target."""

    target_obs = torch.tensor(
        [[[[1.0 + observed_offset], [2.0 + observed_offset]]]],
        dtype=torch.float32,
    )  # [B=1, It=1, To=2, 1]
    target_obs_time = torch.tensor(
        [[[[0.0], [1.0]]]],
        dtype=torch.float32,
    )  # [1, 1, 2, 1]
    target_obs_mask = torch.tensor([[[True, True]]], dtype=torch.bool)  # [1, 1, 2]

    target_rem_sim = torch.tensor(
        [[[[3.0 + observed_offset], [4.0 + observed_offset]]]],
        dtype=torch.float32,
    )  # [1, 1, 2, 1]
    target_rem_sim_time = torch.tensor(
        [[[[2.0], [3.0]]]],
        dtype=torch.float32,
    )  # [1, 1, 2, 1]
    target_rem_sim_mask = torch.tensor([[[True, True]]], dtype=torch.bool)  # [1, 1, 2]

    context_obs = torch.tensor(
        [[[[0.5], [0.75]]]],
        dtype=torch.float32,
    )  # [1, 1, 2, 1]
    context_obs_time = torch.tensor(
        [[[[0.0], [1.0]]]],
        dtype=torch.float32,
    )  # [1, 1, 2, 1]
    context_obs_mask = torch.tensor([[[True, True]]], dtype=torch.bool)  # [1, 1, 2]

    empty_context_rem = torch.zeros(1, 1, 0, 1, dtype=torch.float32)  # [1, 1, 0, 1]
    empty_context_rem_time = torch.zeros(1, 1, 0, 1, dtype=torch.float32)  # [1, 1, 0, 1]
    empty_context_rem_mask = torch.zeros(1, 1, 0, dtype=torch.bool)  # [1, 1, 0]

    return AICMECompartmentsDataBatch(
        target_obs=target_obs,
        target_obs_time=target_obs_time,
        target_obs_mask=target_obs_mask,
        target_rem_sim=target_rem_sim,
        target_rem_sim_time=target_rem_sim_time,
        target_rem_sim_mask=target_rem_sim_mask,
        context_obs=context_obs,
        context_obs_time=context_obs_time,
        context_obs_mask=context_obs_mask,
        context_rem_sim=empty_context_rem,
        context_rem_sim_time=empty_context_rem_time,
        context_rem_sim_mask=empty_context_rem_mask,
        target_dosing_amounts=torch.ones(1, 1, dtype=torch.float32),
        target_dosing_route_types=torch.zeros(1, 1, dtype=torch.long),
        context_dosing_amounts=torch.ones(1, 1, dtype=torch.float32),
        context_dosing_route_types=torch.zeros(1, 1, dtype=torch.long),
        mask_context_individuals=torch.tensor([[True]], dtype=torch.bool),
        mask_target_individuals=torch.tensor([[True]], dtype=torch.bool),
        study_name=[study_name],
        context_subject_name=[["context_0"]],
        target_subject_name=[["target_0"]],
        substance_name=[substance_name],
        time_scales=torch.ones(1, 2, dtype=torch.float32),
        is_empirical=True,
    )


class _IdentityScaler:
    """Minimal scaler stub used to exercise unbound helper methods."""

    @staticmethod
    def forward(values: torch.Tensor, times: torch.Tensor, stats: torch.Tensor):
        del stats
        return values, times


class _DummyEmpiricalDatamodule:
    """Datamodule stub returning one repo-specific held-out permutation list."""

    def __init__(self, batches: list[AICMECompartmentsDataBatch]) -> None:
        self._batches = batches
        self.calls: list[tuple[str, str]] = []

    def get_empirical_batches(
        self,
        *,
        split: str,
        empirical_name: str,
        device=None,
    ) -> list[AICMECompartmentsDataBatch]:
        del device
        self.calls.append((split, empirical_name))
        return self._batches


class _DummyGenerativeModel:
    """Generative-model stub exposing target-resolved sampling."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.calls: list[dict[str, object]] = []
        self.model_config = SimpleNamespace(
            mix_data=SimpleNamespace(test_empirical_datasets=["repo_a"])
        )

    def sample_new_individual(self, batch, **kwargs):
        self.calls.append(dict(kwargs))
        full_batch = _reconstruct_full_target_databatch(batch)
        generated_samples = (
            full_batch.target_obs.permute(1, 0, 2, 3).contiguous() + 0.1
        )  # [It, B, Tfull, 1]
        generated_times = full_batch.target_obs_time[:, 0].clone()  # [B, Tfull, 1]
        generated_mask = full_batch.target_obs_mask[:, 0].clone()  # [B, Tfull]
        return generated_samples, generated_times, generated_mask


def test_empirical_heldout_collection_requests_target_remainder_times() -> None:
    """Held-out empirical collection should request full target schedules."""

    batch = _make_empirical_batch(observed_offset=0.0)
    model = _DummyGenerativeModel()

    _collect_empirical_heldout_generated_classifier_collection(
        [batch],
        model=model,
        num_steps=2,
    )

    assert model.calls
    assert model.calls[0]["resolve_sampling_from_target"] is True
    assert model.calls[0]["include_rem"] is True


def test_empirical_heldout_collection_skips_permutations_without_valid_targets() -> None:
    """Permutations with no comparable held-out target should simply not be counted."""

    valid_batch = _make_empirical_batch(observed_offset=0.0)
    invalid_batch = valid_batch._replace(
        target_obs_mask=torch.zeros_like(valid_batch.target_obs_mask),
        mask_target_individuals=torch.zeros_like(valid_batch.mask_target_individuals),
    )
    model = _DummyGenerativeModel()

    with pytest.warns(UserWarning, match="Skipping empirical held-out classifier permutation"):
        collection, output_substances = _collect_empirical_heldout_generated_classifier_collection(
            [invalid_batch, valid_batch],
            model=model,
            num_steps=2,
        )

    assert len(model.calls) == 1
    assert output_substances == ["Drug_A"]
    assert int(collection.dataset_bundle.mask.any(dim=-1).sum().item()) == 1


def test_task_empirical_heldout_generated_classifier_pools_all_configured_repos() -> None:
    """Held-out task should pool repos and dispatch classifier and MMD distances."""

    datamodule_calls: list[tuple[str, str]] = []

    def _get_empirical_batches(*, split: str, empirical_name: str, device=None):
        del device
        datamodule_calls.append((split, empirical_name))
        return [_make_empirical_batch(observed_offset=float(len(datamodule_calls)))]

    trainer = SimpleNamespace(
        datamodule=SimpleNamespace(get_empirical_batches=_get_empirical_batches),
        default_root_dir=Path("."),
        current_epoch=0,
        global_step=0,
    )
    model = SimpleNamespace(
        model_config=SimpleNamespace(
            mix_data=SimpleNamespace(test_empirical_datasets=["repo_a", "repo_b"])
        )
    )
    captured: dict[str, object] = {}

    aligned_bundle = SyntheticMMDSeriesBundle(
        observed_values=torch.tensor(
            [
                [[[1.0], [2.0], [3.0], [4.0]]],
                [[[5.0], [6.0], [7.0], [8.0]]],
            ],
            dtype=torch.float32,
        ),  # [B=2, It=1, Tobs=4, 1]
        generated_values=torch.tensor(
            [
                [[[1.1], [2.1], [3.1], [4.1]]],
                [[[5.1], [6.1], [7.1], [8.1]]],
            ],
            dtype=torch.float32,
        ),  # [B=2, It=1, Tobs=4, 1]
        times=torch.tensor(
            [
                [[[0.0], [1.0], [2.0], [3.0]]],
                [[[0.0], [1.0], [2.0], [3.0]]],
            ],
            dtype=torch.float32,
        ),  # [B=2, It=1, Tobs=4, 1]
        mask=torch.ones(2, 1, 4, dtype=torch.bool),  # [B=2, It=1, Tobs=4]
    )

    def _fake_collect(
        batch_list,
        *,
        model,
        num_steps: int | None = None,
    ):
        del model
        captured["batch_list"] = list(batch_list)
        captured["num_steps"] = num_steps
        return (
            DiverseExperimentDistanceCollection(
                dataset_bundle=aligned_bundle,
                aligned_bundles=[aligned_bundle],
                plot_batches=[],
                plot_aligned_bundles=[],
            ),
            ["Drug_A"],
        )

    def _fake_run_classifier_auc_distance(
        *,
        collection,
        classifier_cfg,
        output_root,
        save_details: bool,
    ):
        del classifier_cfg
        del output_root
        del save_details
        captured["classifier_bundle_shape"] = tuple(collection.dataset_bundle.observed_values.shape)
        return {"classifier_auc": 0.75}

    def _fake_run_mmd2_distance(
        *,
        bundle,
        mmd_cfg,
        output_root,
        save_details: bool,
    ):
        del output_root
        del save_details
        captured["mmd_bundle_shape"] = tuple(bundle.observed_values.shape)
        captured["mmd_cfg"] = dict(mmd_cfg)
        return {"mmd2": 0.125}

    with patch(
        "pff.training.callbacks.pk_task_empirical._shared."
        "_collect_empirical_heldout_generated_classifier_collection",
        side_effect=_fake_collect,
    ), patch(
        "pff.training.callbacks.pk_task_empirical._run_classifier_auc_distance",
        side_effect=_fake_run_classifier_auc_distance,
    ), patch(
        "pff.training.callbacks.pk_task_empirical._run_mmd2_distance",
        side_effect=_fake_run_mmd2_distance,
    ):
        result = task_empirical_heldout_generated_classifier(
            samples=None,
            batches=[],
            task_cfg={
                "split": "empirical_heldout",
                "distance_metrics": ["classifier_auc", "mmd2"],
                "save_details": False,
                "num_steps": 2,
                "classifier_auc": {"mode": "joint", "show_progress": False},
                "mmd": {"python_executable": "/tmp/ksig-env/bin/python"},
            },
            trainer=trainer,
            pl_module=model,
        )

    assert datamodule_calls == [
        ("empirical_heldout", "repo_a"),
        ("empirical_heldout", "repo_b"),
    ]
    assert len(captured["batch_list"]) == 2
    assert captured["num_steps"] == 2
    assert captured["classifier_bundle_shape"] == (2, 1, 4, 1)
    assert captured["mmd_bundle_shape"] == (1, 2, 4, 1)
    assert captured["mmd_cfg"] == {
        "python_executable": "/tmp/ksig-env/bin/python",
        "signature_levels": 4,
        "estimator": "unbiased",
        "include_time_channel": True,
    }
    assert result == {
        "classifier_auc": pytest.approx(0.75),
        "mmd2": pytest.approx(0.125),
    }


def test_task_empirical_heldout_generated_classifier_smoke_from_aicme_experiment() -> None:
    """Empirical held-out classifier should run end to end from a YAML-backed AICME config."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch.object(BasicLightningExperiment, "_setup_logger", _setup_dummy_logger):
            exp_config = _build_small_aicme_empirical_classifier_config(Path(tmp_dir))
            basic_experiment = BasicLightningExperiment.from_config(exp_config, map_location="cpu")
            datamodule = basic_experiment.get_datamodule()
            model = basic_experiment.get_module().to(torch.device("cpu"))
            empirical_name = exp_config.mix_data.test_empirical_datasets[0]
            trainer = SimpleNamespace(
                datamodule=datamodule,
                default_root_dir=Path(tmp_dir),
                current_epoch=0,
                global_step=0,
            )

            result = task_empirical_heldout_generated_classifier(
                samples=None,
                batches=[],
                task_cfg={
                    "empirical_name": empirical_name,
                    "split": "empirical_heldout",
                    "repo_id": empirical_name,
                    "save_details": False,
                    "num_steps": 2,
                    "classifier_auc": {
                        "mode": "joint",
                        "include_time_channel": True,
                        "hidden_dim": 32,
                        "num_hidden_layers": 2,
                        "learning_rate": 1.0e-3,
                        "weight_decay": 1.0e-4,
                        "epochs": 5,
                        "batch_size": 32,
                        "seed": 0,
                        "show_progress": False,
                    },
                },
                trainer=trainer,
                pl_module=model,
            )

            assert "classifier_auc" in result
            assert torch.isfinite(torch.tensor(result["classifier_auc"], dtype=torch.float32))
            assert 0.0 <= float(result["classifier_auc"]) <= 1.0


if __name__ == "__main__":
    test_task_empirical_heldout_generated_classifier_smoke_from_aicme_experiment()
