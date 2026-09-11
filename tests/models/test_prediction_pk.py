"""Unit tests for the current ``PredictionPK`` integration surface.

These tests cover the active prediction-only model path:
- forward/sampling on held-out target individuals,
- scheduler payload generation,
- empirical scheduler task expansion,
- predictive-only empirical summary evaluation.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pff import config_dir
from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.data.datasets.aicme_datasets import AICMECompartmentsDataModule
from pff.models.amortized_inference.prediction_pk import (
    PredictionForwardOutputs,
    PredictionPK,
)
from pff.training.callbacks.pk_task_empirical import task_empirical_summary
from pff.training.callbacks.pk_task_synthetic import (
    task_predictive_images,
)
from pff.training.callbacks.pk_tasks import (
    PredictiveTaskSamples,
)
from pff.training.callbacks.scheduler import BaseSchedulerCallback


def _prediction_config() -> NodePKExperimentConfig:
    """Return a compact held-out-target config for ``PredictionPK`` tests."""

    callbacks_scheduler = {
        "percent_step": 0.5,
        "include_end": True,
        "skip_sanity_check": True,
        "store_samples": True,
        "max_samples_per_group": 4,
        "task_during": [
            {
                "name": "empirical/predictive_metrics",
                "fn_key": "pk.empirical.predictive.metrics",
                "n_samples": 1,
                "sample_source": "empirical_set",
                "split": "empirical_heldout",
            }
        ],
        "tasks_end": [],
        "tasks_validation": [],
    }

    cfg = NodePKExperimentConfig()
    cfg.train = replace(
        cfg.train,
        batch_size=1,
        num_workers=0,
        persistent_workers=False,
        epochs=1,
        callbacks_scheduler=callbacks_scheduler,
    )
    cfg.mix_data = replace(
        cfg.mix_data,
        train_size=1,
        val_size=1,
        test_size=1,
        n_of_permutations=2,
        n_of_target_individuals=1,
        test_empirical_datasets=[],
    )
    cfg.meta_study = replace(cfg.meta_study, num_individuals_range=(3, 3))
    cfg.network = replace(cfg.network, aggregator_type="mean")
    cfg.context_observations = replace(
        cfg.context_observations,
        split_past_future=False,
        add_rem=True,
        max_num_obs=10,
        min_past=3,
        max_past=5,
    )
    cfg.target_observations = replace(
        cfg.target_observations,
        split_past_future=True,
        add_rem=True,
        max_num_obs=10,
        min_past=3,
        max_past=5,
    )
    return cfg


def _prediction_config_from_file() -> NodePKExperimentConfig:
    """Load the AISTATS NODE-PK config and shrink it for unit tests."""

    default_yaml = Path(config_dir) / "experiment_configs" / "AISTATS" / "node-pk" / "base.yaml"
    cfg = NodePKExperimentConfig.from_yaml(str(default_yaml))
    cfg.train = replace(
        cfg.train,
        batch_size=1,
        num_workers=0,
        persistent_workers=False,
        epochs=1,
    )
    cfg.mix_data = replace(
        cfg.mix_data,
        train_size=1,
        val_size=1,
        test_size=1,
        n_of_permutations=1,
        n_of_target_individuals=1,
        test_empirical_datasets=[],
    )
    cfg.meta_study = replace(cfg.meta_study, num_individuals_range=(3, 3))
    cfg.network = replace(cfg.network, aggregator_type="mean")
    cfg.context_observations = replace(
        cfg.context_observations,
        split_past_future=False,
        add_rem=True,
        max_num_obs=10,
        min_past=3,
        max_past=5,
    )
    cfg.target_observations = replace(
        cfg.target_observations,
        split_past_future=True,
        add_rem=True,
        max_num_obs=10,
        min_past=3,
        max_past=5,
    )
    return cfg


def _first_batch_list(dm: AICMECompartmentsDataModule):
    """Return the first permutation list from the training loader on CPU."""

    dm.prepare_data()
    dm.setup()
    batch_list = next(iter(dm.train_dataloader()))
    assert isinstance(batch_list, list) and len(batch_list) >= 1
    return [batch.to_device("cpu") for batch in batch_list]


def test_prediction_forward_pass() -> None:
    """Forward pass returns aggregated predictive losses and heads."""

    cfg = _prediction_config()
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)
    model = PredictionPK(cfg)
    outputs = model(batch_list)

    assert isinstance(outputs, PredictionForwardOutputs)
    losses = outputs.to_dict()
    assert "pred_loss" in losses
    assert "pred_rmse" in losses

    head = outputs.heads["prediction"]
    for key in ("mean", "logvar", "target", "mask"):
        assert key in head
        assert head[key] is not None
    assert head["mean"].shape[-1] == 1


def test_prediction_forward_pass_from_aistats_config() -> None:
    """Forward pass works when the test config is loaded from AISTATS YAML files."""

    cfg = _prediction_config_from_file()
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)
    model = PredictionPK(cfg)
    outputs = model(batch_list)

    assert isinstance(outputs, PredictionForwardOutputs)
    losses = outputs.to_dict()
    assert "pred_loss" in losses
    assert "pred_rmse" in losses

    head = outputs.heads["prediction"]
    for key in ("mean", "logvar", "target", "mask"):
        assert key in head
        assert head[key] is not None
    assert head["mean"].shape[-1] == 1


def test_prediction_pk_sample_individual_prediction_shapes() -> None:
    """Sampling returns the expected predictive tensor shapes."""

    cfg = _prediction_config()
    dm = AICMECompartmentsDataModule(cfg)
    db0 = _first_batch_list(dm)[0]
    model = PredictionPK(cfg)
    sampled, tgrid, real, mask = model.sample_individual_prediction(db0, sample_size=5)

    assert sampled.dim() == 5 and tgrid.dim() == 5
    assert sampled.size(0) == 5 and tgrid.size(0) == 5
    assert sampled.size(-1) == 1 and tgrid.size(-1) == 1
    assert real.dim() == 4 and real.size(-1) == 1
    assert mask.dim() == 3


def test_predictionpk_build_visualization_callback_keeps_one_empirical_task() -> None:
    """PredictionPK should keep one cross-repo empirical scheduler task."""

    cfg = _prediction_config()
    cfg.mix_data = replace(
        cfg.mix_data,
        test_empirical_datasets=["org/repo_a", "org/repo_b"],
    )
    model = PredictionPK(cfg)

    callbacks = model.build_visualization_callback()

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], BaseSchedulerCallback)
    assert len(callbacks[0].task_during) == 1
    assert callbacks[0].task_during[0].empirical_name is None


def test_predictionpk_generate_returns_predictive_scheduler_payload(tmp_path: Path) -> None:
    """PredictionPK scheduler payload should remain predictive-only for image tasks."""

    cfg = _prediction_config()
    dm = AICMECompartmentsDataModule(cfg)
    db0 = _first_batch_list(dm)[0]
    model = PredictionPK(cfg)

    payload = model.generate(db0, num_samples=3)

    assert isinstance(payload, PredictiveTaskSamples)
    assert payload.predictive.samples_S.shape[0] == 3

    trainer = SimpleNamespace(current_epoch=1, default_root_dir=str(tmp_path))
    images = task_predictive_images(
        samples=payload,
        batches=[db0],
        task_cfg={"label": "Synthetic", "model_label": "PredictionPK"},
        trainer=trainer,
        pl_module=model,
    )
    assert images
    for image_path in images.values():
        assert Path(image_path).exists()


def test_empirical_predictive_images_default_to_one_prediction_per_drug(tmp_path: Path) -> None:
    """Empirical predictive images should default to one plot per drug across batches."""

    cfg = _prediction_config()
    dm = AICMECompartmentsDataModule(cfg)
    db0 = _first_batch_list(dm)[0]
    model = PredictionPK(cfg)
    payload = model.generate(db0, num_samples=3)
    trainer = SimpleNamespace(current_epoch=1, default_root_dir=str(tmp_path))
    plot_calls: list[dict[str, object]] = []

    def _fake_plot(**kwargs):
        plot_calls.append(kwargs)
        return str(tmp_path / f"empirical_{len(plot_calls)}.png")

    with patch(
        "pff.training.callbacks.pk_tasks.plot_list_list_study_json",
        side_effect=_fake_plot,
    ):
        images = task_predictive_images(
            samples=[payload, payload],
            batches=[db0, db0],
            task_cfg={"label": "Empirical", "model_label": "PredictionPK"},
            trainer=trainer,
            pl_module=model,
        )

        assert images
    assert len(images) == 1
    assert len(plot_calls) == 1
    assert plot_calls[0]["plot_all_separately"] is True
    assert plot_calls[0]["plot_kwargs"]["number_of_predictions_plot_per_drug"] == 1


def test_empirical_predictive_images_respect_prediction_limit_override(tmp_path: Path) -> None:
    """Empirical predictive images should respect the configured per-drug plot limit."""

    cfg = _prediction_config()
    dm = AICMECompartmentsDataModule(cfg)
    db0 = _first_batch_list(dm)[0]
    model = PredictionPK(cfg)
    payload = model.generate(db0, num_samples=3)
    trainer = SimpleNamespace(current_epoch=1, default_root_dir=str(tmp_path))
    plot_calls: list[dict[str, object]] = []

    def _fake_plot(**kwargs):
        plot_calls.append(kwargs)
        return str(tmp_path / f"empirical_{len(plot_calls)}.png")

    with patch(
        "pff.training.callbacks.pk_tasks.plot_list_list_study_json",
        side_effect=_fake_plot,
    ):
        images = task_predictive_images(
            samples=[payload, payload],
            batches=[db0, db0],
            task_cfg={
                "label": "Empirical",
                "model_label": "PredictionPK",
                "number_of_predictions_plot_per_drug": 2,
            },
            trainer=trainer,
            pl_module=model,
        )

    assert images
    assert len(images) == 2
    assert len(plot_calls) == 2
    assert plot_calls[0]["plot_kwargs"]["number_of_predictions_plot_per_drug"] == 2


def test_task_empirical_summary_predictive_scope_uses_heldout_only(monkeypatch) -> None:
    """Predictive-only summary should read only held-out empirical batches."""

    dm_cfg = _prediction_config()
    dm = AICMECompartmentsDataModule(dm_cfg)
    batch = _first_batch_list(dm)[0]._replace(substance_name=["test_drug"])

    model_cfg = _prediction_config()
    model_cfg.mix_data = replace(model_cfg.mix_data, test_empirical_datasets=["repo_a"])
    model = PredictionPK(model_cfg)

    calls: list[tuple[str, str]] = []

    def fake_get_empirical_batches(*, split: str, empirical_name: str, device=None):
        del device
        calls.append((split, empirical_name))
        return [batch]

    monkeypatch.setattr(dm, "get_empirical_batches", fake_get_empirical_batches)

    out = task_empirical_summary(
        samples=None,
        batches=[batch],
        task_cfg={
            "summary_metric": "log_rmse",
            "summary_scope": "predictive",
            "selected_summary_drugs": ["test_drug"],
        },
        trainer=SimpleNamespace(datamodule=dm),
        pl_module=model,
    )

    assert "log_rmse" in out
    assert calls == [("empirical_heldout", "repo_a")]


if __name__ == "__main__":
    test_prediction_forward_pass_from_aistats_config()
