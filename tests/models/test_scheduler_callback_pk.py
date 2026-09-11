"""Unit tests for the PK scheduler integration surface."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pff.config_classes.training_config import SchedulerConfig
from pff.training.callbacks.pk_task_empirical import (
    task_empirical_predictive_metrics,
)
from pff.training.callbacks.pk_tasks import (
    GenerativeBundle,
    PKTaskSamples,
    PredictiveBundle,
    VPCBundle,
    summarize_across_repos,
)
from pff.training.callbacks.scheduler import BaseSchedulerCallback, SampleSource, TaskSpec
from pff.training.callbacks.task_registry import TASK_REGISTRY


def test_task_registry_exposes_pk_scheduler_functions() -> None:
    expected = {
        "pk.empirical.predictive.metrics",
        "pk.predictive.images",
        "pk.generative.metrics",
        "pk.generative.images",
        "pk.vpc.npde_pvalues",
        "pk.vpc.images",
        "pk.synthetic.vpc.paired_images",
        "pk.empirical.summary",
        "pk.diverse_experiment.distances",
        "pk.diverse_synthetic_experiment.sample_distances",
    }
    assert expected.issubset(TASK_REGISTRY.keys())


def test_pk_task_samples_scheduler_slice_trims_all_sample_axes() -> None:
    payload = PKTaskSamples(
        predictive=PredictiveBundle(
            samples_S=torch.randn(5, 2, 1, 3, 1),
            times_S=torch.randn(5, 2, 1, 3, 1),
            target_raw=torch.randn(2, 1, 3, 1),
            target_mask=torch.ones(2, 1, 3, dtype=torch.bool),
        ),
        generative=GenerativeBundle(
            samples=torch.randn(5, 2, 4, 1),
            times=torch.randn(2, 4, 1),
            mask=torch.ones(2, 4, dtype=torch.bool),
            pred_values=torch.randn(2, 5, 4, 1),
        ),
        vpc=VPCBundle(
            observed_studies=[{"context": [], "target": [], "meta_data": {}}],
            simulated_studies_by_substance=[[{"context": []}] * 5],
        ),
    )

    sliced = payload.scheduler_slice(3)

    assert sliced.predictive.samples_S.shape[0] == 3
    assert sliced.predictive.times_S.shape[0] == 3
    assert sliced.generative.samples.shape[0] == 3
    assert sliced.generative.pred_values.shape[1] == 3
    assert len(sliced.vpc.simulated_studies_by_substance[0]) == 3


def test_filter_tasks_for_milestone_respects_stride() -> None:
    always = TaskSpec(name="always", fn_key="always", fn=lambda **_: {}, task_cfg={})
    every_five = TaskSpec(
        name="five",
        fn_key="five",
        fn=lambda **_: {},
        task_cfg={"milestone_stride": 5},
    )

    first = BaseSchedulerCallback._filter_tasks_for_milestone(
        [always, every_five],
        milestone_idx=1,
    )
    fifth = BaseSchedulerCallback._filter_tasks_for_milestone(
        [always, every_five],
        milestone_idx=5,
    )

    assert [task.name for task in first] == ["always"]
    assert [task.name for task in fifth] == ["always", "five"]


def test_scheduler_resolves_last_best_and_task_checkpoint_paths(tmp_path: Path) -> None:
    last_path = tmp_path / "last.ckpt"
    best_path = tmp_path / "best.ckpt"
    task_metric_path = tmp_path / "best-log-rmse.ckpt"
    for path in (last_path, best_path, task_metric_path):
        path.write_text("x", encoding="utf-8")

    callback = BaseSchedulerCallback(
        config=SchedulerConfig(),
        tasks_validation=[],
        task_during=[],
        tasks_end=[],
    )
    callback.attach_experiment_checkpoints(
        checkpoint_callback_last=SimpleNamespace(last_model_path=str(last_path)),
        checkpoint_callback_best=SimpleNamespace(best_model_path=str(best_path)),
    )
    callback._best_checkpoint_path_by_name["log_rmse"] = task_metric_path

    assert callback._resolve_checkpoint_path("last") == last_path
    assert callback._resolve_checkpoint_path("best") == best_path
    assert callback._resolve_checkpoint_path("log_rmse") == task_metric_path


def test_scheduler_task_internal_skips_batch_resolution_and_generation(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def task_fn(**kwargs):
        captured.update(kwargs)
        return {}

    callback = BaseSchedulerCallback(
        config=SchedulerConfig(store_samples=False),
        tasks_validation=[],
        task_during=[],
        tasks_end=[],
    )

    def fail_resolve_batches(**kwargs):
        raise AssertionError("_resolve_batches should not be used for task_internal")

    def fail_generate_once(**kwargs):
        raise AssertionError("_generate_once should not be used for task_internal")

    callback._resolve_batches = fail_resolve_batches  # type: ignore[method-assign]
    callback._generate_once = fail_generate_once  # type: ignore[method-assign]

    trainer = SimpleNamespace(
        default_root_dir=tmp_path,
        global_step=0,
        current_epoch=0,
        logger=None,
        loggers=[],
    )
    pl_module = SimpleNamespace(device=torch.device("cpu"))
    task = TaskSpec(
        name="synthetic/diverse_experiment_distances",
        fn_key="pk.diverse_experiment.distances",
        fn=task_fn,
        n_samples=0,
        sample_source=SampleSource.TASK_INTERNAL,
        split="val",
        task_cfg={},
    )

    callback._run_task_list(
        tasks=[task],
        trainer=trainer,
        pl_module=pl_module,
        current_val_batch=None,
        tag="train_end__end",
        checkpoint_label="end",
    )

    assert captured["samples"] is None
    assert captured["batches"] == []


def test_scheduler_logs_png_paths_as_images_and_pdf_paths_as_assets(tmp_path: Path) -> None:
    class RecordingExperiment:
        def __init__(self) -> None:
            self.image_calls: list[tuple[str, str, int]] = []
            self.asset_calls: list[tuple[str, str, int]] = []

        def log_image(self, image_path: str, name: str, step: int) -> None:
            self.image_calls.append((image_path, name, step))

        def log_asset(self, asset_path: str, name: str, step: int) -> None:
            self.asset_calls.append((asset_path, name, step))

    experiment = RecordingExperiment()
    trainer = SimpleNamespace(
        global_step=7,
        logger=SimpleNamespace(experiment=experiment),
        loggers=[SimpleNamespace(experiment=experiment)],
    )
    callback = BaseSchedulerCallback(
        config=SchedulerConfig(),
        tasks_validation=[],
        task_during=[],
        tasks_end=[],
    )
    task = TaskSpec(
        name="synthetic/new_individuals_images",
        fn_key="pk.generative.images",
        fn=lambda **_: {},
        log_prefix="Synthetic",
    )

    png_path = tmp_path / "epoch_019.png"
    pdf_path = tmp_path / "epoch_019.pdf"
    png_path.write_text("png", encoding="utf-8")
    pdf_path.write_text("pdf", encoding="utf-8")

    callback._log_and_save(
        trainer=trainer,
        pl_module=SimpleNamespace(),
        task=task,
        out={"image_000": png_path, "asset_000": pdf_path},
    )

    assert experiment.image_calls == [
        (
            str(png_path),
            "Synthetic/synthetic/new_individuals_images/image_000",
            7,
        )
    ]
    assert experiment.asset_calls == [
        (
            str(pdf_path),
            "Synthetic_synthetic_new_individuals_images_asset_000",
            7,
        )
    ]


@pytest.mark.parametrize("section_name", ["tasks_validation", "task_during"])
def test_diverse_experiment_distances_is_rejected_outside_tasks_end(section_name: str) -> None:
    cfg = {
        section_name: [
            {
                "name": "synthetic/diverse_experiment_distances",
                "fn_key": "pk.diverse_experiment.distances",
                "n_samples": 0,
                "sample_source": "task_internal",
                "split": "val",
                "task_cfg": {
                    "distance_metrics": ["mmd2"],
                    "mmd": {
                        "python_executable": "/home/cesarali/miniconda3/envs/ksig/bin/python",
                    },
                    "synthetic_loader": {
                        "n_targets": 6,
                        "n_dosings": 1,
                        "dosing_mode": "diverse_dosing",
                        "dataset_size": 1,
                    },
                },
            }
        ]
    }

    with pytest.raises(ValueError, match="tasks_end"):
        BaseSchedulerCallback.from_config(cfg=cfg, registry=TASK_REGISTRY)


def test_diverse_experiment_distances_is_accepted_in_tasks_end() -> None:
    callback = BaseSchedulerCallback.from_config(
        cfg={
            "tasks_end": [
                {
                    "name": "synthetic/diverse_experiment_distances",
                    "fn_key": "pk.diverse_experiment.distances",
                    "n_samples": 0,
                    "sample_source": "task_internal",
                    "split": "val",
                    "task_cfg": {
                        "distance_metrics": ["mmd2"],
                        "mmd": {
                            "python_executable": "/home/cesarali/miniconda3/envs/ksig/bin/python",
                        },
                        "synthetic_loader": {
                            "n_targets": 6,
                            "n_dosings": 1,
                            "dosing_mode": "diverse_dosing",
                            "dataset_size": 1,
                        },
                    },
                }
            ]
        },
        registry=TASK_REGISTRY,
    )

    assert len(callback.tasks_end) == 1
    assert callback.tasks_end[0].sample_source is SampleSource.TASK_INTERNAL


@pytest.mark.parametrize("section_name", ["tasks_validation", "task_during"])
def test_synthetic_vpc_paired_images_is_rejected_outside_tasks_end(section_name: str) -> None:
    cfg = {
        section_name: [
            {
                "name": "synthetic/vpc_paired_images",
                "fn_key": "pk.synthetic.vpc.paired_images",
                "n_samples": 0,
                "sample_source": "task_internal",
                "split": "val",
                "log_prefix": "Synthetic",
                "task_cfg": {
                    "n_cases": 1,
                    "sample_size": 4,
                    "n_observed_individuals": 3,
                },
            }
        ]
    }

    with pytest.raises(ValueError, match="tasks_end"):
        BaseSchedulerCallback.from_config(cfg=cfg, registry=TASK_REGISTRY)


def test_synthetic_vpc_paired_images_is_accepted_in_tasks_end() -> None:
    callback = BaseSchedulerCallback.from_config(
        cfg={
            "tasks_end": [
                {
                    "name": "synthetic/vpc_paired_images",
                    "fn_key": "pk.synthetic.vpc.paired_images",
                    "n_samples": 0,
                    "sample_source": "task_internal",
                    "split": "val",
                    "log_prefix": "Synthetic",
                    "task_cfg": {
                        "n_cases": 1,
                        "sample_size": 4,
                        "n_observed_individuals": 3,
                    },
                }
            ]
        },
        registry=TASK_REGISTRY,
    )

    assert len(callback.tasks_end) == 1
    assert callback.tasks_end[0].sample_source is SampleSource.TASK_INTERNAL
    assert callback.tasks_end[0].fn_key == "pk.synthetic.vpc.paired_images"


def test_task_internal_rejects_positive_n_samples() -> None:
    with pytest.raises(ValueError, match="n_samples=0"):
        BaseSchedulerCallback.from_config(
            cfg={
                "tasks_end": [
                    {
                        "name": "synthetic/diverse_experiment_distances",
                        "fn_key": "pk.diverse_experiment.distances",
                        "n_samples": 1,
                        "sample_source": "task_internal",
                        "split": "val",
                        "task_cfg": {
                            "distance_metrics": ["mmd2"],
                            "mmd": {
                                "python_executable": "/home/cesarali/miniconda3/envs/ksig/bin/python",
                            },
                            "synthetic_loader": {
                                "n_targets": 6,
                                "n_dosings": 1,
                                "dosing_mode": "diverse_dosing",
                                "dataset_size": 1,
                            },
                        },
                    }
                ]
            },
            registry=TASK_REGISTRY,
        )


def test_task_empirical_predictive_metrics_resolves_and_aggregates_internal_sampling() -> None:
    class DummyEmpiricalPredictiveModel:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.sample_calls: list[tuple[str, int]] = []

        def sample_individual_prediction(self, batch, sample_size: int = 1):
            self.sample_calls.append((batch.permutation_name, sample_size))
            samples = batch.prediction_values.unsqueeze(0).repeat(sample_size, 1, 1, 1, 1)
            times = torch.zeros_like(samples)
            return samples, times, batch.target_obs, batch.target_obs_mask

        @staticmethod
        def masked_rmse_loss(pred, target, mask):
            del mask
            return {"rmse": (pred - target).mean().abs()}

        @staticmethod
        def masked_log_rmse_loss(pred, target, mask):
            del mask
            return {"rmse": (pred - target).mean().abs() + 10.0}

        @staticmethod
        def masked_r2_score(pred, target, mask):
            del mask
            return (pred - target).mean() + 20.0

        @staticmethod
        def masked_log_r2_score(pred, target, mask):
            del mask
            return (pred - target).mean() + 30.0

    def make_batch(
        *,
        permutation_name: str,
        prediction_values: torch.Tensor,
        valid_targets: torch.Tensor,
    ) -> SimpleNamespace:
        batch_size, num_targets, num_times, _ = prediction_values.shape
        return SimpleNamespace(
            permutation_name=permutation_name,
            prediction_values=prediction_values,
            target_obs=torch.zeros(batch_size, num_targets, num_times, 1),
            target_obs_mask=torch.ones(batch_size, num_targets, num_times, dtype=torch.bool),
            mask_target_individuals=valid_targets,
            substance_name=["Drug_A", "Drug_B"],
            study_name=["Study_A", "Study_B"],
        )

    perm_0 = make_batch(
        permutation_name="perm_0",
        prediction_values=torch.tensor(
            [
                [[[1.0]], [[99.0]]],
                [[[2.0]], [[99.0]]],
            ],
            dtype=torch.float32,
        ),
        valid_targets=torch.tensor(
            [
                [True, False],
                [True, False],
            ],
            dtype=torch.bool,
        ),
    )
    perm_1 = make_batch(
        permutation_name="perm_1",
        prediction_values=torch.tensor(
            [
                [[[99.0]], [[3.0]]],
                [[[99.0]], [[99.0]]],
            ],
            dtype=torch.float32,
        ),
        valid_targets=torch.tensor(
            [
                [False, True],
                [False, False],
            ],
            dtype=torch.bool,
        ),
    )

    datamodule_calls: list[tuple[str, str]] = []

    def get_empirical_batches(*, split: str, empirical_name: str, device=None):
        del device
        datamodule_calls.append((split, empirical_name))
        return [perm_0, perm_1]

    model = DummyEmpiricalPredictiveModel()
    out = task_empirical_predictive_metrics(
        samples=["ignored"],
        batches=[perm_0],
        task_cfg={
            "sample_size": 2,
            "split": "empirical_heldout",
            "empirical_name": "org/repo_a",
            "repo_id": "org/repo_a",
        },
        trainer=SimpleNamespace(datamodule=SimpleNamespace(get_empirical_batches=get_empirical_batches)),
        pl_module=model,
    )

    assert datamodule_calls == [("empirical_heldout", "org/repo_a")]
    assert model.sample_calls == [("perm_0", 2), ("perm_1", 2)]
    assert out["Drug_A/rmse"] == pytest.approx(2.0)
    assert out["Drug_A/rmse_std"] == pytest.approx(2.0**0.5)
    assert out["Drug_A/log_rmse"] == pytest.approx(12.0)
    assert out["Drug_A/r2"] == pytest.approx(22.0)
    assert out["Drug_A/log_r2"] == pytest.approx(32.0)
    assert out["Drug_B/rmse"] == pytest.approx(2.0)
    assert out["Drug_B/rmse_std"] == pytest.approx(0.0)


def test_summarize_across_repos_averages_selected_substances_only() -> None:
    summary = summarize_across_repos(
        {
            "repo_a": {
                "midazolam": {"rmse": 2.0},
                "warfarin": {"rmse": 99.0},
            },
            "repo_b": {
                "paracetamol glucuronide": {"rmse": 4.0},
            },
        },
        selected_drugs=["midazolam", "paracetamol glucuronide"],
        metric_name="rmse",
    )

    assert summary == 3.0
