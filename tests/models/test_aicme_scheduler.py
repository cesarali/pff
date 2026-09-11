"""Unit tests for ACIME scheduler callback construction."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import torch

from pff import config_dir
from pff.config_classes.flow_pk_config import FlowPKExperimentConfig
from pff.config_classes.training_config import SchedulerConfig
from pff.config_classes.training_config import TrainingConfig
from pff.models.amortized_inference.aicme import (
    AICMEPK,
    _expand_empirical_scheduler_tasks,
)
from pff.models.amortized_inference.generative_pk import NewBasePKModel
from pff.training.callbacks.scheduler import BaseSchedulerCallback, SampleSource
from pff.training.callbacks.task_registry import TASK_REGISTRY

FakeBatch = namedtuple("FakeBatch", ["value"])


def test_training_config_filter_keeps_callbacks_scheduler() -> None:
    filtered = TrainingConfig._filter_kwargs(
        {
            "epochs": 5,
            "callbacks_scheduler": {"percent_step": 0.5},
            "unknown_key": 123,
        }
    )
    assert filtered == {
        "epochs": 5,
        "callbacks_scheduler": {"percent_step": 0.5},
    }


def test_expand_empirical_scheduler_tasks_rewrites_internal_empirical_tasks_once() -> None:
    expanded = _expand_empirical_scheduler_tasks(
        {
            "task_during": [
                {
                    "name": "empirical/predictive_metrics",
                    "fn_key": "pk.empirical.predictive.metrics",
                    "n_samples": 3,
                    "sample_source": "empirical_set",
                    "split": "empirical_heldout",
                },
                {
                    "name": "summary",
                    "fn_key": "pk.empirical.summary",
                    "sample_source": "val_batch",
                    "split": "val",
                },
            ]
        },
        empirical_datasets=["org/repo_a", "org/repo_b"],
        model_label="AICME",
    )

    tasks = expanded["task_during"]
    names = [task["name"] for task in tasks]
    assert "empirical/predictive_metrics" in names
    assert "summary" in names

    empirical_tasks = [task for task in tasks if task["name"].startswith("empirical/")]
    assert len(empirical_tasks) == 1
    assert empirical_tasks[0]["empirical_name"] is None
    assert all(task["fn_key"] == "pk.empirical.predictive.metrics" for task in empirical_tasks)
    assert all(task["sample_source"] == "task_internal" for task in empirical_tasks)
    assert all(task["n_samples"] == 0 for task in empirical_tasks)
    assert [task["task_cfg"]["sample_size"] for task in empirical_tasks] == [3]
    assert [task["task_cfg"]["split"] for task in empirical_tasks] == ["empirical_heldout"]


def test_expand_empirical_scheduler_tasks_rewrites_classifier_once() -> None:
    expanded = _expand_empirical_scheduler_tasks(
        {
            "task_during": [
                {
                    "name": "empirical/heldout_generated_classifier",
                    "fn_key": "pk.empirical.heldout_generated_classifier",
                    "n_samples": 0,
                    "sample_source": "empirical_set",
                    "split": "empirical_heldout",
                }
            ]
        },
        empirical_datasets=["org/repo_a", "org/repo_b"],
        model_label="AICME",
    )

    tasks = expanded["task_during"]
    assert len(tasks) == 1
    assert tasks[0]["name"] == "empirical/heldout_generated_classifier"
    assert tasks[0]["empirical_name"] is None
    assert tasks[0]["fn_key"] == "pk.empirical.heldout_generated_classifier"
    assert tasks[0]["sample_source"] == "task_internal"
    assert tasks[0]["n_samples"] == 0
    assert tasks[0]["task_cfg"]["split"] == "empirical_heldout"
    assert tasks[0]["task_cfg"]["model_label"] == "AICME"


def test_build_visualization_callback_returns_scheduler_when_configured() -> None:
    train_cfg = TrainingConfig(
        callbacks_scheduler={
            "percent_step": 0.5,
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
    )
    fake_model = SimpleNamespace(
        model_config=SimpleNamespace(
            train=train_cfg,
            mix_data=SimpleNamespace(test_empirical_datasets=["org/repo_a", "org/repo_b"]),
            name_str="AICME",
        )
    )

    callbacks = AICMEPK.build_visualization_callback(fake_model)

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], BaseSchedulerCallback)
    assert len(callbacks[0].task_during) == 1
    assert callbacks[0].task_during[0].empirical_name is None
    assert callbacks[0].task_during[0].fn_key == "pk.empirical.predictive.metrics"
    assert callbacks[0].task_during[0].sample_source is SampleSource.TASK_INTERNAL
    assert callbacks[0].task_during[0].n_samples == 0
    assert callbacks[0].task_during[0].task_cfg["sample_size"] == 1


def test_base_pk_build_visualization_callback_keeps_one_empirical_task() -> None:
    """Generative PK base models should keep one cross-repo empirical task."""

    train_cfg = TrainingConfig(
        callbacks_scheduler={
            "percent_step": 0.5,
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
    )
    fake_model = SimpleNamespace(
        model_config=SimpleNamespace(
            train=train_cfg,
            mix_data=SimpleNamespace(test_empirical_datasets=["org/repo_a", "org/repo_b"]),
            name_str="FlowPK",
        )
    )

    callbacks = NewBasePKModel.build_visualization_callback(fake_model)

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], BaseSchedulerCallback)
    assert len(callbacks[0].task_during) == 1
    assert callbacks[0].task_during[0].empirical_name is None
    assert callbacks[0].task_during[0].fn_key == "pk.empirical.predictive.metrics"
    assert callbacks[0].task_during[0].sample_source is SampleSource.TASK_INTERNAL
    assert callbacks[0].task_during[0].n_samples == 0
    assert callbacks[0].task_during[0].task_cfg["sample_size"] == 1


def test_scheduler_resolves_empirical_set_across_all_configured_repos() -> None:
    callback = BaseSchedulerCallback(
        config=SchedulerConfig(store_samples=False),
        tasks_validation=[],
        task_during=[],
        tasks_end=[],
    )
    datamodule_calls: list[tuple[str, str]] = []

    def _get_empirical_batches(*, split: str, empirical_name: str):
        datamodule_calls.append((split, empirical_name))
        return [FakeBatch(f"{empirical_name}_perm_0"), FakeBatch(f"{empirical_name}_perm_1")]

    trainer = SimpleNamespace(
        datamodule=SimpleNamespace(get_empirical_batches=_get_empirical_batches),
        lightning_module=SimpleNamespace(
            model_config=SimpleNamespace(
                mix_data=SimpleNamespace(test_empirical_datasets=["org/repo_a", "org/repo_b"])
            )
        ),
    )

    resolved = callback._resolve_batches(
        trainer=trainer,
        sample_source=SampleSource.EMPIRICAL_SET,
        split="empirical_heldout",
        empirical_name=None,
        current_val_batch=None,
    )

    assert datamodule_calls == [
        ("empirical_heldout", "org/repo_a"),
        ("empirical_heldout", "org/repo_b"),
    ]
    assert resolved == [
        FakeBatch("org/repo_a_perm_0"),
        FakeBatch("org/repo_a_perm_1"),
        FakeBatch("org/repo_b_perm_0"),
        FakeBatch("org/repo_b_perm_1"),
    ]


def test_scheduler_resolves_val_batch_permutations_as_flat_batch_list() -> None:
    callback = BaseSchedulerCallback(
        config=SchedulerConfig(store_samples=False),
        tasks_validation=[],
        task_during=[],
        tasks_end=[],
    )
    current_val_batch = [FakeBatch("perm_0"), FakeBatch("perm_1")]

    resolved = callback._resolve_batches(
        trainer=SimpleNamespace(),
        sample_source=SampleSource.VAL_BATCH,
        split="val",
        empirical_name=None,
        current_val_batch=current_val_batch,
    )

    assert resolved == current_val_batch


def test_scheduler_flattens_nested_full_split_batches() -> None:
    callback = BaseSchedulerCallback(
        config=SchedulerConfig(store_samples=False),
        tasks_validation=[],
        task_during=[],
        tasks_end=[],
    )
    nested_batches = [[FakeBatch("perm_0"), FakeBatch("perm_1")], [FakeBatch("perm_2")]]
    callback._iter_full_split = lambda trainer, split: nested_batches  # type: ignore[method-assign]

    resolved = callback._resolve_batches(
        trainer=SimpleNamespace(),
        sample_source=SampleSource.FULL_SPLIT,
        split="val",
        empirical_name=None,
        current_val_batch=None,
    )

    assert resolved == [FakeBatch("perm_0"), FakeBatch("perm_1"), FakeBatch("perm_2")]


def test_scheduler_generates_for_each_validation_permutation() -> None:
    class DummyModel:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def generate(self, batch, num_samples: int = 1):
            self.calls.append((batch.value, num_samples))
            return {"value": batch.value, "num_samples": num_samples}

    callback = BaseSchedulerCallback(
        config=SchedulerConfig(store_samples=False),
        tasks_validation=[],
        task_during=[],
        tasks_end=[],
    )
    model = DummyModel()

    generated = callback._generate_once(
        pl_module=SimpleNamespace(model=model, device=torch.device("cpu")),
        sample_source=SampleSource.VAL_BATCH,
        n_samples=3,
        batches=[FakeBatch("perm_0"), FakeBatch("perm_1")],
    )

    assert generated == [
        {"value": "perm_0", "num_samples": 3},
        {"value": "perm_1", "num_samples": 3},
    ]
    assert model.calls == [("perm_0", 3), ("perm_1", 3)]


def test_scheduler_prefers_to_device_when_available() -> None:
    class DeviceAwareBatch:
        def __init__(self) -> None:
            self.seen_device = None

        def to_device(self, device):
            self.seen_device = device
            return ("moved", device)

    batch = DeviceAwareBatch()
    moved = BaseSchedulerCallback._to_device_if_possible(batch, torch.device("cpu"))

    assert moved == ("moved", torch.device("cpu"))
    assert batch.seen_device == torch.device("cpu")


def test_flowpk_yaml_with_diverse_experiment_distances_scheduler_builds() -> None:
    experiment_yaml = (
        Path(config_dir)
        / "experiment_configs"
        / "UAI"
        / "Submission"
        / "flow-pk-predict-n-generate-test"
        / "flowPK.yaml"
    )

    cfg = FlowPKExperimentConfig.from_yaml(str(experiment_yaml))
    callback = BaseSchedulerCallback.from_config(
        cfg=cfg.train.callbacks_scheduler,
        registry=TASK_REGISTRY,
    )

    synthetic_tasks = [
        task
        for task in callback.tasks_end
        if task.fn_key == "pk.diverse_experiment.distances"
    ]
    assert len(synthetic_tasks) == 1
    assert synthetic_tasks[0].sample_source is SampleSource.TASK_INTERNAL
