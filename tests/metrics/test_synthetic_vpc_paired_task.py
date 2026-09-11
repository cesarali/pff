"""Tests for the synthetic truth-vs-model paired VPC scheduler task."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from pff import config_dir
from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.data.data_empirical.json_schema import IndividualJSON, StudyJSON
from pff.data.datasets.aicme_batch import AICMECompartmentsDataBatch
from pff.data.datasets.aicme_datasets import AICMECompartmentsDataModule
from pff.models.amortized_inference.aicme import AICMEPK
from pff.training.callbacks import pk_task_synthetic
from pff.training.callbacks.pk_task_synthetic import (
    task_synthetic_vpc_paired_images,
)


def _make_observed_study(case_index: int = 0) -> StudyJSON:
    """Create one compact observed StudyJSON for paired-VPC task tests."""

    return StudyJSON(
        context=[
            IndividualJSON(
                name_id=f"ctx_{case_index}_0",
                observation_times=[0.0, 1.0, 2.0],
                observations=[10.0 + case_index, 20.0 + case_index, 30.0 + case_index],
                dosing=[100.0],
                dosing_type=["oral"],
                dosing_times=[12.0],
                dosing_name=["oral"],
            ),
            IndividualJSON(
                name_id=f"ctx_{case_index}_1",
                observation_times=[0.0, 1.0, 2.0],
                observations=[11.0 + case_index, 21.0 + case_index, 31.0 + case_index],
                dosing=[120.0],
                dosing_type=["iv"],
                dosing_times=[12.0],
                dosing_name=["iv"],
            ),
        ],
        target=[],
        meta_data={
            "study_name": f"study_{case_index}",
            "substance_name": f"drug_{case_index}",
        },
    )


def _make_replicates(
    observed: StudyJSON,
    *,
    sample_size: int,
    source: str,
) -> list[StudyJSON]:
    """Create deterministic replicate studies preserving the observed schedule."""

    replicates: list[StudyJSON] = []
    for sample_idx in range(sample_size):
        context: list[IndividualJSON] = []
        for individual in observed["context"]:
            delta = float(sample_idx + 1)
            context.append(
                IndividualJSON(
                    name_id=individual["name_id"],
                    observation_times=list(individual["observation_times"]),
                    observations=[
                        float(value) + delta for value in individual["observations"]
                    ],
                    dosing=list(individual["dosing"]),
                    dosing_type=list(individual["dosing_type"]),
                    dosing_times=list(individual["dosing_times"]),
                    dosing_name=list(individual["dosing_name"]),
                )
            )
        replicates.append(
            StudyJSON(
                context=context,
                target=[],
                meta_data={
                    "study_name": observed["meta_data"]["study_name"],
                    "substance_name": observed["meta_data"]["substance_name"],
                    "source": source,
                },
            )
        )
    return replicates


def _make_vpc_batch(observed: StudyJSON) -> AICMECompartmentsDataBatch:
    """Convert a tiny observed study into the VPC batch shape expected by the sampler."""

    B, c_ind, t_ind, T = 1, 2, 1, 3
    context_obs = torch.tensor(
        [
            [
                [[10.0], [20.0], [30.0]],
                [[11.0], [21.0], [31.0]],
            ]
        ],
        dtype=torch.float32,
    )  # [B, c_ind, T, 1]
    context_time = torch.tensor(
        [
            [
                [[0.0], [1.0], [2.0]],
                [[0.0], [1.0], [2.0]],
            ]
        ],
        dtype=torch.float32,
    )  # [B, c_ind, T, 1]
    context_mask = torch.ones(B, c_ind, T, dtype=torch.bool)  # [B, c_ind, T]

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
        context_dosing_amounts=torch.tensor([[100.0, 120.0]], dtype=torch.float32),
        context_dosing_route_types=torch.tensor([[0, 1]], dtype=torch.long),
        mask_context_individuals=torch.ones(B, c_ind, dtype=torch.bool),
        mask_target_individuals=torch.zeros(B, t_ind, dtype=torch.bool),
        study_name=[observed["meta_data"]["study_name"]],
        context_subject_name=[
            [individual["name_id"] for individual in observed["context"]],
        ],
        target_subject_name=[["target_0"]],
        substance_name=[observed["meta_data"]["substance_name"]],
        time_scales=torch.zeros(B, 2),
        is_empirical=True,
    )


class _DummySyntheticVPCDataModule:
    """Minimal datamodule stub for the paired-VPC task tests."""

    def __init__(self) -> None:
        self.generate_calls: list[tuple[int, int, int]] = []
        self.batch_build_calls: list[str] = []

    def generate_synthetic_vpc_data_list(
        self,
        n_cases: int,
        n_observed_individuals: int,
        sample_size: int,
    ) -> list[tuple[StudyJSON, list[StudyJSON]]]:
        self.generate_calls.append((n_cases, n_observed_individuals, sample_size))
        cases: list[tuple[StudyJSON, list[StudyJSON]]] = []
        for case_index in range(n_cases):
            observed = _make_observed_study(case_index)
            cases.append(
                (
                    observed,
                    _make_replicates(observed, sample_size=sample_size, source="truth"),
                )
            )
        return cases

    def _build_synthetic_vpc_evaluation_batch(
        self,
        study: StudyJSON,
    ) -> AICMECompartmentsDataBatch:
        self.batch_build_calls.append(study["meta_data"]["study_name"])
        return _make_vpc_batch(study)


class _DummyModelVPCSampler:
    """Minimal sampler stub exposing the model-side VPC API."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.model_config = SimpleNamespace(name_str="DummySampler")
        self.sample_calls: list[dict[str, int | None]] = []

    def sample_new_individuals_to_vpc_format(
        self,
        batch: AICMECompartmentsDataBatch,
        sample_size: int = 8,
        num_steps: int | None = None,
    ) -> list[list[StudyJSON]]:
        observed = _make_observed_study(0)
        observed["meta_data"]["study_name"] = batch.study_name[0]
        observed["meta_data"]["substance_name"] = batch.substance_name[0]
        self.sample_calls.append(
            {
                "batch_size": int(batch.context_obs.shape[0]),
                "sample_size": int(sample_size),
                "num_steps": None if num_steps is None else int(num_steps),
            }
        )
        replicates = _make_replicates(observed, sample_size=sample_size, source="model")
        return [replicates]


def _patch_lightweight_vpc_helpers(monkeypatch):
    """Replace heavy VPC helpers with small deterministic plot spies."""

    compute_calls: list[tuple[str, str]] = []
    plot_calls: list[str] = []

    def fake_compute_vpc_data(observed, simulated, **kwargs):
        _ = kwargs
        branch = str(simulated[0]["meta_data"].get("source", "unknown"))
        compute_calls.append((observed["meta_data"]["study_name"], branch))
        return {"branch": branch}

    def fake_vpc_plot(vpc_data, ax, log_y=False):
        _ = log_y
        plot_calls.append(str(vpc_data["branch"]))
        ax.plot([0.0, 1.0], [0.0, 1.0])

    monkeypatch.setattr(pk_task_synthetic, "compute_vpc_data", fake_compute_vpc_data)
    monkeypatch.setattr(pk_task_synthetic, "vpc_plot", fake_vpc_plot)
    return compute_calls, plot_calls


def test_task_synthetic_vpc_paired_images_uses_default_task_cfg(tmp_path, monkeypatch) -> None:
    """Default task_cfg should drive native case generation with the planned values."""

    compute_calls, plot_calls = _patch_lightweight_vpc_helpers(monkeypatch)
    datamodule = _DummySyntheticVPCDataModule()
    model = _DummyModelVPCSampler()
    trainer = SimpleNamespace(
        datamodule=datamodule,
        default_root_dir=tmp_path,
        current_epoch=2,
    )

    outputs = task_synthetic_vpc_paired_images(
        samples=None,
        batches=[],
        task_cfg={},
        trainer=trainer,
        pl_module=model,
    )

    assert datamodule.generate_calls == [(10, 10, 500)]
    assert len(datamodule.batch_build_calls) == 10
    assert all(call["sample_size"] == 500 for call in model.sample_calls)
    assert len(outputs) == 10
    assert len(compute_calls) == 20
    assert len(plot_calls) == 20


def test_task_synthetic_vpc_paired_images_uses_explicit_sample_size(tmp_path, monkeypatch) -> None:
    """Explicit task_cfg.sample_size should override the default model sampling count."""

    _patch_lightweight_vpc_helpers(monkeypatch)
    datamodule = _DummySyntheticVPCDataModule()
    model = _DummyModelVPCSampler()
    trainer = SimpleNamespace(
        datamodule=datamodule,
        default_root_dir=tmp_path,
        current_epoch=1,
    )

    _ = task_synthetic_vpc_paired_images(
        samples=None,
        batches=[],
        task_cfg={
            "n_cases": 2,
            "sample_size": 7,
            "n_observed_individuals": 3,
        },
        trainer=trainer,
        pl_module=model,
    )

    assert datamodule.generate_calls == [(2, 3, 7)]
    assert [call["sample_size"] for call in model.sample_calls] == [7, 7]


def test_task_synthetic_vpc_paired_images_saves_one_image_per_case(tmp_path, monkeypatch) -> None:
    """The paired-VPC task should save one non-empty PNG for each generated case."""

    _patch_lightweight_vpc_helpers(monkeypatch)
    datamodule = _DummySyntheticVPCDataModule()
    model = _DummyModelVPCSampler()
    trainer = SimpleNamespace(
        datamodule=datamodule,
        default_root_dir=tmp_path,
        current_epoch=3,
    )

    outputs = task_synthetic_vpc_paired_images(
        samples=None,
        batches=[],
        task_cfg={
            "n_cases": 3,
            "sample_size": 4,
            "n_observed_individuals": 2,
        },
        trainer=trainer,
        pl_module=model,
    )

    assert len(outputs) == 3
    for image_path in outputs.values():
        assert image_path.exists()
        assert image_path.stat().st_size > 0


def test_task_synthetic_vpc_paired_images_uses_vpc_helpers_for_truth_and_model(
    tmp_path,
    monkeypatch,
) -> None:
    """Each case should call compute/plot once for truth and once for model samples."""

    compute_calls, plot_calls = _patch_lightweight_vpc_helpers(monkeypatch)
    datamodule = _DummySyntheticVPCDataModule()
    model = _DummyModelVPCSampler()
    trainer = SimpleNamespace(
        datamodule=datamodule,
        default_root_dir=tmp_path,
        current_epoch=0,
    )

    _ = task_synthetic_vpc_paired_images(
        samples=None,
        batches=[],
        task_cfg={
            "n_cases": 2,
            "sample_size": 3,
            "n_observed_individuals": 2,
        },
        trainer=trainer,
        pl_module=model,
    )

    assert compute_calls == [
        ("study_0", "truth"),
        ("study_0", "model"),
        ("study_1", "truth"),
        ("study_1", "model"),
    ]
    assert plot_calls == ["truth", "model", "truth", "model"]


def _aicme_potsdam_config_for_synthetic_vpc() -> NodePKExperimentConfig:
    """Load the canonical Potsdam AICME config used by the paired-VPC smoke test."""

    base_yaml = (
        Path(config_dir)
        / "experiment_configs"
        / "UAI"
        / "Rebuttal"
        / "Potsdam"
        / "aicme-t-pk"
        / "base.yaml"
    )
    cfg = NodePKExperimentConfig.from_yaml(str(base_yaml))
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
    return cfg


def test_task_synthetic_vpc_paired_images_aicme_potsdam_smoke(tmp_path) -> None:
    """A small CPU Potsdam+AICME run should save one paired synthetic VPC image."""

    torch.manual_seed(19)
    cfg = _aicme_potsdam_config_for_synthetic_vpc()
    datamodule = AICMECompartmentsDataModule(cfg)
    datamodule.prepare_data()
    datamodule.setup()

    model = AICMEPK(cfg)
    model.eval()

    outputs = task_synthetic_vpc_paired_images(
        samples=None,
        batches=[],
        task_cfg={
            "n_cases": 1,
            "sample_size": 4,
            "n_observed_individuals": 3,
        },
        trainer=SimpleNamespace(
            datamodule=datamodule,
            default_root_dir=tmp_path,
            current_epoch=0,
        ),
        pl_module=model,
    )

    assert len(outputs) == 1
    image_path = next(iter(outputs.values()))
    assert image_path.exists()
    assert image_path.stat().st_size > 0
