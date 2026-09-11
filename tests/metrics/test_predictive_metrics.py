"""Smoke tests for predictive-metrics task execution."""

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
from pff.config_classes.flow_pk_config import FlowPKExperimentConfig
from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.data.datasets.aicme_batch import AICMECompartmentsDataBatch
from pff.training.basic_experiment import BasicLightningExperiment
from pff.training.callbacks.pk_task_empirical import (
    task_empirical_predictive_metrics,
)

FLOW_EXPERIMENT_YAML = (
    Path(config_dir)
    / "experiment_configs"
    / "UAI"
    / "Submission"
    / "flow-pk-predict-n-generate-test"
    / "flowPK.yaml"
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


class _DummyCrossRepoEmpiricalDatamodule:
    """Datamodule stub returning one held-out permutation list per repo."""

    def __init__(self, batch_lists_by_repo: dict[str, list[object]]) -> None:
        self._batch_lists_by_repo = batch_lists_by_repo
        self.calls: list[tuple[str, str]] = []

    def get_empirical_batches(
        self,
        *,
        split: str,
        empirical_name: str,
        device=None,
    ) -> list[object]:
        del device
        self.calls.append((split, empirical_name))
        return self._batch_lists_by_repo[empirical_name]


def _build_small_flow_predictive_config(tmp_path: Path) -> FlowPKExperimentConfig:
    """Load the real FlowPK config and shrink it for a fast smoke test."""

    cfg = FlowPKExperimentConfig.from_yaml(str(FLOW_EXPERIMENT_YAML))
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
        test_empirical_datasets=[],
    )
    cfg.meta_study = replace(
        cfg.meta_study,
        num_individuals_range=(3, 3),
        time_num_steps=20,
    )
    cfg.context_observations = replace(cfg.context_observations, max_num_obs=10)
    cfg.target_observations = replace(cfg.target_observations, max_num_obs=10)
    cfg.upload_to_hf_hub = False
    cfg.my_results_path = str(tmp_path)
    return cfg


def _build_small_flow_empirical_predictive_config(tmp_path: Path) -> FlowPKExperimentConfig:
    """Load the real FlowPK config and shrink it for real empirical predictive smoke tests."""

    cfg = FlowPKExperimentConfig.from_yaml(str(FLOW_EXPERIMENT_YAML))
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
    cfg.upload_to_hf_hub = False
    cfg.my_results_path = str(tmp_path)
    return cfg


def _build_small_aicme_predictive_config(tmp_path: Path) -> NodePKExperimentConfig:
    """Load the real AICMEPK config and shrink it for a fast smoke test."""

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
        test_empirical_datasets=[],
    )
    cfg.meta_study = replace(
        cfg.meta_study,
        num_individuals_range=(3, 3),
        time_num_steps=20,
    )
    cfg.context_observations = replace(cfg.context_observations, max_num_obs=10)
    cfg.target_observations = replace(cfg.target_observations, max_num_obs=10)
    cfg.network = replace(cfg.network, aggregator_type="mean")
    cfg.upload_to_hf_hub = False
    cfg.my_results_path = str(tmp_path)
    return cfg


def _build_small_aicme_empirical_predictive_config(tmp_path: Path) -> NodePKExperimentConfig:
    """Load the real AICME config and shrink it for real empirical predictive smoke tests."""

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


def _first_batch_list_from_datamodule(datamodule) -> list:
    """Return the first permutation list from the train dataloader on CPU."""

    datamodule.prepare_data()
    datamodule.setup()
    batch_list = next(iter(datamodule.train_dataloader()))
    assert isinstance(batch_list, list) and len(batch_list) >= 1
    return [batch.to_device(torch.device("cpu")) for batch in batch_list]


def _slice_batch_axis(
    batch: AICMECompartmentsDataBatch,
    keep_indices: list[int],
) -> AICMECompartmentsDataBatch:
    """Return one databatch restricted to selected batch-axis entries."""

    batch_size = int(batch.target_obs.shape[0])
    index_tensor = torch.tensor(keep_indices, dtype=torch.long, device=batch.target_obs.device)
    sliced_fields = {}
    for field_name in batch._fields:
        value = getattr(batch, field_name)
        if isinstance(value, torch.Tensor):
            if value.ndim > 0 and int(value.shape[0]) == batch_size:
                sliced_fields[field_name] = value.index_select(0, index_tensor)
            else:
                sliced_fields[field_name] = value
        elif isinstance(value, list) and len(value) == batch_size:
            sliced_fields[field_name] = [value[idx] for idx in keep_indices]
        else:
            sliced_fields[field_name] = value
    return batch._replace(**sliced_fields)


def _filter_batches_with_valid_heldout_targets(
    batch_list: list[AICMECompartmentsDataBatch],
) -> list[AICMECompartmentsDataBatch]:
    """Keep only empirical batch rows with at least one valid held-out target."""

    filtered_batches: list[AICMECompartmentsDataBatch] = []
    for batch in batch_list:
        valid_rows = (
            batch.target_obs_mask.bool() & batch.mask_target_individuals.unsqueeze(-1)
        ).any(dim=(1, 2))
        keep_indices = torch.nonzero(valid_rows, as_tuple=False).view(-1).tolist()
        if not keep_indices:
            continue
        filtered_batches.append(_slice_batch_axis(batch, keep_indices))
    return filtered_batches


def _align_empirical_batches_on_shared_slots(
    batch_list: list[AICMECompartmentsDataBatch],
) -> list[AICMECompartmentsDataBatch]:
    """Restrict a compatible permutation subset to one shared ordered set of slots."""

    if not batch_list:
        return []

    label_lists = [
        list(zip(batch.study_name, batch.substance_name, strict=False)) for batch in batch_list
    ]

    best_batch_indices: list[int] = []
    best_common_labels: list[tuple[str, str]] = []
    for start_idx, base_labels in enumerate(label_lists):
        common_labels = list(base_labels)
        current_indices = [start_idx]
        if len(common_labels) >= 2 and len(common_labels) > len(best_common_labels):
            best_batch_indices = list(current_indices)
            best_common_labels = list(common_labels)

        for next_idx in range(start_idx + 1, len(label_lists)):
            next_label_set = set(label_lists[next_idx])
            candidate_common = [label for label in common_labels if label in next_label_set]
            if len(candidate_common) < 2:
                continue
            current_indices.append(next_idx)
            common_labels = candidate_common
            if len(common_labels) > len(best_common_labels) or (
                len(common_labels) == len(best_common_labels)
                and len(current_indices) > len(best_batch_indices)
            ):
                best_batch_indices = list(current_indices)
                best_common_labels = list(common_labels)

    if not best_batch_indices or not best_common_labels:
        return []

    aligned_batches: list[AICMECompartmentsDataBatch] = []
    for batch_idx in best_batch_indices:
        batch = batch_list[batch_idx]
        labels = label_lists[batch_idx]
        index_by_label = {label: idx for idx, label in enumerate(labels)}
        keep_indices = [index_by_label[label] for label in best_common_labels]
        aligned_batches.append(_slice_batch_axis(batch, keep_indices))
    return aligned_batches


def _assert_task_empirical_predictive_metrics_smoke_from_experiment(
    exp_config: FlowPKExperimentConfig | NodePKExperimentConfig,
) -> None:
    """Build one real experiment and run the empirical predictive metrics task."""

    basic_experiment = BasicLightningExperiment.from_config(exp_config, map_location="cpu")
    datamodule = basic_experiment.get_datamodule()
    model = basic_experiment.get_module().to(torch.device("cpu"))
    empirical_name = exp_config.mix_data.test_empirical_datasets[0]
    original_get_empirical_batches = datamodule.get_empirical_batches

    def _get_empirical_batches_with_valid_targets(
        *,
        split: str,
        empirical_name: str,
        device=None,
    ) -> list[AICMECompartmentsDataBatch]:
        batch_list = original_get_empirical_batches(
            split=split,
            empirical_name=empirical_name,
            device=device,
        )
        filtered = _align_empirical_batches_on_shared_slots(
            _filter_batches_with_valid_heldout_targets(batch_list)
        )
        assert filtered, "Expected at least one empirical held-out batch with valid targets."
        return filtered

    datamodule.get_empirical_batches = _get_empirical_batches_with_valid_targets  # type: ignore[method-assign]
    trainer = SimpleNamespace(
        datamodule=datamodule,
        default_root_dir=Path(str(exp_config.my_results_path)),
        current_epoch=0,
        global_step=0,
    )

    result = task_empirical_predictive_metrics(
        samples=None,
        batches=[],
        task_cfg={
            "sample_size": 2,
            "split": "empirical_heldout",
            "empirical_name": empirical_name,
            "repo_id": empirical_name,
        },
        trainer=trainer,
        pl_module=model,
    )

    assert result
    assert any(key.endswith("/rmse") for key in result)
    assert all(
        torch.isfinite(torch.tensor(value, dtype=torch.float32)) for value in result.values()
    )


@pytest.mark.parametrize(
    ("model_name", "build_config"),
    [
        ("flowpk", _build_small_flow_predictive_config),
        ("aicme", _build_small_aicme_predictive_config),
    ],
)
def test_task_predictive_metrics_smoke_from_experiment(
    model_name: str,
    build_config,
) -> None:
    """Empirical predictive metrics task should run end to end for both PK models."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch.object(BasicLightningExperiment, "_setup_logger", _setup_dummy_logger):
            exp_config = build_config(Path(tmp_dir))
            basic_experiment = BasicLightningExperiment.from_config(exp_config, map_location="cpu")
            datamodule = basic_experiment.get_datamodule()
            model = basic_experiment.get_module().to(torch.device("cpu"))
            batch_list = _first_batch_list_from_datamodule(datamodule)

            def _get_empirical_batches(*, split: str, empirical_name: str, device=None):
                del empirical_name
                del device
                assert split == "empirical_heldout"
                return batch_list

            datamodule.get_empirical_batches = _get_empirical_batches  # type: ignore[method-assign]
            trainer = SimpleNamespace(
                datamodule=datamodule,
                default_root_dir=Path(tmp_dir),
                current_epoch=0,
                global_step=0,
            )

            result = task_empirical_predictive_metrics(
                samples=None,
                batches=[],
                task_cfg={
                    "sample_size": 2,
                    "split": "empirical_heldout",
                    "empirical_name": f"local/{model_name}",
                    "repo_id": f"local/{model_name}",
                    "model_label": model_name,
                },
                trainer=trainer,
                pl_module=model,
            )

            assert result
            assert any(key.endswith("/rmse") for key in result)
            assert all(
                torch.isfinite(torch.tensor(value, dtype=torch.float32))
                for value in result.values()
            )


def test_task_empirical_predictive_metrics_pools_all_configured_repos_by_substance() -> None:
    """Predictive metrics should pool raw held-out observations across repos per substance."""

    datamodule = _DummyCrossRepoEmpiricalDatamodule(
        batch_lists_by_repo={
            "repo_a": ["repo_a_batch_list"],
            "repo_b": ["repo_b_batch_list"],
        }
    )
    trainer = SimpleNamespace(
        datamodule=datamodule,
        default_root_dir=Path("."),
        current_epoch=0,
        global_step=0,
    )
    model = SimpleNamespace(
        model_config=SimpleNamespace(
            mix_data=SimpleNamespace(test_empirical_datasets=["repo_a", "repo_b"])
        )
    )

    def _fake_collect_observations(
        batch_list,
        *,
        model,
        sample_size: int,
    ) -> dict[str, list[dict[str, float]]]:
        del model
        assert sample_size == 2
        repo_token = batch_list[0]
        if repo_token == "repo_a_batch_list":
            return {
                "Indometacin": [
                    {"rmse": 1.0, "log_rmse": 11.0, "r2": 0.1, "log_r2": 1.1},
                ],
                "Theophylline": [
                    {"rmse": 8.0, "log_rmse": 18.0, "r2": 0.8, "log_r2": 1.8},
                ],
            }
        if repo_token == "repo_b_batch_list":
            return {
                "indometacin": [
                    {"rmse": 3.0, "log_rmse": 13.0, "r2": 0.3, "log_r2": 1.3},
                    {"rmse": 5.0, "log_rmse": 15.0, "r2": 0.5, "log_r2": 1.5},
                ],
                "THEOPHYLLINE": [
                    {"rmse": 10.0, "log_rmse": 20.0, "r2": 1.0, "log_r2": 2.0},
                ],
            }
        raise AssertionError(f"Unexpected batch_list token: {repo_token}")

    with patch(
        "pff.training.callbacks.pk_tasks."
        "_collect_empirical_predictive_metric_observations_from_batch_list",
        side_effect=_fake_collect_observations,
    ):
        result = task_empirical_predictive_metrics(
            samples=None,
            batches=[],
            task_cfg={
                "sample_size": 2,
                "split": "empirical_heldout",
            },
            trainer=trainer,
            pl_module=model,
        )

    assert datamodule.calls == [
        ("empirical_heldout", "repo_a"),
        ("empirical_heldout", "repo_b"),
    ]
    assert result["Indometacin/rmse"] == pytest.approx(3.0)
    assert result["Indometacin/rmse_std"] == pytest.approx(2.0)
    assert result["Indometacin/log_rmse"] == pytest.approx(13.0)
    assert result["Theophylline/rmse"] == pytest.approx(9.0)
    assert result["Theophylline/rmse_std"] == pytest.approx(2.0**0.5)


def test_task_empirical_predictive_metrics_smoke_from_flow_experiment() -> None:
    """Empirical predictive metrics should run end to end from a YAML-backed FlowPK config."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch.object(BasicLightningExperiment, "_setup_logger", _setup_dummy_logger):
            _assert_task_empirical_predictive_metrics_smoke_from_experiment(
                _build_small_flow_empirical_predictive_config(Path(tmp_dir))
            )


def test_task_empirical_predictive_metrics_smoke_from_aicme_experiment() -> None:
    """Empirical predictive metrics should run end to end from a YAML-backed AICME config."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch.object(BasicLightningExperiment, "_setup_logger", _setup_dummy_logger):
            _assert_task_empirical_predictive_metrics_smoke_from_experiment(
                _build_small_aicme_empirical_predictive_config(Path(tmp_dir))
            )


if __name__ == "__main__":
    test_task_empirical_predictive_metrics_smoke_from_aicme_experiment()
