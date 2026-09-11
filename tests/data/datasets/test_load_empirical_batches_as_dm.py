"""Tests for :func:`load_empirical_json_batches_as_dm`."""

from pathlib import Path

import pytest
import torch

from pff.data.datasets.aicme_datasets import (
    AICMECompartmentsDataBatch,
    AICMECompartmentsDataModule,
)
from pff.data.data_empirical import (
    load_empirical_json_batches_as_dm,
    load_empirical_hf_batches_as_dm
)

from tests.models.test_aicme_pk import _aicme_config

def _add_batch_dim(batch: AICMECompartmentsDataBatch) -> AICMECompartmentsDataBatch:
    """Ensure a leading batch dimension ``B=1`` in tensor fields."""

    def maybe_unsqueeze(t):
        return t.unsqueeze(0) if isinstance(t, torch.Tensor) else t

    return AICMECompartmentsDataBatch(*[maybe_unsqueeze(t) for t in batch])


def _assert_empirical_shapes_match_dm_strategies(
    dm: AICMECompartmentsDataModule, batch: AICMECompartmentsDataBatch
) -> None:
    """Validate empirical batch capacities against datamodule strategy shapes."""

    ctx_obs_cap, ctx_rem_cap = dm.context_strategy.get_shapes()
    tgt_strategy = dm.empirical_target_strategy or dm.target_strategy
    tgt_obs_cap, tgt_rem_cap = tgt_strategy.get_shapes()

    assert batch.context_obs.shape[2] == ctx_obs_cap
    assert batch.context_obs_time.shape[2] == ctx_obs_cap
    assert batch.context_obs_mask.shape[2] == ctx_obs_cap
    assert batch.context_rem_sim.shape[2] == ctx_rem_cap
    assert batch.context_rem_sim_time.shape[2] == ctx_rem_cap
    assert batch.context_rem_sim_mask.shape[2] == ctx_rem_cap

    assert batch.target_obs.shape[2] == tgt_obs_cap
    assert batch.target_obs_time.shape[2] == tgt_obs_cap
    assert batch.target_obs_mask.shape[2] == tgt_obs_cap
    assert batch.target_rem_sim.shape[2] == tgt_rem_cap
    assert batch.target_rem_sim_time.shape[2] == tgt_rem_cap
    assert batch.target_rem_sim_mask.shape[2] == tgt_rem_cap

def test_load_empirical_batches_as_dm_matches_datamodule() -> None:
    cfg = _aicme_config()
    cfg.mix_data.test_empirical_datasets = []
    dm = AICMECompartmentsDataModule(cfg)
    dm.prepare_data()
    dm.setup()

    json_path = Path(__file__).resolve().parents[1] / "fixtures" / "studies_long_list.json"
    batches = load_empirical_json_batches_as_dm(json_path, meta_dosing=cfg.dosing, datamodule=dm)

    assert batches
    first_batch = batches[0]
    _assert_empirical_shapes_match_dm_strategies(dm, first_batch)

def test_load_empirical_batches_as_dm_matches_datamodule_from_lenuzza() -> None:
    from pff import data_dir
    cfg = _aicme_config()
    cfg.mix_data.test_empirical_datasets = []
    dm = AICMECompartmentsDataModule(cfg)
    dm.prepare_data()
    dm.setup()

    json_path = Path(data_dir) / "preprocessed" / "lenuzza_2016.json"

    batches = load_empirical_json_batches_as_dm(json_path, meta_dosing=cfg.dosing, datamodule=dm)

    assert batches
    first_batch = batches[0]
    _assert_empirical_shapes_match_dm_strategies(dm, first_batch)

def test_load_empirical_batches_as_dm_matches_datamodule_from_hf() -> None:
    cfg = _aicme_config()
    cfg.mix_data.test_empirical_datasets = []
    dm = AICMECompartmentsDataModule(cfg)
    dm.prepare_data()
    dm.setup()

    try:
        batches = load_empirical_hf_batches_as_dm(
            "cesarali/Indometacin",
            meta_dosing=cfg.dosing,
            datamodule=dm,
        )
    except PermissionError as exc:
        pytest.skip(f"Hugging Face cache lock is not writable in this environment: {exc}")

    assert batches
    first_batch = batches[0]
    _assert_empirical_shapes_match_dm_strategies(dm, first_batch)


if __name__ == "__main__":
    test_load_empirical_batches_as_dm_matches_datamodule_from_lenuzza()
