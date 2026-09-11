"""Tests for training utility checkpoint resolution."""

from __future__ import annotations

from pathlib import Path

import torch

from pff.training.utils import load_model_from_checkpoint_path
from pff.training.utils import get_lightning_checkpoint_path


def test_get_lightning_checkpoint_path_resolves_scheduler_metric_checkpoint(
    tmp_path: Path,
) -> None:
    scheduler_checkpoint = (
        tmp_path
        / "scheduler_metric_checkpoints"
        / "empirical_summary"
        / "best-epoch_049-step_0003350-log_rmse=1.523099.ckpt"
    )
    scheduler_checkpoint.parent.mkdir(parents=True)
    scheduler_checkpoint.write_text("x", encoding="utf-8")

    resolved = get_lightning_checkpoint_path(tmp_path, "log_rmse")

    assert resolved == scheduler_checkpoint


def test_get_lightning_checkpoint_path_accepts_explicit_checkpoint_file(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "custom.ckpt"
    checkpoint_path.write_text("x", encoding="utf-8")

    resolved = get_lightning_checkpoint_path(tmp_path, str(checkpoint_path))

    assert resolved == checkpoint_path


class _DummyModule(torch.nn.Module):
    def __init__(self, model_config=None) -> None:
        super().__init__()
        self.model_config = model_config
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.loaded_from: str | None = None

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        map_location,
        strict: bool,
        model_config,
    ) -> "_DummyModule":
        module = cls(model_config=model_config)
        module.loaded_from = "lightning"
        return module


def test_load_model_from_checkpoint_path_uses_lightning_loader_for_full_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "lightning.ckpt"
    torch.save({"pytorch-lightning_version": "2.5.0"}, checkpoint_path)

    model = load_model_from_checkpoint_path(
        _DummyModule,
        checkpoint_path,
        model_config={"name": "cfg"},
    )

    assert isinstance(model, _DummyModule)
    assert model.loaded_from == "lightning"
    assert model.model_config == {"name": "cfg"}


def test_load_model_from_checkpoint_path_loads_scheduler_state_dict(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "scheduler.ckpt"
    expected = _DummyModule(model_config={"name": "cfg"})
    with torch.no_grad():
        expected.weight.copy_(torch.tensor([1.5, -2.0]))
    torch.save({"state_dict": expected.state_dict()}, checkpoint_path)

    model = load_model_from_checkpoint_path(
        _DummyModule,
        checkpoint_path,
        model_config={"name": "cfg"},
    )

    assert isinstance(model, _DummyModule)
    assert model.loaded_from is None
    assert model.model_config == {"name": "cfg"}
    torch.testing.assert_close(model.weight.detach(), torch.tensor([1.5, -2.0]))
