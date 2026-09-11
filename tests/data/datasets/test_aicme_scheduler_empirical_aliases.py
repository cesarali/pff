"""Unit tests for scheduler-facing empirical batch aliases."""

from __future__ import annotations

import pytest

from pff.data.datasets.aicme_datasets import AICMECompartmentsDataModule


class _DummyEmpiricalModule:
    def get_empirical_test_batches(self, *, no_heldout: bool = False, device=None):
        del device
        if no_heldout:
            return {"repo_b": ["no_heldout_batch"]}
        return {"repo_a": ["heldout_batch"]}


def test_get_empirical_batches_routes_to_heldout_alias() -> None:
    dm = _DummyEmpiricalModule()
    batches = AICMECompartmentsDataModule.get_empirical_batches(
        dm,
        split="empirical_heldout",
        empirical_name="repo_a",
    )
    assert batches == ["heldout_batch"]


def test_get_empirical_batches_routes_to_no_heldout_alias() -> None:
    dm = _DummyEmpiricalModule()
    batches = AICMECompartmentsDataModule.get_empirical_batches(
        dm,
        split="empirical_no_heldout",
        empirical_name="repo_b",
    )
    assert batches == ["no_heldout_batch"]


def test_get_empirical_batches_rejects_unknown_alias() -> None:
    dm = _DummyEmpiricalModule()
    with pytest.raises(ValueError, match="Unsupported empirical split alias"):
        AICMECompartmentsDataModule.get_empirical_batches(
            dm,
            split="empirical",
            empirical_name="repo_a",
        )
