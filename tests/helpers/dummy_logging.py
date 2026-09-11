"""Simple logging stubs shared across test suites.

These helpers simulate the Lightning trainer/logger/experiment stack so
model logging utilities can be exercised without relying on external
services.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DummyExperiment:
    """Minimal experiment handle recording metrics and images."""

    logged_metrics: List[Dict[str, Any]] = field(default_factory=list)
    logged_images: List[Dict[str, Any]] = field(default_factory=list)
    datamodule: Any | None = None

    def log_metric(self, name: str, value: float, step: int) -> None:
        self.logged_metrics.append({"name": name, "value": value, "step": step})

    def log_image(self, path: str, name: str, step: int) -> None:
        self.logged_images.append({"path": path, "name": name, "step": step})


@dataclass
class DummyLogger:
    """Simple logger wrapper exposing the experiment object."""

    experiment: DummyExperiment


class _DummyProgressBar:
    """No-op stub for the trainer progress bar callback."""

    def __init__(self) -> None:
        self.is_enabled = False

    def disable(self) -> None:  # pragma: no cover - behaviour-less stub
        pass

    def enable(self) -> None:  # pragma: no cover - behaviour-less stub
        pass


class DummyTrainer:
    """Trainer stub exposing expected Lightning attributes."""

    def __init__(self, logger: DummyLogger, datamodule=None, epoch: int = 0):
        self.logger = logger
        self.datamodule = datamodule
        self.is_global_zero = True
        self.progress_bar_callback = _DummyProgressBar()
        self.current_epoch = epoch  # <-- THIS IS ALL LIGHTNING CHECKS FOR
