from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
import torch
from torch import nn

from transport_processes.config_classes.training_config import TrainerConfig
from transport_processes.models.base_outputs import BaseForwardOutputs


class BaseLightningModule(pl.LightningModule):
    """Shared LightningModule that consumes ``BaseForwardOutputs`` instances."""

    def __init__(self, model: nn.Module, trainer_cfg: TrainerConfig):
        super().__init__()
        self.model = model
        self.trainer_cfg = trainer_cfg
        self._visualization_checkpoint = self._build_visualization_checkpoint()

    def _build_visualization_checkpoint(self):
        build_fn = getattr(self.model, "build_visualization_checkpoint", None)
        if callable(build_fn):
            return build_fn()
        return None

    def forward(self, batch: Any) -> BaseForwardOutputs:
        return self.model(batch)

    def _log_comet_metric(self, name: str, value: torch.Tensor, *, step: int | None) -> None:
        trainer = getattr(self, "_trainer", None)
        logger = getattr(trainer, "logger", None) if trainer is not None else None
        experiment = getattr(logger, "experiment", None) if logger else None
        if experiment is None:
            return
        experiment.log_metric(name=name, value=value, step=step)

    def _shared_step(self, batch: Any, prefix: str) -> torch.Tensor:
        fwd = self.model(batch)
        if not isinstance(fwd, BaseForwardOutputs):
            raise TypeError("Model forward must return a BaseForwardOutputs instance.")

        loss_combiner = None
        if getattr(self.model, "use_multihead_loss", False):
            loss_combiner = getattr(self.model, "multihead_loss", None)
        loss = fwd.aggregate_loss(loss_combiner=loss_combiner)
        checkpoint_key = fwd.CHECKPOINT_KEY
        on_step = prefix == "train"

        for key in fwd.LOG_KEYS:
            value = fwd.get(key)
            self.log(
                f"{prefix}/{key}",
                value,
                prog_bar=(key == checkpoint_key),
                on_step=on_step,
                on_epoch=True,
            )
            # self._log_comet_metric(name=f"{prefix}/{key}", value=value, step=self.global_step)

        self.log(
            f"{prefix}/loss",
            loss,
            prog_bar=True,
            on_step=on_step,
            on_epoch=True,
        )
        # self._log_comet_metric(name=f"{prefix}/loss", value=loss, step=self.global_step)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, prefix="train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, prefix="val")

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, prefix="test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.trainer_cfg.learning_rate,
            weight_decay=self.trainer_cfg.weight_decay,
        )
        return optimizer
