import importlib
import os
from dataclasses import asdict, is_dataclass
from typing import List, Optional, Tuple, Type, Union

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CometLogger

from transport_processes import COMET_KEY, results_dir
from transport_processes.config_classes.data_config import TransportProcessDataConfig
from transport_processes.config_classes.experiment_config import (
    NeuralProcessExperimentConfig,
    WassersteinAutoencoderProcessExperimentConfig,
)
from transport_processes.data import build_datamodule
from transport_processes.data.base_datamodule import BaseTransportProcessDataModule
from transport_processes.models.neural_process import NeuralProcessModel
from transport_processes.models.wasserstein_autoencoder_process.wae import (
    WassersteinAutoencoderProcessModel,
)
from transport_processes.training.base_lightning_module import BaseLightningModule


def _select_devices_and_strategy(
    devices: Optional[Union[int, List[int]]],
    strategy: Optional[str],
) -> Tuple[Union[int, List[int]], str]:
    """Resolve accelerator devices and distributed strategy."""

    if devices is None:
        devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

    if isinstance(devices, (list, tuple)):
        ddp_flag = len(devices) > 1
    else:
        ddp_flag = bool(devices and devices > 1)

    resolved_strategy = strategy if strategy is not None else ("ddp" if ddp_flag else "auto")
    return devices, resolved_strategy


def _parse_devices_value(
    raw_value: Optional[str],
) -> Optional[Union[int, List[int]]]:
    if raw_value is None:
        return None
    value = raw_value.strip()
    if not value:
        return None
    lowered = value.lower()
    if lowered in ("none", "null", "auto"):
        return None
    if "," in value:
        items = [item.strip() for item in value.split(",") if item.strip()]
        if not items:
            return None
        try:
            return [int(item) for item in items]
        except ValueError as exc:
            raise ValueError(
                f"Invalid devices list '{raw_value}'. Expected comma-separated integers."
            ) from exc
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid devices value '{raw_value}'. Expected an integer or comma list."
        ) from exc


def _resolve_devices_arg(
    exp_config,
    devices: Optional[Union[int, List[int]]],
) -> Optional[Union[int, List[int]]]:
    if devices is not None:
        return devices

    env_devices = _parse_devices_value(os.getenv("TRANSPORT_PROCESSES_DEVICES"))
    if env_devices is not None:
        return env_devices

    trainer_cfg = getattr(exp_config, "trainer", None)
    if trainer_cfg is not None:
        cfg_devices = getattr(trainer_cfg, "devices", None)
        if isinstance(cfg_devices, str):
            cfg_devices = _parse_devices_value(cfg_devices)
        if cfg_devices is not None:
            return cfg_devices

    return None


def _resolve_strategy_arg(strategy: Optional[str]) -> Optional[str]:
    if strategy is not None:
        return strategy
    env_strategy = os.getenv("TRANSPORT_PROCESSES_STRATEGY")
    if env_strategy is None:
        return None
    env_strategy = env_strategy.strip()
    if not env_strategy or env_strategy.lower() in ("none", "null"):
        return None
    return env_strategy


def get_model_class(config):
    if isinstance(config, NeuralProcessExperimentConfig):
        model_class_path = getattr(config, "model_class", None)
        if model_class_path:
            module_name, class_name = model_class_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        return NeuralProcessModel
    if isinstance(config, WassersteinAutoencoderProcessExperimentConfig):
        return WassersteinAutoencoderProcessModel
    raise TypeError(f"Unsupported experiment config type: {type(config)}")


def get_datamodule_class(config):
    if hasattr(config, "data") and isinstance(config.data, TransportProcessDataConfig):
        return BaseTransportProcessDataModule
    raise TypeError("Experiment config must define a data section compatible with the datamodule.")


class BaseLightningExperiment:
    """High-level wrapper orchestrating Lightning training runs."""

    experiment_name: str = ""

    def __init__(
        self,
        *,
        exp_config=None,
        map_location="cuda",
        devices=None,
        strategy=None,
        strict: bool = True,
        results_root: str | None = None,
    ) -> None:
        self.exp_config = exp_config
        self.map_location = map_location
        resolved_devices = _resolve_devices_arg(exp_config, devices)
        resolved_strategy = _resolve_strategy_arg(strategy)
        self.devices, self.strategy = _select_devices_and_strategy(
            resolved_devices, resolved_strategy
        )
        self.strict = strict
        self._results_root = results_root

        self.MODEL_CLASS_TYPE: Optional[Type] = (
            get_model_class(exp_config) if exp_config is not None else None
        )
        self.DATAMODULE_CLASS_TYPE: Optional[Type] = (
            get_datamodule_class(exp_config) if exp_config is not None else None
        )
        self.model: Optional[BaseLightningModule] = None
        self.datamodule: Optional[BaseTransportProcessDataModule] = None
        self.logger: Optional[CometLogger] = None
        self.experiment_dir: Optional[str] = None
        self.results_dir: Optional[str] = None
        self.callbacks: list[ModelCheckpoint] = []
        self.logger_folder: Optional[str] = None
        self.checkpoint_metric: Optional[str] = None
        self.checkpoint_mode: Optional[str] = None

    @classmethod
    def from_config(
        cls,
        exp_config,
        map_location="cuda",
        devices=None,
        strategy=None,
        strict: bool = True,
        results_root: str | None = None,
    ) -> "BaseLightningExperiment":
        self = cls(
            exp_config=exp_config,
            map_location=map_location,
            devices=devices,
            strategy=strategy,
            strict=strict,
            results_root=results_root,
        )

        self.exp_config = exp_config
        self.MODEL_CLASS_TYPE = get_model_class(exp_config)
        self.DATAMODULE_CLASS_TYPE = get_datamodule_class(exp_config)
        self.experiment_name = self.exp_config.experiment_name

        self._setup_logger()
        self._setup_datamodule()
        self._setup_model()
        self._setup_callbacks()
        return self

    @classmethod
    def from_experiment(
        cls,
        experiment_key: str,
        *,
        model_config_override=None,
        map_location="cuda",
        checkpoint_type: str = "best",
        devices=None,
        strategy=None,
        strict: bool = True,
        results_root: str | None = None,
    ) -> "BaseLightningExperiment":
        raise NotImplementedError

    @classmethod
    def from_hf(
        cls,
        hf_model_id: str,
        *,
        map_location="cuda",
        devices=None,
        strategy=None,
        strict: bool = True,
        results_root: str | None = None,
    ) -> "BaseLightningExperiment":
        raise NotImplementedError

    def _resolve_results_root(self) -> str:
        if self.results_dir is not None:
            return self.results_dir
        if self.exp_config is not None and getattr(self.exp_config, "results_dir", None):
            return self.exp_config.results_dir
        return results_dir

    def _resolve_experiment_dir(self) -> str:
        if self.experiment_dir is not None:
            return self.experiment_dir
        if self.logger is None:
            raise RuntimeError("Logger must be initialised before resolving experiment dir.")

        key = getattr(self.logger, "version", None) or "unknown"
        experiment_dir = os.path.join(
            self.logger_folder,
            self.exp_config.experiment_name,
            str(key),
        )
        self.experiment_dir = experiment_dir
        if self.exp_config is not None:
            self.exp_config.experiment_dir = self.experiment_dir
        return experiment_dir

    def _resolve_checkpoint_metric(self) -> tuple[str, str]:
        if self.model is None or self.model.model is None:
            raise RuntimeError("Model must be initialised before resolving checkpoint metric.")
        checkpoint_key = "val/" + self.model.model.forward_output_class.CHECKPOINT_KEY
        return checkpoint_key, "min"

    def _setup_logger(self, experiment_key: Optional[str] = None) -> None:
        if self.exp_config is None:
            raise RuntimeError("model_config must be set before calling _setup_logger().")

        my_results_path = self._resolve_results_root()
        self.logger_folder = os.path.join(my_results_path, "comet")

        self.logger = CometLogger(
            api_key=COMET_KEY or None,
            project_name=self.exp_config.experiment_name,
            experiment_key=experiment_key,
        )
        self._resolve_experiment_dir()

    def _setup_callbacks(self, experiment_dir: Optional[str] = None) -> None:
        if self.exp_config is None:
            raise RuntimeError("model_config must be set before calling _setup_callbacks().")
        if self.logger is None:
            raise RuntimeError("logger must be configured before _setup_callbacks().")

        metric, mode = self._resolve_checkpoint_metric()
        self.checkpoint_metric = metric
        self.checkpoint_mode = mode

        self.checkpoint_callback_best = ModelCheckpoint(
            dirpath=self.experiment_dir if experiment_dir is None else experiment_dir,
            save_top_k=1,
            monitor=metric,
            mode=mode,
            filename="best-{epoch:02d}-{" + metric + ":.4f}",
        )
        self.checkpoint_callback_last = ModelCheckpoint(
            dirpath=self.experiment_dir if experiment_dir is None else experiment_dir,
            save_last=True,
            monitor=None,
            filename="last",
            save_top_k=0,
        )

        self.callbacks = [
            self.checkpoint_callback_last,
            self.checkpoint_callback_best,
        ]
        visualization_cb = getattr(self.model, "_visualization_checkpoint", None)
        if visualization_cb is not None:
            self.callbacks.append(visualization_cb)

    def _setup_datamodule(self) -> None:
        if self.exp_config is None:
            raise RuntimeError("model_config must be set before calling _setup_datamodule().")
        self.DATAMODULE_CLASS_TYPE = get_datamodule_class(self.exp_config)
        self.datamodule = build_datamodule(self.exp_config.data)

    def _setup_model(self) -> None:
        if self.exp_config is None:
            raise RuntimeError("model_config must be set before calling _setup_model().")
        core_model_class = get_model_class(self.exp_config)
        core_model = core_model_class(self.exp_config)
        self.model = BaseLightningModule(core_model, self.exp_config.trainer)

    def get_module(self) -> BaseLightningModule:
        if self.model is None:
            self._setup_model()
        return self.model  # type: ignore[return-value]

    def get_datamodule(self) -> BaseTransportProcessDataModule:
        if self.datamodule is None:
            self._setup_datamodule()
        return self.datamodule  # type: ignore[return-value]

    def train(self) -> None:
        if self.model is None or self.datamodule is None:
            raise RuntimeError("Model and datamodule must be configured before training.")
        if self.logger is None:
            raise RuntimeError("Logger must be configured before training.")

        if self.experiment_dir is None:
            self._resolve_experiment_dir()

        self.model.save_hyperparameters(ignore=["config"], logger=False)
        self._log_hyperparameters()

        trainer = Trainer(
            default_root_dir=self.experiment_dir,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=self.devices,
            strategy=self.strategy,
            logger=self.logger,
            max_epochs=self.exp_config.trainer.max_epochs,  # type: ignore[union-attr]
            callbacks=self.callbacks or [],
            log_every_n_steps=1,
            gradient_clip_val=self.exp_config.trainer.gradient_clip_val,  # type: ignore[union-attr]
        )

        trainer.fit(self.model, datamodule=self.datamodule)

    def _log_hyperparameters(self) -> None:
        """Log current configuration as hyper‑parameters in Comet."""
        if hasattr(self.exp_config, "to_log_dict"):
            cfg_dict = self.exp_config.to_log_dict()
        elif is_dataclass(self.exp_config):
            cfg_dict = asdict(self.exp_config)
        else:
            cfg_dict = {}
        self.logger.experiment.log_parameters(cfg_dict)
