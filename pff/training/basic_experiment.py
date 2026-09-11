"""Callback-driven experiment runner for PK training."""

from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from types import SimpleNamespace
from typing import List, Optional, Tuple, Type, Union

import comet_ml
import torch
from huggingface_hub import HfApi, create_repo
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from pff import (
    COMET_KEY,
    HUGGINGFACE_KEY,
    config_dir,  # project root injected into PYTHONPATH
    results_dir,
)
from pff.config_classes.flow_pk_config import FlowPKExperimentConfig, HFFlowPKConfig
from pff.data.datasets.aicme_datasets import AICMECompartmentsDataModule
from pff.models import get_model_class, get_model_config
from pff.training.utils import (
    EXPERIMENT_CONFIG_FILENAME,
    get_lightning_checkpoint_path,
    load_experiment_config_yaml,
    load_model_from_checkpoint_path,
    parse_comet_parameters_summary,
    resolve_model_config_kwarg,
    save_experiment_config_yaml,
)


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

    resolved_strategy = (
        strategy
        if strategy is not None
        else ("ddp_find_unused_parameters_true" if ddp_flag else "auto")
    )
    return devices, resolved_strategy


def _normalize_optional_token(value: Optional[str]) -> Optional[str]:
    """Normalize optional token values from configs or env files."""
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if not value or value.lower() in ("none", "null"):
        return None
    return value


def _resolve_comet_key(
    exp_config: Optional[FlowPKExperimentConfig],
) -> Optional[str]:
    """Prefer config-provided Comet keys, with COMET_KEYS.txt as fallback."""
    cfg_key = _normalize_optional_token(getattr(exp_config, "comet_ai_key", None))
    return cfg_key or _normalize_optional_token(COMET_KEY)


def _resolve_hf_token(
    exp_config: Optional[FlowPKExperimentConfig],
) -> Optional[str]:
    """Prefer config-provided HF tokens, with KEYS.txt as fallback."""
    cfg_token = _normalize_optional_token(getattr(exp_config, "hugging_face_token", None))
    return cfg_token or _normalize_optional_token(HUGGINGFACE_KEY)


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
    exp_config: Optional[FlowPKExperimentConfig],
    devices: Optional[Union[int, List[int]]],
) -> Optional[Union[int, List[int]]]:
    if devices is not None:
        return devices

    env_devices = _parse_devices_value(os.getenv("PFF_DEVICES"))
    if env_devices is not None:
        return env_devices

    train_cfg = getattr(exp_config, "train", None) if exp_config is not None else None
    if train_cfg is not None:
        cfg_devices = getattr(train_cfg, "devices", None)
        if isinstance(cfg_devices, str):
            cfg_devices = _parse_devices_value(cfg_devices)
        if cfg_devices is not None:
            return cfg_devices

    return None


def _resolve_strategy_arg(strategy: Optional[str]) -> Optional[str]:
    if strategy is not None:
        return strategy
    env_strategy = os.getenv("PFF_STRATEGY")
    if env_strategy is None:
        return None
    env_strategy = env_strategy.strip()
    if not env_strategy or env_strategy.lower() in ("none", "null"):
        return None
    return env_strategy


def get_datamodule_class(config):
    """Return the FlowPK datamodule class for a validated experiment config."""
    if isinstance(config, FlowPKExperimentConfig):
        return AICMECompartmentsDataModule
    raise TypeError("Experiment config must be a FlowPKExperimentConfig instance.")


class BasicLightningExperiment:
    """High-level wrapper orchestrating Lightning training runs."""

    experiment_name: str = ""

    def __init__(
        self,
        *,
        exp_config: Optional[FlowPKExperimentConfig] = None,
        map_location: str = "cuda",
        devices: Optional[Union[int, List[int]]] = None,
        results_root: str | None = None,
        strategy: str | None = None,
        strict: bool = True,
        checkpoint_type: str = "best",
    ) -> None:
        self.exp_config = exp_config
        self.map_location = map_location
        resolved_devices = _resolve_devices_arg(exp_config, devices)
        resolved_strategy = _resolve_strategy_arg(strategy)
        self.devices, self.strategy = _select_devices_and_strategy(
            resolved_devices, resolved_strategy
        )
        self.strict = strict
        self.checkpoint_type = checkpoint_type
        self._results_root = results_root
        self.hf_token = _resolve_hf_token(exp_config)
        self.upload_to_hf_hub = False

        self.MODEL_CLASS_TYPE: Optional[Type] = (
            get_model_class(exp_config) if exp_config is not None else None
        )
        self.DATAMODULE_CLASS_TYPE: Optional[Type] = (
            get_datamodule_class(exp_config) if exp_config is not None else None
        )
        self.model: Optional[torch.nn.Module] = None
        self.datamodule: Optional[AICMECompartmentsDataModule] = None
        self.logger: Optional[CometLogger] = None
        self.experiment_dir: Optional[str] = None
        self.results_dir: Optional[str] = None
        self.callbacks: list[Callback] = []
        self.logger_folder: Optional[str] = None
        self.checkpoint_metric: Optional[str] = None
        self.checkpoint_mode: Optional[str] = None

    @classmethod
    def from_config(
        cls,
        exp_config: FlowPKExperimentConfig,
        map_location: str = "cuda",
        devices: Optional[Union[int, List[int]]] = None,
        results_root: str | None = None,
        strategy: str | None = None,
        strict: bool = True,
    ) -> "BasicLightningExperiment":
        self = cls(
            exp_config=exp_config,
            map_location=map_location,
            devices=devices,
            results_root=results_root,
            strategy=strategy,
            strict=strict,
        )

        self.exp_config = exp_config
        self.MODEL_CLASS_TYPE = get_model_class(exp_config)
        self.DATAMODULE_CLASS_TYPE = get_datamodule_class(exp_config)
        self.experiment_name = self.exp_config.experiment_name

        self._setup_logger()
        self._setup_datamodule()
        self._setup_model()
        self._setup_callbacks()
        self.upload_to_hf_hub = exp_config.upload_to_hf_hub
        return self

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        map_location: str = "cuda",
        devices: Optional[Union[int, List[int]]] = None,
        results_root: str | None = None,
        strategy: str | None = None,
        strict: bool = True,
    ) -> "BasicLightningExperiment":
        """Load an experiment config from YAML and instantiate the experiment."""
        exp_config = get_model_config(yaml_path)
        return cls.from_config(
            exp_config=exp_config,
            map_location=map_location,
            devices=devices,
            results_root=results_root,
            strategy=strategy,
            strict=strict,
        )

    @classmethod
    def from_experiment_comet(
        cls,
        experiment_key: str,
        model_config_override: Optional[FlowPKExperimentConfig] = None,
        map_location: str = "cuda",
        checkpoint_type: str = "best",
        results_root: str | None = None,
        strict: bool = True,
    ) -> "BasicLightningExperiment":
        """
        Sets up the experiment for resumption from an existing experiment key.

        Args:
            experiment_key: Key of the experiment to resume.
            checkpoint_type: Type of checkpoint to load ("best" or "last").
            model_config_override: Optional config values to override after loading.

        Behavior:
            - Retrieves experiment details from Comet.
            - Loads the model checkpoint and configuration.
            - Initializes the data module and sets the experiment to resume mode.
        """
        self = cls(
            exp_config=None,
            map_location=map_location,
            results_root=results_root,
            strict=strict,
        )

        self.experiment_key_0 = experiment_key
        api = comet_ml.API(api_key=_resolve_comet_key(model_config_override or self.exp_config))
        self.api_experiment = api.get_experiment_by_key(experiment_key)
        resolved_results_root = results_root or str(results_dir)
        self.experiment_dir, self.model_class_name_str = self._get_experiment_meta(
            self.api_experiment,
            experiment_key,
            resolved_results_root,
        )
        self.checkpoint_path = get_lightning_checkpoint_path(self.experiment_dir, checkpoint_type)
        self.MODEL_CLASS_TYPE = get_model_class(None, self.model_class_name_str)

        # fallback: reconstruct config from Comet
        parameters_list = self.api_experiment.get_parameters_summary()
        self.exp_config = parse_comet_parameters_summary(parameters_list)
        self.hf_token = _resolve_hf_token(self.exp_config)
        if model_config_override is not None:
            self._update_config(model_config_override)

        self.model = self._load_model_from_checkpoint(self.checkpoint_path)
        self.DATAMODULE_CLASS_TYPE = get_datamodule_class(self.exp_config)
        self.datamodule = self.DATAMODULE_CLASS_TYPE(self.exp_config)

        self.experiment_name = self.exp_config.experiment_name
        self.api_experiment.end()  # the api was only need in order to obtain the experiment name and experiment dir
        self._resume_posible = True
        self._setup_logger(experiment_key)
        self._setup_callbacks(self.experiment_dir)
        self.model._trainer = SimpleNamespace(
            logger=self.logger, current_epoch=0, is_global_zero=True
        )
        self.upload_to_hf_hub = self.exp_config.upload_to_hf_hub
        return self

    @classmethod
    def from_experiment_dir(
        cls,
        experiment_dir: str,
        model_config_override: Optional[FlowPKExperimentConfig] = None,
        map_location: str = "cuda",
        checkpoint_type: str = "best",
        results_root: str | None = None,
        strict: bool = True,
        experiment_key: Optional[str] = None,
        config_filename: str = EXPERIMENT_CONFIG_FILENAME,
    ) -> "BasicLightningExperiment":
        """
        Resume an experiment from a local directory and saved YAML config.

        Unlike :meth:`from_experiment_comet`, this method loads the experiment
        configuration from ``<experiment_dir>/<config_filename>`` instead of
        querying the Comet API. Use ``experiment_key`` if you want the logger
        to attach to an existing Comet run.
        """
        self = cls(
            exp_config=None,
            map_location=map_location,
            results_root=results_root,
            strict=strict,
        )

        self.experiment_dir = os.path.abspath(experiment_dir)
        self.checkpoint_path = get_lightning_checkpoint_path(self.experiment_dir, checkpoint_type)
        if self.checkpoint_path is None:
            raise FileNotFoundError(
                f"No checkpoint found for '{checkpoint_type}' in {self.experiment_dir}."
            )

        self.exp_config = load_experiment_config_yaml(self.experiment_dir, filename=config_filename)
        self.hf_token = _resolve_hf_token(self.exp_config)
        if model_config_override is not None:
            self._update_config(model_config_override)

        self.MODEL_CLASS_TYPE = get_model_class(self.exp_config)
        self.model = self._load_model_from_checkpoint(self.checkpoint_path)
        self.DATAMODULE_CLASS_TYPE = get_datamodule_class(self.exp_config)
        self.datamodule = self.DATAMODULE_CLASS_TYPE(self.exp_config)

        self.experiment_name = self.exp_config.experiment_name
        self._resume_posible = True
        self._setup_logger(experiment_key)
        self._setup_callbacks(self.experiment_dir)
        self.model._trainer = SimpleNamespace(
            logger=self.logger, current_epoch=0, is_global_zero=True
        )
        self.upload_to_hf_hub = self.exp_config.upload_to_hf_hub
        return self

    def _get_experiment_meta(
        self,
        api_experiment,
        experiment_key: str,
        results_root: str,
    ) -> tuple[str | None, str | None]:
        """
        Return ``(experiment_dir, name_str)`` for a Comet run.

        Priority for *experiment_dir*
        1. value stored under ``model_config/experiment_dir`` (new runs)
        2. value stored under ``config/experiment_dir`` (legacy runs)
        3. reconstructed path ``<results_root>/comet/node_pk_compartments/<key>``

        Priority for *name_str*
        1. value stored under ``model_config/name_str``
        2. Comet run display name (fallback)
        """
        exp_dir, name_str = None, None
        for prefix in ("", "model_config/", "config/"):
            try:
                param = api_experiment.get_parameters_summary(prefix + "experiment_dir")
                if isinstance(param, dict) and param.get("valueCurrent"):
                    exp_dir = param["valueCurrent"]
            except Exception:
                pass
            try:
                param = api_experiment.get_parameters_summary(prefix + "name_str")
                if isinstance(param, dict) and param.get("valueCurrent"):
                    name_str = param["valueCurrent"]
            except Exception:
                pass

        if not name_str:
            try:
                name_str = api_experiment.get_name()
            except Exception:
                name_str = None

        if not exp_dir or exp_dir == "null":
            exp_dir = os.path.join(
                results_root,
                "comet",
                "node_pk_compartments",
                experiment_key,
            )

        return exp_dir, name_str

    def _update_config(self, user_model_config: FlowPKExperimentConfig) -> None:
        """
        TODO: THIS UPDATE CONFIG IS NATIVE TO ACIMET AND MUST BE DEFINED MODELWISE
        """
        if self.exp_config is None:
            raise RuntimeError("exp_config must be set before applying overrides.")

        self.exp_config.meta_study = user_model_config.meta_study
        self.exp_config.mix_data = user_model_config.mix_data
        self.exp_config.train = user_model_config.train
        self.exp_config.mix_data.recreate_tempfile = True
        self.exp_config.debug_test = user_model_config.debug_test

        # Allow overriding model-specific sections when resuming.
        if hasattr(self.exp_config, "network") and hasattr(user_model_config, "network"):
            self.exp_config.network = user_model_config.network
        if hasattr(self.exp_config, "vector_field") and hasattr(user_model_config, "vector_field"):
            self.exp_config.vector_field = user_model_config.vector_field
        self.hf_token = _resolve_hf_token(self.exp_config)

        if self.model is not None:
            self.model.model_config = self.exp_config

    @classmethod
    def from_hf(
        cls,
        hf_model_id: str,
        *,
        map_location: str = "cuda",
        devices: Optional[Union[int, List[int]]] = None,
        results_root: str | None = None,
        strategy: str | None = None,
        strict: bool = True,
    ) -> "BasicLightningExperiment":
        raise NotImplementedError

    def _resolve_results_root(self) -> str:
        if self.results_dir is not None:
            return self.results_dir
        if self._results_root is not None:
            return self._results_root
        if self.exp_config is not None and getattr(self.exp_config, "my_results_path", None):
            return self.exp_config.my_results_path
        return str(results_dir)

    def _resolve_experiment_dir(self) -> str:
        if self.experiment_dir is not None:
            return self.experiment_dir
        if self.logger is None:
            raise RuntimeError("Logger must be initialised before resolving experiment dir.")

        key = getattr(self.logger, "version", None) or "unknown"
        self.experiment_dir = os.path.join(
            self.logger_folder,
            self.exp_config.experiment_name,
            str(key),
        )
        if self.exp_config is not None:
            self.exp_config.experiment_dir = self.experiment_dir
        return self.experiment_dir

    def _resolve_checkpoint_metric(self) -> tuple[str, str]:
        if self.exp_config is None:
            raise RuntimeError("Experiment config must be set before resolving checkpoint metric.")
        metric = getattr(self.exp_config, "checkpoint_metric", None)
        mode = getattr(self.exp_config, "checkpoint_mode", None)
        return metric or "val_rmse", mode or "min"

    def _setup_logger(self, experiment_key: Optional[str] = None) -> None:
        """Initialise a Comet logger for the experiment.

        Uses ``exp_config.comet_ai_key`` when provided, otherwise falls back to
        the COMET_KEYS.txt value loaded at import time.
        """
        if self.exp_config is None:
            raise RuntimeError("exp_config must be set before calling _setup_logger().")

        my_results_path = self._resolve_results_root()
        self.logger_folder = os.path.join(my_results_path, "comet")

        self.logger = CometLogger(
            api_key=_resolve_comet_key(self.exp_config) or None,
            project_name=self.exp_config.experiment_name,
            experiment_key=experiment_key,
        )
        self._resolve_experiment_dir()

    def _setup_callbacks(self, experiment_dir: Optional[str] = None) -> None:
        if self.exp_config is None:
            raise RuntimeError("exp_config must be set before calling _setup_callbacks().")
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
        build_cb = getattr(self.model, "build_visualization_callback", None)
        if callable(build_cb):
            visualization_cb = build_cb()
            if visualization_cb is not None:
                if isinstance(visualization_cb, (list, tuple)):
                    self.callbacks.extend([cb for cb in visualization_cb if cb is not None])
                else:
                    self.callbacks.append(visualization_cb)

        for callback in self.callbacks:
            attach = getattr(callback, "attach_experiment_checkpoints", None)
            if not callable(attach):
                continue
            attach(
                checkpoint_callback_last=self.checkpoint_callback_last,
                checkpoint_callback_best=self.checkpoint_callback_best,
            )

    def _setup_datamodule(self) -> None:
        if self.exp_config is None:
            raise RuntimeError("exp_config must be set before calling _setup_datamodule().")
        self.DATAMODULE_CLASS_TYPE = get_datamodule_class(self.exp_config)
        self.datamodule = self.DATAMODULE_CLASS_TYPE(self.exp_config)

    def _setup_model(self) -> None:
        if self.exp_config is None:
            raise RuntimeError("exp_config must be set before calling _setup_model().")
        core_model_class = get_model_class(self.exp_config)
        self.model = core_model_class(self.exp_config)

    def _resolve_model_config_kwarg(self) -> str:
        """Return the constructor kwarg name for passing the experiment config."""
        if self.MODEL_CLASS_TYPE is None:
            raise RuntimeError("MODEL_CLASS_TYPE must be set before resolving config kwargs.")
        return resolve_model_config_kwarg(self.MODEL_CLASS_TYPE)

    def _load_model_from_checkpoint(self, checkpoint_path: str) -> torch.nn.Module:
        """Load a model from either a Lightning or scheduler-managed checkpoint."""
        if self.MODEL_CLASS_TYPE is None:
            raise RuntimeError("MODEL_CLASS_TYPE must be set before loading a checkpoint.")
        if self.exp_config is None:
            raise RuntimeError("exp_config must be set before loading a checkpoint.")
        return load_model_from_checkpoint_path(
            self.MODEL_CLASS_TYPE,
            checkpoint_path,
            model_config=self.exp_config,
            map_location=self.map_location,
            strict=self.strict,
        )

    def get_module(self) -> torch.nn.Module:
        if self.model is None:
            self._setup_model()
        return self.model

    def get_datamodule(self) -> AICMECompartmentsDataModule:
        if self.datamodule is None:
            self._setup_datamodule()
        return self.datamodule

    def _log_hyperparameters(self) -> None:
        """Log the current model configuration to the Comet logger."""
        if self.logger is None:
            raise RuntimeError("Logger must be configured before logging hyperparameters.")
        if is_dataclass(self.exp_config):
            cfg_dict = asdict(self.exp_config)
        else:
            cfg_dict = {}
        self.logger.experiment.log_parameters(cfg_dict)
        self._save_experiment_config_yaml()

    def _save_experiment_config_yaml(self) -> None:
        """Write the current experiment config to the experiment directory."""
        if self.exp_config is None:
            raise RuntimeError("Experiment config must be set before saving YAML.")
        if self.experiment_dir is None:
            self._resolve_experiment_dir()
        save_experiment_config_yaml(self.exp_config, self.experiment_dir)

    def train(self) -> None:
        """Train the model with the configured Trainer and callbacks."""
        if self.model is None or self.datamodule is None:
            raise RuntimeError("Model and datamodule must be configured before training.")
        if self.logger is None:
            raise RuntimeError("Logger must be configured before training.")

        if self.experiment_dir is None:
            self._resolve_experiment_dir()

        # requiered for checkpoint lightning
        self.model.save_hyperparameters(ignore=["config"], logger=False)
        # requiered for commet logger
        self._log_hyperparameters()

        trainer = Trainer(
            default_root_dir=self.experiment_dir,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=self.devices,
            strategy=self.strategy,
            logger=self.logger,
            max_epochs=self.exp_config.train.epochs,
            callbacks=self.callbacks or [],
            log_every_n_steps=self.exp_config.train.log_interval,
            gradient_clip_val=self.exp_config.train.gradient_clip_val,
        )

        trainer.fit(self.model, datamodule=self.datamodule)

        # send model to huggingface
        if self.hf_token is not None and self.upload_to_hf_hub:
            self._push_best_model_to_hub()

    @rank_zero_only
    def _push_best_model_to_hub(self, force_push: bool = False) -> None:
        """
        Wrapper: Loads the checkpoint, compares local RMSE vs remote,
        and calls `_push_model_to_hub` if conditions are satisfied.
        """
        ckpt_path = get_lightning_checkpoint_path(self.experiment_dir, self.checkpoint_type)
        if not (ckpt_path and os.path.exists(ckpt_path)):
            self.logger.experiment.log_other("hf_push_status", "checkpoint_missing")
            return

        # Load model
        model = self._load_model_from_checkpoint(ckpt_path)

        # Local validation RMSE
        if self.checkpoint_type == "best":
            local_rmse = float(self.checkpoint_callback_best.best_model_score)
        else:
            # if you’re resuming from "last", fall back to whatever the config currently stores
            if hasattr(model.config, "get_best"):
                local_rmse = float(model.config.get_best("val_rmse"))
            else:
                local_rmse = float(getattr(model.config, "best_val_loss", float("inf")))

        # Repo ID
        user = HfApi().whoami(token=self.hf_token)["name"]
        hf_repo_id = f"{user}/{self.exp_config.hf_model_name}"

        # Remote best
        remote_best = self._get_remote_best_val_loss(hf_repo_id)

        # Push if better or forced
        if (local_rmse < remote_best) or force_push:
            # ---- IMPORTANT: mutate config BEFORE pushing so it is serialized by _push_model_to_hub ----
            model.config.set_best("val_rmse", local_rmse)

            self._push_model_to_hub(
                model=model,
                hf_repo_id=hf_repo_id,
                commit_message=f"{self.checkpoint_type} val_rmse {local_rmse:.4f}",
                alias_name="best_model_hf",
            )

            self.logger.experiment.log_metric("hf_pushed", 1)
            self.logger.experiment.log_other("hf_push_repo", hf_repo_id)
            self.logger.experiment.log_other("hf_push_local_rmse", str(local_rmse))
            self.logger.experiment.log_other("hf_push_remote_best_before", str(remote_best))
        else:
            self.logger.experiment.log_metric("hf_pushed", 0)
            self.logger.experiment.log_other("hf_push_repo", "not_pushed")
            self.logger.experiment.log_other("hf_push_local_rmse", str(local_rmse))
            self.logger.experiment.log_other("hf_push_remote_best", str(remote_best))

    def _push_model_to_hub(
        self, model, hf_repo_id: str, commit_message: str, alias_name: str | None = None
    ) -> None:
        """
        Primitive function: Push an *already loaded* model to the Hugging Face Hub.
        """
        create_repo(hf_repo_id, exist_ok=True, token=self.hf_token)

        save_dir = os.path.join(self.experiment_dir, alias_name or "model_hf")
        os.makedirs(save_dir, exist_ok=True)

        # Save binary weights + config
        torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        model.config.save_pretrained(save_dir)

        # Upload the folder
        api = HfApi(token=self.hf_token)
        api.upload_folder(
            folder_path=save_dir,
            repo_id=hf_repo_id,
            commit_message=commit_message,
            token=self.hf_token,
        )

        # Upload model card if present
        hf_model_card_path = os.path.join(config_dir, *self.exp_config.hf_model_card_path)
        if not os.path.isfile(hf_model_card_path):
            raise FileNotFoundError(f"Model card not found at: {hf_model_card_path}")

        api.upload_file(
            path_or_fileobj=hf_model_card_path,
            path_in_repo="README.md",
            repo_id=hf_repo_id,
            repo_type="model",
            token=self.hf_token,
        )

    def _get_remote_best_val_loss(self, hf_repo_id: str) -> float:
        """
        Read the best validation metric from the remote HF config.
        Returns +inf if not found.
        """
        try:
            remote_cfg = HFFlowPKConfig.from_pretrained(
                hf_repo_id,
                token=self.hf_token,
                force_download=True,  # IMPORTANT: avoid reading stale cached config.json
            )

            # Prefer new API
            if hasattr(remote_cfg, "get_best"):
                return float(remote_cfg.get_best("val_rmse", default=float("inf")))

            # Backward-compat fallback
            return float(getattr(remote_cfg, "best_val_loss", float("inf")))

        except Exception as e:
            self.logger.experiment.log_other("hf_remote_check_error", str(e))
            return float("inf")


__all__ = ["BasicLightningExperiment"]
