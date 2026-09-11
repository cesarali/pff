import os
from dataclasses import asdict, is_dataclass
from types import SimpleNamespace
from typing import List, Optional, Union

import comet_ml
import torch
from huggingface_hub import HfApi, create_repo, login
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from pff import (
    config_dir,  # project root injected into PYTHONPATH
    project_dir,
)
from pff.config_classes.node_pk_config import HFNodePKConfig, NodePKExperimentConfig
from pff.data.datasets.aicme_datasets import AICMECompartmentsDataModule
from pff.models import get_model_class
from pff.training.utils import (
    NonFiniteLossCallback,
    dataclass_from_dict,
    get_lightning_checkpoint_path,
    parse_comet_parameters_summary,
)

HF_TOKEN = open(os.path.join(project_dir, "KEYS.txt")).read().strip()
COMET_KEY = open(os.path.join(project_dir, "COMET_KEYS.txt")).read().strip()


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


def _resolve_comet_key(config: Optional[NodePKExperimentConfig]) -> Optional[str]:
    """Prefer config-provided Comet keys, with COMET_KEYS.txt as fallback."""
    cfg_key = _normalize_optional_token(getattr(config, "comet_ai_key", None))
    return cfg_key or _normalize_optional_token(COMET_KEY)


def _resolve_hf_token(config: Optional[NodePKExperimentConfig]) -> Optional[str]:
    """Prefer config-provided HF tokens, with KEYS.txt as fallback."""
    cfg_token = _normalize_optional_token(getattr(config, "hugging_face_token", None))
    return cfg_token or _normalize_optional_token(HF_TOKEN)


def select_strategy(devices, strategy):
    # Devices
    devices = (
        devices
        if devices is not None
        else (torch.cuda.device_count() if torch.cuda.is_available() else 1)
    )
    if isinstance(devices, (list, tuple)):
        ddp_flag = len(devices) > 1
    else:
        ddp_flag = devices and devices > 1
    # NEW
    if strategy is not None:
        return devices, strategy
    else:
        strategy = "ddp" if ddp_flag else "auto"  # ← single GPU / CPU
        return devices, strategy


class BasicLightningExperiment:
    """
    Class defining all objects needed for training, checkpointing,
    and evaluation, as well as calls to the PyTorch Lightning trainer.
    """

    experiment_name: str = ""

    def __init__(
        self,
        model_config: Union[NodePKExperimentConfig] = None,
        hf_model_id: Optional[str] = None,
        experiment_key: Optional[str] = None,
        map_location: str = "cuda",
        checkpoint_type: str = "best",
        comet_key: Optional[str] = None,
        train_new: bool = True,
        devices: Optional[Union[int, List[int]]] = None,
        results_dir: str = None,
        strategy: str = None,
        strict=True,
    ):
        """
        Initializes the experiment.

        Args:
            model_config (NeuralProcessConfig | AmortizedInContextMixEffectsConfig):
                Configuration object for the experiment. If `experiment_key` is provided,
                this will be ignored and the configuration will be loaded from the experiment.
            map_location (str): Device location for loading the model (e.g., "cuda" or "cpu").
            checkpoint_type (str): Type of checkpoint to load ("best" or "last").
            experiment_key (str): Key of the experiment to resume from. If provided, the experiment
                will be set up from the existing experiment.
            comet_key (str): API key for Comet logger.
            train_new (bool): If True, a new experiment will be initialized using the provided
                configuration. If False, the experiment is ready for resumption.

        Behavior:
            - If `experiment_key` is provided, the experiment is set up from the existing experiment
              using the provided `comet_key` and `checkpoint_type`.
            - If `train_new` is True, a new experiment is initialized using the provided `config` or from the config obtained
            - If neither `experiment_key` nor `train_new` is set, the experiment is ready for resumption.
        """
        # tokens & login
        self.hf_token = _resolve_hf_token(model_config)
        if self.hf_token:
            rank_zero_only(login)(token=self.hf_token)

        # shared attributes
        self.map_location = map_location
        self.checkpoint_type = checkpoint_type
        self.datamodule = None
        self.model = None
        self.logger = None
        self.callbacks = None
        self._set_to_train = False
        self._resume_posible = False
        self.strict = strict
        self.devices, self.strategy = select_strategy(devices, strategy)

        # HF login
        if model_config is not None:
            self.MODEL_CLASS_TYPE = get_model_class(model_config)

        # Dispatch based on inputs
        if hf_model_id is not None:
            # 1) HF download path
            self._setup_from_hf(hf_model_id)
        elif experiment_key is not None:
            # 2.1) old logger old weights set up for resume
            model_config = self._setup_from_experiment_key(
                _resolve_comet_key(model_config),
                experiment_key,
                checkpoint_type,
                results_dir,
                model_config,
            )
            if model_config is not None and train_new:
                # 2.2) new comet experiment new weights (just reuses config)
                self._setup_from_config(model_config)

        elif model_config is not None and train_new:
            # 3) new everything
            self._setup_from_config(model_config)
        else:
            raise ValueError(
                "Provide one of: hf_model_id, experiment_key, or model_config+train_new"
            )

        self.force_hf_push = self.model_config.upload_to_hf_hub

    def _setup_from_config(self, config: Union[NodePKExperimentConfig]):
        """
        Sets up the experiment using the provided configuration.

        Args:
            config: Configuration object for the experiment.
        """
        self.model_config = config
        self.hf_token = _resolve_hf_token(self.model_config)
        if self.hf_token:
            rank_zero_only(login)(token=self.hf_token)
        self.experiment_name = config.experiment_name
        self._setup_logger()
        self._setup_callbacks()
        self._setup_datamodule()
        self._setup_model()
        self._set_to_train = True
        self._resume_posible = False

    @rank_zero_only
    def _setup_from_hf(self, hf_model_id: str):
        """Download HF weights + configs, instantiate model only."""
        # load HF config then reconstruct NodePKConfig
        hf_cfg = HFNodePKConfig.from_pretrained(hf_model_id, token=self.hf_token)
        cfg_dict = hf_cfg.to_dict()
        # build nested NodePKConfig automatically
        self.model_config = dataclass_from_dict(NodePKExperimentConfig, cfg_dict)

        # download weights
        bin_path = HfApi().hf_hub_download(
            repo_id=hf_model_id, filename="pytorch_model.bin", token=self.hf_token
        )
        state_dict = torch.load(bin_path, map_location=self.map_location)

        # instantiate model
        self.MODEL_CLASS_TYPE = get_model_class(self.model_config)
        self.model = self.MODEL_CLASS_TYPE(self.model_config)
        self.model.load_state_dict(state_dict)
        self.model.to(self.map_location)
        self.datamodule = AICMECompartmentsDataModule(self.model_config)
        self._resume_posible = False
        self._set_to_train = False

    def _setup_logger(self, experiment_key=None):
        """Initialise the ``CometLogger`` in distributed setups.

        ``COMET_EXPERIMENT_KEY`` is checked when ``experiment_key`` is not
        provided so that non-zero ranks attach to the run created by rank zero.
        Rank zero will create a new run if no key exists and then write the
        resulting key back to ``COMET_EXPERIMENT_KEY`` for the other ranks."""

        if self.model_config.my_results_path is None:
            from pff import results_dir

            my_results_path = results_dir
        else:
            my_results_path = self.model_config.my_results_path
        self.logger_folder = os.path.join(my_results_path, "comet")

        # resolve experiment key
        if experiment_key is None:
            experiment_key = os.environ.get("COMET_EXPERIMENT_KEY")

        rank_zero = os.environ.get("RANK", "0") == "0"

        if experiment_key:
            # attach to an existing experiment
            self.logger = CometLogger(
                api_key=_resolve_comet_key(self.model_config),
                project_name=self.model_config.experiment_name,
                experiment_key=experiment_key,
            )
        else:
            # create a new experiment only on rank zero
            self.logger = CometLogger(
                api_key=_resolve_comet_key(self.model_config),
                project_name=self.model_config.experiment_name,
            )
            if rank_zero:
                os.environ["COMET_EXPERIMENT_KEY"] = self.logger.version

        if rank_zero:
            # 🩹 Ensure tags is a list (added once)
            tags = self.model_config.tags
            if isinstance(tags, str):
                import ast

                try:
                    tags = ast.literal_eval(tags)
                except Exception:
                    tags = [tags]
            self.logger.experiment.add_tags(tags)

    def _setup_callbacks(self, experiment_dir=None):
        """
        Sets up model checkpoint callbacks for saving the best and last model states.

        Args:
            experiment_dir: Optional directory to save checkpoints. If None, it is derived from the logger.
        """
        if experiment_dir is None:
            # use logger version to avoid None when experiment key isn't yet resolved
            key = self.logger.version
            if hasattr(self.logger, "experiment") and self.logger.experiment:
                key = self.logger.experiment.get_key() or key
            self.experiment_dir = os.path.join(
                self.logger_folder,
                self.model_config.experiment_name,
                key,
            )
        else:
            self.experiment_dir = experiment_dir
        rank_zero_only(os.makedirs)(self.experiment_dir, exist_ok=True)
        self.model_config.experiment_dir = self.experiment_dir

        # Monitor validation RMSE
        self.checkpoint_callback_best = ModelCheckpoint(
            dirpath=self.experiment_dir,
            save_top_k=1,
            monitor="val_rmse",
            mode="min",
            filename="best-{epoch:02d}-{val_rmse:.4f}",
        )
        # self.checkpoint_callback_best_log_rmse = ModelCheckpoint(
        #    dirpath=self.experiment_dir,
        #    save_top_k=1,
        #    monitor="avg_log_rmse",
        #    mode="min",
        #    filename="best_log_rmse",
        # )
        self.checkpoint_callback_last = ModelCheckpoint(
            dirpath=self.experiment_dir,
            save_last=True,
            monitor=None,
            filename="last",
            save_top_k=0,
        )
        self.checkpoint_callback_periodic = ModelCheckpoint(
            dirpath=self.experiment_dir,
            every_n_epochs=10,
            monitor=None,
            filename="periodic-{epoch:04d}",
            save_top_k=-1,
        )

        self.nan_callback = NonFiniteLossCallback()
        self.non_fininte_callback = NonFiniteLossCallback()

        # Register callbacks
        self.callbacks = [
            self.checkpoint_callback_last,
            self.checkpoint_callback_best,
            #            self.checkpoint_callback_best_log_rmse,
            self.checkpoint_callback_periodic,
            self.nan_callback,
            self.non_fininte_callback,
        ]

    def _setup_datamodule(self):
        """
        Sets up the data module for the experiment.

        Behavior:
            - Initializes the data module using the model configuration.
            - Currently uses `AICMECompartmentsDataModule`.
        """
        self.datamodule = AICMECompartmentsDataModule(self.model_config)

    def _setup_model(self):
        """
        Sets up the model for the experiment.

        Behavior:
            - Initializes the model using the model configuration and class type.
        """
        self.model = self.MODEL_CLASS_TYPE(self.model_config)

    def _setup_from_experiment_key(
        self,
        comet_key=COMET_KEY,
        experiment_key=None,
        checkpoint_type="best",
        results_dir=None,
        new_model_config=None,
    ):
        """
        Sets up the experiment for resumption from an existing experiment key.

        Args:
            comet_key: API key for accessing Comet experiments.
            experiment_key: Key of the experiment to resume.
            checkpoint_type: Type of checkpoint to load ("best" or "last").

        Behavior:
            - Retrieves experiment details from Comet.
            - Loads the model checkpoint and configuration.
            - Initializes the data module and sets the experiment to resume mode.
        """
        self.experiment_key_0 = experiment_key
        api = comet_ml.API(api_key=comet_key)
        self.api_experiment = api.get_experiment_by_key(experiment_key)
        self.experiment_dir, self.model_class_name_str = self._get_experiment_meta(
            self.api_experiment, experiment_key, results_dir
        )
        self.checkpoint_path = get_lightning_checkpoint_path(self.experiment_dir, checkpoint_type)
        self.MODEL_CLASS_TYPE = get_model_class(None, self.model_class_name_str)

        # try loading checkpoint directly (may contain hparams)
        try:
            self.model = self.MODEL_CLASS_TYPE.load_from_checkpoint(
                checkpoint_path=self.checkpoint_path,
                map_location=self.map_location,
            )
            self.model_config = self.model.model_config
            self.hf_token = _resolve_hf_token(self.model_config)
            if self.hf_token:
                rank_zero_only(login)(token=self.hf_token)
            # Apply user overrides even if checkpoint loading succeeds
            if new_model_config is not None:
                self._update_config(new_model_config)
        except Exception:
            # fallback: reconstruct config from Comet
            parameters_list = self.api_experiment.get_parameters_summary()
            self.model_config = parse_comet_parameters_summary(parameters_list)
            self.hf_token = _resolve_hf_token(self.model_config)
            if self.hf_token:
                rank_zero_only(login)(token=self.hf_token)
            if new_model_config is not None:
                self._update_config(new_model_config)

            self.model = self.MODEL_CLASS_TYPE.load_from_checkpoint(
                checkpoint_path=self.checkpoint_path,
                map_location=self.map_location,
                model_config=self.model_config,
                strict=self.strict,
            )
            self.datamodule = AICMECompartmentsDataModule(self.model_config)

        self.experiment_name = self.model_config.experiment_name
        self.api_experiment.end()  # the api was only need in order to obtain the experiment name and experiment dir
        self._resume_posible = True
        self._setup_logger(experiment_key)
        self.model._trainer = SimpleNamespace(
            logger=self.logger, current_epoch=0, is_global_zero=True
        )
        return self.model_config

    def _get_remote_best_val_loss(self, hf_repo_id: str) -> float:
        """
        Query the Hugging Face Hub for the current best validation loss
        stored in the remote config.

        Parameters
        ----------
        hf_repo_id : str
            Full repository ID, e.g. "user/model-name".

        Returns
        -------
        float
            Remote best_val_loss if found, else +inf.
        """
        try:
            remote_cfg = HFNodePKConfig.from_pretrained(
                hf_repo_id,
                token=self.hf_token,
            )
            return float(getattr(remote_cfg, "best_val_loss", float("inf")))
        except Exception as e:
            # Optional: log the error for debugging
            self.logger.experiment.log_other("hf_remote_check_error", str(e))
            return float("inf")

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
        hf_model_card_path = os.path.join(config_dir, *self.model_config.hf_model_card_path)
        if not os.path.isfile(hf_model_card_path):
            raise FileNotFoundError(f"Model card not found at: {hf_model_card_path}")

        api.upload_file(
            path_or_fileobj=hf_model_card_path,
            path_in_repo="README.md",
            repo_id=hf_repo_id,
            repo_type="model",
            token=self.hf_token,
        )

    @rank_zero_only
    def _push_best_model_to_hub(self):
        """
        Wrapper: Loads the checkpoint, compares local RMSE vs remote,
        and calls `_push_model_to_hub` if conditions are satisfied.
        """
        ckpt_path = get_lightning_checkpoint_path(self.experiment_dir, self.checkpoint_type)
        if not (ckpt_path and os.path.exists(ckpt_path)):
            self.logger.experiment.log_other("hf_push_status", "checkpoint_missing")
            return

        # Load model
        model = self.MODEL_CLASS_TYPE.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            model_config=self.model_config,
            map_location=self.map_location,
        )

        # Local validation RMSE
        if self.checkpoint_type == "best":
            local_rmse = float(self.checkpoint_callback_best.best_model_score)
        else:
            local_rmse = model.config.best_val_loss

        # Repo ID
        user = HfApi().whoami(token=self.hf_token)["name"]
        hf_repo_id = f"{user}/{self.model_config.hf_model_name}"

        # Remote best
        remote_best = self._get_remote_best_val_loss(hf_repo_id)

        # Push if better or forced
        if local_rmse < remote_best or self.force_hf_push:
            model.config.best_val_loss = local_rmse

            self._push_model_to_hub(
                model=model,
                hf_repo_id=hf_repo_id,
                commit_message=f"{self.checkpoint_type} val_rmse {local_rmse:.4f}",
                alias_name="best_model_hf",
            )

            self.logger.experiment.log_metric("hf_pushed", 1)
            self.logger.experiment.log_metric("hf_push_repo", hf_repo_id)
        else:
            self.logger.experiment.log_metric("hf_pushed", 0)
            self.logger.experiment.log_metric("hf_push_repo", "not_pushed")

    def _log_hyperparameters(self):
        """Log the current model configuration to the Comet logger."""
        if isinstance(self.logger, CometLogger):
            cfg_dict = asdict(self.model_config) if is_dataclass(self.model_config) else {}
            try:
                # ensure the lightning module stores the latest parameters
                setattr(self.model, "hparams", cfg_dict)
            except Exception:
                pass
            self.logger.experiment.log_parameters(cfg_dict)

    def train(self):
        """
        Trains the model using PyTorch Lightning's `Trainer`.

        Behavior:
            - Logs model configuration parameters to Comet.
            - Initializes the Trainer with the specified settings.
            - Fits the model using the datamodule or dataloaders.
        """
        if self._set_to_train:
            # store updated hyper-parameters on the model and logger
            try:
                self.model.save_hyperparameters(ignore=["config"], logger=False)
            except Exception:
                pass
            self._log_hyperparameters()
            ckpt_path = None
            # this loop ensures that if the training stops due to nans losses we restart from non nan checkpoint
            attempt = 0
            while attempt < 11:
                attempt += 1
                trainer = Trainer(
                    default_root_dir=self.experiment_dir,
                    accelerator="gpu" if torch.cuda.is_available() else "cpu",
                    devices=self.devices,
                    strategy=self.strategy,
                    logger=self.logger,
                    max_epochs=self.model_config.train.epochs,
                    callbacks=self.callbacks or [],
                    log_every_n_steps=1,
                    gradient_clip_val=self.model_config.train.gradient_clip_val,
                    # reload_dataloaders_every_n_epochs=1,
                )
                trainer.fit(self.model, datamodule=self.datamodule, ckpt_path=ckpt_path)

                if getattr(self.model, "nan_detected", False):
                    ckpt_path = self.nan_callback.last_valid_checkpoint
                    if ckpt_path is None:
                        print(
                            "No valid checkpoint available to resume from. Restarting from scratch."
                        )
                        self.model = self.MODEL_CLASS_TYPE(self.model_config)
                    else:
                        self.model = self.MODEL_CLASS_TYPE.load_from_checkpoint(
                            checkpoint_path=ckpt_path,
                            model_config=self.model_config,
                            map_location=self.map_location,
                        )
                    setattr(self.model, "nan_detected", False)
                    continue
                break

            if self.hf_token:
                self._push_best_model_to_hub()
        else:
            raise Exception("Not set to train!")

    def resume(self, how_many_more_epochs: int = 1, from_new_experiment=False):
        """
        Resumes training from a previously saved checkpoint.

        Args:
            how_many_more_epochs: Number of additional epochs to train.

        Behavior:
            - Sets up the logger and callbacks for the resumed experiment.
            - Adjusts the maximum number of epochs based on the checkpoint.
            - Resumes training using the saved checkpoint.
        """
        if self._resume_posible:
            if from_new_experiment:
                self._setup_logger()
            else:
                self._setup_logger(experiment_key=self.experiment_key_0)

            self._setup_callbacks(experiment_dir=self.experiment_dir)
            # update and log hyper-parameters for resumed run
            try:
                self.model.save_hyperparameters(ignore=["config"], logger=False)
            except Exception:
                pass
            self._log_hyperparameters()

            ckpt_path = self.checkpoint_path
            while True:
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                start_epoch = checkpoint["epoch"]
                new_max_epochs = start_epoch + how_many_more_epochs

                trainer = Trainer(
                    default_root_dir=self.experiment_dir,
                    accelerator="gpu" if torch.cuda.is_available() else "cpu",
                    devices=self.devices,
                    strategy=self.strategy,
                    logger=self.logger,
                    max_epochs=new_max_epochs,
                    callbacks=self.callbacks
                    or [],  # callbacks defined once in rank zero, empty list for workers
                    log_every_n_steps=self.model_config.train.log_interval,
                    gradient_clip_val=self.model_config.train.gradient_clip_val,
                )

                trainer.fit(self.model, datamodule=self.datamodule, ckpt_path=ckpt_path)

                if getattr(self.model, "nan_detected", False):
                    ckpt_path = self.nan_callback.last_valid_checkpoint
                    if ckpt_path is None:
                        print(
                            "No valid checkpoint available to resume from. Restarting from scratch."
                        )
                        self.model = self.MODEL_CLASS_TYPE(self.model_config)
                    else:
                        self.model = self.MODEL_CLASS_TYPE.load_from_checkpoint(
                            checkpoint_path=ckpt_path,
                            model_config=self.model_config,
                            map_location=self.map_location,
                        )
                    setattr(self.model, "nan_detected", False)
                    continue
                break
        else:
            raise Exception("Resume Called without Starting from Experiment Key")

    # ------------------------------------------------------------------ #
    # helper : experiment_dir + name_str                                 #
    # ------------------------------------------------------------------ #
    def _get_experiment_meta(  # NEW
        self,
        api_experiment,
        experiment_key: str,
        results_dir: str,
    ) -> tuple[str | None, str | None]:
        """
        This function allows our legacy models to be loaded

        Return **(experiment_dir, name_str)** for a Comet run.

        Priority for *experiment_dir*
        1.   value stored under «model_config/experiment_dir»          (new runs)
        2.   value stored under «config/experiment_dir»                (very old)
        3.   reconstructed path  <results_dir>/comet/<name_str>/<key>

        Priority for *name_str*
        1.   value stored under «model_config/name_str»
        2.   Comet run display name  (fallback)
        """
        import os

        # ── a) look for the two parameters directly on the run ────────────
        exp_dir, name_str = None, None
        for prefix in ("", "model_config/", "config/"):
            try:  # experiment_dir
                p = api_experiment.get_parameters_summary(prefix + "experiment_dir")
                if isinstance(p, dict) and p.get("valueCurrent"):
                    exp_dir = p["valueCurrent"]
            except Exception:
                pass
            try:  # name_str
                p = api_experiment.get_parameters_summary(prefix + "name_str")
                if isinstance(p, dict) and p.get("valueCurrent"):
                    name_str = p["valueCurrent"]
            except Exception:
                pass

        # ── b) fallback for name_str (Comet’s run label) ──────────────────
        if not name_str:
            try:
                name_str = api_experiment.get_name()
            except Exception:
                name_str = None

        # ── c) fallback for experiment_dir (re-build default path) ────────
        if not exp_dir or exp_dir == "null":
            if results_dir is None:
                from pff import results_dir
            exp_dir = os.path.join(results_dir, "comet", "node_pk_compartments", experiment_key)

        return exp_dir, name_str

    def _update_config(self, user_model_config):
        """ """
        print("Model Config Submitted with Experiment or Model Card")
        print(" UPDATING DATAMODULE METADA STUDY CONFIG")
        # Update dataset related fields
        self.model_config.meta_study = user_model_config.meta_study
        self.model_config.mix_data = user_model_config.mix_data
        self.model_config.train = user_model_config.train
        self.model_config.mix_data.recreate_tempfile = True
        self.model_config.debug_test = user_model_config.debug_test

        # Allow overriding loss and KL regularisation flags when resuming
        self.model_config.network.loss_name = user_model_config.network.loss_name
        self.model_config.network.use_kl_s = user_model_config.network.use_kl_s
        self.model_config.network.use_kl_i = user_model_config.network.use_kl_i
        self.model_config.network.use_kl_init = user_model_config.network.use_kl_init
        self.model_config.network.use_invariance_loss = (
            user_model_config.network.use_invariance_loss
        )

        # If model already exists, propagate changes
        if hasattr(self, "model") and self.model is not None:
            self.model.model_config = self.model_config
