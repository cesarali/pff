import torch
import lightning.pytorch as L
import yaml
import os
from dataclasses import is_dataclass, asdict
import sys

from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.callbacks import ModelCheckpoint

from pff.config_classes.flow_pk_config import FlowPKExperimentConfig
from pff.data.datasets.aicme_datasets import AICMECompartmentsDataModule
from pff.models import get_model_class


from pff.utils import (PASCAL_BASE_DIR, 
                                NERSC_BASE_DIR, 
                                NERSC_EXPERIMENT_DIR, 
                                PROJECT, 
                                WORKSPACE, 
                                COMET_API_KEY, 
                                HF_KEYS)

NUM_NODES = int(os.environ.get("SLURM_NNODES", 1))
NUM_TASKS_PER_NODE = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))
# CONFIG = NERSC_BASE_DIR + "/config_files/experiment_configs/UAI/flow-pk-generate/flowPK.yaml"
CONFIG = NERSC_BASE_DIR + "/config_files/experiment_configs/UAI/flow-pk-predict-n-generate/flowPK.yaml"

class NERSC_Experiment:
    """Class to handle NERSC experiment setup and execution."""

    def __init__(
        self,
        epochs: int = 5,
        batch_size: int = 4,
        train_size: int = 10,
        val_size: int = 4,
        sample_size: int = 4,
        num_workers: int = 1
    ):
        self.rank0_print(
            "\nℹ️  ▶ Running NERSC: {} nodes and {} tasks/node".format(NUM_NODES, NUM_TASKS_PER_NODE)
        )
        self.config = self.experiment_configs(
            epochs, batch_size, train_size, val_size, sample_size, num_workers
        )
        self.datamodule = self.get_datamodule()

    def experiment_configs(self, epochs, batch_size, train_size, val_size, sample_size, num_workers):
        self.rank0_print("ℹ️  ▶ Starting Experiment initialization...")
        self.rank0_print(f"ℹ️    epochs: {epochs}")
        self.rank0_print(f"ℹ️    batch_size: {batch_size}")
        self.rank0_print(f"ℹ️    train_size: {train_size}")
        self.rank0_print(f"ℹ️    val_size: {val_size}")
        self.rank0_print(f"ℹ️    sample_size: {sample_size}")
        self.rank0_print(f"ℹ️    num_workers: {num_workers}")
        self.rank0_print("\nℹ️  ▶ Loading config...")
        config = FlowPKExperimentConfig.from_yaml(CONFIG)
        config.train.epochs = epochs
        config.train.batch_size = batch_size
        config.train.num_workers = num_workers
        config.mix_data.train_size = train_size
        config.mix_data.val_size = val_size
        config.mix_data.sample_size_for_generative_evaluation_val = sample_size
        config.mix_data.sample_size_for_generative_evaluation_end_of_training = sample_size
        self.rank0_print("✅  Config loaded!")
        return config

    def get_datamodule(self):
        self.rank0_print("\nℹ️  ▶ Building datamodule...")
        datamodule = AICMECompartmentsDataModule(self.config)
        self.rank0_print("✅  Datamodule ready!")
        return datamodule

    def train(self):
        MODEL_CLASS = get_model_class(self.config)  # resolves the model class from config
        self.model = MODEL_CLASS(self.config)
        logger = self.set_comet_logger()
        callback = ModelCheckpoint(dirpath=os.path.join(self.config.experiment_dir, "checkpoints")
                                    if rank_zero_only.rank == 0
                                    else None,
                                    monitor="val_mse",
                                    filename="best-{epoch:02d}-{val_mse:.4f}",
                                    mode="min",
                                    save_last=True,
                                    )
        scheduler_callbacks = list(getattr(self.model, "build_visualization_callback", lambda: [])() or [])
        trainer = L.Trainer(max_epochs=self.config.train.epochs,
                            accelerator="gpu",
                            devices="auto",
                            strategy="ddp_find_unused_parameters_true",
                            num_nodes=NUM_NODES,
                            callbacks=[callback, *scheduler_callbacks],
                            logger=logger,
                            sync_batchnorm=True,
                            log_every_n_steps=self.config.train.log_interval,
                            gradient_clip_val=self.config.train.gradient_clip_val,
                            # precision="bf16-mixed"
                            )

        trainer.fit(self.model, datamodule=self.datamodule)

    def resume(self, path_to_experiment: str):
        """Resume training from a previous run directory.

        Expects the directory to contain:
          - config.yaml          – the serialised experiment config
          - checkpoints/last.ckpt – the Lightning checkpoint to resume from
        """
        config_path = os.path.join(path_to_experiment, "config.yaml")
        ckpt_path   = os.path.join(path_to_experiment, "checkpoints", "last.ckpt")

        self.rank0_print(f"\nℹ️  ▶ Resuming experiment from: {path_to_experiment}")
        self.rank0_print(f"ℹ️    config  : {config_path}")
        self.rank0_print(f"ℹ️    ckpt    : {ckpt_path}")

        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        self.config = FlowPKExperimentConfig.from_yaml(config_path)
        self.datamodule = self.get_datamodule()

        MODEL_CLASS = get_model_class(self.config)
        self.model = MODEL_CLASS(self.config)
        logger = self.set_comet_logger()
        callback = ModelCheckpoint(
            dirpath=os.path.join(self.config.experiment_dir, "checkpoints")
                    if rank_zero_only.rank == 0
                    else None,
            monitor="val_mse",
            filename="best-{epoch:02d}-{val_mse:.4f}",
            mode="min",
            save_last=True,
        )
        scheduler_callbacks = list(getattr(self.model, "build_visualization_callback", lambda: [])() or [])

        trainer = L.Trainer(
            max_epochs=self.config.train.epochs,
            accelerator="gpu",
            devices="auto",
            strategy="ddp_find_unused_parameters_true",
            num_nodes=NUM_NODES,
            callbacks=[callback, *scheduler_callbacks],
            logger=logger,
            sync_batchnorm=True,
            log_every_n_steps=self.config.train.log_interval,
            gradient_clip_val=self.config.train.gradient_clip_val,
        )

        self.rank0_print("✅  Resuming trainer from checkpoint...")
        trainer.fit(self.model, datamodule=self.datamodule, ckpt_path=ckpt_path)

    def set_comet_logger(self):
        """Initialise Comet logger, preferring config-provided keys."""
        # Only rank 0 should create a real logger, others get None (which Lightning handles)
        if rank_zero_only.rank != 0:
            return None
            
        comet_key = getattr(self.config, "comet_ai_key", None)
        if isinstance(comet_key, str):
            comet_key = comet_key.strip()
            if not comet_key or comet_key.lower() in ("none", "null"):
                comet_key = None
        if comet_key is None:
            comet_key = COMET_API_KEY

        logger = CometLogger(api_key=COMET_API_KEY,
                             project_name=PROJECT,
                             workspace=WORKSPACE,
                             offline_directory=".",
                             experiment_key=self.config.experiment_indentifier
                            if self.config.experiment_indentifier
                            else None,
                            )

        if self.config.experiment_indentifier is None:  # if new experiment
            self.config.experiment_indentifier = (
                logger.experiment.get_key() if hasattr(logger.experiment, "get_key") else None
            )
            self.rank0_print(f"INFO: ▶ Comet Experiment ID: {self.config.experiment_indentifier}")
            config_dict = asdict(self.config) if is_dataclass(self.config) else {}

            try:
                setattr(self.model, "hparams", self.config)
            except Exception:
                pass

            logger.experiment.log_parameters(config_dict)

            if self.config.experiment_indentifier is not None:
                self.config.experiment_dir = os.path.join(NERSC_EXPERIMENT_DIR, PROJECT, self.config.experiment_name, self.config.experiment_indentifier)
                os.makedirs(self.config.experiment_dir, exist_ok=True)
                self.config.to_yaml(os.path.join(self.config.experiment_dir, "config.yaml"))
        elif not getattr(self.config, "experiment_dir", None):
            # Resuming an existing experiment: experiment_dir must be reconstructed
            # if it wasn't already present in the loaded config.
            self.config.experiment_dir = os.path.join(
                NERSC_EXPERIMENT_DIR, PROJECT,
                self.config.experiment_name,
                self.config.experiment_indentifier,
            )
            os.makedirs(self.config.experiment_dir, exist_ok=True)

        return logger

    def rank0_print(self, message: str):
        if rank_zero_only.rank == 0:
            print(message)


if __name__ == "__main__":

    if len(sys.argv) == 1: # train new experiment 
        experiment = NERSC_Experiment(epochs=100,
                                      batch_size=16,
                                      train_size=12_800,
                                      val_size=256,
                                      sample_size=10,
                                      num_workers=32
                                      )
        experiment.train()

    else: # resume existing experiment with provided key
        exp_key = sys.argv[1]
        experiment = NERSC_Experiment()
        exp_dir = os.path.join(NERSC_EXPERIMENT_DIR, PROJECT, "functional-flow-pk", exp_key)
        experiment.resume(exp_dir) 
