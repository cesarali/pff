import torch
import lightning.pytorch as L
import yaml
import os
from dataclasses import is_dataclass, asdict

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
        num_workers: int = 1,
    ):
        self.rank0_print(
            "\nℹ️  ▶ Running NERSC: {} nodes and {} tasks/node".format(NUM_NODES, NUM_TASKS_PER_NODE)
        )
        self.config = self.experiment_configs(
            epochs, batch_size, train_size, val_size, sample_size, num_workers
        )
        self.datamodule = self.get_datamodule()

    def experiment_configs(
        self, epochs, batch_size, train_size, val_size, sample_size, num_workers
    ):
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
        checkpoint_dir = self._resolve_checkpoint_dir()
        callback = ModelCheckpoint(
            dirpath=checkpoint_dir if rank_zero_only.rank == 0 else None,
            monitor="val_mse",
            filename="best-{epoch:02d}-{val_mse:.4f}",
            mode="min",
            save_last=True,
        )
        scheduler_callbacks = list(getattr(self.model, "build_visualization_callback", lambda: [])() or [])
        callbacks = [callback, *scheduler_callbacks]

        trainer = L.Trainer(
            max_epochs=self.config.train.epochs,
            accelerator="gpu",
            devices="auto",
            strategy="ddp_find_unused_parameters_true",
            num_nodes=NUM_NODES,
            callbacks=callbacks,
            logger=logger,
            default_root_dir=self._resolve_experiment_dir(),
            sync_batchnorm=True,
            gradient_clip_val=self.config.train.gradient_clip_val,
            # precision="bf16-mixed"
        )

        trainer.fit(self.model, datamodule=self.datamodule)

    # def resume(self, ckpt_path):
    #     self.model = MODEL_CLASS.load_from_checkpoint(ckpt_path,
    #                                                   config=self.config,
    #                                                   map_location="cpu"
    #                                                   )
    #     logger = self.set_comet_logger()
    #     callback = L.callbacks.ModelCheckpoint(dirpath=None,
    #                                             monitor="val_mse",
    #                                             filename="best-{epoch:02d}-{val_mse:.4f}",
    #                                             mode="min",
    #                                             save_last=True,
    #                                                 )
    #     trainer = L.Trainer(max_epochs=self.config.train.epochs,
    #                         accelerator='gpu',
    #                         devices='auto',
    #                         strategy='ddp',
    #                         num_nodes=NUM_NODES,
    #                         callbacks=[callback],
    #                         logger=logger,
    #                         sync_batchnorm=True,
    #                         gradient_clip_val=1.0,
    #                         )
    #     trainer.fit(self.model, datamodule=self.datamodule, ckpt_path=ckpt_path)

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

        logger = CometLogger(
            api_key=comet_key,
            project=PROJECT,
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
                self.config.experiment_dir = os.path.join(
                    NERSC_EXPERIMENT_DIR, self.config.experiment_name, self.config.experiment_indentifier
                )
                os.makedirs(self.config.experiment_dir, exist_ok=True)
                self.config.to_yaml(os.path.join(self.config.experiment_dir, "config.yaml"))

        return logger

    def _resolve_experiment_dir(self) -> str:
        """Return a deterministic experiment root directory."""
        if self.config.experiment_dir:
            os.makedirs(self.config.experiment_dir, exist_ok=True)
            return self.config.experiment_dir

        experiment_id = self.config.experiment_indentifier or "local-run"
        self.config.experiment_dir = os.path.join(
            NERSC_EXPERIMENT_DIR, self.config.experiment_name, experiment_id
        )
        os.makedirs(self.config.experiment_dir, exist_ok=True)
        return self.config.experiment_dir

    def _resolve_checkpoint_dir(self) -> str:
        """Return the checkpoint directory used by both checkpoint callbacks."""
        checkpoint_dir = os.path.join(self._resolve_experiment_dir(), "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        return checkpoint_dir

    def rank0_print(self, message: str):
        if rank_zero_only.rank == 0:
            print(message)


if __name__ == "__main__":
    experiment = NERSC_Experiment(
        epochs=20, batch_size=128, train_size=10_000, val_size=2_000, sample_size=20, num_workers=4
    )
    experiment.train()
