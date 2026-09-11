"""NERSC Perlmutter – Optuna hyperparameter scan for FlowPK.

Each Optuna trial runs a short single-GPU training, then returns the best
validation MSE as the objective value.  A shared SQLite database (or any
other Optuna storage URL) makes the study resumable and allows parallel
workers to collaborate without extra infrastructure.

Usage
-----
Sequential (one trial at a time):

    python nersc_perlmutter_experiment_scan.py

Parallel workers on different GPUs (Optuna coordinates via shared DB):

    CUDA_VISIBLE_DEVICES=0 python nersc_perlmutter_experiment_scan.py &
    CUDA_VISIBLE_DEVICES=1 python nersc_perlmutter_experiment_scan.py &

Environment variables
---------------------
SCAN_EPOCHS        Number of training epochs per trial  (default: 20)
SCAN_TRAIN_SIZE    Training set size per trial           (default: 300)
SCAN_VAL_SIZE      Validation set size per trial         (default: 100)
SCAN_SAMPLE_SIZE   Generative sample size per trial      (default: 10)
SCAN_NUM_WORKERS   DataLoader workers per trial          (default: 4)
SCAN_N_TRIALS      Trials to run in this process         (default: 50)
SCAN_STUDY_NAME    Optuna study name                     (default: flowpk_hparam_scan)
SCAN_STORAGE       Optuna storage URL                    (default: sqlite:///<BASE_DIR>/optuna_<study_name>.db)
"""

import copy
import os

import torch
import lightning.pytorch as L
import optuna
from lightning.pytorch.callbacks import ModelCheckpoint

# ---------------------------------------------------------------------------
# Optional: Optuna's PyTorch Lightning pruning callback.
# Install via: pip install optuna-integration
# If unavailable we fall back to a no-op placeholder so the rest of the
# code stays identical.
# ---------------------------------------------------------------------------
try:
    from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
    _PRUNING_AVAILABLE = True
except ImportError:
    try:
        from optuna.integration import PyTorchLightningPruningCallback  # type: ignore[no-redef]
        _PRUNING_AVAILABLE = True
    except ImportError:
        _PRUNING_AVAILABLE = False

        class PyTorchLightningPruningCallback(L.Callback):  # type: ignore[no-redef]
            """No-op fallback when optuna-integration is not installed."""
            def __init__(self, trial, monitor):
                pass

from pff.config_classes.flow_pk_config import FlowPKExperimentConfig
from pff.data.datasets.aicme_datasets import AICMECompartmentsDataModule
from pff.models import get_model_class
from pff.utils import BASE_DIR


# ---------------------------------------------------------------------------
# Base config – same entry-point as NERSC_Experiment
# ---------------------------------------------------------------------------
CONFIG = BASE_DIR + "/config_files/experiment_configs/UAI/flow-pk-predict-n-generate/flowPK.yaml"

# ---------------------------------------------------------------------------
# Scan-level constants (override via environment variables or subclassing)
# ---------------------------------------------------------------------------
SCAN_EPOCHS      = int(os.environ.get("SCAN_EPOCHS",      4))
SCAN_TRAIN_SIZE  = int(os.environ.get("SCAN_TRAIN_SIZE",  3000))
SCAN_VAL_SIZE    = int(os.environ.get("SCAN_VAL_SIZE",    300))
SCAN_SAMPLE_SIZE = int(os.environ.get("SCAN_SAMPLE_SIZE", 10))
SCAN_NUM_WORKERS = int(os.environ.get("SCAN_NUM_WORKERS", 32))
SCAN_N_TRIALS    = int(os.environ.get("SCAN_N_TRIALS",    50))
SCAN_STUDY_NAME  = os.environ.get("SCAN_STUDY_NAME", "flowpk_hparam_scan")
SCAN_STORAGE     = os.environ.get(
    "SCAN_STORAGE",
    f"sqlite:///{BASE_DIR}/optuna_{SCAN_STUDY_NAME}.db",
)


class NERSC_Experiment_Scan:
    """Optuna-based hyperparameter scan wrapping the FlowPK training loop.

    The class loads a base ``FlowPKExperimentConfig`` from YAML once.  For
    every Optuna trial it deep-copies that config, applies the suggested
    hyper-parameter values, builds the data module + model, runs a short
    Lightning training run on a single GPU, and returns the best observed
    ``val_mse`` as the objective.

    Search space
    ------------
    ``SEARCH_SPACE`` is a class-level dict that maps a dot-separated
    ``<config_section>.<attribute>`` string to an Optuna suggestion spec:

        {
            "type": "float" | "int" | "categorical",
            # for float / int:
            "low": ..., "high": ..., "log": True/False,
            # for categorical:
            "choices": [...],
        }

    Subclass and override ``SEARCH_SPACE`` to tune a different set of params.
    """

    # ------------------------------------------------------------------
    # Search space definition
    # Dot-notation: "<config_section>.<attribute>"
    # Sections map directly to FlowPKExperimentConfig fields:
    #   train           -> config.train          (TrainingConfig)
    #   vector_field    -> config.vector_field   (VectorFieldPKConfig)
    #   source_process  -> config.source_process (SourceProcessConfig)
    # ------------------------------------------------------------------
    SEARCH_SPACE: dict = {
        # --- Training hyper-parameters ---
        "train.learning_rate": dict(
            type="float", low=1e-7, high=1e-2, log=True
        ),
        # "train.weight_decay": dict(
        #     type="float", low=1e-6, high=1e-2, log=True
        # ),
        # "train.gradient_clip_val": dict(
        #     type="float", low=0.1, high=2.0
        # ),
        "train.batch_size": dict(
            type="categorical", choices=[16, 32, 64, 128, 256]
        ),
        # --- Vector field architecture ---
        "vector_field.hidden_dim": dict(
            type="categorical", choices=[32, 64, 128, 256]
        ),
        # "vector_field.dropout": dict(
        #     type="float", low=0.0, high=0.5
        # ),
        "vector_field.encoder_attention_layers": dict(
            type="categorical", choices=[2, 4, 8]
        ),
        "vector_field.decoder_attention_layers": dict(
            type="categorical", choices=[2, 4, 8]
        ),
        # "vector_field.fourier_modes": dict(
        #     type="categorical", choices=[16, 20, 32]
        # ),
        # --- Source process ---
        # "source_process.flow_sigma": dict(
        #     type="float", low=1e-6, high=1e-2, log=True
        # ),
        "source_process.gp_variance": dict(
            type="float", low=0.001, high=10., log=True
        ),
        "source_process.gp_length_scale": dict(
            type="float", low=0.001, high=10., log=True
        ),
    }

    def __init__(
        self,
        epochs: int = SCAN_EPOCHS,
        train_size: int = SCAN_TRAIN_SIZE,
        val_size: int = SCAN_VAL_SIZE,
        sample_size: int = SCAN_SAMPLE_SIZE,
        num_workers: int = SCAN_NUM_WORKERS,
    ):
        self.epochs = epochs
        self.train_size = train_size
        self.val_size = val_size
        self.sample_size = sample_size
        self.num_workers = num_workers

        print(f"Loading base config from: {CONFIG}")
        self.base_config = FlowPKExperimentConfig.from_yaml(CONFIG)
        self._apply_scan_sizes(self.base_config)
        print("Base config loaded. Scan ready.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_scan_sizes(self, config: FlowPKExperimentConfig) -> None:
        """Apply fixed dataset / epoch sizes to a config in-place.

        Also disables expensive end-of-training empirical evaluations
        that would slow down every trial.
        """
        config.train.epochs = self.epochs
        config.train.num_workers = self.num_workers
        config.mix_data.train_size = self.train_size
        config.mix_data.val_size = self.val_size
        config.mix_data.sample_size_for_generative_evaluation_val = self.sample_size
        config.mix_data.sample_size_for_generative_evaluation_end_of_training = self.sample_size

        # Disable heavy end-of-training evaluations for scan speed.
        config.train.callbacks_scheduler = None

    @staticmethod
    def _suggest(trial: optuna.Trial, name: str, spec: dict):
        """Dispatch an Optuna suggest call based on the ``type`` key."""
        kind = spec["type"]
        if kind == "float":
            return trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        if kind == "int":
            return trial.suggest_int(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        if kind == "categorical":
            return trial.suggest_categorical(name, spec["choices"])
        raise ValueError(f"Unknown hyperparameter type: {kind!r}")

    def _build_trial_config(self, trial: optuna.Trial) -> FlowPKExperimentConfig:
        """Deep-copy the base config and apply Optuna suggestions."""
        config = copy.deepcopy(self.base_config)

        # Each trial is a fresh experiment (no Comet key / dir reuse).
        config.experiment_indentifier = None
        config.experiment_dir = None

        for param_name, spec in self.SEARCH_SPACE.items():
            value = self._suggest(trial, param_name, spec)
            section_name, attr = param_name.split(".", 1)
            section_obj = getattr(config, section_name)
            setattr(section_obj, attr, value)

        return config

    # ------------------------------------------------------------------
    # Optuna objective
    # ------------------------------------------------------------------

    def objective(self, trial: optuna.Trial) -> float:
        """Train one model and return best ``val_mse`` (lower = better).

        Raises ``optuna.TrialPruned`` if the pruner decides to stop the
        trial early (requires ``optuna-integration`` to be installed).
        """
        config = self._build_trial_config(trial)

        print(f"\n--- Trial #{trial.number} ---")
        for k, v in trial.params.items():
            print(f"  {k}: {v}")

        datamodule = AICMECompartmentsDataModule(config)
        MODEL_CLASS = get_model_class(config)
        model = MODEL_CLASS(config)

        # Build callbacks.
        pruning_cb = PyTorchLightningPruningCallback(trial, monitor="val_mse")
        checkpoint_cb = ModelCheckpoint(
            monitor="val_mse",
            mode="min",
            save_last=False,
            save_top_k=1,
            dirpath=None,  # in-memory; no persistent checkpoint needed
        )
        scheduler_callbacks = list(getattr(model, "build_visualization_callback", lambda: [])() or [])

        accelerator = "gpu" if torch.cuda.is_available() else "cpu"

        trainer = L.Trainer(
            max_epochs=config.train.epochs,
            accelerator=accelerator,
            devices=1,           # single GPU per trial; parallelism is at the study level
            num_nodes=1,
            callbacks=[checkpoint_cb, *scheduler_callbacks, pruning_cb],
            logger=False,        # no logger during scan to keep output clean
            enable_checkpointing=True,
            gradient_clip_val=config.train.gradient_clip_val,
            enable_progress_bar=False,
        )

        try:
            trainer.fit(model, datamodule=datamodule)
        except optuna.exceptions.TrialPruned:
            raise  # let Optuna handle it

        val_mse = trainer.callback_metrics.get("val_mse")
        if val_mse is None:
            # val_mse not logged – prune to avoid polluting the study.
            raise optuna.exceptions.TrialPruned("val_mse not found in callback_metrics")

        result = float(val_mse)
        print(f"  -> val_mse: {result:.6f}")
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_study(
        self,
        n_trials: int = SCAN_N_TRIALS,
        study_name: str = SCAN_STUDY_NAME,
        storage: str = SCAN_STORAGE,
        direction: str = "minimize",
        pruner: optuna.pruners.BasePruner = None,
        sampler: optuna.samplers.BaseSampler = None,
    ) -> optuna.Study:
        """Create (or resume) an Optuna study and run ``n_trials`` trials.

        Parameters
        ----------
        n_trials:
            Number of trials to run in **this process**.  When several
            workers share the same ``storage`` they each call
            ``run_study`` independently and Optuna distributes the work.
        study_name:
            Unique study identifier used by Optuna.
        storage:
            Optuna storage URL.  Defaults to a local SQLite file so that
            results persist across restarts.  For distributed scans use a
            shared database, e.g. ``"mysql://user:pass@host/db"``.
        direction:
            ``"minimize"`` (default) or ``"maximize"``.
        pruner:
            Optuna pruner.  Defaults to ``MedianPruner`` when ``None``.
        sampler:
            Optuna sampler.  Defaults to ``TPESampler`` when ``None``.

        Returns
        -------
        optuna.Study
            The completed study object.  Inspect ``study.best_trial`` and
            ``study.best_params`` for the winning configuration.
        """
        if pruner is None:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=max(1, self.epochs // 5),
            )
        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=42)

        print(f"\nOptuna study: {study_name!r}")
        print(f"Storage     : {storage}")
        print(f"Trials      : {n_trials}")
        if not _PRUNING_AVAILABLE:
            print(
                "WARNING: optuna-integration not found; pruning is disabled. "
                "Install it with: pip install optuna-integration"
            )

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            pruner=pruner,
            sampler=sampler,
            load_if_exists=True,   # resume a previous scan if the DB already exists
        )

        study.optimize(
            self.objective,
            n_trials=n_trials,
            catch=(Exception,),    # log failed trials without aborting the scan
        )

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        print("\n=== Scan complete ===")
        try:
            print(f"  Best trial  : #{study.best_trial.number}")
            print(f"  Best val_mse: {study.best_value:.6f}")
            print("  Best params :")
            for k, v in study.best_trial.params.items():
                print(f"    {k}: {v}")
        except ValueError:
            print("  No completed trials found.")

        return study

    def print_search_space(self) -> None:
        """Pretty-print the configured search space."""
        print("\nSearch space:")
        for name, spec in self.SEARCH_SPACE.items():
            kind = spec["type"]
            if kind in ("float", "int"):
                scale = " (log)" if spec.get("log") else ""
                print(f"  {name}: {kind}[{spec['low']}, {spec['high']}]{scale}")
            else:
                print(f"  {name}: categorical{spec['choices']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scan = NERSC_Experiment_Scan(
        epochs=SCAN_EPOCHS,
        train_size=SCAN_TRAIN_SIZE,
        val_size=SCAN_VAL_SIZE,
        sample_size=SCAN_SAMPLE_SIZE,
        num_workers=SCAN_NUM_WORKERS,
    )

    scan.print_search_space()

    study = scan.run_study(
        n_trials=SCAN_N_TRIALS,
        study_name=SCAN_STUDY_NAME,
        storage=SCAN_STORAGE,
    )
