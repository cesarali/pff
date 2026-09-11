import json
import inspect
import os
import re
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

try:  # pragma: no cover - prefer PyYAML when available
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    from pff.config_classes import yaml_fallback as yaml

import lightning.pytorch as pl
import torch
from lightning import Callback

from pff.config_classes.node_pk_config import (
    EncoderDecoderNetworkConfig,
    MetaDosingConfig,
    MetaStudyConfig,
    MixDataConfig,
    NodePKExperimentConfig,
    ObservationsConfig,
)
from pff.config_classes.training_config import TrainingConfig

EXPERIMENT_CONFIG_FILENAME = "experiment_config.yaml"


def _extract_epoch_and_step_from_checkpoint_name(checkpoint_name: str) -> tuple[int, int]:
    """Return ``(epoch, step)`` parsed from supported checkpoint filename patterns."""
    epoch_match = re.search(r"epoch[=_](\d+)", checkpoint_name)
    step_match = re.search(r"step[=_](\d+)", checkpoint_name)
    epoch = int(epoch_match.group(1)) if epoch_match else -1
    step = int(step_match.group(1)) if step_match else -1
    return epoch, step


def _resolve_scheduler_metric_checkpoint(
    experiment_dir: Path,
    checkpoint_type: str,
) -> Path | None:
    """Resolve a scheduler-managed checkpoint by metric name.

    Scheduler checkpoints are written under:

    ``<experiment_dir>/scheduler_metric_checkpoints/<task_name>/best-epoch_...-<metric>=...ckpt``

    The ``checkpoint_type`` is expected to be the configured scheduler metric
    selector, for example ``"log_rmse"``.
    """

    scheduler_root = experiment_dir / "scheduler_metric_checkpoints"
    if not scheduler_root.is_dir():
        return None

    matches = [
        path
        for path in scheduler_root.rglob("*.ckpt")
        if path.is_file() and f"-{checkpoint_type}=" in path.name
    ]
    if not matches:
        return None

    return max(matches, key=lambda path: _extract_epoch_and_step_from_checkpoint_name(path.name))


def resolve_model_config_kwarg(model_class: type) -> str:
    """Return the constructor kwarg name used to pass experiment config objects."""
    signature = inspect.signature(model_class.__init__)
    parameters = signature.parameters
    if "experiment_config" in parameters:
        return "experiment_config"
    if "model_config" in parameters:
        return "model_config"
    if "config" in parameters:
        return "config"
    raise ValueError(f"{model_class.__name__}.__init__() does not expose a config argument.")


def load_model_from_checkpoint_path(
    model_class: type,
    checkpoint_path: str | os.PathLike[str],
    *,
    model_config: Any,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> torch.nn.Module:
    """Load a model from either a Lightning checkpoint or a state_dict-only checkpoint.

    Scheduler-managed metric checkpoints intentionally store only ``state_dict`` to
    avoid the distributed synchronization semantics of ``trainer.save_checkpoint``.
    Those files cannot be passed to ``LightningModule.load_from_checkpoint(...)``.

    This helper accepts both formats:
    - full Lightning checkpoints with metadata such as
      ``pytorch-lightning_version``
    - light-weight checkpoints containing ``{"state_dict": ...}``
    """

    checkpoint_path = str(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    config_kwarg = resolve_model_config_kwarg(model_class)

    if isinstance(checkpoint, dict) and "pytorch-lightning_version" in checkpoint:
        return model_class.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=map_location,
            strict=strict,
            **{config_kwarg: model_config},
        )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        # Support raw ``state_dict`` payloads saved without an outer wrapper.
        state_dict = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint payload in '{checkpoint_path}'.")

    model = model_class(**{config_kwarg: model_config})
    model.load_state_dict(state_dict, strict=strict)
    return model


def get_lightning_checkpoint_path(experiment_dir, checkpoint_type="best"):
    """
    Checks for lightning checkpoints in experiment folders and returns
    the checkpoint with the highest epoch number for the specified type,
    preferring non-versioned filenames when epochs are equal.

    checkpoint_type: one of ['best', 'best_log', 'last'] or a scheduler metric
    selector such as ``'log_rmse'``.
    - 'best' matches files like 'best-epoch=...'
    - 'best_log' matches 'best_log_rmse.ckpt'
    - 'last' matches 'last.ckpt'
    - scheduler metric selectors match files under
      ``scheduler_metric_checkpoints/**/best-epoch_...-<metric>=...ckpt``
    """
    checkpoints_path = Path(experiment_dir)
    if not checkpoints_path.exists():
        print(f"CHECKPOINTS DIRECTORY DOES NOT EXIST: {checkpoints_path}")
        return None
    if not checkpoints_path.is_dir():
        print(f"CHECKPOINTS PATH IS NOT A DIRECTORY: {checkpoints_path}")
        return None

    explicit_checkpoint_path = Path(str(checkpoint_type))
    if explicit_checkpoint_path.is_file():
        return explicit_checkpoint_path

    all_checkpoints = os.listdir(checkpoints_path)
    checkpoint_epochs = {}

    if len(all_checkpoints) == 0:
        print("NO CHECKPOINTS FOUND IN DIRECTORY.")
        return None

    # Populate available checkpoints
    for ckpt in all_checkpoints:
        ckpt_path = checkpoints_path / ckpt
        if ckpt.startswith("best-epoch="):
            match = re.search(r"epoch=(\d+)", ckpt)
            if match:
                epoch = int(match.group(1))
                if "best" not in checkpoint_epochs or epoch > checkpoint_epochs["best"]["epoch"]:
                    checkpoint_epochs["best"] = {"epoch": epoch, "path": ckpt_path}
        elif ckpt == "best_log_rmse.ckpt":
            checkpoint_epochs["best_log"] = {"epoch": -1, "path": ckpt_path}
        elif ckpt == "last.ckpt":
            checkpoint_epochs["last"] = {"epoch": -1, "path": ckpt_path}
        elif ckpt.startswith("recon-epoch="):
            match = re.search(r"epoch=(\d+)", ckpt)
            if match:
                epoch = int(match.group(1))
                if "recon" not in checkpoint_epochs or epoch > checkpoint_epochs["recon"]["epoch"]:
                    checkpoint_epochs["recon"] = {"epoch": epoch, "path": ckpt_path}

    if checkpoint_type in checkpoint_epochs:
        return checkpoint_epochs[checkpoint_type]["path"]

    scheduler_checkpoint = _resolve_scheduler_metric_checkpoint(checkpoints_path, checkpoint_type)
    if scheduler_checkpoint is not None:
        return scheduler_checkpoint

    if not checkpoint_epochs:
        print("NO VALID CHECKPOINTS FOUND, RETURNING FIRST AVAILABLE")
        return checkpoints_path / all_checkpoints[0]

    for preferred_type in ["last", "best", "recon"]:
        if preferred_type in checkpoint_epochs:
            print(
                f"CHECKPOINT TYPE '{checkpoint_type}' NOT FOUND, RETURNING '{preferred_type}' INSTEAD."
            )
            return checkpoint_epochs[preferred_type]["path"]

    print("NO RECOGNIZED CHECKPOINT TYPES FOUND, RETURNING FIRST AVAILABLE")
    return checkpoints_path / all_checkpoints[0]


# Test the function
class UnusedParamReporter(pl.Callback):
    """
    checks for unussed parameters for ddp strategy compatibility

            trainer = Trainer(
            default_root_dir=self.experiment_dir,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=self.devices,
            strategy=self.strategy,
            logger=self.logger,
            #max_epochs=self.model_config.train.epochs,
            max_epochs=3,
            #callbacks=self.callbacks or [], # the callbacks are defined one in ranck zero empty list for the workers
            callbacks=[UnusedParamReporter()],
            log_every_n_steps=self.model_config.train.log_interval,
            gradient_clip_val=self.model_config.train.gradient_clip_val,
            reload_dataloaders_every_n_epochs=1,
        )
    """

    def on_train_epoch_start(self, trainer, pl_module):
        self.touched = set()  # reset every epoch

    def on_after_backward(self, trainer, pl_module):
        # runs *immediately* after each backward() call
        for n, p in pl_module.named_parameters():
            if p.requires_grad and p.grad is not None:
                self.touched.add(n)

    def on_train_epoch_end(self, trainer, pl_module):
        unused = [
            n for n, p in pl_module.named_parameters() if p.requires_grad and n not in self.touched
        ]
        if unused:
            print("\n[Unused parameters]")
            for n in unused:
                print("  •", n)


class NonFiniteLossCallback(pl.Callback):
    """Abort training when any logged metric is non-finite and record the last
    valid checkpoint path."""

    def __init__(self):
        from typing import Optional

        self.last_valid_checkpoint: Optional[str] = None

    def on_train_start(self, trainer, pl_module) -> None:
        setattr(pl_module, "nan_detected", False)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        metrics = trainer.callback_metrics
        has_invalid = False
        for v in metrics.values():
            if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                has_invalid = True
                break
        if has_invalid:
            setattr(pl_module, "nan_detected", True)
            trainer.should_stop = True
        else:
            for cb in trainer.callbacks:
                if isinstance(cb, pl.callbacks.ModelCheckpoint) and getattr(
                    cb, "last_model_path", None
                ):
                    self.last_valid_checkpoint = cb.last_model_path
                    break


def metrics_are_finite(metrics: dict[str, torch.Tensor]) -> bool:
    """Return ``True`` if all tensor values are finite."""
    for v in metrics.values():
        if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
            return False
    return True


def dataclass_from_dict(klass, dikt):
    if not is_dataclass(klass):
        return dikt
    kwargs = {}
    for f in fields(klass):
        name = f.name
        if name in dikt:
            value = dikt[name]
            # nested dataclass
            if is_dataclass(f.type) and isinstance(value, dict):
                kwargs[name] = dataclass_from_dict(f.type, value)
            else:
                kwargs[name] = value
    return klass(**kwargs)


def _filter_dataclass_kwargs(dataclass_type, raw_mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only fields declared on ``dataclass_type``.

    This makes Comet config reconstruction tolerant to legacy parameters that
    may still exist on older runs but were removed from the current config
    dataclasses.
    """

    if not isinstance(raw_mapping, dict):
        return {}
    valid = {field_.name for field_ in fields(dataclass_type)}
    return {key: value for key, value in raw_mapping.items() if key in valid}


def get_experiment_config_path(
    experiment_dir: str,
    filename: str = EXPERIMENT_CONFIG_FILENAME,
) -> str:
    """Return the expected YAML path for an experiment configuration."""
    return os.path.join(experiment_dir, filename)


def save_experiment_config_yaml(
    exp_config: Any,
    experiment_dir: str,
    filename: str = EXPERIMENT_CONFIG_FILENAME,
) -> str:
    """Persist ``exp_config`` as YAML under ``experiment_dir`` and return the file path."""
    if exp_config is None:
        raise ValueError("exp_config must be provided to save the experiment YAML.")
    os.makedirs(experiment_dir, exist_ok=True)
    yaml_path = get_experiment_config_path(experiment_dir, filename)

    to_yaml = getattr(exp_config, "to_yaml", None)
    if callable(to_yaml):
        to_yaml(yaml_path)
        return yaml_path

    payload = asdict(exp_config) if is_dataclass(exp_config) else exp_config
    with open(yaml_path, "w", encoding="utf-8") as handle:
        yaml.dump(payload, handle, default_flow_style=False)
    return yaml_path


def load_experiment_config_yaml(
    experiment_dir: str,
    filename: str = EXPERIMENT_CONFIG_FILENAME,
):
    """Load an experiment config YAML from ``experiment_dir`` and parse it."""
    yaml_path = get_experiment_config_path(experiment_dir, filename)
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"Experiment config YAML not found at: {yaml_path}")
    from pff.models import get_model_config

    return get_model_config(yaml_path)


def _convert_value(val_str: str) -> Any:
    """
    Convert a Comet string value to an appropriate Python type.

    - "null" -> None
    - JSON lists/dicts -> Python list/dict
    - "true"/"false" -> bool
    - ints/floats -> int/float
    - fallback: original string
    """
    if val_str is None or val_str == "null":
        return None

    # try JSON for list or dict
    if isinstance(val_str, str):
        s = val_str.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                parsed = json.loads(s)
                return parsed
            except json.JSONDecodeError:
                pass

    low = str(val_str).lower()
    if low in ("true", "false"):
        return low == "true"

    # numeric?
    try:
        # int first (so "1" -> 1, not 1.0)
        i = int(val_str)
        return i
    except (ValueError, TypeError):
        try:
            return float(val_str)
        except (ValueError, TypeError):
            return val_str


def parse_comet_parameters_summary(parameters: List[Dict[str, Any]]) -> NodePKExperimentConfig:
    """
    Parse a list of Comet API parameter summaries into a NodePKConfig.

    - Strips "config/" or "model_config/" prefixes if present.
    - Splits names on '/', '.', or '|' to detect section and field.
    - Converts JSON lists/dicts, booleans, numbers, and "null".
    - Builds nested sub-configs for known sections.
    - Filters out any keys not defined on NodePKConfig.
    """
    nested: Dict[str, Dict[str, Any]] = {}
    top_level: Dict[str, Any] = {}

    for entry in parameters:
        raw = entry.get("name", "")
        # split on '/', '.', or '|' and drop empty parts
        parts = [p for p in re.split(r"[\/\.|]", raw) if p]
        # drop "config" or "model_config" prefix
        if parts and parts[0] in ("config", "model_config"):
            parts = parts[1:]
        if not parts:
            continue

        val = _convert_value(entry.get("valueCurrent"))
        if len(parts) == 1:
            top_level[parts[0]] = val
        elif len(parts) == 2:
            sec, fld = parts
            nested.setdefault(sec, {})[fld] = val
        # deeper nesting is ignored

    # assemble constructor kwargs
    init_kwargs: Dict[str, Any] = {}
    for sec, mapping in nested.items():
        if sec == "network":
            init_kwargs["network"] = EncoderDecoderNetworkConfig(
                **_filter_dataclass_kwargs(EncoderDecoderNetworkConfig, mapping)
            )
        elif sec == "mix_data":
            init_kwargs["mix_data"] = MixDataConfig(**_filter_dataclass_kwargs(MixDataConfig, mapping))
        elif sec in ("context_observations", "target_observations"):
            init_kwargs[sec] = ObservationsConfig(
                **_filter_dataclass_kwargs(ObservationsConfig, mapping)
            )
        elif sec == "meta_study":
            init_kwargs["meta_study"] = MetaStudyConfig(
                **_filter_dataclass_kwargs(MetaStudyConfig, mapping)
            )
        elif sec == "dosing":
            init_kwargs["dosing"] = MetaDosingConfig(
                **_filter_dataclass_kwargs(MetaDosingConfig, mapping)
            )
        elif sec == "train":
            init_kwargs["train"] = TrainingConfig(**TrainingConfig._filter_kwargs(mapping))
        # unknown sections get dropped

    init_kwargs.update(top_level)

    # filter to only the real NodePKConfig fields
    valid = {f.name for f in fields(NodePKExperimentConfig)}
    filtered = {k: v for k, v in init_kwargs.items() if k in valid}

    return NodePKExperimentConfig(**filtered)


class SkipNonFiniteGrad(Callback):
    """Cancel the optimizer step when the total grad‑norm is not finite."""

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, opt_idx):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            pl_module.parameters(), max_norm=1.0
        )  # clip + return norm
        if not torch.isfinite(grad_norm):
            pl_module.print(f"⚠️  Non‑finite grad‑norm ({grad_norm}); skipping step.")
            optimizer.zero_grad(set_to_none=True)
            trainer.fit_loop._skip_backward = True  # Lightning ≥2.2


if __name__ == "__main__":
    experiment_dir = "test_dir"
    all_checkpoints = [
        "best-epoch=01.ckpt",
        "best-epoch=19.ckpt",
        "best-epoch=20.ckpt",
        "last-epoch=11.ckpt",
        "last-epoch=20-v1.ckpt",
        "last-epoch=20.ckpt",
        "recon-epoch=01.ckpt",
        "recon-epoch=05.ckpt",
    ]

    # Simulate os.listdir
    os.listdir = lambda x: all_checkpoints

    # Test different checkpoint types
    print(get_lightning_checkpoint_path(experiment_dir, "last"))  # Should return last-epoch=20.ckpt
    print(get_lightning_checkpoint_path(experiment_dir, "best"))  # Should return best-epoch=20.ckpt
    print(
        get_lightning_checkpoint_path(experiment_dir, "recon")
    )  # Should return recon-epoch=05.ckpt
