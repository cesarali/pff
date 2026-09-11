from pathlib import Path
import json

import os
import re
import torch
import lightning.pytorch as pl
from generative_pk.config_classes.node_pk_config import NodePKConfig
from dataclasses import is_dataclass, fields
from lightning import  Callback

import os
from pathlib import Path
import re

def get_lightning_checkpoint_path(experiment_dir, checkpoint_type="best"):
    """
    Checks for lightning checkpoints in experiment folders and returns 
    the checkpoint with the highest epoch number for the specified type,
    preferring non-versioned filenames when epochs are equal.

    checkpoint_type: one of ['best', 'best_log', 'last']
    - 'best' matches files like 'best-epoch=...'
    - 'best_log' matches 'best_log_rmse.ckpt'
    - 'last' matches 'last.ckpt'
    """
    checkpoints_path = Path(experiment_dir)
    all_checkpoints = os.listdir(checkpoints_path)
    checkpoint_epochs = {}

    if len(all_checkpoints) == 0:
        print("NO CHECKPOINTS FOUND IN DIRECTORY.")
        return None

    # Populate available checkpoints
    for ckpt in all_checkpoints:
        ckpt_path = checkpoints_path / ckpt
        if ckpt.startswith("best-epoch="):
            match = re.search(r'epoch=(\d+)', ckpt)
            if match:
                epoch = int(match.group(1))
                if 'best' not in checkpoint_epochs or epoch > checkpoint_epochs['best']['epoch']:
                    checkpoint_epochs['best'] = {'epoch': epoch, 'path': ckpt_path}
        elif ckpt == "best_log_rmse.ckpt":
            checkpoint_epochs['best_log'] = {'epoch': -1, 'path': ckpt_path}
        elif ckpt == "last.ckpt":
            checkpoint_epochs['last'] = {'epoch': -1, 'path': ckpt_path}
        elif ckpt.startswith("recon-epoch="):
            match = re.search(r'epoch=(\d+)', ckpt)
            if match:
                epoch = int(match.group(1))
                if 'recon' not in checkpoint_epochs or epoch > checkpoint_epochs['recon']['epoch']:
                    checkpoint_epochs['recon'] = {'epoch': epoch, 'path': ckpt_path}

    if not checkpoint_epochs:
        print("NO VALID CHECKPOINTS FOUND, RETURNING FIRST AVAILABLE")
        return checkpoints_path / all_checkpoints[0]

    if checkpoint_type in checkpoint_epochs:
        return checkpoint_epochs[checkpoint_type]['path']

    for preferred_type in ['last', 'best', 'recon']:
        if preferred_type in checkpoint_epochs:
            print(f"CHECKPOINT TYPE '{checkpoint_type}' NOT FOUND, RETURNING '{preferred_type}' INSTEAD.")
            return checkpoint_epochs[preferred_type]['path']

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
        self.touched = set()                     # reset every epoch

    def on_after_backward(self, trainer, pl_module):
        # runs *immediately* after each backward() call
        for n, p in pl_module.named_parameters():
            if p.requires_grad and p.grad is not None:
                self.touched.add(n)

    def on_train_epoch_end(self, trainer, pl_module):
        unused = [
            n for n, p in pl_module.named_parameters()
            if p.requires_grad and n not in self.touched
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
                if isinstance(cb, pl.callbacks.ModelCheckpoint) and getattr(cb, "last_model_path", None):
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

from typing import List, Dict, Any
from generative_pk.config_classes.node_pk_config import (
    EncoderDecoderNetworkConfig,
    MixDataConfig,
    ObservationsConfig,
    MetaStudyConfig,
    MetaDosingConfig,
    TrainingConfig,
)

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
        if (s.startswith('[') and s.endswith(']')) or (s.startswith('{') and s.endswith('}')):
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

def parse_comet_parameters_summary(parameters: List[Dict[str, Any]]) -> NodePKConfig:
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
            init_kwargs["network"] = EncoderDecoderNetworkConfig(**mapping)
        elif sec == "mix_data":
            init_kwargs["mix_data"] = MixDataConfig(**mapping)
        elif sec in ("context_observations", "target_observations"):
            init_kwargs[sec] = ObservationsConfig(**mapping)
        elif sec == "meta_study":
            init_kwargs["meta_study"] = MetaStudyConfig(**mapping)
        elif sec == "dosing":
            init_kwargs["dosing"] = MetaDosingConfig(**mapping)
        elif sec == "train":
            init_kwargs["train"] = TrainingConfig(**mapping)
        # unknown sections get dropped

    init_kwargs.update(top_level)

    # filter to only the real NodePKConfig fields
    valid = {f.name for f in fields(NodePKConfig)}
    filtered = {k: v for k, v in init_kwargs.items() if k in valid}

    return NodePKConfig(**filtered)


class SkipNonFiniteGrad(Callback):
    """Cancel the optimizer step when the total grad‑norm is not finite."""

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, opt_idx):
        grad_norm = torch.nn.utils.clip_grad_norm_(pl_module.parameters(), max_norm=1.0)  # clip + return norm
        if not torch.isfinite(grad_norm):
            pl_module.print(f"⚠️  Non‑finite grad‑norm ({grad_norm}); skipping step.")
            optimizer.zero_grad(set_to_none=True)
            trainer.fit_loop._skip_backward = True      # Lightning ≥2.2

if __name__ == "__main__":
    experiment_dir = "test_dir"
    all_checkpoints = ['best-epoch=01.ckpt', 'best-epoch=19.ckpt', 'best-epoch=20.ckpt', 
                      'last-epoch=11.ckpt', 'last-epoch=20-v1.ckpt', 'last-epoch=20.ckpt', 
                      'recon-epoch=01.ckpt', 'recon-epoch=05.ckpt']
    
    # Simulate os.listdir
    os.listdir = lambda x: all_checkpoints
    
    # Test different checkpoint types
    print(get_lightning_checkpoint_path(experiment_dir, "last"))  # Should return last-epoch=20.ckpt
    print(get_lightning_checkpoint_path(experiment_dir, "best"))  # Should return best-epoch=20.ckpt
    print(get_lightning_checkpoint_path(experiment_dir, "recon")) # Should return recon-epoch=05.ckpt
