import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class SchedulerTaskConfig:
    """Typed configuration for one scheduler task.

    ``n_samples`` is interpreted by the scheduler only for non-internal sample
    sources. For ``sample_source='task_internal'`` the task owns its sampling
    logic and ``n_samples`` must stay ``0`` as a sentinel value, even when the
    task internally generates one or more samples per target/trajectory.
    """

    name: str
    fn_key: str
    n_samples: int = 0
    sample_source: str = "unconditional"
    split: str = "val"
    empirical_name: Optional[str] = None
    save_to_disk: bool = True
    log_prefix: str = "val"
    use_ema: bool = False
    checkpoint_metric: bool = False
    checkpoint_metric_name: Optional[str] = None
    checkpoint_mode: str = "min"
    task_cfg: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulerConfig:
    """Typed configuration for scheduler-driven callback execution."""

    percent_step: float = 1.0
    include_end: bool = True
    skip_sanity_check: bool = True
    store_samples: bool = True
    max_samples_per_group: int = 32
    keep_temp_files: bool = False
    cache_dir: Optional[str] = None
    # Supported selectors are:
    # - ``end`` for in-memory train-end weights,
    # - ``last`` / ``best`` for experiment checkpoint callbacks,
    # - scheduler-managed metric checkpoint names emitted by tasks.
    checkpoint_used_in_end: List[str] = field(default_factory=lambda: ["end"])
    tasks_validation: List[SchedulerTaskConfig] = field(default_factory=list)
    task_during: List[SchedulerTaskConfig] = field(default_factory=list)
    tasks_end: List[SchedulerTaskConfig] = field(default_factory=list)


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 8
    gradient_clip_val: float = 1.0
    optimizer_name: str = "AdamW"
    learning_rate: float = 0.0001
    weight_decay: float = 1.0e-4
    num_workers: int = 3
    persistent_workers: bool = True
    shuffle_val: bool = True

    num_batch_plot: int = 1
    log_interval: int = 1  # Frequency of logging and visualization

    # Scheduler-driven PK evaluation and visualization.
    callbacks_scheduler: Optional[Union[SchedulerConfig, Dict[str, Any]]] = None

    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1.0e-8
    amsgrad: bool = False
    scheduler_name: str = "CosineAnnealingLR"
    scheduler_params: Dict[str, Union[float, int]] = field(
        default_factory=lambda: {"T_max": 1000, "eta_min": 5.0e-5, "last_epoch": -1}
    )

    @classmethod
    def from_yaml(cls, file_path: Union[str, os.PathLike]) -> "TrainingConfig":
        """Instantiate the training configuration from a YAML file."""

        with open(file_path, "r", encoding="utf-8") as handle:
            config_dict = yaml.safe_load(handle) or {}

        if isinstance(config_dict, dict) and "train" in config_dict:
            config_dict = config_dict.get("train") or {}

        if not isinstance(config_dict, dict):
            raise TypeError("Expected 'train' section in YAML to be a mapping.")

        return cls(**cls._filter_kwargs(config_dict))

    @classmethod
    def _filter_kwargs(cls, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Drop unknown keys (including deprecated logging flags)."""

        if not isinstance(raw, dict):
            return {}
        valid = {f.name for f in fields(cls)}
        return {key: value for key, value in raw.items() if key in valid}
