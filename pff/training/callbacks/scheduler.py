from __future__ import annotations

import logging
import math
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

try:
    import lightning.pytorch as pl
except Exception:  # pragma: no cover
    import pytorch_lightning as pl  # type: ignore

from pff.config_classes.training_config import SchedulerConfig, SchedulerTaskConfig

LOGGER = logging.getLogger(__name__)

TaskFn = Callable[..., Mapping[str, Any]]


class SampleSource(str, Enum):
    UNCONDITIONAL = "unconditional"
    VAL_BATCH = "val_batch"
    FULL_SPLIT = "full_split"
    EMPIRICAL_SET = "empirical_set"
    TASK_INTERNAL = "task_internal"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    fn_key: str
    fn: TaskFn
    n_samples: int = 0
    sample_source: SampleSource = SampleSource.UNCONDITIONAL
    split: str = "val"
    empirical_name: str | None = None
    save_to_disk: bool = True
    log_prefix: str = "val"
    checkpoint_metric: bool = False
    checkpoint_metric_name: str | None = None
    checkpoint_mode: str = "min"
    task_cfg: Mapping[str, Any] | None = None

    @classmethod
    def from_config(cls, task_cfg: SchedulerTaskConfig, *, fn: TaskFn) -> "TaskSpec":
        return cls(
            name=task_cfg.name,
            fn_key=task_cfg.fn_key,
            fn=fn,
            n_samples=int(task_cfg.n_samples),
            sample_source=SampleSource(task_cfg.sample_source),
            split=task_cfg.split,
            empirical_name=task_cfg.empirical_name,
            save_to_disk=bool(task_cfg.save_to_disk),
            log_prefix=task_cfg.log_prefix,
            checkpoint_metric=bool(task_cfg.checkpoint_metric),
            checkpoint_metric_name=task_cfg.checkpoint_metric_name,
            checkpoint_mode=task_cfg.checkpoint_mode,
            task_cfg=dict(task_cfg.task_cfg),
        )


@dataclass
class _GroupCacheRecord:
    cache_path: Path | None
    batches: Any


def _build_task_config(raw_task: Any) -> SchedulerTaskConfig:
    if isinstance(raw_task, SchedulerTaskConfig):
        return raw_task
    if not isinstance(raw_task, dict):
        raise TypeError("Scheduler task entries must be mappings.")

    known_keys = {
        "name",
        "fn_key",
        "n_samples",
        "sample_source",
        "split",
        "empirical_name",
        "save_to_disk",
        "log_prefix",
        "use_ema",
        "checkpoint_metric",
        "checkpoint_metric_name",
        "checkpoint_mode",
        "task_cfg",
    }

    task_cfg = raw_task.get("task_cfg", {})
    if task_cfg is None:
        task_cfg = {}
    if not isinstance(task_cfg, dict):
        raise TypeError("task_cfg must be a mapping when provided.")

    extra = {k: v for k, v in raw_task.items() if k not in known_keys}
    merged_task_cfg = dict(task_cfg)
    merged_task_cfg.update(extra)

    kwargs = {k: v for k, v in raw_task.items() if k in known_keys and k != "task_cfg"}
    kwargs["task_cfg"] = merged_task_cfg
    return SchedulerTaskConfig(**kwargs)


def _build_scheduler_config(raw_scheduler: Mapping[str, Any]) -> SchedulerConfig:
    kwargs = dict(raw_scheduler)
    for key in ("tasks_validation", "task_during", "tasks_end"):
        raw_tasks = kwargs.get(key, []) or []
        if not isinstance(raw_tasks, list):
            raise TypeError(f"scheduler.{key} must be a list.")
        kwargs[key] = [_build_task_config(task) for task in raw_tasks]
    return SchedulerConfig(**kwargs)


class BaseSchedulerCallback(pl.Callback):
    """Coordinate scheduled evaluation tasks during the Lightning training lifecycle.

    This callback supports three trigger points:
    - validation-batch: run tasks at each validation batch callback.
    - train-epoch-end: run ``task_during`` when epoch progress crosses configured percentages.
    - train end: run tasks once after training finishes.

    To avoid repeated generation work, tasks are grouped by
    ``(sample_source, split, empirical_name)``. For each group, the callback resolves
    batches once, generates a maximum sample set once, caches it to disk, and then
    slices that cached output per task request. Task outputs are finally normalized
    into metric/image/figure logging calls for all attached Lightning loggers.
    """

    def __init__(
        self,
        *,
        config: SchedulerConfig,
        tasks_validation: Sequence[TaskSpec],
        task_during: Sequence[TaskSpec],
        tasks_end: Sequence[TaskSpec],
    ) -> None:
        """Store scheduler configuration, task lists, and runtime state.

        Besides persisting the provided task specs, initialization precomputes
        progress milestones from ``config.percent_step`` and initializes ephemeral
        state used during training:
        - cached first validation batch for current epoch
        - index of next milestone to evaluate
        - temporary root directory for on-disk sample caches
        """
        super().__init__()
        self.config = config
        self.tasks_validation = list(tasks_validation)
        self.task_during = list(task_during)
        self.tasks_end = list(tasks_end)

        self._next_milestone_idx: int = 0
        self._milestones: list[float] = self._build_milestones(config)

        self._epoch_val_batch: Optional[Any] = None
        self._epoch_with_cached_batch: Optional[int] = None

        self._cache_root: Path | None = None
        self._best_metric_by_name: dict[str, float] = {}
        self._best_checkpoint_path_by_name: dict[str, Path] = {}
        self._checkpoint_callback_last: Any | None = None
        self._checkpoint_callback_best: Any | None = None
        self._distributed_mode_logged: bool = False

    @classmethod
    def from_config(
        cls,
        *,
        cfg: Mapping[str, Any] | SchedulerConfig,
        registry: Mapping[str, Callable[..., Mapping[str, Any]]],
    ) -> "BaseSchedulerCallback":
        """Construct callback from typed or raw config plus a function registry.

        The method accepts either a ``SchedulerConfig`` instance or a mapping with
        equivalent fields. Every configured task is resolved through ``registry`` by
        its ``fn_key`` and converted into a ``TaskSpec`` bound to the callable.
        """
        scheduler_cfg = cfg if isinstance(cfg, SchedulerConfig) else _build_scheduler_config(cfg)

        return cls(
            config=scheduler_cfg,
            tasks_validation=cls._build_task_specs(
                scheduler_cfg.tasks_validation,
                registry,
                section_name="tasks_validation",
            ),
            task_during=cls._build_task_specs(
                scheduler_cfg.task_during,
                registry,
                section_name="task_during",
            ),
            tasks_end=cls._build_task_specs(
                scheduler_cfg.tasks_end,
                registry,
                section_name="tasks_end",
            ),
        )

    def attach_experiment_checkpoints(
        self,
        *,
        checkpoint_callback_last: Any | None,
        checkpoint_callback_best: Any | None,
    ) -> None:
        """Attach experiment-managed checkpoint callbacks used by end-task selectors."""
        self._checkpoint_callback_last = checkpoint_callback_last
        self._checkpoint_callback_best = checkpoint_callback_best

    @staticmethod
    def _build_task_specs(
        tasks: Sequence[SchedulerTaskConfig],
        registry: Mapping[str, Callable[..., Mapping[str, Any]]],
        *,
        section_name: str,
    ) -> list[TaskSpec]:
        """Translate task configs into executable specs with resolved callables.

        Raises:
            KeyError: if any ``fn_key`` does not exist in the provided registry.
        """
        specs: list[TaskSpec] = []
        for task_cfg in tasks:
            sample_source = SampleSource(task_cfg.sample_source)
            if sample_source is SampleSource.TASK_INTERNAL and int(task_cfg.n_samples) != 0:
                raise ValueError(
                    "sample_source='task_internal' requires n_samples=0 because the task "
                    "must manage its own sampling."
                )
            if (
                task_cfg.fn_key
                in {
                    "pk.diverse_experiment.distances",
                    "pk.diverse_synthetic_experiment.sample_distances",
                    "pk.synthetic.vpc.paired_images",
                }
                and section_name != "tasks_end"
            ):
                raise ValueError(
                    "Synthetic task-internal analyses like diverse-experiment distances and "
                    "paired synthetic VPC images are supported only in scheduler.tasks_end."
                )
            if task_cfg.fn_key not in registry:
                raise KeyError(
                    f"Unknown fn_key '{task_cfg.fn_key}'. "
                    f"Available keys: {sorted(registry.keys())}."
                )
            specs.append(TaskSpec.from_config(task_cfg, fn=registry[task_cfg.fn_key]))
        return specs

    @staticmethod
    def _build_milestones(config: SchedulerConfig) -> list[float]:
        """Compute percentage milestones that should trigger percent-based tasks.

        Milestones are generated as ``k * percent_step`` until the sequence reaches
        or passes ``1.0``. If ``include_end`` is true, ``1.0`` is always included as
        the final milestone.
        """
        milestones: list[float] = []
        k = 1
        while True:
            candidate = float(k * config.percent_step)
            if candidate < 1.0:
                milestones.append(candidate)
                k += 1
                continue
            if config.include_end:
                milestones.append(1.0)
            elif math.isclose(candidate, 1.0) or candidate <= 1.0:
                milestones.append(candidate)
            break
        return milestones

    def _should_skip(self, trainer: pl.Trainer) -> bool:
        """Return whether hooks should be skipped during Lightning sanity checking.

        This only applies when ``config.skip_sanity_check`` is enabled.
        """
        if not self.config.skip_sanity_check:
            return False
        return bool(getattr(trainer, "sanity_checking", False))

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Handle validation-batch trigger and maintain per-epoch batch cache.

        Behavior:
        - Resets cached validation batch when a new epoch starts.
        - Stores the first encountered validation batch for this epoch so
          end-of-training hooks can reuse it if needed.
        - Executes ``tasks_validation`` immediately, passing the current batch as
          ``current_val_batch`` for source resolution.
        """
        _ = outputs
        _ = dataloader_idx
        if self._should_skip(trainer):
            return

        current_epoch = int(getattr(trainer, "current_epoch", 0))
        if self._epoch_with_cached_batch != current_epoch:
            self._epoch_with_cached_batch = current_epoch
            self._epoch_val_batch = None

        if self._epoch_val_batch is None and batch_idx == 0:
            self._epoch_val_batch = batch
        elif self._epoch_val_batch is None:
            self._epoch_val_batch = batch

        if self.tasks_validation:
            self._run_owner_only(
                trainer=trainer,
                section="validation_batch",
                fn=lambda: self._run_task_list(
                    tasks=self.tasks_validation,
                    trainer=trainer,
                    pl_module=pl_module,
                    current_val_batch=batch,
                    tag="validation_batch",
                ),
            )

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Run percent-based tasks at the end of each training epoch.

        A ``while`` loop is intentionally used so if one epoch jumps across multiple
        milestones, all missed milestones are fired in order.
        """
        if self._should_skip(trainer):
            return
        if not self.task_during:
            return

        while self._should_fire_percent_milestone(trainer):
            milestone_idx = self._next_milestone_idx
            tag = f"percent_{milestone_idx}"
            tasks_for_milestone = self._filter_tasks_for_milestone(
                self.task_during,
                milestone_idx=milestone_idx,
            )
            if not tasks_for_milestone:
                continue
            self._run_owner_only(
                trainer=trainer,
                section=tag,
                fn=lambda tag=tag, tasks_for_milestone=tasks_for_milestone: self._run_task_list(
                    tasks=tasks_for_milestone,
                    trainer=trainer,
                    pl_module=pl_module,
                    current_val_batch=None,
                    tag=tag,
                ),
            )

    @staticmethod
    def _filter_tasks_for_milestone(
        tasks: Sequence[TaskSpec],
        *,
        milestone_idx: int,
    ) -> list[TaskSpec]:
        """Return tasks enabled for a given percent milestone.

        ``milestone_stride`` is read from ``task.task_cfg`` when present. A
        stride of ``N`` means "run every N-th milestone", using one-based
        milestone indexing.
        """

        filtered: list[TaskSpec] = []
        for task in tasks:
            task_cfg = dict(task.task_cfg or {})
            raw_stride = task_cfg.get("milestone_stride", 1)
            try:
                stride = max(1, int(raw_stride))
            except (TypeError, ValueError):
                stride = 1
            if milestone_idx % stride != 0:
                continue
            filtered.append(task)
        return filtered

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Execute end-of-training tasks and perform final temporary-cache cleanup."""
        if self.tasks_end:
            self._run_owner_only(
                trainer=trainer,
                section="train_end",
                fn=lambda: self._run_end_tasks(trainer=trainer, pl_module=pl_module),
            )

        if not self.config.keep_temp_files:
            self._cleanup_cache_root()

    def _run_end_tasks(self, *, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        for checkpoint_label in self.config.checkpoint_used_in_end:
            self._log_block(
                "Scheduler End Tasks",
                selector=checkpoint_label,
                task_count=len(self.tasks_end),
            )
            if not self.load_checkpoint(
                trainer=trainer,
                pl_module=pl_module,
                checkpoint_label=checkpoint_label,
            ):
                continue
            self._run_task_list(
                tasks=self.tasks_end,
                trainer=trainer,
                pl_module=pl_module,
                current_val_batch=self._epoch_val_batch,
                tag=f"train_end__{checkpoint_label}",
                checkpoint_label=checkpoint_label,
            )

    def _run_owner_only(
        self,
        *,
        trainer: pl.Trainer,
        section: str,
        fn: Callable[[], None],
    ) -> None:
        if not self._is_distributed_scheduler_run(trainer):
            fn()
            return

        self._log_distributed_mode(trainer, section=section)
        self._distributed_barrier(trainer, section=section, when="before")

        owner_error: Exception | None = None
        try:
            if self._is_scheduler_owner(trainer):
                fn()
        except Exception as exc:  # pragma: no cover - exercised via distributed runtime
            owner_error = exc
        finally:
            self._distributed_barrier(trainer, section=section, when="after")

        if owner_error is not None:
            raise owner_error

    def _is_distributed_scheduler_run(self, trainer: pl.Trainer) -> bool:
        world_size = getattr(trainer, "world_size", None)
        if world_size is not None:
            try:
                return int(world_size) > 1
            except (TypeError, ValueError):
                pass

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size() > 1
        return False

    @staticmethod
    def _is_scheduler_owner(trainer: pl.Trainer) -> bool:
        return bool(getattr(trainer, "is_global_zero", True))

    def _log_distributed_mode(self, trainer: pl.Trainer, *, section: str) -> None:
        if self._distributed_mode_logged:
            return
        self._distributed_mode_logged = True
        LOGGER.info(
            "Scheduler distributed owner-only mode active: rank 0 executes scheduler tasks while "
            "other ranks synchronize (section=%s, global_rank=%s).",
            section,
            getattr(trainer, "global_rank", 0),
        )

    @staticmethod
    def _distributed_barrier(
        trainer: pl.Trainer,
        *,
        section: str,
        when: str,
    ) -> None:
        strategy = getattr(trainer, "strategy", None)
        barrier = getattr(strategy, "barrier", None) if strategy is not None else None
        if callable(barrier):
            try:
                barrier(f"scheduler:{section}:{when}")
            except TypeError:
                barrier()
            return

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def load_checkpoint(
        self,
        *,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint_label: str,
    ) -> bool:
        """Load one end-task checkpoint selector into the active Lightning module."""
        if self._is_distributed_scheduler_run(trainer) and not self._is_scheduler_owner(trainer):
            return False
        if checkpoint_label == "end":
            self._log_block(
                "Scheduler Checkpoint",
                selector=checkpoint_label,
                source="in-memory train-end weights",
            )
            return True

        checkpoint_path = self._resolve_checkpoint_path(checkpoint_label)
        if checkpoint_path is None:
            LOGGER.warning(
                "Skipping scheduler end-task checkpoint '%s' because no checkpoint file is available.",
                checkpoint_label,
            )
            return False

        self._log_block(
            "Scheduler Checkpoint",
            selector=checkpoint_label,
            source=str(checkpoint_path),
        )
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception:
            LOGGER.exception(
                "Failed to load scheduler end-task checkpoint '%s' from '%s'.",
                checkpoint_label,
                checkpoint_path,
            )
            return False

        state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
        if state_dict is None:
            LOGGER.warning(
                "Skipping scheduler end-task checkpoint '%s' because '%s' has no state_dict.",
                checkpoint_label,
                checkpoint_path,
            )
            return False

        try:
            pl_module.load_state_dict(state_dict, strict=True)
        except Exception:
            LOGGER.exception(
                "Failed to apply scheduler end-task checkpoint '%s' from '%s'.",
                checkpoint_label,
                checkpoint_path,
            )
            return False
        return True

    def _resolve_checkpoint_path(self, checkpoint_label: str) -> Path | None:
        if checkpoint_label == "best":
            return self._resolve_experiment_checkpoint_path(
                callback=self._checkpoint_callback_best,
                attribute_name="best_model_path",
            )
        if checkpoint_label == "last":
            return self._resolve_experiment_checkpoint_path(
                callback=self._checkpoint_callback_last,
                attribute_name="last_model_path",
            )
        return self._best_checkpoint_path_by_name.get(checkpoint_label)

    @staticmethod
    def _resolve_experiment_checkpoint_path(
        *,
        callback: Any | None,
        attribute_name: str,
    ) -> Path | None:
        if callback is None:
            return None
        raw_path = getattr(callback, attribute_name, None)
        if not raw_path:
            return None
        path = Path(str(raw_path))
        return path if path.exists() else None

    def _should_fire_percent_milestone(self, trainer: pl.Trainer) -> bool:
        """Check whether the next progress milestone has been reached.

        Progress is defined as ``(current_epoch + 1) / max_epochs`` to match
        end-of-epoch semantics. When reached, this method increments the internal
        milestone pointer and returns ``True`` exactly once per milestone.
        """
        if self._next_milestone_idx >= len(self._milestones):
            return False

        max_epochs = int(getattr(trainer, "max_epochs", 0) or 0)
        if max_epochs <= 0:
            return False

        progress = float(int(getattr(trainer, "current_epoch", 0)) + 1) / float(max_epochs)
        milestone = self._milestones[self._next_milestone_idx]
        if progress + 1e-12 >= milestone:
            self._next_milestone_idx += 1
            return True
        return False

    def _run_task_list(
        self,
        *,
        tasks: Sequence[TaskSpec],
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        current_val_batch: Optional[Any],
        tag: str,
        checkpoint_label: str | None = None,
    ) -> None:
        """Run tasks through a grouped generate-cache-slice-execute pipeline.

        Pipeline details:
        1. Group tasks by sample source parameters so generation is reused.
        2. For each group, clamp to ``max_samples_per_group``, resolve batches, and
           generate a single maximal sample payload.
        3. Cache generated payload on disk and reload lazily per task.
        4. Slice payload to each task's ``n_samples``, execute task callable, and
           log/save returned artifacts.

        Temporary cache files are removed in ``finally`` unless
        ``config.keep_temp_files`` is enabled.
        """
        if not tasks:
            return

        grouped: dict[tuple[SampleSource, str, str | None], list[TaskSpec]] = defaultdict(list)
        for task in tasks:
            key = (task.sample_source, task.split, task.empirical_name)
            grouped[key].append(task)

        group_cache: dict[tuple[SampleSource, str, str | None], _GroupCacheRecord] = {}
        created_paths: list[Path] = []
        for group_key, grouped_tasks in grouped.items():
            sample_source, split, empirical_name = group_key
            task_names = [task.name for task in grouped_tasks]

            max_requested = max(task.n_samples for task in grouped_tasks)
            n_samples = min(max_requested, self.config.max_samples_per_group)
            if n_samples < max_requested:
                LOGGER.warning(
                    "Scheduler clamped n_samples for group %s from %d to %d (max_samples_per_group=%d)",
                    group_key,
                    max_requested,
                    n_samples,
                    self.config.max_samples_per_group,
                )
            self._log_block(
                "Scheduler Task Group",
                tag=tag,
                checkpoint=checkpoint_label or "none",
                source=sample_source.value,
                split=split,
                empirical=empirical_name or "-",
                tasks=", ".join(task_names),
                requested_samples=max_requested,
                generated_samples=n_samples,
            )

            if sample_source is SampleSource.TASK_INTERNAL:
                group_cache[group_key] = _GroupCacheRecord(
                    cache_path=None,
                    batches=[],
                )
                continue

            batches = self._resolve_batches(
                trainer=trainer,
                sample_source=sample_source,
                split=split,
                empirical_name=empirical_name,
                current_val_batch=current_val_batch,
            )

            samples_max = None
            stored_path: Path | None = None
            if n_samples > 0:
                samples_max = self._generate_once(
                    pl_module=pl_module,
                    sample_source=sample_source,
                    n_samples=n_samples,
                    batches=batches,
                )
                stored_path = self._store_group_samples(
                    trainer=trainer,
                    samples_max=samples_max,
                    sample_source=sample_source,
                    split=split,
                    empirical_name=empirical_name,
                    tag=tag,
                )

            if stored_path is not None:
                cache_path = stored_path
            else:
                cache_path = self._cache_samples(samples_max, trainer=trainer)
                if cache_path is not None:
                    created_paths.append(cache_path)

            group_cache[group_key] = _GroupCacheRecord(
                cache_path=cache_path,
                batches=batches,
            )

        loaded_cache: dict[tuple[SampleSource, str, str | None], Any] = {}
        try:
            for task in tasks:
                group_key = (task.sample_source, task.split, task.empirical_name)
                group_record = group_cache[group_key]

                samples = None
                if task.n_samples > 0:
                    if group_key not in loaded_cache:
                        loaded_cache[group_key] = self._load_cached(group_record.cache_path)
                    samples = self._slice_samples(loaded_cache[group_key], task.n_samples)

                task_cfg = dict(task.task_cfg or {})
                task_cfg.setdefault("tag", tag)
                if checkpoint_label is not None:
                    task_cfg["checkpoint_label"] = checkpoint_label
                    model_label = task_cfg.get("model_label")
                    if model_label:
                        task_cfg["model_label"] = f"{model_label}__{checkpoint_label}"
                self._log_block(
                    "Scheduler Task",
                    name=task.name,
                    fn=self._task_fn_name(task.fn),
                    tag=tag,
                    checkpoint=checkpoint_label or "none",
                    n_samples=task.n_samples,
                    source=task.sample_source.value,
                    split=task.split,
                )
                out = task.fn(
                    samples=samples,
                    batches=group_record.batches,
                    task_cfg=task_cfg,
                    trainer=trainer,
                    pl_module=pl_module,
                )
                self._log_and_save(
                    trainer=trainer,
                    pl_module=pl_module,
                    task=task,
                    out=out,
                    task_name=self._task_name_for_run(task=task, checkpoint_label=checkpoint_label),
                )
        finally:
            if not self.config.keep_temp_files:
                for path in created_paths:
                    try:
                        path.unlink(missing_ok=True)
                    except Exception:
                        LOGGER.exception("Failed to remove scheduler temp cache file '%s'.", path)

    def _resolve_batches(
        self,
        *,
        trainer: pl.Trainer,
        sample_source: SampleSource,
        split: str,
        empirical_name: str | None,
        current_val_batch: Any | None,
    ) -> Any:
        """Resolve batch container used as generation context for a task group.

        Source mapping:
        - ``UNCONDITIONAL``: uses current validation batch, or first split batch as fallback.
          Nested permutation containers are unwrapped to a single representative batch.
        - ``VAL_BATCH``: reuses the current validation batch, or first split batch as fallback.
          Nested permutation containers are flattened into a list of batch objects.
        - ``FULL_SPLIT``: materializes all batches from ``{split}_dataloader`` and flattens
          nested permutation containers.
        - ``EMPIRICAL_SET``: materializes batches from named empirical set and flattens
          nested permutation containers.
        - ``TASK_INTERNAL``: provides no scheduler batch context.
        """
        if sample_source is SampleSource.TASK_INTERNAL:
            return []

        if sample_source is SampleSource.UNCONDITIONAL:
            if current_val_batch is None:
                current_val_batch = self._get_first_split_batch(trainer=trainer, split=split)
            return self._select_first_batch(current_val_batch, source=sample_source.value)

        if sample_source is SampleSource.VAL_BATCH:
            if current_val_batch is None:
                current_val_batch = self._get_first_split_batch(trainer=trainer, split=split)
            return self._flatten_batch_collection(current_val_batch)

        if sample_source is SampleSource.FULL_SPLIT:
            return self._flatten_batch_collection(
                list(self._iter_full_split(trainer=trainer, split=split))
            )

        if sample_source is SampleSource.EMPIRICAL_SET:
            empirical_names = (
                [str(empirical_name).strip()]
                if empirical_name and str(empirical_name).strip()
                else self._resolve_empirical_dataset_names(trainer=trainer)
            )
            flattened_batches: list[Any] = []
            for repo_id in empirical_names:
                flattened_batches.extend(
                    self._flatten_batch_collection(
                        list(
                            self._get_empirical_batches(
                                trainer=trainer,
                                split=split,
                                empirical_name=repo_id,
                            )
                        )
                    )
                )
            return flattened_batches

        raise ValueError(f"Unsupported sample_source='{sample_source.value}'.")

    @staticmethod
    def _resolve_empirical_dataset_names(*, trainer: pl.Trainer) -> list[str]:
        """Resolve configured empirical repos for cross-repo scheduler tasks."""

        candidate_sources = [
            getattr(
                getattr(getattr(trainer, "lightning_module", None), "model_config", None),
                "mix_data",
                None,
            ),
            getattr(getattr(trainer, "datamodule", None), "data_config", None),
        ]
        resolved: list[str] = []
        seen: set[str] = set()
        for source in candidate_sources:
            for raw_repo_id in list(getattr(source, "test_empirical_datasets", []) or []):
                repo_id = str(raw_repo_id).strip()
                if not repo_id or repo_id in seen:
                    continue
                seen.add(repo_id)
                resolved.append(repo_id)

        if not resolved:
            raise ValueError(
                "sample_source='empirical_set' requires either task.empirical_name or at least "
                "one configured repo in mix_data.test_empirical_datasets."
            )
        return resolved

    @staticmethod
    def _resolve_empirical_eval_fix_past_value(
        *,
        trainer: pl.Trainer,
        datamodule: Any,
        split: str,
    ) -> int | None:
        """Resolve fixed-past evaluation count for empirical held-out batches."""

        if str(split).strip().lower() != "empirical_heldout":
            return None
        if not callable(getattr(datamodule, "fix_past_selection", None)):
            return None

        mix_cfg = getattr(
            getattr(getattr(trainer, "lightning_module", None), "model_config", None),
            "mix_data",
            None,
        )
        raw_value = getattr(mix_cfg, "evaluate_prediction_steps_past", None)
        if raw_value is None:
            raw_value = getattr(getattr(datamodule, "data_config", None), "evaluate_prediction_steps_past", None)
        if raw_value is None:
            return None
        return int(raw_value)

    @classmethod
    def _flatten_batch_collection(cls, batches: Any) -> list[Any]:
        """Flatten nested batch containers while preserving atomic batch objects.

        The PK dataloaders can yield either a single databatch or a list of
        permutation-specific databatches. This helper recursively flattens plain
        ``list``/``tuple`` containers but treats NamedTuple-like batch objects
        (for example ``AICMECompartmentsDataBatch``) as atomic.
        """
        if batches is None:
            return []

        if isinstance(batches, (list, tuple)) and not hasattr(batches, "_fields"):
            flattened: list[Any] = []
            for batch in batches:
                flattened.extend(cls._flatten_batch_collection(batch))
            return flattened

        return [batches]

    @classmethod
    def _select_first_batch(cls, batches: Any, *, source: str) -> Any:
        """Return the first atomic batch from a possibly nested batch container."""
        flattened = cls._flatten_batch_collection(batches)
        if not flattened:
            raise ValueError(f"Scheduler source '{source}' did not provide any batches.")
        return flattened[0]

    def _get_first_split_batch(self, *, trainer: pl.Trainer, split: str) -> Any:
        """Return first batch from a split dataloader for epoch-end task execution."""
        dataloader = self._iter_full_split(trainer=trainer, split=split)
        iterator = iter(dataloader)
        try:
            return next(iterator)
        except StopIteration as exc:
            raise ValueError(
                f"Datamodule '{split}_dataloader()' returned no batches; cannot run scheduler task."
            ) from exc

    def _iter_full_split(self, *, trainer: pl.Trainer, split: str):
        """Return dataloader iterable for a full split from the active datamodule.

        If datamodule exposes ``setup``, it is invoked with stage ``fit`` for
        ``train``/``val`` splits and with ``split`` for other split names.
        """
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            raise RuntimeError("Scheduler requires trainer.datamodule for FULL_SPLIT tasks.")

        if hasattr(datamodule, "setup"):
            stage = "fit" if split in {"train", "val"} else split
            datamodule.setup(stage)

        loader_name = f"{split}_dataloader"
        loader_fn = getattr(datamodule, loader_name, None)
        if not callable(loader_fn):
            raise ValueError(
                f"Datamodule does not provide '{loader_name}()' required for FULL_SPLIT tasks."
            )
        return loader_fn()

    def _get_empirical_batches(self, *, trainer: pl.Trainer, split: str, empirical_name: str):
        """Return empirical batches for ``(split, empirical_name)`` from datamodule.

        Preferred interface is ``datamodule.get_empirical_batches(...)``; fallback
        accesses ``datamodule.empirical_batches[split][empirical_name]``.
        """
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            raise RuntimeError("Scheduler requires trainer.datamodule for EMPIRICAL_SET tasks.")

        getter = getattr(datamodule, "get_empirical_batches", None)
        if callable(getter):
            fix_past_value = self._resolve_empirical_eval_fix_past_value(
                trainer=trainer,
                datamodule=datamodule,
                split=split,
            )
            if fix_past_value is None:
                return getter(split=split, empirical_name=empirical_name)

            datamodule.fix_past_selection(fix_past_value, who="target")
            try:
                return getter(split=split, empirical_name=empirical_name)
            finally:
                releaser = getattr(datamodule, "release_past_selection", None)
                if callable(releaser):
                    releaser(who="target")

        empirical_batches = getattr(datamodule, "empirical_batches", None)
        if empirical_batches is None:
            raise ValueError(
                "Datamodule missing empirical batch interface. Provide get_empirical_batches(...) "
                "or datamodule.empirical_batches[split][empirical_name]."
            )

        try:
            return empirical_batches[split][empirical_name]
        except Exception as exc:
            raise ValueError(
                f"Could not resolve empirical batches for split='{split}', empirical_name='{empirical_name}'."
            ) from exc

    def _generate_once(
        self,
        *,
        pl_module: pl.LightningModule,
        sample_source: SampleSource,
        n_samples: int,
        batches: Any,
    ) -> Any:
        """Generate one maximal sample payload for a grouped set of tasks.

        Dispatch rules:
        - ``UNCONDITIONAL`` delegates to ``model.generate_unconditional`` only when
          that method exists.
        - ``VAL_BATCH`` calls ``model.generate`` for every resolved validation batch.
        - ``FULL_SPLIT`` / ``EMPIRICAL_SET`` call ``model.generate`` per batch and
          return a list aligned to input batch order.
        """
        model = getattr(pl_module, "model", pl_module)

        if sample_source is SampleSource.UNCONDITIONAL:
            generate_unconditional = getattr(model, "generate_unconditional", None)
            if not callable(generate_unconditional):
                raise NotImplementedError(
                    "Scheduler sample_source='unconditional' requires model.generate_unconditional(...). "
                    "The current PolyFold v1 integration only guarantees conditional sampling via model.generate(...)."
                )
            self._log_block(
                "Scheduler Generation",
                source="unconditional",
                samples=n_samples,
            )
            batch = self._to_device_if_possible(batches, pl_module.device)
            return generate_unconditional(batch, num_samples=n_samples)

        if sample_source is SampleSource.TASK_INTERNAL:
            raise ValueError(
                "Scheduler sample_source='task_internal' does not support scheduler-managed "
                "generation."
            )

        if sample_source in {
            SampleSource.VAL_BATCH,
            SampleSource.FULL_SPLIT,
            SampleSource.EMPIRICAL_SET,
        }:
            generated: list[Any] = []
            self._log_block(
                "Scheduler Generation",
                source=sample_source.value,
                samples=n_samples,
                batches=len(batches),
            )
            for batch in batches:
                batch_device = self._to_device_if_possible(batch, pl_module.device)
                generated.append(model.generate(batch_device, num_samples=n_samples))
            return generated

        raise ValueError(f"Unsupported sample source '{sample_source.value}'.")

    @staticmethod
    def _to_device_if_possible(batch: Any, device: Any) -> Any:
        """Move batch to target device when object provides a device-transfer method."""
        to_device = getattr(batch, "to_device", None)
        if callable(to_device):
            return to_device(device)

        if hasattr(batch, "to"):
            return batch.to(device)
        return batch

    def _slice_samples(self, samples_max: Any, n_samples: int) -> Any:
        """Slice generated payload down to task-specific ``n_samples``.

        Supports nested list payloads, scheduler-aware composite payloads
        exposing ``scheduler_slice(...)``, and tensors.
        """
        if samples_max is None or n_samples <= 0:
            return None

        if isinstance(samples_max, list):
            return [self._slice_samples(item, n_samples) for item in samples_max]

        scheduler_slice = getattr(samples_max, "scheduler_slice", None)
        if callable(scheduler_slice):
            return scheduler_slice(int(n_samples))

        if isinstance(samples_max, torch.Tensor):
            return samples_max[:n_samples]

        return samples_max

    def _cache_samples(self, samples_max: Any, *, trainer: pl.Trainer) -> Path | None:
        """Serialize generated payload to a temporary ``.pt`` file and return path.

        Returns ``None`` when there is no payload to cache.
        """
        if samples_max is None:
            return None
        cache_root = self._ensure_cache_root(trainer)
        fd, raw_path = tempfile.mkstemp(
            prefix="scheduler_group_", suffix=".pt", dir=str(cache_root)
        )
        import os

        os.close(fd)
        torch.save(samples_max, raw_path)
        return Path(raw_path)

    def _store_group_samples(
        self,
        *,
        trainer: pl.Trainer,
        samples_max: Any,
        sample_source: SampleSource,
        split: str,
        empirical_name: str | None,
        tag: str,
    ) -> Path | None:
        """Persist grouped samples by protocol when ``config.store_samples`` is enabled."""
        if not self.config.store_samples or samples_max is None:
            return None

        root_dir = getattr(trainer, "default_root_dir", None)
        root = Path(root_dir) if root_dir is not None else Path(".")
        protocol_dir = self._protocol_dir_name(
            sample_source=sample_source,
            split=split,
            empirical_name=empirical_name,
        )
        output_dir = root / "scheduler_samples" / protocol_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        epoch = int(getattr(trainer, "current_epoch", 0))
        step = int(getattr(trainer, "global_step", 0))
        safe_tag = self._sanitize_path_component(tag)
        output_path = output_dir / f"epoch_{epoch:03d}_step_{step:07d}_{safe_tag}.pt"
        torch.save(samples_max, output_path)
        return output_path

    def _protocol_dir_name(
        self,
        *,
        sample_source: SampleSource,
        split: str,
        empirical_name: str | None,
    ) -> str:
        """Build folder name for a generated sample protocol."""
        components = [sample_source.value, f"split_{split}"]
        if empirical_name:
            components.append(f"empirical_{empirical_name}")
        return "__".join(self._sanitize_path_component(component) for component in components)

    @staticmethod
    def _sanitize_path_component(value: str) -> str:
        """Normalize arbitrary names for filesystem-safe path components."""
        normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
        return normalized.strip("_") or "unknown"

    @staticmethod
    def _load_cached(cache_path: Path | None) -> Any:
        """Load cached payload from disk with CPU map-location safety."""
        if cache_path is None:
            return None
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    def _ensure_cache_root(self, trainer: pl.Trainer) -> Path:
        """Create and memoize directory used to store per-group cache artifacts.

        Priority for base directory:
        1. ``config.cache_dir``
        2. ``trainer.default_root_dir / "scheduler_cache"``
        3. system temp directory
        """
        if self._cache_root is not None:
            return self._cache_root

        configured_root = self.config.cache_dir
        if configured_root is None:
            default_root = getattr(trainer, "default_root_dir", None)
            if default_root is not None:
                configured_root = str(Path(default_root) / "scheduler_cache")
            else:
                configured_root = tempfile.gettempdir()

        base_path = Path(configured_root)
        base_path.mkdir(parents=True, exist_ok=True)
        self._cache_root = Path(tempfile.mkdtemp(prefix="tp_scheduler_", dir=str(base_path)))
        return self._cache_root

    def _cleanup_cache_root(self) -> None:
        """Delete cache root tree and clear internal pointer to that directory."""
        if self._cache_root is None:
            return
        try:
            shutil.rmtree(self._cache_root, ignore_errors=True)
        finally:
            self._cache_root = None

    def _log_and_save(
        self,
        *,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        task: TaskSpec,
        out: Mapping[str, Any] | None,
        task_name: str | None = None,
    ) -> None:
        """Normalize task outputs into logger-compatible metric/image events.

        Supported output value types:
        - scalar tensors and numeric scalars -> metric logging
        - filesystem ``Path`` -> image logging for image suffixes, otherwise asset logging
        - strings -> text logging when the active experiment supports it
        - Matplotlib ``Figure`` -> optional disk save + figure/image logging

        Metric names are namespaced as ``{log_prefix}/{task_name}/{key}``.
        """
        _ = pl_module
        if not out:
            return

        resolved_task_name = task_name or task.name
        for key, value in out.items():
            metric_name = f"{task.log_prefix}/{resolved_task_name}/{key}"
            if isinstance(value, torch.Tensor) and value.ndim == 0:
                scalar_value = float(value.item())
                self._log_metric(trainer=trainer, name=metric_name, value=scalar_value)
                self._maybe_update_callback_metric(
                    trainer=trainer,
                    task=task,
                    key=key,
                    metric_name=metric_name,
                    value=scalar_value,
                )
                self._maybe_save_task_checkpoint(
                    trainer=trainer,
                    pl_module=pl_module,
                    task=task,
                    key=key,
                    metric_name=metric_name,
                    value=scalar_value,
                )
                continue

            if isinstance(value, (float, int)):
                scalar_value = float(value)
                self._log_metric(trainer=trainer, name=metric_name, value=scalar_value)
                self._maybe_update_callback_metric(
                    trainer=trainer,
                    task=task,
                    key=key,
                    metric_name=metric_name,
                    value=scalar_value,
                )
                self._maybe_save_task_checkpoint(
                    trainer=trainer,
                    pl_module=pl_module,
                    task=task,
                    key=key,
                    metric_name=metric_name,
                    value=scalar_value,
                )
                continue

            if isinstance(value, Path):
                if value.exists():
                    if value.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                        self._log_image(trainer=trainer, image_path=value, name=metric_name)
                    else:
                        self._log_asset(trainer=trainer, path=value, name=metric_name)
                continue

            if isinstance(value, str):
                self._log_text(trainer=trainer, name=metric_name, text=value)
                continue

            if self._is_matplotlib_figure(value):
                figure_path = None
                if task.save_to_disk:
                    figure_path = self._save_figure(
                        trainer=trainer,
                        task_name=resolved_task_name,
                        key=key,
                        fig=value,
                    )
                self._log_figure(
                    trainer=trainer,
                    fig=value,
                    name=metric_name,
                    fallback_image_path=figure_path,
                )

    @staticmethod
    def _is_matplotlib_figure(value: Any) -> bool:
        """Return whether value is a Matplotlib ``Figure`` without hard dependency."""
        try:
            from matplotlib.figure import Figure

            return isinstance(value, Figure)
        except Exception:
            return False

    @staticmethod
    def _iter_loggers(trainer: pl.Trainer):
        """Return non-null logger list across Lightning single/multi logger APIs."""
        loggers = getattr(trainer, "loggers", None)
        if loggers is None:
            logger = getattr(trainer, "logger", None)
            return [logger] if logger is not None else []
        return [logger for logger in loggers if logger is not None]

    def _log_metric(self, *, trainer: pl.Trainer, name: str, value: float) -> None:
        """Log a scalar metric to each experiment exposing ``log_metric``."""
        step = int(getattr(trainer, "global_step", 0))
        for logger in self._iter_loggers(trainer):
            experiment = getattr(logger, "experiment", None)
            if experiment is None or not hasattr(experiment, "log_metric"):
                continue
            experiment.log_metric(name=name, value=value, step=step)

    def _log_text(self, *, trainer: pl.Trainer, name: str, text: str) -> None:
        """Log text payloads to experiments exposing a compatible text API."""
        step = int(getattr(trainer, "global_step", 0))
        for logger in self._iter_loggers(trainer):
            experiment = getattr(logger, "experiment", None)
            if experiment is None:
                continue

            log_text = getattr(experiment, "log_text", None)
            if callable(log_text):
                try:
                    log_text(text, metadata={"name": name}, step=step)
                except TypeError:
                    try:
                        log_text(text, step=step, metadata={"name": name})
                    except TypeError:
                        try:
                            log_text(text)
                        except Exception:
                            LOGGER.debug(
                                "Experiment text logger rejected payload '%s'.", name, exc_info=True
                            )
                continue

            log_asset_data = getattr(experiment, "log_asset_data", None)
            if callable(log_asset_data):
                try:
                    log_asset_data(
                        text, name=f"{self._sanitize_path_component(name)}.txt", step=step
                    )
                except TypeError:
                    try:
                        log_asset_data(text, name=f"{self._sanitize_path_component(name)}.txt")
                    except Exception:
                        LOGGER.debug(
                            "Experiment asset-data logger rejected payload '%s'.",
                            name,
                            exc_info=True,
                        )

    def _log_asset(self, *, trainer: pl.Trainer, path: Path, name: str) -> None:
        """Log non-image artifacts to experiments exposing an asset API."""
        step = int(getattr(trainer, "global_step", 0))
        for logger in self._iter_loggers(trainer):
            experiment = getattr(logger, "experiment", None)
            if experiment is None:
                continue

            log_asset = getattr(experiment, "log_asset", None)
            if not callable(log_asset):
                continue

            try:
                log_asset(str(path), name=self._sanitize_path_component(name), step=step)
            except TypeError:
                try:
                    log_asset(str(path), step=step)
                except TypeError:
                    try:
                        log_asset(str(path))
                    except Exception:
                        LOGGER.debug(
                            "Experiment asset logger rejected payload '%s'.", name, exc_info=True
                        )

    @staticmethod
    def _maybe_update_callback_metric(
        *,
        trainer: pl.Trainer,
        task: TaskSpec,
        key: str,
        metric_name: str,
        value: float,
    ) -> None:
        """Expose configured scheduler metric in ``trainer.callback_metrics`` for checkpointing."""
        if not task.checkpoint_metric:
            return

        configured_name = str(task.checkpoint_metric_name or "").strip()
        if not configured_name:
            return

        if configured_name not in {key, metric_name}:
            return

        callback_metrics = getattr(trainer, "callback_metrics", None)
        if callback_metrics is None:
            callback_metrics = {}
            setattr(trainer, "callback_metrics", callback_metrics)
        callback_metrics[metric_name] = torch.tensor(float(value))

    def _maybe_save_task_checkpoint(
        self,
        *,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        task: TaskSpec,
        key: str,
        metric_name: str,
        value: float,
    ) -> None:
        """Save model checkpoint when configured task metric improves."""
        if self._is_distributed_scheduler_run(trainer) and not self._is_scheduler_owner(trainer):
            return
        if not task.checkpoint_metric:
            return

        configured_name = str(task.checkpoint_metric_name or "").strip()
        if not configured_name:
            return

        # Accept either bare output key (recommended) or full metric path.
        if configured_name not in {key, metric_name}:
            return

        mode = str(task.checkpoint_mode).strip().lower()
        if mode not in {"min", "max"}:
            return

        best = self._best_metric_by_name.get(configured_name)
        improved = best is None or (
            (value < best - 1e-12) if mode == "min" else (value > best + 1e-12)
        )
        if not improved:
            return

        self._best_metric_by_name[configured_name] = float(value)
        checkpoint_path = self._save_task_checkpoint(
            trainer=trainer,
            pl_module=pl_module,
            task=task,
            metric_name=configured_name,
            value=float(value),
        )
        if checkpoint_path is None:
            return

        previous_path = self._best_checkpoint_path_by_name.get(configured_name)
        self._best_checkpoint_path_by_name[configured_name] = checkpoint_path
        if previous_path is not None and previous_path != checkpoint_path:
            try:
                previous_path.unlink(missing_ok=True)
            except Exception:
                LOGGER.exception(
                    "Failed to remove previous scheduler checkpoint '%s'.", previous_path
                )

    def _save_task_checkpoint(
        self,
        *,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        task: TaskSpec,
        metric_name: str,
        value: float,
    ) -> Path | None:
        """Persist a model checkpoint for a task-monitored metric.

        This intentionally bypasses ``trainer.save_checkpoint(...)`` because the
        Lightning strategy implementation may treat that path as collective under
        DDP, which can deadlock when scheduler checkpointing is owner-rank only.
        """
        if not bool(getattr(trainer, "is_global_zero", True)):
            return None

        root_dir = getattr(trainer, "default_root_dir", None)
        root = Path(root_dir) if root_dir is not None else Path(".")
        output_dir = root / "scheduler_metric_checkpoints" / task.name.replace("/", "_")
        output_dir.mkdir(parents=True, exist_ok=True)

        epoch = int(getattr(trainer, "current_epoch", 0))
        step = int(getattr(trainer, "global_step", 0))
        safe_metric = self._sanitize_path_component(metric_name)
        output_path = output_dir / (
            f"best-epoch_{epoch:03d}-step_{step:07d}-{safe_metric}={value:.6f}.ckpt"
        )

        self._log_block(
            "Scheduler Metric Checkpoint",
            task=task.name,
            metric=metric_name,
            value=f"{value:.6f}",
            path=str(output_path),
        )
        state_dict = {
            key: tensor.detach().cpu() if isinstance(tensor, torch.Tensor) else tensor
            for key, tensor in pl_module.state_dict().items()
        }
        torch.save({"state_dict": state_dict}, output_path)
        return output_path

    @staticmethod
    def _task_name_for_run(*, task: TaskSpec, checkpoint_label: str | None) -> str:
        if checkpoint_label is None:
            return task.name
        return f"{task.name}__{checkpoint_label}"

    @staticmethod
    def _task_fn_name(fn: TaskFn) -> str:
        return getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn)))

    def _log_block(self, title: str, **fields: Any) -> None:
        lines = [f"\n{title}"]
        for key, value in fields.items():
            lines.append(f"  {key}: {value}")
        LOGGER.info("\n".join(lines))

    def _log_image(self, *, trainer: pl.Trainer, image_path: Path, name: str) -> None:
        """Log image path to each experiment exposing ``log_image``."""
        step = int(getattr(trainer, "global_step", 0))
        for logger in self._iter_loggers(trainer):
            experiment = getattr(logger, "experiment", None)
            if experiment is None:
                continue
            if hasattr(experiment, "log_image"):
                experiment.log_image(str(image_path), name=name, step=step)

    def _save_figure(self, *, trainer: pl.Trainer, task_name: str, key: str, fig: Any) -> Path:
        """Persist figure to ``training_images/<task_name>/epoch_<N>_<key>.png``."""
        root_dir = getattr(trainer, "default_root_dir", None)
        root = Path(root_dir) if root_dir is not None else Path("training_images")
        output_dir = root / "training_images" / task_name.replace("/", "_")
        output_dir.mkdir(parents=True, exist_ok=True)
        epoch = int(getattr(trainer, "current_epoch", 0))
        image_path = output_dir / f"epoch_{epoch:03d}_{key}.png"
        fig.savefig(image_path, dpi=150, bbox_inches="tight")
        return image_path

    def _log_figure(
        self,
        *,
        trainer: pl.Trainer,
        fig: Any,
        name: str,
        fallback_image_path: Path | None,
    ) -> None:
        """Log figure via ``add_figure`` or fallback to file-based image logging."""
        step = int(getattr(trainer, "global_step", 0))
        for logger in self._iter_loggers(trainer):
            experiment = getattr(logger, "experiment", None)
            if experiment is None:
                continue
            if hasattr(experiment, "add_figure"):
                experiment.add_figure(name, fig, global_step=step)
            elif fallback_image_path is not None and hasattr(experiment, "log_image"):
                experiment.log_image(str(fallback_image_path), name=name, step=step)
