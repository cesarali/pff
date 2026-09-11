# CALLBACKS

## What this file is about

This project can attach a scheduler-driven Lightning callback that runs extra PK evaluation and plotting tasks during training.

In the training YAML, this is configured under:

```yaml
train:
  callbacks_scheduler:
    ...
```

The callback implementation lives in [pff/training/callbacks/scheduler.py](/home/cesarali/Pharma/pff/pff/training/callbacks/scheduler.py), and the PK task functions live in [pff/training/callbacks/pk_tasks.py](/home/cesarali/Pharma/pff/pff/training/callbacks/pk_tasks.py).

## In simple terms: what the scheduler does

The scheduler is an automatic "run extra checks while training" system.

It can:

- run metric computations,
- create prediction or generation plots,
- summarize empirical performance,
- run those tasks at validation time, at selected training progress milestones, or once at the very end.

Instead of generating samples separately for every task, it tries to be efficient:

- tasks using the same source of data are grouped together,
- samples are generated once for that group,
- the generated results are cached,
- each task then reuses the cached results.

This avoids repeating expensive model sampling work.

## Where the scheduler is called

The callback is wired in three steps:

1. The training config is loaded from YAML, including `train.callbacks_scheduler`.
2. The model builds the callback with `BaseSchedulerCallback.from_config(...)`.
3. The experiment runner attaches that callback to Lightning together with checkpoint callbacks.

Main code path:

- Training entry point: [scripts/training/train_model.py](/home/cesarali/Pharma/pff/scripts/training/train_model.py)
- Model callback creation for `PredictionPK`: [pff/models/amortized_inference/prediction_pk.py](/home/cesarali/Pharma/pff/pff/models/amortized_inference/prediction_pk.py)
- Lightning callback attachment: [pff/training/basic_experiment.py](/home/cesarali/Pharma/pff/pff/training/basic_experiment.py)

For `PredictionPK`, the callback is built in `build_visualization_callback()`. Then `BasicLightningExperiment._setup_callbacks()` adds it to the trainer and also gives it access to the experiment checkpoints (`last` and `best`).

## When the scheduler runs

The scheduler supports three moments:

- `tasks_validation`: runs on validation batches.
- `task_during`: runs at training progress milestones such as 20%, 40%, 60%, ...
- `tasks_end`: runs once after training ends.

Internally, these are triggered by Lightning hooks:

- `on_validation_batch_end(...)`
- `on_train_epoch_end(...)`
- `on_train_end(...)`

## Registered task functions

These task keys are registered in [pff/training/callbacks/task_registry.py](/home/cesarali/Pharma/pff/pff/training/callbacks/task_registry.py).

### Predictive tasks

- `pk.empirical.predictive.metrics`
  Computes held-out empirical predictive metrics such as `rmse`, `log_rmse`, `r2`, and `log_r2`.

- `pk.predictive.images`
  Creates plots comparing predictive trajectories and observations.
  For empirical predictive plots, `task_cfg.number_of_predictions_plot_per_drug`
  controls how many held-out predictive plots are created per drug across
  empirical batches and defaults to `1`.

### Generative tasks

- `pk.generative.metrics`
  Computes coverage-style metrics for generated new individuals.

- `pk.generative.images`
  Creates plots for generated new-individual trajectories.

### VPC tasks

- `pk.vpc.npde_pvalues`
  Computes NPDE-based p-values from VPC-style empirical simulations.

- `pk.vpc.images`
  Creates VPC plots for empirical studies.

- `pk.synthetic_experiment.vpc.images`
  Creates VPC plots from synthetic study experiments built inside the task via
  `trainer.datamodule.get_synthetic_experiment_dataloader(...)`.
  This task is self-contained and currently supported only in `tasks_end`.

### Summary task

- `pk.empirical.summary`
  Computes one summary scalar across selected empirical datasets and selected drugs.
  This is useful when you want a single checkpoint-driving metric such as a mean `log_rmse`.

### Diverse Experiment Distances Task

- `pk.diverse_experiment.distances`
  Computes one or more distance metrics on a synthetic experiment dataloader built inside the task.
  The current metrics are signature-kernel `MMD^2` and classifier-based `classifier_auc`.
  Optionally, `plot_num_studies` can be set to save one overlay image showing the
  first synthetic studies with shared context plus semi-transparent real/model
  target trajectories on the same axes.
  This task is self-contained and currently supported only in `tasks_end`.

## Important note for `PredictionPK`

`PredictionPK` is prediction-only. In practice, the current `node-pk` setup mainly uses:

- `pk.empirical.predictive.metrics`
- `pk.predictive.images`
- `pk.empirical.summary`

The generative and VPC tasks are registered globally, but they only make sense when the model exposes the required generation functionality.

## Meaning of the top-level `callbacks_scheduler` parameters

Below is the meaning of the fields in:

```yaml
train:
  callbacks_scheduler:
    ...
```

### `percent_step`

Defines the spacing between training-progress milestones for `task_during`.

Example:

- `percent_step: 0.2` means 20% steps.
- With `epochs: 10`, milestones are reached at 20%, 40%, 60%, 80%, and 100%.
- In epoch terms, that means after epochs 2, 4, 6, 8, and 10.

### `include_end`

If `true`, the scheduler always includes `100%` as the final milestone.

With `percent_step: 0.2`, this ensures the last `task_during` run also happens at the end of training.

### `skip_sanity_check`

If `true`, the scheduler does not run during Lightning sanity-check validation.

This is usually the safe default because sanity checks are only meant to verify that validation runs at all.

### `store_samples`

If `true`, generated scheduler samples are saved under:

```text
<trainer.default_root_dir>/scheduler_samples/
```

This is useful for inspection and debugging.

### `max_samples_per_group`

Caps how many samples are generated for one grouped task run.

If several tasks ask for different `n_samples`, the scheduler generates only once using the maximum requested value, but never above `max_samples_per_group`.

Example:

- one task asks for `1`,
- another asks for `20`,
- `max_samples_per_group` is `10`,
- the scheduler generates `10`, logs a warning, and each task receives a sliced view of that cached output.

### `keep_temp_files`

Controls cleanup of temporary cache files created during task execution.

- `false`: remove temporary cache files after use.
- `true`: keep them for debugging.

### `cache_dir`

Optional custom directory for temporary scheduler cache files.

If not set, the scheduler uses:

1. `trainer.default_root_dir/scheduler_cache`, if available
2. otherwise the system temp directory

### `checkpoint_used_in_end`

Defines which model weights should be loaded when running `tasks_end`.

Supported values are:

- `end`: use the in-memory weights at the end of training
- `last`: load the experiment's last checkpoint
- `best`: load the experiment's best checkpoint
- a custom scheduler metric checkpoint name, such as `log_rmse`, if a task created one

Example:

```yaml
checkpoint_used_in_end:
  - end
  - log_rmse
```

This means:

- run end tasks once using the final training weights,
- then run them again using the best checkpoint saved for the scheduler metric named `log_rmse`.

### `tasks_validation`

List of tasks to run during validation batches.

In your current config this is empty:

```yaml
tasks_validation: []
```

### `task_during`

List of tasks to run during training at milestone percentages defined by `percent_step`.

### `tasks_end`

List of tasks to run once training is finished, optionally for multiple checkpoint selections from `checkpoint_used_in_end`.

## Meaning of each task entry

Each element inside `tasks_validation`, `task_during`, or `tasks_end` is one task definition.

Example:

```yaml
- name: empirical/predictive_metrics
  fn_key: pk.empirical.predictive.metrics
  n_samples: 1
  sample_source: empirical_set
  split: empirical_heldout
  save_to_disk: false
  log_prefix: Empirical
  task_cfg:
    label: Empirical
```

### `name`

Human-readable task name used in logs and metric names.

### `fn_key`

Selects the task function from the registry.

Examples:

- `pk.empirical.predictive.metrics`
- `pk.predictive.images`
- `pk.empirical.summary`

### `n_samples`

How many model samples the task wants.

- Use `0` if the task does not need generated samples.
- Use `1` or more for predictive, generative, or VPC tasks that require sampling.

For example, `pk.empirical.summary` can use `n_samples: 0` because it computes its own summary logic internally.

### `sample_source`

Tells the scheduler where the input context should come from.

Supported values:

- `unconditional`
  Use unconditional generation from the model. This requires `generate_unconditional(...)`.

- `val_batch`
  Use the current validation batch, or the first validation batch if needed.

- `full_split`
  Iterate through the full dataloader for the selected split.

- `empirical_set`
  Use one empirical dataset identified by `empirical_name`, or all configured
  datasets from `mix_data.test_empirical_datasets` when `empirical_name` is omitted.

- `task_internal`
  Use no scheduler-resolved batch context and no scheduler-managed sample generation.
  This is intended for self-contained tasks such as `pk.diverse_experiment.distances`.

### `split`

Which split to use when resolving batches.

Examples:

- `val`
- `train`
- `empirical_heldout`
- `empirical_no_heldout`

### `empirical_name`

Optional when `sample_source: empirical_set`.

When provided, it selects one empirical dataset from the datamodule.
When omitted, the scheduler resolves all configured empirical repos in
`mix_data.test_empirical_datasets` for the same task call.

### `save_to_disk`

Controls whether figures produced by the task should also be written to disk.

For image tasks in this repo, plots are typically saved under:

```text
<trainer.default_root_dir>/training_images/
```

### `log_prefix`

Prefix used for logged metric names.

The scheduler logs outputs using a structure like:

```text
<log_prefix>/<task_name>/<metric_key>
```

Example:

```text
Empirical/empirical/summary/log_rmse
```

### `use_ema`

This field exists in the typed config, but the current scheduler implementation does not use it.

Right now it is effectively unused.

### `checkpoint_metric`

If `true`, the task output can be used to drive a scheduler-managed checkpoint.

This is mainly relevant for scalar outputs such as the summary task.

### `checkpoint_metric_name`

Name used to store the best checkpoint for that task metric.

In your config:

```yaml
checkpoint_metric_name: log_rmse
```

This lets `checkpoint_used_in_end` later refer to `log_rmse`.

### `checkpoint_mode`

How the scheduler decides whether a new metric value is better:

- `min`: smaller is better
- `max`: larger is better

For `log_rmse`, `min` is the correct choice.

### `task_cfg`

Extra task-specific options passed directly to the task function.

This is where task-specific behavior is configured.

Common examples in this repo:

- `label`
  A display label such as `Synthetic` or `Empirical`.

- `milestone_stride`
  Run this task only every N-th percent milestone.
  Example: with `milestone_stride: 2` and milestones at 20%, 40%, 60%, 80%, 100%, the task runs only at 20%, 60%, and 100% because the internal milestone index starts at 0.

- `summary_metric`
  Which metric to summarize in `pk.empirical.summary`.

- `summary_scope`
  Scope of the summary computation.
  `predictive` means predictive metrics only.
  `full` includes predictive, generative, and VPC-style summary inputs when available.

- `selected_summary_drugs`
  Which drugs are included in the cross-repo summary.

- `plot_kwargs`, `n_bins`, `binning`, `log_y`, `repo_id`, `model_label`,
  `number_of_predictions_plot_per_drug`
  Optional extra controls used by specific plotting or VPC tasks.

For `pk.diverse_experiment.distances`, prefer a nested configuration:

```yaml
task_cfg:
  distance_metrics: [mmd2, classifier_auc]
  save_details: false
  mmd:
    python_executable: /home/cesarali/miniconda3/envs/ksig/bin/python
    signature_levels: 4
    estimator: unbiased
    include_time_channel: true
  classifier_auc:
    mode: joint
    include_time_channel: true
    hidden_dim: 64
    num_hidden_layers: 2
    learning_rate: 1.0e-3
    weight_decay: 1.0e-4
    epochs: 100
    batch_size: 128
    seed: 0
    show_progress: true
  synthetic_loader:
    n_targets: 6
    n_dosings: 1
    dosing_mode: diverse_dosing
    dataset_size: 1
```

Optional `synthetic_loader` keys are forwarded to
`get_synthetic_experiment_dataloader(...)` when provided:

`pk.empirical.heldout_generated_classifier` accepts the same
`distance_metrics`, `mmd`, and `classifier_auc` blocks. When `mmd2` is
requested, the task pools all valid held-out empirical series across the
selected repos and compares those pooled held-out versus generated samples via
the same signature-kernel MMD runner used by
`pk.diverse_experiment.distances`.

- `dosing_list_generation`
- `logdose_range`
- `synthetic_target_observation_config`
- `shuffle`

For `pk.synthetic_experiment.vpc.images`, `synthetic_loader.dosing_mode` must
be `vpc_context`. That mode creates context-only synthetic studies for VPC:

- `context` holds the observed synthetic individuals,
- `target` is empty,
- `n_targets` controls the number of observed individuals per study,
- `task_cfg.sample_size` controls the number of Monte Carlo VPC replicates.

## How your current `node-pk` config behaves

The current file is [config_files/experiment_configs/AISTATS/node-pk/base.training.yaml](/home/cesarali/Pharma/pff/config_files/experiment_configs/AISTATS/node-pk/base.training.yaml).

In simple terms, it does this:

- no validation-batch tasks are run,
- during training, every 20% of progress it runs:
  - synthetic predictive images,
  - empirical predictive metrics,
  - empirical summary metric,
- at the very end, it runs end tasks twice:
  - once with the final in-memory weights (`end`),
  - once with the best checkpoint for the scheduler metric named `log_rmse`, if that checkpoint was created.

More specifically:

- `synthetic/predictive_images`
  Uses `val_batch`, samples once, saves images, and labels them as `Synthetic`.

- `empirical/predictive_metrics`
  Uses `empirical_set` with split `empirical_heldout`, samples once, and logs metrics for empirical data.

- `empirical/summary`
  Uses `pk.empirical.summary` to compute a scalar `log_rmse` summary over the selected drug list, currently `Indometacin`.
  Because `checkpoint_metric: true`, this summary can save the best scheduler-managed checkpoint under the name `log_rmse`.

## Practical mental model

If you want to reason about `callbacks_scheduler`, think of it like this:

- `tasks_validation` = "run on validation"
- `task_during` = "run during training at selected percentages"
- `tasks_end` = "run after training ends"
- `fn_key` = "what job should be done"
- `sample_source` and `split` = "which data should the job use"
- `n_samples` = "how much sampling is needed"
- `task_cfg` = "extra options for that specific job"
- `checkpoint_metric*` = "should this task produce a metric that can define a best checkpoint"

## Output locations

Depending on the task and config, the scheduler can write outputs in these places:

- sampled payloads: `scheduler_samples/`
- temporary caches: `scheduler_cache/`
- generated images: `training_images/`
- logger metrics and images: sent through the active Lightning logger, which in this project is typically Comet

## Relevant source files

- Scheduler config types: [pff/config_classes/training_config.py](/home/cesarali/Pharma/pff/pff/config_classes/training_config.py)
- Scheduler callback: [pff/training/callbacks/scheduler.py](/home/cesarali/Pharma/pff/pff/training/callbacks/scheduler.py)
- Task registry: [pff/training/callbacks/task_registry.py](/home/cesarali/Pharma/pff/pff/training/callbacks/task_registry.py)
- PK task implementations: [pff/training/callbacks/pk_tasks.py](/home/cesarali/Pharma/pff/pff/training/callbacks/pk_tasks.py)
- Current training config example: [config_files/experiment_configs/AISTATS/node-pk/base.training.yaml](/home/cesarali/Pharma/pff/config_files/experiment_configs/AISTATS/node-pk/base.training.yaml)
