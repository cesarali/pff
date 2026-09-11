# Technical Design / TDD: Synthetic Truth-vs-Model VPC End Task

## Summary

This document specifies the next synthetic VPC phase after the native
generator rebuild.

The native generator already exists as:

```python
AICMECompartmentsDataModule.generate_synthetic_vpc_data_list(
    n_cases: int,
    n_observed_individuals: int,
    sample_size: int,
) -> list[tuple[StudyJSON, list[StudyJSON]]]
```

That phase established the correct native data contract:

- one observed synthetic study per case,
- one list of truth replicates per case,
- direct compatibility with `compute_vpc_data(...)`,
- direct compatibility with `vpc_plot(...)`.

This new phase adds exactly one scheduler task on top of that native
generator.

The task goal is:

- build synthetic VPC "truth" cases from
  `generate_synthetic_vpc_data_list(...)`,
- sample model-side replicate studies from
  `sample_new_individuals_to_vpc_format(...)`,
- render **paired VPC plots** for each case:
  one truth panel and one model panel.

This task is an end-of-training self-contained task with:

- `sample_source: task_internal`
- `log_prefix: Synthetic`

## Prior Phase Dependency

This task depends on the phase-1 synthetic VPC rebuild documented in:

- [synthetic_vpc_dual_track_tdd.md](/home/cesarali/Pharma/pff/technical_design/metrics/synthetic_vpc_dual_track_tdd.md)

That earlier document deliberately excluded task integration.
This document now adds that deferred task layer.

## Goal

For each synthetic VPC case, create one output image containing a paired
comparison:

1. left panel: VPC computed from synthetic truth replicates,
2. right panel: VPC computed from model-sampled replicates.

Both panels must use the **same observed study** for that case.

The scientific comparison is therefore:

- same observed synthetic study,
- same realized dosing layout,
- same observation schedule,
- truth replicates from the synthetic generator,
- model replicates from the model API.

## Scope

### In Scope

- add one new scheduler task function
- register one new task key
- support `tasks_end` with `sample_source: task_internal`
- build native synthetic VPC cases via
  `generate_synthetic_vpc_data_list(...)`
- convert each observed study into the model input batch expected by
  `sample_new_individuals_to_vpc_format(...)`
- compute truth-side VPC data with `compute_vpc_data(...)`
- compute model-side VPC data with `compute_vpc_data(...)`
- plot both sides with `vpc_plot(...)`
- save one paired image per case

### Out Of Scope

- validation-time synthetic VPC task runs
- milestone-time synthetic VPC task runs
- new synthetic VPC datamodule APIs
- new synthetic VPC bundle abstractions
- new synthetic VPC dataloaders
- scalar metric summaries for synthetic truth-vs-model VPC
- a third "difference" panel
- empirical VPC changes

## New Task Contract

### Proposed Task Key

```text
pk.synthetic.vpc.paired_images
```

This is intentionally a new task key.
The previously removed synthetic VPC task key should not be revived as-is,
because it represented the wrong abstraction layer.

### Intended Scheduler Placement

This task is supported only in:

- `tasks_end`

It must be rejected from:

- `tasks_validation`
- `task_during`

### Scheduler Entry Shape

The intended scheduler entry is:

```yaml
- name: synthetic/vpc_paired_images
  fn_key: pk.synthetic.vpc.paired_images
  n_samples: 0
  sample_source: task_internal
  split: val
  save_to_disk: true
  log_prefix: Synthetic
  task_cfg:
    n_cases: 10
    sample_size: 500
    n_observed_individuals: 10
```

### Default Task Configuration

The task should use the following defaults when fields are absent:

- `n_cases = 10`
- `sample_size = 500`
- `n_observed_individuals = 10`

Optional plot-facing fields may also be supported, for example:

- `model_label`
- `log_y`
- `n_bins`
- `binning`

But the default scientific configuration must be the three fields above.

### Important Clarification About `n_samples`

The scheduler-level `n_samples` field must remain:

- `n_samples: 0`

because this is a `task_internal` task.

The replicate count used for both truth and model generation belongs in:

- `task_cfg.sample_size`

These two notions must not be mixed.

## Model API Requirement

The task must use the existing model-side API already exposed by:

- `AICMEPK`
- `FlowPK`

namely:

```python
sample_new_individuals_to_vpc_format(batch, sample_size=...)
```

As with the existing VPC-related helpers, the task may resolve this API from:

- `pl_module`, or
- `pl_module.model`

The task must not introduce a second model-side VPC sampling API.

## High-Level Semantics

For each synthetic VPC case:

1. build one observed synthetic study plus truth replicates by calling
   `datamodule.generate_synthetic_vpc_data_list(...)`
2. use the observed study from that case as the shared VPC reference
3. compute truth-side VPC data:
   `compute_vpc_data(observed_study, truth_replicates, ...)`
4. convert the observed study into the internal databatch form expected by the
   model-side API
5. sample model replicates with:
   `sample_new_individuals_to_vpc_format(batch, sample_size=...)`
6. compute model-side VPC data:
   `compute_vpc_data(observed_study, model_replicates, ...)`
7. render both VPC results into one paired figure

## Paired Figure Semantics

Each case should produce one figure with two subplots:

- subplot 1: `"Synthetic Truth"`
- subplot 2: `"Model Samples"`

Both subplots must be computed against the same `observed_study`.

This is critical.
The comparison is only interpretable if both branches are aligned to the same
observed synthetic design.

## Relationship Between Truth And Model Branches

For one case:

- `observed_study` comes from the datamodule native synthetic VPC generator
- `truth_replicates` also come from that same datamodule call
- `model_replicates` come from the model API, but must be conditioned on the
  same observed study structure

Therefore:

- truth branch and model branch are directly comparable
- the observed study is shared
- the case identity is shared
- only the source of replicate studies differs

## Internal Data Flow

### 1. Generate Native VPC Cases

The task must call:

```python
datamodule.generate_synthetic_vpc_data_list(
    n_cases=n_cases,
    n_observed_individuals=n_observed_individuals,
    sample_size=sample_size,
)
```

This is the only supported source of synthetic truth cases for this task.

### 2. Rebuild One Evaluation Batch Per Case

The model-side API `sample_new_individuals_to_vpc_format(...)` works from an
`AICMECompartmentsDataBatch`, not directly from `StudyJSON`.

So the task should reuse the existing internal study-to-batch conversion path
inside the datamodule.

That means:

- build one batch from `[observed_study]`
- keep this conversion private/internal
- do not create a new public abstraction for synthetic VPC evaluation batches

Reusing the existing internal builder is acceptable in this task phase.

### 3. Sample Model Replicates

For one case batch:

```python
model_replicates_by_substance = vpc_sampler.sample_new_individuals_to_vpc_format(
    batch_device,
    sample_size=sample_size,
)
```

Because one case corresponds to one observed study, the task should require
that the returned model-side structure resolve to exactly one replicate-study
list for that case.

If the returned structure is inconsistent with the one-case assumption, the
task should fail loudly rather than silently guessing.

## Plotting Contract

Truth and model branches must each use:

1. `compute_vpc_data(...)`
2. `vpc_plot(...)`

directly.

This task must not introduce a separate synthetic-VPC-specific plotting stack.

### Truth Branch

```python
truth_vpc = compute_vpc_data(observed_study, truth_replicates, ...)
vpc_plot(truth_vpc, ax=axes[0], ...)
```

### Model Branch

```python
model_vpc = compute_vpc_data(observed_study, model_replicates, ...)
vpc_plot(model_vpc, ax=axes[1], ...)
```

## Output Policy

The task should save:

- one PNG per case

Suggested filename shape:

```text
epoch_{epoch:03d}_case_{case_idx:03d}_{substance_name}.png
```

Suggested output directory shape:

```text
<plot_root>/<model_label>/synthetic_vpc_paired/
```

The exact folder name may be finalized in implementation, but the task must
avoid creating one tree for truth and another for model because the artifact
semantic unit is now one paired comparison per case.

## Progress Reporting

The task builds on `generate_synthetic_vpc_data_list(...)`, which already
shows progress for:

- case generation
- replicate generation

The task itself may additionally report progress over:

- paired image rendering per case

but this is optional.
The main requirement is that native case generation progress remains visible.

## Default Task Config Semantics

The defaults:

- `n_cases = 10`
- `sample_size = 500`
- `n_observed_individuals = 10`

mean:

- create 10 independent synthetic observed studies,
- each observed study has 10 individuals,
- truth branch uses 500 replicates per case,
- model branch also uses 500 replicates per case.

This symmetry between truth and model sample size must remain explicit.

## Error Handling

The task should raise clear errors when:

- `trainer.datamodule` is missing
- the datamodule does not support `generate_synthetic_vpc_data_list(...)`
- `sample_size <= 0`
- `n_cases <= 0`
- `n_observed_individuals <= 0`
- the model does not expose `sample_new_individuals_to_vpc_format(...)`
- model-side returned replicate structure does not match the expected one-case
  format
- `compute_vpc_data(...)` fails for either branch

Silent skipping is not preferred here because the point of the task is a
direct paired scientific comparison.

## Proposed Implementation Plan

### Step 1: Add New Task Function

Add a new task function in:

- `pff/training/callbacks/pk_task_synthetic.py`

Suggested name:

- `task_synthetic_vpc_paired_images(...)`

### Step 2: Register New Task Key

Register:

- `pk.synthetic.vpc.paired_images`

in:

- `pff/training/callbacks/task_registry.py`

### Step 3: Restrict To `tasks_end`

Mirror the existing `task_internal` synthetic-task restriction logic in:

- `pff/training/callbacks/scheduler.py`

so this task is accepted only in `tasks_end`.

### Step 4: Resolve The Model VPC Sampler

Reuse the same model-resolution pattern already used elsewhere:

- prefer `pl_module.sample_new_individuals_to_vpc_format`
- fallback to `pl_module.model.sample_new_individuals_to_vpc_format`

### Step 5: Build One Paired Figure Per Case

For each case:

1. compute truth-side VPC
2. build evaluation batch from observed study
3. sample model replicates
4. compute model-side VPC
5. render paired subplots
6. save one PNG

### Step 6: Stop

Do not add in this phase:

- synthetic VPC scalar metrics
- synthetic VPC checkpoint-driving summaries
- validation-time task variants
- task-specific bundle abstractions

## Test Plan

### Test 1: Registry Exposure

Verify that:

- `TASK_REGISTRY` contains `pk.synthetic.vpc.paired_images`

### Test 2: Scheduler Placement Restriction

Verify that the task:

- is rejected outside `tasks_end`
- is accepted inside `tasks_end`

with:

- `sample_source: task_internal`
- `log_prefix: Synthetic`

### Test 3: Default Task Config Semantics

With a dummy datamodule:

- call the task without explicitly passing
  `n_cases`, `sample_size`, `n_observed_individuals`
- verify the task calls
  `generate_synthetic_vpc_data_list(n_cases=10, n_observed_individuals=10, sample_size=500)`

### Test 4: Model Sampling Invocation

With a dummy model:

- verify `sample_new_individuals_to_vpc_format(..., sample_size=500)` is used
  by default
- verify explicit `task_cfg.sample_size` overrides the default

### Test 5: One Image Per Case

With a dummy datamodule returning `n_cases = 3` cases:

- verify the task saves exactly 3 images
- verify each image exists
- verify each image is non-empty

### Test 6: Truth And Model Both Use Native VPC Helpers

Patch or spy on:

- `compute_vpc_data`
- `vpc_plot`

and verify:

- each case performs one truth-side `compute_vpc_data(...)`
- each case performs one model-side `compute_vpc_data(...)`
- each case performs two `vpc_plot(...)` calls, one per subplot

### Test 7: Potsdam Integration Smoke Test

Using the canonical config path:

`config_files/experiment_configs/UAI/Rebuttal/Potsdam/aicme-t-pk`

1. instantiate the datamodule
2. call `prepare_data()`
3. call `setup()`
4. instantiate an `AICMEPK` model on CPU
5. run the task with a small test override such as:
   - `n_cases=1`
   - `sample_size=4`
   - `n_observed_individuals=3`
6. verify the task saves one paired image successfully

Optionally, repeat the same smoke test with `FlowPK`.

## Example Scheduler Config

The intended final scheduler entry should look like:

```yaml
- name: synthetic/vpc_paired_images
  fn_key: pk.synthetic.vpc.paired_images
  n_samples: 0
  sample_source: task_internal
  split: val
  save_to_disk: true
  log_prefix: Synthetic
  task_cfg:
    n_cases: 10
    sample_size: 500
    n_observed_individuals: 10
```

## Acceptance Criteria

This phase is complete only when all of the following are true:

1. a new task key `pk.synthetic.vpc.paired_images` exists
2. the task is supported only in `tasks_end`
3. the task uses `generate_synthetic_vpc_data_list(...)` as the truth source
4. the task uses `sample_new_individuals_to_vpc_format(...)` as the model
   source
5. the task default config is:
   - `n_cases=10`
   - `sample_size=500`
   - `n_observed_individuals=10`
6. one paired image is saved per case
7. each paired image contains:
   - one truth VPC subplot
   - one model VPC subplot
8. both branches use the same observed study for the case
9. both branches are computed via `compute_vpc_data(...)`
10. both branches are rendered via `vpc_plot(...)`
11. the task works with models exposing
    `sample_new_individuals_to_vpc_format(...)`, including `AICMEPK` and
    `FlowPK`
