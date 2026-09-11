# Technical Design: No-Duplication PK Task Split

## Context

The PK scheduler task layer is currently in a transitional state:

- empirical task entrypoints were introduced in
  `pff/training/callbacks/pk_task_empirical.py`,
- synthetic task entrypoints were introduced in
  `pff/training/callbacks/pk_task_synthetic.py`,
- but `pff/training/callbacks/pk_tasks.py` still contains old task
  entrypoints and shared helpers.

As a result, several task functions now exist twice, for example:

- `task_predictive_images`
- `task_generative_images`
- `task_generative_metrics`
- `task_vpc_images`
- `task_vpc_npde_pvalues`

This violates the desired rule that task implementations should exist in one
place only.

## Problem

The current split mixes three concerns:

1. task registry ownership,
2. task entrypoint implementation,
3. shared helper implementation.

The registry already points to the new empirical/synthetic modules, but
`pk_tasks.py` still contains duplicated task bodies and partial compatibility
wrappers. This makes ownership unclear and increases the chance of drift between
two copies of the same task.

## Goals

1. Each task entrypoint must have exactly one canonical implementation.
2. Empirical tasks must live only in `pk_task_empirical.py`.
3. Synthetic tasks must live only in `pk_task_synthetic.py`.
4. Metric-specific utilities such as sampled-distance MMD and classifier AUC
   must live under `pff/metrics/`.
5. `pk.diverse_experiment.distances` must be treated as synthetic-only.
6. The clearer synthetic task name
   `task_diverse_synthetic_experiment_sample_distances` should be the canonical
   Python entrypoint.
7. Existing config `fn_key` values may remain temporarily supported through
   aliases, but aliases must not duplicate bodies.

## Non-Goals

1. Rewriting all experiment YAML files in the same change.
2. Renaming every historical test immediately.
3. Refactoring shared non-task helper logic unless needed to remove task
   duplication.

## Desired End State

### Canonical task ownership

#### Empirical task entrypoints

File:
`pff/training/callbacks/pk_task_empirical.py`

Canonical task functions:

- `task_empirical_predictive_metrics`
- `task_empirical_heldout_generated_classifier`
- `task_empirical_summary`

#### Synthetic task entrypoints

File:
`pff/training/callbacks/pk_task_synthetic.py`

Canonical task functions:

- `task_predictive_images`
- `task_generative_metrics`
- `task_generative_images`
- `task_vpc_npde_pvalues`
- `task_vpc_images`
- `task_diverse_synthetic_experiment_sample_distances`

Compatibility alias allowed:

- `task_diverse_experiment_distances`

This alias must call the canonical synthetic task function and must not contain
its own implementation body.

### Metrics ownership

File:
`pff/metrics/sample_distance_metrics.py`

Canonical contents:

- `_run_mmd2_distance`
- `_run_classifier_auc_distance`
- `_run_signature_mmd_runner`
- `_write_synthetic_mmd_payload`
- `_resolve_mmd_task_cfg`
- `_resolve_classifier_auc_cfg`
- `_resolve_distance_metric_names`
- classifier AUC support utilities used only by sampled-distance evaluation

This module owns sampled-distance metric computation. Task modules should only
coordinate data collection, task-specific artifact layout, and scheduler-facing
outputs.

### Shared callback helpers

File:
`pff/training/callbacks/pk_tasks.py`

This file should become a shared-helper and compatibility module only.

Allowed contents:

- shared dataclasses such as `PredictiveBundle`, `GenerativeBundle`,
  `VPCBundle`, `SyntheticMMDSeriesBundle`,
  `DiverseExperimentDistanceCollection`
- shared sampling helpers
- shared plotting helpers
- shared tensor aggregation helpers
- empirical helper logic genuinely used by empirical tasks
- compatibility aliases that forward to canonical task entrypoints

Disallowed contents in final state:

- duplicated task bodies already owned by `pk_task_empirical.py`
- duplicated task bodies already owned by `pk_task_synthetic.py`

## Registry and fn_key Policy

### Canonical registry mapping

`pff/training/callbacks/task_registry.py` should map:

- `pk.predictive.images` -> synthetic task module
- `pk.generative.metrics` -> synthetic task module
- `pk.generative.images` -> synthetic task module
- `pk.vpc.npde_pvalues` -> synthetic task module
- `pk.vpc.images` -> synthetic task module
- `pk.empirical.predictive.metrics` -> empirical task module
- `pk.empirical.heldout_generated_classifier` -> empirical task module
- `pk.empirical.summary` -> empirical task module
- `pk.diverse_synthetic_experiment.sample_distances` -> synthetic task module

Temporary compatibility mapping allowed:

- `pk.diverse_experiment.distances` ->
  `task_diverse_synthetic_experiment_sample_distances`

### Scheduler policy

The scheduler should treat both:

- `pk.diverse_experiment.distances`
- `pk.diverse_synthetic_experiment.sample_distances`

as synthetic end-only tasks during the migration window.

After config migration is complete, the old `fn_key` may be removed.

## Concrete Refactor Tasks

### Phase 1: Make ownership explicit

1. Keep canonical empirical task bodies only in `pk_task_empirical.py`.
2. Keep canonical synthetic task bodies only in `pk_task_synthetic.py`.
3. Convert any same-named task functions left in `pk_tasks.py` into thin
   forwarding aliases or remove them.

Success condition:

- there is only one implementation body per task entrypoint.

### Phase 2: Move sampled-distance metric logic out of callbacks

1. Keep `_run_mmd2_distance` only in `pff/metrics/sample_distance_metrics.py`.
2. Keep `_run_classifier_auc_distance` only in
   `pff/metrics/sample_distance_metrics.py`.
3. Move any classifier/MMD helpers needed exclusively by sampled-distance tasks
   into that metrics module.
4. Replace old copies in `pk_tasks.py` with imports or forwarding aliases during
   migration, then remove them if not needed.

Success condition:

- sampled-distance metrics are defined only under `pff/metrics/`.

### Phase 3: Make the synthetic diverse-distance task fully synthetic

1. Keep the canonical task body only in
   `pk_task_synthetic.py::task_diverse_synthetic_experiment_sample_distances`.
2. Keep `task_diverse_experiment_distances` only as an alias.
3. Ensure no empirical module imports or owns this task.

Success condition:

- the task is conceptually and physically located in the synthetic module.

### Phase 4: Remove duplicated generic synthetic task wrappers

Move to single ownership for:

- `task_predictive_images`
- `task_generative_metrics`
- `task_generative_images`
- `task_vpc_npde_pvalues`
- `task_vpc_images`

Recommended approach:

1. Keep the canonical versions in `pk_task_synthetic.py`.
2. Replace old versions in `pk_tasks.py` with forwarding aliases if external
   imports still depend on them.
3. Eventually remove those aliases once downstream imports are updated.

### Phase 5: Update imports and tests

1. Tests for sampled-distance metrics should import metric utilities from
   `pff.metrics.sample_distance_metrics`.
2. Tests for synthetic tasks should import canonical task entrypoints from
   `pk_task_synthetic.py`.
3. Tests for empirical tasks should import canonical task entrypoints from
   `pk_task_empirical.py`.
4. Monkeypatch targets should follow the canonical implementation module, not a
   compatibility alias.

### Phase 6: Migrate configs

1. New configs should use
   `fn_key: pk.diverse_synthetic_experiment.sample_distances`.
2. Existing configs using `pk.diverse_experiment.distances` may continue to work
   during migration.
3. Once configs are migrated, remove the old `fn_key` alias from the registry
   and scheduler special case.

## Compatibility Strategy

Short-term compatibility is acceptable, but only in this form:

- compatibility alias with a one-line forward call,
- no second implementation body,
- no branching logic copied into the alias.

Examples of acceptable compatibility:

- `pk_tasks.task_predictive_images` forwards to
  `pk_task_synthetic.task_predictive_images`
- `pk_tasks.task_diverse_experiment_distances` forwards to
  `pk_task_synthetic.task_diverse_synthetic_experiment_sample_distances`

Examples of unacceptable compatibility:

- full duplicate task bodies in both files,
- a task body in `pk_tasks.py` and another slightly different body in the new
  task module.

## Risks

1. Tests may still monkeypatch old module paths.
2. Old imports from `pk_tasks.py` may be widespread.
3. Configs may rely on historical `fn_key` strings.
4. Scheduler special cases may silently refer to old `fn_key` values.

## Mitigations

1. Keep aliases during migration, but keep them body-less.
2. Update tests to patch canonical modules first.
3. Add one migration pass for YAML configs after code stabilization.
4. Search for old `fn_key` strings before removing the compatibility alias.

## Validation Plan

### Static validation

1. `python -m py_compile` on task modules, metrics module, registry, scheduler,
   and updated tests.
2. `rg` checks confirming one canonical definition per task function name.

Suggested checks:

```bash
rg -n "^def task_predictive_images\\(" pff
rg -n "^def task_diverse_synthetic_experiment_sample_distances\\(" pff
rg -n "^def _run_mmd2_distance\\(" pff
rg -n "^def _run_classifier_auc_distance\\(" pff
```

### Runtime validation

1. Synthetic sampled-distance tests
2. Empirical held-out classifier tests
3. Scheduler registry tests
4. Prediction and AICME scheduler expansion tests

## Definition of Done

This refactor is complete when all of the following are true:

1. Every task entrypoint has exactly one canonical implementation body.
2. `pk_task_empirical.py` owns empirical task entrypoints.
3. `pk_task_synthetic.py` owns synthetic task entrypoints.
4. Sample-distance metric computation lives under
   `pff/metrics/sample_distance_metrics.py`.
5. `task_diverse_synthetic_experiment_sample_distances` is the canonical
   synthetic distance task.
6. Old names, if still present, are forwarding aliases only.
7. Tests and monkeypatch targets use canonical modules.
8. No duplicated task wrappers remain across `pk_tasks.py`,
   `pk_task_empirical.py`, and `pk_task_synthetic.py`.
