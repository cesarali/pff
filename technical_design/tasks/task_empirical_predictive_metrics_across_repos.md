# Technical Design: Empirical Predictive Metrics Across Repos

## Context

The current empirical predictive scheduler task
`pk.empirical.predictive.metrics` is implemented in
`pff/training/callbacks/pk_task_empirical.py` and currently operates
on exactly one empirical repo at a time.

That current contract is visible in two places:

1. the task requires one `empirical_name`,
2. the datamodule interface `get_empirical_batches(split, empirical_name)`
   returns one batch list for one repo.

As a result, scheduler configs currently need one task entry per repo when the
user wants predictive metrics for multiple empirical datasets.

## Problem

The desired scientific behavior is not "one task per repo". The desired
behavior is:

1. collect all valid held-out predictive comparisons from all configured
   empirical repos,
2. group those comparisons by substance name,
3. compute one metric mean and one metric standard deviation for each
   substance,
4. report those per-substance metrics directly.

In other words, the semantic unit of aggregation should be:

- one valid held-out target individual,

and the final reporting key should be:

- substance name,

not repo id.

## Desired Metric Semantics

### Observation unit

One observation unit is one valid held-out target individual from one empirical
permutation.

For each such held-out individual we compute predictive metrics such as:

- `rmse`
- `log_rmse`
- `r2`
- `log_r2`

### Pooling rule

Pooling happens **separately for each substance**.

More precisely:

- within one empirical batch, many substances may be present along the batch
  axis,
- each valid held-out target individual already belongs to exactly one
  substance,
- when observations are collected across permutations and across repos, they
  are appended only to the bucket for that same substance,
- there is no global pool mixing `Indometacin`, `Theophylline`, and other
  substances together.

### Grouping rule

The grouping key is normalized substance name, and pooling is performed inside
each such substance group.

Examples:

- all `Indometacin` observations across all repos contribute to the
  `Indometacin` metric group,
- all `Theophylline` observations across all repos contribute to the
  `Theophylline` metric group.

### Aggregation rule

For each substance and each metric:

1. collect the full list of metric values from all valid held-out individuals
   assigned to that substance,
2. compute the arithmetic mean,
3. compute the sample standard deviation over the same list.

This means the reported mean/std for one substance are computed over the full
set of held-out predictive observations assigned to that substance, aggregated
across permutations and across repos, but never mixed with observations from a
different substance.

## Important Clarification

This design does **not** mean:

- average repo-level means and then average those again,
- merge all substances into one global empirical score,
- keep the scheduler task surface one-task-per-repo.

This design **does** mean:

- every valid held-out target individual counts once,
- repo boundaries disappear after observation collection,
- substance boundaries are preserved.

## Goals

1. `pk.empirical.predictive.metrics` should work over all configured empirical
   repos in one task call.
2. The returned outputs should be per-substance metrics, not per-repo metrics.
3. The task should no longer require one scheduler entry per empirical repo.
4. The implementation should keep the current valid-held-out-target semantics:
   only real comparable held-out target individuals are counted.
5. Documentation should make the aggregation semantics explicit.

## Non-Goals

1. Changing predictive image tasks in the same refactor.
2. Changing VPC tasks in the same refactor.
3. Changing the held-out classifier task to cross-repo aggregation in the same
   change.
4. Defining a cross-substance global scalar here. That remains the job of
   summary-style tasks.

## Desired End State

### Task contract

The canonical empirical predictive task should:

1. read the configured empirical repo list from
   `pl_module.model_config.mix_data.test_empirical_datasets`,
2. load held-out empirical batches for each repo,
3. collect predictive metric observations for every valid held-out target
   individual,
4. assign each observation to its substance,
5. aggregate per substance across the full pooled observation set,
6. return one flat metric dictionary keyed by substance.

### Scheduler contract

The scheduler config should need only one predictive-metrics task entry, for
example:

```yaml
- name: empirical/predictive_metrics
  fn_key: pk.empirical.predictive.metrics
  n_samples: 0
  sample_source: task_internal
  split: empirical_heldout
  save_to_disk: false
  log_prefix: Empirical
  task_cfg:
    label: Empirical
    sample_size: 1
    split: empirical_heldout
```

No explicit per-repo `empirical_name` should be required for this task in the
desired final state.

## Proposed Implementation

### 1. Keep one low-level helper for one repo

The existing helper
`_compute_empirical_predictive_metrics_from_batch_list(batch_list, ...)`
already knows how to:

- consume one repo's permutation list,
- collect valid held-out predictive observations,
- aggregate over those observations.

That helper should remain available as the one-repo building block.

### 2. Add a new cross-repo collection helper

Add one shared helper that:

1. loops over all configured empirical repos,
2. loads `datamodule.get_empirical_batches(split="empirical_heldout", empirical_name=repo_id)`,
3. collects the per-target metric observations from each repo,
4. merges those observations into one dictionary keyed by substance.

Important detail:

- this helper should merge raw observation lists, not already-aggregated repo
  means.

### 3. Aggregate once at the end

After all repos have contributed their held-out observations, run one final
aggregation pass:

- per substance,
- per metric name,
- mean and sample standard deviation over the pooled observation list.

### 4. Flat scheduler-facing outputs

Return values should remain scheduler-friendly flat metrics, for example:

- `Indometacin/rmse`
- `Indometacin/rmse_std`
- `Indometacin/log_rmse`
- `Theophylline/rmse`

The exact flattening convention may reuse the existing callback naming helpers.

## Shape and Data Semantics

### Existing empirical predictive path

For one repo:

- `batch_list[p]` is one empirical leave-one-out permutation,
- within each batch, axis `B` indexes study/drug slots,
- within each slot, valid held-out target individuals are the observations that
  contribute metrics.

### New cross-repo path

The new task should not try to align permutations across repos.

That is unnecessary because the aggregation unit is not "repo permutation", but
"valid held-out target individual assigned to one substance".

So the cross-repo logic should:

- preserve the one-repo permutation logic inside each repo,
- concatenate the resulting observation lists by substance across repos.

## Logging Policy

The task output should be interpretable without repo-specific task names.

Recommended naming policy:

- task name stays generic, for example `empirical/predictive_metrics`,
- metric keys carry substance identity,
- repo identity is not part of the final predictive metric namespace unless
  explicitly requested for debugging.

## Backward Compatibility

### Temporary compatibility

During migration, the task may temporarily support:

- explicit `task_cfg.empirical_name` for one-repo execution,
- implicit "all repos" behavior when `empirical_name` is absent.

### Preferred final behavior

For the long-term contract, the preferred meaning of
`pk.empirical.predictive.metrics` is:

- all configured empirical repos,
- aggregated per substance across repos.

If one-repo execution is still needed later, it should ideally be exposed via a
different explicit task or helper, not by keeping the primary task semantics
ambiguous.

## Open Questions

1. Should substance matching across repos use exact string equality or the same
   normalization helper already used elsewhere in the datamodule?
2. Should outputs include the number of held-out observations contributing to
   each substance metric?
3. Should empty substances be omitted silently or reported with warnings?

## Recommendation

Implement this behavior directly in the canonical task
`task_empirical_predictive_metrics` and document it in the task docstring.

That gives the cleanest scheduler surface:

- one task entry,
- one clear scientific meaning,
- per-substance metrics computed across all configured empirical repos.
