# Technical Design / TDD: Synthetic VPC Cleanup And Rebuild

## Summary

This document specifies a full cleanup-and-rebuild of the synthetic VPC path.
The current implementation is to be removed, not repaired incrementally.

The old synthetic VPC path built around:

- `generate_synthetic_vpc_reference(...)`
- `SyntheticVPCReferenceData`
- `SyntheticVPCReferenceBundle`
- `task_synthetic_experiment_vpc_images(...)`

is treated as the wrong abstraction and must be replaced rather than extended.

The redesign introduces exactly one new public API on the existing
`AICMECompartmentsDataModule`, tentatively named
`generate_synthetic_vpc_data_list(...)`, and nothing else at this stage:

- no scheduler task
- no plotting task wrapper
- no model-sampling integration
- no reference bundle abstraction
- no extra synthetic VPC loader/task plumbing

The sole purpose of this API is to generate native synthetic VPC inputs that
can be passed directly to `compute_vpc_data(...)` and then `vpc_plot(...)`
from `sampling_quality.py`.

The phase-1 public interface must remain minimal and must not expose extra
knobs inherited from the old broken abstraction.

## Why The Previous Design Failed

The previous implementation became brittle because it mixed native VPC
generation with downstream task/model machinery.

That coupling created the wrong development pressure:

- generator semantics were shaped by scheduler/task requirements
- VPC-native data generation was hidden behind task-layer abstractions
- synthetic oracle generation and downstream evaluation concerns were mixed
- debugging became difficult because basic VPC input correctness was entangled
  with task execution and model-side integration

This redesign intentionally reduces scope to the smallest correct
implementation:

- one datamodule method
- native `StudyJSON` outputs
- direct compatibility with `compute_vpc_data(...)`
- minimal tests focused only on native VPC generation and plotting

## Scope

### In Scope

- remove the current synthetic VPC reference/task path
- add one new public datamodule method:
  `generate_synthetic_vpc_data_list(...)`
- implement only the minimal generation logic required to produce native VPC
  inputs
- verify the resulting outputs with `compute_vpc_data(...)` and `vpc_plot(...)`

### Out Of Scope

- scheduler integration
- callback integration
- task wrappers
- model-sampling branches
- synthetic VPC loaders
- training integration
- artifact naming policies
- multi-branch oracle-vs-model comparison
- any new public abstraction besides the single datamodule method

## New Public API

### Location

The new API must be added as a method on the existing
`AICMECompartmentsDataModule`.

It must not be implemented:

- as a free function
- as a task-layer API
- as a callback helper
- as a loader abstraction

### Tentative Name

```python
generate_synthetic_vpc_data_list(
    n_cases: int,
    n_observed_individuals: int,
    sample_size: int,
) -> list[tuple[StudyJSON, list[StudyJSON]]]
```

The preferred phase-1 signature above is the intended public API.

Here:

- `n_cases` means the number of independent observed synthetic VPC studies to
  generate
- `n_observed_individuals` means the number of individuals in each observed
  synthetic VPC study
- `sample_size` means the number of simulated replicate studies generated for
  each observed study

The public API must not add:

- `n_dosings`
- `n_targets`
- scheduler-facing parameters
- bundle options
- loader options
- task-layer arguments

The outer `n_cases` parameter is acceptable in phase 1 because it is still
native to the intended VPC object and does not recreate the old synthetic VPC
task/reference abstraction.

### Return Type

The method returns a list of VPC cases.

Each VPC case is a tuple:

```python
(observed_study, simulated_replicates)
```

with semantics:

- `observed_study: StudyJSON`
- `simulated_replicates: list[StudyJSON]`

So the full return value is conceptually:

```python
list[tuple[StudyJSON, list[StudyJSON]]]
```

## Configuration Inheritance

All non-VPC-specific generative ingredients come from the existing datamodule
configuration and internal synthetic data generation pipeline already used
elsewhere in the project.

In other words, the method should only ask for:

- `n_cases`
- `n_observed_individuals`
- `sample_size`

while all other behavior is inherited from the
`AICMECompartmentsDataModule` instance constructed from the canonical test
config, including:

- study-level sampling
- simulator behavior
- default observation strategy behavior
- dosing distributions
- related synthetic generation details

This new API is not a new synthetic data framework, but only a native-VPC view
over the existing synthetic generation pipeline.

## API Semantics

The semantics of `generate_synthetic_vpc_data_list(...)` must be defined
precisely.

For each VPC case:

1. sample one study configuration once
2. keep that study configuration fixed across the whole case
3. build one observed synthetic study from that fixed study-level state
4. generate `sample_size` replicate studies
5. each replicate study contains one simulated realization for every observed
   individual, serialized in the `StudyJSON` structure expected by the native
   VPC code

For phase 1, each observed synthetic VPC study should be serialized as a
context-only `StudyJSON` with `target=[]`.

The returned replicate studies must preserve that same effective structure so
they are directly consumable by `compute_vpc_data(...)`.

The intended semantics are:

- the study config is fixed for the whole case
- the realized dosing layout is taken from the observed study and preserved
  exactly across all simulated replicates
- the observation structure required for native VPC alignment is preserved
- only the individual latent configurations are resampled

Replicate studies are therefore conditional resamples given the observed study
design, not newly randomized study designs.

This point must remain explicit in the implementation and in the docs:

**preserve the observed study’s realized dosing layout**

This wording is required so there is no ambiguity between:

- a dosing sampling distribution, and
- the actual sampled dose/route/time values stored in the reference observed
  study

## Structural Compatibility With Native VPC Code

The new generator exists to feed the native pharmacology VPC code directly.

`compute_vpc_data(data, simulations)` converts both the observed study and all
simulated replicates into rows indexed by:

- `(Type, ID, Time)`

and validates that observations and predictions are structurally identical
before computing VPC summaries.

Therefore, the new generator must return objects that preserve the observed
study’s effective VPC structure closely enough for:

```python
compute_vpc_data(observed_study, simulated_replicates)
```

to run without structural mismatch.

Concretely, each simulated replicate must preserve:

- the same effective `Type` layout
- the same effective individual IDs
- the same effective observation times
- the same individual-wise observation structure required by native VPC input
  validation

The generator does not need to imitate old scheduler payloads or reference
bundles. It only needs to produce native `StudyJSON` objects that satisfy the
native VPC contract.

## Conceptual Relationship To Existing Low-Level Logic

The new generator is conceptually related to the existing repeated-dosing
sample-experiment logic, because both workflows:

- sample one shared study-level state
- then repeatedly resample fresh individuals

That relationship is useful only at the low level.

The new implementation must not be routed through the current synthetic VPC
task/reference abstraction.

Reuse only genuinely useful low-level pieces, for example:

- shared-state sampling
- study-level config sampling
- any helper that serializes simulated trajectories onto an explicit reference
  schedule

Ignore and remove the current higher-level synthetic VPC bundle/task plumbing.

## Proposed Minimal API Shape

The exact signature can be finalized during implementation, but the public API
should be shaped around native VPC semantics rather than task semantics.

The phase-1 public signature should be:

```python
generate_synthetic_vpc_data_list(
    n_cases: int,
    n_observed_individuals: int,
    sample_size: int,
) -> list[tuple[StudyJSON, list[StudyJSON]]]
```

Important note:

- `sample_size` means the number of replicate studies per observed study
- it does not mean the number of observation points
- it does not mean the number of scheduler/model samples

The public API should not expose the current synthetic-VPC-specific reference
bundle concepts.

All other behavior must be inherited from the existing datamodule instance and
its configured synthetic generation pipeline.

## Removal Plan

The current synthetic VPC path must be removed rather than extended.

This cleanup includes removing the current synthetic VPC reference/task path
associated with:

- `generate_synthetic_vpc_reference(...)`
- `SyntheticVPCReferenceData`
- `SyntheticVPCReferenceBundle`
- `task_synthetic_experiment_vpc_images(...)`

The goal of this phase is not to preserve compatibility with the old synthetic
VPC abstraction. The goal is to replace it with one minimal, correct native
generator.

## Canonical Test Config Fixture

The datamodule-based tests must use this exact config path:

`config_files/experiment_configs/UAI/Rebuttal/Potsdam/aicme-t-pk`

This exact config path is the canonical fixture for reproducing the behavior.

The test plan must load this exact config through the existing project
config-loading path, then:

1. instantiate the datamodule
2. call `prepare_data()`
3. call `setup()`
4. call `generate_synthetic_vpc_data_list(...)`

This config is not optional for the test.

It is used only to build the datamodule fixture and must not be reinterpreted
as a broader training, scheduler, or callback integration requirement.

## Narrow Implementation Plan

### Step 1: Remove The Current Synthetic VPC Reference/Task Path

Delete the current synthetic VPC reference/task path and stop treating it as a
foundation for future work.

This should be a delete-first rebuild strategy.

The current synthetic VPC path is sufficiently misleading that it should be
removed before implementing the replacement.

In particular, the old synthetic VPC path built around:

- `generate_synthetic_vpc_reference(...)`
- `SyntheticVPCReferenceData`
- `SyntheticVPCReferenceBundle`
- `task_synthetic_experiment_vpc_images(...)`

should be deleted first so the new implementation is developed from a clean
slate rather than constrained by the old abstraction.

The one safeguard is that this delete-first step applies only to the synthetic
VPC-specific path, not to shared low-level utilities that may still be reused
by the new implementation.

### Step 2: Add One Datamodule Method

Add exactly one new public method on `AICMECompartmentsDataModule`:

- `generate_synthetic_vpc_data_list(...)`

This is the only new public API in this phase.

### Step 3: Factor Only Minimal Native Generation Logic

Implement only the minimal generation logic required to produce native VPC
inputs:

- sample one fixed study-level state per VPC case
- build one observed study from that state
- serialize the observed study as context-only `StudyJSON` with `target=[]`
- preserve the observed study’s realized dosing layout
- resample only individual latent configurations across replicate studies
- serialize replicate trajectories onto the observed study’s explicit VPC
  structure

### Step 4: Stop

Do not add:

- scheduler tasks
- callback wrappers
- model-in-the-loop branches
- synthetic VPC bundle abstractions
- new dataloaders or task-only helpers

That work is intentionally deferred.

## Narrow Test Plan

All tests in this phase must run using the datamodule instantiated from the
canonical config fixture:

`config_files/experiment_configs/UAI/Rebuttal/Potsdam/aicme-t-pk`

No scheduler tests, callback tests, or model-in-the-loop tests belong in this
phase.

### Test 1: Return Shape

Using the datamodule instantiated from the canonical config fixture:

- call
  `generate_synthetic_vpc_data_list(n_cases=..., n_observed_individuals=..., sample_size=...)`
- verify it returns the requested number of VPC cases
- verify the returned list length equals `n_cases`
- verify each case has the exact shape:
  `(StudyJSON, list[StudyJSON])`

### Test 2: Native VPC Computation

Using the same datamodule fixture:

- call
  `generate_synthetic_vpc_data_list(n_cases=..., n_observed_individuals=..., sample_size=...)`
- for every returned case, run:
  `compute_vpc_data(observed_study, simulated_replicates)`
- verify execution succeeds
- verify the returned VPC data is non-empty

### Test 3: Native VPC Plotting

Using the same datamodule fixture:

- call
  `generate_synthetic_vpc_data_list(n_cases=..., n_observed_individuals=..., sample_size=...)`
- for every returned case, run `compute_vpc_data(...)`
- pass the output to `vpc_plot(...)`
- verify plotting can be called without failure

That is all that should be implemented and tested in this phase.

## Acceptance Criteria

This cleanup-and-rebuild phase is complete only when all of the following are
true:

1. the old synthetic VPC reference/task abstraction is removed rather than
   extended
2. there is exactly one new public synthetic VPC API:
   `AICMECompartmentsDataModule.generate_synthetic_vpc_data_list(...)`
3. the new API returns native VPC cases as
   `(observed_study, simulated_replicates)`
4. each VPC case fixes one study-level state for the observed study and all
   replicates
5. replicate studies preserve the observed study’s realized dosing layout
6. replicate studies resample only the individual latent configurations
7. `compute_vpc_data(observed_study, simulated_replicates)` runs successfully
   on the generated outputs
8. `vpc_plot(...)` can be called on the resulting VPC data
9. no synthetic VPC scheduler task, callback integration, loader abstraction,
   or model-sampling branch is added in this phase
