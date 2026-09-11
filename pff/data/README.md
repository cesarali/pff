# `pff.data` Package Guide

This guide documents the purpose of every subpackage that lives under `pff/data`. Use it as a quick reference when wiring new data pipelines or navigating the simulated pharmacokinetic (PK) workflow.

## Configuration Preamble

Simulations are configured by combining reusable YAML files with the dataclasses that live in `pff.config_classes`. YAMLS are the file that populates those classes, we can define configs from the classes or reading the files. 

- **YAML files (`config_files/`)** – Ready-to-use experiment definitions grouped under `config_files/experiment_configs`. For example, the `node-pk` folder contains `base-homogeneous.*.yaml` files that describe meta-study, dosing, and observation settings.
- **Config dataclasses (`pff/config_classes/`)** – Python dataclasses (`MetaStudyConfig`, `MetaDosingConfig`, `ObservationsConfig`, and friends) that parse those YAML files or can be instantiated directly in code when you need programmatic overrides.

When you load configurations in tests or scripts, prefer `MetaStudyConfig.from_yaml(...)` and similar helpers. They keep the simulation code aligned with the canonical YAML layout while still allowing you to craft configurations in pure Python when necessary.

## Top-Level Layout

This are the files that matter for the handling of simulations anda data:

```

pff/
├── config_files/
│   └── experiment_configs/
├── scripts/
├── pff/
│   ├── config_classes/
│   └── data/
│       ├── data_empirical/
│       ├── data_generation/
│       ├── data_preprocessing/
│       ├── datasets/
│       └── extra/
└── tests/
    └── data/
        └── simulation_data/
            └── test_simulations.py
```

Each directory is described below together with the most important entry points it exposes.

## `data_empirical`

Defines the data contracts used across the project.
These contracts specify the canonical JSON schema (StudyJSON, IndividualJSON) that standardizes how pharmacokinetic studies are represented — both empirical and simulated.
They serve as the interface between raw datasets, tensor batches, and model-ready data structures, ensuring a unified format throughout the pipeline. These helpers make it straightforward to load Hugging Face datasets or local JSON files, validate them, and materialise PyTorch-compatible batches.


## `data_generation`

Simulation building blocks used to synthesise PK trajectories under configurable dosing and observation schemes.

* [`compartment_models.py`](data_generation/compartment_models.py) implements the stochastic sampling of population/individual PK parameters and the compartmental simulation loops.
* [`observations_classes.py`](data_generation/observations_classes.py) describe observation strategies (e.g. sparse vs. dense sampling) and utilities to realise them.
* [`compartment_models_management.py`](data_generation/compartment_models_management.py) orchestrates the full simulation workflow: it takes the meta-configuration, samples individual and dosing configurations, runs the compartmental simulations, applies the observation strategy, and assembles complete ensembles of studies in the data contracts.

Together these modules allow you to go from configuration dataclasses to simulated studies that mirror the empirical format.

## `data_preprocessing`

deprecated

## `datasets`

Lightning-ready dataset/dataloader factories.

- [`aicme_datasets.py`](datasets/aicme_datasets.py) defines `AICMECompartmentsDataBatch` and related PyTorch Lightning `DataModule` wrappers that harmonise both empirical and simulated studies for downstream training.


## Putting It All Together

A typical workflow is:

1. **Configure**: Use `pff.config_classes` to describe study, dosing, and observation priors.
2. **Simulate**: Call into `data_generation` to sample synthetic studies or to augment empirical cohorts.
3. **Serialise or load**: Store simulations as JSON, or load existing JSON/CSV with `data_empirical` and `data_preprocessing`.
4. **Batch**: Wrap tensors using `datasets.AICMECompartmentsDataModule` for consumption by modules in `pff.models` and training scripts.

Refer back to this document whenever you onboard a new collaborator or reorganise data flows—the sections above stay aligned with the current code base.

## Worked Examples and Tests

Integration-style tests in `tests/data/simulation_data/test_simulations.py` demonstrate how the configuration pieces fit together:

- `test_prepare_full_simulation_to_study_json` shows how YAML-driven configs from `config_files/experiment_configs/node-pk` feed into `prepare_full_simulation_to_study_json` and culminate in a canonical `StudyJSON`.
- `test_prepare_ensemble_of_simulations` builds on the same configuration files to generate an ensemble of studies and persists them to disk, illustrating how bulk simulations can be orchestrated.

Use these tests as executable documentation whenever you need to follow the end-to-end flow from configuration files to simulated study artefacts.
