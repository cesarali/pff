#!/usr/bin/env python3
"""
Script to prepare an ensemble of PK simulations and save them as JSON.

Usage:
    python prepare_ensemble_of_simulations.py
"""

import tempfile
from pathlib import Path
from typing import List

from pff import config_dir
from pff.config_classes.data_config import (
    MetaDosingConfig,
    MetaStudyConfig,
    ObservationsConfig,
)
from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.data.data_empirical.builder import load_empirical_json_batches_as_dm
from pff.data.data_empirical.json_schema import StudyJSON
from pff.data.data_generation.compartment_models_management import (
    prepare_ensemble_of_simulations,
)
from pff.data.datasets.aicme_datasets import (
    AICMECompartmentsDataBatch,
    AICMECompartmentsDataModule,
)
from pff.utils.plots.databatch_plot import plot_list_list_study_json
from pff import reports_dir


def _aicme_config_from_file() -> NodePKExperimentConfig:
    """Load the default YAML config and shrink it for unit tests."""

    default_yaml = Path(config_dir) / "experiment_configs" / "AISTATS" / "aicme-t-pk" / "base.yaml"
    cfg = NodePKExperimentConfig.from_yaml(str(default_yaml))
    return cfg


def main():
    # --- keep this import HERE (like in the test) ---
    model_config = _aicme_config_from_file()
    dm = AICMECompartmentsDataModule(model_config)
    size = 10
    # READ CONFIGS (unchanged paths & API)
    experiment_dir = config_dir / "experiment_configs" / "node-pk"
    meta_study_cfg = MetaStudyConfig.from_yaml(experiment_dir / "base-homogeneous.meta_study.yaml")
    dosing_cfg = MetaDosingConfig.from_yaml(experiment_dir / "base-homogeneous.dosing.yaml")
    context_obs_cfg = ObservationsConfig.from_yaml(
        experiment_dir / "base-homogeneous.observations.yaml",
        section="context_observations",
    )
    meta_study_cfg.num_individuals_range = (size, size)
    # Create a temp dir to mirror pytest's tmp_path behavior
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        output_path = tmp_path / "ensemble.json"

        number_of_samples = 1
        # CALL THE SIMULATIONS (same positional + kwargs as in the test)
        studies, failure_rate = prepare_ensemble_of_simulations(
            meta_study_cfg,
            context_obs_cfg,
            dosing_cfg,
            number_of_samples=number_of_samples,
            file_name=str(output_path),
        )

        plot_file_name = reports_dir / "study_different_sizes.png"
        studies: List[StudyJSON]
        list_databatches: List[AICMECompartmentsDataBatch]
        list_databatches = load_empirical_json_batches_as_dm(
            None, model_config.dosing, None, dm, studies
        )
        plot_list_list_study_json(list_databatches, file_name=str(plot_file_name))


if __name__ == "__main__":
    main()
