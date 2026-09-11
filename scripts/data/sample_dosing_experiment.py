#!/usr/bin/env python3
"""
Script to prepare an ensemble of PK simulations representing a dosing experiment and save them as JSON.

Usage:
    python sample_dosing_experiment.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Ensure plot rendering works in headless or VS Code RPC environments.

from pff import config_dir, data_dir
from pff.config_classes.data_config import (
    MetaDosingConfig,
    MetaStudyConfig,
    ObservationsConfig,
)
from pff.data.data_generation.compartment_models_management import (
    prepare_full_simulation_to_study_json_context_target,
)

from pff.utils.plots.databatch_plot import plot_list_list_study_json, plot_study_json
from pff import reports_dir


def main():
    # READ CONFIGS (unchanged paths & API)
    experiment_dir = config_dir / "experiment_configs" / "node-pk"
    meta_study_cfg = MetaStudyConfig.from_yaml(experiment_dir / "base-homogeneous.meta_study.yaml")

    # two dosing configs
    meta_dosing_cfg_context = MetaDosingConfig(
        logdose_mean_range=[1.0, 1.0],
        logdose_std_range=[0.0, 0.0],
    )
    meta_dosing_cfg_target = MetaDosingConfig(
        logdose_mean_range=[2.0, 2.0],
        logdose_std_range=[0.0, 0.0],
    )

    context_obs_cfg = ObservationsConfig.from_yaml(
        experiment_dir / "base-homogeneous.observations.yaml",
        section="context_observations",
    )

    # CALL THE SIMULATIONS
    studies, _ = prepare_full_simulation_to_study_json_context_target(
        meta_study_cfg,
        context_obs_cfg,
        meta_dosing_cfg_context,
        meta_dosing_cfg_target,
    )

    # Save json to file:
    final_out = data_dir / "preprocessed" / Path("dosing-experiment.json")
    final_out.write_text(json.dumps(studies, indent=2))
    print(f"✅ Ensemble saved to: {final_out}")
    plot_file_name = reports_dir / "dosing-experiment.png"
    plot_study_json(studies, file_name=str(plot_file_name))


if __name__ == "__main__":
    main()
