#!/usr/bin/env python3
"""
Script to prepare an ensemble of PK simulations and save them as JSON.

Usage:
    python prepare_ensemble_of_simulations.py
"""

import json
import tempfile
import pandas as pd
from pathlib import Path
from pff import config_dir, data_dir
from pff.config_classes.data_config import (
    MetaDosingConfig,
    MetaStudyConfig,
    ObservationsConfig,
)
from pff.data.data_generation.compartment_models_management import (
    prepare_ensemble_of_simulations,
)
from pff.data.data_generation.study_population_stats import (
    ListedObservationStats,
)


def main():
    # --- keep this import HERE (like in the test) ---

    # READ CONFIGS (unchanged paths & API)
    experiment_dir = config_dir / "experiment_configs" / "node-pk"
    meta_study_cfg = MetaStudyConfig.from_yaml(experiment_dir / "base-homogeneous.meta_study.yaml")
    dosing_cfg = MetaDosingConfig.from_yaml(experiment_dir / "base-homogeneous.dosing.yaml")
    context_obs_cfg = ObservationsConfig.from_yaml(
        experiment_dir / "base-homogeneous.observations.yaml",
        section="context_observations",
    )

    # Create a temp dir to mirror pytest's tmp_path behavior
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        output_path = tmp_path / "ensemble.json"

        number_of_samples = 200
        # CALL THE SIMULATIONS (same positional + kwargs as in the test)
        studies, failure_rate = prepare_ensemble_of_simulations(
            meta_study_cfg,
            context_obs_cfg,
            dosing_cfg,
            number_of_samples=number_of_samples,
            file_name=str(output_path),
        )

        # Calculate study population stats
        stats_calculator = ListedObservationStats()
        stats = stats_calculator.compute_study_population_statistics(studies)

        print("✅ Computed study population statistics:")
        # print(stats)

        # Export study population statistics to .csv
        df = pd.DataFrame(stats)
        df.to_csv("output.csv", index=False)
        print("✅ Study population statistics saved to output.csv")

        print("✅ Failure rate of simulations:")
        print(failure_rate)

        # Same integrity checks
        assert len(studies) == number_of_samples
        for idx, study in enumerate(studies):
            assert study["meta_data"]["study_name"] == f"simulated_study_{idx}"
            assert "context" in study and isinstance(study["context"], list)

        assert output_path.exists()
        saved = json.loads(output_path.read_text())
        assert saved == studies

        # If you want the file in CWD too, also write it here:
        final_out = data_dir / "preprocessed" / Path("ensemble.json")
        final_out.write_text(json.dumps(studies, indent=2))
        print(f"✅ Ensemble saved to: {final_out}")


if __name__ == "__main__":
    main()
