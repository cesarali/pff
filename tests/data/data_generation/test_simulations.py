import json
import math
from dataclasses import replace
from pathlib import Path

import matplotlib
import pytest

from pff.config_classes.data_config import (
    MetaDosingConfig,
    MetaDosingWithDurationConfig,
    MetaStudyConfig,
    MixDataConfig,
    ObservationsConfig,
)
from pff.data.data_empirical.json_schema import canonicalize_study
from pff.data.data_generation.compartment_models import (
    derive_timescale_parameters,
    sample_dosing_arrays,
    sample_dosing_configs,
    sample_study_dosing_config,
    sample_dosing_with_duration_configs,
    sample_individual_configs,
    sample_study,
    sample_study_config,
    sample_study_with_duration,
)
from pff.data.data_generation.compartment_models_management import (
    prepare_ensemble_of_simulations,
    prepare_full_simulation,
    prepare_full_simulation_to_study_json,
)
from pff.data.data_generation.observations_classes import (
    PKPeakHalfLifeStrategy,
)
from pff.utils.plots import plot_study_json

torch = pytest.importorskip("torch")
matplotlib.use("Agg")


def get_base_configs():
    meta_study_config = MetaStudyConfig()
    mix_data_config = MixDataConfig()
    meta_dosing_config = MetaDosingConfig()
    return meta_study_config, mix_data_config, meta_dosing_config


def test_sample_study_dosing_config_and_arrays() -> None:
    """Study-level dosing sampling should preserve shared route and dose parameters."""

    meta_dosing_config = MetaDosingConfig(
        num_individuals=4,
        same_route=True,
        logdose_mean_range=(1.0, 1.0),
        logdose_std_range=(0.0, 0.0),
        route_options=["iv"],
        route_weights=[1.0],
        time=2.5,
    )

    study_dosing_config = sample_study_dosing_config(meta_dosing_config)

    assert study_dosing_config.logdose_mean == pytest.approx(1.0)
    assert study_dosing_config.logdose_std == pytest.approx(0.0)
    assert study_dosing_config.route == "iv"
    assert study_dosing_config.time == pytest.approx(2.5)

    dosing_config_array = sample_dosing_arrays(study_dosing_config, num_individuals=4)

    assert len(dosing_config_array) == 4
    assert [cfg.route for cfg in dosing_config_array] == ["iv"] * 4
    assert [cfg.time for cfg in dosing_config_array] == pytest.approx([2.5] * 4)
    assert [cfg.dose for cfg in dosing_config_array] == pytest.approx([math.exp(1.0)] * 4)


def test_simulation():
    meta_study_config, mix_data_config, meta_dosing_config = get_base_configs()
    study_config = sample_study_config(meta_study_config)
    individual_config_array = sample_individual_configs(study_config)
    time_scales = derive_timescale_parameters(study_config, meta_study_config)

    # Generate time points
    time_points = torch.linspace(
        meta_study_config.time_start,
        meta_study_config.time_stop,
        meta_study_config.time_num_steps,
        dtype=torch.float32,
    )
    # Sample individual dosing configurations
    local_meta_dosing = replace(meta_dosing_config, num_individuals=study_config.num_individuals)
    dosing_config_array = sample_dosing_configs(local_meta_dosing)

    # Generate full simulation data
    full_simulation, full_simulation_times, dosing_amounts, dosing_route_types = sample_study(
        individual_config_array,
        dosing_config_array,
        time_points,
        meta_study_config.solver_method,
    )
    print(full_simulation.shape)

    return full_simulation, full_simulation_times, dosing_amounts, dosing_route_types


def test_simulation_with_duration():
    meta_study_config, mix_data_config, _ = get_base_configs()
    meta_dosing_config = MetaDosingWithDurationConfig()
    study_config = sample_study_config(meta_study_config)
    individual_config_array = sample_individual_configs(study_config)
    time_scales = derive_timescale_parameters(study_config, meta_study_config)

    # Generate time points
    time_points = torch.linspace(
        meta_study_config.time_start,
        meta_study_config.time_stop,
        meta_study_config.time_num_steps,
        dtype=torch.float32,
    )
    # Sample individual dosing configurations
    local_meta_dosing = replace(meta_dosing_config, num_individuals=study_config.num_individuals)
    dosing_config_array = sample_dosing_with_duration_configs(local_meta_dosing)

    # Generate full simulation data
    full_simulation, full_simulation_times, dosing_amounts, dosing_route_types = sample_study_with_duration(
        individual_config_array,
        dosing_config_array,
        time_points,
        meta_study_config.solver_method,
    )
    print(full_simulation.shape)

    return full_simulation, full_simulation_times, dosing_amounts, dosing_route_types


def test_observations():
    # --- prepare configs and simulated data -------------------------------
    meta_study_config, mix_data_config, meta_dosing_config = get_base_configs()
    obs_config = ObservationsConfig(
        max_num_obs=10,
        add_rem=True,
        split_past_future=True,
        min_past=2,
        max_past=4,
    )

    full_simulation, full_simulation_times, dosing_amounts, dosing_route_types = test_simulation()

    # --- create the strategy ----------------------------------------------
    observation_strategy = PKPeakHalfLifeStrategy(obs_config, meta_study_config)

    # --- determine expected tensor shapes ---------------------------------
    # NOTE: The PKPeakHalfLifeStrategy exposes its raw shape computation via
    # `_get_shapes_raw`.  The test uses these values so that the assertions
    # remain aligned with any future adjustments to the strategy logic.
    expected_obs, expected_rem = observation_strategy._get_shapes_raw()

    # --- run generate() ---------------------------------------------------
    obs_out, time_out, mask_out, rem_sim, rem_time, rem_mask, rescaled_scales = (
        observation_strategy.generate(
            full_simulation=full_simulation,
            full_simulation_times=full_simulation_times,
            time_scales=torch.tensor([0.3, 0.6]),
        )
    )

    # --- basic sanity checks ----------------------------------------------
    print("obs_out.shape:", obs_out.shape)
    print("time_out.shape:", time_out.shape)
    print("mask_out.sum:", mask_out.sum().item())
    print("rem_sim.shape:" if rem_sim is not None else "rem_sim: None")
    print("rescaled scales:", rescaled_scales)

    # --- assertions for automated testing --------------------------------
    batch_size = full_simulation.shape[0]
    assert obs_out.shape == (batch_size, expected_obs), "Observation output shape mismatch"
    assert time_out.shape == (batch_size, expected_obs), "Observation time shape mismatch"
    assert mask_out.shape == (batch_size, expected_obs), "Observation mask shape mismatch"
    assert time_out.max() <= 1.0 + 1e-6, "Time outputs must be normalized"
    assert mask_out.dtype == torch.bool, "Mask must be boolean"

    if expected_rem == 0:
        assert rem_sim is None and rem_time is None and rem_mask is None, (
            "No remaining tensors expected when canonical remainder capacity is zero"
        )
    else:
        assert rem_sim is not None and rem_time is not None and rem_mask is not None, (
            "Remaining tensors should be returned when canonical remainder capacity is positive"
        )
        assert rem_sim.shape == (batch_size, expected_rem), "Remaining simulation shape mismatch"
        assert rem_time.shape == (batch_size, expected_rem), "Remaining time shape mismatch"
        assert rem_mask.shape == (batch_size, expected_rem), "Remaining mask shape mismatch"
        assert rem_mask.dtype == torch.bool, "Remaining mask must be boolean"

    return obs_out, time_out, mask_out, rem_sim, rem_time, rem_mask, rescaled_scales


def test_full_simulation():
    from pff import config_dir

    # get configurations from file
    experiment_dir = config_dir / "experiment_configs" / "node-pk"
    meta_study_cfg = MetaStudyConfig.from_yaml(experiment_dir / "base-homogeneous.meta_study.yaml")
    dosing_cfg = MetaDosingConfig.from_yaml(experiment_dir / "base-homogeneous.dosing.yaml")
    observations_path = experiment_dir / "base-homogeneous.observations.yaml"
    context_obs_cfg = ObservationsConfig.from_yaml(
        observations_path, section="context_observations"
    )
    observation_strategy = PKPeakHalfLifeStrategy(context_obs_cfg, meta_study_cfg)

    full_sim, full_times, dosing_amounts, dosing_routes, time_points, time_scales = (
        prepare_full_simulation(meta_study_cfg, dosing_cfg)
    )

    obs_out, time_out, mask_out, rem_sim, rem_time, rem_mask, rescaled_scales = (
        observation_strategy.generate(
            full_simulation=full_sim,
            full_simulation_times=full_times,
            time_scales=time_scales,
        )
    )


def test_full_simple_simulation():
    from pff import config_dir

    # get configurations from file
    experiment_dir = config_dir / "experiment_configs" / "node-pk"
    meta_study_cfg = MetaStudyConfig.from_yaml(experiment_dir / "base-homogeneous.meta_study.yaml")
    dosing_cfg = MetaDosingConfig.from_yaml(experiment_dir / "base-homogeneous.dosing.yaml")
    observations_path = experiment_dir / "base-homogeneous.observations.yaml"
    context_obs_cfg = ObservationsConfig.from_yaml(
        observations_path, section="context_observations"
    )
    observation_strategy = PKPeakHalfLifeStrategy(context_obs_cfg, meta_study_cfg)

    full_sim, full_times, dosing_amounts, dosing_routes, time_points, time_scales = (
        prepare_full_simulation(meta_study_cfg, dosing_cfg)
    )

    obs_out, time_out, mask_out, rem_sim, rem_time, rem_mask, rescaled_scales = (
        observation_strategy.generate(
            full_simulation=full_sim,
            full_simulation_times=full_times,
            time_scales=time_scales,
        )
    )


def test_prepare_full_simulation_to_study_json():
    from pff import config_dir

    experiment_dir = config_dir / "experiment_configs" / "node-pk"
    meta_study_cfg = MetaStudyConfig.from_yaml(experiment_dir / "base-homogeneous.meta_study.yaml")
    dosing_cfg = MetaDosingConfig.from_yaml(experiment_dir / "base-homogeneous.dosing.yaml")
    observations_path = experiment_dir / "base-homogeneous.observations.yaml"
    context_obs_cfg = ObservationsConfig.from_yaml(
        observations_path, section="context_observations"
    )

    study_json, failed_attempts = prepare_full_simulation_to_study_json(
        meta_study_cfg, context_obs_cfg, dosing_cfg
    )

    assert "context" in study_json and isinstance(study_json["context"], list)
    assert "target" in study_json and study_json["target"] == []
    assert "meta_data" in study_json
    assert study_json["meta_data"].get("study_name")
    assert study_json["meta_data"].get("substance_name")

    assert len(study_json["context"]) > 0
    first_individual = study_json["context"][0]
    assert len(first_individual["observations"]) == len(first_individual["observation_times"])
    assert "dosing" in first_individual and len(first_individual["dosing"]) == 1
    assert len(first_individual["dosing_type"]) == 1
    assert len(first_individual["dosing_times"]) == 1

    canonical = canonicalize_study(study_json, drop_tgt_too_few=False)
    assert len(canonical["context"]) == len(study_json["context"])

    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    requested_plot_path = reports_dir / "study_plot.png"
    saved_plot_path = reports_dir / "study_plot.pdf"

    saved_path = plot_study_json(study_json, file_name=str(requested_plot_path))

    assert saved_path == str(saved_plot_path)
    assert saved_plot_path.exists()
    assert failed_attempts >= 0


def test_prepare_ensemble_of_simulations(tmp_path):
    from pff import config_dir

    # READ CONFIGS
    experiment_dir = config_dir / "experiment_configs" / "node-pk"
    meta_study_cfg = MetaStudyConfig.from_yaml(experiment_dir / "base-homogeneous.meta_study.yaml")
    dosing_cfg = MetaDosingConfig.from_yaml(experiment_dir / "base-homogeneous.dosing.yaml")
    context_obs_cfg = ObservationsConfig.from_yaml(
        experiment_dir / "base-homogeneous.observations.yaml", section="context_observations"
    )

    # CALL THE SIMULATIONS
    output_path = tmp_path / "ensemble.json"
    studies, failure_rate = prepare_ensemble_of_simulations(
        meta_study_cfg,
        context_obs_cfg,
        dosing_cfg,
        number_of_samples=2,
        file_name=str(output_path),
    )

    assert len(studies) == 2
    assert 0.0 <= failure_rate <= 1.0
    for idx, study in enumerate(studies):
        assert study["meta_data"]["study_name"] == f"simulated_study_{idx}"
        assert "context" in study and isinstance(study["context"], list)

    assert output_path.exists()
    saved = json.loads(output_path.read_text())
    assert saved == studies


if __name__ == "__main__":
    test_simulation_with_duration()
