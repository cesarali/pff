
import torch

from pff.config_classes.data_config import (
    MetaDosingConfig,
    MetaDosingWithDurationConfig,
    MetaStudyConfig,
)
from pff.data.data_generation.compartment_models import (
    sample_dosing_configs,
    sample_dosing_with_duration_configs,
    sample_individual_configs,
    sample_study,
    sample_study_config,
    sample_study_with_duration,
)


def test_dosing_with_duration_refactoring():

    meta_study_config = MetaStudyConfig()
    study_config = sample_study_config(meta_study_config)
    individual_config_array = sample_individual_configs(study_config)

    meta_dosing_config = MetaDosingConfig(
        logdose_mean_range=[1.0, 1.0],
        logdose_std_range=[0.0, 0.0],
        num_individuals=study_config.num_individuals,
    )
    meta_dosing_with_duration_config = MetaDosingWithDurationConfig(
        logdose_mean_range=meta_dosing_config.logdose_mean_range,
        logdose_std_range=meta_dosing_config.logdose_std_range,
        num_individuals=meta_dosing_config.num_individuals,
        route_duration_weights={"oral": 0.0, "iv": 0.0},
    )

    # Generate time points
    time_points = torch.linspace(
        meta_study_config.time_start,
        meta_study_config.time_stop,
        meta_study_config.time_num_steps,
        dtype=torch.float32,
    )
    # Sample individual dosing configurations
    dosing_config_array = sample_dosing_configs(meta_dosing_config)
    dosing_with_duration_config_array = sample_dosing_with_duration_configs(meta_dosing_with_duration_config)

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Generate simulation data with old function
    full_simulation, full_simulation_times, dosing_amounts, dosing_route_types = sample_study(
        individual_config_array,
        dosing_config_array,
        time_points,
        meta_study_config.solver_method,
    )

    # Set same seed as above to generate the same ensembles
    torch.manual_seed(42)

    # Generate simulation data with new function supporting dosing with duration
    full_simulation_duration, full_simulation_times_duration, dosing_amounts_duration, dosing_route_types_duration = sample_study_with_duration(
        individual_config_array,
        dosing_with_duration_config_array,
        time_points,
        meta_study_config.solver_method,
    )
    # Compare results
    assert torch.allclose(full_simulation, full_simulation_duration)
    assert torch.allclose(full_simulation_times, full_simulation_times_duration)
    assert torch.allclose(dosing_amounts, dosing_amounts_duration)
    assert torch.allclose(dosing_route_types, dosing_route_types_duration)

if __name__ == "__main__":
    test_dosing_with_duration_refactoring()
