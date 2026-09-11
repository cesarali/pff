# pyright: reportAssignmentType=false
# compartment_models_management.py
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import torch
from torchtyping import TensorType
from tqdm.auto import tqdm

from pff.config_classes.data_config import (
    DosingConfig,
    DosingWithDurationConfig,
    MetaDosingConfig,
    MetaDosingWithDurationConfig,
    MetaStudyConfig,
    ObservationsConfig,
)
from pff.data.data_empirical.json_schema import StudyJSON
from pff.data.data_generation.compartment_models import (
    StudyConfig,
    StudyDosingConfig,
    derive_timescale_parameters,
    sample_dosing_arrays,
    sample_dosing_configs,
    sample_dosing_with_duration_configs,
    sample_individual_configs,
    sample_study_dosing_config,
    sample_study,
    sample_study_config,
    sample_study_with_duration,
)
from pff.data.data_generation.observations_classes import ObservationStrategyFactory

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pff.data.data_empirical.json_schema import IndividualJSON, StudyJSON
else:  # pragma: no cover - runtime fallback avoids heavy import cycle
    IndividualJSON = Dict[str, object]
    StudyJSON = Dict[str, object]


@dataclass
class SyntheticSampleExperimentConfig:
    """Reusable shared study-level state for sample-experiment generation."""

    study_config: StudyConfig
    time_scales: torch.Tensor
    time_points: torch.Tensor
    meta_dosing_config: MetaDosingConfig


@dataclass
class SampleExperimentSharedState:
    """Shared state reused across all sample-experiment dosing modes."""

    study_config: StudyConfig
    context_study_dosing_config: StudyDosingConfig
    solver_method: str
    time_scales: torch.Tensor
    time_points: torch.Tensor
    context_observation_strategy: object
    target_observation_strategy: object
    context: list[IndividualJSON]


_SAMPLE_EXPERIMENT_DOSING_MODES = {
    "repeated_dosing",
    "dosing_list",
    "diverse_dosing",
    "vpc_context",
}
_SAMPLE_EXPERIMENT_DOSING_LIST_GENERATION_MODES = {
    "dosing_from_samples",
    "dosing_from_range",
}
_SAMPLE_EXPERIMENT_MAX_RETRY_ATTEMPTS = 100


class _SampleExperimentInvalidSimulationError(RuntimeError):
    """Private error used to trigger whole-experiment replacement resampling."""

    def __init__(
        self,
        *,
        block_name: str,
        dosing_signature: str,
        debug_message: str = "",
    ) -> None:
        self.block_name = str(block_name)
        self.dosing_signature = str(dosing_signature)
        self.debug_message = str(debug_message)

        message = (
            f"Invalid {self.block_name} simulation encountered while building sample experiment. "
            f"dosing_signature={self.dosing_signature}"
        )
        if self.debug_message:
            message = f"{message} {self.debug_message}"
        super().__init__(message)


def is_valid_simulation(sim: torch.Tensor) -> bool:
    """Returns True if the simulation is numerically valid and all values are < 10."""
    return torch.isfinite(sim).all() and (sim >= 0).all() and (sim < 10).all()


def normalize_sample_experiment_dosing_mode(dosing_mode: str) -> str:
    """Validate and return the sample-experiment dosing mode."""

    if dosing_mode not in _SAMPLE_EXPERIMENT_DOSING_MODES:
        valid_modes = "', '".join(sorted(_SAMPLE_EXPERIMENT_DOSING_MODES))
        raise ValueError(f"dosing_mode must be one of '{valid_modes}'")
    return dosing_mode


def sample_dosing_configs_repeated_target(config: MetaDosingConfig, n_targets: int):
    """
    Generate dosing configs where all target individuals share the same
    dose and route.

    Parameters
    ----------
    config : MetaDosingConfig
        Meta dosing configuration (num_individuals field may be ignored).
    n_targets : int
        Number of target individuals to generate.

    Returns
    -------
    List[DosingConfig]
        Identical dosing configs repeated `n_targets` times.
    """
    if n_targets < 0:
        raise ValueError("n_targets must be non-negative")
    if n_targets == 0:
        return []

    study_dosing_config = sample_study_dosing_config(config)
    return sample_dosing_configs_repeated_target_from_study_dosing_config(
        study_dosing_config,
        n_targets,
    )


def sample_dosing_configs_repeated_target_from_study_dosing_config(
    study_dosing_config: StudyDosingConfig,
    n_targets: int,
) -> list[DosingConfig]:
    """Repeat one dosing draw sampled from a shared study-level dosing config."""

    if n_targets < 0:
        raise ValueError("n_targets must be non-negative")
    if n_targets == 0:
        return []

    # Sample exactly one target dosing realization from the shared study-level
    # distribution, then repeat it across all targets in the study.
    sampled_config = sample_dosing_arrays(study_dosing_config, 1)[0]
    return build_repeated_target_dosing_configs(
        n_targets,
        dose_value=float(sampled_config.dose),
        route=str(sampled_config.route),
        time=float(study_dosing_config.time),
    )


def build_repeated_target_dosing_configs(
    n_targets: int,
    *,
    dose_value: float,
    route: str,
    time: float,
) -> list[DosingConfig]:
    """Build identical target dosing configs for a fixed dose and route."""

    return [DosingConfig(dose=dose_value, route=route, time=time) for _ in range(n_targets)]


def sample_dosing_with_duration_configs_repeated_target(
    config: MetaDosingWithDurationConfig, n_targets: int
):
    """
    Generate dosing configs where all target individuals share the same
    dose and route.

    Parameters
    ----------
    config : MetaDosingWithDurationConfig
        Meta dosing configuration with duration (num_individuals field may be ignored).
    n_targets : int
        Number of target individuals to generate.

    Returns
    -------
    List[DosingConfig]
        Identical dosing configs repeated `n_targets` times.
    """
    # Choose one route for all targets
    route = np.random.choice(config.route_options, p=config.route_weights)

    # Handling the duration logic
    duration_weight = config.route_duration_weights[route]
    duration_range = np.random.uniform(*config.duration_range)
    duration = duration_weight * duration_range

    # Sample one dose (lognormal)
    logdose_mean = np.random.uniform(*config.logdose_mean_range)
    logdose_std = np.random.uniform(*config.logdose_std_range)
    dose_value = float(np.random.lognormal(logdose_mean, logdose_std))

    # Build identical configs
    dosing_configs = [
        DosingWithDurationConfig(dose=dose_value, route=route, time=config.time, duration=duration)
        for _ in range(n_targets)
    ]
    return dosing_configs


def sample_synthetic_sample_experiment_config(
    meta_study_config: MetaStudyConfig,
    meta_dosing_config: MetaDosingConfig,
) -> SyntheticSampleExperimentConfig:
    """Sample shared study-level state for a synthetic sample experiment."""

    if getattr(meta_study_config, "simple_mode", False):
        raise ValueError(
            "sample experiments are not implemented for simple_mode meta-study configs"
        )

    study_config = sample_study_config(meta_study_config)
    setattr(study_config, "solver_method", meta_study_config.solver_method)
    time_scales = derive_timescale_parameters(study_config, meta_study_config)
    time_points = torch.linspace(
        meta_study_config.time_start,
        meta_study_config.time_stop,
        meta_study_config.time_num_steps,
        dtype=torch.float32,
    )

    return SyntheticSampleExperimentConfig(
        study_config=study_config,
        time_scales=time_scales,
        time_points=time_points,
        meta_dosing_config=meta_dosing_config,
    )


def _generate_observation_pack(
    observation_strategy,
    *,
    full_simulation: torch.Tensor,
    full_simulation_times: torch.Tensor,
    time_scales: torch.Tensor,
):
    """Retry randomized observation generation until a non-empty pack is produced."""

    for _ in range(10):
        obs_pack = observation_strategy.generate(
            full_simulation=full_simulation,
            full_simulation_times=full_simulation_times,
            time_scales=time_scales,
        )
        if obs_pack[0] is not None:
            return obs_pack
    raise RuntimeError(
        "Unable to generate non-empty observations after 10 attempts for the sample experiment."
    )


def _serialize_simulation_block(
    *,
    full_simulation: torch.Tensor,
    full_simulation_times: torch.Tensor,
    dosing_config_array: list[DosingConfig],
    dosing_amounts: torch.Tensor,
    observation_strategy,
    time_scales: torch.Tensor,
    name_prefix: str,
) -> list[IndividualJSON]:
    """Serialize one simulation block into StudyJSON individuals."""

    obs_out, time_out, mask_out, rem_sim, rem_time, rem_mask, _ = _generate_observation_pack(
        observation_strategy,
        full_simulation=full_simulation,
        full_simulation_times=full_simulation_times,
        time_scales=time_scales,
    )

    individuals: list[IndividualJSON] = []
    num_individuals = full_simulation.shape[0]
    for ind_idx in range(num_individuals):
        # obs_row: [T_obs], obs_time_row: [T_obs], obs_mask_row: [T_obs]
        obs_mask_row = mask_out[ind_idx].to(torch.bool)
        observations = obs_out[ind_idx][obs_mask_row].tolist()
        observation_times = time_out[ind_idx][obs_mask_row].tolist()

        _ensure_strictly_increasing_observations(
            observation_times,
            observations,
            individual_id=f"{name_prefix}_{ind_idx}",
        )

        individual: IndividualJSON = {
            "name_id": f"{name_prefix}_{ind_idx}",
            "observations": observations,
            "observation_times": observation_times,
        }

        if rem_sim is not None and rem_time is not None and rem_mask is not None:
            rem_mask_row = rem_mask[ind_idx].to(torch.bool)
            if rem_mask_row.any():
                individual["remaining"] = rem_sim[ind_idx][rem_mask_row].tolist()
                individual["remaining_times"] = rem_time[ind_idx][rem_mask_row].tolist()

        dosing_cfg = dosing_config_array[ind_idx]
        dose = float(dosing_amounts[ind_idx].item())
        route = str(getattr(dosing_cfg, "route", ""))
        dosing_time = float(getattr(dosing_cfg, "time", 0.0))
        if dose or route:
            individual["dosing"] = [dose]
            individual["dosing_type"] = [route]
            individual["dosing_times"] = [dosing_time]
            individual["dosing_name"] = [route]

        individuals.append(individual)

    return individuals


def _lookup_simulation_values_at_reference_times(
    *,
    simulation_values: torch.Tensor,
    simulation_times: torch.Tensor,
    reference_times: list[float],
    individual_id: str,
    schedule_name: str,
    atol: float = 1.0e-5,
) -> list[float]:
    """Project one simulated trajectory onto an explicit reference schedule."""

    if not reference_times:
        return []

    matched_indices: list[int] = []
    for ref_time in reference_times:
        matches = torch.isclose(
            simulation_times,
            torch.tensor(
                float(ref_time),
                dtype=simulation_times.dtype,
                device=simulation_times.device,
            ),
            atol=atol,
            rtol=0.0,
        )
        if not bool(matches.any()):
            preview_count = min(5, int(simulation_times.numel()))
            time_preview = [float(x) for x in simulation_times[:preview_count].tolist()]
            raise ValueError(
                "Synthetic VPC reference schedules must align with the sampled simulation grid. "
                f"Could not match {schedule_name} time {float(ref_time):.6g} for {individual_id}. "
                f"Time preview={time_preview}."
            )
        matched_indices.append(int(torch.nonzero(matches, as_tuple=False)[0].item()))

    return [float(x) for x in simulation_values[matched_indices].tolist()]


def _serialize_simulation_block_on_reference_schedule(
    *,
    full_simulation: torch.Tensor,
    full_simulation_times: torch.Tensor,
    dosing_config_array: list[DosingConfig],
    dosing_amounts: torch.Tensor,
    reference_individuals: list[IndividualJSON],
    name_prefix: str,
) -> list[IndividualJSON]:
    """Serialize one simulation block on the exact schedule stored in ``reference_individuals``."""

    num_individuals = full_simulation.shape[0]
    if len(reference_individuals) != num_individuals:
        raise ValueError(
            "Reference individuals must match the simulated block size, got "
            f"{len(reference_individuals)} reference individuals for {num_individuals} "
            f"simulated individuals."
        )

    individuals: list[IndividualJSON] = []
    for ind_idx, reference_individual in enumerate(reference_individuals):
        simulation_row = full_simulation[ind_idx]  # [T]
        simulation_times_row = full_simulation_times[ind_idx]  # [T]
        name_id = str(reference_individual.get("name_id", f"{name_prefix}_{ind_idx}"))
        observation_times = [float(x) for x in reference_individual.get("observation_times", [])]
        observations = _lookup_simulation_values_at_reference_times(
            simulation_values=simulation_row,
            simulation_times=simulation_times_row,
            reference_times=observation_times,
            individual_id=name_id,
            schedule_name="observation",
        )
        _ensure_strictly_increasing_observations(
            observation_times,
            observations,
            individual_id=name_id,
        )

        individual: IndividualJSON = {
            "name_id": name_id,
            "observations": observations,
            "observation_times": observation_times,
        }

        remaining_times = [float(x) for x in reference_individual.get("remaining_times", [])]
        if remaining_times:
            individual["remaining_times"] = remaining_times
            individual["remaining"] = _lookup_simulation_values_at_reference_times(
                simulation_values=simulation_row,
                simulation_times=simulation_times_row,
                reference_times=remaining_times,
                individual_id=name_id,
                schedule_name="remaining",
            )

        dosing_cfg = dosing_config_array[ind_idx]
        dose = float(dosing_amounts[ind_idx].item())
        route = str(getattr(dosing_cfg, "route", ""))
        dosing_time = float(getattr(dosing_cfg, "time", 0.0))
        if dose or route:
            individual["dosing"] = [dose]
            individual["dosing_type"] = [route]
            individual["dosing_times"] = [dosing_time]
            individual["dosing_name"] = [route]

        individuals.append(individual)

    return individuals


def _target_dosing_signature(dosing_config_array: list[DosingConfig]) -> str:
    """Return a stable string signature describing one target dosing realization."""

    if not dosing_config_array:
        return "none"

    parts = []
    for idx, dosing_cfg in enumerate(dosing_config_array):
        route = str(getattr(dosing_cfg, "route", ""))
        dose = float(getattr(dosing_cfg, "dose", 0.0))
        dosing_time = float(getattr(dosing_cfg, "time", 0.0))
        parts.append(f"target_{idx}:route={route}|dose={dose:.12g}|time={dosing_time:.12g}")
    return ";".join(parts)


def _simulate_and_serialize_sample_experiment_block(
    *,
    block_name: str,
    individual_configs,
    dosing_config_array: list[DosingConfig],
    time_points: torch.Tensor,
    solver_method: str,
    observation_strategy,
    time_scales: torch.Tensor,
) -> list[IndividualJSON]:
    """Simulate one study block and serialize it to study JSON individuals."""

    full_simulation, full_simulation_times, dosing_amounts = _simulate_sample_experiment_block(
        block_name=block_name,
        individual_configs=individual_configs,
        dosing_config_array=dosing_config_array,
        time_points=time_points,
        solver_method=solver_method,
    )
    return _serialize_simulation_block(
        full_simulation=full_simulation,
        full_simulation_times=full_simulation_times,
        dosing_config_array=dosing_config_array,
        dosing_amounts=dosing_amounts,
        observation_strategy=observation_strategy,
        time_scales=time_scales,
        name_prefix=block_name,
    )


def _simulate_sample_experiment_block(
    *,
    block_name: str,
    individual_configs,
    dosing_config_array: list[DosingConfig],
    time_points: torch.Tensor,
    solver_method: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulate one sample-experiment block and return the raw trajectories."""

    full_simulation, full_simulation_times, dosing_amounts, _ = sample_study(
        individual_configs,
        dosing_config_array,
        time_points,
        solver_method,
    )
    if not is_valid_simulation(full_simulation):
        dosing_signature = _target_dosing_signature(dosing_config_array)
        raise _SampleExperimentInvalidSimulationError(
            block_name=block_name,
            dosing_signature=dosing_signature,
        )
    return full_simulation, full_simulation_times, dosing_amounts


def _prepare_sample_experiment_shared_state(
    synthetic_config: SyntheticSampleExperimentConfig,
    context_observation_config: ObservationsConfig,
    target_observation_config: ObservationsConfig,
) -> SampleExperimentSharedState:
    """Prepare the shared context and target-independent state once per experiment.

    Sample experiments preserve the native context observation strategy while
    serializing targets on a caller-provided deterministic target schedule so
    downstream tasks such as signature-kernel MMD receive one shared target
    grid without coupling that choice to the regular training datamodule path.
    """

    study_config = synthetic_config.study_config
    solver_method = str(getattr(study_config, "solver_method", "rk4"))
    time_scales = synthetic_config.time_scales
    time_points = synthetic_config.time_points
    observation_meta_config = replace(
        MetaStudyConfig(),
        time_start=study_config.time_start,
        time_stop=study_config.time_stop,
        time_num_steps=int(time_points.shape[0]),
        solver_method=solver_method,
    )
    context_observation_strategy = ObservationStrategyFactory.from_config(
        context_observation_config,
        observation_meta_config,
    )
    target_observation_strategy = ObservationStrategyFactory.from_config(
        target_observation_config,
        observation_meta_config,
    )

    context_individual_configs = sample_individual_configs(study_config)
    context_study_dosing_config = sample_study_dosing_config(synthetic_config.meta_dosing_config)
    context_dosing_configs = sample_dosing_arrays(
        context_study_dosing_config,
        study_config.num_individuals,
    )
    context = _simulate_and_serialize_sample_experiment_block(
        block_name="context",
        individual_configs=context_individual_configs,
        dosing_config_array=context_dosing_configs,
        time_points=time_points,
        solver_method=solver_method,
        observation_strategy=context_observation_strategy,
        time_scales=time_scales,
    )

    return SampleExperimentSharedState(
        study_config=study_config,
        context_study_dosing_config=context_study_dosing_config,
        solver_method=solver_method,
        time_scales=time_scales,
        time_points=time_points,
        context_observation_strategy=context_observation_strategy,
        target_observation_strategy=target_observation_strategy,
        context=context,
    )


def _build_sample_experiment_study(
    shared_state: SampleExperimentSharedState,
    *,
    target_individual_configs: list,
    target_dosing_configs: list[DosingConfig],
    dosing_mode: str,
    dosing_idx: int,
) -> StudyJSON:
    """Build one sample-experiment study for a fully specified target dosing realization."""

    target = _simulate_and_serialize_sample_experiment_block(
        block_name="target",
        individual_configs=target_individual_configs,
        dosing_config_array=target_dosing_configs,
        time_points=shared_state.time_points,
        solver_method=shared_state.solver_method,
        observation_strategy=shared_state.target_observation_strategy,
        time_scales=shared_state.time_scales,
    )

    return {
        "context": shared_state.context,
        "target": target,
        "meta_data": {
            "study_name": (
                f"sample_experiment_{shared_state.study_config.drug_id}_{dosing_mode}_{dosing_idx}"
            ),
            "substance_name": str(shared_state.study_config.drug_id),
            "dosing_mode": dosing_mode,
            "dosing_realization_index": str(dosing_idx),
        },
    }


def _build_sample_experiment_vpc_context_study(
    shared_state: SampleExperimentSharedState,
    *,
    observed_individual_configs: list,
    observed_dosing_configs: list[DosingConfig],
    dosing_idx: int,
) -> StudyJSON:
    """Build one context-only synthetic study used as observed input for VPC."""

    observed_context = _simulate_and_serialize_sample_experiment_block(
        block_name="context",
        individual_configs=observed_individual_configs,
        dosing_config_array=observed_dosing_configs,
        time_points=shared_state.time_points,
        solver_method=shared_state.solver_method,
        observation_strategy=shared_state.target_observation_strategy,
        time_scales=shared_state.time_scales,
    )

    return {
        "context": observed_context,
        "target": [],
        "meta_data": {
            "study_name": (
                f"sample_experiment_{shared_state.study_config.drug_id}_vpc_context_{dosing_idx}"
            ),
            "substance_name": str(shared_state.study_config.drug_id),
            "dosing_mode": "vpc_context",
            "dosing_realization_index": str(dosing_idx),
        },
    }


def _build_sample_experiment_studies_repeated_dosing(
    shared_state: SampleExperimentSharedState,
    meta_dosing_config: MetaDosingConfig,
    *,
    n_targets: int,
    n_dosings: int,
) -> list[StudyJSON]:
    """Build sample-experiment studies with one repeated target dosing per study."""

    target_individual_configs = sample_individual_configs(shared_state.study_config, n=n_targets)
    studies: list[StudyJSON] = []

    for dosing_idx in range(n_dosings):
        target_dosing_configs = sample_dosing_configs_repeated_target_from_study_dosing_config(
            sample_study_dosing_config(meta_dosing_config),
            n_targets,
        )
        studies.append(
            _build_sample_experiment_study(
                shared_state,
                target_individual_configs=target_individual_configs,
                target_dosing_configs=target_dosing_configs,
                dosing_mode="repeated_dosing",
                dosing_idx=dosing_idx,
            )
        )

    return studies


def _build_sample_experiment_studies_dosing_list(
    shared_state: SampleExperimentSharedState,
    _meta_dosing_config: MetaDosingConfig,
    *,
    n_targets: int,
    n_dosings: int,
    dosing_list_generation: str,
    logdose_range: Optional[Tuple[float, float]],
) -> list[StudyJSON]:
    """Build sample-experiment studies for dosing-list mode.

    ``dosing_from_samples`` reuses the same shared study-level dosing
    distribution stored in ``shared_state`` and draws one repeated-target
    dosing realization per returned study. ``dosing_from_range`` instead
    builds a deterministic repeated-target dose grid while keeping the shared
    route/time realization already stored in ``shared_state``.
    """

    if dosing_list_generation not in _SAMPLE_EXPERIMENT_DOSING_LIST_GENERATION_MODES:
        raise ValueError(
            "dosing_list_generation must be one of {'dosing_from_samples', 'dosing_from_range'}"
        )

    target_individual_configs = sample_individual_configs(shared_state.study_config, n=n_targets)
    studies: list[StudyJSON] = []

    if dosing_list_generation == "dosing_from_range" and n_dosings > 0:
        if logdose_range is None:
            raise ValueError(
                "logdose_range must be provided when dosing_list_generation='dosing_from_range'"
            )
        if len(logdose_range) != 2:
            raise ValueError("logdose_range must contain exactly two values")
        if float(logdose_range[0]) > float(logdose_range[1]):
            raise ValueError("logdose_range must satisfy min_logdose <= max_logdose")

        target_study_dosing_config = shared_state.context_study_dosing_config
        dosing_list_route = str(sample_dosing_arrays(target_study_dosing_config, 1)[0].route)
        dosing_list_grid = np.linspace(
            float(logdose_range[0]),
            float(logdose_range[1]),
            num=n_dosings,
        )

        for dosing_idx, logdose in enumerate(dosing_list_grid):
            target_dosing_configs = build_repeated_target_dosing_configs(
                n_targets,
                dose_value=float(np.exp(logdose)),
                route=dosing_list_route,
                time=float(target_study_dosing_config.time),
            )
            studies.append(
                _build_sample_experiment_study(
                    shared_state,
                    target_individual_configs=target_individual_configs,
                    target_dosing_configs=target_dosing_configs,
                    dosing_mode="dosing_list",
                    dosing_idx=dosing_idx,
                )
            )

        return studies

    for dosing_idx in range(n_dosings):
        target_dosing_configs = sample_dosing_configs_repeated_target_from_study_dosing_config(
            shared_state.context_study_dosing_config,
            n_targets,
        )
        studies.append(
            _build_sample_experiment_study(
                shared_state,
                target_individual_configs=target_individual_configs,
                target_dosing_configs=target_dosing_configs,
                dosing_mode="dosing_list",
                dosing_idx=dosing_idx,
            )
        )

    return studies


def _build_sample_experiment_studies_diverse_dosing(
    shared_state: SampleExperimentSharedState,
    _meta_dosing_config: MetaDosingConfig,
    *,
    n_targets: int,
    n_dosings: int,
) -> list[StudyJSON]:
    """Build sample-experiment studies where context and target share one meta dosing config."""

    target_individual_configs = sample_individual_configs(shared_state.study_config, n=n_targets)
    studies: list[StudyJSON] = []

    for dosing_idx in range(n_dosings):
        target_dosing_configs = sample_dosing_arrays(
            shared_state.context_study_dosing_config,
            n_targets,
        )
        studies.append(
            _build_sample_experiment_study(
                shared_state,
                target_individual_configs=target_individual_configs,
                target_dosing_configs=target_dosing_configs,
                dosing_mode="diverse_dosing",
                dosing_idx=dosing_idx,
            )
        )

    return studies


def _build_sample_experiment_studies_vpc_context(
    shared_state: SampleExperimentSharedState,
    _meta_dosing_config: MetaDosingConfig,
    *,
    n_targets: int,
    n_dosings: int,
) -> list[StudyJSON]:
    """Build context-only synthetic studies used as observed VPC inputs."""

    observed_individual_configs = sample_individual_configs(shared_state.study_config, n=n_targets)
    studies: list[StudyJSON] = []

    for dosing_idx in range(n_dosings):
        observed_dosing_configs = sample_dosing_arrays(
            shared_state.context_study_dosing_config,
            n_targets,
        )
        studies.append(
            _build_sample_experiment_vpc_context_study(
                shared_state,
                observed_individual_configs=observed_individual_configs,
                observed_dosing_configs=observed_dosing_configs,
                dosing_idx=dosing_idx,
            )
        )

    return studies


def _build_synthetic_vpc_case_single_attempt(
    synthetic_config: SyntheticSampleExperimentConfig,
    observation_config: ObservationsConfig,
    *,
    case_index: int,
    n_observed_individuals: int,
    sample_size: int,
) -> tuple[StudyJSON, list[StudyJSON]]:
    """Build one native synthetic VPC case from one sampled study-level state.

    The observed study and all replicate studies share one sampled study-level
    state and one realized dosing layout. Replicates therefore preserve the
    observed study's realized dosing layout and explicit observation schedule
    while resampling only the individual latent configurations.
    """

    study_config = synthetic_config.study_config
    solver_method = str(getattr(study_config, "solver_method", "rk4"))
    time_points = synthetic_config.time_points
    time_scales = synthetic_config.time_scales
    observation_meta_config = replace(
        MetaStudyConfig(),
        time_start=study_config.time_start,
        time_stop=study_config.time_stop,
        time_num_steps=int(time_points.shape[0]),
        solver_method=solver_method,
    )
    observation_strategy = ObservationStrategyFactory.from_config(
        observation_config,
        observation_meta_config,
    )

    observed_individual_configs = sample_individual_configs(
        study_config,
        n=n_observed_individuals,
    )
    observed_study_dosing_config = sample_study_dosing_config(synthetic_config.meta_dosing_config)
    observed_dosing_configs = sample_dosing_arrays(
        observed_study_dosing_config,
        n_observed_individuals,
    )
    observed_context = _simulate_and_serialize_sample_experiment_block(
        block_name="context",
        individual_configs=observed_individual_configs,
        dosing_config_array=observed_dosing_configs,
        time_points=time_points,
        solver_method=solver_method,
        observation_strategy=observation_strategy,
        time_scales=time_scales,
    )
    observed_study: StudyJSON = {
        "context": observed_context,
        "target": [],
        "meta_data": {
            "study_name": f"synthetic_vpc_{study_config.drug_id}_case_{case_index}",
            "substance_name": str(study_config.drug_id),
            "synthetic_vpc_case_index": str(case_index),
        },
    }

    simulated_replicates: list[StudyJSON] = []
    reference_context = list(observed_study.get("context", []))
    replicate_iterator = tqdm(
        range(sample_size),
        desc=f"Synthetic VPC replicas for case {case_index + 1}",
        leave=False,
        dynamic_ncols=True,
    )
    for replicate_index in replicate_iterator:
        # Preserve the observed study's realized dosing layout and explicit
        # VPC schedule; only the individual latent configurations are resampled.
        replicate_individual_configs = sample_individual_configs(
            study_config,
            n=n_observed_individuals,
        )
        replicate_full_simulation, replicate_full_times, replicate_dosing_amounts = (
            _simulate_sample_experiment_block(
                block_name="context",
                individual_configs=replicate_individual_configs,
                dosing_config_array=observed_dosing_configs,
                time_points=time_points,
                solver_method=solver_method,
            )
        )
        replicate_context = _serialize_simulation_block_on_reference_schedule(
            full_simulation=replicate_full_simulation,
            full_simulation_times=replicate_full_times,
            dosing_config_array=observed_dosing_configs,
            dosing_amounts=replicate_dosing_amounts,
            reference_individuals=reference_context,
            name_prefix="context",
        )
        simulated_replicates.append(
            {
                "context": replicate_context,
                "target": [],
                "meta_data": {
                    **dict(observed_study.get("meta_data", {})),
                    "synthetic_vpc_replicate_index": str(replicate_index),
                },
            }
        )

    return observed_study, simulated_replicates


# ──────────────────────────────────────────────────────────────
# NEW: split where *all* individuals are target
# ──────────────────────────────────────────────────────────────
def split_context_only(
    full_simulation: torch.Tensor,
    full_simulation_times: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Return all individuals as context, no targets."""
    num_individuals = full_simulation.shape[0]
    context_indices = list(range(num_individuals))
    return full_simulation, full_simulation_times, context_indices


def split_simulations_repeated_target(
    full_simulation: torch.Tensor,
    full_simulation_times: torch.Tensor,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[int],
]:
    """
    Variant of split_simulations where **all individuals are in the target set**
    and no context individuals are returned.

    Parameters
    ----------
    full_simulation : torch.Tensor [N, T]
    full_simulation_times : torch.Tensor [N, T]

    Returns
    -------
    context_simulation : None
    context_simulation_times : None
    target_simulation : torch.Tensor [N, T]
    target_simulation_times : torch.Tensor [N, T]
    context_indices : []
    target_indices : list[int] = [0,...,N-1]
    """
    num_individuals = full_simulation.shape[0]
    target_indices = list(range(num_individuals))

    return (
        None,
        None,
        full_simulation,
        full_simulation_times,
        [],
        target_indices,
    )


def _generate_full_simulation(
    meta_study_config: MetaStudyConfig,
    meta_dosing_config: MetaDosingConfig,
    *,
    retry_on_invalid: bool = True,
    idx: int = 0,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    StudyConfig,
    list[DosingConfig],
    int,
]:
    """Internal helper returning the raw tensors alongside sampling metadata."""
    study_config = sample_study_config(meta_study_config)
    indiv_config_array = sample_individual_configs(study_config)
    time_scales = derive_timescale_parameters(study_config, meta_study_config)

    time_points = torch.linspace(
        meta_study_config.time_start,
        meta_study_config.time_stop,
        meta_study_config.time_num_steps,
        dtype=torch.float32,
    )

    local_meta_dosing = replace(meta_dosing_config, num_individuals=study_config.num_individuals)
    dosing_config_array = sample_dosing_configs(local_meta_dosing)

    full_sim, full_times, dosing_amounts, dosing_routes = sample_study(
        indiv_config_array,
        dosing_config_array,
        time_points,
        meta_study_config.solver_method,
    )

    if not is_valid_simulation(full_sim):
        attempt_number = idx + 1
        if attempt_number > 5:
            logger.warning(
                "Invalid simulation encountered during attempt %d (recursion depth %d); retry_on_invalid=%s.",
                attempt_number,
                idx,
                retry_on_invalid,
            )
        if retry_on_invalid:
            (
                full_sim,
                full_times,
                dosing_amounts,
                dosing_routes,
                time_points,
                time_scales,
                study_config,
                dosing_config_array,
                downstream_failures,
            ) = _generate_full_simulation(
                meta_study_config,
                meta_dosing_config,
                retry_on_invalid=retry_on_invalid,
                idx=idx + 1,
            )
            return (
                full_sim,
                full_times,
                dosing_amounts,
                dosing_routes,
                time_points,
                time_scales,
                study_config,
                dosing_config_array,
                downstream_failures + 1,
            )
        raise RuntimeError("Invalid simulation")

    return (
        full_sim,
        full_times,
        dosing_amounts,
        dosing_routes,
        time_points,
        time_scales,
        study_config,
        dosing_config_array,
        0,
    )


def _generate_full_simulation_with_duration(
    meta_study_config: MetaStudyConfig,
    meta_dosing_config: MetaDosingWithDurationConfig,
    *,
    retry_on_invalid: bool = True,
    idx: int = 0,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    StudyConfig,
    list[DosingConfig],
    int,
]:
    """
    Internal helper returning the raw tensors alongside sampling metadata.
    This is a parallel implementation to `_generate_full_simulation` that supports
    dosing with duration. Once validated, the two can be merged.
    """
    study_config = sample_study_config(meta_study_config)
    indiv_config_array = sample_individual_configs(study_config)
    time_scales = derive_timescale_parameters(study_config, meta_study_config)

    time_points = torch.linspace(
        meta_study_config.time_start,
        meta_study_config.time_stop,
        meta_study_config.time_num_steps,
        dtype=torch.float32,
    )

    local_meta_dosing = replace(meta_dosing_config, num_individuals=study_config.num_individuals)
    dosing_config_array = sample_dosing_with_duration_configs(local_meta_dosing)

    full_sim, full_times, dosing_amounts, dosing_routes = sample_study_with_duration(
        indiv_config_array,
        dosing_config_array,
        time_points,
        meta_study_config.solver_method,
    )

    if not is_valid_simulation(full_sim):
        attempt_number = idx + 1
        if attempt_number > 5:
            logger.warning(
                "Invalid simulation encountered during attempt %d (recursion depth %d); retry_on_invalid=%s.",
                attempt_number,
                idx,
                retry_on_invalid,
            )
        if retry_on_invalid:
            (
                full_sim,
                full_times,
                dosing_amounts,
                dosing_routes,
                time_points,
                time_scales,
                study_config,
                dosing_config_array,
                downstream_failures,
            ) = _generate_full_simulation_with_duration(
                meta_study_config,
                meta_dosing_config,
                retry_on_invalid=retry_on_invalid,
                idx=idx + 1,
            )
            return (
                full_sim,
                full_times,
                dosing_amounts,
                dosing_routes,
                time_points,
                time_scales,
                study_config,
                dosing_config_array,
                downstream_failures + 1,
            )
        raise RuntimeError("Invalid simulation")

    return (
        full_sim,
        full_times,
        dosing_amounts,
        dosing_routes,
        time_points,
        time_scales,
        study_config,
        dosing_config_array,
        0,
    )


def _generate_simple_exp_simulation(
    meta_study_config,
) -> Tuple[
    TensorType["N", "T"],  # full_simulation
    TensorType["N", "T"],  # full_simulation_times
    TensorType["N"],  # dosing_amounts
    TensorType["N"],  # dosing_route_types
    TensorType["T"],  # time_points
    TensorType[2],  # time_scales [tmax, t12]
]:
    """
    Minimal synthetic PK-like simulator.

    Changes:
      - Samples a single per-RUN decay rate k ~ U(decay_rate_range) and uses it for all individuals.
      - Uses only band_scale_range, baseline_range, and (new) decay_rate_range.

    Derivations per RUN:
      baseline_run   ~ U(baseline_range)
      band_scale_run ~ U(band_scale_range)
      decay_rate     ~ U(decay_rate_range)
      intercept_mean = 1.0 + baseline_run
      intercept_std  = 0.5 * band_scale_run
    """

    # ---------------------------
    # Basic hyperparameters
    # ---------------------------
    N: int = getattr(meta_study_config, "num_individuals", 16)
    Tn: int = getattr(meta_study_config, "time_num_steps", 40)
    t_min: float = getattr(meta_study_config, "time_start", 0.0)
    t_max: float = getattr(meta_study_config, "time_stop", 24.0)

    band_scale_range = getattr(meta_study_config, "band_scale_range", (0.1, 0.3))
    baseline_range = getattr(meta_study_config, "baseline_range", (0.0, 0.1))
    decay_rate_range = getattr(meta_study_config, "decay_rate_range", (0.3, 0.6))  # NEW

    # ---------------------------
    # Per-RUN draws (no seeds)
    # ---------------------------
    def _urun(lo, hi):  # uniform helper
        return (torch.rand(1) * (hi - lo) + lo).item()

    band_scale_run = _urun(*band_scale_range)
    baseline_run = _urun(*baseline_range)
    decay_rate_k = _urun(*decay_rate_range)  # shared across all individuals this run

    intercept_mean = 1.0 + baseline_run
    intercept_std = 0.5 * band_scale_run

    # ---------------------------
    # Time grid & single-exp shape
    # ---------------------------
    t: TensorType["T", 1] = torch.linspace(t_min, t_max, Tn).unsqueeze(-1)  # [T,1]
    f_t: TensorType["T", 1] = torch.exp(-decay_rate_k * t)  # f_t(0)=1, shared shape

    # ---------------------------
    # Per-individual intercepts
    # ---------------------------
    intercepts: TensorType["N", 1, 1] = torch.normal(
        mean=float(intercept_mean),
        std=float(intercept_std),
        size=(N, 1, 1),
    ).clamp_min(0.0)

    # Build samples: scaled shape + shared run baseline.
    samples: TensorType["N", "T", 1] = intercepts * f_t.unsqueeze(0) + baseline_run
    samples = samples.clamp_min(0.0)  # numerical safety

    # ---------------------------
    # Dummy dosing / time scales
    # ---------------------------
    dosing_amounts: TensorType["N"] = torch.zeros(N)
    dosing_routes: TensorType["N"] = torch.zeros(N)
    duration = float(t_max - t_min)
    tmax = 0.3 * duration
    t12 = 0.75 * duration
    time_scales: TensorType[2] = torch.tensor([tmax, t12], dtype=torch.float32)

    # ---------------------------
    # Construct outputs
    # ---------------------------
    full_sim = samples.squeeze(-1)  # [N, T]
    full_sim_times = t.expand(N, -1, -1).squeeze(-1)  # [N, T]
    time_points = t.squeeze(-1)  # [T]

    return (
        full_sim,
        full_sim_times,
        dosing_amounts,
        dosing_routes,
        time_points,
        time_scales,
    )


def _generate_pulse_simulation(
    meta_study_config,
) -> Tuple[
    TensorType["N", "T"],  # full_simulation
    TensorType["N", "T"],  # full_simulation_times
    TensorType["N"],  # dosing_amounts
    TensorType["N"],  # dosing_route_types
    TensorType["T"],  # time_points
    TensorType[2],  # time_scales [t_peak, t_half_tail]
]:
    """
    Pulse-like PK-style simulator (rise -> peak -> decay).

    Config used (all optional with safe defaults):
      - num_individuals (int)
      - time_start, time_stop, time_num_steps
      - band_scale_range: (lo, hi)   # controls intercept std via 0.5 * band_scale_run
      - baseline_range:   (lo, hi)   # per-run vertical offset added to all traces
      - decay_rate_range: (lo, hi)   # per-run tail rate r; larger r => faster decay

    Construction (per RUN):
      duration = time_stop - time_start
      t_peak   = 0.30 * duration
      r        ~ U(decay_rate_range)
      beta     = 1 / r
      alpha    = 1 + r * t_peak              # => peak near t_peak for Gamma(alpha, beta)
      f(t)     = t^(alpha-1) * exp(-t/beta)  # normalized so max(f)=1

      baseline_run   ~ U(baseline_range)
      band_scale_run ~ U(band_scale_range)
      intercept_mean = 1.0 + baseline_run
      intercept_std  = 0.5 * band_scale_run

    Per INDIVIDUAL:
      intercept_i ~ Normal(intercept_mean, intercept_std), clamped to >= 0

    Output:
      samples_i(t) = intercept_i * f_norm(t) + baseline_run, clamped to >= 0
    """

    # ---------------------------
    # Basics
    # ---------------------------
    N: int = getattr(meta_study_config, "num_individuals", 16)
    Tn: int = getattr(meta_study_config, "time_num_steps", 40)
    t_min: float = getattr(meta_study_config, "time_start", 0.0)
    t_max: float = getattr(meta_study_config, "time_stop", 24.0)

    band_scale_range = getattr(meta_study_config, "band_scale_range", (0.1, 0.3))
    baseline_range = getattr(meta_study_config, "baseline_range", (0.0, 0.1))
    decay_rate_range = getattr(meta_study_config, "decay_rate_range", (0.3, 0.6))

    # ---------------------------
    # Per-RUN draws (no seeds)
    # ---------------------------
    def _urun(lo, hi):
        return (torch.rand(1) * (hi - lo) + lo).item()

    band_scale_run = _urun(*band_scale_range)
    baseline_run = _urun(*baseline_range)
    r_tail = _urun(*decay_rate_range)  # shared by all individuals this run

    duration = float(t_max - t_min)
    t_peak = 0.30 * duration  # desired peak position
    beta = 1.0 / max(r_tail, 1e-6)  # tail scale
    alpha = 1.0 + r_tail * t_peak  # ensures peak near t_peak (alpha>1)

    # Guardrails: make sure alpha > 1 for a proper rise-then-decay
    if alpha <= 1.05:
        alpha = 1.05

    # ---------------------------
    # Time grid & Gamma-shaped pulse
    # ---------------------------
    t: TensorType["T"] = torch.linspace(t_min, t_max, Tn)  # [T]
    t_shift = t - t_min  # start at 0
    # Gamma shape (unnormalized). For t=0, t^(alpha-1) is 0 if alpha>1.
    f_t = (t_shift.clamp_min(0.0) ** (alpha - 1.0)) * torch.exp(-t_shift / beta)

    # Normalize to max=1 so intercept controls amplitude
    f_max = torch.amax(f_t).clamp_min(1e-12)
    f_t = f_t / f_max  # [T]

    # ---------------------------
    # Per-individual intercepts
    # ---------------------------
    intercept_mean = 1.0 + baseline_run
    intercept_std = 0.5 * band_scale_run

    intercepts: TensorType["N", 1] = torch.normal(
        mean=float(intercept_mean),
        std=float(intercept_std),
        size=(N, 1),
    ).clamp_min(0.0)

    # Samples: scale by intercept, add per-run baseline
    samples: TensorType["N", "T"] = (intercepts * f_t.unsqueeze(0)) + baseline_run
    samples = samples.clamp_min(0.0)

    # ---------------------------
    # Dummy dosing / time scales
    # ---------------------------
    dosing_amounts: TensorType["N"] = torch.zeros(N)
    dosing_routes: TensorType["N"] = torch.zeros(N)

    # Report t_peak and an approximate tail half-life (after the peak)
    t_half_tail = t_peak + (torch.log(torch.tensor(2.0)) / max(r_tail, 1e-6)).item()
    time_scales: TensorType[2] = torch.tensor([t_peak, t_half_tail], dtype=torch.float32)

    # ---------------------------
    # Construct outputs
    # ---------------------------
    full_sim = samples  # [N, T]
    full_sim_times: TensorType = t.unsqueeze(0).expand(N, -1)  # [N, T]
    time_points = t  # [T]

    return (
        full_sim,
        full_sim_times,
        dosing_amounts,
        dosing_routes,
        time_points,
        time_scales,
    )


def _generate_simple_simulation(
    meta_study_config,
) -> Tuple[
    TensorType["N", "T"],
    TensorType["N", "T"],
    TensorType["N"],
    TensorType["N"],
    TensorType["T"],
    TensorType[2],
]:
    """
    Dispatcher that mixes two generators:
      - with probability p1: _generate_simple_exp_simulation(...)
      - with probability 1 - p1: _generate_pulse_simulation(...)

    Config:
      - p1 (float in [0,1]), default 0.5
    """
    p1 = float(getattr(meta_study_config, "p1", 0.5))
    # clamp to [0,1]
    p1 = 0.0 if p1 < 0.0 else (1.0 if p1 > 1.0 else p1)

    if torch.rand(1).item() < p1:
        return _generate_simple_exp_simulation(meta_study_config)
    else:
        return _generate_pulse_simulation(meta_study_config)


def prepare_full_simulation(
    meta_study_config,
    meta_dosing_config,
    *,
    retry_on_invalid: bool = True,
    idx: int = 0,
) -> Tuple[
    TensorType["N", "T", 1],
    TensorType["N", "T"],
    TensorType["N"],
    TensorType["N"],
    TensorType["T"],
    TensorType[2],
]:
    """
    Generate a full INDIVIDUAL study simulation (before context/target split).

    This bundles the common steps shared across all dataset generators.
    If `meta_study_config.simple_mode=True`, uses `_generate_simple_simulation`.
    """

    if getattr(meta_study_config, "simple_mode", False):
        return _generate_simple_simulation(meta_study_config)

    (
        full_sim,
        full_times,
        dosing_amounts,
        dosing_routes,
        time_points,
        time_scales,
        _,
        _,
        _,
    ) = _generate_full_simulation(
        meta_study_config,
        meta_dosing_config,
        retry_on_invalid=retry_on_invalid,
        idx=idx,
    )

    return full_sim, full_times, dosing_amounts, dosing_routes, time_points, time_scales


def prepare_full_simulation_with_duration(
    meta_study_config,
    meta_dosing_config,
    *,
    retry_on_invalid: bool = True,
    idx: int = 0,
) -> Tuple[
    TensorType["N", "T", 1],
    TensorType["N", "T"],
    TensorType["N"],
    TensorType["N"],
    TensorType["T"],
    TensorType[2],
]:
    """
    Generate a full INDIVIDUAL study simulation (before context/target split).

    This bundles the common steps shared across all dataset generators.
    If `meta_study_config.simple_mode=True`, uses `_generate_simple_simulation`.

    This is a parallel implementation to `prepare_full_simulation` that supports
    dosing with duration. Once validated, the two can be merged.
    """

    if getattr(meta_study_config, "simple_mode", False):
        return _generate_simple_simulation(meta_study_config)

    (
        full_sim,
        full_times,
        dosing_amounts,
        dosing_routes,
        time_points,
        time_scales,
        _,
        _,
        _,
    ) = _generate_full_simulation_with_duration(
        meta_study_config,
        meta_dosing_config,
        retry_on_invalid=retry_on_invalid,
        idx=idx,
    )

    return full_sim, full_times, dosing_amounts, dosing_routes, time_points, time_scales


def _ensure_strictly_increasing_observations(
    obs_times: list[float], obs_vals: list[list[float]], *, individual_id: str
) -> None:
    """Validate that the provided observation times are strictly increasing.

    Parameters
    ----------
    obs_times:
        Sequence of observation timestamps extracted from the simulator.
    obs_vals:
        Sequence of observation values sampled at ``obs_times``.
    individual_id:
        Identifier of the individual being validated. Included in the
        diagnostic error message to simplify debugging when duplicates are
        detected in batched runs.
    """

    if len(obs_times) != len(obs_vals):
        raise ValueError(
            "Observation times must be sorted and match the number of observations. "
            f"Received lengths times={len(obs_times)} and values={len(obs_vals)} for "
            f"{individual_id}. Observations={obs_vals}, times={obs_times}."
        )

    for idx_time in range(len(obs_times) - 1):
        if obs_times[idx_time] >= obs_times[idx_time + 1]:
            raise ValueError(
                "Observation times must be sorted and match the number of observations. "
                f"Detected non-increasing times for {individual_id} at position {idx_time}. "
                f"Observations={obs_vals}, times={obs_times}."
            )


def _build_sample_experiment_studies_single_attempt(
    synthetic_config: SyntheticSampleExperimentConfig,
    context_observation_config: ObservationsConfig,
    target_observation_config: ObservationsConfig,
    n_targets: int,
    n_dosings: int,
    dosing_mode: str,
    dosing_list_generation: str = "dosing_from_samples",
    logdose_range: Optional[Tuple[float, float]] = None,
) -> list[StudyJSON]:
    """Build sample-experiment studies from one sampled study state attempt.

    Parameters
    ----------
    synthetic_config:
        Shared synthetic study state sampled once for the whole experiment.
    context_observation_config:
        Observation sampling strategy used to serialize the shared context
        individuals.
    target_observation_config:
        Observation sampling strategy used to serialize target individuals for
        the synthetic sample experiment.
    n_targets:
        Number of target individuals in each returned study.
    n_dosings:
        Number of returned study JSONs / target dosing realizations. All
        returned studies reuse the same shared context, while target dosing is
        resampled according to the selected dosing mode.
    dosing_mode:
        Supported modes are ``repeated_dosing``, ``dosing_list``,
        ``diverse_dosing``, and ``vpc_context``.
    dosing_list_generation:
        Strategy used only when ``dosing_mode="dosing_list"`` and ignored by
        the other dosing modes:
        ``dosing_from_samples`` reuses one shared ``StudyDosingConfig`` and
        draws one repeated-target dosing realization per returned study, while
        ``dosing_from_range`` builds a deterministic grid in log-dose space
        while reusing the shared route/time realization from that same
        ``StudyDosingConfig``.
    logdose_range:
        Inclusive ``(min_logdose, max_logdose)`` range used when
        ``dosing_list_generation="dosing_from_range"``.
    """
    shared_state = _prepare_sample_experiment_shared_state(
        synthetic_config=synthetic_config,
        context_observation_config=context_observation_config,
        target_observation_config=target_observation_config,
    )

    if dosing_mode == "repeated_dosing":
        return _build_sample_experiment_studies_repeated_dosing(
            shared_state,
            synthetic_config.meta_dosing_config,
            n_targets=n_targets,
            n_dosings=n_dosings,
        )
    if dosing_mode == "dosing_list":
        return _build_sample_experiment_studies_dosing_list(
            shared_state,
            synthetic_config.meta_dosing_config,
            n_targets=n_targets,
            n_dosings=n_dosings,
            dosing_list_generation=dosing_list_generation,
            logdose_range=logdose_range,
        )
    if dosing_mode == "vpc_context":
        return _build_sample_experiment_studies_vpc_context(
            shared_state,
            synthetic_config.meta_dosing_config,
            n_targets=n_targets,
            n_dosings=n_dosings,
        )
    return _build_sample_experiment_studies_diverse_dosing(
        shared_state,
        synthetic_config.meta_dosing_config,
        n_targets=n_targets,
        n_dosings=n_dosings,
    )


def build_sample_experiment_studies(
    meta_study_config: MetaStudyConfig,
    meta_dosing_config: MetaDosingConfig,
    context_observation_config: ObservationsConfig,
    target_observation_config: ObservationsConfig,
    n_targets: int,
    n_dosings: int,
    dosing_mode: str,
    dosing_list_generation: str = "dosing_from_samples",
    logdose_range: Optional[Tuple[float, float]] = None,
    *,
    max_attempts: Optional[int] = None,
) -> list[StudyJSON]:
    """Build sample experiments with internal replacement resampling.

    The generator samples one shared study-level state per attempt. When a
    sampled context or target simulation is numerically invalid, the whole
    sample experiment is discarded and resampled so callers keep receiving the
    requested number of studies without changing loader semantics.
    """

    if n_targets < 0:
        raise ValueError("n_targets must be non-negative")
    if n_dosings < 0:
        raise ValueError("n_dosings must be non-negative")

    resolved_dosing_mode = normalize_sample_experiment_dosing_mode(dosing_mode)
    if resolved_dosing_mode == "dosing_list" and dosing_list_generation not in {
        "dosing_from_samples",
        "dosing_from_range",
    }:
        raise ValueError(
            "dosing_list_generation must be one of {'dosing_from_samples', 'dosing_from_range'}"
        )

    resolved_max_attempts = (
        _SAMPLE_EXPERIMENT_MAX_RETRY_ATTEMPTS if max_attempts is None else int(max_attempts)
    )
    if resolved_max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    last_error: Optional[_SampleExperimentInvalidSimulationError] = None
    for attempt_idx in range(1, resolved_max_attempts + 1):
        synthetic_config = sample_synthetic_sample_experiment_config(
            meta_study_config,
            meta_dosing_config,
        )
        try:
            return _build_sample_experiment_studies_single_attempt(
                synthetic_config=synthetic_config,
                context_observation_config=context_observation_config,
                target_observation_config=target_observation_config,
                n_targets=n_targets,
                n_dosings=n_dosings,
                dosing_mode=resolved_dosing_mode,
                dosing_list_generation=dosing_list_generation,
                logdose_range=logdose_range,
            )
        except _SampleExperimentInvalidSimulationError as err:
            last_error = err
            if attempt_idx >= resolved_max_attempts:
                break

    assert last_error is not None
    raise RuntimeError(
        "Unable to build a valid sample experiment after "
        f"{resolved_max_attempts} attempts for dosing_mode='{resolved_dosing_mode}'. "
        f"Last failed block='{last_error.block_name}' "
        f"dosing_signature={last_error.dosing_signature}"
    ) from last_error


def _build_synthetic_vpc_case(
    meta_study_config: MetaStudyConfig,
    meta_dosing_config: MetaDosingConfig,
    observation_config: ObservationsConfig,
    *,
    case_index: int,
    n_observed_individuals: int,
    sample_size: int,
    max_attempts: Optional[int] = None,
) -> tuple[StudyJSON, list[StudyJSON]]:
    """Build one native synthetic VPC case with bounded replacement resampling."""

    if n_observed_individuals <= 0:
        raise ValueError("n_observed_individuals must be positive for synthetic VPC generation")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive for synthetic VPC generation")

    resolved_max_attempts = (
        _SAMPLE_EXPERIMENT_MAX_RETRY_ATTEMPTS if max_attempts is None else int(max_attempts)
    )
    if resolved_max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    last_error: Optional[_SampleExperimentInvalidSimulationError] = None
    for attempt_idx in range(1, resolved_max_attempts + 1):
        synthetic_config = sample_synthetic_sample_experiment_config(
            meta_study_config,
            meta_dosing_config,
        )
        try:
            return _build_synthetic_vpc_case_single_attempt(
                synthetic_config=synthetic_config,
                observation_config=observation_config,
                case_index=case_index,
                n_observed_individuals=n_observed_individuals,
                sample_size=sample_size,
            )
        except _SampleExperimentInvalidSimulationError as err:
            last_error = err
            if attempt_idx >= resolved_max_attempts:
                break

    assert last_error is not None
    raise RuntimeError(
        "Unable to build a valid synthetic VPC case after "
        f"{resolved_max_attempts} attempts for case_index={case_index}. "
        f"Last failed block='{last_error.block_name}' "
        f"dosing_signature={last_error.dosing_signature}"
    ) from last_error


def _build_synthetic_vpc_data_list(
    meta_study_config: MetaStudyConfig,
    meta_dosing_config: MetaDosingConfig,
    observation_config: ObservationsConfig,
    *,
    n_cases: int,
    n_observed_individuals: int,
    sample_size: int,
    max_attempts: Optional[int] = None,
) -> list[tuple[StudyJSON, list[StudyJSON]]]:
    """Build native synthetic VPC inputs as ``(observed_study, replicates)`` tuples.

        Each case samples one study-level state once, then returns:
    - one observed context-only ``StudyJSON``
    - ``sample_size`` context-only replicate studies aligned to that observed
      study's explicit VPC schedule

    The observed study's realized dosing layout is preserved exactly across
    all replicate studies within the case, and observation sampling inherits
    the datamodule's native observation strategy configuration provided by the
    caller.
    """

    if n_cases < 0:
        raise ValueError("n_cases must be non-negative for synthetic VPC generation")
    if n_cases == 0:
        return []

    case_iterator = tqdm(
        range(n_cases),
        desc="Synthetic VPC cases",
        dynamic_ncols=True,
    )
    return [
        _build_synthetic_vpc_case(
            meta_study_config=meta_study_config,
            meta_dosing_config=meta_dosing_config,
            observation_config=observation_config,
            case_index=case_index,
            n_observed_individuals=n_observed_individuals,
            sample_size=sample_size,
            max_attempts=max_attempts,
        )
        for case_index in case_iterator
    ]


def prepare_full_simulation_to_study_json(
    meta_study_config: MetaStudyConfig,
    observation_config: ObservationsConfig,
    meta_dosing_config: MetaDosingConfig,
    *,
    retry_on_invalid: bool = True,
    idx: int = 0,
) -> tuple[StudyJSON, int]:
    """Generate a full simulation and convert it into a :class:`StudyJSON` record.

    Parameters
    ----------
    meta_study_config:
        Sampling configuration describing the population and numerical solver.
        If meta_study_config.simple_mode is True, uses simplified synthetic data.
    observation_config:
        Configuration for the observation strategy used to extract measurements
        from the raw simulation.  All generated observations are stored under
        the ``context`` section of the returned study.
    meta_dosing_config:
        Configuration describing the dosing regimen for each simulated
        individual.
    retry_on_invalid:
        When ``True`` (default) the function retries simulation sampling if the
        generated trajectories are numerically invalid.
    idx:
        Internal recursion depth counter exposed for debugging and testing.

    Returns
    -------
    tuple[StudyJSON, int]
        Canonical JSON representation of the simulated study with all
        individuals stored in the ``context`` field and an empty ``target``
        list, alongside the number of failed attempts before obtaining the
        valid simulation.
    """
    if getattr(meta_study_config, "simple_mode", False):
        # Handle simple synthetic data generation
        (
            full_sim,
            full_times,
            dosing_amounts,
            dosing_routes,
            _time_points,
            time_scales,
        ) = _generate_simple_simulation(meta_study_config)
        study_config = {""}
        dosing_config_array = [
            DosingConfig(dose=float(d), route="", time=0.0) for d in dosing_amounts
        ]
        failed_attempts = 0
    else:
        # Original mechanistic simulation code
        (
            full_sim,
            full_times,
            dosing_amounts,
            _dosing_routes,
            _time_points,
            time_scales,
            study_config,
            dosing_config_array,
            failed_attempts,
        ) = _generate_full_simulation(
            meta_study_config,
            meta_dosing_config,
            retry_on_invalid=retry_on_invalid,
            idx=idx,
        )

    observation_strategy = ObservationStrategyFactory.from_config(
        observation_config, meta_study_config
    )
    obs_out, time_out, mask_out, rem_sim, rem_time, rem_mask, _ = observation_strategy.generate(
        full_simulation=full_sim,
        full_simulation_times=full_times,
        time_scales=time_scales,
    )

    context: list[IndividualJSON] = []
    num_individuals = full_sim.shape[0]

    for ind_idx in range(num_individuals):
        mask = mask_out[ind_idx].to(torch.bool)
        observations = obs_out[ind_idx][mask].tolist()
        observation_times = time_out[ind_idx][mask].tolist()

        _ensure_strictly_increasing_observations(
            observation_times,
            observations,
            individual_id=f"context_{ind_idx}",
        )

        individual: IndividualJSON = {
            "name_id": f"context_{ind_idx}",
            "observations": observations,
            "observation_times": observation_times,
        }

        if rem_sim is not None and rem_time is not None and rem_mask is not None:
            rem_mask_row = rem_mask[ind_idx].to(torch.bool)
            if rem_mask_row.any():
                individual["remaining"] = rem_sim[ind_idx][rem_mask_row].tolist()
                individual["remaining_times"] = rem_time[ind_idx][rem_mask_row].tolist()

        dosing_cfg = dosing_config_array[ind_idx]
        dose = float(dosing_amounts[ind_idx].item())
        route = getattr(dosing_cfg, "route", "")
        dosing_time = float(getattr(dosing_cfg, "time", 0.0))

        if dose or route:
            individual["dosing"] = [dose]
            individual["dosing_type"] = [route]
            individual["dosing_times"] = [dosing_time]
            individual["dosing_name"] = [route]

        context.append(individual)

    study_json: StudyJSON = {
        "context": context,
        "target": [],
        "meta_data": {
            "study_name": f"simulated_study_{idx}",
            "substance_name": getattr(study_config, "drug_id", "simulated_substance"),
        },
    }

    return study_json, failed_attempts


def prepare_full_simulation_with_repeated_targets(
    meta_study_config: MetaStudyConfig,
    meta_dosing_config: MetaDosingConfig,
    n_targets: int,
    *,
    different_dosing: bool = False,
    retry_on_invalid: bool = True,
    idx: int = 0,
):
    """
    Generate a context simulation (normal dosing) plus a new set of target
    individuals.

    Parameters
    ----------
    different_dosing:
        If ``False`` (default), all target individuals share one repeated
        dosing configuration.
        If ``True``, each target individual gets an independent dosing sample
        from the same distribution used for context individuals.

    Returns
    -------
    context_sim, context_times,
    target_sim, target_times,
    dosing_amounts_ctx, dosing_routes_ctx,
    dosing_amounts_tgt, dosing_routes_tgt,
    time_points, time_scales
    """
    study_config = sample_study_config(meta_study_config)
    indiv_config_array = sample_individual_configs(study_config)
    time_scales = derive_timescale_parameters(study_config, meta_study_config)

    time_points = torch.linspace(
        meta_study_config.time_start,
        meta_study_config.time_stop,
        meta_study_config.time_num_steps,
        dtype=torch.float32,
    )

    # Context part
    local_meta_dosing_ctx = replace(
        meta_dosing_config, num_individuals=study_config.num_individuals
    )
    dosing_config_array_ctx = sample_dosing_configs(local_meta_dosing_ctx)

    full_sim, full_times, dosing_amounts_all, dosing_routes_all = sample_study(
        indiv_config_array,
        dosing_config_array_ctx,
        time_points,
        meta_study_config.solver_method,
    )
    if not is_valid_simulation(full_sim):
        if retry_on_invalid:
            return prepare_full_simulation_with_repeated_targets(
                meta_study_config,
                meta_dosing_config,
                n_targets,
                different_dosing=different_dosing,
                idx=idx + 1,
            )
        raise RuntimeError("Invalid context simulation")

    context_sim, context_times, ctx_idx = split_context_only(full_sim, full_times)
    dosing_amounts_ctx = dosing_amounts_all[ctx_idx]
    dosing_routes_ctx = dosing_routes_all[ctx_idx]

    dosing_amounts_ctx = dosing_amounts_all[ctx_idx]
    dosing_routes_ctx = dosing_routes_all[ctx_idx]

    # Target part
    indiv_cfg_targets = sample_individual_configs(study_config, n=n_targets)
    local_meta_dosing_tgt = replace(meta_dosing_config, num_individuals=n_targets)
    if different_dosing:
        dosing_config_array_tgt = sample_dosing_configs(local_meta_dosing_tgt)
    else:
        dosing_config_array_tgt = sample_dosing_configs_repeated_target(
            local_meta_dosing_tgt, n_targets
        )

    full_sim_tgt, full_times_tgt, dosing_amounts_tgt, dosing_routes_tgt = sample_study(
        indiv_cfg_targets,
        dosing_config_array_tgt,
        time_points,
        meta_study_config.solver_method,
    )
    if not is_valid_simulation(full_sim_tgt):
        if retry_on_invalid:
            return prepare_full_simulation_with_repeated_targets(
                meta_study_config,
                meta_dosing_config,
                n_targets,
                different_dosing=different_dosing,
                idx=idx + 1,
            )
        raise RuntimeError("Invalid target simulation")

    _, _, target_sim, target_times, _, tgt_idx = split_simulations_repeated_target(
        full_sim_tgt, full_times_tgt
    )

    return (
        context_sim,
        context_times,
        target_sim,
        target_times,
        dosing_amounts_ctx,
        dosing_routes_ctx,
        dosing_amounts_tgt[tgt_idx],
        dosing_routes_tgt[tgt_idx],
        time_points,
        time_scales,
    )


def prepare_full_simulation_list_with_repeated_targets(
    meta_study_config: MetaStudyConfig,
    meta_dosing_config: MetaDosingConfig,
    n_targets: int,
    num_of_different_dosages: int,
    *,
    retry_on_invalid: bool = True,
    idx: int = 0,
):
    """Generate one shared context and ``L`` target sets with repeated dosing.

    Parameters
    ----------
    meta_study_config:
        Sampling configuration controlling PK population and solver behaviour.
    meta_dosing_config:
        Dosing-distribution configuration used for both context and targets.
    n_targets:
        Number of target individuals for each dosing condition.
    num_of_different_dosages:
        Number of target dosing conditions ``L``.
    retry_on_invalid:
        Whether to retry sampling when numerical invalid simulations are found.
    idx:
        Retry depth / attempt index used for diagnostics.

    Returns
    -------
    tuple
        ``(context_sim, context_times, dosing_amounts_ctx, dosing_routes_ctx,``
        ``target_simulations, target_times_list, target_dosing_amounts_list,``
        ``target_dosing_routes_list, time_points, time_scales)`` where each
        target list has length ``num_of_different_dosages``.
    """

    if num_of_different_dosages < 0:
        raise ValueError("num_of_different_dosages must be non-negative")

    study_config = sample_study_config(meta_study_config)
    indiv_config_array = sample_individual_configs(study_config)
    time_scales = derive_timescale_parameters(study_config, meta_study_config)

    # [T]
    time_points = torch.linspace(
        meta_study_config.time_start,
        meta_study_config.time_stop,
        meta_study_config.time_num_steps,
        dtype=torch.float32,
    )

    # Context is sampled exactly once.
    local_meta_dosing_ctx = replace(
        meta_dosing_config, num_individuals=study_config.num_individuals
    )
    dosing_config_array_ctx = sample_dosing_configs(local_meta_dosing_ctx)
    full_sim, full_times, dosing_amounts_all, dosing_routes_all = sample_study(
        indiv_config_array,
        dosing_config_array_ctx,
        time_points,
        meta_study_config.solver_method,
    )
    if not is_valid_simulation(full_sim):
        if retry_on_invalid:
            return prepare_full_simulation_list_with_repeated_targets(
                meta_study_config,
                meta_dosing_config,
                n_targets,
                num_of_different_dosages,
                idx=idx + 1,
            )
        raise RuntimeError("Invalid context simulation")

    # context_sim: [N_ctx, T], context_times: [N_ctx, T]
    context_sim, context_times, ctx_idx = split_context_only(full_sim, full_times)
    dosing_amounts_ctx = dosing_amounts_all[ctx_idx]
    dosing_routes_ctx = dosing_routes_all[ctx_idx]

    # Keep the same target PK individuals across all dosing conditions so that
    # only dosing changes across list elements.
    indiv_cfg_targets = sample_individual_configs(study_config, n=n_targets)
    local_meta_dosing_tgt = replace(meta_dosing_config, num_individuals=n_targets)

    target_simulations = []
    target_times_list = []
    target_dosing_amounts_list = []
    target_dosing_routes_list = []
    seen_dosing_signatures: set[tuple[str, float]] = set()

    for _ in range(num_of_different_dosages):
        attempts = 0
        while True:
            attempts += 1
            dosing_config_array_tgt = sample_dosing_configs_repeated_target(
                local_meta_dosing_tgt, n_targets
            )

            # Ensure distinct dosing regimens across list elements.
            dosing_signature = ("", 0.0)
            if n_targets > 0 and len(dosing_config_array_tgt) > 0:
                first_cfg = dosing_config_array_tgt[0]
                dosing_signature = (
                    str(getattr(first_cfg, "route", "")),
                    float(getattr(first_cfg, "dose", 0.0)),
                )
                if dosing_signature in seen_dosing_signatures and num_of_different_dosages > 1:
                    if attempts < 100:
                        continue
                    logger.warning(
                        "Could not sample a unique repeated target dosing signature after %d attempts.",
                        attempts,
                    )

            full_sim_tgt, full_times_tgt, dosing_amounts_tgt, dosing_routes_tgt = sample_study(
                indiv_cfg_targets,
                dosing_config_array_tgt,
                time_points,
                meta_study_config.solver_method,
            )
            if not is_valid_simulation(full_sim_tgt):
                if retry_on_invalid and attempts < 100:
                    continue
                if retry_on_invalid:
                    return prepare_full_simulation_list_with_repeated_targets(
                        meta_study_config,
                        meta_dosing_config,
                        n_targets,
                        num_of_different_dosages,
                        idx=idx + 1,
                    )
                raise RuntimeError("Invalid target simulation")

            _, _, target_sim, target_times, _, tgt_idx = split_simulations_repeated_target(
                full_sim_tgt, full_times_tgt
            )

            target_simulations.append(target_sim)
            target_times_list.append(target_times)
            target_dosing_amounts_list.append(dosing_amounts_tgt[tgt_idx])
            target_dosing_routes_list.append(dosing_routes_tgt[tgt_idx])
            if n_targets > 0:
                seen_dosing_signatures.add(dosing_signature)
            break

    return (
        context_sim,
        context_times,
        dosing_amounts_ctx,
        dosing_routes_ctx,
        target_simulations,
        target_times_list,
        target_dosing_amounts_list,
        target_dosing_routes_list,
        time_points,
        time_scales,
    )


def prepare_ensemble_of_simulations(
    meta_study_config: MetaStudyConfig,
    observation_config: ObservationsConfig,
    meta_dosing_config: MetaDosingConfig,
    number_of_samples: int,
    file_name: Optional[str] = None,
    group_size: Optional[int] = None,
) -> tuple[list[StudyJSON] | list[list[StudyJSON]], float]:
    """Generate an ensemble of simulated studies.

    The helper repeatedly calls :func:`prepare_full_simulation_to_study_json`
    to produce ``number_of_samples`` independent simulations. When ``file_name``
    is provided, the resulting list is serialized as JSON for reproducibility
    and downstream processing.

    Parameters
    ----------
    meta_study_config:
        Sampling configuration controlling the pharmacokinetic population and
        solver settings.
    observation_config:
        Observation strategy applied to each generated simulation.
    meta_dosing_config:
        Configuration describing the dosing regimen per simulated individual.
    number_of_samples:
        Number of simulations to generate.
    file_name:
        Optional path used to persist the generated ensemble as a JSON file.
    group_size:
        Optional number of studies per group. If provided, the return value is
        a list of lists where each sublist has ``group_size`` elements. Extra
        simulations that do not fit evenly into the last group are ignored.

    Returns
    -------
    tuple[list[StudyJSON] | list[list[StudyJSON]], float]
        Ensemble of simulated studies (flat or grouped) and the proportion of
        failed simulation attempts encountered while generating the ensemble.
    """

    studies: list[StudyJSON] = []
    total_failed_attempts = 0
    for idx in range(number_of_samples):
        study, failed_attempts = prepare_full_simulation_to_study_json(
            meta_study_config=meta_study_config,
            observation_config=observation_config,
            meta_dosing_config=meta_dosing_config,
            idx=idx,
        )
        studies.append(study)
        total_failed_attempts += failed_attempts

    # --- Optional serialization ---
    if file_name:
        path = Path(file_name)
        path.write_text(json.dumps(studies, indent=2))

    # --- Compute failure rate ---
    total_successful = len(studies)
    total_attempts = total_failed_attempts + total_successful
    failure_rate = total_failed_attempts / total_attempts if total_attempts > 0 else 0.0

    # --- Optional grouping ---
    if group_size and group_size > 0:
        n_full_groups = len(studies) // group_size
        grouped_studies = [
            studies[i * group_size : (i + 1) * group_size] for i in range(n_full_groups)
        ]
        return grouped_studies, failure_rate

    return studies, failure_rate


def prepare_full_simulation_to_study_json_context_target(
    meta_study_config: MetaStudyConfig,
    observation_config: ObservationsConfig,
    meta_dosing_config_context: MetaDosingConfig,
    meta_dosing_config_target: MetaDosingConfig,
    *,
    retry_on_invalid: bool = True,
    idx: int = 0,
) -> tuple[StudyJSON, int]:
    """Generate a full simulation and convert it into a :class:`StudyJSON` record.
    Different dosing regimens are used for context and target individuals.

    Parameters
    ----------
    meta_study_config:
        Sampling configuration describing the population and numerical solver.
        If meta_study_config.simple_mode is True, uses simplified synthetic data.
    observation_config:
        Configuration for the observation strategy used to extract measurements
        from the raw simulation.  All generated observations are stored under
        the ``context`` section of the returned study.
    meta_dosing_config_context:
        Configuration describing the dosing regimen for each simulated
        individual in the context set.
    meta_dosing_config_target:
        Configuration describing the dosing regimen for each simulated
        individual in the target set.
    retry_on_invalid:
        When ``True`` (default) the function retries simulation sampling if the
        generated trajectories are numerically invalid.
    idx:
        Internal recursion depth counter exposed for debugging and testing.

    Returns
    -------
    tuple[StudyJSON, int]
        Canonical JSON representation of the simulated study with all
        individuals stored in the ``context`` field and an empty ``target``
        list, alongside the number of failed attempts before obtaining the
        valid simulation.
    """

    def prepare_section(name, meta_dosing_config):
        (
            full_sim,
            full_times,
            dosing_amounts,
            _dosing_routes,
            _time_points,
            time_scales,
            study_config,
            dosing_config_array,
            failed_attempts,
        ) = _generate_full_simulation(
            meta_study_config,
            meta_dosing_config,
            retry_on_invalid=retry_on_invalid,
            idx=idx,
        )

        observation_strategy = ObservationStrategyFactory.from_config(
            observation_config, meta_study_config
        )
        obs_out, time_out, mask_out, rem_sim, rem_time, rem_mask, _ = observation_strategy.generate(
            full_simulation=full_sim,
            full_simulation_times=full_times,
            time_scales=time_scales,
        )

        section: list[IndividualJSON] = []
        num_individuals = full_sim.shape[0]

        for ind_idx in range(num_individuals):
            mask = mask_out[ind_idx].to(torch.bool)
            observations = obs_out[ind_idx][mask].tolist()
            observation_times = time_out[ind_idx][mask].tolist()

            _ensure_strictly_increasing_observations(
                observation_times,
                observations,
                individual_id=f"{name}_{ind_idx}",
            )

            individual: IndividualJSON = {
                "name_id": f"{name}_{ind_idx}",
                "observations": observations,
                "observation_times": observation_times,
            }

            if rem_sim is not None and rem_time is not None and rem_mask is not None:
                rem_mask_row = rem_mask[ind_idx].to(torch.bool)
                if rem_mask_row.any():
                    individual["remaining"] = rem_sim[ind_idx][rem_mask_row].tolist()
                    individual["remaining_times"] = rem_time[ind_idx][rem_mask_row].tolist()

            dosing_cfg = dosing_config_array[ind_idx]
            dose = float(dosing_amounts[ind_idx].item())
            route = getattr(dosing_cfg, "route", "")
            dosing_time = float(getattr(dosing_cfg, "time", 0.0))

            if dose or route:
                individual["dosing"] = [dose]
                individual["dosing_type"] = [route]
                individual["dosing_times"] = [dosing_time]
                individual["dosing_name"] = [route]

            section.append(individual)

        return section, study_config, failed_attempts

    # Set RNG to have the same study config for both context and target
    torch.manual_seed(42)
    context, study_config, failed_attempts_context = prepare_section(
        "context", meta_dosing_config_context
    )
    torch.manual_seed(42)
    target, _, failed_attempts_target = prepare_section("target", meta_dosing_config_target)

    study_json: StudyJSON = {
        "context": context,
        "target": target,
        "meta_data": {
            "study_name": f"simulated_study_{idx}",
            "substance_name": getattr(study_config, "drug_id", "simulated_substance"),
        },
    }
    failed_attempts = failed_attempts_context + failed_attempts_target

    return study_json, failed_attempts
