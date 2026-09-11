import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torchdiffeq import odeint
from torchtyping import TensorType

from pff.config_classes.data_config import (
    DosingConfig,
    DosingWithDurationConfig,
    MetaDosingConfig,
    MetaDosingWithDurationConfig,
    MetaStudyConfig,
)
from pff.config_classes.node_pk_config import NodePKExperimentConfig


@dataclass
class StudyConfig:
    """
    This corresponds to the configuration of one study
    """

    drug_id: str  # Identifier for the drug
    num_individuals: int  # Number of individuals in the population
    num_peripherals: int  # Number of peripheral compartments
    log_k_a_mean: float  # Mean absorption rate constant
    log_k_a_std: float  # Standard deviation for absorption rate constant
    k_a_tmag: float  # Magnitude of time-dependent variation of absorption rate constant
    k_a_tscl: float  # Scale of time-dependent variation of absorption rate constant
    log_k_e_mean: float  # Mean elimination rate constant
    log_k_e_std: float  # Standard deviation for elimination rate constant
    k_e_tmag: float  # Magnitude of time-dependent variation of elimination rate constant
    k_e_tscl: float  # Scale of time-dependent variation of elimination rate constant
    log_V_mean: float  # Mean volume of central compartment
    log_V_std: float  # Standard deviation for volume of central compartment
    V_tmag: float  # Magnitude of time-dependent variation of volume of central compartment
    V_tscl: float  # Scale of time-dependent variation of volume of central compartment
    log_k_1p_mean: List[float]  # Mean rate constants (central to other peripherals)
    log_k_1p_std: List[float]  # Standard deviations for k_1p
    k_1p_tmag: List[float]  # Magnitude of time-dependent variation of k_1p
    k_1p_tscl: List[float]  # Scale of time-dependent variation of k_1p
    log_k_p1_mean: List[float]  # Mean rate constants (other peripherals to central)
    log_k_p1_std: List[float]  # Standard deviations for k_p1
    k_p1_tmag: List[float]  # Magnitude of time-dependent variation of k_p1
    k_p1_tscl: List[float]  # Scale of time-dependent variation of k_p1
    time_start: float  # Start time for the study
    time_stop: float  # End time for the study
    rel_ruv: float  # Relative residual unexplained variability for the study


@dataclass
class StudyDosingConfig:
    """Study-level dosing distribution reused across individuals in one study.

    The object stores the log-dose distribution parameters sampled once for a
    study together with the route sampling information needed to generate
    per-individual :class:`DosingConfig` objects.
    """

    logdose_mean: float
    logdose_std: float
    time: float = 0.0
    same_route: bool = True
    route: Optional[str] = None
    route_options: List[str] = field(default_factory=lambda: ["oral", "iv"])
    route_weights: List[float] = field(default_factory=lambda: [0.8, 0.2])


@dataclass
class IndividualConfig:
    """
    This corresponds to the configuration of one individual.
    """

    num_peripherals: int = 2  # Number of peripheral compartments
    k_a: Callable[[float], float] = lambda t: 0.1  # Absorption rate constant (gut to central)
    k_e: Callable[[float], float] = lambda t: 0.05  # Elimination rate constant (central)
    V: Callable[[float], float] = lambda t: 0.05  # Volume of central compartment
    k_1p: List[Callable[[float], float]] = field(
        default_factory=lambda: [lambda t: 0.01, lambda t: 0.01]
    )  # Rate constants from central to other peripherals
    k_p1: List[Callable[[float], float]] = field(
        default_factory=lambda: [lambda t: 0.01, lambda t: 0.01]
    )  # Rate constants from other peripherals to central
    rel_ruv: float = 0.1  # Relative residual unexplained variability per individual


def sample_study_config(config: MetaStudyConfig):
    """
    Samples a StudyConfig object based on the MetaStudyConfig.
    """
    # Generate random values for each parameter
    drug_id = random.choice(config.drug_id_options)
    num_individuals = random.randint(*config.num_individuals_range)
    num_peripherals = random.randint(*config.num_peripherals_range)

    # Sample mean, std, and tmag for each rate constant
    log_k_a_mean = random.uniform(*config.log_k_a_mean_range)
    log_k_a_std = random.uniform(*config.log_k_a_std_range)
    k_a_tmag = random.uniform(*config.k_a_tmag_range)
    k_a_tscl = random.uniform(*config.k_a_tscl_range)

    log_k_e_mean = random.uniform(*config.log_k_e_mean_range)
    log_k_e_std = random.uniform(*config.log_k_e_std_range)
    k_e_tmag = random.uniform(*config.k_e_tmag_range)
    k_e_tscl = random.uniform(*config.k_e_tscl_range)

    log_V_mean = random.uniform(*config.log_V_mean_range)
    log_V_std = random.uniform(*config.log_V_std_range)
    V_tmag = random.uniform(*config.V_tmag_range)
    V_tscl = random.uniform(*config.V_tscl_range)

    log_k_1p_mean = [random.uniform(*config.log_k_1p_mean_range) for _ in range(num_peripherals)]
    log_k_1p_std = [random.uniform(*config.log_k_1p_std_range) for _ in range(num_peripherals)]
    k_1p_tmag = [random.uniform(*config.k_1p_tmag_range) for _ in range(num_peripherals)]
    k_1p_tscl = [random.uniform(*config.k_1p_tscl_range) for _ in range(num_peripherals)]

    log_k_p1_mean = [random.uniform(*config.log_k_p1_mean_range) for _ in range(num_peripherals)]
    log_k_p1_std = [random.uniform(*config.log_k_p1_std_range) for _ in range(num_peripherals)]
    k_p1_tmag = [random.uniform(*config.k_p1_tmag_range) for _ in range(num_peripherals)]
    k_p1_tscl = [random.uniform(*config.k_p1_tscl_range) for _ in range(num_peripherals)]

    rel_ruv = random.uniform(*config.rel_ruv_range)

    return StudyConfig(
        drug_id=drug_id,
        num_individuals=num_individuals,
        num_peripherals=num_peripherals,
        log_k_a_mean=log_k_a_mean,
        log_k_a_std=log_k_a_std,
        k_a_tmag=k_a_tmag,
        k_a_tscl=k_a_tscl,
        log_k_e_mean=log_k_e_mean,
        log_k_e_std=log_k_e_std,
        k_e_tmag=k_e_tmag,
        k_e_tscl=k_e_tscl,
        log_V_mean=log_V_mean,
        log_V_std=log_V_std,
        V_tmag=V_tmag,
        V_tscl=V_tscl,
        log_k_1p_mean=log_k_1p_mean,
        log_k_1p_std=log_k_1p_std,
        k_1p_tmag=k_1p_tmag,
        k_1p_tscl=k_1p_tscl,
        log_k_p1_mean=log_k_p1_mean,
        log_k_p1_std=log_k_p1_std,
        k_p1_tmag=k_p1_tmag,
        k_p1_tscl=k_p1_tscl,
        time_start=config.time_start,
        time_stop=config.time_stop,
        rel_ruv=rel_ruv,
    )


def sample_rate_function(mean_rate, variability, variability_type="sinusoidal"):
    """
    Samples a time-dependent rate function.
    :param mean_rate: Mean rate constant
    :param variability: Variability in the rate constant
    :param variability_type: Type of variability ("sinusoidal" or "decaying")
    :return: A time-dependent rate function
    """
    if variability_type == "sinusoidal":

        def rate_function(t):
            return mean_rate + variability * torch.sin(t)  # Sinusoidal variability
    elif variability_type == "decaying":

        def rate_function(t):
            return mean_rate * torch.exp(-variability * t)  # Decaying variability
    else:
        raise ValueError(f"Unknown variability_type: {variability_type}")
    return rate_function


def simulate_ou_process(
    mu: float, sigma: float, theta: float, dt: float, T: float, seed: Optional[int] = None
) -> np.ndarray:
    """Simulate a mean-reverting Ornstein-Uhlenbeck process."""
    if seed is not None:
        np.random.seed(seed)

    N = int(T / dt)
    X = np.zeros(N)

    # Start from the stationary distribution
    X[0] = np.random.normal(mu, np.sqrt(sigma**2 / (2 * theta)))

    for t in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt))
        X[t] = X[t - 1] + theta * (mu - X[t - 1]) * dt + sigma * dW

    return X


def sample_individual_configs(study_config: StudyConfig, n: Optional[int] = None):
    """
    Samples parameters for a population of individuals.

    Parameters
    ----------
    study_config : StudyConfig
        Configuration object with parameter distributions.
    n : int, optional
        Number of individuals to sample. If None, defaults to
        study_config.num_individuals.

    Returns
    -------
    List[IndividualConfig]
        A list of sampled individual configurations.
    """
    num_individuals = n if n is not None else study_config.num_individuals
    individual_configs = []

    for _ in range(num_individuals):
        # Sample parameters from lognormal distributions
        k_a = np.random.lognormal(study_config.log_k_a_mean, study_config.log_k_a_std)
        k_e = np.random.lognormal(study_config.log_k_e_mean, study_config.log_k_e_std)
        V = np.random.lognormal(study_config.log_V_mean, study_config.log_V_std)
        k_1p = [
            np.random.lognormal(mean, std)
            for mean, std in zip(study_config.log_k_1p_mean, study_config.log_k_1p_std)
        ]
        k_p1 = [
            np.random.lognormal(mean, std)
            for mean, std in zip(study_config.log_k_p1_mean, study_config.log_k_p1_std)
        ]

        # Ornstein–Uhlenbeck processes for time-dependent variability
        dt = 0.1
        ou_times = np.arange(study_config.time_start, study_config.time_stop, dt)
        ou_k_a = k_a * np.exp(
            simulate_ou_process(
                0,
                study_config.k_a_tmag * np.sqrt(2 * study_config.k_a_tscl),
                study_config.k_a_tmag,
                dt,
                study_config.time_stop - study_config.time_start,
            )
        )
        ou_k_e = k_e * np.exp(
            simulate_ou_process(
                0,
                study_config.k_e_tmag * np.sqrt(2 * study_config.k_e_tscl),
                study_config.k_e_tmag,
                dt,
                study_config.time_stop - study_config.time_start,
            )
        )
        ou_V = V * np.exp(
            simulate_ou_process(
                0,
                study_config.V_tmag * np.sqrt(2 * study_config.V_tscl),
                study_config.V_tmag,
                dt,
                study_config.time_stop - study_config.time_start,
            )
        )

        # Time-dependent rate functions
        def k_a_fn(t, ou_k_a=ou_k_a):
            return np.interp(t, ou_times, ou_k_a)

        def k_e_fn(t, ou_k_e=ou_k_e):
            return np.interp(t, ou_times, ou_k_e)

        def V_fn(t, ou_V=ou_V):
            return np.interp(t, ou_times, ou_V)

        # Peripheral exchange rates (sinusoidal modulation as placeholder)
        k_1p_fn = [
            lambda t,
            k_1p_i=k_1p[i],
            tmag_i=study_config.k_1p_tmag[i],
            tscl_i=study_config.k_1p_tscl[i]: k_1p_i * (1 + tmag_i * np.sin(t / tscl_i))
            for i in range(len(k_1p))
        ]
        k_p1_fn = [
            lambda t,
            k_p1_i=k_p1[i],
            tmag_i=study_config.k_p1_tmag[i],
            tscl_i=study_config.k_p1_tscl[i]: k_p1_i * (1 + tmag_i * np.sin(t / tscl_i))
            for i in range(len(k_p1))
        ]

        # Create config for this individual
        config = IndividualConfig(
            num_peripherals=study_config.num_peripherals,
            k_a=k_a_fn,
            k_e=k_e_fn,
            V=V_fn,
            k_1p=k_1p_fn,
            k_p1=k_p1_fn,
            rel_ruv=study_config.rel_ruv,
        )
        individual_configs.append(config)

    return individual_configs


def create_dynamic_ode_matrix(config: IndividualConfig, t: float):
    """
    Creates the ODE matrix for the compartment model at time t.
    :param config: IndividualConfig object
    :param t: Current time
    :return: ODE matrix as a torch tensor
    """
    num_compartments = 2 + config.num_peripherals  # gut, central, and peripherals
    ode_matrix = torch.zeros((num_compartments, num_compartments))

    # Gut compartment
    ode_matrix[0, 0] = -config.k_a(t)  # d_gut/dt = -k_a(t) * gut

    # Central compartment
    ode_matrix[1, 0] = config.k_a(t)  # d_central/dt += k_a(t) * gut
    ode_matrix[1, 1] = -config.k_e(t)  # d_central/dt += -k_e(t) * central

    # Peripheral compartments
    for i in range(config.num_peripherals):
        ode_matrix[1, 1] -= config.k_1p[i](t)  # d_central/dt += - sum_p(k_1p(t)) * central
        ode_matrix[1, 2 + i] = config.k_p1[i](t)  # d_central/dt += k_p1[i](t) * peripheral(i)
        ode_matrix[2 + i, 1] = config.k_1p[i](t)  # d_peripheral(i)/dt += k_1p[i](t) * central
        ode_matrix[2 + i, 2 + i] = -config.k_p1[i](
            t
        )  # d_peripheral(i)/dt += -k_p1[i](t) * peripheral(i)

    return ode_matrix


def create_dynamic_ode_matrix_batched(configs, t, num_peripherals):
    """
    Creates batched ODE matrices for multiple individuals.

    Parameters:
    ----------
    configs : list
        List of IndividualConfig objects.
    t : float
        Current time point.
    num_peripherals : int
        Number of peripheral compartments (same for all individuals).

    Returns:
    -------
    A_all : torch.Tensor
        Tensor of shape (N, M, M) containing ODE matrices for all individuals.
    """
    import torch

    N = len(configs)
    M = 2 + num_peripherals
    A_all = torch.zeros((N, M, M), dtype=torch.float32)

    # Compute batched rate parameters
    k_a_all = torch.tensor([config.k_a(t) for config in configs], dtype=torch.float32)
    k_e_all = torch.tensor([config.k_e(t) for config in configs], dtype=torch.float32)
    k_1p_all = torch.tensor(
        [[config.k_1p[i](t) for i in range(num_peripherals)] for config in configs],
        dtype=torch.float32,
    )
    k_p1_all = torch.tensor(
        [[config.k_p1[i](t) for i in range(num_peripherals)] for config in configs],
        dtype=torch.float32,
    )

    # Populate the batched ODE matrices
    A_all[:, 0, 0] = -k_a_all  # Gut compartment
    A_all[:, 1, 0] = k_a_all  # Absorption into central
    A_all[:, 1, 1] = -k_e_all - k_1p_all.sum(dim=1)  # Central compartment
    A_all[:, 1, 2 : 2 + num_peripherals] = k_p1_all  # Central to peripheral
    A_all[:, 2 : 2 + num_peripherals, 1] = k_1p_all  # Peripheral to central
    for i in range(num_peripherals):
        A_all[:, 2 + i, 2 + i] = -k_p1_all[:, i]  # Peripheral compartments

    return A_all


def sample_study(
    individual_config_array, dosing_config_array, t: torch.Tensor, solver_method: str = "rk4"
) -> Tuple[
    torch.Tensor,  # [N, T] concentration profiles
    torch.Tensor,  # [N, T] time points
    torch.Tensor,  # [N] dosing amounts
    torch.Tensor,  # [N] dosing route types (0 = oral, 1 = iv)
]:
    """
    Simulates the pharmacokinetic study for a group of individuals and returns
    concentration profiles, time points, and dosing metadata.

    Parameters:
    ----------
    individual_config_array : list
        List of IndividualConfig objects for each individual.
    dosing_config_array : list
        List of DosingConfig objects for each individual.
    t : torch.Tensor
        A 1D tensor of time points [T].

    Returns:
    -------
    full_simulation : torch.Tensor
        Concentration profiles [N, T].
    full_simulation_times : torch.Tensor
        Time points [N, T].
    dosing_amounts : torch.Tensor
        Dosing amounts [N].
    dosing_route_types : torch.Tensor
        Route types [N], 0 = oral, 1 = iv.
    """
    # Sanity check
    if len(individual_config_array) != len(dosing_config_array):
        raise ValueError("Number of individuals and dosing configurations must match.")

    N = len(individual_config_array)
    num_peripherals_list = [cfg.num_peripherals for cfg in individual_config_array]
    all_same_peripherals = all(n == num_peripherals_list[0] for n in num_peripherals_list)

    # Extract dosing info
    dosing_amounts = torch.tensor(
        [cfg.dose for cfg in dosing_config_array], dtype=torch.float32
    )  # [N]
    routes_str = [cfg.route for cfg in dosing_config_array]
    route_map = {"oral": 0, "iv": 1}
    dosing_route_types = torch.tensor([route_map[r] for r in routes_str], dtype=torch.int64)  # [N]

    if all_same_peripherals:
        P = num_peripherals_list[0]
        M = 2 + P
        y0 = torch.zeros((N, M), dtype=torch.float32)
        is_oral = dosing_route_types == 0
        is_iv = dosing_route_types == 1
        y0[is_oral, 0] = dosing_amounts[is_oral]
        y0[is_iv, 1] = dosing_amounts[is_iv]

        def ode_func(t, y):
            A_all = create_dynamic_ode_matrix_batched(individual_config_array, t.item(), P)
            return torch.bmm(A_all, y.unsqueeze(-1)).squeeze(-1)

        y = odeint(ode_func, y0, t, method=solver_method)  # [T, N, M]
        V_all = torch.tensor(
            [[cfg.V(ti.item()) for ti in t] for cfg in individual_config_array], dtype=torch.float32
        )  # [N, T]
        full_simulation = y[:, :, 1].T / V_all  # [N, T]
        full_simulation *= (
            1 + torch.randn_like(full_simulation) * individual_config_array[0].rel_ruv
        )
    else:
        full_simulation = []
        for config, dosing_config in zip(individual_config_array, dosing_config_array):
            P = config.num_peripherals
            M = 2 + P
            if dosing_config.route == "oral":
                y0 = torch.tensor([dosing_config.dose] + [0.0] * (M - 1), dtype=torch.float32)
            elif dosing_config.route == "iv":
                y0 = torch.tensor([0.0, dosing_config.dose] + [0.0] * (M - 2), dtype=torch.float32)
            else:
                raise ValueError(f"Unsupported route: {dosing_config.route}")

            def ode_func(t, y):
                A = create_dynamic_ode_matrix(config, t.item())
                return torch.matmul(A, y)

            y = odeint(ode_func, y0, t, method=solver_method)  # [T, M]
            V = torch.tensor([config.V(ti.item()) for ti in t], dtype=torch.float32)  # [T]
            concentration = y[:, 1] / V
            concentration *= 1 + torch.randn_like(concentration) * config.rel_ruv
            full_simulation.append(concentration)

        full_simulation = torch.stack(full_simulation)  # [N, T]

    full_times = t.unsqueeze(0).repeat(N, 1)  # [N, T]

    return full_simulation, full_times, dosing_amounts, dosing_route_types


def sample_study_with_duration(
    individual_config_array,
    dosing_config_array: List[DosingWithDurationConfig],
    t: torch.Tensor,
    solver_method: str = "rk4",
) -> Tuple[
    torch.Tensor,  # [N, T] concentration profiles
    torch.Tensor,  # [N, T] time points
    torch.Tensor,  # [N] dosing amounts
    torch.Tensor,  # [N] dosing route types (0 = oral, 1 = iv)
]:
    """
    Simulates the pharmacokinetic study for a group of individuals and returns
    concentration profiles, time points, and dosing metadata.

    This is a parallel implementation to sample_study that supports infusion dosing with duration.
    Once validated, the two can be merged.

    Parameters:
    ----------
    individual_config_array : list
        List of IndividualConfig objects for each individual.
    dosing_config_array : list
        List of DosingWithDurationConfig objects for each individual.
    t : torch.Tensor
        A 1D tensor of time points [T].

    Returns:
    -------
    full_simulation : torch.Tensor
        Concentration profiles [N, T].
    full_simulation_times : torch.Tensor
        Time points [N, T].
    dosing_amounts : torch.Tensor
        Dosing amounts [N].
    dosing_route_types : torch.Tensor
        Route types [N], 0 = oral, 1 = iv.
    """
    # Sanity check
    if len(individual_config_array) != len(dosing_config_array):
        raise ValueError("Number of individuals and dosing configurations must match.")

    N = len(individual_config_array)
    num_peripherals_list = [cfg.num_peripherals for cfg in individual_config_array]
    all_same_peripherals = all(n == num_peripherals_list[0] for n in num_peripherals_list)

    # Extract dosing info
    dosing_amounts = torch.tensor(
        [cfg.dose for cfg in dosing_config_array], dtype=torch.float32
    )  # [N]
    routes_str = [cfg.route for cfg in dosing_config_array]
    route_map = {"oral": 0, "iv": 1}
    dosing_route_types = torch.tensor([route_map[r] for r in routes_str], dtype=torch.int64)  # [N]
    dosing_durations = torch.tensor(
        [cfg.duration for cfg in dosing_config_array], dtype=torch.float32
    )  # [N]

    if all_same_peripherals and all(dosing_durations == 0):
        P = num_peripherals_list[0]
        M = 2 + P  # gut, central, peripherals
        y0 = torch.zeros((N, M), dtype=torch.float32)

        is_oral = dosing_route_types == 0
        is_iv_bolus = dosing_route_types == 1

        y0[is_oral, 0] = dosing_amounts[is_oral]
        y0[is_iv_bolus, 1] = dosing_amounts[is_iv_bolus]

        def ode_func(t, y):
            A_all = create_dynamic_ode_matrix_batched(individual_config_array, t.item(), P)
            return torch.bmm(A_all, y.unsqueeze(-1)).squeeze(-1)

        # ODE solving during infusion
        y = odeint(ode_func, y0, t, method=solver_method)  # [T, N, M]
        V_all = torch.tensor(
            [[cfg.V(ti.item()) for ti in t] for cfg in individual_config_array], dtype=torch.float32
        )  # [N, T]
        full_simulation = y[:, :, 1].T / V_all  # [N, T]
        full_simulation *= (
            1 + torch.randn_like(full_simulation) * individual_config_array[0].rel_ruv
        )
    else:
        full_simulation = []
        for config, dosing_config in zip(individual_config_array, dosing_config_array):
            P = config.num_peripherals
            M = 2 + P  # gut, central, peripherals
            if dosing_config.route == "oral":
                assert dosing_config.duration == 0, "Oral dosing cannot have a duration."
                y0 = torch.tensor([dosing_config.dose] + [0.0] * (M - 1), dtype=torch.float32)
            elif dosing_config.route == "iv":
                if dosing_config.duration > 0:
                    # Infusion dosing
                    y0 = torch.tensor(
                        [0.0, 0.0] + [0.0] * (M - 2),
                        dtype=torch.float32,
                    )
                else:  # Bolus dosing
                    y0 = torch.tensor(
                        [0.0, dosing_config.dose] + [0.0] * (M - 2), dtype=torch.float32
                    )
            else:
                raise ValueError(f"Unsupported route: {dosing_config.route}")

            def ode_func(t, y):
                A = create_dynamic_ode_matrix(config, t.item())
                b = torch.zeros_like(y)
                if (
                    dosing_config.route == "iv"
                    and dosing_config.duration > 0
                    and t.item() < dosing_config.duration
                ):
                    # During infusion, add rate to central compartment
                    b[1] = dosing_config.dose / dosing_config.duration
                return torch.matmul(A, y) + b

            y = odeint(ode_func, y0, t, method=solver_method)  # [T, M]
            V = torch.tensor([config.V(ti.item()) for ti in t], dtype=torch.float32)  # [T]
            concentration = y[:, 1] / V
            concentration *= 1 + torch.randn_like(concentration) * config.rel_ruv
            full_simulation.append(concentration)

        full_simulation = torch.stack(full_simulation)  # [N, T]

    full_times = t.unsqueeze(0).repeat(N, 1)  # [N, T]

    return full_simulation, full_times, dosing_amounts, dosing_route_types


def derive_timescale_parameters(config: StudyConfig, meta_config: MetaStudyConfig):
    """
    Derive peak time and terminal half life for typical parameters,
    which can then be used to inform a study-specific sampling schedule.
    """
    k_a = np.exp(config.log_k_a_mean)
    k_e = np.exp(config.log_k_e_mean)
    tmax = (np.log(k_e) - np.log(k_a)) / (k_e - k_a)

    # mean residence time approximation for terminal half-life
    MRT = 1 / k_e
    # for i in range(config.num_peripherals):
    #     k_1i = np.exp(config.log_k_p1_mean[i])
    #     MRT += 1/k_1i
    t12 = np.log(2) * MRT
    if t12 > meta_config.time_stop:
        t12 = float(meta_config.time_stop / 2.0)
    if tmax > t12:
        tmax = float(t12 * 0.5)
    return torch.Tensor([tmax, t12])


def sample_study_dosing_config(config: MetaDosingConfig) -> StudyDosingConfig:
    """Sample the study-level dosing distribution from a meta-dosing config.

    This samples the shared log-dose distribution parameters once per study and
    carries the route information needed later by :func:`sample_dosing_arrays`.
    """

    study_route: Optional[str] = None
    if config.same_route:
        study_route = str(np.random.choice(config.route_options, p=config.route_weights))

    return StudyDosingConfig(
        logdose_mean=float(np.random.uniform(*config.logdose_mean_range)),
        logdose_std=float(np.random.uniform(*config.logdose_std_range)),
        time=float(config.time),
        same_route=bool(config.same_route),
        route=study_route,
        route_options=list(config.route_options),
        route_weights=list(config.route_weights),
    )


def sample_dosing_arrays(
    study_dosing_config: StudyDosingConfig,
    num_individuals: int,
) -> List[DosingConfig]:
    """Sample per-individual dosing configs from one study-level dosing object."""

    if num_individuals < 0:
        raise ValueError("num_individuals must be non-negative")
    if num_individuals == 0:
        return []

    if study_dosing_config.same_route:
        if study_dosing_config.route is None:
            raise ValueError("StudyDosingConfig.route must be set when same_route=True")
        route = np.repeat(study_dosing_config.route, num_individuals)
    else:
        route = np.random.choice(
            study_dosing_config.route_options,
            p=study_dosing_config.route_weights,
            size=num_individuals,
        )

    dose = np.random.lognormal(
        study_dosing_config.logdose_mean,
        study_dosing_config.logdose_std,
        size=num_individuals,
    )

    dosing_configs: List[DosingConfig] = []
    for i in range(num_individuals):
        dosing_configs.append(
            DosingConfig(
                dose=float(dose[i]),
                route=str(route[i]),
                time=float(study_dosing_config.time),
            )
        )

    return dosing_configs


def sample_dosing_configs(config: MetaDosingConfig):
    """
    Sample a dosing configuration based on the meta dosing configuration.
    Route may be the same for all individuals in the study or not.
    Doses are lognormally distributed with log-mean and log-std sample uniformly from the specified range.
    In the special case of logdose_std_range = (0, 0), the dose is identical for all individuals.
    """
    study_dosing_config = sample_study_dosing_config(config)
    return sample_dosing_arrays(study_dosing_config, config.num_individuals)


def sample_dosing_with_duration_configs(config: MetaDosingWithDurationConfig):
    """
    Sample a dosing configuration based on the meta dosing configuration.
    Route may be the same for all individuals in the study or not.
    Doses are lognormally distributed with log-mean and log-std sample uniformly from the specified range.
    In the special case of logdose_std_range = (0, 0), the dose is identical for all individuals.
    """
    study_dosing_config = sample_study_dosing_config(config)
    base_dosing_configs = sample_dosing_arrays(study_dosing_config, config.num_individuals)
    dosing_configs = []

    # Draw durations for infusion dosing depending on route
    duration_raw = np.random.uniform(
        config.duration_range[0], config.duration_range[1], size=config.num_individuals
    )

    for i in range(config.num_individuals):
        dosing_config = base_dosing_configs[i]

        # Add duration flag based on route duration weights
        duration_flag = np.random.binomial(
            1, config.route_duration_weights[dosing_config.route], size=1
        )[0]

        this_config = DosingWithDurationConfig(
            dose=dosing_config.dose,
            route=dosing_config.route,
            time=dosing_config.time,
            duration=duration_raw[i] * duration_flag,
        )

        dosing_configs.append(this_config)

    return dosing_configs


def get_random_simulation(
    model_config: NodePKExperimentConfig,
) -> Tuple[TensorType["I", "T"], TensorType["I", "T"]]:
    """
    Generates random simulation data based on the model configuration.

    Args:
        model_config (NodePKConfig): Configuration for the simulation.

    Returns:
        Tuple[TensorType["I", "T"], TensorType["I", "T"]]: Time steps and simulation points.
    """
    I = model_config.meta_study.num_individuals_range[0]
    T = model_config.meta_study.time_num_steps

    # Generate time steps using linspace
    time_steps = (
        torch.linspace(
            model_config.meta_study.time_start,
            model_config.meta_study.time_stop,
            T,
            dtype=torch.float32,
        )
        .unsqueeze(0)
        .repeat(I, 1)
    )  # Shape: [I, T]

    # Generate random simulation points with the same shape
    simulation_points = torch.rand(I, T)  # Shape: [I, T]
    simulation_points = simulation_points / model_config.meta_study.time_stop

    return simulation_points, time_steps
