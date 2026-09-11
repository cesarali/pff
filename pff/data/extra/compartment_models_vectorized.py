import numpy as np
import torch

def sample_individual_configs_vectorized(study_config):
    """
    Vectorizes the sampling of parameters for a population of individuals.
    
    Parameters
    ----------
    study_config : StudyConfig
        Contains the study settings and distribution parameters.
    
    Returns
    -------
    config_dict : dict
        Dictionary containing the vectorized parameters and time-magnitudes.
        Keys:
          'k_a', 'k_e', 'V': Tensors of shape (N,)
          'k_1p', 'k_p1': Tensors of shape (N, P)
          'k_a_tmag', 'k_e_tmag', 'V_tmag': Scalars
          'k_1p_tmag', 'k_p1_tmag': Tensors of shape (P,)
          'num_peripherals': int
    """
    N = study_config.num_individuals
    P = study_config.num_peripherals
    
    # Sample the central parameters as tensors of shape (N,)
    k_a = torch.from_numpy(np.random.lognormal(study_config.log_k_a_mean, study_config.log_k_a_std, size=N)).float()
    k_e = torch.from_numpy(np.random.lognormal(study_config.log_k_e_mean, study_config.log_k_e_std, size=N)).float()
    V   = torch.from_numpy(np.random.lognormal(study_config.log_V_mean,   study_config.log_V_std,   size=N)).float()
    
    # Sample the peripheral parameters as tensors of shape (N, P)
    k_1p = []
    k_p1 = []
    for i in range(P):
        k_1p_i = torch.from_numpy(np.random.lognormal(study_config.log_k_1p_mean[i],
                                                       study_config.log_k_1p_std[i], size=N)).float()
        k_p1_i = torch.from_numpy(np.random.lognormal(study_config.log_k_p1_mean[i],
                                                       study_config.log_k_p1_std[i], size=N)).float()
        k_1p.append(k_1p_i)
        k_p1.append(k_p1_i)
    # Stack along the peripheral dimension: shape becomes (N, P)
    k_1p = torch.stack(k_1p, dim=1)
    k_p1 = torch.stack(k_p1, dim=1)
    
    # Pack time-magnitudes (assumed scalars for central parameters and lists for peripherals)
    k_a_tmag = study_config.k_a_tmag  # scalar
    k_e_tmag = study_config.k_e_tmag  # scalar
    V_tmag   = study_config.V_tmag    # scalar
    # For peripherals, we assume the study_config gives lists/arrays of length P.
    k_1p_tmag = torch.tensor(study_config.k_1p_tmag).float()  # shape (P,)
    k_p1_tmag = torch.tensor(study_config.k_p1_tmag).float()    # shape (P,)

    config_dict = {
        'k_a': k_a,
        'k_e': k_e,
        'V': V,
        'k_1p': k_1p,
        'k_p1': k_p1,
        'k_a_tmag': k_a_tmag,
        'k_e_tmag': k_e_tmag,
        'V_tmag': V_tmag,
        'k_1p_tmag': k_1p_tmag,
        'k_p1_tmag': k_p1_tmag,
        'num_peripherals': P,
    }
    return config_dict

import torch

def compute_rates(config, t):
    """
    Computes the dynamic rates for all individuals at a given time t.
    
    Parameters
    ----------
    config : dict
        Dictionary returned by sample_individual_configs_vectorized.
    t : float or torch.Tensor
        Current time point.
    
    Returns
    -------
    k_a, k_e, V : torch.Tensor
        Tensors of shape (N,).
    k_1p, k_p1 : torch.Tensor
        Tensors of shape (N, P).
    """
    # Ensure t is a tensor
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=config['k_a_tmag'].dtype, device=config['k_a_tmag'].device)

    k_a = config['k_a'] * torch.exp(-config['k_a_tmag'] * t)
    k_e = config['k_e'] * torch.exp(-config['k_e_tmag'] * t)
    V   = config['V']   * torch.exp(-config['V_tmag']   * t)
    
    # Use broadcasting for peripheral compartments
    k_1p = config['k_1p'] * torch.exp(-config['k_1p_tmag'] * t)
    k_p1 = config['k_p1'] * torch.exp(-config['k_p1_tmag'] * t)
    
    return k_a, k_e, V, k_1p, k_p1

def ode_func(t_val, y, config):
    """
    ODE function using vectorized rate computations.
    
    Parameters
    ----------
    t_val : torch.Tensor
        Current time point.
    y : torch.Tensor
        Current state, shape (N, M) where M = 2 + num_peripherals.
    config : dict
        Vectorized individual configuration dictionary.
    
    Returns
    -------
    dy_dt : torch.Tensor
        Time derivative of y, shape (N, M).
    """
    # Get the dynamic rates for all individuals at time t_val.
    k_a, k_e, _, k_1p, k_p1 = compute_rates(config, t_val)
    N = y.size(0)
    P = config['num_peripherals']
    M = 2 + P

    # Build the ODE rate matrix A(t) in a vectorized fashion
    A_all = torch.zeros((N, M, M), dtype=torch.float32)
    A_all[:, 0, 0] = -k_a          # Loss from gut
    A_all[:, 1, 0] = k_a           # Transfer gut -> central
    A_all[:, 1, 1] = -k_e - k_1p.sum(dim=1)  # Loss from central and distribution to peripherals
    A_all[:, 1, 2:2+P] = k_p1       # Transfer central -> peripherals
    A_all[:, 2:2+P, 1] = k_1p       # Transfer peripherals -> central
    # Peripheral compartments clearance:
    for i in range(P):
        A_all[:, 2 + i, 2 + i] = -k_p1[:, i]
    
    # Compute dy/dt = A_all @ y for each individual.
    dy_dt = torch.bmm(A_all, y.unsqueeze(-1)).squeeze(-1)
    return dy_dt

def sample_study_vectorized(study_config, dosing_config, t, solver_method="rk4"):
    """
    Simulates the pharmacokinetic study using vectorized individual configurations.
    
    Parameters
    ----------
    study_config : StudyConfig
        Contains global study settings and distribution parameters.
    dosing_config : DosingConfig
        Contains dosing information.
    t : torch.Tensor
        Time points at which the simulation is evaluated.
    
    Returns
    -------
    full_simulation : torch.Tensor
        Concentration profiles (N, len(t)).
    full_times : torch.Tensor
        Time points replicated for each individual.
    """
    from torchdiffeq import odeint

    # Get the vectorized configuration dictionary
    config = sample_individual_configs_vectorized(study_config)
    N = study_config.num_individuals
    P = study_config.num_peripherals
    M = 2 + P

    # Initial conditions: dose in the gut (first compartment), zeros elsewhere.
    y0 = torch.zeros((N, M), dtype=torch.float32)
    y0[:, 0] = dosing_config.dose

    def wrapped_ode(t_val, y):
        return ode_func(t_val, y, config)

    # Solve the ODE system for all individuals in batch
    y = odeint(wrapped_ode, y0, t, method=solver_method)
    # Extract central compartment (index 1) for each individual
    full_simulation = y[:, :, 1].T
    full_times = t.unsqueeze(0).repeat(N, 1)
    return full_simulation, full_times
