import torch
import numpy as np
from typing import List,Tuple
from matplotlib import pyplot as plt
from torchtyping import TensorType, patch_typeguard
from pff.data.datasets.aicme_batch import AICMECompartmentsDataBatch
from scipy import stats
import torch
from typing import Tuple

Tensor = torch.Tensor          # for brevity – keep your own alias if you prefer
import os

def ensure_folder_exists(folder_name: str):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"✅ Created folder: {folder_name}")
    else:
        print(f"📁 Folder already exists: {folder_name}")
        
def combine_samples(
    samples_list: list[TensorType["S", "B", "I", "T", 1]]
) -> TensorType["S", "P", "T"]:
    """
    Given:
      samples_list: list of length P, each tensor of shape [S, B, I, T, 1]
                    (here B = I = 1)

    Returns:
      combined: tensor of shape [S, P, T]
    """
    # 1) Extract the [S, T] slice from each sample (drop B=1, I=1, last dim=1)
    #    - s[:, 0, 0, :, 0] has shape [S, T]
    squeezed: list[TensorType["S", "T"]] = [
        s[:, 0, 0, :, 0]
        for s in samples_list
    ]

    # 2) Stack along a new “permutation” axis P → [S, P, T]
    combined: TensorType["S", "P", "T"] = torch.stack(squeezed, dim=1)
    return combined

def extract_context_by_mask(
    db: AICMECompartmentsDataBatch
) -> Tuple[
    List[TensorType["n_i"]],  # context observations per compartment
    List[TensorType["n_i"]]   # context times per compartment
]:
    """
    For B=1, from a single AICMECompartmentsDataBatch:
      - db.context_obs:      [1, c_ind, num_obs_c, 1]
      - db.context_obs_time: [1, c_ind, num_obs_c, 1]
      - db.context_obs_mask: [1, c_ind, num_obs_c]

    Returns two lists of length c_ind:
      obs_list[i].shape == (n_i,)   selects those obs where mask==1
      time_list[i].shape == (n_i,)  selects the corresponding times
    """
    # Unpack and assert B=1
    B, c_ind, num_obs_c, one = db.context_obs.shape
    assert B == 1 and one == 1, f"Expected B=1 and last dim=1, got B={B}, last={one}"

    # Drop the batch and singleton dims:
    #   [1, c_ind, num_obs_c, 1] → [c_ind, num_obs_c]
    obs   = db.context_obs.squeeze(0).squeeze(-1)       # TensorType["c_ind", "num_obs_c"]
    times = db.context_obs_time.squeeze(0).squeeze(-1)  # TensorType["c_ind", "num_obs_c"]
    mask  = db.context_obs_mask.squeeze(0)              # TensorType["c_ind", "num_obs_c"]

    obs_list: List[torch.Tensor] = []
    time_list: List[torch.Tensor] = []

    for i in range(c_ind):
        mi      = mask[i].bool()    # [num_obs_c]
        obs_i   = obs[i][mi]        # [n_i]
        times_i = times[i][mi]      # [n_i]
        obs_list.append(obs_i)
        time_list.append(times_i)

    return obs_list, time_list

def compute_pd(
    y_obs : TensorType["I", "T"],             # observed data
    y_sim : TensorType["S", "I", "T"],        # S simulated datasets
    mask  : TensorType["I", "T"],             # True/1 = valid obs
) -> TensorType["I", "T"]:                    # pd, NaN where mask == 0
    """
    NOTICE THAT THERE IS NO BATCH INDEX, this works only on individual substances

    Prediction discrepancy (pd)   —  Eq. (4)  Comets et al. 2008

    Parameters
    ----------
    y_obs : [I, T]           observed values (padding value doesn't matter,
                             because `mask` says which entries to trust)
    y_sim : [S, I, T]        S Monte-Carlo replicates generated from the model
    mask  : [I, T]           binary mask — True at valid observation points

    Returns
    -------
    pd :   [I, T]            empirical CDF value at (i,j); NaN where mask==0
    """
    S, I, T = y_sim.shape
    assert y_obs.shape == (I, T),  "y_obs must be [I,T]"
    assert mask.shape  == (I, T),  "mask  must be [I,T]"

    # Expand y_obs to [S,I,T] so we can broadcast the < comparison
    y_obs_exp = y_obs.unsqueeze(0).expand(S, -1, -1)      # [S,I,T]

    # δ_{ijk} = 1 if y_sim < y_obs else 0
    delta = (y_sim < y_obs_exp).float()                   # [S,I,T]

    # average over the S simulations   →   empirical CDF
    pd = delta.mean(dim=0)                                # [I,T]

    # put NaN where mask == 0 so the caller knows which are pads
    pd = torch.where(mask.bool(), pd, torch.full_like(pd, float("nan")))

    return pd

def sample_covariance_manual_torch(
    X: TensorType["S", "Tv"]  # simulations for one subject, S×Tᵥ
):
    """
    Pure-Torch analogue of your NumPy helper.
    Returns unbiased covariance [Tᵥ,Tᵥ] and mean vector [Tᵥ].
    """
    S, _ = X.shape
    mean_vec = X.mean(dim=0)                       # [Tᵥ]
    Xc = X - mean_vec                              # [S,Tᵥ]
    cov = Xc.t() @ Xc / (S - 1)                    # [Tᵥ,Tᵥ]
    return cov, mean_vec

def whiten_manual_torch_old(
    X: TensorType["S", "Tv"],        # data to whiten
    eps: float = 1e-8                # ridge for numerical safety
):
    """
    Manual whitening à la your NumPy code.
    Returns whitened X and the whitening matrix W (Σ^{-1/2}).
    """
    cov, mean_vec = sample_covariance_manual_torch(X)   # Σ, μ
    eigvals, eigvecs = torch.linalg.eigh(cov + eps * torch.eye(cov.size(0), device=X.device))
    D_inv_sqrt = torch.diag(torch.rsqrt(eigvals))       # diag(1/√λ)
    W = eigvecs @ D_inv_sqrt @ eigvecs.t()              # Σ^{-1/2}
    X_white = (X - mean_vec) @ W                        # apply whitening
    return X_white, W, mean_vec

def compute_npde_full_old(
    y_obs: TensorType["I", "T"],
    y_sim: TensorType["S", "I", "T"],
    mask : TensorType["I", "T"],
    eps  : float = 1e-8
) -> TensorType["I", "T"]:
    """
    Full NPDE with within-subject decorrelation (Σ^{-1/2}) computed
    **exactly** as in your NumPy snippet.

    NOTICE THAT THERE IS NO BATCH INDEX, this works only on individual substances

    Shapes
    ------
    y_obs : [I,T]     observations (padding allowed)
    y_sim : [S,I,T]   S Monte-Carlo replicates
    mask  : [I,T]     True/1 = valid time-points
    """
    S, I, T = y_sim.shape
    N01 = torch.distributions.Normal(0.0, 1.0)
    out = torch.full_like(y_obs, float("nan"))          # result placeholder

    for i in range(I):
        # ---- select the irregular grid for subject i -------------------
        valid_idx = mask[i].bool()
        if not valid_idx.any():
            continue                                    # nothing to do

        y_i_obs = y_obs[i, valid_idx]                  # [Tᵥ]
        y_i_sim = y_sim[:, i, valid_idx]               # [S,Tᵥ]

        # ---- whitening per your NumPy logic ----------------------------
        y_i_sim_white, W, mean_vec = whiten_manual_torch(y_i_sim, eps)  # [S,Tᵥ]
        if W is None:
            # Whitening degraded → set result to NaN or skip this subject
            out[i, valid_idx] = float("nan")
            continue

        # same transform for the single observation vector
        y_i_obs_white = (y_i_obs - mean_vec) @ W                       # [Tᵥ]

        # ---- empirical CDF on whitened scale (Eq. 4) -------------------
        delta = (y_i_sim_white < y_i_obs_white).float()                # [S,Tᵥ]
        pde   = delta.mean(dim=0)                                      # [Tᵥ]

        # ---- edge-case rule (Eq. 6) ------------------------------------
        one_over_S = 1.0 / S
        pde = torch.where(pde == 0, torch.full_like(pde, one_over_S), pde)
        pde = torch.where(pde == 1, torch.full_like(pde, 1 - one_over_S), pde)

        # ---- NPDE (Eq. 7) ---------------------------------------------
        npde = N01.icdf(pde)                                           # [Tᵥ]

        # ---- write back to full-size tensor ----------------------------
        out[i, valid_idx] = npde

    return out

# ---------------------------------------------------------------------
# 1. Robust whitening
# ---------------------------------------------------------------------
def whiten_manual_torch(
    X: Tensor,                       # [S, Tᵥ]
    eps: float = 1e-8,
    max_attempts: int = 5,
    base_jitter: float = 1e-6
) -> Tuple[Tensor, torch.Tensor | None, Tensor, bool]:
    """
    Returns
    -------
    X_white : [S,Tᵥ]           whitened simulations
    W       : [Tᵥ,Tᵥ] | None   Σ^{-½} (None ⇒ degraded to diag)
    mean    : [Tᵥ]             sample mean
    ok      : bool             True if full Σ^{-½} was used
    """
    S, T = X.shape
    X64   = X.double()
    mean  = X64.mean(dim=0)
    Xm    = X64 - mean
    cov   = (Xm.T @ Xm) / (S - 1)
    I     = torch.eye(T, dtype=X64.dtype, device=X.device)

    W = None
    for k in range(max_attempts):
        jitter = base_jitter * (10.0 ** k)
        try:
            eigvals, eigvecs = torch.linalg.eigh(cov + (eps + jitter) * I)
            if torch.any(eigvals <= 0):
                raise RuntimeError("non-positive eigenvalues")
            inv_sqrt = torch.rsqrt(eigvals)
            W = eigvecs @ torch.diag(inv_sqrt) @ eigvecs.T
            break
        except RuntimeError:
            pass  # try bigger jitter

    if W is None:                               # final fallback
        var = cov.diag().clamp_min(eps)
        W   = torch.diag(torch.rsqrt(var))      # diagonal only
        ok  = False
    else:
        ok  = True

    X_white = (Xm @ W).float()
    return X_white, W.float() if ok else None, mean.float(), ok


# ---------------------------------------------------------------------
# 2. NPDE with an *output* validity mask
# ---------------------------------------------------------------------

def compute_npde_full(
    y_obs: TensorType["I", "T"],
    y_sim: TensorType["S", "I", "T"],
    mask : TensorType["I", "T"],
    eps  : float = 1e-8,
) -> Tuple[TensorType["I", "T"], TensorType["I", "T"]]:
    """
    Full NPDE with within-subject decorrelation (Σ^{-1/2}) computed
    **exactly** as in your NumPy snippet.

    NOTICE THAT THERE IS NO BATCH INDEX, this works only on individual substances

    Args
    ------
    y_obs : [I,T]     observations (padding allowed)
    y_sim : [S,I,T]   S Monte-Carlo replicates
    mask  : [I,T]     True/1 = valid time-points

    Returns
    -------
    npde       : [I,T]   – same shape as `y_obs`
    valid_mask : [I,T]   – True where npde is statistically valid
    """
    S, I, T = y_sim.shape
    N01 = torch.distributions.Normal(0.0, 1.0)

    npde_out   = torch.full_like(y_obs, float("nan"))
    valid_out  = mask.clone().bool()            # start with the user mask

    for i in range(I):
        # ---- select the irregular grid for subject i -------------------
        valid_idx = mask[i].bool()
        if not valid_idx.any():
            valid_out[i] = False
            continue

        y_i_obs = y_obs[i, valid_idx]           # [Tᵥ]
        y_i_sim = y_sim[:, i, valid_idx]        # [S,Tᵥ]

        # ---- whitening per your NumPy logic ----------------------------
        y_i_sim_white, W, mean_vec, ok = whiten_manual_torch(y_i_sim, eps)

        if not ok:                              # whitening failed → invalidate
            valid_out[i, valid_idx] = False
            continue

        # same transform for the single observation vector
        y_i_obs_white = (y_i_obs - mean_vec) @ W

        # ---- empirical CDF on whitened scale (Eq. 4) -------------------
        delta = (y_i_sim_white < y_i_obs_white).float()
        pde   = delta.mean(dim=0)

        # ---- edge-case rule (Eq. 6) ------------------------------------
        one_over_S = 1.0 / S
        pde = torch.where(pde == 0, torch.full_like(pde, one_over_S), pde)
        pde = torch.where(pde == 1, torch.full_like(pde, 1 - one_over_S), pde)

        # ---- NPDE (Eq. 7) ---------------------------------------------
        npde = N01.icdf(pde)
        npde_out[i, valid_idx] = npde

    return npde_out, valid_out


def compute_npde_in_batch(
    y_obs: TensorType["B", "I", "T"],
    y_sim: TensorType["S", "B", "I", "T"],
    mask: TensorType["B", "I", "T"],
    eps: float = 1e-8,
) -> TensorType["B", "I", "T"]:
    """Compute NPDE for each element in a batch.

    Parameters
    ----------
    y_obs : [B, I, T]  Observed values per batch item (context observations).
    y_sim : [S, B, I, T]  Simulated predictions.
    mask  : [B, I, T]  Validity mask for observations.

    Returns
    -------
    Tensor of shape [B, I, T] with NPDE values.
    """
    B = y_obs.size(0)
    results = []
    for b in range(B):
        npde_b = compute_npde_full(y_obs[b], y_sim[:, b], mask[b], eps)
        results.append(npde_b)
    return torch.stack(results, dim=0)

def shapiro_wilk_normality(npde: TensorType["T"]) -> Tuple[float, float]:
    """Return Shapiro-Wilk normality test statistic and p-value for a 1-D tensor."""
    npde_np = npde[torch.isfinite(npde)].detach().cpu().numpy()
    w, p = stats.shapiro(npde_np)
    return float(w), float(p)

def qq_plot(npde: TensorType["T"], train:bool =False, epoch:str|int = "na", **kwargs) -> str | None:
    """
    Generate and optionally save/show a QQ plot of NPDE values.
    
    Args:
        npde: Tensor containing NPDE values.
        train (bool, optional): If True (default), saves plot to file.
        model (optional): Lightning model, used to name the file with `current_epoch`.
    
    Returns:
        File path if saved, None otherwise.
    """
    npde_np = npde[torch.isfinite(npde)].detach().cpu().numpy()

    fig = plt.figure()
    stats.probplot(npde_np, dist="norm", plot=plt)

    if train:
        # Use model.current_epoch if provided
        path = f"qq_plot_epoch_{epoch}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
    else:
        plt.show()
        return None

def vcp_from_sample(model,databatch_list,empirical_databatch,train=False):
    """
    in order to have a shape [S,I,T] vs [I,T] the models concatenates all samples for each held out individuals
    which are of shape [S,B=1,I=1,T,1] (held out sample) -> [S,I,T] (required by vpc)
    """
    samples_list = model.sample(databatch_list,use_unique_times=True,num_samples=30)
    combined_observation = combine_samples([pair[0] for pair in samples_list])
    combined_times = combine_samples([pair[1] for pair in samples_list])
    print(combined_observation.shape)
    simulation_times = combined_times[0,0,:]
    print(simulation_times.shape)
    patients, patients_time = extract_context_by_mask(empirical_databatch)
    img = vpc(simulation_times, combined_observation, patients, patients_time,train=train)
    return img

def vpc_from_empirical(databatch_list,databatch_list_context,model,train=False,image_name="vpc.png",samples_number=100,y_scale=None):
    aicme = databatch_list_context[0]
    patients = [db_tuple[0].target_obs.cpu().detach().numpy() for db_tuple in databatch_list]
    patients_time = [db_tuple[0].target_obs_time.cpu().detach().numpy() for db_tuple in databatch_list]
    max_time_index = aicme.context_obs_mask.sum(axis=2).squeeze().argmax()
    all_samples_times = aicme.context_obs_time[0,max_time_index,aicme.context_obs_mask[0,max_time_index]]
    all_samples = []
    for db_tuple in databatch_list:
        samples,samples_time = model.sample_new_individual(db_tuple,samples_number)
        all_samples.append(samples)
    all_samples = torch.cat(all_samples,dim=2).squeeze()
    all_samples = all_samples[:,:,aicme.context_obs_mask[0,max_time_index]]
    vpc(all_samples_times, all_samples, patients, patients_time,train=train,image_name=image_name,y_scale=y_scale)

def vpc(test_time, MetaStudies, patients, patients_time, train=True, image_name="vpc.png", y_scale=None):
    """
    Generate a Visual Predictive Check (VPC) plot with PyTorch tensor inputs.
    
    Parameters:
    - test_time: 1D PyTorch tensor of fixed time points for simulated data (shape [T])
    - MetaStudies: 3D PyTorch tensor of simulated data (shape [M, P, T])
    - patients: List of 1D PyTorch tensors, each with observed concentrations
    - patients_time: List of 1D PyTorch tensors, each with corresponding times
    - train: If True, save plot; else show it
    - image_name: File name to save the image if train=True
    - y_scale: Set to "log" for log-scale y-axis; None for linear
    """
    if len(test_time.shape) > 1:
        test_time = test_time.squeeze()

    test_time_np = test_time.detach().cpu().numpy()
    MetaStudies_np = MetaStudies.detach().cpu().numpy()

    percentiles = [5, 25, 50, 75, 95]
    sim_percentiles = np.percentile(MetaStudies_np, percentiles, axis=1)  # [5, M, T]
    sim_percentiles_agg = np.percentile(sim_percentiles, 50, axis=1)      # [5, T]

    p05, p25, p50, p75, p95 = sim_percentiles_agg

    plt.figure(figsize=(10, 6))
    plt.fill_between(test_time_np, p05, p95, color='blue', alpha=0.2, label='5th-95th Percentile')
    plt.fill_between(test_time_np, p25, p75, color='blue', alpha=0.4, label='25th-75th Percentile')
    plt.plot(test_time_np, p50, color='blue', label='Median (50th Percentile)')

    for obs, times in zip(patients, patients_time):
        plt.scatter(times, obs, color='red', alpha=0.6, s=20)

    plt.xlabel('Time (hours)')
    plt.ylabel('Concentration (g/L)')
    if y_scale == "log":
        plt.yscale('log')

    plt.title('Visual Predictive Check (VPC)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    if train:
        plt.savefig(image_name)
        plt.close()
        return image_name
    else:
        plt.show()
        plt.close()


def get_unique_target_times(
    db_list: List[AICMECompartmentsDataBatch]
) -> TensorType["U", 1]:
    """
    Given P databatches, each with
        .target_obs_time: [B, t_ind, num_obs_t, 1]
    returns a tensor of shape [U, 1] containing the sorted unique times
    across *all* batches and *all* target time points.

    Args:
        db_list: list of length P of AICMECompartmentsDataBatch

    Returns:
        unique_times: Tensor of shape [U, 1], where U is the number of
                      unique target‐observation times across every batch.
    """
    # 1) Flatten each batch's times:
    #    db.target_obs_time.squeeze(-1).reshape(-1) has shape [B * t_ind * num_obs_t]
    flat_times = [
        db.target_obs_time.squeeze(-1).reshape(-1)  # [B * t_ind * num_obs_t]
        for db in db_list
    ]
    # 2) Concatenate all P batches → [(P * B * t_ind * num_obs_t)]
    all_times = torch.cat(flat_times, dim=0)

    # 3) Compute sorted unique values → [U]
    unique = torch.unique(all_times)

    # 4) Return as column vector → [U, 1]
    return unique.unsqueeze(-1).unsqueeze(0).unsqueeze(0)  # TensorType[1,1,"U", 1]