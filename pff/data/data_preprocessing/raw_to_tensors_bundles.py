"""
Here we define the functions requiered to process the data

    https://pk-db.com/

"""
import torch
import numpy as np
import pandas as pd
from typing import Tuple
from torchtyping import TensorType
from typing import NamedTuple, List, Dict
from torchtyping import TensorType
from typing import Dict, Tuple, List
import numpy as np
import torch
from torchtyping import TensorType

from typing import Dict, Tuple, List, Optional
import numpy as np
import torch
from torchtyping import TensorType

lenuzza_doses_mg_per_g = {
    "memantine": 0.005,
    "omeprazole": 0.010,
    "repaglinide": 0.00025,
    "rosuvastatin": 0.005,
    "tolbutamide": 0.010,
    "dextromethorphan": 0.018,
    "digoxin": 0.00025,
    "paracetamol": 0.060,
    "caffeine": 0.073,
    "midazolam": 0.004,
    "paraxanthine":0.073,
    "dextrorphan":0.018,
}

class EmpiricalSubstanceTensorBundle(NamedTuple):
    observations:           TensorType["S", "I", "T"]  # padded concentration values
    times:                  TensorType["S", "I", "T"]  # padded normalized times [0,1]
    masks:                  TensorType["S", "I", "T"]  # 1 = observed, 0 = missing or padded
    individuals_mask:       TensorType["S", "I"]       # 1 = real subject, 0 = padded row
    study_names:            List[str]                  # [S] → one study name per substance
    individuals_names:      List[List[str]]            # [S][I] → subject name per padded subject
    substance_names:        List[str]                 # [S] substance_label entries
    mapping:                Dict[str, Dict[str, object]]
    dosing_amounts:         TensorType["S", "I"]      # dose mg/g per subject
    dosing_route_types:     TensorType["S", "I"]      # route type index per subject

def map_substance_to_index_and_study(
    drug_data_frame
) -> dict[str, dict[str, object]]:
    """
    Returns a dictionary mapping each substance_label to its index (in np.unique order)
    and its associated study_name (taken from the first row where that label appears).

    Returns
    -------
    dict: {
        "substance_label": {
            "index": int,
            "study_name": str
        },
        ...
    }
    """
    substance_labels = np.unique(drug_data_frame["substance_label"].values)

    mapping = {}
    for idx, label in enumerate(substance_labels):
        study_name = drug_data_frame.loc[
            drug_data_frame["substance_label"] == label, "study_name"
        ].iloc[0]
        mapping[label] = {
            "index": idx,
            "study_name": study_name
        }

    return mapping

def substances_csv_to_tensors(drug_data_frame, substance_label='omeprazole'):
    """
    The function groups by substance_label and obtains the time series 
    for each subject, pads when necessary, and returns observations, times, and masks.
    
    Params:
        drug_data_frame (pd.DataFrame): Input DataFrame with specified columns.
        substance_label (str): The substance label to filter by. Defaults to 'omeprazole'.
    
    Returns:
        observations (torch.Tensor): Padded observation values tensor of shape [num_subjects, max_time].
        observations_times (torch.Tensor): Padded time points tensor of shape [num_subjects, max_time].
        observations_mask (torch.Tensor): Mask tensor indicating valid data points, shape [num_subjects, max_time].
        dosing_amounts (torch.Tensor): Dose amount per subject [num_subjects].
        dosing_route_types (torch.Tensor): Route type index per subject [num_subjects].
    """
    # Filter the DataFrame by the given substance_label
    substance_data = drug_data_frame[drug_data_frame['substance_label'] == substance_label]

    # Group by subject_name
    subject_groups = substance_data.groupby('subject_name')

    # Collect sorted time and value arrays for each subject
    times_list = []
    values_list = []
    dosing_amounts_list = []
    route_list = []
    for subject_name, group in subject_groups:
        # Sort the group by 'time' to ensure chronological order
        sorted_group = group.sort_values('time')
        times = sorted_group['time'].values.astype(np.float32)
        values = sorted_group['value'].values.astype(np.float32)
        times_list.append(times)
        values_list.append(values)

        # Determine dosing amount based on substance name
        if 'substance_name' in group.columns:
            s_name = str(group['substance_name'].iloc[0]).lower()
        else:
            s_name = str(substance_label).lower()

        dose_value = 0.5
        for key, val in lenuzza_doses_mg_per_g.items():
            if key in s_name:
                dose_value = val
                break
        dosing_amounts_list.append(dose_value)
        route_list.append(0)  # oral
    
    # Determine the maximum time sequence length
    max_len = max(len(times) for times in times_list) if times_list else 0
    
    # Pad each subject's time and value arrays, and create the mask
    padded_times = []
    padded_values = []
    masks = []
    for times, values in zip(times_list, values_list):
        current_len = len(times)
        pad_len = max_len - current_len
        
        # Pad with zeros
        padded_time = np.pad(times, (0, pad_len), mode='constant', constant_values=0)
        padded_value = np.pad(values, (0, pad_len), mode='constant', constant_values=0)
        
        # Create mask (1 for real data, 0 for padding)
        mask = np.ones(max_len, dtype=np.float32)
        mask[current_len:] = 0
        
        padded_times.append(padded_time)
        padded_values.append(padded_value)
        masks.append(mask)
    
    # Convert to PyTorch tensors
    observations = torch.tensor(padded_values, dtype=torch.float32)      # [P, T]
    observations_times = torch.tensor(padded_times, dtype=torch.float32) # [P, T]
    observations_mask = torch.tensor(masks, dtype=torch.float32)         # [P, T]

    dosing_amounts = torch.tensor(dosing_amounts_list, dtype=torch.float32)  # [P]
    dosing_route_types = torch.tensor(route_list, dtype=torch.long)          # [P]

    return observations, observations_times, observations_mask, dosing_amounts, dosing_route_types

def substance_dict_to_tensors(
    selected_series: Optional[Dict[str, Dict[str, List[float]]]],
    hidden_series: Optional[Dict[str, Dict[str, List[float]]]],
) -> Tuple[
    Optional[TensorType["N_sel", "T"]], Optional[TensorType["N_sel", "T"]], Optional[TensorType["N_sel", "T"]],
    Optional[TensorType["N_hid", "T"]], Optional[TensorType["N_hid", "T"]], Optional[TensorType["N_hid", "T"]],
]:
    """
    Converts two dictionaries of time series into padded tensors, sharing a common maximum sequence length.
    Typically comming from the frontend payload

    Args:
        selected_series: Mapping subject_name -> {'timepoints': [...], 'values': [...]}.
        hidden_series:   Mapping subject_name -> {'timepoints': [...], 'values': [...]}.

    Returns:
        sel_obs, sel_times, sel_mask: [N_sel, T] or None.
        hid_obs, hid_times, hid_mask: [N_hid, T] or None.
    """
    def _extract_sorted(series: Dict[str, Dict[str, List[float]]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        times_list, values_list = [], []
        for subj, data in series.items():
            t = np.array(data['timepoints'], dtype=np.float32)
            v = np.array(data['values'], dtype=np.float32)
            idx = np.argsort(t)
            times_list.append(t[idx])
            values_list.append(v[idx])
        return times_list, values_list

    def _pad(times_list: List[np.ndarray], vals_list: List[np.ndarray], T: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        padded_times, padded_vals, masks = [], [], []
        for t, v in zip(times_list, vals_list):
            pad = T - len(t)
            t_pad = np.pad(t, (0, pad), mode='constant', constant_values=0)
            v_pad = np.pad(v, (0, pad), mode='constant', constant_values=0)
            mask  = np.ones(T, dtype=np.float32)
            mask[len(t):] = 0
            padded_times.append(t_pad)
            padded_vals.append(v_pad)
            masks.append(mask)
        return (
            torch.tensor(padded_vals, dtype=torch.float32),   # [N, T]
            torch.tensor(padded_times, dtype=torch.float32),  # [N, T]
            torch.tensor(masks, dtype=torch.float32),         # [N, T]
        )

    # Handle selected_series
    if selected_series:
        sel_times_list, sel_vals_list = _extract_sorted(selected_series)
        max_len_sel = max((len(t) for t in sel_times_list), default=0)
    else:
        sel_times_list = sel_vals_list = []
        max_len_sel = 0

    # Handle hidden_series
    if hidden_series:
        hid_times_list, hid_vals_list = _extract_sorted(hidden_series)
        max_len_hid = max((len(t) for t in hid_times_list), default=0)
    else:
        hid_times_list = hid_vals_list = []
        max_len_hid = 0

    # Determine shared max length
    T = max(max_len_sel, max_len_hid)

    # Pad or return None depending on presence of data
    if sel_times_list:
        sel_obs, sel_times, sel_mask = _pad(sel_times_list, sel_vals_list, T)
    else:
        sel_obs = sel_times = sel_mask = None

    if hid_times_list:
        hid_obs, hid_times, hid_mask = _pad(hid_times_list, hid_vals_list, T)
    else:
        hid_obs = hid_times = hid_mask = None

    return sel_obs, sel_times, sel_mask, hid_obs, hid_times, hid_mask

def substance_cvs_to_tensors_bundle(
    drug_data_frame: pd.DataFrame,
    **kwargs
) -> EmpiricalSubstanceTensorBundle:
    """
    Groups by substance_label and returns padded tensors for:
      - observations,
      - times (normalized per-substance to [0, 1]),
      - observation masks.

    Handles invalid (NaN) values in observations, applies optional normalization,
    and constructs per-substance tensors.

    Also returns metadata:
      - study_names: one per substance,
      - subject_names: one per subject (padded to max P).

    Returns:
        observations:     TensorType["S", "I", "T"]
        times:            TensorType["S", "I", "T"]
        masks:            TensorType["S", "I", "T"]
        subjects_mask:    TensorType["S", "I"]
        substance_labels: np.ndarray of length S
        mapping:          metadata dictionary
        study_names:      list of S strings
        subject_names:    list of S lists of I strings
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    substance_labels = np.unique(drug_data_frame["substance_label"].values)
    mapping = map_substance_to_index_and_study(drug_data_frame)

    substance_observations = []
    substance_times = []
    substance_masks = []
    subject_masks = []
    substance_doses = []
    substance_routes = []

    study_names_per_substance = []
    subject_names_per_substance = []

    max_time_steps = 0
    max_subjects = 0

    for substance_label in substance_labels:
        df_sub = drug_data_frame[drug_data_frame["substance_label"] == substance_label]
        obs, times, masks, doses, routes = substances_csv_to_tensors(
            drug_data_frame, substance_label=substance_label
        )
        # obs, times, masks: [P, T]

        valid_obs_mask = ~torch.isnan(obs)
        masks = masks.bool() & valid_obs_mask
        obs = obs.nan_to_num(nan=0.0)

        max_time_steps = max(max_time_steps, obs.shape[1])
        max_subjects = max(max_subjects, obs.shape[0])

        # --- Metadata collection ---
        grouped = df_sub.groupby("subject_name").first()
        subject_names = list(grouped.index)
        study_name = grouped["study_name"].iloc[0] if len(grouped) > 0 else ""

        study_names_per_substance.append(study_name)
        subject_names_per_substance.append(subject_names)

        substance_observations.append(obs)
        substance_times.append(times)
        substance_masks.append(masks)
        subject_masks.append(torch.ones(obs.shape[0], dtype=torch.float32))  # [P]
        substance_doses.append(doses)
        substance_routes.append(routes)

    # Padding pass
    all_observations, all_times, all_masks, all_subjects_mask = [], [], [], []
    all_doses, all_routes = [], []

    for obs, time, mask, subj_mask, subj_names, doses, routes in zip(
        substance_observations,
        substance_times,
        substance_masks,
        subject_masks,
        subject_names_per_substance,
        substance_doses,
        substance_routes,
    ):
        pad_subjects = max_subjects - obs.shape[0]
        pad_timesteps = max_time_steps - obs.shape[1]

        obs_padded = F.pad(obs, (0, pad_timesteps, 0, pad_subjects))         # [I, T]
        time_padded = F.pad(time, (0, pad_timesteps, 0, pad_subjects))       # [I, T]
        mask_padded = F.pad(mask, (0, pad_timesteps, 0, pad_subjects))       # [I, T]
        subj_mask_padded = F.pad(subj_mask, (0, pad_subjects))               # [I]
        dose_padded = F.pad(doses, (0, pad_subjects))          # [I]
        route_padded = F.pad(routes, (0, pad_subjects))        # [I]
        subj_names += [""] * pad_subjects                      # [I] → pad with ""

        all_observations.append(obs_padded)
        all_times.append(time_padded)
        all_masks.append(mask_padded)
        all_subjects_mask.append(subj_mask_padded)
        all_doses.append(dose_padded)
        all_routes.append(route_padded)

    return EmpiricalSubstanceTensorBundle(
        observations=torch.stack(all_observations),       # [S, I, T]
        times=torch.stack(all_times),                     # [S, I, T]
        masks=torch.stack(all_masks),                     # [S, I, T]
        individuals_mask=torch.stack(all_subjects_mask),     # [S, I]
        substance_names=list(substance_labels),                # [S]
        mapping=mapping,
        study_names=study_names_per_substance,            # [S]
        individuals_names=subject_names_per_substance,         # [S][I]
        dosing_amounts=torch.stack(all_doses),            # [S, I]
        dosing_route_types=torch.stack(all_routes)        # [S, I]
    )
