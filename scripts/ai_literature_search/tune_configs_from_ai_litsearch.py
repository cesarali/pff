# Match key metrics from the AI-based literature search to the synthetic data generation.

#%% Imports and setup

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

#%% Literature data loading and processing

# Get the key metrics from the AI-based literature search (e.g., from a JSON file or API response)
json_path = "scripts/ai_literature_search/search_results/extracted_pk_information_V2.json"

with open(json_path, 'r') as file:
    data = json.load(file)

# Format the data to a data frame for easier plotting
df = pd.DataFrame(data)

# Unpack pk_parameters dictionary into separate columns
pk_params_df = pd.json_normalize(df['pk_parameters'])
df = pd.concat([df.drop(columns=['pk_parameters']), pk_params_df], axis=1)

# Unit conversion dictionaries
conc_to_mg_per_L = {
    "ng/mL": 1e3,
    "μg/mL": 1,
    "mg/L": 1,
    "pg/mL": 1e6,
    "µg/mL": 1,
    "%": 1,          # Percentage values are already in the correct format (CV% as a percentage), we can keep them as is.
    "percent": 1,
}

dose_to_mg = {
    "mg": 1,
    "g": 1e-3,
    "μg": 1e3,
}

auc_to_h_mg_per_L = {
    "h·ng/mL": 1e3,
    "h×ng/mL": 1e3,
    "ng*h/mL": 1e3,
    "μg/mL/hour": 1,
    "ng·h/mL": 1e3,
    "μg·h/mL": 1,
    "ng·hr/mL": 1e3,
    "ng×h/mL": 1e3,
    "mg/L*h": 1,
    "pg·h/mL": 1e6,
    "ng h/mL": 1e3,
    "µg·hr/mL": 1,
    "μg*h/mL": 1,
    "ng/mL*h": 1e3,
    "h*ng/mL": 1e3,
    "pg*h/mL": 1e6,
    "h*mg/L": 1,
    "ng.hour/mL": 1e3,
    "hr*ng/mL": 1e3,
    "μg/mL·h": 1,
    "mg*h/L": 1,
    "μg h/mL": 1,
    "μg/mL.h": 1,
    "hr*μg/mL": 1,
    "µg·h/mL": 1,
    "µg h/mL": 1,
    "hr × ng/mL": 1e3,
    "µg⋅h/mL": 1,
    "ng-h/L": 1e3,
    "h·pg/mL": 1e6,
    "μg.h/mL": 1,
    "h·µg/mL": 1,
    "μg/mL*h": 1,
    "%": 1,          # Percentage values are already in the correct format (CV% as a percentage), we can keep them as is.
    "percent": 1,
}

time_to_h = {
    "min": 1/60,
    "h": 1,
    "hr": 1,
    "hrs": 1,
    "hour": 1,
    "hours": 1,
    "days": 24,
}

# Unit standardization (dose, AUC central tendency, Cmax central tendency, Tmax central tendency, AUC dispersion, Cmax dispersion);
# Tmax dispersion comes later since it has some special handling due to the different dispersion types)
df["dose"] /= df["dose_unit"].map(dose_to_mg)
df["AUC0_t.central_tendency"] /= df["AUC0_t.central_tendency_unit"].map(auc_to_h_mg_per_L)
df["Cmax.central_tendency"] /= df["Cmax.central_tendency_unit"].map(conc_to_mg_per_L)
df["Tmax.central_tendency"] /= df["Tmax.central_tendency_unit"].map(time_to_h)

df["AUC0_t.dispersion"] = pd.to_numeric(df["AUC0_t.dispersion"], errors="coerce")
df["Cmax.dispersion"] = pd.to_numeric(df["Cmax.dispersion"], errors="coerce")
df["AUC0_t.dispersion"] /= df["AUC0_t.dispersion_unit"].map(auc_to_h_mg_per_L)
df["Cmax.dispersion"] /= df["Cmax.dispersion_unit"].map(conc_to_mg_per_L)

df["last_sample_time"] = pd.to_numeric(df["last_sample_time"], errors="coerce")
df["last_sample_time"] /= df["last_sample_time_unit"].map(time_to_h)

# Values above the last sample time are not physiologically plausible, likely due to errors in the literature or unit conversion, hence we set them to NaN to exclude them from the comparison
df.loc[df['Tmax.central_tendency'] > df['last_sample_time'],'Tmax.central_tendency'] = np.nan

# For Tmax dispersion, if the dispersion type is "range" or "min-max", we can calculate an approximate SD by assuming a normal
# distribution and using the formula: SD ≈ range / 4 (since ~95% of data falls within ±2 SD in a normal distribution).
def range_to_sd(val):
    try:
        low, high = map(float, val.split('-'))
        return (high - low) / 4
    except:
        return np.nan

df["Tmax_SD"] = np.where(
    df["Tmax.dispersion_type"].isin(["range","min-max"]),
    df["Tmax.dispersion"].apply(range_to_sd),
    pd.to_numeric(df["Tmax.dispersion"], errors="coerce") # already SD
)
df["Tmax_SD"] /= df["Tmax.dispersion_unit"].map(time_to_h)


df["AUC0_t.AUC0_t.central_tendency_unit"] = "h·mg/L"
df["Cmax.central_tendency_unit"] = "mg/L"
df["Tmax.central_tendency_unit"] = "h"
df["dose_unit"] = "mg"

# Calculate dose-normalized AUC and Cmax
df['nAUC_central_tendency'] = df['AUC0_t.central_tendency'] / df['dose']
df['nCmax_central_tendency'] = df['Cmax.central_tendency'] / df['dose']

# Calculate AUC and Cmax variability (in %CV)
df['AUC_CV'] = np.nan
df['Cmax_CV'] = np.nan

mask_AUC_sd = df["AUC0_t.dispersion_type"] == "SD"
mask_Cmax_sd = df["Cmax.dispersion_type"] == "SD"
mask_AUC_cv = df["AUC0_t.dispersion_type"].isin(["CV","CV%"])
mask_Cmax_cv = df["Cmax.dispersion_type"].isin(["CV","CV%"])

df.loc[mask_AUC_sd,  'AUC_CV']  = 100 * df.loc[mask_AUC_sd, 'AUC0_t.dispersion'] / df.loc[mask_AUC_sd, 'AUC0_t.central_tendency']
df.loc[mask_Cmax_sd, 'Cmax_CV'] = 100 * df.loc[mask_Cmax_sd, 'Cmax.dispersion'] / df.loc[mask_Cmax_sd, 'Cmax.central_tendency']
df.loc[mask_AUC_cv,  'AUC_CV']  = df.loc[mask_AUC_cv, 'AUC0_t.dispersion']
df.loc[mask_Cmax_cv, 'Cmax_CV'] = df.loc[mask_Cmax_cv, 'Cmax.dispersion']

# Calculate Tmax variability (in %CV)
df['Tmax_CV']  = 100 * df['Tmax_SD'] / df['Tmax.central_tendency']

# Use only oral administration data for comparison with the simulations (since the current simulations are for oral administration)
df = df[df["administration_route"] == "oral"]

# Final literature data frame for comparison
lit_df = df[["nAUC_central_tendency", "nCmax_central_tendency", "Tmax.central_tendency", "AUC_CV", "Cmax_CV", "Tmax_CV", "number_of_patients"]].rename(
    columns={
        "nAUC_central_tendency": "nAUC",
        "nCmax_central_tendency": "nCmax",
        "Tmax.central_tendency": "Tmax",
        "AUC_CV": "AUC_cv",
        "Cmax_CV": "Cmax_cv",
        "Tmax_CV": "Tmax_cv",
        "number_of_patients": "nID"
    }
)

#%% ----- Simulation data according to current configurations and a tunable config -----

stats_calculator = ListedObservationStats()

# Load the current simulated data for speed, no need to simulate again 
# every time I explore something below
data_path = data_dir / "preprocessed" / "ensemble.json"
current_studies = json.loads(data_path.read_text())


# Load the configs (same as in the test, but we will modify them below for tuning)
experiment_dir = config_dir / "experiment_configs" / "node-pk"
meta_study_cfg = MetaStudyConfig.from_yaml(experiment_dir / "base-homogeneous.meta_study.yaml")
dosing_cfg = MetaDosingConfig.from_yaml(experiment_dir / "base-homogeneous.dosing.yaml")
context_obs_cfg = ObservationsConfig.from_yaml(
    experiment_dir / "base-homogeneous.observations.yaml",
    section="context_observations",
)

meta_study_cfg.k_a_tmag_range = (0.01, 0.1)
meta_study_cfg.k_e_tmag_range = (0.01, 0.1)
meta_study_cfg.V_tmag_range = (0.001, 0.01) # not a kinetic parameter, hence less variability
meta_study_cfg.k_1p_tmag_range = (0.01, 0.1)
meta_study_cfg.k_p1_tmag_range = (0.01, 0.1)

meta_study_cfg.rel_ruv_range = (0.01, 0.1)

meta_study_cfg.log_V_mean_range = (1,7)

meta_study_cfg.log_V_std_range = (0.15, 0.45)
meta_study_cfg.log_k_e_std_range = (0.15, 0.45)
meta_study_cfg.log_k_a_std_range = (0.15, 0.45)
meta_study_cfg.log_k_1p_std_range = (0.15, 0.45)
meta_study_cfg.log_k_p1_std_range = (0.15, 0.45)

meta_dosing_cfg = MetaDosingConfig()
meta_dosing_cfg.route_weights = [1.0,0.0] # oral dosing only for comparability to literature data

obs_cfg = ObservationsConfig()

number_of_samples = 1000

tuned_studies, _ = prepare_ensemble_of_simulations(
    meta_study_config=meta_study_cfg,
    observation_config=obs_cfg,
    meta_dosing_config=meta_dosing_cfg,
    number_of_samples=number_of_samples,
)

def study_to_df(studies):
    stats = stats_calculator.compute_study_population_statistics(studies)
    df = pd.DataFrame(stats)[["nAUC_mean_list", "nCmax_mean_list", "nAUC_cv_list", "nCmax_cv_list", "Tmax_mean_list", "Tmax_cv_list", "nID_list"]].rename(
        columns={
            "nAUC_mean_list": "nAUC",
            "nCmax_mean_list": "nCmax",
            "nAUC_cv_list": "AUC_cv",
            "nCmax_cv_list": "Cmax_cv",
            "Tmax_mean_list": "Tmax",
            "Tmax_cv_list": "Tmax_cv",
            "nID_list": "nID",
        }
    )
    return df

current_df = study_to_df(current_studies)
tuned_df = study_to_df(tuned_studies)

#%% Create a joint object for plotting
datasets = {
    "Literature": lit_df,
    # "Current": current_df,
    "Simulation": tuned_df,
}

metrics = ["nAUC", "nCmax", "Tmax", "AUC_cv", "Cmax_cv", "Tmax_cv"]

cleaned_data = {}
for name, data in datasets.items():
    cleaned_data[name] = {
        metric: data[metric].dropna().values
        for metric in metrics
    }

nAUC_data = [cleaned_data[name]["nAUC"] for name in datasets]
nCmax_data = [cleaned_data[name]["nCmax"] for name in datasets]
Tmax_data = [cleaned_data[name]["Tmax"] for name in datasets]
AUC_cv_data = [cleaned_data[name]["AUC_cv"] for name in datasets]
Cmax_cv_data = [cleaned_data[name]["Cmax_cv"] for name in datasets]
Tmax_cv_data = [cleaned_data[name]["Tmax_cv"] for name in datasets]

#%% ---- Plotting the comparison of the key metrics between the AI literature search and the simulated data (current and tuned) -----


fig, axes = plt.subplots(2, 3, figsize=(12, 6))
colors = ["skyblue", "lightgreen", "salmon"]  # one per dataset
labels = list(datasets.keys())  # for xticks

def plot_box(ax, data, title, ylabel, log=False, ylim=None):
    b = ax.boxplot(data, patch_artist=True)
    for patch, color in zip(b['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(ylim)

# Top-left: nAUC, Top-right: nCmax, Bottom-left: AUC CV, Bottom-right: Cmax CV
plot_box(axes[0,0], nAUC_data, "Dose-Normalized Mean AUC", "h/L", log=True)
plot_box(axes[0,1], nCmax_data, "Dose-Normalized Mean Cmax", "1/L", log=True)
plot_box(axes[0,2], Tmax_data, "Mean Tmax", "h", log=True)
plot_box(axes[1,0], AUC_cv_data, "AUC Variability", "%CV", ylim=(0,120))
plot_box(axes[1,1], Cmax_cv_data, "Cmax Variability", "%CV", ylim=(0,120))
plot_box(axes[1,2], Tmax_cv_data, "Tmax Variability", "%CV", ylim=(0,120))

plt.tight_layout()
plt.show()

#%%  ------- Individual plots to see how the generated samples look like ------

sample_to_plot = tuned_studies[0:9]
num_cols = 3
num_rows = 3
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8), squeeze=False)

total_plots = num_rows * num_cols

for idx, study in enumerate(sample_to_plot):
    row = idx // num_cols
    col = idx % num_cols
    ax = axes[row][col]

    context = study.get("context", [])
    for individual in context:
        times = individual.get("observation_times", [])
        observations = individual.get("observations", [])
        if times and observations:
            ax.plot(times, observations, marker="o", linestyle="-")

    study_meta = study.get("meta_data", {})
    study_name = study_meta.get("study_name", f"study_{idx}")
    ax.set_title(study_name)
    ax.set_yscale("log")
    ax.set_xlabel("Time")
    ax.set_ylabel("Observation")
    ax.grid(True, alpha=0.3)

for remaining_idx in range(len(sample_to_plot), total_plots):
    row = remaining_idx // num_cols
    col = remaining_idx % num_cols
    axes[row][col].axis("off")

fig.tight_layout()
plt.show()
