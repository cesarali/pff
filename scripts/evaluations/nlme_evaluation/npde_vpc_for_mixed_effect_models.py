"""VPC evaluation for mixed effect models.

This script consumes Simulx CSV outputs stored in ``simulation_results/`` and
builds VPC diagnostics from them.

Developer note
--------------
The datamodule fetches empirical studies from Hugging Face via
``datasets.load_dataset``. For convenience, this file includes a small example
call that downloads the Lenuzza study JSON payload
(``cesarali/lenuzza-2016``), so future scripts can reuse the same pattern.
"""

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset

from pff import reports_dir
from pff.data.data_empirical.simulx_to_json import simulx_to_json
from pff.metrics.sampling_quality import (
    compute_npde_data,
    compute_vpc_data,
    npde_pvalues,
    vpc_plot,
)

substance_info = [
    ("indometacin", "iv"),
    ("theophylline", "oral"),
    ("caffeine", "oral"),
    ("dextromethorphan", "oral"),
    ("digoxin", "oral"),
    ("memantine", "oral"),
    ("midazolam", "oral"),
    ("omeprazole", "oral"),
    ("paracetamol", "oral"),
    ("repaglinide", "oral"),
    ("rosuvastatin", "oral"),
    ("tolbutamide", "oral"),
]

# Simulated data loading and conversion

simulations_csv = [
    f"scripts/nlme_evaluation/simulation_results/simulatedData-{dosing_type}-{substance}.csv"
    for substance, dosing_type in substance_info
]


def csvpath_to_json(path):
    # Extract route and substance from file name, assuming format "simulatedData-{dosing_type}-{substance}.csv"
    parts = path.split("/")[-1].replace("simulatedData-", "").replace(".csv", "").split("-")
    if len(parts) == 2:
        dosing_type, substance = parts[0], parts[1]
    else:
        raise ValueError(f"Unexpected file name format: {path}")

    # Call the conversion function
    json = simulx_to_json(
        path,
        study_name=f"simulated_{dosing_type}_{substance}",
        substance_name=substance,
        dosing_type=dosing_type,
    )
    return json


simulations_json = [csvpath_to_json(path) for path in simulations_csv]


def load_empirical_study_json_from_hf(repo_id: str = "cesarali/lenuzza-2016", split: str = "train"):
    """Download JSON from Hugging Face datasets."""
    ds = load_dataset(repo_id, split=split)
    return [dict(study) for study in ds]  # pyright: ignore[reportCallIssue, reportArgumentType]


empirical_json_lenuzza_raw = load_empirical_study_json_from_hf("cesarali/lenuzza-2016")
empirical_json_indometacin_raw = load_empirical_study_json_from_hf("cesarali/Indometacin")
empirical_json_theophylline_raw = load_empirical_study_json_from_hf("cesarali/Theophylline")

# standardize capitalization
empirical_json_lenuzza_raw[15]["meta_data"]["substance_name"] = "caffeine"
empirical_json_indometacin_raw[0]["meta_data"]["substance_name"] = "indometacin"
empirical_json_theophylline_raw[0]["meta_data"]["substance_name"] = "theophylline"

empirical_json_raw = (
    empirical_json_lenuzza_raw + empirical_json_indometacin_raw + empirical_json_theophylline_raw
)

# align empirical data with simulated data for VPC comparison
lookup_empirical = {study["meta_data"]["substance_name"]: study for study in empirical_json_raw}
empirical_json = [
    lookup_empirical[name]
    for name in [substance for substance, _ in substance_info]
    if name in lookup_empirical
]


def convert_name_id_to_numeric(json_data):
    for study in json_data:
        for obs in study["context"]:
            obs["name_id"] = int(
                obs["name_id"].split("_")[-1]
            )  # Extract numeric part from "name_id" field
    return json_data


convert_name_id_to_numeric(empirical_json)
simulations_json = [convert_name_id_to_numeric(sim_json) for sim_json in simulations_json]


fig, axes = plt.subplots(
    nrows=3, ncols=4, figsize=(18, 12), sharex=False, sharey=False, squeeze=False
)

axes = axes.flatten()

for ax, emp_data, sim_data in zip(axes, empirical_json, simulations_json, strict=True):
    # Sanity check: ensure that the substance names match between empirical and simulated data
    emp_substance = emp_data["meta_data"]["substance_name"]
    sim_substance = sim_data[0]["meta_data"]["substance_name"]
    assert emp_substance == sim_substance, (
        f"Substance name mismatch: {emp_substance} vs {sim_substance}"
    )

    vpc_results = compute_vpc_data(emp_data, sim_data, n_bins=10, binning="equal_count")
    vpc_plot(vpc_results, ax=ax, log_y=False)
    ax.set_title(emp_substance.capitalize())

for ax in axes[1:]:
    ax.get_legend().remove()

fig.supxlabel("Time")
fig.supylabel("Concentration")
fig.suptitle("Visual Predictive Check (VPC) for Nonlinear Mixed Effect Models")
plt.tight_layout()
plt.savefig(reports_dir / "vpc_nlme_benchmark.png")


npde_values = [
    compute_npde_data(emp_data, sim_data)
    for emp_data, sim_data in zip(empirical_json, simulations_json, strict=True)
]
npde_statistics = [npde_pvalues(npde) for npde in npde_values]

# Convert to DataFrame for better visualization
npde_df = pd.DataFrame(npde_statistics)
npde_df["substance"] = [emp_data["meta_data"]["substance_name"] for emp_data in empirical_json]

# Save as CSV for further analysis
npde_df.to_csv(reports_dir / "npde_statistics.csv", index=False)
