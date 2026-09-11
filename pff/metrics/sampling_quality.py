# Evaluate sampling quality of a model based on Visual Predictive Checks or Normalized Prediction Distribution Errors (NPDEs).
# Input for both evaluations: a StudyJSON object containing the observed data and a List[StudyJSON] containing replicates of simulated data from the model.
# This way, both neural networks and NLME models can be evaluated using the same code, as long as they can produce the required StudyJSON objects.

from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, shapiro, ttest_1samp

from pff.data.data_empirical.json_schema import IndividualJSON, StudyJSON


def json_to_dataframe(study_json: StudyJSON) -> pd.DataFrame:
    """
    Convert a StudyJSON object to a pandas DataFrame for easier analysis.

    Args:
        study_json (StudyJSON): The StudyJSON object to convert.
    Returns:
        pd.DataFrame: A DataFrame with columns ["Type", "ID", "Time", "Value"] from the StudyJSON data.
    """

    frames = []

    for data_type in ["context", "target"]:
        entries = study_json.get(data_type, [])

        for j, entry in enumerate(entries):
            # Prefer name_id, else _id, else a deterministic fallback
            name_id = entry.get("name_id") or entry.get("_id") or f"{data_type}_{j}"

            df_entry = pd.DataFrame(
                {
                    "Type": data_type,
                    "ID": str(name_id),  # ensure it's a string
                    "Time": entry["observation_times"],
                    "Value": entry["observations"],
                }
            )
            frames.append(df_entry)

    if frames:
        return pd.concat(frames, ignore_index=True)
    else:
        return pd.DataFrame(columns=["Type", "ID", "Time", "Value"])


def json_list_to_dataframe(study_list: List[StudyJSON]) -> pd.DataFrame:
    """
    Convert a list of StudyJSON objects to a pandas DataFrame for easier analysis.

    Args:
        study_list (List[StudyJSON]): The list of StudyJSON objects to convert.

    Returns:
        pd.DataFrame: A DataFrame with columns ["Type", "ID", "Time", "Value", "Replicate"] from the StudyJSON data.
    """

    frames = []

    for replicate_idx, study in enumerate(study_list):
        df = json_to_dataframe(study)
        df["Replicate"] = replicate_idx
        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def validate_npde_vpc_inputs(
    data: pd.DataFrame, simulations: List[pd.DataFrame], differentTimesError: bool = True
) -> None:
    """
    Validate the inputs for NPDE / VPC calculation.

    Args:
        data (pd.DataFrame): The observed data in DataFrame format with columns ["Type", "ID", "Time", "Value"].
        simulations (List[pd.DataFrame]): A list of DataFrames with columns ["Type", "ID", "Time", "Value","Replicate"]
        representing simulated data from the model.
        differentTimesError (bool): Whether to raise an error if observation times differ between individuals (default: True).

    Returns:
        None: If the inputs are valid, otherwise raises a ValueError.
    """

    key_cols = ["Type", "ID", "Time"]

    obs_keys = data[key_cols].drop_duplicates().sort_values(key_cols).reset_index(drop=True)

    pred_keys = simulations[key_cols].drop_duplicates().sort_values(key_cols).reset_index(drop=True)  # type: ignore

    if not obs_keys.equals(pred_keys):
        raise ValueError("Observations and predictions are not structurally identical.")

    if differentTimesError:
        if (data.groupby("ID")["Time"].apply(lambda x: tuple(sorted(x))).nunique()) != 1:
            raise ValueError("Observation times differ between individuals.")

    return None


def compute_npde_data(data: StudyJSON, simulations: List[StudyJSON]) -> np.ndarray:
    """
    Compute Normalized Prediction Distribution Errors (NPDEs) for a given StudyJSON and a list of simulated StudyJSONs.

    Args:
        data (StudyJSON): The observed data in StudyJSON
        simulations (List[StudyJSON]): A list of StudyJSON objects representing simulated data from the model.

    Returns:
        np.ndarray: An array of NPDE values.
    """
    # Extract observed values and predicted values from the StudyJSON objects and validate them before calculating NPDEs.
    observed_values = json_to_dataframe(data)
    predicted_values = json_list_to_dataframe(simulations)

    validate_npde_vpc_inputs(observed_values, predicted_values, differentTimesError=False)

    # Merge observations and predictions into a single DataFrame for NPDE calculation.
    key_cols = ["Type", "ID", "Time"]

    pred_wide = predicted_values.pivot(index=key_cols, columns="Replicate", values="Value")

    obs_indexed = observed_values.set_index(key_cols)

    combined = pred_wide.join(obs_indexed["Value"].rename("Observed"))

    # Calculate NPDEs for each replicate and return as a numpy array.
    replicate_cols = pred_wide.columns

    pred_vals = combined[replicate_cols].values
    obs_vals = combined["Observed"].values

    # Empirical CDF (truncated to avoid 0 and 1) for each observation based on the predicted distribution from the replicates.
    pde = (pred_vals <= obs_vals[:, None]).sum(axis=1) / (len(replicate_cols) + 1) + 0.5 / (
        len(replicate_cols) + 1
    )
    npde = norm.ppf(pde)

    return npde


def npde_plot(npde_values: np.ndarray) -> None:
    """
    Create a quantile-quantile-plot of NPDE values.

    Args:
        npde_values (np.ndarray): An array of NPDE values to plot.

    Returns:
        None
    """
    plt.figure(figsize=(6, 6))
    plt.title("Q-Q Plot of NPDE Values")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Empirical Quantiles")
    norm_qq = np.sort(npde_values)
    theoretical_qq = norm.ppf((np.arange(len(npde_values)) + 1) / (len(npde_values) + 1))
    plt.plot(theoretical_qq, norm_qq, marker="o", linestyle="")
    plt.plot(theoretical_qq, theoretical_qq, color="red", linestyle="--")
    plt.grid()
    plt.show()


def npde_pvalues(npde_values: np.ndarray) -> dict:
    """
    Calculate p-values based on the theoretical N(0,1) distribution of NPDE values.

    Args:
        npde_values (np.ndarray): An array of NPDE values to summarize.

    Returns:
        dict: A dictionary containing p-values for different tests applied to the NPDE values:
            - "mean": The (one-sample) t-test for zero mean of the NPDE values.
            - "variance": The (one-sample) chi-squared test for unit variance of the NPDE values.
            - "normality": The Shapiro-Wilk test for normality of the NPDE values.
    """

    # variance test not implemented in scipy, so we calculate the p-value manually based on
    # the chi-squared distribution of the sample variance under the null hypothesis of unit variance.
    n = len(npde_values)
    sample_var = np.var(npde_values, ddof=1)
    chi2_stat = (n - 1) * sample_var
    p_lower = chi2.cdf(chi2_stat, df=n - 1)
    p_upper = 1 - p_lower
    p_var = 2 * min(p_lower, p_upper)

    return {
        "mean": ttest_1samp(npde_values, 0).pvalue,  # type: ignore
        "variance": p_var,
        "normality": shapiro(npde_values).pvalue,
    }


def compute_vpc_data(
    data: StudyJSON,
    simulations: Sequence[StudyJSON],
    quantiles: List[float] = [0.05, 0.5, 0.95],
    confidence: float = 0.9,
    n_bins: Optional[int] = None,
    binning: str = "equal_count",  # "equal_count" or "equal_width"
) -> pd.DataFrame:
    """
    Compute data for a Visual Predictive Check (VPC) plot for the given StudyJSON and a list of simulated StudyJSONs.

    Args:
        data (StudyJSON): The observed data in StudyJSON
        simulations (List[StudyJSON]): A list of simulated data in StudyJSON format.
        quantiles (List[float]): The quantiles to display in the VPC plot (default: [0.05, 0.5, 0.95]).
        confidence (float): The confidence level for the prediction intervals (default: 0.9).
    Returns:
        pd.DataFrame: A DataFrame containing the VPC data.
    """

    observed_values = json_to_dataframe(data)
    predicted_values = json_list_to_dataframe(simulations)

    alpha_low = (1 - confidence) / 2
    alpha_high = 1 - alpha_low

    # --------------------------------
    # Binning (if requested OR if needed)
    # --------------------------------
    if n_bins is not None:
        validate_npde_vpc_inputs(observed_values, predicted_values, differentTimesError=False)

        match binning:
            case "equal_count":
                observed_values["TimeBin"] = pd.qcut(
                    observed_values["Time"], q=n_bins, duplicates="drop"
                )

                # Use same bin edges for predicted
                bins = observed_values["TimeBin"].cat.categories
                predicted_values["TimeBin"] = pd.cut(predicted_values["Time"], bins=bins)

            case "equal_width":
                tmin = observed_values["Time"].min()
                tmax = observed_values["Time"].max()
                bins = np.linspace(tmin, tmax, n_bins + 1)

                observed_values["TimeBin"] = pd.cut(
                    observed_values["Time"], bins=bins, include_lowest=True
                )
                predicted_values["TimeBin"] = pd.cut(
                    predicted_values["Time"], bins=bins, include_lowest=True
                )

            case _:
                raise ValueError("binning must be 'equal_width' or 'equal_count'")

        # Use bin midpoint for plotting
        bin_midpoints = (
            observed_values.groupby("TimeBin", observed=False)["Time"].mean().rename("Time")
        )

        # Replace Time with bin midpoint
        observed_values["Time"] = observed_values["TimeBin"].map(bin_midpoints)
        predicted_values["Time"] = predicted_values["TimeBin"].map(bin_midpoints)

        # Drop bin column
        observed_values = observed_values.drop(columns="TimeBin")
        predicted_values = predicted_values.drop(columns="TimeBin")

    else:
        validate_npde_vpc_inputs(observed_values, predicted_values, differentTimesError=True)  # type: ignore

    # --------------------------------
    # Quantile calculation
    # --------------------------------
    vpc_obs = (
        observed_values.groupby("Time")["Value"]
        .quantile(quantiles)  # type: ignore
        .rename("Obs")
        .reset_index()
        .rename(columns={"level_1": "Quantile"})
    )

    vpc_pred = (
        predicted_values.groupby(["Time", "Replicate"])["Value"]
        .quantile(quantiles)  # type: ignore
        .rename("SimQuantile")
        .reset_index()
        .rename(columns={"level_2": "Quantile"})
        .groupby(["Time", "Quantile"])["SimQuantile"]
        .quantile([alpha_low, alpha_high])  # type: ignore
        .rename("VPC")
        .reset_index()
        .rename(columns={"level_2": "PI"})
        .pivot(index=["Time", "Quantile"], columns="PI", values="VPC")
        .reset_index()
        .rename(columns={alpha_low: "LowerPred", alpha_high: "UpperPred"})
    )

    vpc_data = vpc_obs.merge(vpc_pred, on=["Time", "Quantile"], how="left")

    return vpc_data


def vpc_plot(vpc_data: pd.DataFrame, ax=None, log_y: bool = False) -> None:
    """
    Create a Visual Predictive Check (VPC) plot for the given VPC data.

    Args:
        vpc_data (pd.DataFrame): The VPC data to plot.
        ax: Optional matplotlib axis to plot on. If None, a new figure and axis will be created.
        log_y: Whether to use a logarithmic scale for the y-axis (default: False).

    Returns:
        None
    """

    quantiles = np.sort(vpc_data["Quantile"].unique())

    # Enforce exactly 3 quantiles
    if len(quantiles) != 3:
        raise ValueError(f"Expected exactly 3 quantiles, got {len(quantiles)}: {quantiles}")

    # Default axis management
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Log-scale option
    if log_y:
        # Safety check: log scale requires strictly positive values
        y_cols = ["Obs", "LowerPred", "UpperPred"]
        if (vpc_data[y_cols] <= 0).any().any():
            raise ValueError("Log scale requested but non-positive values detected.")
        ax.set_yscale("log")

    # Color scheme: lower, median, upper
    colors = ["tab:blue", "tab:orange", "tab:blue"]

    # Map sorted quantiles to colors
    q_to_color = dict(zip(quantiles, colors))

    # Plot observed quantiles
    quantiles = vpc_data["Quantile"].unique()
    for q in quantiles:
        subset = vpc_data[vpc_data["Quantile"] == q]

        color = q_to_color[q]
        is_median = np.isclose(q, 0.5)

        ax.plot(
            subset["Time"],
            subset["Obs"],
            marker="o",
            color=color,
            linewidth=2 if is_median else 1,
            label=f"Observed {q:.0%}",
        )

        ax.fill_between(
            subset["Time"],
            subset["LowerPred"],
            subset["UpperPred"],
            color=color,
            alpha=0.25,
            label=f"Simulated {q:.0%} PI",
        )

    # Keep legend outside the plotting area to avoid occluding trajectories.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
    )
    ax.figure.subplots_adjust(bottom=0.25)
    return ax


if __name__ == "__main__":
    # Example usage
    observed_data = StudyJSON(
        context=[
            IndividualJSON(name_id="1", observation_times=[0, 1, 2], observations=[10, 20, 30]),
            IndividualJSON(name_id="2", observation_times=[0, 1, 2], observations=[11, 21, 31]),
        ]
    )

    simulated_data = [
        StudyJSON(
            context=[
                IndividualJSON(name_id="1", observation_times=[0, 1, 2], observations=[12, 22, 32]),
                IndividualJSON(name_id="2", observation_times=[0, 1, 2], observations=[13, 21, 30]),
            ]
        ),
        StudyJSON(
            context=[
                IndividualJSON(name_id="1", observation_times=[0, 1, 2], observations=[8, 18, 28]),
                IndividualJSON(name_id="2", observation_times=[0, 1, 2], observations=[11, 19, 27]),
            ]
        ),
    ]
    # Convert to dataframes for visualization (optional)
    observed_values = json_to_dataframe(observed_data)
    simulated_values = json_list_to_dataframe(simulated_data)

    validate_npde_inputs(observed_values, simulated_values)
    npde_results = calculate_npde(observed_data, simulated_data)

    print("NPDE Results:", npde_results)

    vpc_data = create_vpc_data(observed_data, simulated_data)

    vpc_plot(vpc_data)
