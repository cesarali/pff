"""This is used for calculating summary statistics over ensembles of StudyJSONs to check that
the distribution of simulated data matches empirical data."""

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

from pff.data.data_empirical.json_schema import IndividualJSON, StudyJSON


class StudyPopulationStats(ABC):
    """Abstract interface for computing and aggregating statistics over ensembles of StudyJSONs."""

    @abstractmethod
    def compute_per_individual(self, ind: IndividualJSON) -> Dict[str, float]:
        """Compute statistics for a single individual (e.g., min/max observation value, count)."""

    @abstractmethod
    def compute_per_study(self, study: StudyJSON) -> Dict[str, float]:
        """Compute statistics for a single study (e.g., min/max observation value, count)."""

    @abstractmethod
    def aggregate(
        self,
        per_study: List[Dict[str, float]],
    ) -> Dict[str, object]:
        """Aggregate statistics across studies (e.g., global extrema, averages, or histograms)."""

    def compute_study_population_statistics(
        self,
        studies: List[StudyJSON],
    ) -> Dict[str, object]:
        """Compute and aggregate statistics for a StudyJSON ensemble."""
        per_study = [self.compute_per_study(study) for study in studies]
        return self.aggregate(per_study)


class BasicObservationStats(StudyPopulationStats):
    """Compute descriptive statistics for observation values across individuals.
    For each individual, computes:
    - nAUC: Area Under the Curve (AUC), normalized by dose, using trapezoidal rule.
    - nCmax: Maximum observed concentration, normalized by dose.
    - Tmax: Time at which Cmax occurs.
    - Nobs: Number of observations.
    - Duration: Duration of the observation period (max observation time).
    For each study, computes:
    - Mean and standard deviation of nAUC, nCmax, Tmax across individuals.
    - Mean and total number of observations (Nobs) across all individuals.
    - Total study duration (max Duration across individuals).
    Aggregates across studies to provide percentiles of each study-level statistic.
    """

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def compute_per_individual(self, ind: IndividualJSON) -> Dict[str, float]:
        obs_vals = ind.get("observations", [])
        obs_times = ind.get("observation_times", [])
        dose = ind.get("dosing", [])
        dosing_time = ind.get("dosing_times", [])
        route = ind.get("dosing_type", [])

        if not obs_vals:
            return {"nAUC": np.nan, "nCmax": np.nan, "Tmax": np.nan, "Nobs": 0, "Duration": np.nan}

        # Check that input times are sorted and match the number of observations
        if len(obs_times) != len(obs_vals) or any(
            obs_times[i] >= obs_times[i + 1] for i in range(len(obs_times) - 1)
        ):
            raise ValueError(
                "Observation times must be sorted and match the number of observations."
            )

        # Check that there is only a single positive dose
        if len(dose) != 1 or len(dosing_time) != 1 or len(route) != 1:
            raise ValueError("Only single dosing is supported in this statistic.")
        if dose[0] <= 0 or np.isnan(dose) or np.isnan(dosing_time[0]):
            raise ValueError("Dose must be positive.")

        # Check that dose precedes observations
        if any(t < dosing_time[0] for t in obs_times):
            raise ValueError("Dosing time must precede observation times.")

        # calculate AUC using the trapezoidal rule:
        # - for oral dosing, add a value of 0 at dosing time
        # - for iv bolus, add the first observation at dosing time

        obs_times_trapz = dosing_time + obs_times
        if route[0] == "oral":
            obs_vals_trapz = [0.0] + obs_vals
        elif route[0] == "iv":
            obs_vals_trapz = [obs_vals[0]] + obs_vals
        else:
            raise ValueError("Only 'oral' and 'iv' dosing types are supported.")

        auc = np.trapezoid(obs_vals_trapz, obs_times_trapz) if len(obs_vals) > 0 else np.nan
        auc /= dose[0]

        # Calculate Cmax and Tmax
        Cmax_idx = np.argmax(obs_vals)
        Cmax = obs_vals[Cmax_idx]
        Tmax = obs_times[Cmax_idx]
        Cmax /= dose[0]

        return {
            "nAUC": float(auc),
            "nCmax": float(Cmax),
            "Tmax": float(Tmax),
            "Nobs": len(obs_vals),
            "Duration": np.max(obs_times),
        }

    def compute_per_study(self, study: StudyJSON) -> Dict[str, float]:
        ind_stats = [
            self.compute_per_individual(ind)
            for block in ("context", "target")
            for ind in study.get(block, [])
        ]
        if not ind_stats:
            return {"max_obs": np.nan, "min_obs": np.nan, "mean_obs": np.nan, "num_obs": 0}

        # Calculate statistics (maybe a bit too much, can be simplified later)
        metrics = {
            "nAUC_mean": ("nAUC", np.mean),
            "nAUC_sd": ("nAUC", np.std),
            "nAUC_cv": ("nAUC", lambda x: np.std(x) / np.mean(x) * 100 if np.mean(x) != 0 else np.nan),
            "nCmax_mean": ("nCmax", np.mean),
            "nCmax_sd": ("nCmax", np.std),
            "nCmax_cv": ("nCmax", lambda x: np.std(x) / np.mean(x) * 100 if np.mean(x) != 0 else np.nan),
            "Tmax_mean": ("Tmax", np.mean),
            "Tmax_sd": ("Tmax", np.std),
            "Tmax_cv": ("Tmax", lambda x: np.std(x) / np.mean(x) * 100 if np.mean(x) != 0 else np.nan),
            "Nobs_mean": ("Nobs", np.mean),
            "Nobs_total": ("Nobs", np.sum),
            "Duration_max": ("Duration", np.max),
            "nID": ("Nobs", lambda x: len(x)),
        }

        results = {name: func([d[key] for d in ind_stats]) for name, (key, func) in metrics.items()}

        # Ensure all values are floats for JSON-friendliness or downstream compatibility
        return {k: float(v) for k, v in results.items()}

    def aggregate(
        self,
        per_study: List[Dict[str, float]],
    ) -> Dict[str, object]:
        """Aggregate statistics across studies."""
        # Calculate percentiles of study-level statistics
        percentiles = [5, 50, 95]
        summary: Dict[str, object] = {}
        for key in per_study[0].keys():
            values = [s[key] for s in per_study if not np.isnan(s[key])]
            if values:
                summary[f"{key}_percentiles"] = {
                    f"P{p}": float(np.percentile(values, p)) for p in percentiles
                }
            else:
                summary[f"{key}_percentiles"] = {f"P{p}": np.nan for p in percentiles}
        summary["Nstudy"] = len(per_study)

        return summary


class ListedObservationStats(BasicObservationStats):
    """Variant of BasicObservationStats that returns lists of study-level statistics instead of percentiles.
    This is useful for more detailed analyses or visualizations of the distribution of study-level statistics.
    """
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def aggregate(
        self,
        per_study: List[Dict[str, float]],
    ) -> Dict[str, object]:
        """Aggregate statistics across studies."""
        # Collect lists of study-level statistics
        summary: Dict[str, object] = {}
        for key in per_study[0].keys():
            values = [s[key] for s in per_study]
            summary[f"{key}_list"] = [float(v) for v in values]
        summary["Nstudy"] = len(per_study)

        return summary
