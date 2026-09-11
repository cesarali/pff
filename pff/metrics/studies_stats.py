import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Callable, Optional, List
from pff.data.datasets.aicme_batch import AICMECompartmentsDataBatch

class StatisticsAggregator:
    """
    # ---------------------
    # Example Usage:
    # ---------------------
    # Assume that 'batch' is an object (e.g. a namedtuple) with attributes:
    #   context_obs, context_obs_time, context_obs_mask, target_obs, target_obs_time, target_obs_mask
    #
    # metrics = {
    #     'max_steepness': max_steepness_per_study,
    #     'individual_mean': individual_mean
    # }
    #
    # aggregator = StatisticsAggregator(metrics)
    # for batch in data_loader:
    #     aggregator.update(batch)
    #
    # summary = aggregator.compute()
    # print(summary)
    # aggregator.plot_histograms()

    """
    def __init__(self, metrics: Dict[str, Callable]):
        """
        metrics: a dictionary mapping metric names to functions that take a batch
                 and return a torch.Tensor.
        """
        self.metrics = metrics
        self.reset()

    def reset(self) -> None:
        self.storage = {name: [] for name in self.metrics}
        self.summary_df = pd.DataFrame()
        self.raw_df = pd.DataFrame()

    def histogram_arrays(self, metrics: Optional[List[str]] = None, bins: int = 20):
        """Return histogram counts and bin edges for each metric.

        Parameters
        ----------
        metrics: list of str, optional
            Metrics to include (defaults to all metrics).
        bins: int
            Number of histogram bins.
        """
        if self.raw_df.empty:
            raise ValueError("First run compute() to generate histograms")

        metrics = metrics or self.raw_df.columns.tolist()
        histograms = {}
        for metric in metrics:
            data = self.raw_df[metric].dropna()
            data = data[np.isfinite(data)]
            counts, edges = np.histogram(data, bins=bins)
            histograms[metric] = {
                "counts": counts.tolist(),
                "bin_edges": edges.tolist(),
                "summary": (
                    self.summary_df.loc[metric].to_dict()
                    if not self.summary_df.empty and metric in self.summary_df.index
                    else {}
                ),
            }
        return histograms

    def update(self, batch) -> None:
        """
        For each registered metric function, compute the metric on the given batch.
        If the returned tensor is 2D (e.g. individual-level metrics of shape [B, N]),
        it will be flattened to a 1D tensor.
        """
        for metric_name, metric_fn in self.metrics.items():
            try:
                values = metric_fn(batch)
                if not isinstance(values, torch.Tensor):
                    values = torch.tensor(values)
                # Flatten 2D tensors to 1D (assuming [B, N] => individual metrics)
                if values.ndim == 2:
                    values = values.reshape(-1)
                self.storage[metric_name].append(values.detach().cpu())
            except Exception as e:
                print(f"Error in {metric_name}: {str(e)}")
                self.storage[metric_name].append(torch.tensor([float('nan')]))

    def compute(self) -> pd.DataFrame:
        """
        Concatenate stored metrics, compute summary statistics for each metric,
        and return a Pandas DataFrame with the results.
        """
        stats_list = []
        raw_data = {}
        
        for name, batches in self.storage.items():
            if not batches:
                raw_data[name] = np.array([])
                continue
                
            all_values = torch.cat(batches).numpy()
            raw_data[name] = all_values
            valid_values = all_values[~np.isnan(all_values)]  # Excludes NaN
            finite_values = all_values[np.isfinite(all_values)]  # Excludes NaN, inf, -inf

            stats = {
                'metric': name,
                'mean': np.nanmean(all_values),  # Ignores NaN, includes inf
                'std': np.nanstd(all_values),   # Ignores NaN, includes inf
                'min': np.nanmin(valid_values) if valid_values.size > 0 else np.nan,
                'max': np.nanmax(valid_values) if valid_values.size > 0 else np.nan,
                'count': valid_values.size,      # Count of non-NaN values
                'num_nan': np.isnan(all_values).sum(),  # Count of NaN
                'nan_pct': np.isnan(all_values).mean() * 100,  # Percentage of NaN
                'num_finite': finite_values.size,  # Count of finite values (excludes NaN, inf, -inf)
                'finite_pct': (finite_values.size / all_values.size) * 100 if all_values.size > 0 else 0  # Percentage of finite values
            }
            stats_list.append(stats)

        self.summary_df = pd.DataFrame(stats_list).set_index('metric')
        
        # Pad arrays to the maximum length so we can create a DataFrame
        max_length = max((len(arr) for arr in raw_data.values()), default=0)
        padded_data = {}
        for name, arr in raw_data.items():
            if len(arr) < max_length:
                # Pad with NaN values
                pad_width = max_length - len(arr)
                arr = np.concatenate([arr, np.full(pad_width, np.nan)])
            padded_data[name] = arr
        
        self.raw_df = pd.DataFrame(padded_data)
        return self.summary_df

    def plot_histograms(self, metrics: Optional[List[str]] = None, 
                        figsize: tuple = (15, 10), bins: int = 20,
                        save_path: Optional[str] = None):
        """
        Plot histograms for selected metrics. By default, it plots all metrics present 
        in the raw dataframe.
        """
        if self.raw_df.empty:
            raise ValueError("First run compute() to generate histograms")
            
        metrics = metrics or self.raw_df.columns.tolist()
        n_metrics = len(metrics)
        cols = 2
        rows = (n_metrics + cols - 1) // cols
        
        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        axs = axs.ravel() if isinstance(axs, np.ndarray) else [axs]
        
        for i, metric in enumerate(metrics):
            print(metric)
            ax = axs[i]
            data = self.raw_df[metric].dropna()
            data = data[np.isfinite(data)]       # Remove inf and -inf values
            
            if not data.empty:
                ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)
                ax.set_title(f'{metric}\nDistribution', fontsize=12)
                ax.grid(True, alpha=0.2)
                
                stats_text = (
                    f"Mean: {self.summary_df.loc[metric, 'mean']:.2f}\n"
                    f"Std: {self.summary_df.loc[metric, 'std']:.2f}\n"
                    f"Min: {self.summary_df.loc[metric, 'min']:.2f}\n"
                    f"Max: {self.summary_df.loc[metric, 'max']:.2f}\n"
                    f"NaNs: {self.summary_df.loc[metric, 'num_nan']} "
                    f"({self.summary_df.loc[metric, 'nan_pct']:.1f}%)"
                )
                ax.text(0.95, 0.95, stats_text, 
                        transform=ax.transAxes,
                        verticalalignment='top',
                        horizontalalignment='right',
                        bbox=dict(facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No valid data', 
                        ha='center', va='center')
                
        for j in range(i+1, len(axs)):
            axs[j].axis('off')
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()


def max_steepness_per_study(batch) -> torch.Tensor:
    """
    Calculate the maximum absolute slope between consecutive observations per study.
    It combines both context and target observations.
    Returns a tensor of shape [B] where each entry is the maximum slope observed in the study.
    """
    # Combine context and target observations: shape [B, total_indiv, max_obs]
    all_obs = torch.cat([batch.context_obs, batch.target_obs], dim=1).squeeze(-1)
    all_times = torch.cat([batch.context_obs_time, batch.target_obs_time], dim=1).squeeze(-1)
    all_masks = torch.cat([batch.context_obs_mask, batch.target_obs_mask], dim=1).bool()

    B, num_individuals, num_obs = all_obs.shape
    max_slopes = torch.full((B,), float('nan'), device=all_obs.device)

    for b in range(B):
        study_max = -np.inf
        for i in range(num_individuals):
            valid = all_masks[b, i]
            times = all_times[b, i][valid]
            values = all_obs[b, i][valid]

            if times.numel() < 2:
                continue

            # Sort observations by time
            sorted_idx = torch.argsort(times)
            times = times[sorted_idx]
            values = values[sorted_idx]

            dt = times[1:] - times[:-1] + 1e-6  # prevent division by zero
            dv = values[1:] - values[:-1]
            slopes = dv / dt

            if slopes.numel() > 0:
                current_max = slopes.abs().max().item()
                study_max = max(study_max, current_max)
        if study_max != -np.inf:
            max_slopes[b] = study_max

    return max_slopes

def individual_mean(batch) -> torch.Tensor:
    """
    Compute the mean of observations per individual.
    Combines context and target data and returns a tensor of shape [B, total_individuals].
    """
    all_obs = torch.cat([
        batch.context_obs.squeeze(-1),
        batch.target_obs.squeeze(-1)
    ], dim=1)
    
    all_mask = torch.cat([
        batch.context_obs_mask,
        batch.target_obs_mask
    ], dim=1).bool()

    B, num_indiv, _ = all_obs.shape
    means = torch.full((B, num_indiv), float('nan'), device=all_obs.device)
    
    for b in range(B):
        for i in range(num_indiv):
            valid_obs = all_obs[b, i][all_mask[b, i]]
            if valid_obs.numel() > 0:
                means[b, i] = valid_obs.mean()
    return means

def max_per_study(batch) -> torch.Tensor:
    """
    For each study in the batch, compute the maximum observation value (across all individuals).
    Returns a tensor of shape [B].
    """
    # Concatenate context and target observations along the individual axis.
    # Shape: [B, total_individuals, max_obs]
    all_obs = torch.cat([batch.context_obs, batch.target_obs], dim=1).squeeze(-1)
    all_mask = torch.cat([batch.context_obs_mask, batch.target_obs_mask], dim=1).bool()
    
    B, num_individuals, max_obs = all_obs.shape
    max_study = torch.full((B,), float('nan'), device=all_obs.device)
    
    for b in range(B):
        # Select all valid observations across individuals for this study.
        valid_obs = all_obs[b][all_mask[b]]
        if valid_obs.numel() > 0:
            max_study[b] = valid_obs.max()
    
    return max_study

def min_per_study(batch) -> torch.Tensor:
    """
    For each study in the batch, compute the minimum observation value (across all individuals).
    Returns a tensor of shape [B].
    """
    all_obs = torch.cat([batch.context_obs, batch.target_obs], dim=1).squeeze(-1)
    all_mask = torch.cat([batch.context_obs_mask, batch.target_obs_mask], dim=1).bool()
    
    B, num_individuals, max_obs = all_obs.shape
    min_study = torch.full((B,), float('nan'), device=all_obs.device)
    
    for b in range(B):
        valid_obs = all_obs[b][all_mask[b]]
        if valid_obs.numel() > 0:
            min_study[b] = valid_obs.min()
    
    return min_study

def max_per_user(batch) -> torch.Tensor:
    """
    Compute the maximum observation per individual (user) across both context and target.
    Returns a tensor of shape [B, total_individuals].
    """
    all_obs = torch.cat([batch.context_obs, batch.target_obs], dim=1).squeeze(-1)
    all_mask = torch.cat([batch.context_obs_mask, batch.target_obs_mask], dim=1).bool()
    
    B, num_individuals, max_obs = all_obs.shape
    max_user = torch.full((B, num_individuals), float('nan'), device=all_obs.device)
    
    for b in range(B):
        for i in range(num_individuals):
            valid_obs = all_obs[b, i][all_mask[b, i]]
            if valid_obs.numel() > 0:
                max_user[b, i] = valid_obs.max()
    return max_user

def time_scale(batch:AICMECompartmentsDataBatch,time_index=0) -> torch.Tensor:
    """
    obtains the first time scale 
    """ 
    return batch.time_scales[:,time_index].clone()

def var_max_per_user_per_study(batch) -> torch.Tensor:
    """
    For each study, compute the variance of the maximum observation per user.
    Uses the max_per_user function to compute per-user maximum values,
    then computes the variance (unbiased estimator set to False) across individuals.
    Returns a tensor of shape [B].
    """
    # Get max per user first: shape [B, total_individuals]
    max_user = max_per_user(batch)
    B, num_individuals = max_user.shape
    var_study = torch.full((B,), float('nan'), device=max_user.device)
    
    for b in range(B):
        # Only consider individuals with valid (non-NaN) max values.
        valid = ~torch.isnan(max_user[b])
        if valid.sum() > 0:
            var_study[b] = torch.var(max_user[b][valid], unbiased=False)
    
    return var_study

