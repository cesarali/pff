# Quantile Coverage Metric and PK Sampling — Function Signatures Reference

This document summarizes all the key function signatures and their purposes across the two complementary modules:

* **Quantile Coverage Metrics:**  `pff/metrics/quantiles_coverage.py`
* **PK-like Synthetic Sampling and Visualization:**  `pff/utils/synthetic_samples/quantiles_coverage_samples.py`

These together form the complete testbed for evaluating uncertainty calibration in generative pharmacokinetic (PK) models.

---

## 🧩 Part I — Quantile Coverage Metrics

### `compute_percentile_coverage`

```python
def compute_percentile_coverage(
    pred_values,
    pred_times,
    pred_mask,
    real_values,
    real_times,
    real_mask,
    alpha: float = 0.05,
):
```

Compute the time-weighted coverage and interval score for irregular time series.

Returns a dictionary:

```python
{"coverage": TensorType["B"], "interval_score": TensorType["B"]}
```

---

### `compute_predictive_quantiles`

```python
def compute_predictive_quantiles(
    pred_values: TensorType["B", "S", "Tdistinct_max", 1],
    pred_mask: TensorType["B", "Tdistinct_max"],
    alpha: float,
) -> Tuple[
    TensorType["B", "Tdistinct_max", 1],
    TensorType["B", "Tdistinct_max", 1],
]:
```

Compute lower and upper predictive quantiles (α/2, 1−α/2) across stochastic samples.

---

### `interpolate_quantiles_to_obs_times`

```python
def interpolate_quantiles_to_obs_times(
    q_low: TensorType["B", "Tpred", 1],
    q_high: TensorType["B", "Tpred", 1],
    pred_times: TensorType["B", "Tpred", 1],
    pred_mask: TensorType["B", "Tpred"],
    real_times: TensorType["B", "I", "Treal", 1],
    real_mask: TensorType["B", "I", "Treal"],
) -> Tuple[
    TensorType["B", "I", "Treal", 1],
    TensorType["B", "I", "Treal", 1],
]:
```

Interpolate predictive quantile bands to match the irregular observation times.

---

### `compute_time_weighted_coverage`

```python
def compute_time_weighted_coverage(
    real_values: TensorType["B", "I", "Treal", 1],
    real_times: TensorType["B", "I", "Treal", 1],
    real_mask: TensorType["B", "I", "Treal"],
    q_low_interp: TensorType["B", "I", "Treal", 1],
    q_high_interp: TensorType["B", "I", "Treal", 1],
) -> TensorType["B"]:
```

Compute the Δt-weighted fraction of observed values inside the predictive quantile bands.

---

### `compute_interval_score`

```python
def compute_interval_score(
    real_values: TensorType["B", "I", "Treal", 1],
    real_times: TensorType["B", "I", "Treal", 1],
    real_mask: TensorType["B", "I", "Treal"],
    q_low_interp: TensorType["B", "I", "Treal", 1],
    q_high_interp: TensorType["B", "I", "Treal", 1],
    alpha: float,
) -> TensorType["B"]:
```

Compute the **Interval Score** (Gneiting & Raftery, 2007):
[IS_\alpha = (q_{high} - q_{low}) + \frac{2}{\alpha}[(q_{low} - y)*+ + (y - q*{high})_+]]
weighted by local Δt.

---

## 🧪 Part II — PK-like Synthetic Sampling and Visualization

### Imports

```python
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import torch
```

---

### `sample_dummy_band`

```python
def sample_dummy_band(
    n_samples: int = 100,
    n_points: int = 80,
    t_min: float = 0.0,
    t_max: float = 24.0,
    noise_scale: float = 0.05,
    band_scale: float = 0.2,
    baseline: float = 0.05,
    irregular: bool = False,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
```

Generate PK-like decaying trajectories with per-individual level variability and baseline offset.

Each trajectory:
[ y_i(t) = (1 + \delta_i) (f(t) + b) + \epsilon_i ]

Returns dictionary:

```python
{"t": [T, 1], "ref": [T, 1], "samples": [S, T, 1], "mask": [T]}
```

---

### `plot_dummy_band`

```python
def plot_dummy_band(
    data: Dict[str, torch.Tensor],
    alpha: float = 0.05,
    title: str = "Dummy Band",
    save_dir: Optional[Path] = None,
    show: bool = True,
) -> Path:
```

Plot a single set of trajectories with mean and percentile shading, optionally saving the result.

---

### `compare_bands_plot`

```python
def compare_bands_plot(
    data_left: Dict[str, torch.Tensor],
    data_right: Dict[str, torch.Tensor],
    alpha: float = 0.05,
    labels: Tuple[str, str] = ("Sample", "Reference"),
    save_dir: Optional[Path] = None,
    show: bool = True,
    log_y: bool = True,
) -> Path:
```

Visualize two synthetic PK bands side-by-side, comparing, for instance, low vs high variance cases. Supports log-scale y-axis for PK-style decays.

---

## 📜 Summary Table

| Category              | Function                             | Purpose                                               |
| --------------------- | ------------------------------------ | ----------------------------------------------------- |
| **Metrics**           | `compute_predictive_quantiles`       | Compute α/2 and 1−α/2 quantiles across samples.       |
|                       | `interpolate_quantiles_to_obs_times` | Interpolate quantiles to irregular observation times. |
|                       | `compute_time_weighted_coverage`     | Δt-weighted fraction of covered points.               |
|                       | `compute_interval_score`             | Proper interval score per batch.                      |
|                       | `compute_percentile_coverage`        | Full coverage and score pipeline.                     |
| **Synthetic Samples** | `sample_dummy_band`                  | Generate PK-like synthetic curves.                    |
|                       | `plot_dummy_band`                    | Plot one synthetic dataset.                           |
|                       | `compare_bands_plot`                 | Compare two datasets side-by-side.                    |

---

These functions are intended for testing and validating the **quantile coverage metric** pipeline. Together, they allow visual and quantitative evaluation of calibration behavior in generative PK models.
