"""StudyJSON plotting helpers."""

from __future__ import annotations

from typing import Tuple

from pff.data.data_empirical.json_schema import StudyJSON

try:  # pragma: no cover - plotting is optional in CI environments
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - fallback handled at runtime
    plt = None


def plot_studyjson_ensemble(
    study_list: list[StudyJSON],
    num_rows: int,
    num_cols: int,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """Visualise a list of studies as observation curves.

    Parameters
    ----------
    study_list:
        Collection of :class:`StudyJSON` records to render.
    num_rows, num_cols:
        Dimensions of the subplot grid. The product must be greater or equal to
        ``len(study_list)`` so that each study receives its own axis.
    figsize:
        Matplotlib ``figure`` size argument applied when creating the subplot
        grid.
    """

    if plt is None:  # pragma: no cover - only triggered when matplotlib missing
        raise ImportError(
            "matplotlib is required for plot_studyjson_ensemble but is not installed"
        )

    if not study_list:
        raise ValueError("study_list must contain at least one StudyJSON entry")

    total_plots = num_rows * num_cols
    if len(study_list) > total_plots:
        raise ValueError(
            "The subplot grid is too small for the provided study list: "
            f"{len(study_list)} studies for {total_plots} subplots."
        )

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)

    for idx, study in enumerate(study_list):
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
        ax.set_xlabel("Time")
        ax.set_ylabel("Observation")
        ax.grid(True, alpha=0.3)

    for remaining_idx in range(len(study_list), total_plots):
        row = remaining_idx // num_cols
        col = remaining_idx % num_cols
        axes[row][col].axis("off")

    fig.tight_layout()
    plt.show()

