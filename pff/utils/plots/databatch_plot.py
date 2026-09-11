"""Utilities for plotting :class:`AICMECompartmentsDataBatch` objects."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from torchtyping import TensorType

from pff.data.data_empirical.builder import EmpiricalBatchConfig, JSON2AICMEBuilder
from pff.data.data_empirical.json_schema import (
    StudyJSON,
    canonicalize_study,
    prediction_stats,
)
from pff.data.datasets.aicme_batch import AICMECompartmentsDataBatch
import matplotlib

matplotlib.use("Agg")  # Ensure plot rendering works in headless or VS Code RPC environments.

# An IndividualJSON is simply a mapping, documented in ``data_empirical/json_schema.py``.
IndividualJSON = Dict[str, object]
from pff.config_classes.data_config import MetaDosingConfig

# Colors used for different elements in the plot
CONTEXT_OBS_COLOR = "tab:green"
CONTEXT_REM_COLOR = "lightgreen"
TARGET_OBS_COLOR = "blue"
TARGET_REM_COLOR = "red"


def _resolve_plot_file_name(file_name: Optional[str]) -> Optional[str]:
    """Return the on-disk path used for saved plots.

    The plotting helpers default to PDF output only when the caller omits a
    suffix entirely. Explicit suffixes are preserved so scheduler tasks and
    other callers can request concrete image formats such as ``.png`` for
    Comet-compatible logging.
    """

    if file_name is None:
        return None

    file_path = Path(file_name)
    if file_path.suffix.lower() == "":
        file_path = file_path.with_suffix(".pdf")
    return str(file_path)


def _detach_to_cpu(batch: AICMECompartmentsDataBatch) -> AICMECompartmentsDataBatch:
    """Detach all tensors from computation graph and move to CPU."""
    return batch.detach_all().to_device(torch.device("cpu"))


def _squeeze_plot_payload(tensor: torch.Tensor) -> torch.Tensor:
    """Drop only trailing singleton payload axes used by scalar trajectories.

    Legacy ``AICMECompartmentsDataBatch`` tensors store scalar values as
    ``[..., T, 1]``. The synthetic experiment loader can additionally emit an
    extra channel axis and sometimes keeps times as ``[..., T, 1]`` while the
    corresponding values/masks are ``[..., T, C, 1]`` / ``[..., T, C]``.
    Removing only trailing singleton axes preserves the leading time axis while
    keeping non-trivial channel axes intact.
    """

    squeezed = tensor
    while squeezed.ndim > 0 and squeezed.shape[-1] == 1:
        squeezed = squeezed.squeeze(-1)
    return squeezed


def _extract_plot_series(
    values: torch.Tensor,
    times: torch.Tensor,
    mask: torch.Tensor,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Convert one databatch slice into one or more 1D plotting series.

    The helper supports both legacy scalar layouts
    ``values.shape == [T]`` / ``mask.shape == [T]`` and channel-augmented
    layouts such as ``values.shape == [T, C]`` with shared times
    ``times.shape == [T]`` or ``[T, 1]``. Extra non-time axes are flattened
    into independent plotted series after broadcasting ``times`` and ``mask``
    to the value tensor shape.
    """

    values_squeezed = _squeeze_plot_payload(values)
    times_squeezed = _squeeze_plot_payload(times)
    mask_squeezed = _squeeze_plot_payload(mask.bool())

    max_ndim = max(values_squeezed.ndim, times_squeezed.ndim, mask_squeezed.ndim)
    while values_squeezed.ndim < max_ndim:
        values_squeezed = values_squeezed.unsqueeze(-1)
    while times_squeezed.ndim < max_ndim:
        times_squeezed = times_squeezed.unsqueeze(-1)
    while mask_squeezed.ndim < max_ndim:
        mask_squeezed = mask_squeezed.unsqueeze(-1)

    try:
        values_b, times_b, mask_b = torch.broadcast_tensors(
            values_squeezed,
            times_squeezed,
            mask_squeezed,
        )
    except RuntimeError as exc:  # pragma: no cover - defensive error path
        raise ValueError(
            "Could not align value/time/mask tensors for plotting. "
            f"Got shapes values={tuple(values.shape)}, times={tuple(times.shape)}, "
            f"mask={tuple(mask.shape)}."
        ) from exc

    if values_b.ndim == 0:
        values_b = values_b.unsqueeze(0)
        times_b = times_b.unsqueeze(0)
        mask_b = mask_b.unsqueeze(0)

    time_steps = values_b.shape[0]
    if time_steps == 0 or values_b.numel() == 0 or times_b.numel() == 0 or mask_b.numel() == 0:
        return []

    values_2d = values_b.reshape(time_steps, -1)
    times_2d = times_b.reshape(time_steps, -1)
    mask_2d = mask_b.reshape(time_steps, -1).bool()

    series: List[Tuple[np.ndarray, np.ndarray]] = []
    for col in range(values_2d.shape[1]):
        valid = mask_2d[:, col]
        if not valid.any():
            continue
        series.append(
            (
                times_2d[:, col][valid].detach().cpu().numpy(),
                values_2d[:, col][valid].detach().cpu().numpy(),
            )
        )
    return series


def plot_aicme_databatch(
    databatch: AICMECompartmentsDataBatch,
    *,
    batch_index: int = 0,
    ax: Optional[plt.Axes] = None,
    log_scale: bool = True,
    file_name: Optional[str] = None,
    point_size: int = 5,
    line_width: float = 0.75,
    point_marker: str = "o",
    context_obs_color: str = CONTEXT_OBS_COLOR,
    context_rem_color: str = CONTEXT_REM_COLOR,
    target_obs_color: str = TARGET_OBS_COLOR,
    target_rem_color: str = TARGET_REM_COLOR,
    axis_label_font_size: Optional[float] = None,
    tick_label_font_size: Optional[float] = None,
) -> plt.Axes:
    """Plot one batch entry with configurable marker size/style and colors.

    The helper accepts both classic scalar PK tensors (for example
    ``[B, I, T, 1]``) and the synthetic-experiment layout where values and
    masks may include an extra channel axis (for example ``[B, I, T, C, 1]``
    and ``[B, I, T, C]``) while times remain shared across channels.
    """

    batch_cpu = _detach_to_cpu(databatch)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    context_alpha = 0.5  # transparent context
    line_alpha = 0.6  # slightly transparent context lines

    # Context observations
    for ind in range(batch_cpu.context_obs.shape[1]):
        context_obs_series = _extract_plot_series(
            batch_cpu.context_obs[batch_index, ind],
            batch_cpu.context_obs_time[batch_index, ind],
            batch_cpu.context_obs_mask[batch_index, ind],
        )
        for times, values in context_obs_series:
            ax.scatter(
                times,
                values,
                color=context_obs_color,
                s=point_size,
                alpha=context_alpha,
                marker=point_marker,
            )
            ax.plot(
                times,
                values,
                color=context_obs_color,
                linewidth=line_width,
                alpha=line_alpha,
            )

    # Context remainder
    for ind in range(batch_cpu.context_rem_sim.shape[1]):
        context_rem_series = _extract_plot_series(
            batch_cpu.context_rem_sim[batch_index, ind],
            batch_cpu.context_rem_sim_time[batch_index, ind],
            batch_cpu.context_rem_sim_mask[batch_index, ind],
        )
        for times, values in context_rem_series:
            ax.scatter(
                times,
                values,
                color=context_rem_color,
                s=point_size,
                alpha=context_alpha,
                marker=point_marker,
            )
            ax.plot(
                times,
                values,
                color=context_rem_color,
                linewidth=line_width,
                alpha=line_alpha,
            )

    # Target observations
    for ind in range(batch_cpu.target_obs.shape[1]):
        target_obs_series = _extract_plot_series(
            batch_cpu.target_obs[batch_index, ind],
            batch_cpu.target_obs_time[batch_index, ind],
            batch_cpu.target_obs_mask[batch_index, ind],
        )
        target_rem_series = _extract_plot_series(
            batch_cpu.target_rem_sim[batch_index, ind],
            batch_cpu.target_rem_sim_time[batch_index, ind],
            batch_cpu.target_rem_sim_mask[batch_index, ind],
        )

        for times, values in target_obs_series:
            ax.scatter(times, values, color=target_obs_color, s=point_size, marker=point_marker)
            ax.plot(times, values, color=target_obs_color, linewidth=line_width, alpha=0.8)

        # Connect the end of each observed series to the first remainder point
        # of the corresponding flattened channel whenever both exist.
        for (obs_times, obs_values), (rem_times, rem_values) in zip(
            target_obs_series,
            target_rem_series,
        ):
            ax.plot(
                [float(obs_times[-1]), float(rem_times[0])],
                [float(obs_values[-1]), float(rem_values[0])],
                color="gray",
                linestyle="--",
                linewidth=line_width,
                alpha=0.7,
            )

    # Target remainders
    for ind in range(batch_cpu.target_rem_sim.shape[1]):
        target_rem_series = _extract_plot_series(
            batch_cpu.target_rem_sim[batch_index, ind],
            batch_cpu.target_rem_sim_time[batch_index, ind],
            batch_cpu.target_rem_sim_mask[batch_index, ind],
        )
        for times, values in target_rem_series:
            ax.scatter(times, values, color=target_rem_color, s=point_size, marker=point_marker)
            ax.plot(times, values, color=target_rem_color, linewidth=1, alpha=0.8)

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("time")
    ax.set_ylabel("concentration")
    if axis_label_font_size is not None:
        ax.set_xlabel("time", fontsize=float(axis_label_font_size))
        ax.set_ylabel("concentration", fontsize=float(axis_label_font_size))
    if tick_label_font_size is not None:
        ax.tick_params(axis="both", labelsize=float(tick_label_font_size))

    resolved_file_name = _resolve_plot_file_name(file_name)
    if resolved_file_name is not None:
        Path(resolved_file_name).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(resolved_file_name, bbox_inches="tight")
        plt.close(fig)

    return ax


def plot_list_aicme_databatch(
    databatch_list: List[AICMECompartmentsDataBatch],
    file_name: Optional[str] = None,
    number_of_rows: int = 3,
    number_of_columns: int = 3,
    log_scale: bool = True,
) -> Optional[str]:
    """Plot a grid of :class:`AICMECompartmentsDataBatch` objects.

    Parameters
    ----------
    databatch_list:
        List of batches to plot.
    file_name:
        Path where the figure should be saved. If ``None`` the plot is not saved.
    number_of_rows:
        Maximum number of rows in the grid.
    number_of_columns:
        Maximum number of columns in the grid.
    log_scale:
        If ``True`` (default) the y-axis is set to logarithmic scale for all subplots.

    Returns
    -------
    str | None
        ``file_name`` if provided else ``None``.
    """
    if not databatch_list:
        return file_name

    batch_size = databatch_list[0].target_obs.shape[0]  # shape: [B, ...]
    n_rows = min(number_of_rows, batch_size)
    n_cols = min(number_of_columns, len(databatch_list))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.asarray(axes, dtype=object).reshape(n_rows, n_cols)

    for col in range(n_cols):
        for row in range(n_rows):
            ax = axes[row, col]
            plot_aicme_databatch(
                databatch_list[col],
                batch_index=row,
                ax=ax,
                log_scale=log_scale,
            )

    # Label rows with substance names from the first column's batch
    first_batch = databatch_list[0]
    for row in range(n_rows):
        if row < len(first_batch.substance_name):
            label = first_batch.substance_name[row]
            if isinstance(label, tuple):
                label = ""
            axes[row, 0].set_ylabel(f"concentration\n{label}")

    # Hide unused axes
    for col in range(n_cols, axes.shape[1]):
        for row in range(axes.shape[0]):
            axes[row, col].axis("off")
    for row in range(n_rows, axes.shape[0]):
        for col in range(axes.shape[1]):
            axes[row, col].axis("off")

    fig.tight_layout()

    resolved_file_name = _resolve_plot_file_name(file_name)
    if resolved_file_name is not None:
        Path(resolved_file_name).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(resolved_file_name, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return resolved_file_name


def plot_ind_json(
    individual: IndividualJSON,
    *,
    file_name: Optional[str] = None,
    log_scale: bool = True,
) -> Optional[str]:
    """Plot a single ``IndividualJSON`` record.

    Parameters
    ----------
    individual:
        Mapping describing one subject with keys such as ``"observations"`` and
        ``"observation_times"``.
    file_name:
        Optional path where the resulting figure should be saved. If ``None``
        the figure is not written to disk. Missing suffixes default to
        ``.pdf`` while explicit suffixes are preserved.
    log_scale:
        If ``True`` (default) the y-axis of the plot is drawn on a logarithmic scale.

    Returns
    -------
    str | None
        ``file_name`` if provided else ``None``.
    """

    max_obs = len(individual.get("observations", []))
    max_rem = len(individual.get("remaining", []))

    study = {"context": [], "target": [individual]}

    builder = JSON2AICMEBuilder(
        EmpiricalBatchConfig(max_observations=max_obs, max_remaining=max_rem, max_individuals=1)
    )
    batch = builder.build_one_aicmebatch([study], MetaDosingConfig())
    B = batch.target_obs.shape[0]  # [B, ...] -> [1, ...]

    return plot_list_aicme_databatch(
        [batch],
        file_name=file_name,
        number_of_rows=B,
        number_of_columns=1,
        log_scale=log_scale,
    )


def plot_study_json(
    study: StudyJSON,
    *,
    file_name: Optional[str] = None,
    log_scale: bool = True,
) -> Optional[str]:
    """Plot a full ``StudyJSON`` record after canonicalization.

    The study is canonicalized using :func:`canonicalize_study` to ensure
    consistent ordering and validation of individuals. The resulting study is
    converted to an :class:`AICMECompartmentsDataBatch` and displayed using
    :func:`plot_aicme_databatch`.

    Parameters
    ----------
    study:
        Mapping describing a study with ``"context"`` and ``"target"``
        individuals.
    file_name:
        Optional path where the resulting figure should be saved. If ``None``
        the figure is not written to disk. Missing suffixes default to
        ``.pdf`` while explicit suffixes are preserved.
    log_scale:
        If ``True`` (default) the y-axis of the plot is drawn on a logarithmic scale.

    Returns
    -------
    str | None
        ``file_name`` if provided else ``None``.
    """

    canon = canonicalize_study(study)
    all_inds = canon["context"] + canon["target"]

    max_obs = max((len(ind.get("observations", [])) for ind in all_inds), default=0)
    max_rem = max((len(ind.get("remaining", [])) for ind in all_inds), default=0)
    max_inds = max(len(canon["context"]), len(canon["target"]))

    builder = JSON2AICMEBuilder(
        EmpiricalBatchConfig(
            max_observations=max_obs,
            max_remaining=max_rem,
            max_individuals=max_inds,
        )
    )
    batch = builder.build_one_aicmebatch([canon], MetaDosingConfig())

    resolved_file_name = _resolve_plot_file_name(file_name)
    plot_aicme_databatch(
        batch, batch_index=0, log_scale=log_scale, file_name=resolved_file_name
    )
    return resolved_file_name


def plot_study_json_with_prediction(
    study: StudyJSON,
    *,
    ax: Optional[plt.Axes] = None,
    file_name: Optional[str] = None,
    log_scale: bool = True,
    point_size: int = 5,
    line_width: float = 0.75,
    point_marker: str = "o",
    context_obs_color: str = CONTEXT_OBS_COLOR,
    context_rem_color: str = CONTEXT_REM_COLOR,
    target_obs_color: str = TARGET_OBS_COLOR,
    target_rem_color: str = TARGET_REM_COLOR,
    prediction_marker: str = "o",
    prediction_marker_size: float = 4.0,
    prediction_color: str = "black",
    prediction_error_color: str = "gray",
    prediction_line_style: str = "-",
    figure_size: Optional[Tuple[float, float]] = None,
    show_legend: bool = False,
    legend_font_size: float = 10.0,
    legend_loc: str = "best",
    axis_label_font_size: Optional[float] = None,
    tick_label_font_size: Optional[float] = None,
    number_of_predictions_plot_per_drug: Optional[int] = None,
) -> Optional[str]:
    """Plot a ``StudyJSON`` record including prediction statistics.

    The function computes prediction means and standard deviations using
    :func:`prediction_stats` and overlays them on top of the canonicalized study
    plot. Prediction overlays respect the non‑contiguous
    ``target_rem_sim_mask`` from the constructed
    :class:`AICMECompartmentsDataBatch`, so padded/invalid future points are
    never shown.

    Parameters
    ----------
    study:
        Study description to plot.
    ax:
        Optional Matplotlib axis to draw on. If ``None`` a new figure and axis
        are created.
    file_name:
        If provided, the figure is stored at this path. Missing suffixes
        default to ``.pdf`` while explicit suffixes are preserved.
    log_scale:
        Whether to draw the y-axis on a logarithmic scale (``True`` by default).
    point_size:
        Marker area passed to the underlying observed/target scatter calls.
    line_width:
        Width of observed/target connecting lines.
    point_marker:
        Marker used for observed/target points (for example ``"o"`` for circles).
    context_obs_color:
        Color for context observation trajectories.
    context_rem_color:
        Color for context remainder trajectories.
    target_obs_color:
        Color for target observation trajectories.
    target_rem_color:
        Color for target remainder trajectories.
    prediction_marker:
        Marker used for predictive mean points.
    prediction_marker_size:
        Marker size used for predictive mean points.
    prediction_color:
        Color used for predictive mean markers and connecting line.
    prediction_error_color:
        Color used for predictive error bars.
    prediction_line_style:
        Linestyle used for predictive mean line.
    figure_size:
        Reserved for compatibility with ``plot_kwargs`` forwarding. Figure size
        is managed by :func:`plot_list_list_study_json`, so this argument is
        ignored here.
    show_legend:
        If ``True``, draws a legend for context/target/prediction elements.
    legend_font_size:
        Font size used for the legend.
    legend_loc:
        Matplotlib legend location string.
    axis_label_font_size:
        Font size for x/y axis labels.
    tick_label_font_size:
        Font size for x/y tick labels.
    number_of_predictions_plot_per_drug:
        Optional upper bound on how many target-individual prediction overlays
        are drawn for this drug plot. When ``None``, all available target
        predictions are shown.

    Returns
    -------
    str | None
        ``file_name`` if provided else ``None``.
    """

    _ = figure_size
    if (
        number_of_predictions_plot_per_drug is not None
        and int(number_of_predictions_plot_per_drug) <= 0
    ):
        raise ValueError(
            "'number_of_predictions_plot_per_drug' must be a positive integer or None."
        )
    study = prediction_stats(study)
    canon = canonicalize_study(study, drop_tgt_too_few=False)

    all_inds = canon["context"] + canon["target"]
    max_obs = max((len(ind.get("observations", [])) for ind in all_inds), default=0)
    max_rem = max((len(ind.get("remaining", [])) for ind in all_inds), default=0)
    max_inds = max(len(canon["context"]), len(canon["target"]))

    builder = JSON2AICMEBuilder(
        EmpiricalBatchConfig(
            max_observations=max_obs,
            max_remaining=max_rem,
            max_individuals=max_inds,
        )
    )
    batch = builder.build_one_aicmebatch([canon], MetaDosingConfig())
    ax = plot_aicme_databatch(
        batch,
        batch_index=0,
        ax=ax,
        log_scale=log_scale,
        file_name=None,
        point_size=point_size,
        line_width=line_width,
        point_marker=point_marker,
        context_obs_color=context_obs_color,
        context_rem_color=context_rem_color,
        target_obs_color=target_obs_color,
        target_rem_color=target_rem_color,
        axis_label_font_size=axis_label_font_size,
        tick_label_font_size=tick_label_font_size,
    )

    max_predictions = (
        len(canon["target"])
        if number_of_predictions_plot_per_drug is None
        else min(int(number_of_predictions_plot_per_drug), len(canon["target"]))
    )
    for it_idx, ind in enumerate(canon["target"][:max_predictions]):
        has_pred = (
            "prediction_times" in ind and "prediction_mean" in ind and "prediction_std" in ind
        )
        if not has_pred:
            continue

        # Use prediction data as-is
        times = torch.as_tensor(ind["prediction_times"], dtype=torch.float32).view(-1)
        mean = torch.as_tensor(ind["prediction_mean"], dtype=torch.float32).view(-1)
        std = torch.as_tensor(ind["prediction_std"], dtype=torch.float32).view(-1)

        if times.numel() == 0:
            continue

        # Drop padded entries: all the trailing zeros in times (and their mean/std)
        # If you want to be extra safe, use (times > 0) instead of (times != 0)
        keep_mask = times != 0
        # keep_mask = times > 0  # <- alternative

        if not keep_mask.any():
            continue

        times = times[keep_mask]
        mean = mean[keep_mask]
        std = std[keep_mask]

        ax.errorbar(
            times,
            mean,
            yerr=std,
            fmt=prediction_marker,
            linestyle=prediction_line_style,
            color=prediction_color,
            ecolor=prediction_error_color,
            elinewidth=1,
            capsize=3,
            markersize=prediction_marker_size,
        )

    if show_legend:
        marker_size = max(3.0, float(point_size) ** 0.5)
        handles = [
            Line2D(
                [0],
                [0],
                color=context_obs_color,
                marker=point_marker,
                linewidth=line_width,
                markersize=marker_size,
                label="Context Obs",
            ),
            Line2D(
                [0],
                [0],
                color=context_rem_color,
                marker=point_marker,
                linewidth=line_width,
                markersize=marker_size,
                label="Context Remainder",
            ),
            Line2D(
                [0],
                [0],
                color=target_obs_color,
                marker=point_marker,
                linewidth=line_width,
                markersize=marker_size,
                label="Target Obs",
            ),
            Line2D(
                [0],
                [0],
                color=target_rem_color,
                marker=point_marker,
                linewidth=line_width,
                markersize=marker_size,
                label="Target Remainder",
            ),
            Line2D(
                [0],
                [0],
                color=prediction_color,
                marker=prediction_marker,
                linewidth=1.0,
                markersize=prediction_marker_size,
                label="Prediction Mean",
            ),
        ]
        ax.legend(handles=handles, fontsize=legend_font_size, loc=legend_loc)
    resolved_file_name = _resolve_plot_file_name(file_name)
    if resolved_file_name is not None:
        Path(resolved_file_name).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(resolved_file_name, bbox_inches="tight")
    return resolved_file_name


def _extract_study_dosing_routes_and_values(
    study: StudyJSON,
) -> Tuple[List[str], Dict[str, List[float]]]:
    """Return per-event dosing routes and numeric dose values grouped by route.

    The helper walks through all context and target individuals after
    canonicalisation. Each dosing event contributes one route entry to the
    returned list, which is later used for route-count histograms. Numeric dose
    values are collected separately per route so the value histogram can be
    overlaid by route.
    """

    canon = canonicalize_study(study, drop_tgt_too_few=False)
    route_sequence: List[str] = []
    doses_by_route: Dict[str, List[float]] = {}

    for ind in canon["context"] + canon["target"]:
        raw_routes = ind.get("dosing_type", [])
        raw_doses = ind.get("dosing", [])

        routes = list(raw_routes) if isinstance(raw_routes, list) else [raw_routes]
        doses = list(raw_doses) if isinstance(raw_doses, list) else [raw_doses]
        n_events = max(len(routes), len(doses))

        for event_idx in range(n_events):
            route_name = "unknown"
            if event_idx < len(routes):
                candidate = str(routes[event_idx]).strip()
                if candidate:
                    route_name = candidate

            route_sequence.append(route_name)

            if event_idx >= len(doses):
                continue

            try:
                dose_value = float(doses[event_idx])
            except (TypeError, ValueError):
                continue

            if not np.isfinite(dose_value):
                continue

            if route_name not in doses_by_route:
                doses_by_route[route_name] = []
            doses_by_route[route_name].append(dose_value)

    return route_sequence, doses_by_route


def _decode_dosing_route_label(
    route_idx: int,
    route_options: Optional[Sequence[str]],
) -> str:
    """Return a readable dosing-route label for one encoded route index."""

    if route_options is not None and 0 <= route_idx < len(route_options):
        route_name = str(route_options[route_idx]).strip()
        if route_name:
            return route_name
    return f"route_{route_idx}"


def _extract_batch_dosing_routes_and_values(
    batch: AICMECompartmentsDataBatch,
    *,
    study_idx: int,
    route_options: Optional[Sequence[str]] = None,
) -> Tuple[List[str], Dict[str, List[float]]]:
    """Return per-individual dosing routes and amounts for one databatch study.

    The databatch carries one dosing amount and one encoded dosing route per
    individual. This helper mirrors
    :func:`_extract_study_dosing_routes_and_values` for synthetic MMD plots by
    aggregating valid context and target individuals of one batch-study row.
    """

    route_sequence: List[str] = []
    doses_by_route: Dict[str, List[float]] = {}

    dosing_sources = (
        (
            batch.context_dosing_amounts[study_idx],  # shape: [Ic]
            batch.context_dosing_route_types[study_idx],  # shape: [Ic]
            batch.mask_context_individuals[study_idx],  # shape: [Ic]
        ),
        (
            batch.target_dosing_amounts[study_idx],  # shape: [It]
            batch.target_dosing_route_types[study_idx],  # shape: [It]
            batch.mask_target_individuals[study_idx],  # shape: [It]
        ),
    )

    for amounts, routes, individual_mask in dosing_sources:
        for ind_idx in range(int(individual_mask.shape[0])):
            if not bool(individual_mask[ind_idx]):
                continue

            route_name = _decode_dosing_route_label(
                int(routes[ind_idx].item()),
                route_options=route_options,
            )
            route_sequence.append(route_name)

            dose_value = float(amounts[ind_idx].item())
            if not np.isfinite(dose_value):
                continue
            doses_by_route.setdefault(route_name, []).append(dose_value)

    return route_sequence, doses_by_route


def _plot_dosing_histograms(
    route_ax: plt.Axes,
    dose_ax: plt.Axes,
    *,
    route_sequence: Sequence[str],
    doses_by_route: Dict[str, List[float]],
    legend_font_size: float = 10.0,
    legend_loc: str = "best",
    route_hist_color: str = "tab:orange",
    route_hist_alpha: float = 0.8,
    dose_hist_bins: Union[int, str] = "auto",
    dose_hist_alpha: float = 0.6,
    axis_label_font_size: Optional[float] = None,
    tick_label_font_size: Optional[float] = None,
) -> None:
    """Render dosing-route and dose-value histograms on the provided axes."""

    route_counts: Dict[str, int] = {}
    for route_name in route_sequence:
        route_counts[route_name] = route_counts.get(route_name, 0) + 1

    route_ax.set_title("Dosing Route Counts")
    route_ax.set_xlabel("Route")
    route_ax.set_ylabel("Count")
    if axis_label_font_size is not None:
        route_ax.set_xlabel("Route", fontsize=float(axis_label_font_size))
        route_ax.set_ylabel("Count", fontsize=float(axis_label_font_size))
    if tick_label_font_size is not None:
        route_ax.tick_params(axis="both", labelsize=float(tick_label_font_size))
    if route_counts:
        route_names = list(route_counts.keys())
        route_values = list(route_counts.values())
        route_ax.bar(
            route_names,
            route_values,
            color=route_hist_color,
            alpha=route_hist_alpha,
            edgecolor="black",
        )
        for tick_label in route_ax.get_xticklabels():
            tick_label.set_rotation(45)
            tick_label.set_horizontalalignment("right")
    else:
        route_ax.text(
            0.5,
            0.5,
            "No dosing routes",
            ha="center",
            va="center",
            transform=route_ax.transAxes,
        )
        route_ax.set_xticks([])
        route_ax.set_yticks([])

    dose_ax.set_title("Dose Values by Route")
    dose_ax.set_xlabel("Dose")
    dose_ax.set_ylabel("Frequency")
    if axis_label_font_size is not None:
        dose_ax.set_xlabel("Dose", fontsize=float(axis_label_font_size))
        dose_ax.set_ylabel("Frequency", fontsize=float(axis_label_font_size))
    if tick_label_font_size is not None:
        dose_ax.tick_params(axis="both", labelsize=float(tick_label_font_size))
    non_empty_dose_routes = {
        route_name: route_values
        for route_name, route_values in doses_by_route.items()
        if len(route_values) > 0
    }
    if non_empty_dose_routes:
        for route_name, route_values in non_empty_dose_routes.items():
            dose_ax.hist(
                route_values,
                bins=dose_hist_bins,
                alpha=dose_hist_alpha,
                edgecolor="black",
                label=route_name,
            )
        if len(non_empty_dose_routes) > 1:
            dose_ax.legend(fontsize=legend_font_size, loc=legend_loc)
    else:
        dose_ax.text(
            0.5,
            0.5,
            "No dosing values",
            ha="center",
            va="center",
            transform=dose_ax.transAxes,
        )
        dose_ax.set_xticks([])
        dose_ax.set_yticks([])


def plot_study_json_with_dosing_histograms(
    study: StudyJSON,
    *,
    axes: Optional[Sequence[plt.Axes]] = None,
    file_name: Optional[str] = None,
    log_scale: bool = True,
    point_size: int = 5,
    line_width: float = 0.75,
    point_marker: str = "o",
    context_obs_color: str = CONTEXT_OBS_COLOR,
    context_rem_color: str = CONTEXT_REM_COLOR,
    target_obs_color: str = TARGET_OBS_COLOR,
    target_rem_color: str = TARGET_REM_COLOR,
    prediction_marker: str = "o",
    prediction_marker_size: float = 4.0,
    prediction_color: str = "black",
    prediction_error_color: str = "gray",
    prediction_line_style: str = "-",
    figure_size: Optional[Tuple[float, float]] = None,
    show_legend: bool = False,
    legend_font_size: float = 10.0,
    legend_loc: str = "best",
    axis_label_font_size: Optional[float] = None,
    tick_label_font_size: Optional[float] = None,
    route_hist_color: str = "tab:orange",
    route_hist_alpha: float = 0.8,
    dose_hist_bins: Union[int, str] = "auto",
    dose_hist_alpha: float = 0.6,
) -> Optional[str]:
    """Plot a study together with dosing-route and dose-value histograms.

    The layout contains three columns:
    1. The existing study/prediction trajectory plot from
       :func:`plot_study_json_with_prediction`.
    2. A categorical histogram of dosing routes used across all dosing events.
    3. Numeric dose histograms overlaid by dosing route.

    Parameters
    ----------
    study:
        Study description to plot.
    axes:
        Optional sequence of exactly three Matplotlib axes. When omitted, the
        function creates a new 1x3 figure.
    file_name:
        If provided, the figure is stored at this path. Missing suffixes
        default to ``.pdf`` while explicit suffixes are preserved.
    log_scale:
        Whether to draw the first-column PK y-axis on a logarithmic scale.
    point_size, line_width, point_marker, context_obs_color, context_rem_color,
    target_obs_color, target_rem_color, prediction_marker,
    prediction_marker_size, prediction_color, prediction_error_color,
    prediction_line_style, figure_size, show_legend, legend_font_size,
    legend_loc, axis_label_font_size, tick_label_font_size:
        Forwarded to :func:`plot_study_json_with_prediction` for the first
        column.
    route_hist_color:
        Fill color used for the dosing-route count bars.
    route_hist_alpha:
        Transparency applied to the dosing-route count bars.
    dose_hist_bins:
        Bin configuration forwarded to :meth:`matplotlib.axes.Axes.hist` for
        the dose-value histogram panel.
    dose_hist_alpha:
        Transparency applied to the per-route dose histograms.

    Returns
    -------
    str | None
        ``file_name`` if provided else ``None``.
    """

    created_figure = False
    if axes is None:
        local_figure_size = figure_size if figure_size is not None else (12.0, 3.5)
        fig, axes_array = plt.subplots(1, 3, figsize=local_figure_size)
        axes_flat = np.asarray(axes_array, dtype=object).reshape(-1)
        created_figure = True
    else:
        axes_flat = np.asarray(axes, dtype=object).reshape(-1)
        if axes_flat.size != 3:
            raise ValueError("'axes' must contain exactly three Matplotlib axes.")
        fig = axes_flat[0].figure
        if any(ax.figure is not fig for ax in axes_flat):
            raise ValueError("All provided axes must belong to the same Matplotlib figure.")

    pk_ax, route_ax, dose_ax = axes_flat.tolist()

    plot_study_json_with_prediction(
        study,
        ax=pk_ax,
        file_name=None,
        log_scale=log_scale,
        point_size=point_size,
        line_width=line_width,
        point_marker=point_marker,
        context_obs_color=context_obs_color,
        context_rem_color=context_rem_color,
        target_obs_color=target_obs_color,
        target_rem_color=target_rem_color,
        prediction_marker=prediction_marker,
        prediction_marker_size=prediction_marker_size,
        prediction_color=prediction_color,
        prediction_error_color=prediction_error_color,
        prediction_line_style=prediction_line_style,
        figure_size=figure_size,
        show_legend=show_legend,
        legend_font_size=legend_font_size,
        legend_loc=legend_loc,
        axis_label_font_size=axis_label_font_size,
        tick_label_font_size=tick_label_font_size,
    )

    route_sequence, doses_by_route = _extract_study_dosing_routes_and_values(study)
    _plot_dosing_histograms(
        route_ax,
        dose_ax,
        route_sequence=route_sequence,
        doses_by_route=doses_by_route,
        legend_font_size=legend_font_size,
        legend_loc=legend_loc,
        route_hist_color=route_hist_color,
        route_hist_alpha=route_hist_alpha,
        dose_hist_bins=dose_hist_bins,
        dose_hist_alpha=dose_hist_alpha,
        axis_label_font_size=axis_label_font_size,
        tick_label_font_size=tick_label_font_size,
    )

    if created_figure:
        fig.tight_layout()

    resolved_file_name = _resolve_plot_file_name(file_name)
    if resolved_file_name is not None:
        Path(resolved_file_name).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(resolved_file_name, bbox_inches="tight")
    return resolved_file_name


def _normalise_substance_name(raw_name: object, fallback: str) -> str:
    """Return a clean substance name extracted from StudyJSON metadata.

    The empirical StudyJSON metadata occasionally stores the ``"substance_name"``
    field as strings, tuples or other containers.  This helper converts the
    value into a readable string and falls back to ``fallback`` whenever the
    metadata entry is missing or empty.  The normalised name is used both for
    axis labelling and for generating deterministic file names when each study
    is plotted separately.
    """

    if raw_name is None:
        return fallback

    if isinstance(raw_name, str):
        candidate = raw_name.strip()
        return candidate or fallback

    if isinstance(raw_name, (list, tuple)):
        parts = [str(part).strip() for part in raw_name if str(part).strip()]
        if parts:
            return " ".join(parts)
        return fallback

    try:
        candidate = str(raw_name).strip()
    except Exception:  # pragma: no cover - extremely defensive
        return fallback

    return candidate or fallback


def _separate_plot_file_name(
    base_file_name: str,
    *,
    substance_name: str,
    permutation_index: int,
) -> str:
    """Return a filename for a single-study plot derived from ``base_file_name``.

    Parameters
    ----------
    base_file_name:
        Reference file name used when plotting multiple studies in a single figure.
    substance_name:
        Name of the simulated substance associated with the plot.  The value is
        sanitised so that it can safely be used inside the file name.
    permutation_index:
        Index of the permutation that produced the study.  Including the
        permutation makes every generated file name deterministic and unique.

    Returns
    -------
    str
        A new filename that appends ``substance_name`` and ``permutation``
        information to ``base_file_name`` while preserving the original suffix.
    """

    resolved_base_file_name = _resolve_plot_file_name(base_file_name)
    if resolved_base_file_name is None:  # pragma: no cover - defensive guard
        raise ValueError("'base_file_name' must not be None")

    base_path = Path(resolved_base_file_name)
    stem = base_path.stem
    suffix = base_path.suffix

    safe_substance = re.sub(r"[^0-9A-Za-z]+", "_", substance_name).strip("_")
    if not safe_substance:
        safe_substance = "substance"

    new_stem = f"{stem}_{safe_substance}_permutation_{permutation_index}"
    return str(base_path.with_name(f"{new_stem}{suffix}"))


def _plot_series_collection(
    ax: plt.Axes,
    *,
    series: List[Tuple[np.ndarray, np.ndarray]],
    color: str,
    point_size: float,
    line_width: float,
    alpha: float,
    marker: str,
) -> None:
    """Draw one flattened collection of time series on ``ax``."""

    for times, values in series:
        ax.scatter(
            times,
            values,
            color=color,
            s=point_size,
            alpha=alpha,
            marker=marker,
        )
        ax.plot(
            times,
            values,
            color=color,
            linewidth=line_width,
            alpha=alpha,
        )


def _build_synthetic_mmd_overlay_figure(
    batch: AICMECompartmentsDataBatch,
    *,
    observed_values: TensorType["B", "It", "Tobs", 1],
    generated_values: TensorType["B", "It", "Tobs", 1],
    times: TensorType["B", "It", "Tobs", 1],
    mask: TensorType["B", "It", "Tobs"],
    num_studies: int,
    route_options: Optional[Sequence[str]] = None,
    log_scale: bool = True,
    plot_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Return a figure overlaying observed and generated synthetic MMD targets.

    Parameters
    ----------
    batch:
        Original synthetic databatch carrying the shared context trajectories.
    observed_values:
        Real synthetic target values with shape ``[B, It, Tobs, 1]``.
    generated_values:
        Model-generated target values aligned to the same schedule as
        ``observed_values`` with shape ``[B, It, Tobs, 1]``.
    times:
        Shared target-observation times with shape ``[B, It, Tobs, 1]``.
    mask:
        Valid-time mask for ``times`` with shape ``[B, It, Tobs]``.
    num_studies:
        Maximum number of batch-study rows to render.
    route_options:
        Optional route labels decoding ``*_dosing_route_types``. When omitted,
        the default :class:`MetaDosingConfig` routes are used.
    log_scale:
        Whether to draw the y-axis on a logarithmic scale.
    plot_kwargs:
        Optional style overrides controlling colors, transparency and sizing.

    Returns
    -------
    tuple[matplotlib.figure.Figure, numpy.ndarray]
        Figure plus the stacked PK-overlay axes used for plotting.
    """

    if num_studies <= 0:
        raise ValueError("'num_studies' must be a positive integer.")

    batch_cpu = _detach_to_cpu(batch)
    observed_values_cpu = observed_values.detach().cpu()
    generated_values_cpu = generated_values.detach().cpu()
    times_cpu = times.detach().cpu()
    mask_cpu = mask.detach().cpu().bool()

    # observed_values_cpu: [B, It, Tobs, 1]
    # generated_values_cpu: [B, It, Tobs, 1]
    # times_cpu: [B, It, Tobs, 1]
    # mask_cpu: [B, It, Tobs]
    if observed_values_cpu.ndim != 4 or observed_values_cpu.shape[-1] != 1:
        raise ValueError(
            "Expected observed_values to have shape [B, It, Tobs, 1], got "
            f"{tuple(observed_values_cpu.shape)}."
        )
    if generated_values_cpu.shape != observed_values_cpu.shape:
        raise ValueError(
            "generated_values must match observed_values shape exactly, got "
            f"{tuple(generated_values_cpu.shape)} vs {tuple(observed_values_cpu.shape)}."
        )
    if times_cpu.shape != observed_values_cpu.shape:
        raise ValueError(
            "times must match observed_values shape exactly, got "
            f"{tuple(times_cpu.shape)} vs {tuple(observed_values_cpu.shape)}."
        )
    if mask_cpu.shape != observed_values_cpu.shape[:-1]:
        raise ValueError(
            "mask must have shape [B, It, Tobs], got "
            f"{tuple(mask_cpu.shape)} for observed_values shape {tuple(observed_values_cpu.shape)}."
        )

    max_studies = min(
        int(num_studies),
        int(batch_cpu.target_obs.shape[0]),
        int(observed_values_cpu.shape[0]),
    )
    if max_studies <= 0:
        raise ValueError("No valid studies are available for synthetic MMD plotting.")

    style_kwargs = dict(plot_kwargs) if plot_kwargs else {}
    figure_size = style_kwargs.pop("figure_size", (12.0, 3.25 * max_studies))
    if not isinstance(figure_size, (tuple, list)) or len(figure_size) != 2:
        raise ValueError("'plot_kwargs[\"figure_size\"]' must be a 2-item tuple/list.")
    figure_size = (float(figure_size[0]), float(figure_size[1]))
    route_options_resolved = route_options
    if route_options_resolved is None:
        route_options_resolved = list(MetaDosingConfig().route_options)

    context_obs_color = str(style_kwargs.pop("context_obs_color", CONTEXT_OBS_COLOR))
    context_rem_color = str(style_kwargs.pop("context_rem_color", CONTEXT_REM_COLOR))
    observed_target_color = str(style_kwargs.pop("observed_target_color", "tab:blue"))
    generated_target_color = str(style_kwargs.pop("generated_target_color", "tab:orange"))
    context_alpha = float(style_kwargs.pop("context_alpha", 0.45))
    observed_target_alpha = float(style_kwargs.pop("observed_target_alpha", 0.50))
    generated_target_alpha = float(style_kwargs.pop("generated_target_alpha", 0.50))
    point_size = float(style_kwargs.pop("point_size", 8.0))
    line_width = float(style_kwargs.pop("line_width", 0.90))
    point_marker = str(style_kwargs.pop("point_marker", "o"))
    title_override = style_kwargs.pop("title", None)
    title_font_size = style_kwargs.pop("title_font_size", None)
    axis_label_font_size = style_kwargs.pop("axis_label_font_size", None)
    tick_label_font_size = style_kwargs.pop("tick_label_font_size", None)
    show_legend = bool(style_kwargs.pop("show_legend", True))
    legend_font_size = float(style_kwargs.pop("legend_font_size", 9.0))
    legend_loc = str(style_kwargs.pop("legend_loc", "best"))
    route_hist_color = str(style_kwargs.pop("route_hist_color", "tab:orange"))
    route_hist_alpha = float(style_kwargs.pop("route_hist_alpha", 0.8))
    dose_hist_bins = style_kwargs.pop("dose_hist_bins", "auto")
    dose_hist_alpha = float(style_kwargs.pop("dose_hist_alpha", 0.6))

    fig, axes = plt.subplots(
        max_studies,
        3,
        figsize=figure_size,
        squeeze=False,
        gridspec_kw={"width_ratios": [2.6, 1.0, 1.2]},
    )
    pk_axes = np.asarray(axes[:, 0], dtype=object).reshape(max_studies)

    if title_override is not None:
        if title_font_size is not None:
            fig.suptitle(str(title_override), fontsize=float(title_font_size))
        else:
            fig.suptitle(str(title_override))

    for study_idx in range(max_studies):
        pk_ax = axes[study_idx, 0]
        route_ax = axes[study_idx, 1]
        dose_ax = axes[study_idx, 2]

        for ind_idx in range(batch_cpu.context_obs.shape[1]):
            if not bool(batch_cpu.mask_context_individuals[study_idx, ind_idx]):
                continue

            context_obs_series = _extract_plot_series(
                batch_cpu.context_obs[study_idx, ind_idx],
                batch_cpu.context_obs_time[study_idx, ind_idx],
                batch_cpu.context_obs_mask[study_idx, ind_idx],
            )
            _plot_series_collection(
                pk_ax,
                series=context_obs_series,
                color=context_obs_color,
                point_size=point_size,
                line_width=line_width,
                alpha=context_alpha,
                marker=point_marker,
            )

            context_rem_series = _extract_plot_series(
                batch_cpu.context_rem_sim[study_idx, ind_idx],
                batch_cpu.context_rem_sim_time[study_idx, ind_idx],
                batch_cpu.context_rem_sim_mask[study_idx, ind_idx],
            )
            _plot_series_collection(
                pk_ax,
                series=context_rem_series,
                color=context_rem_color,
                point_size=point_size,
                line_width=line_width,
                alpha=context_alpha,
                marker=point_marker,
            )

        for target_idx in range(batch_cpu.target_obs.shape[1]):
            if not bool(batch_cpu.mask_target_individuals[study_idx, target_idx]):
                continue

            observed_series = _extract_plot_series(
                observed_values_cpu[study_idx, target_idx],
                times_cpu[study_idx, target_idx],
                mask_cpu[study_idx, target_idx],
            )
            _plot_series_collection(
                pk_ax,
                series=observed_series,
                color=observed_target_color,
                point_size=point_size,
                line_width=line_width,
                alpha=observed_target_alpha,
                marker=point_marker,
            )

            generated_series = _extract_plot_series(
                generated_values_cpu[study_idx, target_idx],
                times_cpu[study_idx, target_idx],
                mask_cpu[study_idx, target_idx],
            )
            _plot_series_collection(
                pk_ax,
                series=generated_series,
                color=generated_target_color,
                point_size=point_size,
                line_width=line_width,
                alpha=generated_target_alpha,
                marker=point_marker,
            )

        if log_scale:
            pk_ax.set_yscale("log")

        study_name = (
            batch_cpu.study_name[study_idx]
            if study_idx < len(batch_cpu.study_name) and batch_cpu.study_name[study_idx]
            else f"study_{study_idx}"
        )
        substance_name = _normalise_substance_name(
            batch_cpu.substance_name[study_idx]
            if study_idx < len(batch_cpu.substance_name)
            else None,
            f"substance_{study_idx}",
        )
        row_title = f"{study_name} | {substance_name}"
        if title_font_size is not None:
            pk_ax.set_title(row_title, fontsize=float(title_font_size))
        else:
            pk_ax.set_title(row_title)

        pk_ax.set_xlabel("time")
        pk_ax.set_ylabel("concentration")
        if axis_label_font_size is not None:
            pk_ax.set_xlabel("time", fontsize=float(axis_label_font_size))
            pk_ax.set_ylabel("concentration", fontsize=float(axis_label_font_size))
        if tick_label_font_size is not None:
            pk_ax.tick_params(axis="both", labelsize=float(tick_label_font_size))

        route_sequence, doses_by_route = _extract_batch_dosing_routes_and_values(
            batch_cpu,
            study_idx=study_idx,
            route_options=route_options_resolved,
        )
        _plot_dosing_histograms(
            route_ax,
            dose_ax,
            route_sequence=route_sequence,
            doses_by_route=doses_by_route,
            legend_font_size=legend_font_size,
            legend_loc=legend_loc,
            route_hist_color=route_hist_color,
            route_hist_alpha=route_hist_alpha,
            dose_hist_bins=dose_hist_bins,
            dose_hist_alpha=dose_hist_alpha,
            axis_label_font_size=axis_label_font_size,
            tick_label_font_size=tick_label_font_size,
        )

        if show_legend and study_idx == 0:
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    color=context_obs_color,
                    marker=point_marker,
                    linewidth=line_width,
                    alpha=context_alpha,
                    label="Context observed",
                ),
                Line2D(
                    [0],
                    [0],
                    color=context_rem_color,
                    marker=point_marker,
                    linewidth=line_width,
                    alpha=context_alpha,
                    label="Context remainder",
                ),
                Line2D(
                    [0],
                    [0],
                    color=observed_target_color,
                    marker=point_marker,
                    linewidth=line_width,
                    alpha=observed_target_alpha,
                    label="Synthetic target",
                ),
                Line2D(
                    [0],
                    [0],
                    color=generated_target_color,
                    marker=point_marker,
                    linewidth=line_width,
                    alpha=generated_target_alpha,
                    label="Model target",
                ),
            ]
            pk_ax.legend(handles=legend_handles, fontsize=legend_font_size, loc=legend_loc)

    tight_layout_kwargs = {"rect": (0.0, 0.0, 1.0, 0.97)} if title_override is not None else {}
    fig.tight_layout(**tight_layout_kwargs)
    return fig, pk_axes


def plot_synthetic_mmd_overlay(
    batch: AICMECompartmentsDataBatch,
    *,
    observed_values: TensorType["B", "It", "Tobs", 1],
    generated_values: TensorType["B", "It", "Tobs", 1],
    times: TensorType["B", "It", "Tobs", 1],
    mask: TensorType["B", "It", "Tobs"],
    num_studies: int,
    route_options: Optional[Sequence[str]] = None,
    file_name: Optional[str] = None,
    log_scale: bool = True,
    plot_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Plot overlaid synthetic-MMD studies and optionally save the figure.

    Unlike the legacy StudyJSON plotting helpers, this function preserves an
    explicit ``.png`` output path when provided so scheduler tasks can return
    image artifacts directly.
    """

    fig, _ = _build_synthetic_mmd_overlay_figure(
        batch,
        observed_values=observed_values,
        generated_values=generated_values,
        times=times,
        mask=mask,
        num_studies=num_studies,
        route_options=route_options,
        log_scale=log_scale,
        plot_kwargs=plot_kwargs,
    )

    if file_name is not None:
        output_path = Path(file_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return str(output_path)

    plt.show()
    return None


def plot_list_list_study_json(
    studies: List[List[StudyJSON]],
    *,
    file_name: Optional[str] = None,
    number_of_rows: Optional[int] = 3,
    number_of_columns: Optional[int] = 3,
    log_scale: bool = True,
    plot_all_separately: bool = False,
    plot_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[Union[str, List[str]]]:
    """Plot ``StudyJSON`` records either as a grid or as individual figures.

    When ``plot_all_separately`` is ``False`` (the default) the function retains
    the historical behaviour and renders the studies on a single grid.  When the
    flag is ``True`` each study is saved to its own image whose name is derived
    from ``file_name`` via :func:`_separate_plot_file_name`, and the list of
    generated filenames is returned.

    Parameters
    ----------
    number_of_rows:
        Maximum number of different substances to render per permutation.  When
        ``None`` every available substance is shown.
    number_of_columns:
        Maximum number of permutations to render.  When ``None`` every
        permutation is shown.
    plot_kwargs:
        Optional keyword arguments forwarded to
        :func:`plot_study_json_with_prediction` to control visual styling
        (for example marker size, marker type, or colors). The special key
        ``"figure_size"`` is consumed by this function to control Matplotlib
        figure size:
        - ``plot_all_separately=True``: per-image figure size.
        - ``plot_all_separately=False``: full grid figure size.
        The optional key ``"title"`` overrides the plot title text.
    """

    # ``log_scale`` defaults to ``True`` to match the single-study plotting helpers
    # and ensure consistent logarithmic y-axes across all plotting utilities.

    resolved_file_name = _resolve_plot_file_name(file_name)

    if not studies or not studies[0]:
        return resolved_file_name

    study_plot_kwargs = dict(plot_kwargs) if plot_kwargs else {}
    figure_size = study_plot_kwargs.pop("figure_size", None)
    title_font_size = study_plot_kwargs.pop("title_font_size", None)
    title_override = study_plot_kwargs.pop("title", None)
    if title_override is not None and not isinstance(title_override, str):
        raise ValueError("'plot_kwargs[\"title\"]' must be a string when provided.")
    if figure_size is None:
        separate_figsize = (4, 3)
        grid_figsize = None
    else:
        if not isinstance(figure_size, (list, tuple)) or len(figure_size) != 2:
            raise ValueError("'plot_kwargs[\"figure_size\"]' must be a 2-item tuple/list.")
        width = float(figure_size[0])
        height = float(figure_size[1])
        if width <= 0 or height <= 0:
            raise ValueError("'plot_kwargs[\"figure_size\"]' values must be > 0.")
        separate_figsize = (width, height)
        grid_figsize = (width, height)

    if plot_all_separately:
        if resolved_file_name is None:
            raise ValueError("'file_name' must be provided when plotting separately")

        separate_files: List[str] = []
        total_permutations = len(studies)
        if number_of_columns is not None and number_of_columns <= 0:
            raise ValueError("'number_of_columns' must be a positive integer or None")
        if number_of_rows is not None and number_of_rows <= 0:
            raise ValueError("'number_of_rows' must be a positive integer or None")

        max_permutations = (
            total_permutations
            if number_of_columns is None
            else min(number_of_columns, total_permutations)
        )

        for permutation_index in range(max_permutations):
            permutation_studies = studies[permutation_index]
            max_rows = (
                len(permutation_studies)
                if number_of_rows is None
                else min(number_of_rows, len(permutation_studies))
            )

            for row, study in enumerate(permutation_studies[:max_rows]):
                fig, ax = plt.subplots(figsize=separate_figsize)
                plot_study_json_with_prediction(
                    study,
                    ax=ax,
                    log_scale=log_scale,
                    **study_plot_kwargs,
                )

                raw_substance_name = study["meta_data"].get("substance_name")
                substance_name = _normalise_substance_name(raw_substance_name, f"substance_{row}")
                display_title = title_override if title_override else substance_name
                if title_font_size is not None:
                    ax.set_title(display_title, fontsize=float(title_font_size))
                else:
                    ax.set_title(display_title)
                study_file_name = _separate_plot_file_name(
                    resolved_file_name,
                    substance_name=substance_name,
                    permutation_index=permutation_index,
                )

                Path(study_file_name).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(study_file_name, bbox_inches="tight")
                plt.close(fig)
                separate_files.append(study_file_name)

        return separate_files

    batch_size = len(studies[0])  # shape: [B]

    if number_of_rows is not None and number_of_rows <= 0:
        raise ValueError("'number_of_rows' must be a positive integer or None")
    if number_of_columns is not None and number_of_columns <= 0:
        raise ValueError("'number_of_columns' must be a positive integer or None")

    n_rows = batch_size if number_of_rows is None else min(number_of_rows, batch_size)
    total_permutations = len(studies)
    n_cols = (
        total_permutations
        if number_of_columns is None
        else min(number_of_columns, total_permutations)
    )

    if grid_figsize is None:
        grid_figsize = (4 * n_cols, 3 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=grid_figsize)
    axes = np.atleast_2d(axes).reshape(n_rows, n_cols)

    for col in range(n_cols):
        permutation_studies = studies[col]
        for row in range(n_rows):
            ax = axes[row, col]
            if row >= len(permutation_studies):
                ax.axis("off")
                continue
            study = permutation_studies[row]
            plot_study_json_with_prediction(
                study,
                ax=ax,
                log_scale=log_scale,
                **study_plot_kwargs,
            )

            # Label left-most column with the substance name
            if col == 0:
                raw_substance_name = study["meta_data"].get("substance_name")
                substance_name = _normalise_substance_name(raw_substance_name, f"substance_{row}")
                ax.set_ylabel(substance_name, fontsize=10, rotation=90, labelpad=10)

    # Hide unused axes
    for col in range(n_cols, axes.shape[1]):
        for row in range(axes.shape[0]):
            axes[row, col].axis("off")
    for row in range(n_rows, axes.shape[0]):
        for col in range(axes.shape[1]):
            axes[row, col].axis("off")

    fig.tight_layout()

    if file_name is not None:
        if resolved_file_name is None:  # pragma: no cover - explicit for type narrowing
            raise ValueError("'file_name' must not be None when saving a grid plot")
        Path(resolved_file_name).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(resolved_file_name, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return resolved_file_name
