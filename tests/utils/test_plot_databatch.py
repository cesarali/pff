import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")
import matplotlib.pyplot as plt

from pff.data.datasets.aicme_batch import AICMECompartmentsDataBatch
from pff.data.datasets.aicme_datasets import _collate_aicme_batches
from pff.utils.plots.databatch_plot import (
    _build_synthetic_mmd_overlay_figure,
    plot_aicme_databatch,
    plot_list_aicme_databatch,
    plot_synthetic_mmd_overlay,
)


def _build_builder_style_databatch(
    *,
    study_name: str,
    substance_name: str,
    include_remainder: bool = True,
) -> AICMECompartmentsDataBatch:
    """Create one builder-style batch carrying an explicit leading ``B=1`` axis."""

    n_context = 1
    n_target = 2
    n_obs = 6
    n_rem = 3 if include_remainder else 0

    context_obs = torch.linspace(1.0, 2.0, n_context * n_obs, dtype=torch.float32).view(
        1, n_context, n_obs, 1
    )
    target_obs = torch.linspace(1.5, 3.0, n_target * n_obs, dtype=torch.float32).view(
        1, n_target, n_obs, 1
    )

    context_obs_time = torch.linspace(0.5, 6.0, n_obs, dtype=torch.float32).view(1, 1, n_obs, 1)
    target_obs_time = torch.linspace(0.5, 6.0, n_obs, dtype=torch.float32).view(1, 1, n_obs, 1)
    target_obs_time = target_obs_time.repeat(1, n_target, 1, 1)

    context_obs_mask = torch.ones(1, n_context, n_obs, dtype=torch.bool)
    target_obs_mask = torch.ones(1, n_target, n_obs, dtype=torch.bool)

    if include_remainder:
        context_rem = torch.linspace(0.8, 1.0, n_context * n_rem, dtype=torch.float32).view(
            1, n_context, n_rem, 1
        )
        target_rem = torch.linspace(0.9, 1.4, n_target * n_rem, dtype=torch.float32).view(
            1, n_target, n_rem, 1
        )
        context_rem_time = torch.linspace(6.5, 8.5, n_rem, dtype=torch.float32).view(
            1, 1, n_rem, 1
        )
        target_rem_time = torch.linspace(6.5, 8.5, n_rem, dtype=torch.float32).view(
            1, 1, n_rem, 1
        )
        target_rem_time = target_rem_time.repeat(1, n_target, 1, 1)
        context_rem_mask = torch.ones(1, n_context, n_rem, dtype=torch.bool)
        target_rem_mask = torch.ones(1, n_target, n_rem, dtype=torch.bool)
    else:
        context_rem = torch.zeros(1, n_context, 0, 1, dtype=torch.float32)
        target_rem = torch.zeros(1, n_target, 0, 1, dtype=torch.float32)
        context_rem_time = torch.zeros(1, n_context, 0, 1, dtype=torch.float32)
        target_rem_time = torch.zeros(1, n_target, 0, 1, dtype=torch.float32)
        context_rem_mask = torch.zeros(1, n_context, 0, dtype=torch.bool)
        target_rem_mask = torch.zeros(1, n_target, 0, dtype=torch.bool)

    return AICMECompartmentsDataBatch(
        target_obs=target_obs,
        target_obs_time=target_obs_time,
        target_obs_mask=target_obs_mask,
        target_rem_sim=target_rem,
        target_rem_sim_time=target_rem_time,
        target_rem_sim_mask=target_rem_mask,
        context_obs=context_obs,
        context_obs_time=context_obs_time,
        context_obs_mask=context_obs_mask,
        context_rem_sim=context_rem,
        context_rem_sim_time=context_rem_time,
        context_rem_sim_mask=context_rem_mask,
        target_dosing_amounts=torch.ones(1, n_target, dtype=torch.float32),
        target_dosing_route_types=torch.zeros(1, n_target, dtype=torch.long),
        context_dosing_amounts=torch.ones(1, n_context, dtype=torch.float32),
        context_dosing_route_types=torch.zeros(1, n_context, dtype=torch.long),
        mask_context_individuals=torch.ones(1, n_context, dtype=torch.bool),
        mask_target_individuals=torch.ones(1, n_target, dtype=torch.bool),
        study_name=[study_name],
        context_subject_name=[["ctx_0"]],
        target_subject_name=[["tgt_0", "tgt_1"]],
        substance_name=[substance_name],
        time_scales=torch.ones(1, 2, dtype=torch.float32),
        is_empirical=False,
    )


def test_collate_aicme_batches_concatenates_existing_batch_axis() -> None:
    """Builder-style ``B=1`` items must collate to ``[B, I, T, 1]`` tensors."""

    batch_a = _build_builder_style_databatch(study_name="study_a", substance_name="Drug_A")
    batch_b = _build_builder_style_databatch(study_name="study_b", substance_name="Drug_B")

    collated = _collate_aicme_batches([[batch_a], [batch_b]])[0]

    assert collated.context_obs.shape == (2, 1, 6, 1)
    assert collated.target_obs.shape == (2, 2, 6, 1)
    assert collated.context_obs_mask.shape == (2, 1, 6)
    assert collated.target_obs_mask.shape == (2, 2, 6)
    assert collated.study_name == ["study_a", "study_b"]
    assert collated.substance_name == ["Drug_A", "Drug_B"]


def test_plot_list_aicme_databatch_handles_collated_builder_batches(tmp_path) -> None:
    """Plotting should work after collating synthetic-study builder batches."""

    batch_a = _build_builder_style_databatch(study_name="study_a", substance_name="Drug_A")
    batch_b = _build_builder_style_databatch(study_name="study_b", substance_name="Drug_B")
    collated = _collate_aicme_batches([[batch_a], [batch_b]])[0]

    requested_out = tmp_path / "synthetic_loader_plot.png"
    saved_out = tmp_path / "synthetic_loader_plot.png"
    returned_path = plot_list_aicme_databatch(
        [collated],
        file_name=str(requested_out),
        number_of_rows=2,
        number_of_columns=1,
        log_scale=True,
    )

    assert returned_path == str(saved_out)
    assert saved_out.exists()


def test_plot_aicme_databatch_handles_empty_remainder_axis() -> None:
    """Empty remainder tensors should be ignored instead of raising reshape errors."""

    batch = _build_builder_style_databatch(
        study_name="study_empty_rem",
        substance_name="Drug_A",
        include_remainder=False,
    )
    ax = plot_aicme_databatch(batch, batch_index=0, log_scale=True)
    assert ax is not None


def test_plot_synthetic_mmd_overlay_handles_multi_study_multi_target_batch(tmp_path) -> None:
    """Synthetic MMD overlay plotting should handle multiple studies and targets."""

    batch_a = _build_builder_style_databatch(study_name="study_a", substance_name="Drug_A")
    batch_b = _build_builder_style_databatch(study_name="study_b", substance_name="Drug_B")
    collated = _collate_aicme_batches([batch_a, batch_b])

    observed_values = collated.target_obs.clone()
    generated_values = observed_values * 1.10
    aligned_times = collated.target_obs_time.clone()
    aligned_mask = collated.target_obs_mask.clone()

    fig, axes = _build_synthetic_mmd_overlay_figure(
        collated,
        observed_values=observed_values,
        generated_values=generated_values,
        times=aligned_times,
        mask=aligned_mask,
        num_studies=1,
    )
    assert len(axes) == 1
    assert len(fig.axes) == 3
    assert fig.axes[1].get_title() == "Dosing Route Counts"
    assert fig.axes[2].get_title() == "Dose Values by Route"
    plt.close(fig)

    output_path = tmp_path / "synthetic_mmd_overlay.png"
    returned_path = plot_synthetic_mmd_overlay(
        collated,
        observed_values=observed_values,
        generated_values=generated_values,
        times=aligned_times,
        mask=aligned_mask,
        num_studies=2,
        file_name=str(output_path),
    )

    assert returned_path == str(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
