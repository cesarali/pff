"""Forward pass smoke test for :class:`ContextVAEPK`."""

import pytest
import torch
import numpy as np
from dataclasses import replace
from pathlib import Path

from pff import config_dir
from pff.config_classes.flow_pk_config import FlowPKExperimentConfig
from pff.data.datasets.aicme_batch import AICMECompartmentsDataBatch
from pff.data.datasets.aicme_datasets import AICMECompartmentsDataModule
from pff.models.amortized_inference.flows_pk import (
    FlowForwardOutputs,
    FlowPK,
    OTSampler,
)
from pff.metrics.quantiles_coverage import compute_percentile_coverage
from pff.utils.tensors_operations import gather_distinct_times_per_substance
from pff.utils.plots.databatch_plot import plot_list_list_study_json

from tests.models.test_aicme_pk import _first_batch_list


def _flow_pk_config_from_file(inference_type) -> FlowPKExperimentConfig:
    if inference_type == "flow-pk-generative":
        default_yaml = (
            Path(config_dir)
            / "experiment_configs"
            / "UAI"
            / "flow-pk-generate-test"
            / "flowPK.yaml"
        )
    elif inference_type == "flow-pk-predictive":
        default_yaml = (
            Path(config_dir)
            / "experiment_configs"
            / "UAI"
            / "flow-pk-predict-n-generate-test"
            / "flowPK.yaml"
        )
    cfg: FlowPKExperimentConfig = FlowPKExperimentConfig.from_yaml(str(default_yaml))
    cfg.train = replace(cfg.train, batch_size=3)
    cfg.mix_data = replace(cfg.mix_data, train_size=4)
    return cfg


def _random_batch(
    cfg: FlowPKExperimentConfig,
    batch_size: int = 2,
    n_context_individuals: int = 3,
    n_target_individuals: int = 1,
    n_context_obs: int | None = None,
    n_target_obs: int | None = None,
    n_target_future: int | None = None,
) -> AICMECompartmentsDataBatch:
    """Build a small random databatch for FlowPK tests.

    By default, tensor lengths follow the observation configuration so random
    smoke tests mirror the real dataset behaviour (e.g. no past target points
    when ``target_observations.max_past == 0``).
    """
    context_cfg = cfg.context_observations
    target_cfg = cfg.target_observations

    if n_context_obs is None:
        n_context_obs = int(context_cfg.max_num_obs)

    if n_target_obs is None:
        if target_cfg.split_past_future:
            if target_cfg.max_past is None:
                raise ValueError(
                    "target_observations.max_past must be set when split_past_future=True."
                )
            n_target_obs = int(target_cfg.max_past)
        else:
            n_target_obs = int(target_cfg.max_num_obs)

    if n_target_future is None:
        if target_cfg.split_past_future and target_cfg.add_rem:
            n_target_future = max(0, int(target_cfg.max_num_obs) - int(n_target_obs))
        else:
            n_target_future = 0

    n_context_rem = 1 if context_cfg.add_rem else 0

    B = batch_size
    Ic = n_context_individuals
    It = n_target_individuals
    C = n_context_obs
    T = n_target_obs
    Tr = n_target_future
    Cr = n_context_rem

    target_obs = torch.rand(B, It, T, 1)
    target_obs_time = torch.sort(torch.rand(B, It, T, 1), dim=2).values
    target_obs_mask = torch.ones(B, It, T, dtype=torch.bool)

    target_rem_sim = torch.rand(B, It, Tr, 1)
    target_rem_sim_time = 1.0 + torch.sort(torch.rand(B, It, Tr, 1), dim=2).values
    target_rem_sim_mask = torch.ones(B, It, Tr, dtype=torch.bool)

    context_obs = torch.rand(B, Ic, C, 1)
    context_obs_time = torch.sort(torch.rand(B, Ic, C, 1), dim=2).values
    context_obs_mask = torch.ones(B, Ic, C, dtype=torch.bool)

    context_rem_sim = torch.rand(B, Ic, Cr, 1)
    context_rem_sim_time = 1.0 + torch.sort(torch.rand(B, Ic, Cr, 1), dim=2).values
    context_rem_sim_mask = torch.ones(B, Ic, Cr, dtype=torch.bool)

    target_dosing_amounts = torch.rand(B, It) * 10.0 + 1.0
    target_dosing_route_types = torch.randint(0, 2, (B, It))
    context_dosing_amounts = torch.rand(B, Ic) * 10.0 + 1.0
    context_dosing_route_types = torch.randint(0, 2, (B, Ic))

    mask_context_individuals = torch.ones(B, Ic, dtype=torch.bool)
    mask_target_individuals = torch.ones(B, It, dtype=torch.bool)

    study_name = [f"study_{i}" for i in range(B)]
    context_subject_name = [[f"c_{j}" for j in range(Ic)] for _ in range(B)]
    target_subject_name = [[f"t_{j}" for j in range(It)] for _ in range(B)]
    substance_name = ["drug_x"] * B
    time_scales = torch.ones(B, 2)

    return AICMECompartmentsDataBatch(
        target_obs=target_obs,
        target_obs_time=target_obs_time,
        target_obs_mask=target_obs_mask,
        target_rem_sim=target_rem_sim,
        target_rem_sim_time=target_rem_sim_time,
        target_rem_sim_mask=target_rem_sim_mask,
        context_obs=context_obs,
        context_obs_time=context_obs_time,
        context_obs_mask=context_obs_mask,
        context_rem_sim=context_rem_sim,
        context_rem_sim_time=context_rem_sim_time,
        context_rem_sim_mask=context_rem_sim_mask,
        target_dosing_amounts=target_dosing_amounts,
        target_dosing_route_types=target_dosing_route_types,
        context_dosing_amounts=context_dosing_amounts,
        context_dosing_route_types=context_dosing_route_types,
        mask_context_individuals=mask_context_individuals,
        mask_target_individuals=mask_target_individuals,
        study_name=study_name,
        context_subject_name=context_subject_name,
        target_subject_name=target_subject_name,
        substance_name=substance_name,
        time_scales=time_scales,
        is_empirical=False,
    )


def _target_resolved_batch(cfg: FlowPKExperimentConfig) -> AICMECompartmentsDataBatch:
    """Return a compact multi-target batch with distinct target schedules."""

    batch = _random_batch(
        cfg,
        batch_size=2,
        n_target_individuals=2,
        n_target_obs=3,
        n_target_future=2,
    )
    target_obs_time = torch.tensor(
        [
            [
                [[1.0], [2.0], [0.0]],
                [[2.0], [4.0], [0.0]],
            ],
            [
                [[0.5], [1.5], [0.0]],
                [[1.0], [3.0], [0.0]],
            ],
        ],
        dtype=batch.target_obs_time.dtype,
    )
    target_obs_mask = target_obs_time.squeeze(-1) > 0
    target_dosing_amounts = torch.tensor(
        [[11.0, 22.0], [33.0, 44.0]], dtype=batch.target_dosing_amounts.dtype
    )
    target_dosing_route_types = torch.tensor(
        [[0, 1], [1, 0]], dtype=batch.target_dosing_route_types.dtype
    )
    return batch._replace(
        target_obs_time=target_obs_time,
        target_obs_mask=target_obs_mask,
        target_dosing_amounts=target_dosing_amounts,
        target_dosing_route_types=target_dosing_route_types,
        target_subject_name=[["tgt_a0", "tgt_a1"], ["tgt_b0", "tgt_b1"]],
        mask_target_individuals=torch.ones_like(batch.mask_target_individuals),
    )


@pytest.mark.skip("Long loading")
def test_flow_pk_from_file(inference_type):
    cfg = _flow_pk_config_from_file(inference_type)
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)
    model = FlowPK(cfg)
    outputs = model(batch_list)
    # Structural checks
    assert isinstance(outputs, FlowForwardOutputs)


def test_flow_pk_random_tensors(inference_type):
    """Forward smoke test without dataloader using random tensors."""
    cfg = _flow_pk_config_from_file(inference_type)
    model = FlowPK(cfg)
    batch = _random_batch(cfg, batch_size=10)
    outputs = model([batch])
    assert isinstance(outputs, FlowForwardOutputs)
    assert outputs.total_loss is not None


def test_flow_pk_random_tensors_multi_target_forward():
    """Forward smoke test for multi-target settings (`It > 1`)."""
    cfg = _flow_pk_config_from_file("flow-pk-predictive")
    model = FlowPK(cfg)
    batch = _random_batch(cfg, batch_size=4, n_target_individuals=2, n_target_future=3)
    outputs = model([batch])
    assert isinstance(outputs, FlowForwardOutputs)
    assert outputs.total_loss is not None
    assert torch.isfinite(outputs.total_loss)


def test_flow_pk_prediction():
    """Predictive sampling should ignore extra targets and return one target path."""
    cfg = _flow_pk_config_from_file("flow-pk-predictive")
    model = FlowPK(cfg)
    batch = _random_batch(cfg, batch_size=2, n_target_individuals=2, n_target_future=3)
    S = 2
    sampled, tgrid, real, mask = model.sample_individual_prediction(batch, sample_size=S)
    Tr = batch.target_rem_sim.shape[2]

    assert sampled.dim() == 5 and tgrid.dim() == 5
    assert sampled.size(0) == S and tgrid.size(0) == S
    assert sampled.shape == (S, batch.target_obs.size(0), 1, Tr, 1)
    assert tgrid.shape == (S, batch.target_obs.size(0), 1, Tr, 1)
    assert sampled.size(-1) == 1 and tgrid.size(-1) == 1
    assert real.shape == (batch.target_obs.size(0), 1, Tr, 1)
    assert mask.shape == (batch.target_obs.size(0), 1, Tr)


def test_flow_pk_prediction_single_target_regression():
    """Regression check that single-target predictive sampling still works."""
    cfg = _flow_pk_config_from_file("flow-pk-predictive")
    model = FlowPK(cfg)
    batch = _random_batch(cfg, batch_size=2, n_target_individuals=1, n_target_future=3)
    S = 2
    sampled, tgrid, real, mask = model.sample_individual_prediction(batch, sample_size=S)
    Tr = batch.target_rem_sim.shape[2]

    assert sampled.shape == (S, batch.target_obs.size(0), 1, Tr, 1)
    assert tgrid.shape == (S, batch.target_obs.size(0), 1, Tr, 1)
    assert real.shape == batch.target_rem_sim.shape
    assert mask.shape == batch.target_rem_sim_mask.shape


def test_flow_pk_resolve_target_dosing_returns_target_axis():
    """Target-resolved dosing helper should preserve the full target axis."""

    cfg = _flow_pk_config_from_file("flow-pk-predictive")
    model = FlowPK(cfg)
    batch = _target_resolved_batch(cfg)

    dose, route = model.resolve_target_dosing_from_databatch(batch)

    assert torch.equal(dose, batch.target_dosing_amounts)
    assert torch.equal(route, batch.target_dosing_route_types)


def test_flow_pk_sample_new_individual_resolves_from_target():
    """Target-resolved generative sampling should use the target count and grid."""

    cfg = _flow_pk_config_from_file("flow-pk-predictive")
    model = FlowPK(cfg)
    batch = _target_resolved_batch(cfg)
    samples, times, mask = model.sample_new_individual(
        batch,
        sample_size=99,
        num_steps=2,
        resolve_sampling_from_target=True,
    )

    assert samples.dim() == 4 and times.dim() == 3 and mask.dim() == 2
    assert samples.shape[0] == batch.target_obs.shape[1]
    assert samples.shape[1] == batch.target_obs.shape[0]
    assert samples.shape[-1] == 1
    assert times.shape == (batch.target_obs.shape[0], times.shape[1], 1)
    assert mask.shape == (batch.target_obs.shape[0], times.shape[1])
    assert torch.equal(times[0, mask[0], 0], torch.tensor([1.0, 2.0, 4.0], dtype=times.dtype))
    assert torch.equal(
        times[1, mask[1], 0],
        torch.tensor([0.5, 1.0, 1.5, 3.0], dtype=times.dtype),
    )


def test_flow_pk_sample_new_individual_can_include_target_remainder_times():
    """Target-resolved sampling can extend the grid with target remainder times."""

    cfg = _flow_pk_config_from_file("flow-pk-predictive")
    model = FlowPK(cfg)
    batch = _target_resolved_batch(cfg)._replace(
        target_rem_sim_time=torch.tensor(
            [
                [
                    [[5.0], [0.0]],
                    [[4.0], [0.0]],
                ],
                [
                    [[7.0], [0.0]],
                    [[3.0], [0.0]],
                ],
            ],
            dtype=torch.float32,
        ),
        target_rem_sim_mask=torch.tensor(
            [
                [[True, False], [True, False]],
                [[True, False], [True, False]],
            ],
            dtype=torch.bool,
        ),
    )

    _, times, mask = model.sample_new_individual(
        batch,
        sample_size=1,
        num_steps=2,
        resolve_sampling_from_target=True,
        include_rem=True,
    )

    assert torch.equal(times[0, mask[0], 0], torch.tensor([1.0, 2.0, 4.0, 5.0], dtype=times.dtype))
    assert torch.equal(
        times[1, mask[1], 0],
        torch.tensor([0.5, 1.0, 1.5, 3.0, 7.0], dtype=times.dtype),
    )


def test_flow_pk_sample_new_individual_resolve_target_requires_target_observations():
    """Target-resolved sampling should fail when no target observation grid exists."""

    cfg = _flow_pk_config_from_file("flow-pk-predictive")
    model = FlowPK(cfg)
    batch = _target_resolved_batch(cfg)._replace(
        target_obs_time=torch.zeros(2, 2, 3, 1),
        target_obs_mask=torch.zeros(2, 2, 3, dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="requires at least one valid target observation"):
        model.sample_new_individual(
            batch,
            sample_size=5,
            num_steps=2,
            resolve_sampling_from_target=True,
        )


def test_flow_pk_source_process_is_restored_for_legacy_style_loading():
    """FlowPK should rebuild source-process modules if a loaded checkpoint misses them."""
    cfg = _flow_pk_config_from_file("flow-pk-predictive")
    cfg.source_process = replace(cfg.source_process, source_type="gaussian_process")
    model = FlowPK(cfg)

    assert model.source_process.__class__.__name__ == "GaussianProcessRegression"
    assert hasattr(model, "source_is_time_series")

    # Simulate legacy checkpoints missing runtime-only attributes.
    delattr(model, "source_process")
    delattr(model, "source_is_time_series")

    batch = _random_batch(cfg, batch_size=2, n_target_individuals=1, n_target_future=2)
    samples, _, _ = model.sample_new_individual(batch, sample_size=1, num_steps=1)

    assert samples.shape[0] == 1
    assert model.source_process.__class__.__name__ == "GaussianProcessRegression"


def test_ot_sample_plan_with_conditioning_applies_same_batch_map():
    """OT conditional plan should reindex all conditioning tensors with the sampled target map."""
    B = 3
    T = 4
    N = 6
    Ic = 2

    x0 = torch.arange(B, dtype=torch.float32).view(B, 1, 1, 1).expand(B, 1, T, 1)
    x1 = (100.0 + torch.arange(B, dtype=torch.float32)).view(B, 1, 1, 1).expand(B, 1, T, 1)
    obs_times = (10.0 + torch.arange(B, dtype=torch.float32)).view(B, 1, 1, 1).expand(B, 1, T, 1)
    mask_obs = torch.tensor(
        [
            [[True, True, False, False]],
            [[True, False, True, False]],
            [[False, True, True, True]],
        ]
    )
    dose = (20.0 + torch.arange(B, dtype=torch.float32)).view(B, 1, 1).expand(B, T, 2)

    x_ctx = (30.0 + torch.arange(B, dtype=torch.float32)).view(B, 1, 1, 1).expand(B, Ic, 3, 1)
    obs_times_ctx = (
        (40.0 + torch.arange(B, dtype=torch.float32)).view(B, 1, 1, 1).expand(B, Ic, 3, 1)
    )
    context_obs_mask = torch.ones(B, Ic, 3, dtype=torch.bool)
    mask_context_individuals = torch.ones(B, Ic, dtype=torch.bool)
    dose_ctx = (50.0 + torch.arange(B, dtype=torch.float32)).view(B, 1, 1).expand(B, N, 2)
    study_ctx = (x_ctx, obs_times_ctx, context_obs_mask, mask_context_individuals, dose_ctx)

    ot = OTSampler(batch_size=B, replace=False)
    i = np.array([2, 0, 1])
    j = np.array([1, 2, 0])
    ot.get_map = lambda _x0, _x1: np.ones((B, B), dtype=np.float64)  # type: ignore[method-assign]
    ot.sample_map = lambda _pi: (i, j)  # type: ignore[method-assign]

    (
        x0_mapped,
        x1_mapped,
        obs_times_mapped,
        mask_obs_mapped,
        dose_mapped,
        study_ctx_mapped,
    ) = ot.sample_plan_with_conditioning(x0, x1, obs_times, mask_obs, dose, study_ctx)

    assert torch.equal(x0_mapped, x0[i])
    assert torch.equal(x1_mapped, x1[j])
    assert torch.equal(obs_times_mapped, obs_times[j])
    assert torch.equal(mask_obs_mapped, mask_obs[j])
    assert torch.equal(dose_mapped, dose[j])

    (
        x_ctx_mapped,
        obs_times_ctx_mapped,
        context_obs_mask_mapped,
        mask_context_individuals_mapped,
        dose_ctx_mapped,
    ) = study_ctx_mapped
    assert torch.equal(x_ctx_mapped, x_ctx[j])
    assert torch.equal(obs_times_ctx_mapped, obs_times_ctx[j])
    assert torch.equal(context_obs_mask_mapped, context_obs_mask[j])
    assert torch.equal(mask_context_individuals_mapped, mask_context_individuals[j])
    assert torch.equal(dose_ctx_mapped, dose_ctx[j])


@pytest.mark.skip("Long loading")
def test_flow_pk_sample(inference_type):
    sample_size = 2  # number of new individuals to sample
    num_steps = 10  # number of flow integration steps
    cfg = _flow_pk_config_from_file(inference_type)
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)
    batch: AICMECompartmentsDataBatch = batch_list[0]
    model = FlowPK(cfg)
    pred_values, pred_times, mask = model.sample_new_individual(
        batch, sample_size=sample_size, num_steps=num_steps
    )
    pred_values = pred_values.transpose(0, 1)  # (B, S, T, 1)
    assert pred_values.shape[1] == sample_size


@pytest.mark.skipif(True, reason="Skipping because to long")
def test_flow_pk_sample_individual_prediction_shapes(inference_type):
    """Test `sample_individual_prediction` returns expected tuple and shapes.

    Returns:
      - sampled: [S, B, It, Tr, 1]
      - time:    [S, B, It, Tr, 1]
      - real:    [B, It, Tr, 1]
      - mask:    [B, It, Tr]
    """
    S = 2  # number of new individuals to sample
    num_steps = 10  # number of flow integration steps
    cfg = _flow_pk_config_from_file(inference_type)
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)
    batch: AICMECompartmentsDataBatch = batch_list[0]
    model = FlowPK(cfg)
    sampled, tgrid, real, mask = model.sample_individual_prediction(batch, sample_size=S)
    assert sampled.dim() == 5 and tgrid.dim() == 5
    assert sampled.size(0) == S and tgrid.size(0) == S
    assert sampled.size(-1) == 1 and tgrid.size(-1) == 1
    assert real.dim() == 4 and real.size(-1) == 1
    assert mask.dim() == 3


def test_flow_pk_sample_plot(inference_type):
    """Generate and save sample plots for visual inspection."""
    cfg = _flow_pk_config_from_file(inference_type)
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)
    model = FlowPK(cfg)

    # gather decode times and generate new individuals
    decode_times = gather_distinct_times_per_substance(batch_list[0]) if batch_list else None
    studies = model.sample_new_individuals_from_batchlist_to_study_json(
        batch_list,
        decode_times=decode_times,
        num_steps=10,
        sample_size=1,
    )
    assert studies, "No studies generated for plotting"
    from pff import reports_dir

    save_dir = reports_dir / "test" / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    requested_out_path = save_dir / "flow_pk_sample_plot.png"
    saved_out_path = save_dir / "flow_pk_sample_plot.png"

    returned_path = plot_list_list_study_json(studies=studies, file_name=str(requested_out_path))
    assert returned_path == str(saved_out_path)
    assert saved_out_path.exists(), f"Plot file not found: {saved_out_path}"


if __name__ == "__main__":
    inference_type = "flow-pk-predictive"  # or 'flow-pk-predictive'
    # test_flow_pk_from_file(inference_type)
    test_flow_pk_from_file(inference_type)
    # test_flow_pk_sample(inference_type)
    # test_flow_pk_sample_individual_prediction_shapes(inference_type)
