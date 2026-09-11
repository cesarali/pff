"""Unit tests for NewAICMEPK model (ContextVAE PK).

Covers:
- Training step smoke test on a tiny config.
- Sampling of new individuals from study context.
- Individual prediction sampling for target individuals.

We keep sizes tiny for speed and rely on the synthetic data module.
"""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from pff import config_dir
from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.data.data_empirical.json_schema import (
    canonicalize_study,
    prediction_stats,
)
from pff.data.datasets.aicme_datasets import (
    AICMECompartmentsDataBatch,
    AICMECompartmentsDataModule,
)
from pff.models.amortized_inference.aicme import (
    AICMEPK,
    AICMEForwardOutputs,
)
from pff.models.utils.new_pk_callbacks import NewPKEmpiricalEvaluationCallback
from pff.utils.plots.databatch_plot import (
    _normalise_substance_name,
    plot_list_list_study_json,
)
from tests.helpers import DummyExperiment, DummyLogger, DummyTrainer
from pff.training.basic_experiment import BasicLightningExperiment


def _aicme_config() -> NodePKExperimentConfig:
    """Return a minimal but representative configuration for NewAICMEPK tests.

    - Keeps everything small and deterministic.
    """
    cfg = NodePKExperimentConfig()
    cfg.train = replace(
        cfg.train,
        batch_size=4,
        num_workers=0,
        persistent_workers=False,
        epochs=1,
    )
    cfg.mix_data = replace(
        cfg.mix_data,
        train_size=2,
        val_size=2,
        test_size=2,
        n_of_permutations=3,  # >1 to exercise permutation aggregation
        n_of_target_individuals=1,
        test_empirical_datasets=["cesarali/lenuzza-2016", "cesarali/Indometacin"],
    )
    # cesarali/Indometacin
    cfg.meta_study = replace(cfg.meta_study, num_individuals_range=(3, 3))
    cfg.network = replace(cfg.network, aggregator_type="mean")

    cfg.target_observations = replace(
        cfg.target_observations,
        split_past_future=True,
        min_past=3,
        max_past=5,
        max_num_obs=10,
    )

    cfg.context_observations = replace(
        cfg.target_observations,
        split_past_future=False,
        max_num_obs=10,
    )

    return cfg


def _aicme_config_from_file() -> NodePKExperimentConfig:
    """Load the default YAML config and shrink it for unit tests."""

    default_yaml = Path(config_dir) / "experiment_configs" / "node-pk" / "base-homogeneous.yaml"
    cfg = NodePKExperimentConfig.from_yaml(str(default_yaml))
    return cfg


def _first_batch_list(dm: AICMECompartmentsDataModule):
    """Get the first list of batches (permutations) from the train loader on CPU."""
    dm.prepare_data()
    dm.setup()
    batch_list = next(iter(dm.train_dataloader()))
    assert isinstance(batch_list, list) and len(batch_list) >= 1
    return [b.to_device("cpu") for b in batch_list]


def _target_resolved_aicme_batch(db0: AICMECompartmentsDataBatch) -> AICMECompartmentsDataBatch:
    """Expand a single-target AICME batch into a deterministic two-target batch."""

    B = db0.target_obs.shape[0]
    target_obs = torch.cat([db0.target_obs, db0.target_obs], dim=1)
    target_rem_sim = torch.cat([db0.target_rem_sim, db0.target_rem_sim], dim=1)
    target_rem_sim_time = torch.cat([db0.target_rem_sim_time, db0.target_rem_sim_time], dim=1)
    target_rem_sim_mask = torch.cat([db0.target_rem_sim_mask, db0.target_rem_sim_mask], dim=1)

    target_obs_time = torch.tensor(
        [
            [
                [[1.0], [2.0], [0.0]],
                [[2.0], [4.0], [0.0]],
            ]
            for _ in range(B)
        ],
        dtype=db0.target_obs_time.dtype,
    )
    target_obs_mask = target_obs_time.squeeze(-1) > 0
    target_dosing_amounts = torch.stack(
        [
            torch.linspace(10.0, 10.0 + (B - 1), B, dtype=db0.target_dosing_amounts.dtype),
            torch.linspace(20.0, 20.0 + (B - 1), B, dtype=db0.target_dosing_amounts.dtype),
        ],
        dim=1,
    )
    target_dosing_route_types = torch.tensor(
        [[0, 1] for _ in range(B)], dtype=db0.target_dosing_route_types.dtype
    )

    return db0._replace(
        target_obs=target_obs[:, :, : target_obs_time.shape[2], :],
        target_obs_time=target_obs_time,
        target_obs_mask=target_obs_mask,
        target_rem_sim=target_rem_sim,
        target_rem_sim_time=target_rem_sim_time,
        target_rem_sim_mask=target_rem_sim_mask,
        target_dosing_amounts=target_dosing_amounts,
        target_dosing_route_types=target_dosing_route_types,
        mask_target_individuals=torch.ones(B, 2, dtype=torch.bool),
        target_subject_name=[[f"tgt_{b}_0", f"tgt_{b}_1"] for b in range(B)],
    )


def test_aicmepk_requires_target_split():
    """NewAICMEPK raises when target observations do not split past/future."""
    cfg = _aicme_config()
    cfg.target_observations = replace(cfg.target_observations, split_past_future=False)
    with pytest.raises(ValueError):
        AICMEPK(cfg)


def test_aicmepk_sample_new_individual_shapes():
    """Test `sample_new_individual` returns tensors with expected shapes.

    Returns samples: [S, B, Tdistinct_max, 1],
    time grid: [B, Tdistinct_max, 1],
    and mask: [B, Tdistinct_max].
    """
    cfg = _aicme_config()
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)
    db0: AICMECompartmentsDataBatch = batch_list[0]

    model = AICMEPK(cfg)
    S = 2

    samples, times, mask = model.sample_new_individual(db0, sample_size=S)

    # samples: [S,B,T,1]
    assert samples.dim() == 4
    assert samples.size(0) == S
    assert samples.size(-1) == 1

    # times: [B,T,1], mask: [B,T]
    B = samples.size(1)
    T = samples.size(2)
    assert times.shape == (B, T, 1)
    assert mask.shape == (B, T)


def test_aicmepk_select_unseen_dosing_prefers_target_and_context_fallback():
    """`select_unseen_dosing_from_databatch` prioritises available dosing."""

    cfg = _aicme_config()
    dm = AICMECompartmentsDataModule(cfg)
    db0 = _first_batch_list(dm)[0]

    model = AICMEPK(cfg)

    dose, route = model.select_unseen_dosing_from_databatch(db0)
    for b in range(db0.mask_target_individuals.size(0)):
        valid_targets = torch.nonzero(db0.mask_target_individuals[b], as_tuple=False).view(-1)
        if valid_targets.numel() == 0:
            continue
        idx = int(valid_targets[0].item())
        assert torch.isclose(dose[b], db0.target_dosing_amounts[b, idx])
        assert route[b] == db0.target_dosing_route_types[b, idx]

    empty_target_db = db0._replace(
        mask_target_individuals=torch.zeros_like(db0.mask_target_individuals),
        target_dosing_amounts=torch.zeros_like(db0.target_dosing_amounts),
        target_dosing_route_types=torch.zeros_like(db0.target_dosing_route_types),
    )

    dose_ctx, route_ctx = model.select_unseen_dosing_from_databatch(empty_target_db)
    for b in range(empty_target_db.mask_context_individuals.size(0)):
        valid_context = torch.nonzero(
            empty_target_db.mask_context_individuals[b], as_tuple=False
        ).view(-1)
        if valid_context.numel() == 0:
            assert dose_ctx[b] == 0
            assert route_ctx[b] == 0
            continue
        ctx_amounts = empty_target_db.context_dosing_amounts[b, valid_context]
        ctx_routes = empty_target_db.context_dosing_route_types[b, valid_context]
        assert torch.isclose(ctx_amounts, dose_ctx[b]).any()
        assert ctx_routes.eq(route_ctx[b]).any()


def test_aicmepk_sample_new_individual_accepts_custom_dosing():
    """Providing explicit dosing tensors is supported when sampling."""

    cfg = _aicme_config()
    dm = AICMECompartmentsDataModule(cfg)
    db0 = _first_batch_list(dm)[0]

    model = AICMEPK(cfg)

    B = db0.context_obs.size(0)
    custom_dose = torch.full((B,), 1.5, dtype=db0.context_dosing_amounts.dtype)
    custom_route = torch.full((B,), 2, dtype=db0.context_dosing_route_types.dtype)

    samples, _, _ = model.sample_new_individual(
        db0,
        sample_size=2,
        dosing=(custom_dose, custom_route),
    )

    assert samples.size(0) == 2


def test_aicmepk_resolve_target_dosing_returns_target_axis():
    """Target-resolved dosing helper should preserve the full target axis."""

    cfg = _aicme_config()
    dm = AICMECompartmentsDataModule(cfg)
    db0 = _target_resolved_aicme_batch(_first_batch_list(dm)[0])

    model = AICMEPK(cfg)
    dose, route = model.resolve_target_dosing_from_databatch(db0)

    assert torch.equal(dose, db0.target_dosing_amounts)
    assert torch.equal(route, db0.target_dosing_route_types)


def test_aicmepk_sample_new_individual_resolves_from_target():
    """Target-resolved sampling should use target count and target-only times."""

    cfg = _aicme_config()
    dm = AICMECompartmentsDataModule(cfg)
    db0 = _target_resolved_aicme_batch(_first_batch_list(dm)[0])

    model = AICMEPK(cfg)
    samples, times, mask = model.sample_new_individual(
        db0,
        sample_size=99,
        resolve_sampling_from_target=True,
    )

    assert samples.shape[0] == db0.target_obs.shape[1]
    assert samples.shape[1] == db0.target_obs.shape[0]
    assert samples.shape[-1] == 1
    assert torch.equal(times[0, mask[0], 0], torch.tensor([1.0, 2.0, 4.0], dtype=times.dtype))


def test_aicmepk_sample_new_individual_can_include_target_remainder_times():
    """Target-resolved AICME sampling can extend the decode grid with remainder times."""

    cfg = _aicme_config()
    dm = AICMECompartmentsDataModule(cfg)
    base_batch = _target_resolved_aicme_batch(_first_batch_list(dm)[0])
    batch_size = int(base_batch.target_obs.shape[0])
    db0 = base_batch._replace(
        target_rem_sim_time=torch.tensor(
            [
                [
                    [[5.0], [0.0]],
                    [[4.0], [0.0]],
                ]
                for _ in range(batch_size)
            ],
            dtype=base_batch.target_rem_sim_time.dtype,
        ),
        target_rem_sim_mask=torch.tensor(
            [
                [[True, False], [True, False]]
                for _ in range(batch_size)
            ],
            dtype=torch.bool,
        ),
    )

    model = AICMEPK(cfg)
    _, times, mask = model.sample_new_individual(
        db0,
        sample_size=1,
        resolve_sampling_from_target=True,
        include_rem=True,
    )

    assert torch.equal(times[0, mask[0], 0], torch.tensor([1.0, 2.0, 4.0, 5.0], dtype=times.dtype))


def test_aicmepk_sample_new_individual_resolve_target_requires_target_observations():
    """Target-resolved AICME sampling should reject empty target observation grids."""

    cfg = _aicme_config()
    dm = AICMECompartmentsDataModule(cfg)
    target_batch = _target_resolved_aicme_batch(_first_batch_list(dm)[0])
    db0 = target_batch._replace(
        target_obs_time=torch.zeros_like(target_batch.target_obs_time),
        target_obs_mask=torch.zeros_like(target_batch.target_obs_mask),
    )

    model = AICMEPK(cfg)

    with pytest.raises(ValueError, match="requires at least one valid target observation"):
        model.sample_new_individual(
            db0,
            sample_size=5,
            resolve_sampling_from_target=True,
        )


def test_aicmepk_resolves_split_latent_defaults() -> None:
    """AICME resolves local study/individual latent widths from base width."""
    cfg = _aicme_config()
    cfg.network = replace(cfg.network, zi_latent_dim=8, z_s_latent_dim=None, z_i_latent_dim=None)

    model = AICMEPK(cfg)

    assert model.base_latent_dim == 8
    assert model.study_latent_dim == 4
    assert model.individual_latent_dim == 8


def test_aicmepk_forward_pass_with_split_latent_dims() -> None:
    """Forward pass supports smaller study latents and full-width individual latents."""
    cfg = _aicme_config()
    cfg.network = replace(cfg.network, zi_latent_dim=8, z_s_latent_dim=4, z_i_latent_dim=8)
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)

    model = AICMEPK(cfg)
    outputs = model(batch_list)

    assert isinstance(outputs, AICMEForwardOutputs)
    assert outputs.total_loss is not None
    assert torch.isfinite(outputs.total_loss)


def test_aicmepk_sample_new_individual_with_split_latent_dims() -> None:
    """Sampling new individuals uses decoder projections for split latent widths."""
    cfg = _aicme_config()
    cfg.network = replace(cfg.network, zi_latent_dim=8, z_s_latent_dim=4, z_i_latent_dim=8)
    dm = AICMECompartmentsDataModule(cfg)
    db0 = _first_batch_list(dm)[0]

    model = AICMEPK(cfg)
    samples, times, mask = model.sample_new_individual(db0, sample_size=2)

    assert samples.shape[0] == 2
    assert times.shape == samples.shape[1:]
    assert mask.shape == samples.shape[1:3]


def test_aicmepk_sample_individual_prediction_with_split_latent_dims() -> None:
    """Prediction sampling supports split study/individual latent widths."""
    cfg = _aicme_config()
    cfg.network = replace(cfg.network, zi_latent_dim=8, z_s_latent_dim=4, z_i_latent_dim=8)
    dm = AICMECompartmentsDataModule(cfg)
    db0 = _first_batch_list(dm)[0]

    model = AICMEPK(cfg)
    sampled, tgrid, real, mask = model.sample_individual_prediction(db0, sample_size=2)

    assert sampled.shape[0] == 2
    assert tgrid.shape == sampled.shape
    assert real.dim() == 4 and real.size(-1) == 1
    assert mask.dim() == 3


@pytest.mark.skipif(True, reason="Skipping because to long")
def test_aicmepk_sample_new_individuals_to_studyjson():
    """Test `sample_new_individuals_to_studyjson` returns StudyJSONs
    with the expected structure and number of new target individuals.

    Each study should contain its context individuals and exactly `S`
    new target individuals, each with sampled observations and times
    that match the mask.
    """
    cfg = _aicme_config()
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)
    db0: AICMECompartmentsDataBatch = batch_list[0]

    model = AICMEPK(cfg)
    S = 3

    studies = model.sample_new_individuals_to_studyjson(db0, sample_size=S)
    canon_study = canonicalize_study(studies[0])

    # There should be one StudyJSON per batch element
    assert isinstance(studies, list)
    assert all(isinstance(st, dict) for st in studies)

    for study, mask in zip(studies, model.sample_new_individual(db0, sample_size=S)[2]):
        # Schema keys exist
        assert "context" in study and "target" in study and "meta_data" in study

        # Context individuals copied over
        assert isinstance(study["context"], list)
        assert all("observations" in ind for ind in study["context"])

        # Target list should have S new individuals
        targets = study["target"]
        assert len(targets) == S

        for ind in targets:
            # Observations and times aligned
            obs = ind["observations"]
            times = ind["observation_times"]
            assert isinstance(obs, list) and isinstance(times, list)
            assert len(obs) == len(times)

            # Length should equal number of valid decode times
            assert len(times) == int(mask.sum().item())


@pytest.mark.skipif(True, reason="Skipping because to long")
def test_aicmepk_sample_individual_prediction_shapes():
    """Test `sample_individual_prediction` returns expected tuple and shapes.

    Returns:
      - sampled: [S, B, It, Tr, 1]
      - time:    [S, B, It, Tr, 1]
      - real:    [B, It, Tr, 1]
      - mask:    [B, It, Tr]
    """
    cfg = _aicme_config()
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)
    db0: AICMECompartmentsDataBatch = batch_list[0]

    model = AICMEPK(cfg)
    S = 5
    sampled, tgrid, real, mask = model.sample_individual_prediction(db0, sample_size=S)

    assert sampled.dim() == 5 and tgrid.dim() == 5
    assert sampled.size(0) == S and tgrid.size(0) == S
    assert sampled.size(-1) == 1 and tgrid.size(-1) == 1
    assert real.dim() == 4 and real.size(-1) == 1
    assert mask.dim() == 3


@pytest.mark.skipif(True, reason="Skipping because to long")
def test_aicmepk_batch_list_to_tensors_shapes():
    """Verify tensor version aggregates permutations correctly."""
    cfg = _aicme_config()
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)

    model = AICMEPK(cfg)
    S = 2
    sampled, tgrid, real, mask = model.sample_individual_prediction_from_batch_list_to_tensors(
        batch_list, sample_size=S
    )

    assert sampled.size(0) == S and tgrid.shape == sampled.shape
    assert real.dim() == 4 and mask.dim() == 3


@pytest.mark.skipif(True, reason="Skipping because to long")
def test_aicmepk_prediction_only_zeroes_reconstruction_losses():
    """`prediction_only` skips reconstruction losses while keeping prediction terms."""

    cfg = _aicme_config()
    cfg.network = replace(cfg.network, prediction_only=True)
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)

    model = AICMEPK(cfg)
    outputs = model(batch_list)
    losses = outputs.to_dict()

    zero = torch.zeros_like(losses["pred_loss"])
    recon_keys = [
        "recon_loss",
        "kl_s",
        "kl_i",
        "kl_init",
        "kl_zs_zsN",
        "rmse",
        "log_rmse",
        "r2",
        "log_r2",
        "init_rmse",
    ]
    for key in recon_keys:
        assert torch.isclose(losses[key], zero).item(), f"Expected {key} to be zero"

    assert torch.isfinite(losses["pred_loss"]).item()
    assert not torch.isclose(losses["pred_loss"], zero).item()


@pytest.mark.skipif(True, reason="Skipping because to long")
def test_aicmepk_reconstruction_and_prediction_only_are_mutually_exclusive():
    """Setting both switches raises to avoid ambiguous optimisation objectives."""

    cfg = _aicme_config()
    cfg.network = replace(cfg.network, prediction_only=True, reconstruction_only=True)

    with pytest.raises(ValueError):
        AICMEPK(cfg)


@pytest.mark.skipif(True, reason="Skipping because to long")
def test_aicmepk_batch_list_to_studyjson_and_stats(tmp_path, monkeypatch):
    """Ensure StudyJSON conversion stores samples and computes stats."""
    cfg = _aicme_config()
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)

    model = AICMEPK(cfg)
    S = 2
    studies = model.sample_individual_prediction_from_batch_list_to_studyjson(
        batch_list, sample_size=S
    )
    assert isinstance(studies, list) and len(studies) == len(batch_list)
    first_study = studies[0][0]
    tgt_ind = first_study["target"][0]
    assert len(tgt_ind["prediction_samples"]) == S

    prediction_stats(first_study)
    assert "prediction_mean" in tgt_ind and len(tgt_ind["prediction_mean"]) == len(
        tgt_ind["prediction_times"]
    )

    returned_plot_path = plot_list_list_study_json(
        [[first_study]], file_name=str(tmp_path / "plot.png")
    )
    assert returned_plot_path == str(tmp_path / "plot.png")
    assert (tmp_path / "plot.png").is_file()

    from matplotlib import axes

    captured_titles: list[str] = []

    original_set_title = axes.Axes.set_title

    def _capture_title(self, title, *args, **kwargs):
        captured_titles.append(title)
        return original_set_title(self, title, *args, **kwargs)

    monkeypatch.setattr(axes.Axes, "set_title", _capture_title)

    separate_files = plot_list_list_study_json(
        [[first_study]],
        file_name=str(tmp_path / "plot.png"),
        plot_all_separately=True,
    )
    assert isinstance(separate_files, list) and len(separate_files) == 1
    assert Path(separate_files[0]).is_file()
    assert separate_files[0].endswith("_permutation_0.png")
    expected_title = _normalise_substance_name(
        first_study["meta_data"].get("substance_name"), "substance_0"
    )
    assert captured_titles and expected_title in captured_titles

    # Limiting the number of permutations and rows should reduce the files saved
    study_a = deepcopy(first_study)
    study_b = deepcopy(first_study)
    study_b["meta_data"]["substance_name"] = "AltSubstance"
    multi_studies = [[study_a, study_b], [study_a, study_b]]

    limited_files = plot_list_list_study_json(
        multi_studies,
        file_name=str(tmp_path / "plot_multi.png"),
        plot_all_separately=True,
        number_of_rows=1,
        number_of_columns=1,
    )
    assert isinstance(limited_files, list) and len(limited_files) == 1

    all_files = plot_list_list_study_json(
        multi_studies,
        file_name=str(tmp_path / "plot_multi.png"),
        plot_all_separately=True,
        number_of_rows=None,
        number_of_columns=None,
    )
    assert isinstance(all_files, list) and len(all_files) == 4


@pytest.mark.skipif(True, reason="Skipping because to long")
def test_aicmepk_log_empirical_evaluation_logs_metrics_and_images(tmp_path: Path):
    """`_log_empirical_evaluation` logs empirical plots/metrics via DummyLogger."""

    cfg = _aicme_config_from_file()
    dm = AICMECompartmentsDataModule(cfg)
    _first_batch_list(dm)

    model = AICMEPK(cfg)
    model.train()

    dummy_experiment = DummyExperiment(datamodule=dm)
    dummy_logger = DummyLogger(dummy_experiment)
    dummy_trainer = DummyTrainer(dummy_logger, datamodule=dm)
    dummy_trainer.default_root_dir = str(tmp_path)
    reports_test_dir = (tmp_path / "training_images").resolve()
    reports_test_dir.mkdir(parents=True, exist_ok=True)

    callback = NewPKEmpiricalEvaluationCallback(model_label="aicme")
    callback._log_empirical_evaluation(dummy_trainer, model, epoch_label="unit")

    assert callback._last_empirical_logging_epoch == dummy_trainer.current_epoch
    assert model.training, "Model must return to training mode"
    assert dummy_experiment.logged_metrics, "Expected empirical metrics to be logged"
    assert dummy_experiment.logged_images, "Expected empirical plots to be logged"
    for entry in dummy_experiment.logged_images:
        image_path = Path(entry["path"]).resolve()
        assert image_path.exists()
        assert reports_test_dir in image_path.parents or image_path.parent == reports_test_dir


@pytest.mark.skipif(True, reason="Skipping because to long")
def test_aicmepk_forward_on_empirical_batches() -> None:
    """Forward pass on empirical evaluation batches returns correct shapes."""
    cfg = _aicme_config()
    dm = AICMECompartmentsDataModule(cfg)
    dm.prepare_data()

    # repo_id = "cesarali/lenuzza-2016"
    repo_id = "cesarali/Indometacin"
    empirical_batches = dm.get_empirical_test_batches()
    if repo_id not in empirical_batches:
        pytest.skip(f"Empirical dataset '{repo_id}' is unavailable")

    batch_list = empirical_batches[repo_id]
    assert batch_list, "Expected empirical batches for evaluation"

    model = AICMEPK(cfg)
    outputs = model(batch_list)
    losses = outputs.to_dict()

    assert "loss" in losses
    recon = outputs.heads.get("reconstruction")
    assert recon is not None
    mean = recon["mean"]
    logvar = recon["logvar"]
    target = recon["target"]
    mask = recon["mask"]
    assert mean.shape == target.shape  # [B, I, T, 1]
    assert logvar.shape == target.shape
    assert mask.shape == target.shape[:-1]


def test_aicmepk_forward_pass() -> None:
    """Full forward pass through NewAICMEPK using a tiny data batch."""
    cfg = _aicme_config()
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)
    model = AICMEPK(cfg)
    outputs = model(batch_list)

    assert isinstance(outputs, AICMEForwardOutputs)

    losses = outputs.to_dict()
    assert "loss" in losses

    recon = outputs.heads.get("reconstruction")
    assert recon is not None
    mean = recon["mean"]
    logvar = recon["logvar"]
    target = recon["target"]
    mask = recon["mask"]

    assert mean.shape == target.shape  # [B,I,T,1]
    assert logvar.shape == target.shape
    assert mask.shape == target.shape[:-1]


def test_aicmepk_forward_pass_log_and_max_scaler() -> None:
    """Forward smoke test with scaler-owned log+max normalization."""
    cfg = _aicme_config()
    cfg.mix_data = replace(
        cfg.mix_data,
        log_and_max=True,
        normalize_by_max=True,
        z_score_normalization=False,
    )
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)
    model = AICMEPK(cfg)

    assert model.scaler.v_method == "log_and_max"

    outputs = model(batch_list)
    assert isinstance(outputs, AICMEForwardOutputs)
    assert outputs.total_loss is not None
    assert torch.isfinite(outputs.total_loss)


@pytest.mark.skip(reason="model might not exist")
def test_load_aicmepk_from_comet_key():
    experiment = BasicLightningExperiment.from_experiment_comet("5d1f25d0b37f44ea97800700d4769c96")


@pytest.mark.skip(reason="model might not exist")
def test_load_aicmepk_from_experiment_dir():
    experiment = BasicLightningExperiment.from_experiment_dir(
        "/home/cesarali/Pharma/pff/results/comet/functional-flow-pk/e19fb57257cd4501bcdb3a2294dc559e"
    )
    print(experiment.model)


if __name__ == "__main__":
    test_load_aicmepk_from_experiment_dir()
