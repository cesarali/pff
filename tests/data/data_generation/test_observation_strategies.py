import json
from pathlib import Path

import pytest
from torchtyping import TensorType

from pff import config_dir, data_dir
from pff.config_classes.data_config import MetaStudyConfig, ObservationsConfig
from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.data.data_generation.observations_classes import (
    FixedRegularGridStrategy,
    ObservationStrategyFactory,
    PKPeakHalfLifeStrategy,
)

torch = pytest.importorskip("torch")


def _dummy_simulation(meta_cfg: MetaStudyConfig, batch: int = 2):
    """Return random simulation data and a normalised time grid.

    Parameters
    ----------
    meta_cfg : MetaStudyConfig
        Provides ``time_num_steps`` to set the simulation length.
    batch : int, default=2
        Number of individuals in the batch.

    Returns
    -------
    full_simulation, full_times : Tensor[batch, S]
        Random concentrations and a normalised timeline.
    """
    S = meta_cfg.time_num_steps
    full_sim: TensorType["B", "S"] = torch.rand(batch, S)
    time_grid = torch.linspace(0.0, 1.0, S)
    full_times: TensorType["B", "S"] = time_grid.repeat(batch, 1)
    return full_sim, full_times


def _load_empirical_block(max_individuals: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load empirical individuals from Lenuzza or a fixture dataset.

    Parameters
    ----------
    max_individuals:
        Upper bound on the number of individuals returned.

    Returns
    -------
    obs, times, mask : torch.Tensor
        Padded empirical observations, timestamps, and validity mask.
    """

    default_json = Path(data_dir) / "preprocessed" / "lenuzza_2016.json"
    fallback_json = Path(__file__).resolve().parents[1] / "fixtures" / "studies_long_list.json"
    source = default_json if default_json.exists() else fallback_json

    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    studies = payload if isinstance(payload, list) else [payload]
    if not studies:
        pytest.skip(f"No empirical studies available in {source}")

    first_study = studies[0]
    individuals = list(first_study.get("context", []))
    if not individuals:
        individuals = list(first_study.get("target", []))
    if not individuals:
        pytest.skip(f"No individuals found in empirical study from {source}")

    selected = individuals[:max_individuals] if max_individuals else individuals
    if not selected:
        pytest.skip("Requested zero empirical individuals")

    max_len = max(len(ind.get("observations", [])) for ind in selected)
    if max_len == 0:
        pytest.skip("Empirical individuals do not contain observations")

    obs = torch.zeros(len(selected), max_len, dtype=torch.float32)
    times = torch.zeros_like(obs)
    mask = torch.zeros(len(selected), max_len, dtype=torch.bool)

    for row, individual in enumerate(selected):
        obs_values = torch.tensor(individual.get("observations", []), dtype=torch.float32)
        time_values = torch.tensor(individual.get("observation_times", []), dtype=torch.float32)
        valid = min(obs_values.shape[0], max_len)
        if valid:
            obs[row, :valid] = obs_values[:valid]
            times[row, :valid] = time_values[:valid]
            mask[row, :valid] = True

    return obs, times, mask


@pytest.mark.parametrize(
    "strategy_type",
    ["pk_peak_half_life", "random"],
)
def test_strategy_shapes_match_generation(strategy_type: str):
    """Each strategy's ``generate`` output matches ``get_shapes``.

    The test instantiates two observation strategies via
    :class:`ObservationStrategyFactory` and checks that the padded tensor
    sizes produced by :meth:`generate` correspond to ``get_shapes``.
    """
    meta_cfg = MetaStudyConfig(time_num_steps=40)  # S = 40
    obs_cfg = ObservationsConfig(
        type=strategy_type,
        max_num_obs=6,
        add_rem=True,
        split_past_future=True,
        min_past=1,
        max_past=3,
    )
    strategy = ObservationStrategyFactory.from_config(obs_cfg, meta_cfg)

    # Predict maximum shapes
    max_obs, max_rem = strategy.get_shapes()

    full_sim, full_times = _dummy_simulation(meta_cfg, batch=2)

    kwargs = {}
    if strategy_type in {"pk_peak_half_life", "observations_pk_peak_halflife"}:
        kwargs["time_scales"] = torch.tensor([0.3, 0.5])  # [t_peak, t_half]

    out = strategy.generate(full_simulation=full_sim, full_simulation_times=full_times, **kwargs)
    obs, obs_time, obs_mask, rem_sim, rem_time, rem_mask, _ = out

    assert obs.shape == (2, max_obs)
    assert obs_time.shape == (2, max_obs)
    assert obs_mask.shape == (2, max_obs)
    if max_rem:
        assert rem_sim is not None and rem_time is not None and rem_mask is not None
        assert rem_sim.shape == (2, max_rem)
        assert rem_time.shape == (2, max_rem)
        assert rem_mask.shape == (2, max_rem)
    else:
        assert rem_sim is None and rem_time is None and rem_mask is None


def test_pk_peak_halflife_shapes_without_split():
    """``split_past_future=False`` exposes the canonical PK grid capacity."""

    meta_cfg = MetaStudyConfig(time_num_steps=120)
    obs_cfg = ObservationsConfig(
        type="pk_peak_half_life",
        max_num_obs=15,
        add_rem=True,
        split_past_future=False,
    )
    strategy = ObservationStrategyFactory.from_config(obs_cfg, meta_cfg)

    max_obs, max_rem = strategy.get_shapes()

    template_cap = min(
        obs_cfg.max_num_obs,
        meta_cfg.time_num_steps,
        PKPeakHalfLifeStrategy._RAW_CANONICAL_POINTS,
    )

    assert max_obs == template_cap
    assert max_rem == template_cap

    # A coarse simulation grid further limits the remaining capacity
    coarse_meta = MetaStudyConfig(time_num_steps=6)
    coarse_obs_cfg = ObservationsConfig(
        type="pk_peak_half_life",
        max_num_obs=15,
        add_rem=True,
        split_past_future=False,
    )
    coarse_strategy = ObservationStrategyFactory.from_config(coarse_obs_cfg, coarse_meta)
    coarse_obs, coarse_rem = coarse_strategy.get_shapes()
    coarse_cap = min(
        obs_cfg.max_num_obs,
        coarse_meta.time_num_steps,
        PKPeakHalfLifeStrategy._RAW_CANONICAL_POINTS,
    )
    assert coarse_obs == coarse_cap
    assert coarse_rem == coarse_cap


def test_pk_peak_halflife_deterministic_override_matches_raw():
    """Setting ``deterministic_only`` bypasses the randomized branch."""

    meta_cfg = MetaStudyConfig(time_num_steps=40)
    obs_cfg = ObservationsConfig(
        type="pk_peak_half_life",
        max_num_obs=10,
        add_rem=True,
        split_past_future=False,
    )
    strategy = ObservationStrategyFactory.from_config(obs_cfg, meta_cfg)

    full_sim, full_times = _dummy_simulation(meta_cfg, batch=2)
    time_scales = torch.tensor([0.3, 0.6])

    raw_out = strategy._generate_raw(
        full_simulation=full_sim,
        full_simulation_times=full_times,
        time_scales=time_scales,
    )
    gen_out = strategy.generate(
        full_simulation=full_sim,
        full_simulation_times=full_times,
        time_scales=time_scales,
        deterministic_only=True,
    )

    for raw_item, gen_item in zip(raw_out, gen_out):
        if raw_item is None:
            assert gen_item is None
        else:
            assert torch.equal(raw_item, gen_item)


def test_pk_peak_halflife_generate_handles_empty_batch():
    """PK strategy should handle empty [B=0, S] inputs without indexing row 0."""

    meta_cfg = MetaStudyConfig(time_num_steps=40)
    obs_cfg = ObservationsConfig(
        type="pk_peak_half_life",
        max_num_obs=10,
        add_rem=True,
        split_past_future=False,
    )
    strategy = ObservationStrategyFactory.from_config(obs_cfg, meta_cfg)
    max_obs, max_rem = strategy.get_shapes()
    time_scales = torch.tensor([0.3, 0.6], dtype=torch.float32)

    empty_sim = torch.zeros(0, meta_cfg.time_num_steps, dtype=torch.float32)
    empty_times = torch.zeros(0, meta_cfg.time_num_steps, dtype=torch.float32)

    # Deterministic branch: exercises _align_simulation_to_canonical.
    obs, obs_time, obs_mask, rem_sim, rem_time, rem_mask, _ = strategy.generate(
        full_simulation=empty_sim,
        full_simulation_times=empty_times,
        time_scales=time_scales,
        deterministic_only=True,
    )
    assert obs.shape == (0, max_obs)
    assert obs_time.shape == (0, max_obs)
    assert obs_mask.shape == (0, max_obs)
    assert rem_sim is not None and rem_time is not None and rem_mask is not None
    assert rem_sim.shape == (0, max_rem)
    assert rem_time.shape == (0, max_rem)
    assert rem_mask.shape == (0, max_rem)

    # Randomized branch: exercises _generate_random.
    strategy.randomize_prob = 1.0
    obs, obs_time, obs_mask, rem_sim, rem_time, rem_mask, _ = strategy.generate(
        full_simulation=empty_sim,
        full_simulation_times=empty_times,
        time_scales=time_scales,
    )
    assert obs.shape == (0, max_obs)
    assert obs_time.shape == (0, max_obs)
    assert obs_mask.shape == (0, max_obs)
    assert rem_sim is not None and rem_time is not None and rem_mask is not None
    assert rem_sim.shape == (0, max_rem)
    assert rem_time.shape == (0, max_rem)
    assert rem_mask.shape == (0, max_rem)


@pytest.mark.parametrize("deterministic_only", [True, False])
def test_pk_peak_halflife_removes_duplicate_simulation_indices(deterministic_only: bool):
    """A coarse simulator grid never yields duplicated observation timestamps."""

    torch.manual_seed(0)
    full_times = torch.tensor(
        [
            0.0000,
            0.1616,
            0.3232,
            0.4848,
            0.6465,
            0.8081,
            0.9697,
            1.1313,
            1.2929,
            1.4545,
            1.6162,
            1.7778,
            1.9394,
            2.1010,
            2.2626,
            2.4242,
            2.5859,
            2.7475,
            2.9091,
            3.0707,
            3.2323,
            3.3939,
            3.5556,
            3.7172,
            3.8788,
            4.0404,
            4.2020,
            4.3636,
            4.5253,
            4.6869,
            4.8485,
            5.0101,
            5.1717,
            5.3333,
            5.4949,
            5.6566,
            5.8182,
            5.9798,
            6.1414,
            6.3030,
            6.4646,
            6.6263,
            6.7879,
            6.9495,
            7.1111,
            7.2727,
            7.4343,
            7.5960,
            7.7576,
            7.9192,
            8.0808,
            8.2424,
            8.4040,
            8.5657,
            8.7273,
            8.8889,
            9.0505,
            9.2121,
            9.3737,
            9.5354,
            9.6970,
            9.8586,
            10.0202,
            10.1818,
            10.3434,
            10.5051,
            10.6667,
            10.8283,
            10.9899,
            11.1515,
            11.3131,
            11.4747,
            11.6364,
            11.7980,
            11.9596,
            12.1212,
            12.2828,
            12.4444,
            12.6061,
            12.7677,
            12.9293,
            13.0909,
            13.2525,
            13.4141,
            13.5758,
            13.7374,
            13.8990,
            14.0606,
            14.2222,
            14.3838,
            14.5455,
            14.7071,
            14.8687,
            15.0303,
            15.1919,
            15.3535,
            15.5152,
            15.6768,
            15.8384,
            16.0000,
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    experiment_dir = config_dir / "experiment_configs" / "node-pk"
    meta_cfg = MetaStudyConfig.from_yaml(experiment_dir / "base-homogeneous.meta_study.yaml")
    assert meta_cfg.time_num_steps == full_times.shape[1], (
        "Meta-study grid must match the coarse simulation"
    )
    obs_cfg = ObservationsConfig.from_yaml(
        experiment_dir / "base-homogeneous.observations.yaml",
        section="context_observations",
    )
    strategy = ObservationStrategyFactory.from_config(obs_cfg, meta_cfg)

    full_sim = torch.arange(1, full_times.shape[1] + 1, dtype=full_times.dtype).unsqueeze(0)

    if not deterministic_only:
        strategy.randomize_prob = 1.0

    generator = torch.Generator().manual_seed(42)
    obs, obs_time, obs_mask, *_ = strategy.generate(
        full_simulation=full_sim,
        full_simulation_times=full_times,
        time_scales=torch.tensor([3.7203, 8.0], dtype=full_times.dtype),
        deterministic_only=deterministic_only,
        generator=generator,
    )

    for row in range(obs_time.shape[0]):
        valid_times = obs_time[row, obs_mask[row]]
        assert torch.unique(valid_times).numel() == valid_times.numel()
        if valid_times.numel() > 1:
            assert torch.all(valid_times[1:] > valid_times[:-1])


def test_pk_peak_halflife_fix_and_release_past_selection():
    """The strategy can override and restore the sampled past count."""

    meta_cfg = MetaStudyConfig(time_num_steps=40)
    obs_cfg = ObservationsConfig(
        type="pk_peak_half_life",
        max_num_obs=10,
        add_rem=True,
        split_past_future=True,
        min_past=2,
        max_past=4,
    )
    strategy = ObservationStrategyFactory.from_config(obs_cfg, meta_cfg)

    # Prepare canonical tensors with four valid entries per row.
    canonical_vals = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    canonical_times = canonical_vals.clone()
    canonical_mask = torch.ones_like(canonical_vals, dtype=torch.bool)

    strategy.fix_past_selection(3)
    obs, _, obs_mask, _, _, _ = strategy._assemble_from_canonical(
        canonical_vals,
        canonical_times,
        canonical_mask,
    )
    assert obs_mask.sum(dim=1).tolist() == [3, 3]

    # Releasing returns to the RNG-driven behaviour.
    strategy.release_past_selection()
    generator = torch.Generator().manual_seed(1234)
    reference_gen = torch.Generator()
    reference_gen.set_state(generator.get_state())
    expected = int(torch.randint(2, 5, (1,), generator=reference_gen).item())
    obs, _, obs_mask, _, _, _ = strategy._assemble_from_canonical(
        canonical_vals,
        canonical_times,
        canonical_mask,
        generator=generator,
    )
    assert obs_mask.sum(dim=1).tolist() == [expected, expected]

    with pytest.raises(ValueError):
        strategy.fix_past_selection(obs_cfg.max_past + 1)


def test_pk_peak_halflife_generative_bias_true():
    """The biased mode yields ~50% zero past observations and non-zero otherwise."""

    meta_cfg = MetaStudyConfig(time_num_steps=40)
    obs_cfg = ObservationsConfig(
        type="pk_peak_half_life",
        max_num_obs=10,
        add_rem=True,
        split_past_future=True,
        min_past=0,
        max_past=5,
        generative_bias=True,
    )
    strategy = ObservationStrategyFactory.from_config(obs_cfg, meta_cfg)

    batch = 3000
    canonical_len = 8
    canonical_vals = torch.arange(batch * canonical_len, dtype=torch.float32).reshape(
        batch, canonical_len
    )
    canonical_times = canonical_vals.clone()
    canonical_mask = torch.ones_like(canonical_vals, dtype=torch.bool)

    generator = torch.Generator().manual_seed(2026)
    _, _, obs_mask, _, _, _ = strategy._assemble_from_canonical(
        canonical_vals,
        canonical_times,
        canonical_mask,
        generator=generator,
    )
    counts = obs_mask.sum(dim=1).to(torch.int64)

    zero_share = (counts == 0).float().mean().item()
    assert 0.44 <= zero_share <= 0.56

    non_zero_counts = counts[counts > 0]
    assert non_zero_counts.numel() > 0
    assert int(non_zero_counts.min().item()) >= 1
    assert int(non_zero_counts.max().item()) <= obs_cfg.max_past


def test_strategy_from_yaml_config():
    """Factory builds strategies from a full YAML configuration.

    This test loads ``base-homogeneous.yaml`` and verifies that the
    strategy derived from its ``target_observations`` section generates
    tensors with shapes consistent with :meth:`get_shapes`.
    """
    cfg_path = config_dir / "experiment_configs" / "node-pk" / "base-homogeneous.yaml"
    cfg = NodePKExperimentConfig.from_yaml(cfg_path)
    meta_cfg = cfg.meta_study
    obs_cfg = cfg.target_observations  # split_past_future=True
    strategy = ObservationStrategyFactory.from_config(obs_cfg, meta_cfg)

    max_obs, max_rem = strategy.get_shapes()
    full_sim, full_times = _dummy_simulation(meta_cfg, batch=3)
    out = strategy.generate(
        full_simulation=full_sim,
        full_simulation_times=full_times,
        time_scales=torch.tensor([0.25, 0.5]),
    )
    obs, obs_time, obs_mask, rem_sim, rem_time, rem_mask, _ = out

    # --- print shapes for debugging ---
    print("obs.shape:", obs.shape, "expected:", (3, max_obs))
    print("obs_time.shape:", obs_time.shape, "expected:", (3, max_obs))
    print("obs_mask.shape:", obs_mask.shape, "expected:", (3, max_obs))
    print(
        "rem_sim.shape:", rem_sim.shape if rem_sim is not None else None, "expected:", (3, max_rem)
    )
    print(
        "rem_time.shape:",
        rem_time.shape if rem_time is not None else None,
        "expected:",
        (3, max_rem),
    )
    print(
        "rem_mask.shape:",
        rem_mask.shape if rem_mask is not None else None,
        "expected:",
        (3, max_rem),
    )

    # --- assertions ---
    assert obs.shape == (3, max_obs)
    assert obs_time.shape == (3, max_obs)
    assert obs_mask.shape == (3, max_obs)
    if max_rem:
        assert rem_sim is not None and rem_time is not None and rem_mask is not None
        assert rem_sim.shape == (3, max_rem)
        assert rem_time.shape == (3, max_rem)
        assert rem_mask.shape == (3, max_rem)
    else:
        assert rem_sim is None and rem_time is None and rem_mask is None


def test_generate_empirical_matches_synthetic_shapes():
    """Empirical padding mirrors synthetic generation behaviour."""

    empirical_obs, empirical_times, empirical_mask = _load_empirical_block(max_individuals=2)
    batch = empirical_obs.shape[0]

    meta_cfg = MetaStudyConfig(time_num_steps=40)
    obs_cfg = ObservationsConfig(
        type="pk_peak_half_life",
        max_num_obs=6,
        add_rem=True,
        split_past_future=True,
        min_past=1,
        max_past=3,
    )
    strategy = ObservationStrategyFactory.from_config(obs_cfg, meta_cfg)
    max_obs, max_rem = strategy.get_shapes()

    full_sim, full_times = _dummy_simulation(meta_cfg, batch=batch)

    torch.manual_seed(1234)
    synthetic_out = strategy.generate(
        full_simulation=full_sim,
        full_simulation_times=full_times,
        time_scales=torch.tensor([0.3, 0.5]),
    )
    syn_obs, syn_time, syn_mask, syn_rem_sim, syn_rem_time, syn_rem_mask, _ = synthetic_out
    names = ["syn_obs", "syn_time", "syn_mask", "syn_rem_sim", "syn_rem_time", "syn_rem_mask"]
    print(
        {
            name: tensor.shape
            for name, tensor in zip(
                names, (syn_obs, syn_time, syn_mask, syn_rem_sim, syn_rem_time, syn_rem_mask)
            )
        }
    )

    torch.manual_seed(1234)
    empirical_out = strategy.generate_empirical(
        empirical_obs=empirical_obs,
        empirical_times=empirical_times,
        empirical_mask=empirical_mask,
    )
    emp_obs, emp_time, emp_mask, emp_rem_sim, emp_rem_time, emp_rem_mask = empirical_out
    names = ["emp_obs", "emp_time", "emp_mask", "emp_rem_sim", "emp_rem_time", "emp_rem_mask"]
    print(
        {
            name: tensor.shape
            for name, tensor in zip(
                names, (emp_obs, emp_time, emp_mask, emp_rem_sim, emp_rem_time, emp_rem_mask)
            )
        }
    )

    assert syn_obs.shape == (batch, max_obs)
    assert emp_obs.shape == syn_obs.shape
    assert emp_time.shape == syn_time.shape == (batch, max_obs)
    assert emp_mask.shape == syn_mask.shape == (batch, max_obs)
    assert emp_mask.dtype == torch.bool
    assert syn_mask.dtype == torch.bool

    if max_rem:
        assert syn_rem_sim is not None and syn_rem_time is not None and syn_rem_mask is not None
        assert emp_rem_sim is not None and emp_rem_time is not None and emp_rem_mask is not None
        assert syn_rem_sim.shape == (batch, max_rem)
        assert emp_rem_sim.shape == syn_rem_sim.shape
        assert syn_rem_time.shape == (batch, max_rem)
        assert emp_rem_time.shape == syn_rem_time.shape
        assert syn_rem_mask.shape == (batch, max_rem)
        assert emp_rem_mask.shape == syn_rem_mask.shape
        assert syn_rem_mask.dtype == torch.bool
        assert emp_rem_mask.dtype == torch.bool

        syn_total = syn_mask.sum(dim=1).to(torch.int64) + syn_rem_mask.sum(dim=1).to(torch.int64)
        emp_total = emp_mask.sum(dim=1).to(torch.int64) + emp_rem_mask.sum(dim=1).to(torch.int64)
    else:
        assert syn_rem_sim is None and syn_rem_time is None and syn_rem_mask is None
        assert emp_rem_sim is None and emp_rem_time is None and emp_rem_mask is None
        syn_total = syn_mask.sum(dim=1).to(torch.int64)
        emp_total = emp_mask.sum(dim=1).to(torch.int64)

    max_total = max_obs + max_rem
    expected_total = empirical_mask.sum(dim=1).to(torch.int64).clamp(max=max_total)
    assert syn_total.shape == emp_total.shape
    assert torch.all(emp_total <= expected_total)
    assert torch.all(emp_total <= syn_total)


def test_drop_time_zero_observations_filters_synthetic_t0():
    """Synthetic generation can exclude observations sampled at ``t=0``."""

    meta_cfg = MetaStudyConfig(time_num_steps=5, time_start=0.0, time_stop=4.0)
    full_times = torch.linspace(0.0, 4.0, 5, dtype=torch.float32).unsqueeze(0)
    full_sim = torch.arange(1, 6, dtype=torch.float32).unsqueeze(0)
    time_scales = torch.tensor([1e-4, 1e-4], dtype=torch.float32)

    keep_cfg = ObservationsConfig(
        type="pk_peak_half_life",
        max_num_obs=5,
        add_rem=False,
        split_past_future=False,
        drop_time_zero_observations=False,
    )
    keep_strategy = ObservationStrategyFactory.from_config(keep_cfg, meta_cfg)
    keep_obs, keep_time, keep_mask, *_ = keep_strategy.generate(
        full_simulation=full_sim,
        full_simulation_times=full_times,
        time_scales=time_scales,
        deterministic_only=True,
    )
    keep_valid_times = keep_time[keep_mask]
    assert keep_valid_times.numel() > 0
    assert torch.any(keep_valid_times == 0.0)

    drop_cfg = ObservationsConfig(
        type="pk_peak_half_life",
        max_num_obs=5,
        add_rem=False,
        split_past_future=False,
        drop_time_zero_observations=True,
    )
    drop_strategy = ObservationStrategyFactory.from_config(drop_cfg, meta_cfg)
    drop_obs, drop_time, drop_mask, *_ = drop_strategy.generate(
        full_simulation=full_sim,
        full_simulation_times=full_times,
        time_scales=time_scales,
        deterministic_only=True,
    )

    assert keep_obs.shape == drop_obs.shape
    assert keep_time.shape == drop_time.shape
    assert drop_mask.sum().item() == 0


def test_drop_time_zero_observations_filters_empirical_t0():
    """Empirical generation can exclude measurements at non-positive time."""

    meta_cfg = MetaStudyConfig(time_num_steps=3)
    empirical_obs = torch.tensor([[0.2, 3.0, 1.5]], dtype=torch.float32)
    empirical_times = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float32)
    empirical_mask = torch.tensor([[True, True, True]])

    keep_cfg = ObservationsConfig(
        type="pk_peak_half_life",
        max_num_obs=3,
        add_rem=False,
        split_past_future=False,
        drop_time_zero_observations=False,
    )
    keep_strategy = ObservationStrategyFactory.from_config(keep_cfg, meta_cfg)
    _, keep_time, keep_mask, *_ = keep_strategy.generate_empirical(
        empirical_obs=empirical_obs,
        empirical_times=empirical_times,
        empirical_mask=empirical_mask,
    )
    keep_valid_times = keep_time[keep_mask]
    assert keep_valid_times.numel() == 3
    assert torch.any(keep_valid_times == 0.0)

    drop_cfg = ObservationsConfig(
        type="pk_peak_half_life",
        max_num_obs=3,
        add_rem=False,
        split_past_future=False,
        drop_time_zero_observations=True,
    )
    drop_strategy = ObservationStrategyFactory.from_config(drop_cfg, meta_cfg)
    _, drop_time, drop_mask, *_ = drop_strategy.generate_empirical(
        empirical_obs=empirical_obs,
        empirical_times=empirical_times,
        empirical_mask=empirical_mask,
    )
    drop_valid_times = drop_time[drop_mask]
    assert drop_valid_times.numel() == 2
    assert torch.all(drop_valid_times > 0.0)


def test_fixed_regular_grid_strategy_selects_deterministic_even_grid():
    """Fixed-grid sampling should be deterministic and remainder-free."""

    meta_cfg = MetaStudyConfig(time_num_steps=10, time_start=0.0, time_stop=9.0)
    obs_cfg = ObservationsConfig(
        type="fixed_regular_grid",
        max_num_obs=4,
        add_rem=False,
        split_past_future=False,
        drop_time_zero_observations=False,
    )
    strategy = ObservationStrategyFactory.from_config(obs_cfg, meta_cfg)

    assert isinstance(strategy, FixedRegularGridStrategy)
    assert strategy.get_shapes() == (4, 0)

    full_sim = torch.arange(20, dtype=torch.float32).view(2, 10)
    full_times = torch.linspace(0.0, 9.0, 10, dtype=torch.float32).repeat(2, 1)

    obs, obs_time, obs_mask, rem_sim, rem_time, rem_mask, _ = strategy.generate(
        full_simulation=full_sim,
        full_simulation_times=full_times,
    )

    expected_times = torch.tensor([0.0, 3.0, 6.0, 9.0], dtype=torch.float32)
    assert rem_sim is None and rem_time is None and rem_mask is None
    assert torch.equal(obs_mask, torch.ones_like(obs_mask, dtype=torch.bool))
    assert torch.allclose(obs_time[0], expected_times)
    assert torch.allclose(obs_time[1], expected_times)
    assert torch.all(torch.diff(obs_time[0]) > 0.0)


def test_fixed_regular_grid_strategy_drops_time_zero_observations():
    """Fixed-grid sampling should respect ``drop_time_zero_observations``."""

    meta_cfg = MetaStudyConfig(time_num_steps=5, time_start=0.0, time_stop=4.0)
    obs_cfg = ObservationsConfig(
        type="fixed_regular_grid",
        max_num_obs=5,
        add_rem=False,
        split_past_future=False,
        drop_time_zero_observations=True,
    )
    strategy = ObservationStrategyFactory.from_config(obs_cfg, meta_cfg)

    full_sim = torch.arange(1, 6, dtype=torch.float32).unsqueeze(0)
    full_times = torch.linspace(0.0, 4.0, 5, dtype=torch.float32).unsqueeze(0)

    _, obs_time, obs_mask, rem_sim, rem_time, rem_mask, _ = strategy.generate(
        full_simulation=full_sim,
        full_simulation_times=full_times,
    )

    valid_times = obs_time[obs_mask]
    assert rem_sim is None and rem_time is None and rem_mask is None
    assert valid_times.numel() == 4
    assert torch.all(valid_times > 0.0)


def test_fixed_regular_grid_strategy_can_start_after_time_zero():
    """Fixed-grid sampling can skip the first solver steps before spacing."""

    meta_cfg = MetaStudyConfig(time_num_steps=10, time_start=0.0, time_stop=9.0)
    obs_cfg = ObservationsConfig(
        type="fixed_regular_grid",
        max_num_obs=4,
        add_rem=False,
        split_past_future=False,
        drop_time_zero_observations=False,
        fixed_grid_start_index=2,
    )
    strategy = ObservationStrategyFactory.from_config(obs_cfg, meta_cfg)

    full_sim = torch.arange(10, dtype=torch.float32).unsqueeze(0)
    full_times = torch.linspace(0.0, 9.0, 10, dtype=torch.float32).unsqueeze(0)

    obs, obs_time, obs_mask, rem_sim, rem_time, rem_mask, _ = strategy.generate(
        full_simulation=full_sim,
        full_simulation_times=full_times,
    )

    expected_times = torch.tensor([2.0, 4.0, 7.0, 9.0], dtype=torch.float32)
    assert rem_sim is None and rem_time is None and rem_mask is None
    assert torch.equal(obs_mask, torch.ones_like(obs_mask, dtype=torch.bool))
    assert torch.allclose(obs_time[0], expected_times)
    assert torch.allclose(obs[0], expected_times)


def test_fixed_regular_grid_generate_empirical_truncates_without_remainder():
    """Empirical fixed-grid generation should copy and truncate in order."""

    meta_cfg = MetaStudyConfig(time_num_steps=4)
    obs_cfg = ObservationsConfig(
        type="fixed_regular_grid",
        max_num_obs=4,
        add_rem=False,
        split_past_future=False,
    )
    strategy = ObservationStrategyFactory.from_config(obs_cfg, meta_cfg)

    empirical_obs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float32)
    empirical_times = torch.tensor([[0.5, 1.0, 1.5, 2.0, 2.5]], dtype=torch.float32)
    empirical_mask = torch.tensor([[True, True, True, True, True]])

    obs, obs_time, obs_mask, rem_sim, rem_time, rem_mask = strategy.generate_empirical(
        empirical_obs=empirical_obs,
        empirical_times=empirical_times,
        empirical_mask=empirical_mask,
    )

    assert rem_sim is None and rem_time is None and rem_mask is None
    assert obs.shape == (1, 4)
    assert torch.allclose(obs[0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(obs_time[0], torch.tensor([0.5, 1.0, 1.5, 2.0]))
    assert torch.equal(obs_mask, torch.ones_like(obs_mask, dtype=torch.bool))


def test_fixed_regular_grid_generate_empirical_respects_start_offset():
    """Empirical fixed-grid generation should skip the configured initial slots."""

    meta_cfg = MetaStudyConfig(time_num_steps=6)
    obs_cfg = ObservationsConfig(
        type="fixed_regular_grid",
        max_num_obs=3,
        add_rem=False,
        split_past_future=False,
        fixed_grid_start_index=2,
    )
    strategy = ObservationStrategyFactory.from_config(obs_cfg, meta_cfg)

    empirical_obs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float32)
    empirical_times = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=torch.float32)
    empirical_mask = torch.tensor([[True, True, True, True, True]])

    obs, obs_time, obs_mask, rem_sim, rem_time, rem_mask = strategy.generate_empirical(
        empirical_obs=empirical_obs,
        empirical_times=empirical_times,
        empirical_mask=empirical_mask,
    )

    assert rem_sim is None and rem_time is None and rem_mask is None
    assert torch.allclose(obs[0], torch.tensor([3.0, 4.0, 5.0]))
    assert torch.allclose(obs_time[0], torch.tensor([0.5, 0.75, 1.0]))
    assert torch.equal(obs_mask, torch.ones_like(obs_mask, dtype=torch.bool))


@pytest.mark.parametrize("strategy_type", [None, "", "   ", "null", "none"])
def test_observation_factory_defaults_to_pk_peak_half_life(strategy_type):
    """``None``/empty observation types keep legacy PK strategy behaviour."""

    meta_cfg = MetaStudyConfig(time_num_steps=20)
    obs_cfg = ObservationsConfig(
        type=strategy_type,
        max_num_obs=8,
        add_rem=True,
        split_past_future=False,
    )
    strategy = ObservationStrategyFactory.from_config(obs_cfg, meta_cfg)
    assert isinstance(strategy, PKPeakHalfLifeStrategy)


def test_random_split_uses_fixed_capacity_contract():
    """Random split uses ``max_past`` and ``max_num_obs-max_past`` capacities."""

    meta_cfg = MetaStudyConfig(time_num_steps=40)
    obs_cfg = ObservationsConfig(
        type="random",
        max_num_obs=12,
        add_rem=True,
        split_past_future=True,
        min_past=0,
        max_past=5,
        generative_bias=True,
    )
    strategy = ObservationStrategyFactory.from_config(obs_cfg, meta_cfg)
    max_obs, max_rem = strategy.get_shapes()
    assert (max_obs, max_rem) == (5, 7)

    full_sim = torch.rand(3, 40)
    full_times = torch.linspace(0.0, 10.0, 40).repeat(3, 1)
    obs, obs_time, obs_mask, rem_obs, rem_time, rem_mask, _ = strategy.generate(
        full_simulation=full_sim,
        full_simulation_times=full_times,
    )

    assert obs.shape == (3, 5)
    assert obs_time.shape == (3, 5)
    assert obs_mask.shape == (3, 5)
    assert rem_obs is not None and rem_time is not None and rem_mask is not None
    assert rem_obs.shape == (3, 7)
    assert rem_time.shape == (3, 7)
    assert rem_mask.shape == (3, 7)


def test_random_split_truncates_remainder_to_fixed_capacity():
    """Boundary split never exceeds fixed remainder capacity ``M-K_max``."""

    meta_cfg = MetaStudyConfig(time_num_steps=8, time_stop=10.0)
    obs_cfg = ObservationsConfig(
        type="random",
        max_num_obs=8,
        add_rem=True,
        split_past_future=True,
        min_past=0,
        max_past=3,
        generative_bias=False,
        past_time_ratio=0.1,
    )
    strategy = ObservationStrategyFactory.from_config(obs_cfg, meta_cfg)

    # [B=1, M=8] with one past candidate (time <= 1.0) and seven future candidates.
    obs = torch.arange(1, 9, dtype=torch.float32).unsqueeze(0)
    obs_time = torch.tensor([[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]], dtype=torch.float32)
    obs_mask = torch.ones(1, 8, dtype=torch.bool)

    past_obs, past_time, past_mask, rem_obs, rem_time, rem_mask = strategy._split_by_boundary(
        obs, obs_time, obs_mask, generator=torch.Generator().manual_seed(0)
    )

    assert past_obs.shape == (1, 3)
    assert past_time.shape == (1, 3)
    assert past_mask.shape == (1, 3)
    assert rem_obs.shape == (1, 5)
    assert rem_time.shape == (1, 5)
    assert rem_mask.shape == (1, 5)
    assert int(rem_mask.sum().item()) == 5
    rem_valid_times = rem_time[0, rem_mask[0]]
    assert torch.all(rem_valid_times[:-1] <= rem_valid_times[1:])


def test_random_split_keeps_remainder_strictly_future_only():
    """With split enabled, remainder must never contain boundary-past points."""

    meta_cfg = MetaStudyConfig(time_num_steps=10, time_stop=10.0)
    obs_cfg = ObservationsConfig(
        type="random",
        max_num_obs=8,
        add_rem=True,
        split_past_future=True,
        min_past=3,
        max_past=3,
        generative_bias=False,
        past_time_ratio=0.3,  # boundary = 3.0
    )
    strategy = ObservationStrategyFactory.from_config(obs_cfg, meta_cfg)

    # Five past candidates (<= 3.0) and three future candidates (> 3.0).
    obs = torch.arange(1, 9, dtype=torch.float32).unsqueeze(0)
    obs_time = torch.tensor([[0.2, 0.8, 1.5, 2.2, 2.9, 3.5, 5.0, 8.0]], dtype=torch.float32)
    obs_mask = torch.ones(1, 8, dtype=torch.bool)

    past_obs, past_time, past_mask, rem_obs, rem_time, rem_mask = strategy._split_by_boundary(
        obs, obs_time, obs_mask, generator=torch.Generator().manual_seed(11)
    )

    assert past_obs.shape == (1, 3)
    assert past_time.shape == (1, 3)
    assert past_mask.shape == (1, 3)
    assert int(past_mask.sum().item()) == 3
    assert rem_obs.shape == (1, 5)
    assert rem_time.shape == (1, 5)
    assert rem_mask.shape == (1, 5)
    # Only three future points exist; remainder is padded after those.
    assert int(rem_mask.sum().item()) == 3
    rem_valid_times = rem_time[0, rem_mask[0]]
    assert torch.all(rem_valid_times > 3.0)
    assert torch.all(rem_valid_times[:-1] <= rem_valid_times[1:])


def test_random_split_when_no_past_selected_uses_full_domain_for_remainder():
    """If sampled past count is zero, remainder sampling ignores boundary split."""

    meta_cfg = MetaStudyConfig(time_num_steps=8, time_stop=10.0)
    obs_cfg = ObservationsConfig(
        type="random",
        max_num_obs=8,
        add_rem=True,
        split_past_future=True,
        min_past=0,
        max_past=3,
        generative_bias=False,
        past_time_ratio=0.3,  # boundary = 3.0
    )
    strategy = ObservationStrategyFactory.from_config(obs_cfg, meta_cfg)

    # One past point (<= 3.0), four future points, three invalid slots.
    # With seed=0 and k in {0,1}, sampled k is deterministically 0.
    obs = torch.arange(1, 9, dtype=torch.float32).unsqueeze(0)
    obs_time = torch.tensor([[2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]], dtype=torch.float32)
    obs_mask = torch.tensor([[True, True, True, True, True, False, False, False]])

    past_obs, past_time, past_mask, rem_obs, rem_time, rem_mask = strategy._split_by_boundary(
        obs, obs_time, obs_mask, generator=torch.Generator().manual_seed(0)
    )

    assert past_obs.shape == (1, 3)
    assert past_time.shape == (1, 3)
    assert past_mask.shape == (1, 3)
    assert int(past_mask.sum().item()) == 0

    assert rem_obs.shape == (1, 5)
    assert rem_time.shape == (1, 5)
    assert rem_mask.shape == (1, 5)
    assert int(rem_mask.sum().item()) == 5

    rem_valid_times = rem_time[0, rem_mask[0]]
    # Full-domain fallback allows past values inside remainder.
    assert torch.any(rem_valid_times <= 3.0)
    assert torch.all(rem_valid_times[:-1] <= rem_valid_times[1:])


if __name__ == "__main__":
    test_strategy_from_yaml_config()
