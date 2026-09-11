import pytest
from types import SimpleNamespace

torch = pytest.importorskip("torch")

from pff.models.utils.scalers import PKScaler
from pff.models.utils.scaler_selection import resolve_scaler_methods


def test_pk_scaler_forward_inverse_and_dosing_zscore():
    scaler = PKScaler(value_method="zscore", time_method="max")

    context_obs = torch.tensor(
        [
            [[[1.0], [2.0], [0.0]]],
            [[[2.0], [4.0], [6.0]]],
        ],
        dtype=torch.float32,
    )
    context_times = torch.tensor(
        [
            [[[0.5], [1.0], [0.0]]],
            [[[0.25], [0.5], [0.75]]],
        ],
        dtype=torch.float32,
    )
    mask = context_obs.squeeze(-1) > 0

    stats = scaler.stats(context_obs, context_times, mask)

    y_norm, t_norm = scaler.forward(context_obs, context_times, stats)
    y_recovered, t_recovered = scaler.inverse(y_norm, t_norm, stats)

    assert torch.allclose(y_recovered, context_obs)
    assert torch.allclose(t_recovered, context_times)

    dosing = torch.tensor([[3.0], [8.0]], dtype=torch.float32)
    scaled_dosing = scaler.scale_dosing_amounts(dosing, stats)

    expected_mu = stats["v_mu"]
    expected_sigma = stats["v_sigma"]
    expected_scaled = (dosing - expected_mu) / expected_sigma

    assert torch.allclose(scaled_dosing, expected_scaled)


def test_pk_scaler_scale_dosing_amounts_max():
    scaler = PKScaler(value_method="max", time_method="none")

    context_obs = torch.tensor(
        [[[[2.0], [4.0]]]],
        dtype=torch.float32,
    )
    times = torch.zeros_like(context_obs)
    mask = torch.tensor([[[True, True]]])

    stats = scaler.stats(context_obs, times, mask)

    dosing = torch.tensor([[2.0, 4.0]], dtype=torch.float32)
    scaled = scaler.scale_dosing_amounts(dosing, stats)

    expected = dosing / stats["v_sigma"]
    assert torch.allclose(scaled, expected)


def test_pk_scaler_scale_dosing_amounts_1d_preserves_shape():
    """Scaling 1D dosing [B] must not broadcast to [B, B]."""

    scaler = PKScaler(value_method="max", time_method="none")

    context_obs = torch.tensor(
        [
            [[[2.0], [4.0]]],
            [[[3.0], [6.0]]],
        ],
        dtype=torch.float32,
    )
    times = torch.zeros_like(context_obs)
    mask = torch.tensor(
        [
            [[True, True]],
            [[True, True]],
        ]
    )
    stats = scaler.stats(context_obs, times, mask)  # v_sigma: [B, 1]

    dosing = torch.tensor([2.0, 6.0], dtype=torch.float32)  # [B]
    scaled = scaler.scale_dosing_amounts(dosing, stats)

    assert scaled.shape == dosing.shape
    expected = dosing / stats["v_sigma"].squeeze(-1)
    assert torch.allclose(scaled, expected)


def test_pk_scaler_log_and_max_forward_inverse_roundtrip():
    scaler = PKScaler(value_method="log_and_max", time_method="max")

    context_obs = torch.tensor(
        [
            [[[1.0], [2.0], [0.0]]],
            [[[0.5], [4.0], [8.0]]],
        ],
        dtype=torch.float32,
    )
    context_times = torch.tensor(
        [
            [[[0.1], [0.2], [0.0]]],
            [[[0.25], [0.5], [1.0]]],
        ],
        dtype=torch.float32,
    )
    mask = context_obs.squeeze(-1) > 0
    stats = scaler.stats(context_obs, context_times, mask)

    y_scaled, t_scaled = scaler.forward(context_obs, context_times, stats)
    y_back, t_back = scaler.inverse(y_scaled, t_scaled, stats)

    assert torch.allclose(y_back, context_obs, atol=1e-6, rtol=1e-6)
    assert torch.allclose(t_back, context_times)


def test_pk_scaler_log_and_max_stats_use_logged_values():
    scaler = PKScaler(value_method="log_and_max", time_method="none")
    obs = torch.tensor([[[[1e-4], [1.0], [10.0]]]], dtype=torch.float32)
    times = torch.zeros_like(obs)
    mask = torch.tensor([[[True, True, True]]])

    stats = scaler.stats(obs, times, mask)
    expected_sigma = torch.log(torch.tensor(10.0) + 1e-8).view(1, 1)
    assert torch.allclose(stats["v_sigma"], expected_sigma)


def test_pk_scaler_log_and_max_eps_stability_for_small_values():
    scaler = PKScaler(value_method="log_and_max", time_method="none")
    obs = torch.tensor([[[[1e-12], [1e-10]]]], dtype=torch.float32)
    times = torch.zeros_like(obs)
    mask = torch.tensor([[[True, True]]])
    stats = scaler.stats(obs, times, mask)

    y_scaled, _ = scaler.forward(obs, times, stats)
    assert torch.isfinite(y_scaled).all()


def test_pk_scaler_log_and_max_all_values_below_one_uses_logged_max():
    """`log_and_max` must use max(log(y+eps)) directly."""
    scaler = PKScaler(value_method="log_and_max", time_method="none")
    obs = torch.tensor([[[[1e-4], [2e-1]]]], dtype=torch.float32)
    times = torch.zeros_like(obs)
    mask = torch.tensor([[[True, True]]])

    stats = scaler.stats(obs, times, mask)
    # Expected: max value after log transform.
    expected_sigma = torch.max(torch.log(obs[0, 0, :, 0] + 1e-8)).view(1, 1)
    assert torch.allclose(stats["v_sigma"], expected_sigma, atol=1e-6, rtol=1e-6)


def test_pk_scaler_log_and_z_forward_inverse_roundtrip():
    scaler = PKScaler(value_method="log_and_z", time_method="max")

    obs = torch.tensor(
        [
            [[[0.1], [1.0], [5.0]]],
            [[[1e-6], [0.2], [2.5]]],
        ],
        dtype=torch.float32,
    )
    times = torch.tensor(
        [
            [[[0.1], [0.2], [0.5]]],
            [[[0.2], [0.4], [0.8]]],
        ],
        dtype=torch.float32,
    )
    mask = torch.ones(obs.shape[:-1], dtype=torch.bool)

    stats = scaler.stats(obs, times, mask)
    y_scaled, t_scaled = scaler.forward(obs, times, stats)
    y_back, t_back = scaler.inverse(y_scaled, t_scaled, stats)

    assert torch.allclose(y_back, obs, atol=1e-6, rtol=1e-6)
    assert torch.allclose(t_back, times, atol=1e-6, rtol=1e-6)


def test_pk_scaler_log_and_z_stats_use_logged_values():
    scaler = PKScaler(value_method="log_and_z", time_method="none")
    obs = torch.tensor([[[[0.1], [1.0], [10.0]]]], dtype=torch.float32)
    times = torch.zeros_like(obs)
    mask = torch.tensor([[[True, True, True]]])

    stats = scaler.stats(obs, times, mask)
    expected_log = torch.log(obs[0, 0, :, 0] + 1e-8)

    assert torch.allclose(stats["v_mu"], expected_log.mean().view(1, 1), atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        stats["v_sigma"],
        expected_log.std(unbiased=False).view(1, 1),
        atol=1e-6,
        rtol=1e-6,
    )


def test_pk_scaler_log_forward_inverse_roundtrip():
    scaler = PKScaler(value_method="log", time_method="none")

    obs = torch.tensor(
        [
            [[[0.1], [1.0], [5.0]]],
            [[[1e-6], [0.2], [2.5]]],
        ],
        dtype=torch.float32,
    )
    times = torch.zeros_like(obs)
    mask = torch.ones(obs.shape[:-1], dtype=torch.bool)

    stats = scaler.stats(obs, times, mask)
    assert "v_sigma" not in stats
    assert "v_mu" not in stats

    y_scaled, _ = scaler.forward(obs, times, stats)
    y_back, _ = scaler.inverse(y_scaled, times, stats)

    assert torch.allclose(y_back, obs, atol=1e-6, rtol=1e-6)


def test_pk_scaler_log_scale_dosing_amounts():
    scaler = PKScaler(value_method="log", time_method="none")
    dosing = torch.tensor([[0.1, 1.0, 10.0]], dtype=torch.float32)
    scaled = scaler.scale_dosing_amounts(dosing, stats={})
    expected = torch.log(dosing + 1e-8)
    assert torch.allclose(scaled, expected)


def test_pk_scaler_log_and_z_scale_dosing_amounts():
    scaler = PKScaler(value_method="log_and_z", time_method="none")

    obs = torch.tensor([[[[0.2], [1.0], [3.0]]]], dtype=torch.float32)
    times = torch.zeros_like(obs)
    mask = torch.tensor([[[True, True, True]]])
    stats = scaler.stats(obs, times, mask)

    dosing = torch.tensor([[0.5, 2.0]], dtype=torch.float32)
    scaled = scaler.scale_dosing_amounts(dosing, stats)
    expected = (torch.log(dosing + 1e-8) - stats["v_mu"]) / stats["v_sigma"]

    assert torch.allclose(scaled, expected, atol=1e-6, rtol=1e-6)


def test_resolve_scaler_methods_log_and_z_maps_to_log_and_z():
    cfg = SimpleNamespace(
        log_and_z=True,
        log_and_max=True,
        log_transform=True,
        z_score_normalization=True,
        normalize_by_max=True,
        normalize_time=True,
    )
    value_method, time_method = resolve_scaler_methods(cfg)
    assert value_method == "log_and_z"
    assert time_method == "max"


def test_resolve_scaler_methods_log_transform_maps_to_log():
    cfg = SimpleNamespace(
        log_and_z=False,
        log_and_max=False,
        log_transform=True,
        z_score_normalization=False,
        normalize_by_max=True,
        normalize_time=False,
    )
    value_method, time_method = resolve_scaler_methods(cfg)
    assert value_method == "log"
    assert time_method == "none"
