"""Forward pass tests for encoders, decoders, and AICMEPK."""

from dataclasses import replace

import torch
from torchtyping import TensorType

from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.data.datasets.aicme_datasets import (
    AICMECompartmentsDataModule,
)
from pff.models.architectures.decoders_pk import (
    ODEDecoder,
    RNNDecoder,
    TransformerDecoder,
    TransformerDecoderZiQuery,
)
from pff.models.architectures.encoders_pk import (
    RNNContextEncoder,
    RNNContextEncoderDosing,
    TimeObsSeparateEncoder,
    TransformerContextEncoder,
)

# Reuse patched components and helpers from existing model tests
from tests.models.test_aicme_pk import _aicme_config, _first_batch_list


def _tiny_config() -> NodePKExperimentConfig:
    """Create a minimal configuration with small dimensions for fast tests."""
    cfg = NodePKExperimentConfig()
    net = cfg.network
    cfg.network = replace(
        net,
        use_self_attention=True,
        time_obs_encoder_hidden_dim=4,
        time_obs_encoder_output_dim=2,
        encoder_rnn_hidden_dim=4,
        zi_latent_dim=3,
        rnn_individual_encoder_number_of_layers=1,
        individual_encoder_number_of_heads=1,
        use_attention=True,
        decoder_hidden_dim=4,
        decoder_rnn_hidden_dim=4,
        rnn_decoder_number_of_layers=1,
        decoder_num_layers=1,
        decoder_attention_layers=1,
        aggregator_num_heads=1,
        cov_proj_dim=2,
    )
    # Attribute required by ODEDecoder
    cfg.network.rnn_hidden_dim = cfg.network.decoder_rnn_hidden_dim
    return cfg


def test_time_obs_separate_encoder_forward() -> None:
    """Ensure TimeObsSeparateEncoder produces the expected output shape."""
    encoder = TimeObsSeparateEncoder(time_dim=1, obs_dim=1, hidden_dim=4, output_dim=2)
    B, T = 2, 3
    scaled_time: TensorType["B", "T", 1] = torch.randn(B, T, 1)  # [B,T,1]
    observation: TensorType["B", "T", 1] = torch.randn(B, T, 1)  # [B,T,1]
    out = encoder(scaled_time, observation)
    assert out.shape == (B, T, 4)  # 2 * output_dim


def test_rnn_context_encoder_forward() -> None:
    """RNNContextEncoder should return [B,I,zi_latent_dim]."""
    cfg = _tiny_config()
    encoder = RNNContextEncoder(cfg)
    B, I, N = 2, 3, 4
    obs: TensorType["B", "I", "N", 1] = torch.randn(B, I, N, 1)  # [B,I,N,1]
    times: TensorType["B", "I", "N", 1] = torch.randn(B, I, N, 1)  # [B,I,N,1]
    mask: TensorType["B", "I", "N"] = torch.ones(B, I, N)  # [B,I,N]
    dosing_amounts = torch.rand(B, I)
    dosing_routes = torch.zeros(B, I, dtype=torch.long)
    out = encoder(obs, times, mask, dosing_amounts, dosing_routes)
    assert out.shape == (B, I, cfg.network.zi_latent_dim)


def test_transformer_context_encoder_forward() -> None:
    """TransformerContextEncoder forward pass shape check."""
    cfg = _tiny_config()
    cfg.network = replace(cfg.network, individual_encoder_name="TransformerContextEncoder")
    encoder = TransformerContextEncoder(cfg)
    B, I, N = 2, 3, 4
    obs: TensorType["B", "I", "N", 1] = torch.randn(B, I, N, 1)  # [B,I,N,1]
    times: TensorType["B", "I", "N", 1] = torch.randn(B, I, N, 1)  # [B,I,N,1]
    mask: TensorType["B", "I", "N"] = torch.ones(B, I, N)  # [B,I,N]
    dosing_amounts = torch.rand(B, I)
    dosing_routes = torch.zeros(B, I, dtype=torch.long)
    out = encoder(obs, times, mask, dosing_amounts, dosing_routes)
    assert out.shape == (B, I, cfg.network.zi_latent_dim)


def test_rnn_context_encoder_dosing_forward() -> None:
    """RNNContextEncoderDosing should include dosing information."""
    cfg = _tiny_config()
    cfg.network = replace(cfg.network, individual_encoder_name="RNNContextEncoderDosing")
    encoder = RNNContextEncoderDosing(cfg)
    B, I, N = 2, 3, 4
    obs: TensorType["B", "I", "N", 1] = torch.randn(B, I, N, 1)
    times: TensorType["B", "I", "N", 1] = torch.randn(B, I, N, 1)
    mask: TensorType["B", "I", "N"] = torch.ones(B, I, N)
    dosing_amounts = torch.rand(B, I)
    dosing_routes = torch.randint(0, 2, (B, I))
    out = encoder(obs, times, mask, dosing_amounts, dosing_routes)
    assert out.shape == (B, I, cfg.network.zi_latent_dim)


def test_rnn_decoder_forward() -> None:
    """RNNDecoder returns sequences with expected shapes."""
    cfg = _tiny_config()
    decoder = RNNDecoder(cfg)
    BI, T = 2, 5
    B, I = 1, BI
    init: TensorType["BI", 1, 1] = torch.randn(BI, 1, 1)
    decode_t: TensorType["BI", "T", 1] = torch.linspace(0, 1, T).view(1, T, 1).repeat(BI, 1, 1)
    dose: TensorType["BI", 1, 1] = torch.rand(BI, 1, 1)
    route: TensorType["BI", 1, 1] = torch.randint(0, 3, (BI, 1, 1), dtype=torch.float32)
    first_t_s: TensorType["BI", 1, 1] = torch.zeros(BI, 1, 1)
    z_s: TensorType["B", "Z"] = torch.randn(B, cfg.network.zi_latent_dim)
    z_i: TensorType["B", "I", "Z"] = torch.randn(B, I, cfg.network.zi_latent_dim)
    mean, logvar, H, t_out, h = decoder(
        init,
        decode_t,
        z_s,
        z_i,
        dose=dose,
        route=route,
        first_t_s=first_t_s,
    )
    assert mean.shape == (BI, T, 1)
    assert logvar.shape == (BI, T, 1)
    assert t_out.shape == (BI, T, 1)
    assert h.shape == (BI, T, cfg.network.decoder_rnn_hidden_dim)
    assert H is None


def test_transformer_decoder_forward() -> None:
    """TransformerDecoder forward pass returns correct shapes."""
    cfg = _tiny_config()
    cfg.network = replace(cfg.network, decoder_name="TransformerDecoder")
    decoder = TransformerDecoder(cfg)
    BI, T = 2, 5
    B, I = 1, BI
    init: TensorType["BI", 1, 1] = torch.randn(BI, 1, 1)
    decode_t: TensorType["BI", "T", 1] = torch.linspace(0, 1, T).view(1, T, 1).repeat(BI, 1, 1)
    dose: TensorType["BI", 1, 1] = torch.rand(BI, 1, 1)
    route: TensorType["BI", 1, 1] = torch.randint(0, 3, (BI, 1, 1), dtype=torch.float32)
    first_t_s: TensorType["BI", 1, 1] = torch.zeros(BI, 1, 1)
    z_s: TensorType["B", "Z"] = torch.randn(B, cfg.network.zi_latent_dim)
    z_i: TensorType["B", "I", "Z"] = torch.randn(B, I, cfg.network.zi_latent_dim)
    mean, logvar, H, t_out, h = decoder(
        init,
        decode_t,
        z_s,
        z_i,
        dose=dose,
        route=route,
        first_t_s=first_t_s,
    )
    assert mean.shape == (BI, T, 1)
    assert logvar.shape == (BI, T, 1)
    assert t_out.shape == (BI, T, 1)
    assert h.shape == (BI, T, cfg.network.decoder_hidden_dim)
    assert H is None


def test_transformer_decoder_ziquery_forward() -> None:
    """TransformerDecoderZiQuery forward pass returns correct shapes."""
    cfg = _tiny_config()
    cfg.network = replace(cfg.network, decoder_name="TransformerDecoderZiQuery")
    decoder = TransformerDecoderZiQuery(cfg)

    BI, T = 2, 5
    B, I = 1, BI  # so that B*I == BI

    init: TensorType["BI", 1, 1] = torch.randn(BI, 1, 1)
    decode_t: TensorType["BI", "T", 1] = torch.linspace(0, 1, T).view(1, T, 1).repeat(BI, 1, 1)
    dose: TensorType["BI", 1, 1] = torch.rand(BI, 1, 1)
    route: TensorType["BI", 1, 1] = torch.randint(
        0, len(cfg.dosing.route_options), (BI, 1, 1), dtype=torch.float32
    )
    first_t_s: TensorType["BI", 1, 1] = torch.zeros(BI, 1, 1)

    z_s: TensorType["B", "Z"] = torch.randn(B, cfg.network.zi_latent_dim)
    z_i: TensorType["B", "I", "Z"] = torch.randn(B, I, cfg.network.zi_latent_dim)

    mean, logvar, H, t_out, h = decoder(
        init,
        decode_t,
        z_s,
        z_i,
        dose=dose,
        route=route,
        first_t_s=first_t_s,
    )

    assert mean.shape == (BI, T, 1)
    assert logvar.shape == (BI, T, 1)
    assert t_out.shape == (BI, T, 1)
    assert h.shape == (BI, T, cfg.network.decoder_hidden_dim)
    assert H is None


def test_ode_decoder_forward() -> None:
    """ODEDecoder forward pass shape check."""
    cfg = _tiny_config()
    cfg.network = replace(cfg.network, decoder_name="ODEDecoder")
    decoder = ODEDecoder(cfg)
    BI, T = 2, 5
    B, I = 1, BI
    init: TensorType["BI", 1, 1] = torch.randn(BI, 1, 1)
    decode_t: TensorType["BI", "T", 1] = torch.linspace(0, 1, T).view(1, T, 1).repeat(BI, 1, 1)
    dose: TensorType["BI", 1, 1] = torch.rand(BI, 1, 1)
    route: TensorType["BI", 1, 1] = torch.randint(0, 3, (BI, 1, 1), dtype=torch.float32)
    first_t_s: TensorType["BI", 1, 1] = torch.zeros(BI, 1, 1)
    z_s: TensorType["B", "Z"] = torch.randn(B, cfg.network.zi_latent_dim)
    z_i: TensorType["B", "I", "Z"] = torch.randn(B, I, cfg.network.zi_latent_dim)
    mean, logvar, t_out, h, _ = decoder(
        init,
        decode_t,
        z_s,
        z_i,
        dose=dose,
        route=route,
        first_t_s=first_t_s,
    )
    assert mean.shape == (BI, T, 1)
    assert logvar.shape == (BI, T, 1)
    assert t_out.shape == (BI, T, 1)
    assert h.shape == (BI, T, cfg.network.decoder_rnn_hidden_dim)
