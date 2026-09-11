from typing import Annotated, TypeAlias

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from pff.config_classes.node_pk_config import NodePKExperimentConfig

# ────────────────────────────────────────────────────────────────────────────────
#  RNNContextEncoder
# ────────────────────────────────────────────────────────────────────────────────

# Enable runtime shape checking for @typechecked decorated methods
patch_typeguard()


class TimeObsSeparateEncoder(nn.Module):
    def __init__(self, time_dim=1, obs_dim=1, hidden_dim=32, output_dim=16):
        super().__init__()
        self.time_encoder = nn.Sequential(
            nn.Linear(time_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim), nn.ReLU()
        )
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim), nn.ReLU()
        )
        self.layernorm = nn.LayerNorm(2 * output_dim)  #  LayerNorm added after concat

        for m in list(self.time_encoder) + list(self.obs_encoder):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, scaled_time: TensorType["B", "T", 1], observation: TensorType["B", "T", 1]
    ) -> TensorType["B", "T", "2*output_dim"]:
        time_emb = self.time_encoder(scaled_time)  # [B, T, output_dim]
        obs_emb = self.obs_encoder(observation)  # [B, T, output_dim]
        out = torch.cat([time_emb, obs_emb], dim=-1)  # [B, T, 2*output_dim]
        out = self.layernorm(out)  #  stabilize outputs
        return out


class RNNContextEncoder(nn.Module):
    """
    Robust RNN-based context encoder that never produces NaNs, even when some
    individuals have no observations. Drop-in replacement for the original
    implementation.

    Parameters
    ----------
    neural_config:
        Complete :class:`~pff.config_classes.node_pk_config.NodePKConfig`
        configuration. Only the ``network`` section is consumed here.
    """

    def __init__(self, neural_config: "NodePKExperimentConfig"):
        super().__init__()
        self.rnn_hidden_dim = neural_config.network.encoder_rnn_hidden_dim
        self.zi_latent_dim = neural_config.network.zi_latent_dim
        self.use_attention = neural_config.network.use_attention
        self.nhead = neural_config.network.individual_encoder_number_of_heads

        # NEW: config flag for Δt encoding
        self.use_time_deltas = getattr(neural_config.network, "use_time_deltas", True)

        # (t, x) → features
        self.input_encoder = TimeObsSeparateEncoder(
            time_dim=1,
            obs_dim=1,
            hidden_dim=neural_config.network.time_obs_encoder_hidden_dim,
            output_dim=neural_config.network.time_obs_encoder_output_dim,
        )

        if self.use_time_deltas:
            self.delta_t_encoder = nn.Linear(1, neural_config.network.time_obs_encoder_output_dim)
            # x_enc = 2D, delta_enc = D, raw_time = D  → total = 4D
            in_dim = 4 * neural_config.network.time_obs_encoder_output_dim
        else:
            # x_enc = 2D
            in_dim = 2 * neural_config.network.time_obs_encoder_output_dim

        self.rnn = nn.GRU(
            input_size=in_dim,
            hidden_size=self.rnn_hidden_dim,
            batch_first=True,
            num_layers=neural_config.network.rnn_individual_encoder_number_of_layers,
        )

        if self.use_attention:
            self.attn = nn.MultiheadAttention(
                embed_dim=self.rnn_hidden_dim,
                num_heads=self.nhead,
                batch_first=True,
            )
            self.query_proj = nn.Parameter(torch.randn(1, 1, self.rnn_hidden_dim))

        self.proj = nn.Linear(self.rnn_hidden_dim, self.zi_latent_dim)
        nn.init.xavier_uniform_(self.proj.weight)

        # Good GRU init
        for n, p in self.rnn.named_parameters():
            if "weight_ih" in n:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in n:
                nn.init.orthogonal_(p)
            elif "bias" in n:
                nn.init.zeros_(p)

    # ───────────────────────────────────────────────────────────────────── forward
    @typechecked
    def forward(
        self,
        context_obs: Annotated[Tensor, TensorType["B", "I", "N_obs", 1]],
        context_obs_time: Annotated[Tensor, TensorType["B", "I", "N_obs", 1]],
        context_obs_mask: Annotated[Tensor, TensorType["B", "I", "N_obs"]],
        context_dosing_amounts: Annotated[Tensor, TensorType["B", "I"]],
        context_dosing_route_types: Annotated[Tensor, TensorType["B", "I"]],
        mask_individuals: Annotated[Tensor, TensorType["B", "I"]] | None = None,
    ) -> Annotated[Tensor, TensorType["B", "I", "zp_latent_dim"]]:
        B, I, N_obs, _ = context_obs.shape
        _ = context_dosing_amounts  # API compatibility
        _ = context_dosing_route_types

        # ── 1. Encode per-time-step inputs ────────────────────────────────
        x_time = context_obs_time.view(B * I, N_obs, 1)  # [B*I, N_obs, 1]
        x_obs = context_obs.view(B * I, N_obs, 1)  # [B*I, N_obs, 1]
        x_enc = self.input_encoder(x_time, x_obs)  # [B*I, N_obs, D]

        if self.use_time_deltas:
            # compute Δt, with Δt[:,0] = 0
            delta_t = torch.zeros_like(x_time)  # [B*I, N_obs, 1]
            delta_t[:, 1:, :] = x_time[:, 1:, :] - x_time[:, :-1, :]
            delta_enc = self.delta_t_encoder(delta_t)  # [B*I, N_obs, D]

            # feats = enc(t,x) ⊕ enc(Δt) ⊕ raw_time
            feats = torch.cat(
                [x_enc, delta_enc, x_time.expand_as(delta_enc)], dim=-1
            )  # [B*I, N_obs, 3D]
        else:
            feats = x_enc  # [B*I, N_obs, 2D]

        mask = context_obs_mask.view(B * I, N_obs).bool()  # [B*I, N_obs]
        raw_len = mask.sum(-1)  # [B*I], num valid obs
        keep_mask = raw_len > 0  # [B*I]

        # ── 2. GRU over *non-empty* rows ──────────────────────────────────
        if keep_mask.any():
            packed = pack_padded_sequence(
                feats[keep_mask],  # [n_keep, N_obs, in_dim]
                raw_len[keep_mask].cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.rnn(packed)
            rnn_out, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=N_obs
            )  # [n_keep, N_obs, H]
        else:
            rnn_out = feats.new_zeros(0, N_obs, self.rnn_hidden_dim)

        # ── 3. Build final hidden state h ─────────────────────────────────
        h = feats.new_zeros(B * I, self.rnn_hidden_dim)  # [B*I, H]

        if keep_mask.any():
            if self.use_attention:
                Q = self.query_proj.expand(keep_mask.sum(), -1, -1)  # [n_keep, 1, H]
                attn_out, _ = self.attn(
                    query=Q,  # [n_keep, 1, H]
                    key=rnn_out,  # [n_keep, N_obs, H]
                    value=rnn_out,  # [n_keep, N_obs, H]
                    key_padding_mask=~mask[keep_mask],  # [n_keep, N_obs]
                )
                h[keep_mask] = attn_out.squeeze(1)  # [n_keep, H]
            else:
                idx_last = raw_len[keep_mask] - 1
                h[keep_mask] = rnn_out[
                    torch.arange(rnn_out.size(0), device=feats.device),
                    idx_last,  # pick last valid step
                    :,
                ]  # [n_keep, H]

        if mask_individuals is not None:
            h = h * mask_individuals.view(B * I, 1).to(h)  # [B*I, H]

        # ── 4. Project + ℓ2-normalise ─────────────────────────────────────
        z = self.proj(h)  # [B*I, zp_latent_dim]
        z = z / torch.clamp(z.norm(dim=-1, keepdim=True), min=1e-6)
        z = z.view(B, I, self.zi_latent_dim)  # [B, I, zp_latent_dim]

        return z


class TransformerContextEncoder(nn.Module):
    """Processes context sequences using Transformer encoder layers."""

    def __init__(self, neural_config: NodePKExperimentConfig) -> None:
        super().__init__()
        self.model_dim = neural_config.network.encoder_rnn_hidden_dim
        self.zi_latent_dim = neural_config.network.zi_latent_dim
        self.use_attention = neural_config.network.use_attention
        self.nhead = neural_config.network.individual_encoder_number_of_heads
        self.num_layers = neural_config.network.rnn_individual_encoder_number_of_layers

        # Time/observation embedding
        self.input_encoder = TimeObsSeparateEncoder(
            time_dim=1,
            obs_dim=1,
            hidden_dim=neural_config.network.time_obs_encoder_hidden_dim,
            output_dim=neural_config.network.time_obs_encoder_output_dim,
        )

        input_dim = 2 * neural_config.network.time_obs_encoder_output_dim
        self.input_proj = nn.Linear(input_dim, self.model_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.nhead,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers)

        if self.use_attention:
            self.attn = nn.MultiheadAttention(
                embed_dim=self.model_dim,
                num_heads=self.nhead,
                batch_first=True,
            )
            self.query_proj = nn.Parameter(torch.randn(1, 1, self.model_dim))

        self.proj = nn.Linear(self.model_dim, self.zi_latent_dim)
        nn.init.xavier_uniform_(self.proj.weight)

    @typechecked
    def forward(
        self,
        context_obs: Annotated[Tensor, TensorType["B", "I", "N_obs", 1]],
        context_obs_time: Annotated[Tensor, TensorType["B", "I", "N_obs", 1]],
        context_obs_mask: Annotated[Tensor, TensorType["B", "I", "N_obs"]],
        context_dosing_amounts: Annotated[Tensor, TensorType["B", "I"]],
        context_dosing_route_types: Annotated[Tensor, TensorType["B", "I"]],
        mask_individuals: Annotated[Tensor, TensorType["B", "I"]] | None = None,
    ) -> Annotated[Tensor, TensorType["B", "I", "zp_latent_dim"]]:
        B, I, N_obs, _ = context_obs.shape
        _ = context_dosing_amounts  # Interface compatibility placeholder
        _ = context_dosing_route_types

        x_time = context_obs_time.view(B * I, N_obs, 1)
        x_obs = context_obs.view(B * I, N_obs, 1)
        x_encoded = self.input_encoder(x_time, x_obs)
        x_encoded = self.input_proj(x_encoded)

        mask = context_obs_mask.view(B * I, N_obs)
        if mask.dtype != torch.bool:
            mask = mask > 0
        lengths = mask.sum(dim=-1).long()

        src_key_padding_mask = ~mask
        zero_len = lengths == 0
        if zero_len.any():
            src_key_padding_mask[zero_len, 0] = False
            x_encoded[zero_len, 0] = 0.0
            lengths[zero_len] = 1

        trans_out = self.transformer(x_encoded, src_key_padding_mask=src_key_padding_mask)

        if self.use_attention:
            Q = self.query_proj.expand(B * I, -1, -1)
            attn_out, _ = self.attn(
                query=Q,
                key=trans_out,
                value=trans_out,
                key_padding_mask=src_key_padding_mask,
            )
            h = attn_out.squeeze(1)
        else:
            h = trans_out[range(trans_out.size(0)), lengths - 1, :]

        if zero_len.any():
            h[zero_len] = 0.0

        z = self.proj(h)
        z = z / torch.clamp(z.norm(dim=-1, keepdim=True), min=1e-6)
        z = z.view(B, I, self.zi_latent_dim)
        if mask_individuals is not None:
            z = z * mask_individuals.unsqueeze(-1).to(z)
        return z


class RNNContextEncoderDosing(RNNContextEncoder):
    """
    Extension of RNNContextEncoder that also encodes dosing information
    (amount + route type) as an additional "pseudo time step" appended
    to the RNN outputs. Optionally applies a self-attention block on
    the augmented sequence before summarisation.
    """

    def __init__(self, neural_config: "NodePKExperimentConfig"):
        super().__init__(neural_config)

        # Continuous dosing amount → embedding
        self.dose_amount_encoder = nn.Linear(1, self.rnn_hidden_dim)

        # Discrete dosing route type → embedding (use MetaDosingConfig)
        n_routes = len(neural_config.dosing.route_options)
        self.dose_type_embedding = nn.Embedding(
            num_embeddings=n_routes,
            embedding_dim=self.rnn_hidden_dim,
        )

        # Combine [amount_emb, type_emb] → H
        self.dose_proj = nn.Linear(2 * self.rnn_hidden_dim, self.rnn_hidden_dim)

        # Optional self-attention block (default: disabled)
        self.use_self_attention = getattr(neural_config.network, "use_self_attention", False)
        if self.use_self_attention:
            self.self_attn = nn.MultiheadAttention(
                embed_dim=self.rnn_hidden_dim,
                num_heads=self.nhead,
                batch_first=True,
            )

    # ───────────────────────────────────────────────────────────────────── forward
    @typechecked
    def forward(
        self,
        context_obs: Annotated[Tensor, TensorType["B", "I", "N_obs", 1]],
        context_obs_time: Annotated[Tensor, TensorType["B", "I", "N_obs", 1]],
        context_obs_mask: Annotated[Tensor, TensorType["B", "I", "N_obs"]],
        context_dosing_amounts: Annotated[Tensor, TensorType["B", "I"]],
        context_dosing_route_types: Annotated[Tensor, TensorType["B", "I"]],
        mask_individuals: Annotated[Tensor, TensorType["B", "I"]] | None = None,
    ) -> Annotated[Tensor, TensorType["B", "I", "zp_latent_dim"]]:
        B, I, N_obs, _ = context_obs.shape

        # ── Step 1: Encode obs sequence (same as parent) ──────────────────
        x_time = context_obs_time.view(B * I, N_obs, 1)  # [B*I, N_obs, 1]
        x_obs = context_obs.view(B * I, N_obs, 1)  # [B*I, N_obs, 1]
        x_enc = self.input_encoder(x_time, x_obs)  # [B*I, N_obs, D]

        if self.use_time_deltas:
            delta_t = torch.zeros_like(x_time)
            delta_t[:, 1:, :] = x_time[:, 1:, :] - x_time[:, :-1, :]
            delta_enc = self.delta_t_encoder(delta_t)  # [B*I, N_obs, D]
            feats = torch.cat([x_enc, delta_enc, x_time.expand_as(delta_enc)], dim=-1)
        else:
            feats = x_enc  # [B*I, N_obs, 2D]

        mask = context_obs_mask.view(B * I, N_obs).bool()
        raw_len = mask.sum(-1)  # [B*I]
        keep_mask = raw_len > 0  # [B*I]

        if keep_mask.any():
            packed = pack_padded_sequence(
                feats[keep_mask],
                raw_len[keep_mask].cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.rnn(packed)
            rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=N_obs)
        else:
            rnn_out = feats.new_zeros(0, N_obs, self.rnn_hidden_dim)

        # ── Step 2: Dosing as an extra time step ──────────────────────────
        dose_amt = context_dosing_amounts.view(B * I, 1)  # [B*I, 1]
        dose_type = context_dosing_route_types.view(B * I)  # [B*I]

        amt_emb = self.dose_amount_encoder(dose_amt)  # [B*I, H]
        type_emb = self.dose_type_embedding(dose_type)  # [B*I, H]
        dose_feat = self.dose_proj(torch.cat([amt_emb, type_emb], dim=-1))  # [B*I, H]

        if keep_mask.any():
            dose_feat_keep = dose_feat[keep_mask].unsqueeze(1)  # [n_keep, 1, H]
            rnn_out = torch.cat([rnn_out, dose_feat_keep], dim=1)  # [n_keep, N_obs+1, H]

        # ── Step 3: Optional self-attention ───────────────────────────────
        if self.use_self_attention and keep_mask.any():
            rnn_out, _ = self.self_attn(
                query=rnn_out,
                key=rnn_out,
                value=rnn_out,
                key_padding_mask=None,  # dosing step always valid
            )  # [n_keep, N_obs+1, H]

        # ── Step 4: Summary (attention or last step) ──────────────────────
        h = feats.new_zeros(B * I, self.rnn_hidden_dim)

        if keep_mask.any():
            if self.use_attention:
                Q = self.query_proj.expand(keep_mask.sum(), -1, -1)
                attn_out, _ = self.attn(
                    query=Q,
                    key=rnn_out,
                    value=rnn_out,
                )
                h[keep_mask] = attn_out.squeeze(1)  # [n_keep, H]
            else:
                # last step = dosing step if appended
                h[keep_mask] = rnn_out[:, -1, :]

        if mask_individuals is not None:
            h = h * mask_individuals.view(B * I, 1).to(h)

        # ── Step 5: Project + normalise ──────────────────────────────────
        z = self.proj(h)  # [B*I, zp_latent_dim]
        z = z / torch.clamp(z.norm(dim=-1, keepdim=True), min=1e-6)
        z = z.view(B, I, self.zi_latent_dim)  # [B, I, zp_latent_dim]

        return z


ContextEncoderModule: TypeAlias = (
    RNNContextEncoder | RNNContextEncoderDosing | TransformerContextEncoder
)


def get_individual_encoder(model_config: NodePKExperimentConfig):
    """Build the configured individual encoder implementation."""

    if model_config.network.individual_encoder_name == "RNNContextEncoder":
        return RNNContextEncoder(model_config)
    if model_config.network.individual_encoder_name == "RNNContextEncoderDosing":
        return RNNContextEncoderDosing(model_config)
    if model_config.network.individual_encoder_name == "TransformerContextEncoder":
        return TransformerContextEncoder(model_config)
    else:
        raise Exception("Encoder Not Defined")
