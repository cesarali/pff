from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torchdiffeq import odeint
from torchtyping import TensorType, patch_typeguard
from torchtyping import TensorType as TT
from typeguard import typechecked

from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.models.architectures.blocks import MLP, ResidualConcatMLP
from pff.models.architectures.encoders_pk import TimeObsSeparateEncoder

patch_typeguard()


class BaseDecoder(nn.Module):
    """Common decoder utilities shared across concrete architectures."""

    #: Default dimensionality of the conditioning feature vector constructed
    #: from ``init_state``, ``first_t_s``, ``dose`` and ``route`` scalars.
    DEFAULT_CONDITIONING_FEATURE_DIM = 4

    @classmethod
    def _resolve_init_feature_dim(cls, init_dim: Optional[int]) -> int:
        """Return the effective conditioning feature dimension."""
        if init_dim is None:
            return cls.DEFAULT_CONDITIONING_FEATURE_DIM
        if init_dim <= 0:
            raise ValueError("init_dim must be a positive integer")
        return init_dim

    def __init__(
        self,
        *,
        use_covariance: bool,
        cov_proj_dim: int,
        combine_latent_mode: str,
        zi_latent_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.use_covariance = use_covariance
        self.cov_proj_dim = cov_proj_dim
        self.combine_latent_mode = combine_latent_mode
        self.zi_latent_dim = zi_latent_dim

        if use_covariance:
            # Each point i→h_i ∈ ℝᵖ
            self.cov_head = nn.Linear(self.hidden_dim, cov_proj_dim)

        if combine_latent_mode == "mlp":
            self.combine_mlp = MLP(
                in_dim=2 * zi_latent_dim,
                out_dim=zi_latent_dim,
                hidden_dim=zi_latent_dim,
                num_layers=2,
                activation="ReLU",
                dropout=dropout,
                norm="layer",
            )
        elif combine_latent_mode == "sum":
            self.combine_mlp = None
        else:
            raise ValueError(f"Unknown combine_latent_mode '{combine_latent_mode}'")

    # ------------------------------------------------------------------ #
    @typechecked
    def combine_latents(
        self,
        z_s: Optional[TensorType["B", "Z"]],
        z_i: TensorType["B", "I", "Z"],
    ) -> TensorType["B", "I", "Z"]:
        """Fuse study- and individual-level latents according to config."""
        B, I, Z = z_i.shape
        # z_s: [B,Z], z_i: [B,I,Z]
        if z_s is None:
            return z_i
        if z_s.shape != (B, Z):
            raise ValueError(
                f"Expected z_s to have shape [B, Z]; got {tuple(z_s.shape)} "
                f"while z_i has shape {tuple(z_i.shape)}"
            )

        # Broadcast z_s over individuals → [B,I,Z]
        z_s_exp = z_s.unsqueeze(1).expand(B, I, Z)

        if self.combine_latent_mode == "sum":
            # Elementwise sum, preserves [B,I,Z]
            return z_s_exp + z_i
        if self.combine_latent_mode == "mlp":
            assert self.combine_mlp is not None
            # Concatenate along latent dim → [B,I,2Z]
            z_cat = torch.cat([z_s_exp, z_i], dim=-1)
            # Flatten BI dimension → [B*I,2Z]
            z_flat = z_cat.view(B * I, 2 * Z)
            # MLP projection → [B*I,Z]
            z_comb = self.combine_mlp(z_flat)
            # Restore original grouping → [B,I,Z]
            return z_comb.view(B, I, Z)
        raise RuntimeError("Unsupported combine_latent_mode encountered.")

    # ------------------------------------------------------------------ #
    @typechecked
    def _prepare_init_features(
        self,
        init_state: TensorType["BI", 1, 1],  # [B*I, 1, 1]
        time_reference: TensorType["BI", 1, 1],  # [B*I, 1, 1]
        dose: TensorType["BI", 1, 1],  # [B*I, 1, 1]
        route: TensorType["BI", 1, 1],  # [B*I, 1, 1]
        batch_size: int,
    ) -> TensorType["BI", "D_init"]:
        """
        Flatten and concatenate decoder conditioning scalars.

        Each input tensor has shape [B*I, 1, 1].
        After flattening → [B*I, 1], concatenation gives [B*I, 4].
        The output is used to build h₀ token features.
        """

        # Input summary:
        # init_state     [B*I, 1, 1]
        # time_reference [B*I, 1, 1]
        # dose           [B*I, 1, 1]
        # route          [B*I, 1, 1]

        features = []
        for name, tensor in (
            ("initial state", init_state),
            ("time reference", time_reference),
            ("dose", dose),
            ("route", route),
        ):
            if tensor.shape[0] != batch_size:
                raise ValueError(
                    f"Decoder expected first dimension {batch_size} for {name}, "
                    f"but received shape {tuple(tensor.shape)}"
                )

            # Flatten [B*I,1,1] → [B*I,1]
            features.append(tensor.view(batch_size, -1))  # shape: [B*I,1]

        # Concatenate along last dim: [B*I,4]
        init_features = torch.cat(features, dim=-1)

        # Expected feature dim (default 4 unless overridden)
        expected_dim = getattr(self, "init_feature_dim", init_features.shape[-1])
        if init_features.shape[-1] != expected_dim:
            raise ValueError(
                f"Constructed initial feature dimension does not match decoder configuration: "
                f"got {init_features.shape[-1]}, expected {expected_dim}."
            )

        # Final output shape: [B*I, D_init] typically [B*I,4]
        return init_features

    # ------------------------------------------------------------------ #
    def _build_H(
        self,
        hidden: TensorType["BI*T", "hidden_dim"],  # flattened along BI,T
    ) -> TensorType["BI*T", "p"] | None:
        """
        Turns per-point hidden representations → H  (or returns None if diagonal case).

        Input:  hidden [B*I*T, hidden_dim]
        Output: H [B*I*T, p]  (covariance projection)
        """
        if not self.use_covariance:
            return None
        # Linear projection: [B*I*T, hidden_dim] → [B*I*T, p]
        return self.cov_head(hidden)


class CrossAttentionBlock(nn.Module):
    """Single cross-attention block with residual connections."""

    def __init__(self, hidden_dim: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp = MLP(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            activation="ReLU",
            dropout=dropout,
            norm="layer",
        )

    def forward(
        self,
        q: TensorType["B", "T", "H"],
        k: TensorType["B", "L", "H"],
        v: TensorType["B", "L", "H"],
    ) -> TensorType["B", "T", "H"]:
        out, _ = self.attn(q, k, v)
        out = out + q
        ff = self.mlp(out.reshape(-1, out.shape[-1])).reshape(out.shape)
        return out + ff


class TransformerDecoder(BaseDecoder):
    """Transformer-style decoder with cross-attention over (z, h0)."""

    def __init__(self, config: NodePKExperimentConfig, init_dim: Optional[int] = None) -> None:
        use_covariance = False
        if config.network.loss_name == "mv_nll":
            use_covariance = True

        net = config.network
        self.hidden_dim = net.decoder_hidden_dim
        self.decoder_number_of_layers = net.decoder_num_layers
        self.nhead = getattr(net, "aggregator_num_heads", 4)
        self.zi_latent_dim = net.zi_latent_dim
        self.init_feature_dim = self._resolve_init_feature_dim(init_dim)

        super().__init__(
            use_covariance=use_covariance,
            cov_proj_dim=config.network.cov_proj_dim,
            combine_latent_mode=config.network.combine_latent_mode,
            zi_latent_dim=self.zi_latent_dim,
            dropout=net.dropout,
        )
        assert self.hidden_dim % self.nhead == 0, "hidden_dim must be divisible by nhead"

        # ---- projections --------------------------------------------------
        self.time_proj = MLP(
            in_dim=1,
            out_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.decoder_number_of_layers,
            activation="ReLU",
            dropout=net.dropout,
            norm="layer",
        )
        self.z_proj = MLP(
            in_dim=self.zi_latent_dim,
            out_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.decoder_number_of_layers,
            activation="ReLU",
            dropout=net.dropout,
            norm="layer",
        )
        self.init_proj = MLP(
            in_dim=self.init_feature_dim,
            out_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.decoder_number_of_layers,
            activation="ReLU",
            dropout=net.dropout,
            norm="layer",
        )

        # ---- stacked cross-attention blocks ------------------------------
        self.num_attention_layers = getattr(net, "decoder_attention_layers", 1)
        self.attn_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(self.hidden_dim, self.nhead, net.dropout)
                for _ in range(self.num_attention_layers)
            ]
        )

        # ---- output heads -----------------------------------------------
        self.mean_head = MLP(
            in_dim=self.hidden_dim,
            out_dim=1,
            hidden_dim=net.decoder_hidden_dim,
            num_layers=net.output_head_num_layers,
            activation="ReLU",
            dropout=net.dropout,
        )
        self.logvar_head = MLP(
            in_dim=self.hidden_dim,
            out_dim=1,
            hidden_dim=net.decoder_hidden_dim,
            num_layers=net.output_head_num_layers,
            activation="ReLU",
            dropout=net.dropout,
        )

    # --------------------------------------------------------------------- #
    @typechecked
    def forward(  # ───────────────────────────── #
        self,
        init_state: TensorType["BI", 1, 1],
        decode_time: TensorType["BI", "T", 1],
        z_s: Optional[TensorType["B", "zi_latent_dim"]],
        z_i: TensorType["B", "I", "zi_latent_dim"],
        *,
        dose: TensorType["BI", 1, 1],
        route: TensorType["BI", 1, 1],
        first_t_s: TensorType["BI", 1, 1],
    ) -> tuple[  # ───────────────────────────── #
        TensorType["BI", "T", 1],  # mean
        TensorType["BI", "T", 1],  # log-variance (diag)  – *always* returned
        TensorType["BI", "T", "p"] | None,  # H  (projection for Σ = H Hᵀ) – None if diag path
        TensorType["BI", "T", 1],  # decode_time (unchanged; handy for loss)
        TensorType["BI", "T", "hidden_dim"],  # hidden states h
    ]:
        """
        Forward pass for the transformer-style decoder.

        The decoder assembles the conditioning feature vector internally using
        the provided ``init_state``, ``first_t_s``, ``dose`` and ``route``
        tensors, ensuring a consistent interface across caller implementations.

        If `self.use_covariance` is **True** we additionally output `H ∈ ℝ^{BI×T×p}`.
        Downstream code can compute the Cholesky factor via
            L = torch.linalg.cholesky(H @ H.transpose(-1, -2) + εI)
        for each sequence.

        Returns
        -------
        mean, logvar, H (or None), decode_time, h
        """

        # ------------------------------------------------------------------ #
        BI, T, _ = decode_time.shape

        # ----- 1. embed query (decode times) ------------------------------ #
        # decode_time: [BI,T,1] → [BI*T,1] → [BI*T,H] → [BI,T,H]
        q = self.time_proj(decode_time.contiguous().view(BI * T, 1))  # [BI*T,H]
        q = q.view(BI, T, -1)  # [BI,T,H]

        # ----- 2. build key/value sequence (length 2) -------------------- #
        z_comb = self.combine_latents(z_s, z_i)  # [B,I,Z]
        B, I, Z = z_comb.shape
        if decode_time.shape[0] != B * I:
            raise ValueError(
                "Decode time batch does not match latent shapes: "
                f"{decode_time.shape[0]} vs B*I={B * I}"
            )
        if init_state.shape[0] != B * I:
            raise ValueError(
                "Initial state batch does not match latent shapes: "
                f"{init_state.shape[0]} vs B*I={B * I}"
            )

        BI = B * I
        z_flat = z_comb.view(BI, Z)

        z_tok = self.z_proj(z_flat)  # [BI,H]
        z_tok = z_tok.unsqueeze(1)  # [BI,1,H]

        init_features = self._prepare_init_features(
            init_state,
            first_t_s,
            dose,
            route,
            BI,
        )
        h0_tok = self.init_proj(init_features)  # [BI,H]
        h0_tok = h0_tok.unsqueeze(1)  # [BI,1,H]

        kv = torch.cat([h0_tok, z_tok], dim=1)  # [BI,2,H]
        k = v = kv  # [BI,2,H]

        # ----- 3. stacked cross-attention blocks ------------------------- #
        x = q
        for block in self.attn_blocks:
            x = block(x, k, v)
        h = x

        # ----- 5. output heads ------------------------------------------- #
        h_flat = h.reshape(BI * T, -1)  # [BI*T,H]

        # mean ----------------------------------------------------------------
        mean: TensorType["BI*T", 1] = F.softplus(self.mean_head(h_flat))  # [BI*T,1]
        mean = mean.view(BI, T, 1)  # [BI,T,1]

        # log-variance (diagonal) ---------------------------------------------
        logvar: TensorType["BI*T", 1] = self.logvar_head(h_flat)  # [BI*T,1]
        logvar = torch.clamp(logvar, -10.0, 10.0).view(BI, T, 1)  # [BI,T,1]

        # H for multivariate covariance ---------------------------------------
        H = None  # type: ignore
        if getattr(self, "use_covariance", False):
            H_flat: TensorType["BI*T", "p"] = self.cov_head(h_flat)  # [BI*T,p]
            H: TensorType["BI", "T", "p"] = H_flat.view(BI, T, self.cov_proj_dim)

        # ------------------------------------------------------------------ #
        return mean, logvar, H, decode_time, h


class TransformerDecoderZiQuery(BaseDecoder):
    """
    Transformer-style decoder where the *query* uses (time ⊕ z_i),
    and K/V attend over [h0, z_s]. Dosing (amount, route) is encoded
    with learned tables (linear + embedding), mirroring the encoder.
    """

    def __init__(self, config: NodePKExperimentConfig, init_dim: int | None = None) -> None:
        use_covariance = config.network.loss_name == "mv_nll"

        net = config.network
        self.hidden_dim = net.decoder_hidden_dim
        self.decoder_number_of_layers = net.decoder_num_layers
        self.nhead = getattr(net, "aggregator_num_heads", 4)
        self.zi_latent_dim = net.zi_latent_dim

        # (caller passes 4 normally: [init_value, first_t_s or last_t_s, dose, route])
        # We *internally* re-encode dose & route, so keep input dim for API compat.
        self.init_feature_dim = init_dim if init_dim is not None else 4

        super().__init__(
            use_covariance=use_covariance,
            cov_proj_dim=config.network.cov_proj_dim,
            combine_latent_mode=config.network.combine_latent_mode,
            zi_latent_dim=self.zi_latent_dim,
            dropout=net.dropout,
        )
        assert self.hidden_dim % self.nhead == 0, "hidden_dim must be divisible by nhead"

        # ── projections ──────────────────────────────────────────────────
        # time → H
        self.time_proj = MLP(
            in_dim=1,
            out_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.decoder_number_of_layers,
            activation="ReLU",
            dropout=net.dropout,
            norm="layer",
        )
        # z_s → H  (used in KV track)
        self.zs_proj = MLP(
            in_dim=self.zi_latent_dim,
            out_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.decoder_number_of_layers,
            activation="ReLU",
            dropout=net.dropout,
            norm="layer",
        )
        # init features (init value, first/last time, dose(raw), route(raw)) → H
        # NOTE: dose/route re-encoded below; this MLP sees the concatenated 4-tuple
        self.init_proj = MLP(
            in_dim=self.init_feature_dim,
            out_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.decoder_number_of_layers,
            activation="ReLU",
            dropout=net.dropout,
            norm="layer",
        )
        # z_i → H  (for queries)
        self.zi_proj = nn.Linear(self.zi_latent_dim, self.hidden_dim)

        # ── dosing encoders (mirror the encoder) ─────────────────────────
        # amount: ℝ → H
        self.dose_amount_encoder = nn.Linear(1, self.hidden_dim)
        # route: {0..n_routes-1} → H
        n_routes = len(config.dosing.route_options)  # e.g. ['oral', 'iv', ...]
        self.dose_type_embedding = nn.Embedding(
            num_embeddings=n_routes, embedding_dim=self.hidden_dim
        )
        # combine (amount_emb ⊕ type_emb) → H
        self.dose_proj = nn.Linear(2 * self.hidden_dim, self.hidden_dim)

        # ── cross-attention stack ────────────────────────────────────────
        self.num_attention_layers = getattr(net, "decoder_attention_layers", 1)
        self.attn_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(self.hidden_dim, self.nhead, net.dropout)
                for _ in range(self.num_attention_layers)
            ]
        )

        # ── heads ────────────────────────────────────────────────────────
        self.mean_head = MLP(
            in_dim=self.hidden_dim,
            out_dim=1,
            hidden_dim=net.decoder_hidden_dim,
            num_layers=net.output_head_num_layers,
            activation="ReLU",
            dropout=net.dropout,
        )
        self.logvar_head = MLP(
            in_dim=self.hidden_dim,
            out_dim=1,
            hidden_dim=net.decoder_hidden_dim,
            num_layers=net.output_head_num_layers,
            activation="ReLU",
            dropout=net.dropout,
        )
        if use_covariance:
            self.cov_head = MLP(
                in_dim=self.hidden_dim,
                out_dim=self.cov_proj_dim,
                hidden_dim=net.decoder_hidden_dim,
                num_layers=net.output_head_num_layers,
                activation="ReLU",
                dropout=net.dropout,
            )

    # ────────────────────────────────────────────────────────────────────────
    # helpers
    def _encode_dose_route(
        self,
        dose: TT["BI", 1, 1],  # dose amount (raw)
        route: TT["BI", 1, 1],  # route index (float/int)
    ) -> TT["BI", 1, "H"]:
        """Encode dosing amount & route like the encoder; returns a 1-step token."""
        BI = dose.shape[0]
        amt = self.dose_amount_encoder(dose.view(BI, 1))  # [BI,H]  # amt
        rix = route.view(BI).long().clamp_min(0)  # [BI]    # route idx
        typ = self.dose_type_embedding(rix)  # [BI,H]  # type
        tok = self.dose_proj(torch.cat([amt, typ], dim=-1))  # [BI,H]
        return tok.unsqueeze(1)  # [BI,1,H]

    def _prepare_init_features(
        self,
        init_state: TT["BI", 1, 1],
        first_t_s: TT["BI", 1, 1],
        dose: TT["BI", 1, 1],
        route: TT["BI", 1, 1],
        BI: int,
    ) -> TT["BI", "init_feature_dim"]:
        # Concatenate raw tuple to preserve upstream interface (init_dim=4)
        # [BI,1,1]×4 → [BI,4]
        feats = torch.cat([init_state, first_t_s, dose, route], dim=-1).view(
            BI, self.init_feature_dim
        )
        return feats

    # ────────────────────────────────────────────────────────────────────────
    @typechecked
    def forward(
        self,
        init_state: TT["BI", 1, 1],  # initial concentration                             # [BI,1,1]
        decode_time: TT[
            "BI", "T", 1
        ],  # relative decode grid                              # [BI,T,1]
        z_s: Optional[
            TT["B", "zi_latent_dim"]
        ],  # study latent                                      # [B,Z]
        z_i: TT[
            "B", "I", "zi_latent_dim"
        ],  # individual latent                                 # [B,I,Z]
        *,
        dose: TT["BI", 1, 1],  # dosing amount (raw)                               # [BI,1,1]
        route: TT["BI", 1, 1],  # dosing route index                                # [BI,1,1]
        first_t_s: TT["BI", 1, 1],  # reference time (e.g., first or last observed)     # [BI,1,1]
    ) -> tuple[
        TT["BI", "T", 1],  # mean
        TT["BI", "T", 1],  # log-variance (diag)
        TT["BI", "T", "p"] | None,  # H  (for MV NLL) or None
        TT["BI", "T", 1],  # decode_time passthrough
        TT["BI", "T", "hidden_dim"],  # hidden states h
    ]:
        """
        Q = f_t(time) ⊕ g(z_i)    (built per position, broadcasting z_i across T)
        K,V attend over tokens:   [ h0_token , z_s_token , dosing_token ]
        """
        # shapes
        BI, T, _ = decode_time.shape
        # Retrieve (B,I) from z_i
        B, I, Z = z_i.shape
        assert BI == B * I, f"Mismatch: BI={BI}, B*I={B * I}"

        # ── 1) Query from time + z_i ────────────────────────────────────
        # time → [BI,T,H]
        q_time = self.time_proj(decode_time.contiguous().view(BI * T, 1)).view(
            BI, T, self.hidden_dim
        )  # [BI,T,H]
        # z_i → [BI,1,H] → broadcast to [BI,T,H]
        zi_flat = z_i.view(B * I, Z)  # [BI,Z]
        q_zi = self.zi_proj(zi_flat).unsqueeze(1).expand(BI, T, self.hidden_dim)  # [BI,T,H]
        q = q_time + q_zi  # [BI,T,H]

        # ── 2) Keys/Values from [h0(z_s, init, dose, route), z_s, doseTok] ─────
        # z_s → [BI,1,H]
        zs_tok = self.zs_proj(z_s).unsqueeze(1).expand(B, I, -1).contiguous()  # [B,I,H]
        zs_tok = zs_tok.view(BI, 1, self.hidden_dim)  # [BI,1,H]

        # initial features (raw 4-tuple) → [BI,1,H]
        init_feats = self._prepare_init_features(init_state, first_t_s, dose, route, BI)  # [BI,4]
        h0_tok = self.init_proj(init_feats).unsqueeze(1)  # [BI,1,H]

        # dosing token via learned encoders → [BI,1,H]
        dose_tok = self._encode_dose_route(dose, route)  # [BI,1,H]

        # stack KV: [h0_tok, zs_tok, dose_tok] → [BI, 3, H]
        kv = torch.cat([h0_tok, zs_tok, dose_tok], dim=1)  # [BI,3,H]
        k = v = kv  # [BI,3,H]

        # ── 3) Stacked cross-attention ──────────────────────────────────
        x = q  # [BI,T,H]
        for block in self.attn_blocks:
            x = block(x, k, v)  # [BI,T,H]
        h = x

        # ── 4) Output heads ─────────────────────────────────────────────
        h_flat = h.reshape(BI * T, self.hidden_dim)  # [BI*T,H]

        mean = F.softplus(self.mean_head(h_flat))  # [BI*T,1]
        mean = mean.view(BI, T, 1)  # [BI,T,1]

        logvar = self.logvar_head(h_flat)  # [BI*T,1]
        logvar = torch.clamp(logvar, -10.0, 10.0).view(BI, T, 1)  # [BI,T,1]

        H = None
        if getattr(self, "use_covariance", False):
            H_flat = self.cov_head(h_flat)  # [BI*T,p]
            H = H_flat.view(BI, T, self.cov_proj_dim)  # [BI,T,p]

        return mean, logvar, H, decode_time, h


class ResidualConcatGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.cells = nn.ModuleList(
            [nn.GRUCell(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)]
        )

        # Residual projection: (input + hidden) → hidden
        self.residual_proj = nn.ModuleList(
            [
                nn.Linear((input_dim if i == 0 else hidden_dim) + hidden_dim, hidden_dim)
                for i in range(num_layers)
            ]
        )

    def forward(
        self, x_t: TensorType["B", "D_in"], h_t: TensorType["B", "L", "H"]
    ) -> TensorType["B", "L", "H"]:
        B, L, H = h_t.shape
        next_h = []

        for i, (cell, proj) in enumerate(zip(self.cells, self.residual_proj)):
            input_t = x_t if i == 0 else next_h[i - 1]
            h_i = cell(input_t, h_t[:, i])  # GRUCell update
            concat = torch.cat([h_i, input_t], dim=-1)  # Residual via concat
            h_i = proj(concat)  # Project to H
            next_h.append(h_i)

        return torch.stack(next_h, dim=1)  # [B, L, H]


class RNNDecoder(BaseDecoder):
    def __init__(self, neural_config: NodePKExperimentConfig, init_dim: Optional[int] = None):
        p = getattr(neural_config.network, "cov_proj_dim", 16)  # default 16
        use_covariance = False
        if neural_config.network.loss_name == "mv_nll":
            use_covariance = True
        self.decoder_rnn_hidden_dim = neural_config.network.decoder_rnn_hidden_dim
        self.hidden_dim = neural_config.network.decoder_hidden_dim
        self.rnn_decoder_number_of_layers = neural_config.network.rnn_decoder_number_of_layers
        self.zp_latent_dim = neural_config.network.zi_latent_dim
        self.zi_latent_dim = self.zp_latent_dim
        self.node_step = neural_config.network.node_step
        self.exclusive_node_step = getattr(neural_config.network, "exclusive_node_step", False)
        self.use_gru_jump = self.node_step and not self.exclusive_node_step
        self.init_feature_dim = self._resolve_init_feature_dim(init_dim)

        super().__init__(
            use_covariance=use_covariance,
            cov_proj_dim=neural_config.network.cov_proj_dim,
            combine_latent_mode=neural_config.network.combine_latent_mode,
            zi_latent_dim=self.zi_latent_dim,
            dropout=neural_config.network.dropout,
        )
        # 🔑  Re-create `cov_head` so it expects the RNN hidden dimension
        if self.use_covariance:
            self.cov_head = nn.Linear(self.decoder_rnn_hidden_dim, p)

        # Encode (time, z) exactly as before
        self.input_encoder = TimeObsSeparateEncoder(
            time_dim=1,
            obs_dim=self.zp_latent_dim,
            hidden_dim=neural_config.network.time_obs_encoder_hidden_dim,
            output_dim=neural_config.network.time_obs_encoder_output_dim,
        )

        # Compute input dimension of RNN after encoding
        self.rnn_input_dim = 2 * neural_config.network.time_obs_encoder_output_dim

        # ── GRU jump: create it **only when it will be used** -------------
        self.rnn_cell: Optional[ResidualConcatGRUCell]
        if self.use_gru_jump:
            self.rnn_cell = ResidualConcatGRUCell(
                input_dim=self.rnn_input_dim,
                hidden_dim=self.decoder_rnn_hidden_dim,
                num_layers=self.rnn_decoder_number_of_layers,
            )
        else:
            self.rnn_cell = None  # 👈 no unused parameters

        self.init_hidden = ResidualConcatMLP(
            in_dim=self.init_feature_dim,
            out_dim=self.decoder_rnn_hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=neural_config.network.init_hidden_num_layers,
            activation="ReLU",
            dropout=neural_config.network.dropout,
            norm="layer",
            residual=True,
        )

        self.mean_proj = MLP(
            in_dim=self.decoder_rnn_hidden_dim,
            out_dim=1,
            hidden_dim=self.hidden_dim,
            num_layers=neural_config.network.output_head_num_layers,
            activation="ReLU",
            dropout=neural_config.network.dropout,
        )

        self.logvar_proj = MLP(
            in_dim=self.decoder_rnn_hidden_dim,
            out_dim=1,
            hidden_dim=self.hidden_dim,
            num_layers=neural_config.network.output_head_num_layers,
            activation="ReLU",
            dropout=neural_config.network.dropout,
        )

        self.drift = MLP(
            in_dim=self.decoder_rnn_hidden_dim + self.rnn_input_dim,
            out_dim=self.decoder_rnn_hidden_dim,
            hidden_dim=self.decoder_rnn_hidden_dim,
            num_layers=neural_config.network.drift_num_layers,
            activation="Tanh",
            dropout=neural_config.network.dropout,
        )

    # ------------------------------------------------------------
    # forward()  — fixed shapes for multi–layer ODE-RNN decoder
    # ------------------------------------------------------------
    @typechecked
    def forward(  # ─────────────────────── #
        self,
        init_state: TensorType["BI", 1, 1],
        decode_time: TensorType["BI", "T", 1],
        z_s: Optional[TensorType["B", "zi_latent_dim"]],
        z_i: TensorType["B", "I", "zi_latent_dim"],
        *,
        dose: TensorType["BI", 1, 1],
        route: TensorType["BI", 1, 1],
        first_t_s: TensorType["BI", 1, 1],
    ) -> Tuple[  # ─────────────────────── #
        TensorType["BI", "T", 1],  # mean
        TensorType["BI", "T", 1],  # log-var (diag)
        Union[TensorType["BI", "T", "p"], None],  # H  (projection) | None
        TensorType["BI", "T", 1],  # echo decode_time
        TensorType["BI", "T", "Hdim"],  # hidden sequence (last layer)
    ]:
        """
        RNN/Neural-ODE decoder forward pass with optional full-covariance output.

        Conditioning features are constructed inside the decoder from the
        supplied ``init_state``, ``first_t_s``, ``dose`` and ``route`` scalars
        for each trajectory.

        When `self.use_covariance` is **True** the third returned tensor is
        `H` whose rows (one per target point) are projected via an MLP so that
        Σ = H Hᵀ  (Cholesky can be obtained downstream by
        `torch.linalg.cholesky(Σ + εI)`).
        """

        # ------------------------------------------------------------------ #
        BI, T, _ = decode_time.shape
        L = self.rnn_decoder_number_of_layers
        Hdim = self.decoder_rnn_hidden_dim
        D_in = self.rnn_input_dim
        device = decode_time.device
        p = getattr(self, "cov_proj_dim", 0)  # safe even if flag is False

        # ── 0. Encode (t , zπ) ───────────────────────────────────────────── #
        z_comb = self.combine_latents(z_s, z_i)  # [B,I,Z]
        B, I, Z = z_comb.shape
        if decode_time.shape[0] != B * I:
            raise ValueError(
                "Decode time batch does not match latent shapes: "
                f"{decode_time.shape[0]} vs B*I={B * I}"
            )
        if init_state.shape[0] != B * I:
            raise ValueError(
                "Initial state batch does not match latent shapes: "
                f"{init_state.shape[0]} vs B*I={B * I}"
            )

        BI = B * I
        z_flat = z_comb.view(B * I, Z)
        z_expand: TensorType["BI", "T", "zi_latent_dim"] = z_flat.unsqueeze(1).repeat(1, T, 1)
        encoded: TensorType["BI", "T", "D_in"] = self.input_encoder(decode_time, z_expand)

        # ── 1. Init hidden  h₀  (replicate across layers) ─────────────────── #
        init_features = self._prepare_init_features(
            init_state,
            first_t_s,
            dose,
            route,
            BI,
        )
        h0: TensorType["BI", "Hdim"] = torch.tanh(self.init_hidden(init_features))
        h_t: TensorType["BI", "L", "Hdim"] = h0.unsqueeze(1).repeat(1, L, 1)

        # ── 2. Buffers for outputs ────────────────────────────────────────── #
        mean_out = torch.empty(BI, T, 1, device=device)
        logvar_out = torch.empty_like(mean_out)
        h_sequence = torch.empty(BI, T, Hdim, device=device)  # LAST-layer states

        if getattr(self, "use_covariance", False):
            H_out = torch.empty(BI, T, p, device=device)  # projection rows
        else:
            H_out = None

        prev_t = decode_time[:, 0, 0]  # [BI]

        # ── 3. Main loop over target time points ─────────────────────────── #
        for t_i in range(T):
            x_t: TensorType["BI", "D_in"] = encoded[:, t_i, :]  # [BI,D_in]
            cur_t = decode_time[:, t_i, 0]  # [BI]
            dt = (cur_t - prev_t).unsqueeze(-1)  # [BI,1]

            # 3-A) Neural ODE Euler step  (all L layers in parallel) -------- #
            if self.node_step:
                x_rep = x_t.unsqueeze(1).repeat(1, L, 1)  # [BI,L,D_in]
                drift_in = torch.cat([h_t, x_rep], dim=-1)  # [BI,L,Hdim+D_in]

                dh = self.drift(drift_in.view(BI * L, Hdim + D_in))  # [BI*L,Hdim]
                dh = dh.view(BI, L, Hdim)
                h_t = h_t + dh * dt.unsqueeze(-1)  # Euler update

            # 3-B) GRU jump with residual-concat ---------------------------- #
            if self.node_step and not self.exclusive_node_step:
                h_t = self.rnn_cell(x_t, h_t)  # custom cell: [BI,L,Hdim]

            # 3-C) Store LAST-layer hidden ---------------------------------- #
            h_last: TensorType["BI", "Hdim"] = h_t[:, -1, :]  # [BI,Hdim]
            h_sequence[:, t_i, :] = h_last  # log for analysis

            # 3-D) Output heads -------------------------------------------- #
            mean_out[:, t_i, 0] = F.softplus(self.mean_proj(h_last)).squeeze(-1)
            logvar_out[:, t_i, 0] = torch.clamp(  # keep numerics sane
                self.logvar_proj(h_last).squeeze(-1), min=-10.0, max=10.0
            )

            if H_out is not None:
                H_out[:, t_i, :] = self.cov_head(h_last)  # [BI,p]

            prev_t = cur_t  # advance clock

        # ------------------------------------------------------------------ #
        return mean_out, logvar_out, H_out, decode_time, h_sequence


class ODEDecoder(BaseDecoder):
    """
    Hidden-state decoder that integrates dh/dt = fθ(h, x_t) with torchdiffeq.

    • Works with irregular and different time grids for every path.
    • `forward()` has the same TensorType contract as TransformerDecoder:
        mean, logvar, H|None, decode_time, h
    """

    # --------------------------------------------------------------------- #
    def __init__(self, config: NodePKExperimentConfig, init_dim: Optional[int] = None) -> None:
        net = config.network

        use_covariance = False
        if net.loss_name == "mv_nll":
            use_covariance = True

        self.hidden_dim = net.decoder_rnn_hidden_dim
        self.zi_latent_dim = net.zi_latent_dim
        self.init_feature_dim = self._resolve_init_feature_dim(init_dim)

        super().__init__(
            use_covariance=use_covariance,
            cov_proj_dim=net.cov_proj_dim,
            combine_latent_mode=net.combine_latent_mode,
            zi_latent_dim=self.zi_latent_dim,
            dropout=net.dropout,
        )

        H = net.decoder_rnn_hidden_dim
        z_d = net.zi_latent_dim

        # (t , zπ) → x_t
        self.input_encoder = TimeObsSeparateEncoder(
            time_dim=1,
            obs_dim=z_d,
            hidden_dim=net.time_obs_encoder_hidden_dim,
            output_dim=net.time_obs_encoder_output_dim,
        )
        # D = dim(x_t)
        self.x_dim = 2 * net.time_obs_encoder_output_dim

        # h₀ network
        self.init_hidden = MLP(
            in_dim=self.init_feature_dim,
            out_dim=H,
            hidden_dim=H,
            num_layers=net.init_hidden_num_layers,
            activation="ReLU",
            dropout=net.dropout,
            norm="layer",
        )

        # drift fθ(h, x)
        self.drift = MLP(
            in_dim=H + self.x_dim,
            out_dim=H,
            hidden_dim=H,
            num_layers=net.drift_num_layers,
            activation="Tanh",
            dropout=net.dropout,
        )

        # output heads
        self.mean_head = MLP(
            in_dim=H,
            out_dim=1,
            hidden_dim=H,
            num_layers=net.output_head_num_layers,
            activation="ReLU",
            dropout=net.dropout,
        )
        self.logvar_head = MLP(
            in_dim=H,
            out_dim=1,
            hidden_dim=H,
            num_layers=net.output_head_num_layers,
            activation="ReLU",
            dropout=net.dropout,
        )

    # --------------------------------------------------------------------- #
    def _odefunc(self, t: torch.Tensor, h: torch.Tensor, x_const: torch.Tensor) -> torch.Tensor:
        """Piecewise-constant control: x(t) = x_const on (tᵢ , tᵢ₊₁]."""
        return self.drift(torch.cat([h, x_const], dim=-1))

    # --------------------------------------------------------------------- #
    @typechecked
    def forward(
        self,
        init_state: TensorType["BI", 1, 1],
        decode_time: TensorType["BI", "T", 1],
        z_s: Optional[TensorType["B", "zi_latent_dim"]],
        z_i: TensorType["B", "I", "zi_latent_dim"],
        *,
        dose: TensorType["BI", 1, 1],
        route: TensorType["BI", 1, 1],
        first_t_s: TensorType["BI", 1, 1],
    ) -> tuple[
        TensorType["BI", "T", 1],  # mean
        TensorType["BI", "T", 1],  # log-variance (diag)
        TensorType["BI", "T", "p"] | None,  # H for covariance, or None
        TensorType["BI", "T", 1],  # decode_time (echoed)
        TensorType["BI", "T", "hidden_dim"],  # hidden states h
    ]:
        """
        Neural ODE decoder with the same external contract as TransformerDecoder.

        Returns
        -------
        mean, logvar, H|None, decode_time, h_seq
        """

        BI, T, _ = decode_time.shape
        device = decode_time.device
        H = self.hidden_dim

        # ── 1. combine latents and encode inputs x_{i,t} ───────────────────
        z_comb = self.combine_latents(z_s, z_i)  # [B,I,Z]
        B, I, Z = z_comb.shape

        if decode_time.shape[0] != B * I:
            raise ValueError(
                "Decode time batch does not match latent shapes: "
                f"{decode_time.shape[0]} vs B*I={B * I}"
            )
        if init_state.shape[0] != B * I:
            raise ValueError(
                "Initial state batch does not match latent shapes: "
                f"{init_state.shape[0]} vs B*I={B * I}"
            )

        BI = B * I
        z_flat = z_comb.view(BI, Z)
        z_rep = z_flat.unsqueeze(1).expand(-1, T, -1)  # [BI,T,Z]
        x_enc = self.input_encoder(decode_time, z_rep)  # [BI,T,D]

        # ── 2. build global timeline τ₀…τ_{U-1} ────────────────────────────
        times_flat = decode_time.view(-1)  # [BI*T]
        times_global, inv = torch.unique(times_flat, sorted=True, return_inverse=True)
        inv = inv.view(BI, T)  # (i,t) ↦ k
        U = times_global.numel()

        # ── 3. buffers ─────────────────────────────────────────────────────
        mean = torch.empty(BI, T, 1, device=device)
        logvar = torch.empty_like(mean)
        h_seq = torch.empty(BI, T, H, device=device)

        # h₀ per path from (init_state, first_t_s, dose, route, ...)
        init_features = self._prepare_init_features(
            init_state,
            first_t_s,
            dose,
            route,
            BI,
        )
        h_t = torch.tanh(self.init_hidden(init_features))  # [BI,H]

        # per-path column pointer into time dimension
        ptr = torch.zeros(BI, dtype=torch.long, device=device)  # current time index per path
        idx_all = torch.arange(BI, device=device)

        # initial control x_curr
        x_curr = x_enc[:, 0, :]  # [BI,D]

        t_prev = times_global[0].item()
        for k in range(U):
            t_k = times_global[k].item()

            # integrate dh/dt = f(h, x_const) from τ_{k-1} → τ_k (skip k=0)
            if k > 0:
                x_const = x_curr  # [BI,D]

                def f(t, h):
                    return self.drift(torch.cat([h, x_const], dim=-1))

                ts = torch.tensor([t_prev, t_k], device=device, dtype=decode_time.dtype)
                h_t = odeint(f, h_t, ts, rtol=1e-3, atol=1e-4)[-1]  # [BI,H]

            # which paths need output at τ_k?
            # for each path i, look at its current column ptr[i]
            mask_k = inv[idx_all, ptr] == k  # [BI]
            if mask_k.any():
                idx = mask_k.nonzero(as_tuple=True)[0]  # paths to write
                col = ptr[idx]  # their column indices

                h_sel = h_t[idx]  # [m,H]

                # write outputs
                m_sel = F.softplus(self.mean_head(h_sel)).squeeze(-1)  # [m]
                lv_sel = self.logvar_head(h_sel).squeeze(-1)  # [m]

                mean[idx, col, 0] = m_sel
                logvar[idx, col, 0] = torch.clamp(lv_sel, -10.0, 10.0)
                h_seq[idx, col, :] = h_sel

                # advance those paths to their next time step (if any)
                ptr[idx] += 1
                # update controls x_curr for *all* paths based on new ptr
                ptr_clamped = ptr.clamp(max=T - 1)
                x_curr = x_enc[idx_all, ptr_clamped]

            t_prev = t_k

        # ── 4. optional covariance factor H ────────────────────────────────
        H_mat = None  # type: ignore
        h_flat = h_seq.view(BI * T, H)  # [BI*T,H]

        if getattr(self, "use_covariance", False):
            # cov_head is provided by BaseDecoder when use_covariance=True
            H_flat: TensorType["BI*T", "p"] = self.cov_head(h_flat)  # [BI*T,p]
            H_mat: TensorType["BI", "T", "p"] = H_flat.view(BI, T, self.cov_proj_dim)

        # mean / logvar are already set; just ensure proper clamping
        # (done in loop) and shapes are [BI,T,1]

        return mean, logvar, H_mat, decode_time, h_seq
