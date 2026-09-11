from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torchdiffeq import odeint
from torchtyping import TensorType, patch_typeguard
from torchtyping import TensorType as TT
from typeguard import typechecked

from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.models.architectures.blocks import (
    MLP,
    LayerNorm,
    OperatorSelfAttnBlock,
    OperatorCrossAttnBlock,
    TimeFourierEmbedding,
    LearnableFourierEmbedding,
)

from typing import List
import math

patch_typeguard()


def _resolve_vector_field_config(config: Any):
    """Return the vector field configuration for NodePK or FlowPK configs."""
    if hasattr(config, "vector_field") and config.vector_field is not None:
        return config.vector_field
    if hasattr(config, "network") and config.network is not None:
        return config.network
    raise ValueError("Config must define either 'vector_field' or 'network' for vector fields.")


class BaseVectorField(nn.Module):
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

    def compute_trapezoidal_weights(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Compute trapezoidal quadrature weights from sorted grid positions.
        
        Following Eq. (58)-(59) in arXiv:2406.06486, the trapezoidal rule uses:
            Δx_k = x_k - x_{k-1}  for k = 2, ..., N  (1-indexed, paper notation)
        
        Args:
            x: Grid positions of shape (B, N), assumed sorted in ascending order.
            mask: Optional boolean mask of shape (B, N) where True indicates valid positions.
                  If provided, weights for invalid (padded) positions are set to zero.
        
        Returns:
            dx: Quadrature weights of shape (B, N).
                dx[:, 0] = 0 (placeholder, unused in trapezoidal sum which starts at k=2)
                dx[:, k] = x[:, k] - x[:, k-1]  for k = 1, ..., N-1
                
        Usage with OperatorSelfAttention:
            dx = compute_trapezoidal_weights(x, mask)  # (B, N)
            dx = dx.unsqueeze(1)  # (B, 1, N) for broadcasting over heads
            out = self_attn(features, dx=dx)
        """
        B, N = x.shape
        diff = x[:, 1:] - x[:, :-1]  # (B, N-1) dx[k] = x[k] - x[k-1] for k=1..N-1
        dx = torch.cat([torch.zeros(B, 1, device=x.device, dtype=x.dtype), diff], dim=1)  # (B, N)
        if mask is not None:
            dx = dx * mask.float()
        # dx = dx / dx.sum(dim=1, keepdim=True)  # Normalize to sum to 1
        return dx


class _OldTransformerVectorField(BaseVectorField):
    """ Operator Transformer encoder-decoder for irregularly sampled time series.
    """
    def __init__(self, config: NodePKExperimentConfig, init_dim: Optional[int] = None) -> None:

        net = _resolve_vector_field_config(config)
        self.hidden_dim = net.hidden_dim
        self.use_spectral_qkv = net.use_spectral_qkv
        self.fourier_modes = net.fourier_modes
        self.time_fourier_max_freq = net.time_fourier_max_freq
        self.encoder_num_heads = net.encoder_num_heads
        self.decoder_num_heads = net.decoder_num_heads
        self.num_encoder_layers = net.encoder_attention_layers
        self.num_decoder_layers = net.decoder_attention_layers
        self.dropout = net.dropout

        super().__init__(
            use_covariance=False,
            cov_proj_dim=net.cov_proj_dim,
            combine_latent_mode=net.combine_latent_mode,
            zi_latent_dim=net.zi_latent_dim,
            dropout=net.dropout,
        )
        assert self.hidden_dim % self.encoder_num_heads == 0, "hidden_dim of encoder must be divisible by nhead_encoder"
        assert self.hidden_dim % self.decoder_num_heads == 0, "hidden_dim of decoder must be divisible by nhead_encoder"

        # ---- projections --------------------------------------------------

        self.embeddings = nn.ModuleDict(
            dict(
                flow_time=TimeFourierEmbedding(
                    dim=self.hidden_dim,
                    max_freq=self.time_fourier_max_freq,
                ),
                fourier_ctx=LearnableFourierEmbedding(
                    feat_split=[1, 2],  # t | x , f(x)
                    dim_fourier_hidden=[self.hidden_dim, self.hidden_dim],
                    dim_hidden=self.hidden_dim,
                    dim_out=self.hidden_dim,
                    gamma=1.0,
                ),
                norm=LayerNorm(3),
                flow_path=MLP(
                    in_dim=2,
                    out_dim=self.hidden_dim,
                    hidden_dim=self.hidden_dim,
                    num_layers=3,
                    activation="GELU",
                    dropout=net.dropout,
                    norm="layer",
                ),
                study=MLP(
                    in_dim=2,
                    out_dim=self.hidden_dim,
                    hidden_dim=self.hidden_dim,
                    num_layers=3,
                    activation="GELU",
                    dropout=net.dropout,
                    norm="layer",
                ),
            )
        )

        self.encoder = nn.ModuleDict(
            dict(
                norm1=LayerNorm(self.hidden_dim),
                attn_blocks=nn.ModuleList(
                    [
                        OperatorSelfAttnBlock(
                            self.hidden_dim,
                            self.encoder_num_heads,
                            use_spectral_qkv=self.use_spectral_qkv,
                            fourier_modes=self.fourier_modes,
                            dropout=self.dropout,
                            bias=False,
                            qk_layernorm=True,
                        )
                        for _ in range(self.num_encoder_layers)
                    ]
                ),
                norm2=LayerNorm(self.hidden_dim),
            )
        )

        self.decoder = nn.ModuleDict(
            dict(
                norm1=LayerNorm(self.hidden_dim),
                attn_blocks=nn.ModuleList(
                    [
                        OperatorCrossAttnBlock(
                            self.hidden_dim,
                            self.decoder_num_heads,
                            use_spectral_qkv=self.use_spectral_qkv,
                            fourier_modes=self.fourier_modes,
                            dropout=self.dropout,
                            bias=False,
                            qk_layernorm=True,
                        )
                        for _ in range(self.num_decoder_layers)
                    ]
                ),
                norm2=LayerNorm(self.hidden_dim),
                head=MLP(
                    in_dim=self.hidden_dim,
                    out_dim=1,
                    hidden_dim=self.hidden_dim,
                    num_layers=3,
                    activation="GELU",
                    dropout=self.dropout,
                ),
            )
        )

    # --------------------------------------------------------------------- #
    @typechecked
    def forward(
        self,
        x: TensorType["B", "Nt", 2],  # Nt = It * T
        ctx: TensorType["B", "Nc", 2],  # Nc = Ic * S
        mask_pad_x: TensorType["B", "Nt"],
        mask_pad_ctx: TensorType["B", "Nc"],
        mask_attn_ctx: TensorType["B", "Nc", "Nc"],
        flow_t: TensorType["B", 1],
    ):

        B, Nt, _ = x.shape
        Nc = ctx.shape[1]

        # print(f"x:{x[0]}")
        # print(f"ctx:{ctx[0]}")
        # print(f"mask_pad_x: {mask_pad_x[0]}")
        # print(f"mask_pad_ctx: {mask_pad_ctx[0]}")

        #...get trapezoidal weights for target and context grids
        dx = self.compute_trapezoidal_weights(x[..., 0], mask=mask_pad_x)  # (B, Nt)  x[..., 1] (x is [f(x), x] point cloud)
        dx = dx.clamp(min=0.0)
        dctx = self.compute_trapezoidal_weights(ctx[..., 0], mask=mask_pad_ctx)  # (B, Nc)
        dctx = dctx.clamp(min=0.0)

        # print(f"dx: {dx[0]}")
        # print(f"dc: {dctx[0]}")

        #...attention masks
        attn_mask_encoder = mask_attn_ctx.unsqueeze(1).expand(
            -1, self.encoder_num_heads, -1, -1
        )  # (B, n_heads, Nc, Nc) SA mask for encoder
        attn_mask_decoder = self.attention_mask(
            mask_pad_x, self.decoder_num_heads,
        )  # (B, n_heads, Nt, Nt) SA mask for decoder
        attn_mask_cross = self.cross_attention_mask(
            mask_pad_x, mask_pad_ctx, self.decoder_num_heads
        )  # (B, n_heads, Nt, Nc) CA mask for decoder

        #...time/fourier embeddings
        t_emb = self.embeddings.flow_time(flow_t)  # (B, D)
        t_emb = t_emb.unsqueeze(1).expand(-1, Nt, -1)  # (B, Nt, D)
        t = flow_t.unsqueeze(1).expand(-1, Nc, 1)  # (B, Nc, 1)
        f = torch.cat([t, ctx], dim=-1)  # (B, Nc, 3)
        f = self.embeddings.norm(f)
        f_emb = self.embeddings.fourier_ctx(f)  # (B, Nc, D)

        #...context encoder
        c = self.embeddings.study(ctx) + f_emb  # (B, Nc, D)
        c = self.encoder.norm1(c)

        for block in self.encoder.attn_blocks:
            c = block(c, dctx, attn_mask_encoder)
            c = c + f_emb
        c = self.encoder.norm2(c)  # (B, Nc, D)

        #...target decoder
        h = self.embeddings.flow_path(x) + t_emb  # (B, Nt, D)
        h = self.decoder.norm1(h)
        for block in self.decoder.attn_blocks:
            h = block(h, c, dx, dctx, attn_mask_decoder, attn_mask_cross)  
            h = h + t_emb
        h = self.decoder.norm2(h)
        head = self.decoder.head(h)

        return head  # [B,1,T,1]

    def attention_mask(self, mask, num_heads):  # (B, N)
        attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
        attn_mask = attn_mask & attn_mask.transpose(-1, -2)  # (B, 1, N, N)
        attn_mask = attn_mask.expand(-1, num_heads, -1, -1)  # (B, n_heads, N, N)
        return attn_mask

    def cross_attention_mask(self, mask_q, mask_kv, num_heads):  # (B, N)
        attn_mask = mask_q.unsqueeze(2) & mask_kv.unsqueeze(1)  # (B, Nq, N_kv)
        attn_mask = attn_mask.unsqueeze(1)  # (B, 1, Nq, N_kv)
        attn_mask = attn_mask.expand(-1, num_heads, -1, -1)  # (B, n_heads, N_q, N_kv)
        return attn_mask


class TransformerVectorField(BaseVectorField):
    """ Operator Transformer encoder-decoder for irregularly sampled time series.
    """
    def __init__(self, config: NodePKExperimentConfig, init_dim: Optional[int] = None) -> None:

        net = _resolve_vector_field_config(config)
        self.hidden_dim = net.hidden_dim
        self.use_spectral_qkv = net.use_spectral_qkv
        self.fourier_modes = net.fourier_modes
        self.encoder_num_heads = net.encoder_num_heads
        self.decoder_num_heads = net.decoder_num_heads
        self.num_encoder_layers = net.encoder_attention_layers
        self.num_decoder_layers = net.decoder_attention_layers
        self.dropout = net.dropout

        super().__init__(
            use_covariance=False,
            cov_proj_dim=net.cov_proj_dim,
            combine_latent_mode=net.combine_latent_mode,
            zi_latent_dim=net.zi_latent_dim,
            dropout=net.dropout,
        )
        assert self.hidden_dim % self.encoder_num_heads == 0, "hidden_dim of encoder must be divisible by nhead_encoder"
        assert self.hidden_dim % self.decoder_num_heads == 0, "hidden_dim of decoder must be divisible by nhead_encoder"

        self.embeddings = nn.ModuleDict(
            dict(
                 time=TimeFourierEmbedding(
                    dim=self.hidden_dim,
                    max_freq=net.time_fourier_max_freq,
                ),
                target=MLP(
                        in_dim=4,  #  x , f(x), dose, route
                        out_dim=self.hidden_dim,
                        hidden_dim=self.hidden_dim,
                        num_layers=3,
                        activation="GELU",

                ),
                study=MLP(
                        in_dim=4,  #  x , f(x), dose, route
                        out_dim=self.hidden_dim,
                        hidden_dim=self.hidden_dim,
                        num_layers=3,
                        activation="GELU",
                ),
                norm1=LayerNorm(self.hidden_dim),
                norm2=LayerNorm(self.hidden_dim),

        ))

        self.encoder = nn.ModuleDict(
            dict(
                attn_blocks=nn.ModuleList(
                    [
                        OperatorSelfAttnBlock(
                            self.hidden_dim,
                            self.encoder_num_heads,
                            use_spectral_qkv=self.use_spectral_qkv,
                            fourier_modes=self.fourier_modes,
                            dropout=self.dropout,
                            bias=False,
                            qk_layernorm=True,
                        )
                        for _ in range(self.num_encoder_layers)
                    ]
                ),
                norm=LayerNorm(self.hidden_dim),
            )
        )

        self.decoder = nn.ModuleDict(
            dict(
                attn_blocks=nn.ModuleList(
                    [
                        OperatorCrossAttnBlock(
                            self.hidden_dim,
                            self.decoder_num_heads,
                            use_spectral_qkv=self.use_spectral_qkv,
                            fourier_modes=self.fourier_modes,
                            dropout=self.dropout,
                            bias=False,
                            qk_layernorm=True,
                        )
                        for _ in range(self.num_decoder_layers)
                    ]
                ),
                norm=LayerNorm(self.hidden_dim),
                head=MLP(
                    in_dim=self.hidden_dim,
                    out_dim=1,
                    hidden_dim=self.hidden_dim,
                    num_layers=3,
                    activation="GELU",
                    dropout=self.dropout,
                ),
            )
        )

    @typechecked
    def forward(
        self,
        x: TensorType["B", "T", 2], 
        ctx: TensorType["B", "N", 2],  # N = I * C tot number of observation points in context study
        flow_t: TensorType["B", 1],
        mask_pad_x: TensorType["B", "T"],
        mask_pad_ctx: TensorType["B", "N"],
        mask_attn_ctx: TensorType["B", "N", "N"],
        dose: TensorType["B", "T", 2],
        dose_ctx: TensorType["B", "N", 2],
    ):

        B, T, _ = x.shape
        _, N, _ = ctx.shape

        #...get trapezoidal weights for target and context grids

        dx = self.compute_trapezoidal_weights(x[..., 0], mask=mask_pad_x)       # (B, T)
        dx = dx.clamp(min=0.0)
        dx = dx / dx.sum(dim=1, keepdim=True).clamp(min=1e-8)

        dc = self.compute_trapezoidal_weights(ctx[..., 0], mask=mask_pad_ctx)   # (B, N)
        dc = dc.clamp(min=0.0)
        dc = dc / dc.sum(dim=1, keepdim=True).clamp(min=1e-8)

        #...attention masks

        attn_mask_encoder = mask_attn_ctx.unsqueeze(1).expand(
            -1, self.encoder_num_heads, -1, -1
        )  # (B, n_heads, N, N) SA encoder
        attn_mask_decoder = self.attention_mask(
            mask_pad_x, self.decoder_num_heads,
        )  # (B, n_heads, T, T) SA decoder
        attn_mask_cross = self.cross_attention_mask(
            mask_pad_x, mask_pad_ctx, self.decoder_num_heads 
        )  # (B, n_heads, T, N) CA decoder

        #...time fourier embeddings

        t = self.embeddings.time(flow_t)  # (B, T, D)
        t_emb = t.unsqueeze(1).expand(-1, N, -1)                                # (B, N, D)

        #...context encoder

        c = torch.cat([ctx, dose_ctx], dim=-1)                                  # (B, N, 4)
        c = self.embeddings.study(c)                                            # (B, N, D)
        c = self.embeddings.norm1(c + t_emb)

        for block in self.encoder.attn_blocks:
            c = block(c, dc, attn_mask_encoder)
            c = c + t_emb

        c = self.encoder.norm(c)                                                # (B, N, D)

        #...target decoder
        
        t_emb = t.unsqueeze(1).expand(-1, T, -1)                                # (B, T, D)
        h = torch.cat([x, dose], dim=-1)                                        # (B, T, 4)
        h = self.embeddings.target(h)  
        h = self.embeddings.norm2(h + t_emb)                                    # (B, T, D)

        for block in self.decoder.attn_blocks:
            h = block(h, c, dx, dc, attn_mask_decoder, attn_mask_cross)  
            h = h + t_emb

        h = self.decoder.norm(h)   
        h = self.decoder.head(h)                                                # (B, T, 1)

        return h.unsqueeze(1)  # (B, 1, T, 1)
        
    def attention_mask(self, mask, num_heads):                                  # (B, N)
        attn_mask = mask.unsqueeze(1).unsqueeze(2)                              # (B, 1, 1, N)
        attn_mask = attn_mask & attn_mask.transpose(-1, -2)                     # (B, 1, N, N)
        attn_mask = attn_mask.expand(-1, num_heads, -1, -1)                     # (B, n_heads, N, N)
        return attn_mask

    def cross_attention_mask(self, mask_q, mask_kv, num_heads):                 # (B, N)
        attn_mask = mask_q.unsqueeze(2) & mask_kv.unsqueeze(1)                  # (B, Nq, N_kv)
        attn_mask = attn_mask.unsqueeze(1)                                      # (B, 1, Nq, N_kv)
        attn_mask = attn_mask.expand(-1, num_heads, -1, -1)                     # (B, n_heads, N_q, N_kv)
        return attn_mask
