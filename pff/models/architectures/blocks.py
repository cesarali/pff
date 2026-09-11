from typing import Optional, List, Union
from torch import nn
import torch.nn.functional as F
import math
import torch
from torchtyping import TensorType

from pff.models.architectures.operator_attn import scaled_dot_product_attention

class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str = "ReLU",
        dropout: float = 0.0,
        norm: Optional[str] = None,  # "batch", "layer", or None
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            input_size = in_dim if i == 0 else hidden_dim
            output_size = out_dim if i == num_layers - 1 else hidden_dim

            layers.append(nn.Linear(input_size, output_size))

            if i < num_layers - 1:
                if norm == "batch":
                    layers.append(nn.BatchNorm1d(output_size))
                elif norm == "layer":
                    layers.append(nn.LayerNorm(output_size))

                layers.append(getattr(nn, activation)())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
    

class ResidualConcatMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str = "ReLU",
        dropout: float = 0.0,
        norm: Optional[str] = None,
        residual: bool = False  
    ):
        super().__init__()
        self.residual = residual
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        self.norm = norm
        self.dropout = dropout
        self.activation = activation

        for i in range(num_layers):
            if i == 0:
                layer_in = in_dim
            else:
                layer_in = hidden_dim + (in_dim if residual else 0)

            layer_out = out_dim if i == num_layers - 1 else hidden_dim

            block = [nn.Linear(layer_in, layer_out)]

            if i < num_layers - 1:
                if norm == "batch":
                    block.append(nn.BatchNorm1d(layer_out))
                elif norm == "layer":
                    block.append(nn.LayerNorm(layer_out))

                block.append(getattr(nn, activation)())

                if dropout > 0:
                    block.append(nn.Dropout(dropout))

            self.layers.append(nn.Sequential(*block))

        self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            for m in layer:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x: TensorType["B", "D"]) -> TensorType["B", "O"]:
        original_input = x  # [B, in_dim]
        for i, layer in enumerate(self.layers):
            if i > 0 and self.residual:
                x = torch.cat([x, original_input], dim=-1)  # concat on feature axis
            x = layer(x)
        return x


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False
    """
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class OperatorSelfAttnBlock(nn.Module):
    """Pre-LN Transformer block with self-attention and FFN.
    Args:
        dim_embd: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: FFN expansion ratio (default 4x)
        use_spectral_qkv: Use spectral convolution for QKV projection
        fourier_modes: Number of Fourier modes for spectral conv
        dropout: Dropout probability
        bias: Use bias in linear layers
        qk_layernorm: Apply layernorm to Q and K
    """
    def __init__(self, 
                 dim_embd, 
                 num_heads, 
                 mlp_ratio=4,
                 use_spectral_qkv=False, 
                 fourier_modes=None, 
                 dropout=0.1, 
                 bias=False, 
                 qk_layernorm=True):
        super().__init__()
        self.norm1 = LayerNorm(dim_embd)
        self.self_attn = OperatorSelfAttention(dim_embd, 
                                               num_heads,
                                               use_spectral_qkv=use_spectral_qkv, 
                                               fourier_modes=fourier_modes, 
                                               dropout=dropout, 
                                               bias=bias, 
                                               qk_layernorm=qk_layernorm)
        self.norm2 = LayerNorm(dim_embd)
        mlp_hidden = int(dim_embd * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim_embd, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x, dx=None, attn_mask=None):
        x = x + self.self_attn(self.norm1(x), dx, attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class OperatorCrossAttnBlock(nn.Module):
    """Pre-LN Transformer decoder block with self-attention, cross-attention, and FFN.
    Args:
        dim_embd: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: FFN expansion ratio (default 4x)
        use_spectral_qkv: Use spectral convolution for QKV projection
        fourier_modes: Number of Fourier modes for spectral conv
        dropout: Dropout probability
        bias: Use bias in linear layers
        qk_layernorm: Apply layernorm to Q and K
    """
    def __init__(self, 
                 dim_embd, 
                 num_heads, 
                 mlp_ratio=4,
                 use_spectral_qkv=False, 
                 fourier_modes=None, 
                 dropout=0.1, 
                 bias=False, 
                 qk_layernorm=True):        
        super().__init__()

        mlp_hidden = int(dim_embd * mlp_ratio)

        # Self-attention sub-block
        self.norm1 = LayerNorm(dim_embd)
        self.sa = OperatorSelfAttention(dim_embd, 
                                        num_heads,
                                        use_spectral_qkv=use_spectral_qkv, 
                                        fourier_modes=fourier_modes, 
                                        dropout=dropout, 
                                        bias=bias, 
                                        qk_layernorm=qk_layernorm)
        self.norm2 = LayerNorm(dim_embd)
        self.mlp_sa = nn.Sequential(
            nn.Linear(dim_embd, mlp_hidden), 
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim_embd),
            nn.Dropout(dropout),
        )

        # Cross-attention sub-block
        self.norm3 = LayerNorm(dim_embd)
        self.ca = OperatorCrossAttention(dim_embd, 
                                         num_heads,
                                         use_spectral_qkv=use_spectral_qkv, 
                                         fourier_modes=fourier_modes, 
                                         dropout=dropout, 
                                         bias=bias, 
                                         qk_layernorm=qk_layernorm)
        self.norm4 = LayerNorm(dim_embd)
        self.mlp_ca = nn.Sequential(
            nn.Linear(dim_embd, mlp_hidden), 
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x, y, dx=None, dy=None, sa_attn_mask=None, ca_attn_mask=None):
        x = x + self.sa(self.norm1(x), dx, attn_mask=sa_attn_mask)
        x = x + self.mlp_sa(self.norm2(x))
        x = x + self.ca(self.norm3(x), y, dy, attn_mask=ca_attn_mask)
        x = x + self.mlp_ca(self.norm4(x))
        return x


class OperatorCrossAttention(nn.Module):
    def __init__(self, n_embd, n_head, use_spectral_qkv=False, fourier_modes=64, dropout=0.1, bias=False, qk_layernorm=True):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.qk_layernorm = qk_layernorm
        self.use_spec_qkv = use_spectral_qkv

        if use_spectral_qkv:
            self.c_query = LombScargleConv1d(n_embd, n_embd, modes1=fourier_modes)
            self.c_attn = LombScargleConv1d(n_embd, 2 * n_embd, modes1=fourier_modes)
        else:
            self.c_query = nn.Linear(n_embd, n_embd, bias=bias)
            self.c_attn = nn.Linear(n_embd, 2 * n_embd, bias=bias)

        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)

        if qk_layernorm:
            self.q_layernorm = LayerNorm(n_embd // n_head, bias=bias)
            self.k_layernorm = LayerNorm(n_embd // n_head, bias=bias)

        self.query_chunk_size = None  # set externally to enable chunked-query inference

    def forward(self, x, y, dx=None, attn_mask=None):
        """
        x: (B, N, D)
        dx: (B, N) quadrature weights for query tokens (sorted to match x order).
        attn_mask: optional mask for attention (same semantics as scaled_dot_product_attention)
        """
        B, N, D = x.size()
        _, M, _ = y.size()

        if self.use_spec_qkv:
            # SpectralConv1d expects (B, D, N), so transpose before and after
            q = self.c_query(x.transpose(1, 2)).transpose(1, 2)  # (B, N, D)
            kv = self.c_attn(y.transpose(1, 2)).transpose(1, 2)  # (B, M, 2*D)
        else:
            q = self.c_query(x)  # (B, N, D)
            kv = self.c_attn(y)  # (B, M, 2*D)

        k, v = kv.split(self.n_embd, dim=2)
        q = q.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, N, hs)
        k = k.view(B, M, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, M, hs)
        v = v.view(B, M, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, M, hs)

        if dx is not None:
            dx = dx.unsqueeze(1)  # (B, 1, N)

        if self.qk_layernorm:
            q = self.q_layernorm(q)
            k = self.k_layernorm(k)

        a = scaled_dot_product_attention(q, k, v, dx, attn_mask=attn_mask, dropout_p=self.dropout,
                                         query_chunk_size=self.query_chunk_size)
        a = a.transpose(1, 2).contiguous().view(B, N, D)
        a = self.resid_dropout(self.c_proj(a))
        return a


class OperatorSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, use_spectral_qkv=False, fourier_modes=64, dropout=0.1, bias=False, qk_layernorm=True):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.qk_layernorm = qk_layernorm
        self.use_spec_qkv = use_spectral_qkv

        if use_spectral_qkv:
            self.c_attn = LombScargleConv1d(n_embd, 3 * n_embd, modes1=fourier_modes)
        else:
            self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)

        if qk_layernorm:
            self.q_layernorm = LayerNorm(n_embd // n_head, bias=bias)
            self.k_layernorm = LayerNorm(n_embd // n_head, bias=bias)

        self.query_chunk_size = None  # set externally to enable chunked-query inference

    def forward(self, x, dx=None, attn_mask=None):
        """
        x: (B, N, D)
        dx: (B, N) quadrature weights for tokens (sorted to match x order).
        attn_mask: optional mask for attention (same semantics as scaled_dot_product_attention)
        """
        B, N, D = x.size()

        if self.use_spec_qkv:
            # SpectralConv1d expects (B, D, N), so transpose before and after
            qkv = self.c_attn(x.transpose(1, 2)).transpose(1, 2)  # (B, N, 3*D)
        else:
            qkv = self.c_attn(x)  # (B, N, 3*D)

        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, N, hs)
        q = q.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        v = v.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)

        if dx is not None:
            dx = dx.unsqueeze(1)  # (B, 1, N)

        if self.qk_layernorm:
            q = self.q_layernorm(q)
            k = self.k_layernorm(k)

        a = scaled_dot_product_attention(q, k, v, dx, attn_mask=attn_mask, dropout_p=self.dropout,
                                         query_chunk_size=self.query_chunk_size)
        a = a.transpose(1, 2).contiguous().view(B, N, D)
        a = self.resid_dropout(self.c_proj(a))
        return a


class SelfAttnBlock(nn.Module):
    def __init__(self, dim_embd, num_heads, dropout=0.0, bias=False, qk_layernorm=True):
        super().__init__()
        self.self_attn = SelfAttention(dim_embd, num_heads, dropout)
        self.norm1 = LayerNorm(dim_embd)
        self.mlp = nn.Sequential(nn.Linear(dim_embd, dim_embd), 
                                 nn.GELU(), 
                                 nn.Linear(dim_embd, dim_embd))
        self.norm2 = LayerNorm(dim_embd)
    def forward(self, x, attn_mask=None):
        x = x + self.self_attn(x, attn_mask=attn_mask)
        x = x + self.mlp(self.norm1(x))
        return self.norm2(x)


class CrossAttnBlock(nn.Module):
    def __init__(self, dim_embd, num_heads, dropout=0.0, bias=False, qk_layernorm=True):
        super().__init__()

        self.sa = SelfAttention(dim_embd, num_heads, dropout)
        self.norm1 = LayerNorm(dim_embd)
        self.mlp_sa = nn.Sequential(nn.Linear(dim_embd, dim_embd), 
                                    nn.GELU(), 
                                    nn.Linear(dim_embd, dim_embd))

        self.ca = CrossAttention(dim_embd, num_heads, dropout)
        self.norm2 = LayerNorm(dim_embd)
        self.mlp_ca = nn.Sequential(nn.Linear(dim_embd, dim_embd), 
                                    nn.GELU(), 
                                    nn.Linear(dim_embd, dim_embd))
        self.norm = LayerNorm(dim_embd)

    def forward(self, x, y, sa_attn_mask=None, ca_attn_mask=None):
        x = x + self.sa(x, attn_mask=sa_attn_mask)
        x = x + self.mlp_sa(self.norm1(x))
        x = x + self.ca(x, y, attn_mask=ca_attn_mask)
        x = x + self.mlp_ca(self.norm2(x))
        return self.norm(x)


class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.0, bias=True, qk_layernorm=True):
        super().__init__()

        assert n_embd % n_head == 0

        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.qk_layernorm = qk_layernorm
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        if qk_layernorm:
            self.q_layernorm = LayerNorm(n_embd // n_head, bias=bias)
            self.k_layernorm = LayerNorm(n_embd // n_head, bias=bias)

    def forward(self, x, attn_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.qk_layernorm:
            q = self.q_layernorm(q)
            k = self.k_layernorm(k)

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)

        if self.flash: # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout, is_causal=False)
        else:
            raise NotImplementedError
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.resid_dropout(self.c_proj(y))
        return y


class CrossAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.0, bias=True, qk_layernorm=True):
        super().__init__()

        assert n_embd % n_head == 0

        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.qk_layernorm = qk_layernorm
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        self.c_query = nn.Linear(n_embd, n_embd, bias=bias)
        self.c_attn = nn.Linear(n_embd, 2 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        if qk_layernorm:
            self.q_layernorm = LayerNorm(n_embd // n_head, bias=bias)
            self.k_layernorm = LayerNorm(n_embd // n_head, bias=bias)

    def forward(self, x, z, attn_mask=None):
        B, T, C = x.size()  # batch size, query sequence length, embedding dim
        _, S, _ = z.size()  # S = key/value sequence length (may differ from T)

        q = self.c_query(x)
        k, v = self.c_attn(z).split(self.n_embd, dim=2)
        
        # q uses T (query length), k and v use S (key/value length)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, S, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, S, hs)
        v = v.view(B, S, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, S, hs)

        if self.qk_layernorm:
            q = self.q_layernorm(q)
            k = self.k_layernorm(k)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.dropout, is_causal=False
            )
        else:
            raise NotImplementedError

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # output has query length T
        y = self.resid_dropout(self.c_proj(y))
        return y


class TimeFourierEmbedding(nn.Module):
    """
    Turn a scalar t∈[0,1] into a D-dim Fourier feature vector:
      [ sin(t * ω₁), …, sin(t * ω_{D/2}), cos(t * ω₁), …, cos(t * ω_{D/2}) ]
    with frequencies ω log-spaced from 1 to max_freq.
    """
    def __init__(self, dim: int, max_freq: float = 10.0):
        super().__init__()
        half = dim // 2
        inv_freq = 1.0 / ( max_freq ** (torch.arange(half).float() / (half - 1)) )
        self.register_buffer("inv_freq", inv_freq)   # (D/2,)

    def forward(self, t: torch.Tensor):
        # t: (B, 1) or (B,) → ensure (B,1)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        # x = t * ω_i  →  (B, D/2)
        x = t * self.inv_freq.unsqueeze(0)
        emb = torch.cat([x.sin(), x.cos()], dim=-1)  # (B, D)
        return emb                                   # (B, D)


class LearnableFourierEmbedding(nn.Module):
    """
    Learnable Fourier features for particle clouds (B, D, M), with grouping.

    Input:
        x     : Tensor [B, D, M]  (multi-D point features per particle)
        mask  : Optional Bool Tensor [B, D] where False marks padded points (zeroed in output)

    Output:
        pe    : Tensor [B, D, dim_out]

    Args:
        feat_split          : list of ints summing to dim, partitions last-dim features into groups
                              e.g. [1,1,2] for [tau | ] if you concatenated t upstream
        dim_fourier_hidden  : list of per-group Fourier dims D_g (each even), sums to total D_total
        gamma               : float or list per group (Gaussian-kernel init width for W_r)
        dim_hidden          : hidden size for the MLP
        dim_out             : final embedding size
    """
    def __init__(
        self,
        feat_split: List[int],
        dim_fourier_hidden: List[int],
        dim_hidden: int,
        dim_out: int,
        gamma: Union[float, List[float]] = 1.0,
    ):
        super().__init__()
        # groups over input dims
        self.feat_split = feat_split
        self.num_groups = len(feat_split)

        assert all(dg % 2 == 0 for dg in dim_fourier_hidden), "each group Fourier dim must be even."
        self.dim_fourier = dim_fourier_hidden
        self.total_fourier = sum(dim_fourier_hidden)

        # gamma per group
        if isinstance(gamma, (float, int)):
            gammas = [float(gamma)] * self.num_groups
        else:
            assert len(gamma) == self.num_groups, "gamma list size must equal num_groups."
            gammas = [float(g) for g in gamma]

        self.gammas = gammas

        # one W_r per group: [D_g/2, group_input_dim]
        self.Wr = nn.ParameterList()

        for gin, Dg, g in zip(self.feat_split, self.dim_fourier, self.gammas):
            Wr = nn.Parameter(torch.empty(Dg // 2, gin))
            nn.init.normal_(Wr, mean=0.0, std=(1.0 / g))   # kernel-aware init
            self.Wr.append(Wr)

        self.mlp = nn.Sequential(nn.Linear(self.total_fourier, dim_hidden),
                                 nn.GELU(),
                                 nn.Linear(dim_hidden, dim_hidden),
                                 nn.GELU(),
                                 nn.Linear(dim_hidden, dim_out),
                                )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x    : [B, D, M]
        mask : [B, D] (bool). If provided, outputs at padded positions are zeroed.

        returns: [B, D, dim_out]
        """
        assert x.dim() == 3 and x.size(-1) == sum(self.feat_split), "x shape must be [B,D,M] with M=sum(feat_split)."
        chunks = torch.split(x, self.feat_split, dim=-1)  # list of [B, D, gin]

        feats = []
        for chunk, Wr, Dg in zip(chunks, self.Wr, self.dim_fourier):
            proj = F.linear(chunk, Wr)  # [B, D, Dg/2]
            scale = math.sqrt(Dg / 2.0)
            cos = torch.cos(proj)
            sin = torch.sin(proj)
            feats.append(torch.cat([cos, sin], dim=-1) / scale)  # [B, D, Dg]

        ff = torch.cat(feats, dim=-1)  # [B, D, total_fourier]
        out = self.mlp(ff)             # [B, D, dim_out]

        if mask is not None:
            out = out * mask.to(out.dtype)

        return out


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        # Handle variable sequence lengths: use min of modes1 and available freq bins
        n_freq = x_ft.size(-1)
        modes_to_use = min(self.modes1, n_freq)
        
        out_ft = torch.zeros(batchsize, self.out_channels, n_freq, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :modes_to_use] = self.compl_mul1d(
            x_ft[:, :, :modes_to_use], 
            self.weights1[:, :, :modes_to_use]
        )

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class LombScargleConv1d(nn.Module):
    """
    1D Spectral convolution layer for irregularly sampled data using 
    Non-Uniform DFT (Lomb-Scargle inspired approach).
    
    Unlike standard SpectralConv1d which uses FFT (assumes uniform grid),
    this computes the DFT explicitly at arbitrary sample locations:
        Forward:  X(ω_k) = Σ_n x(t_n) * exp(-2πi * ω_k * t_n)
        Inverse:  x(t_n) = Σ_k X(ω_k) * exp(2πi * ω_k * t_n)
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels  
        modes1: Number of Fourier modes (frequencies) to use
        freq_scale: Scale factor for frequency range (default 1.0 corresponds to [0, modes1])
    """
    def __init__(self, in_channels, out_channels, modes1, freq_scale=1.0):
        super(LombScargleConv1d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.freq_scale = freq_scale
        
        # Learnable complex weights for each frequency mode
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )
        
        # Fixed frequency grid (can also be made learnable)
        # Frequencies from 0 to modes1-1, scaled
        freqs = torch.arange(modes1, dtype=torch.float32) * freq_scale
        self.register_buffer('freqs', freqs)
    
    def compl_mul1d(self, input, weights):
        """Complex multiplication: (B, C_in, modes) x (C_in, C_out, modes) -> (B, C_out, modes)"""
        return torch.einsum("bim,iom->bom", input, weights)
    
    def forward(self, x, t=None):
        """
        Args:
            x: Input tensor of shape (B, C, N) - values at sample points
            t: Sample locations of shape (B, N) or (N,), assumed in [0, 1].
               If None, assumes uniform grid (falls back to standard FFT behavior).
        
        Returns:
            y: Output tensor of shape (B, C_out, N)
        """
        B, C, N = x.shape
        device = x.device
        
        if t is None:
            # Fall back to uniform grid assumption
            t = torch.linspace(0, 1, N, device=device).unsqueeze(0).expand(B, -1)
        elif t.dim() == 1:
            t = t.unsqueeze(0).expand(B, -1)
        
        # Compute Non-Uniform DFT basis
        # t: (B, N), freqs: (modes,)
        # phase = 2π * freqs * t -> (B, N, modes)
        phase = 2 * math.pi * t.unsqueeze(-1) * self.freqs.unsqueeze(0).unsqueeze(0)  # (B, N, modes)
        
        # DFT matrix: exp(-i * phase) for forward transform
        dft_matrix = torch.exp(-1j * phase)  # (B, N, modes)
        
        # Forward NUDFT: X(ω) = Σ_n x(t_n) * exp(-2πi * ω * t_n)
        # x: (B, C, N), dft_matrix: (B, N, modes) -> x_ft: (B, C, modes)
        x_ft = torch.einsum("bcn,bnm->bcm", x.to(torch.cfloat), dft_matrix) / N
        
        # Apply learnable spectral weights
        # x_ft: (B, C_in, modes), weights: (C_in, C_out, modes) -> out_ft: (B, C_out, modes)
        out_ft = self.compl_mul1d(x_ft, self.weights)
        
        # Inverse NUDFT: x(t) = Σ_k X(ω_k) * exp(2πi * ω_k * t)
        idft_matrix = torch.exp(1j * phase)  # (B, N, modes) - conjugate of forward
        
        # out_ft: (B, C_out, modes), idft_matrix: (B, N, modes) -> y: (B, C_out, N)
        y = torch.einsum("bcm,bnm->bcn", out_ft, idft_matrix)
        
        return y.real