
import math
import torch
import torch.nn.functional as F

def softmax(x, dx=None, dim=-1, eps=1e-8):
    """ Softmax with trapezoidal quadrature normalization (Eq. 59 in arXiv:2406.06486).
        x: attention scores (B, nh, N, M)
        dx: (B, 1, M) quadrature weights, or None for standard softmax
        eps: small value for numerical stability
    """
    x_max = x.max(dim=dim, keepdim=True).values
    all_inf = torch.isinf(x_max) & (x_max < 0)  # True where entire row is -inf
    x_max = torch.where(all_inf, torch.zeros_like(x_max), x_max) # Nan -> 0 for fully masked rows
    x_exp = (x - x_max).exp()

    if dx is None:
        x_exp_sum = x_exp.sum(dim=dim, keepdim=True)
    else:
        dx = dx.unsqueeze(2)  # (B, 1, 1, S)
        x_exp_sum = 0.5 * ((x_exp[..., 1:] + x_exp[..., :-1]) * dx[..., 1:]).sum(dim=dim, keepdim=True)
    result = x_exp / x_exp_sum.clamp(min=eps)
    result = torch.where(all_inf, torch.zeros_like(result), result)
    return result


def scaled_dot_product_attention(query, key, value, dx=None, attn_mask=None, dropout_p=0.0, scale=None, query_chunk_size=None) -> torch.Tensor:
    """ Memory efficient self-attention following trapezoidal approximations for irregular grids.
        query: (B, nh, N, hs)
        key, value: (B, nh, M, hs)
        dx: (B, 1, M) or None. Uses trapezoidal rule weights in attention computation.
            dx[..., k] = product of |x_k^i - x_{k-1}^i| for i=1,...,d volume element of key/value data
        attn_mask: optional mask for attention (B, nh, N, M)
        dropout_p: dropout probability on attention weights
        scale: optional scaling factor for attention scores (default: 1/sqrt(hs))
        query_chunk_size: if set, process Q in chunks of this size to limit peak memory.
            The full [B, nh, N, M] score matrix is never materialised; at most
            [B, nh, query_chunk_size, M] is live at once.  Result is identical to the
            unchunked path because the trapezoidal softmax normaliser is per-row independent.

        Notes:

        A(v)_j = 0.5 * sum_{k=2}^N (p(k;v,j)*V*v_k + p(k-1;v,j)*V*v_{k-1}) * dx_k  (Eq. 58 arXiv:2406.06486).
        Memory-efficient reformulation using weighted matmul,
        reduces memory from O(B*nh*N*M*hs) to O(B*nh*M*hs):
        Split into two sums and reindex to get standard matmul form:
          Term1: sum_k p[j,k] * v[k] * dx[k]     (for k=1..M-1)
          Term2: sum_k p[j,k] * v[k] * dx[k+1]   (for k=0..M-2)
        Define weights: w1[k] = dx[k], w2[k] = dx[k+1] (with zero padding)
        Result = 0.5 * P @ (V * (w1 + w2))
        w1[k] = dx[k] for k=1..M-1, w1[0]=0 (dx already has this structure)
        w2[k] = dx[k+1] for k=0..M-2, w2[M-1]=0

    """
    N, M = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    if query_chunk_size is not None and query_chunk_size < N:
        # Chunked query path: avoids materialising the full [B, nh, N, M] score matrix.
        # Pre-compute dx-weighted values once — shared across all query chunks.
        if dx is not None:
            dx_next = torch.cat([dx[..., 1:], torch.zeros_like(dx[..., :1])], dim=-1)  # (B, 1, M)
            w = 0.5 * (dx + dx_next).unsqueeze(-1)  # (B, 1, M, 1)
            value_w = value * w  # (B, nh, M, hs)
        else:
            value_w = value

        outputs = []
        for q_start in range(0, N, query_chunk_size):
            q_end = min(q_start + query_chunk_size, N)
            q_c = query[:, :, q_start:q_end, :]  # (B, nh, qc, hs)
            qc = q_end - q_start

            if attn_mask is not None:
                mask_c = attn_mask[:, :, q_start:q_end, :]
                if mask_c.dtype == torch.bool:
                    bias_c = torch.zeros_like(mask_c, dtype=query.dtype)
                    bias_c.masked_fill_(mask_c.logical_not(), float("-inf"))
                else:
                    bias_c = mask_c
            else:
                bias_c = torch.zeros(qc, M, dtype=query.dtype, device=query.device)

            score_c = q_c @ key.transpose(-2, -1) * scale_factor  # (B, nh, qc, M)
            score_c = score_c + bias_c
            score_c = softmax(score_c, dx=dx, dim=-1)  # (B, nh, qc, M)
            if dropout_p > 0.0:
                score_c = F.dropout(score_c, p=dropout_p)
            outputs.append(score_c @ value_w)  # (B, nh, qc, hs)

        return torch.cat(outputs, dim=2)  # (B, nh, N, hs)

    # ── Original single-pass path (unchanged) ────────────────────────────────
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias = torch.zeros_like(attn_mask, dtype=query.dtype)
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask
    else:
        attn_bias = torch.zeros(N, M, dtype=query.dtype, device=query.device)

    attn_score = query @ key.transpose(-2, -1) * scale_factor  # (B, nh, N, M)
    attn_score += attn_bias

    attn_score = softmax(attn_score, dx=dx, dim=-1)  # (B, nh, N, M) attention weights p(k; v, j)

    if dropout_p > 0.0:
        attn_score = F.dropout(attn_score, p=dropout_p)

    if dx is not None:
        # reweight values by trapezoidal weights
        dx_next = torch.cat([dx[..., 1:], torch.zeros_like(dx[..., :1])], dim=-1)  # (B, 1, M)
        w = 0.5 * (dx + dx_next).unsqueeze(-1)  # (B, 1, M, 1)
        value = value * w   # (B, nh, M, hs)

    return attn_score @ value  # (B, nh, N, hs)