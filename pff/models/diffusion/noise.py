from typing import Union

import torch
import torch.nn as nn
from torchtyping import TensorType


class Normal(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.dim = dim

    def forward(self, *shape, **kwargs):
        return torch.randn(*shape, self.dim)

    def covariance(self, **kwargs):
        return torch.eye(self.dim)


class Wiener(nn.Module):
    """
    Wiener process / Brownian motion.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(
        self,
        *shape,
        t: Union[TensorType["seq_len"], TensorType[..., "seq_len", 1], None] = None,
        **kwargs,
    ) -> Union[TensorType["seq_len"], TensorType[..., "seq_len", "dim"]]:
        if t is None and shape:
            # Allow callers to pass t as the first positional argument.
            if torch.is_tensor(shape[0]):
                t = shape[0]
        if t is None:
            raise ValueError("Wiener noise requires a time grid `t`.")

        one_dimensional = len(t.shape) == 1

        if one_dimensional:
            t = t.unsqueeze(-1)
        t = t.repeat_interleave(self.dim, dim=-1)

        dt = torch.diff(t, dim=-2, prepend=torch.zeros_like(t[..., :1, :]).to(t))
        dw = torch.randn_like(dt) * dt.clamp(1e-5).sqrt()
        w = dw.cumsum(dim=-2)

        if one_dimensional and self.dim == 1:
            w = w.squeeze(-1)
        return w

    def covariance(
        self,
        t: TensorType[..., "seq_len", 1],
        diag_epsilon: float = 1e-4,
        **kwargs,
    ) -> TensorType[..., "seq_len", "seq_len"]:
        t = t.squeeze(-1)
        diag = torch.eye(t.shape[-1]).to(t) * diag_epsilon
        cov = torch.minimum(t.unsqueeze(-1), t.unsqueeze(-2))
        return cov + diag


class OrnsteinUhlenbeck(nn.Module):
    """
    Ornstein-Uhlenbeck process.
    """

    def __init__(self, dim: int, variance: float = 1.0, length_scale: float = 1.0, epsilon: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.var = variance
        self.L = length_scale
        self.epsilon = epsilon

    def forward(
        self,
        *args,
        t: TensorType[..., "seq_len", 1],
        **kwargs,
    ) -> TensorType[..., "seq_len", "dim"]:
        delta = torch.diff(t, dim=-2, prepend=torch.zeros_like(t[..., :1, :]))
        coeff = torch.exp(-self.theta * delta)

        sample = []

        x = torch.randn(*t.shape[:-2], 1, self.dim).to(t)
        for i in range(coeff.shape[-2]):
            z = torch.randn(*t.shape[:-2], 1, self.dim).to(t)
            c = coeff[..., i, None, :]
            x = c * x + torch.sqrt(1 - c**2) * z
            sample.append(x)

        sample = torch.cat(sample, dim=-2)
        return sample

    def covariance(
        self,
        t: TensorType[..., "seq_len", 1],
        **kwargs,
    ) -> TensorType[..., "seq_len", "seq_len"]:
        t = t.squeeze(-1)
        distance = t.unsqueeze(-1) - t.unsqueeze(-2)
        diag = torch.eye(t.shape[-1]).to(t) * self.epsilon
        cov = self.var * torch.exp(-distance.abs() / self.L)
        return cov + diag

    def covariance_cholesky(
        self, t: TensorType[..., "seq_len", 1]
    ) -> TensorType[..., "seq_len", "seq_len"]:
        return torch.linalg.cholesky(self.covariance(t))

    def covariance_inverse(
        self, t: TensorType[..., "seq_len", 1]
    ) -> TensorType[..., "seq_len", "seq_len"]:
        return torch.linalg.inv(self.covariance(t))


class GaussianProcess(nn.Module):
    """
    Gaussian random field for one-dimensional (temporal) data.
    """

    def __init__(self, dim: int, variance: float = 1.0, length_scale: float = 1.0, epsilon: float = 1e-8, transform: str = "softplus"):
        super().__init__()
        self.dim = dim
        self.var = variance
        self.L = length_scale
        self.epsilon = epsilon
        self.transform = transform

    def forward(
        self,
        *args,
        t: TensorType[..., "N", 1],
        device = 'cpu',
        **kwargs,
    ) -> TensorType[..., "N", "dim"]:
        # If N is very large this could become slow
        # In that case, consider using sparse GP
        L = self.covariance_cholesky(t)
        e = torch.randn(*t.shape[:-1], self.dim).to(t)
        f = (L @ e).to(device)
        return self._forward_transform(f)

    def _forward_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Map unconstrained GP output → positive space."""
        if self.transform == "exp":
            return torch.exp(x)
        if self.transform == "softplus":
            return torch.nn.functional.softplus(x)
        return x

    def covariance(
        self,
        t: TensorType[..., "N", 1],
        **kwargs,
    ) -> TensorType[..., "N", "N"]:
        if t.shape[-1] != 1 or len(t.shape) < 2:
            t = t.unsqueeze(-1)
        distance = t - t.transpose(-1, -2)
        diag = torch.eye(t.shape[-2]).to(t) * self.epsilon
        cov = self.var * torch.exp(-torch.square(distance / self.L))
        return cov + diag

    def covariance_cholesky(self, t: TensorType[..., "N", 1]) -> TensorType[..., "N", "N"]:
        return torch.linalg.cholesky(self.covariance(t))

    def covariance_inverse(self, t: TensorType[..., "N", 1]) -> TensorType[..., "N", "N"]:
        return torch.linalg.inv(self.covariance(t))


class GaussianProcessRegression(nn.Module):
    """
    GP regression that conditions on a (variable-length) conditional observations
    and draws a posterior sample over the full time grid.

    Parameters
    ----------
    variance     : RBF kernel output scale  σ²
    length_scale : RBF length scale L
    epsilon       : diagonal regularisation added to the prior covariance
                   for Cholesky numerical stability
    transform     : "none" | "exp" | "softplus" applied to the output to ensure positive concentrations

    """

    def __init__(
        self,
        variance: float = 1.0,
        length_scale: float = 1.0,
        epsilon: float = 1e-7, 
        transform: str = "softplus", 
    ):
        super().__init__()
        self.var = variance
        self.L = length_scale
        self.epsilon = epsilon
        self.transform = transform

    def _forward_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Map unconstrained GP output → positive space."""
        if self.transform == "exp":
            return torch.exp(x)
        if self.transform == "softplus":
            return torch.nn.functional.softplus(x)
        return x

    def _rbf(
        self,
        t1: TensorType["B", "N1", 1],
        t2: TensorType["B", "N2", 1],
        ) -> TensorType["B", "N1", "N2"]:
        """Batched RBF (squared-exponential) kernel."""
        dist = t1 - t2.transpose(-1, -2)          # (B, N1, N2)
        return self.var * torch.exp(-torch.square(dist / self.L))

    def sample_prior(
        self,
        t_query:    TensorType["B", 1, "N", 1],
        mask_query: Union[TensorType["B", 1, "N"], None] = None,
        ) -> TensorType["B", 1, "N", 1]:
        """
        Sample from the GP prior (no conditioning data).

        Parameters
        ----------
        t_query    : (B, 1, N, 1)  query time points, zero-padded.
        mask_query : (B, *, N) bool, optional — True = real time point.
                     Inferred from t_query > 0 when not supplied.

        Returns
        -------
        x_query : (B, 1, N, 1)
            GP prior sample at each t_query point.
            Values at padded slots are zeroed out.
        """
        B, _, N, _ = t_query.shape
        device, dtype = t_query.device, t_query.dtype
        tq = t_query.reshape(B, N)                          # (B, N)
        mask_query = mask_query.reshape(B, N) if mask_query  is not None else torch.ones_like(tq)
        K_qq = self._rbf(tq.unsqueeze(-1), tq.unsqueeze(-1))   # (B, N, N)
        diag_eps_q = torch.where(
            mask_query,
            tq.new_full((B, N), self.epsilon),
            tq.new_ones((B, N)),
        )
        K_qq = K_qq + torch.diag_embed(diag_eps_q)             # (B, N, N)
        L_chol  = torch.linalg.cholesky(K_qq)                  # (B, N, N)
        z       = torch.randn(B, N, 1, device=device, dtype=dtype)
        x_query = (L_chol @ z).squeeze(-1)                     # (B, N)
        x_query = self._forward_transform(x_query) * mask_query
        return x_query.reshape(B, 1, N, 1)

    # ------------------------------------------------------------------
    def forward(
        self,
        t_query:    TensorType["B", 1, "N", 1],
        t_cond:     Union[TensorType["B", 1, "M", 1], None] = None,
        x_cond:     Union[TensorType["B", 1, "M", 1], None] = None,
        mask_query: Union[TensorType["B", 1, "N"], None] = None,
        mask_cond:  Union[TensorType["B", 1, "M"], None] = None,
        ) -> TensorType["B", 1, "N", 1]:
        """
        Parameters
        ----------
        t_query    : (B, 1, N, 1)  query time points.
        t_cond     : (B, 1, M, 1)  conditioning time points. If None (along
                     with x_cond), samples from the GP prior.
        x_cond     : (B, 1, M, 1)  observed values at t_cond. If None (along
                     with t_cond), samples from the GP prior.
        mask_query : (B, *, N) bool, zero-padded query points
        mask_cond  : (B, *, M) bool, zero-padded conditioning points
        Returns
        -------
        x_query : (B, 1, N, 1)
            GP posterior sample (or prior sample when no conditioning data)
            at each t_query point. Values at padded future slots are zeroed out.
        """

        B, _, N, _ = t_query.shape
        device, dtype = t_query.device, t_query.dtype

        if t_cond is None or x_cond is None:
            return self.sample_prior(t_query, mask_query=mask_query)

        M = t_cond.shape[2]
        tq = t_query.reshape(B, N)
        tc = t_cond.reshape(B, M)
        xc = x_cond.reshape(B, M)
        mask_cond = mask_cond.reshape(B, M) if mask_cond is not None else torch.ones_like(tc)
        mask_query = mask_query.reshape(B, N) if mask_query is not None else torch.ones_like(tq)

        #...kernel matrices

        K_cc = self._rbf(tc.unsqueeze(-1), tc.unsqueeze(-1))               # (B, M, M)
        K_qc = self._rbf(tq.unsqueeze(-1), tc.unsqueeze(-1))               # (B, N, M)
        K_qq = self._rbf(tq.unsqueeze(-1), tq.unsqueeze(-1))               # (B, N, N)

        #...regularize K's

        diag_eps_c = torch.where(
            mask_cond,
            tc.new_full((B, M), self.epsilon),
            tc.new_full((B, M), 1e+10),
        )
        diag_eps_q = torch.where(
            mask_query,
            tq.new_full((B, N), self.epsilon),
            tq.new_ones((B, N)),
        )
        K_cc = K_cc + torch.diag_embed(diag_eps_c)                  # (B, N, N)
        K_qq = K_qq + torch.diag_embed(diag_eps_q)                  # (B, N, N)
        has_cond_data = mask_cond.any(dim=-1)                       # (B,)

        #...GP posterior

        # alpha = K_cc^{-1} x_cond                                  (B, M, 1)
        alpha = torch.linalg.solve(K_cc, xc.unsqueeze(-1))
        post_mean = (K_qc @ alpha).squeeze(-1)                  #   (B, N)

        # Posterior cov   Σ_q = K_qq − K_qc K_cc^{-1} K_cq          (B, N, N)
        V = torch.linalg.solve(K_cc, K_qc.transpose(-1, -2))    #   (B, M, N)
        post_cov = K_qq - K_qc @ V                              #   (B, N, N)

        #...posterior sample
        L_chol   = torch.linalg.cholesky(post_cov)                   # (B, N, N)
        z        = torch.randn(B, N, 1, device=device, dtype=dtype)
        x_query = post_mean + (L_chol @ z).squeeze(-1)               # (B, N)

        #...prior fallback: batch elements with no observations → sample from prior GP
        if not has_cond_data.all():
            no_cond   = ~has_cond_data
            L_prior  = torch.linalg.cholesky(K_qq[no_cond])           # (n, N, N)
            z_prior  = torch.randn(no_cond.sum(), N, 1, device=device, dtype=dtype)
            x_query = x_query.clone()
            x_query[no_cond] = (L_prior @ z_prior).squeeze(-1)

        x_query = self._forward_transform(x_query) * mask_query 
        
        return x_query.reshape(B, 1, N, 1)


class WhiteNoiseProcess(nn.Module):
    """
    White-noise process that mirrors the GaussianProcessRegression interface.
    The white-noise kernel is K(t, t') = σ² δ(t == t'),

    Parameters
    ----------
    variance  : output variance σ²
    epsilon    : small diagonal added to ensure positivity of padded-slot
                masking (matches GaussianProcessRegression convention)
    transform : "none" | "exp" | "softplus" — maps unconstrained samples
                to positive space (use "none" for unrestricted outputs)
    """

    def __init__(
        self,
        epsilon:    float = 1e-7,
        transform: str   = "softplus",
    ):
        super().__init__()
        self.epsilon    = epsilon
        self.transform = transform

    def _forward_transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform == "exp":
            return torch.exp(x)
        if self.transform == "softplus":
            return torch.nn.functional.softplus(x)
        return x

    def _masked_mean_std(self, x: torch.Tensor, m: torch.Tensor, eps: float):
        valid = m.bool()
        count = valid.sum(dim=-1, keepdim=True).clamp_min(1)
        mean = x.masked_fill(~valid, 0).sum(dim=-1, keepdim=True) / count
        var = ((x - mean).pow(2)).masked_fill(~valid, 0).sum(dim=-1, keepdim=True) / count
        return mean, var.clamp_min(eps**2).sqrt()

    def sample_prior(
        self,
        t_query:    TensorType["B", 1, "N", 1],
        mask_query: Union[TensorType["B", 1, "N"], None] = None,
        ) -> TensorType["B", 1, "N", 1]:
        """
        Sample independently from N(0, σ²) at each query time point.

        Parameters
        ----------
        t_query    : (B, 1, N, 1)  query time points, zero-padded.
        mask_query : (B, *, N) bool, optional — True = real time point.
                     Inferred from t_query > 0 when not supplied.

        Returns
        -------
        x_query : (B, 1, N, 1)
            White-noise sample; padded slots are zeroed out.
        """
        B, _, N, _ = t_query.shape
        device, dtype = t_query.device, t_query.dtype
        tq = t_query.reshape(B, N)                              # (B, N)
        mask_query = mask_query.reshape(B, N) if  mask_query is not None else torch.ones_like(tq)     # (B, N)
        x_query = torch.randn(B, N, device=device, dtype=dtype) 
        x_query = self._forward_transform(x_query) * mask_query   
        return x_query.reshape(B, 1, N, 1)

    # ------------------------------------------------------------------
    def forward(
        self,
        t_query:    TensorType["B", 1, "N", 1],
        t_cond:     Union[TensorType["B", 1, "M", 1], None] = None,
        x_cond:     Union[TensorType["B", 1, "M", 1], None] = None,
        mask_query: Union[TensorType["B", 1, "N"], None] = None,
        mask_cond:  Union[TensorType["B", 1, "M"], None] = None,
        ) -> TensorType["B", 1, "N", 1]:
        """
        Sample white noise centred on the empirical mean of the conditioning

        Parameters
        ----------
        t_query    : (B, 1, N, 1)  query time points.
        t_cond     : (B, 1, M, 1)  conditioning time points (only used to
                     infer mask when mask_cond is None).
        x_cond     : (B, 1, M, 1)  conditioning observations in data space.
        mask_query : (B, *, N) bool, optional.
        mask_cond  : (B, *, M) bool, optional — True = real conditioning point.
                     Inferred from t_cond > 0 when not supplied.
        Returns
        -------
        x_query : (B, 1, N, 1)
        """
        B, _, N, _ = t_query.shape
        device, dtype = t_query.device, t_query.dtype
        if t_cond is None or x_cond is None:
            return self.sample_prior(t_query, mask_query=mask_query)

        M = t_cond.shape[2]
        tq = t_query.reshape(B, N)
        tc = t_cond.reshape(B, M)
        xc = x_cond.reshape(B, M)
        mask_cond = mask_cond.reshape(B, M) if mask_cond is not None else torch.ones_like(tc)
        mask_query = mask_query.reshape(B, N) if mask_query is not None else torch.ones_like(tq)

        mu_cond, std_cond = self._masked_mean_std(xc, mask_cond, 1e-7)  # (B, 1), (B, 1)         
        z = torch.randn(B, N, device=device, dtype=dtype)
        x_query  = mu_cond + z * std_cond # (B, N)

        # No observations 
        has_cond_data = mask_cond.any(dim=-1)                   
        if not has_cond_data.all():
            x_query[~has_cond_data] = z[~has_cond_data]

        # Exactly 1 observation: unit std 
        has_1_obs   = (mask_cond.sum(dim=-1) == 1)                  
        single_val  = (xc * mask_cond.float()).sum(dim=-1, keepdim=True) 
        if has_1_obs.any():
            x_query[has_1_obs] = single_val[has_1_obs] + z[has_1_obs]

        x_query = self._forward_transform(x_query) * mask_query
        return x_query.reshape(B, 1, N, 1)