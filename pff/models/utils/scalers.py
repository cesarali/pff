import torch
from torchtyping import TensorType, patch_typeguard

patch_typeguard()  # makes the >torchtyping< hints runtime-checked


class PKScaler:
    """
    Reversible scaler for concentration *and* time.
    Stats are computed per-substance: one row per B-sample, shared across I individuals.

    Value methods:
    - ``"none"``: no value scaling.
    - ``"max"``: divide by masked max in value space.
    - ``"zscore"``: standardize with masked mean/std in value space.
    - ``"log"``: apply ``log(y + 1e-8)`` without additional normalization.
    - ``"log_and_max"``: apply ``log(y + 1e-8)`` then divide by masked max in
      the log space. This centralizes log preprocessing in the scaler instead
      of the dataset pipeline.
    - ``"log_and_z"``: apply ``log(y + 1e-8)`` then standardize in the log
      space using masked mean/std.
    """

    def __init__(
        self,
        value_method: str = "max",  # "max" | "zscore" | "none" | "log" | "log_and_max" | "log_and_z"
        time_method: str = "max",  # "max" | "none"
        eps: float = 1e-6,
    ):
        assert value_method in {"max", "zscore", "none", "log", "log_and_max", "log_and_z"}
        assert time_method in {"max", "none"}
        self.v_method, self.t_method, self.eps = value_method, time_method, eps
        self.log_eps = 1e-8

    # ---------- helpers -------------------------------------------------
    @staticmethod
    def _masked_max(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        Same job as torch.nanmax (which is still missing in 2.6)﻿:contentReference[oaicite:1]{index=1}.
        Invalid entries (mask==0) are turned to -inf, then torch.max().
        """
        x = x.masked_fill(~m.bool(), float("-inf"))
        out = x.max(dim=-1, keepdim=True).values
        return out.clamp_min(1e-12)  # avoid divide-by-zero later

    @staticmethod
    def _masked_max_signed_stable(x: torch.Tensor, m: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Masked max with sign-preserving epsilon guard around zero."""
        x = x.masked_fill(~m.bool(), float("-inf"))
        out = x.max(dim=-1, keepdim=True).values
        sign = torch.where(out < 0, -torch.ones_like(out), torch.ones_like(out))
        return torch.where(out.abs() < eps, sign * eps, out)

    @staticmethod
    def _masked_mean_std(x: torch.Tensor, m: torch.Tensor, eps: float):
        valid = m.bool()
        count = valid.sum(dim=-1, keepdim=True).clamp_min(1)
        mean = x.masked_fill(~valid, 0).sum(dim=-1, keepdim=True) / count
        var = ((x - mean).pow(2)).masked_fill(~valid, 0).sum(dim=-1, keepdim=True) / count
        return mean, var.clamp_min(eps**2).sqrt()

    # ---------- collect statistics -------------------------------------
    def stats(
        self,
        y: TensorType["B", "I", "T", 1],
        t: TensorType["B", "I", "T", 1],
        m: TensorType["B", "I", "T"],
    ) -> dict[str, torch.Tensor]:
        B, I, T, _ = y.shape
        y_flat, t_flat, m_flat = (
            y.squeeze(-1).reshape(B, I * T),
            t.squeeze(-1).reshape(B, I * T),
            m.reshape(B, I * T),
        )

        s: dict[str, torch.Tensor] = {}
        if self.v_method == "max":
            s["v_sigma"] = self._masked_max(y_flat, m_flat)
        elif self.v_method == "log":
            pass
        elif self.v_method == "log_and_max":
            y_log_flat = torch.log(torch.clamp(y_flat + self.log_eps, min=self.log_eps))
            s["v_sigma"] = self._masked_max_signed_stable(y_log_flat, m_flat)
        elif self.v_method == "log_and_z":
            y_log_flat = torch.log(torch.clamp(y_flat + self.log_eps, min=self.log_eps))
            v_mu, v_sigma = self._masked_mean_std(y_log_flat, m_flat, self.eps)
            s.update(v_mu=v_mu, v_sigma=v_sigma)
        elif self.v_method == "zscore":
            v_mu, v_sigma = self._masked_mean_std(y_flat, m_flat, self.eps)
            s.update(v_mu=v_mu, v_sigma=v_sigma)

        if self.t_method == "max":
            s["t_s"] = self._masked_max(t_flat, m_flat)

        return s  # every tensor has shape [B, 1]

    # ---------- transform / inverse ------------------------------------
    def forward(self, y, t, stats):
        if self.v_method != "none":
            if self.v_method == "log":
                y = torch.log(torch.clamp(y + self.log_eps, min=self.log_eps))
            else:
                vscale = stats["v_sigma"][..., None, None]  # [B,1,1,1]
                if self.v_method == "max":
                    y = y / vscale
                elif self.v_method == "log_and_max":
                    y = torch.log(torch.clamp(y + self.log_eps, min=self.log_eps)) / vscale
                elif self.v_method == "log_and_z":
                    vmu = stats["v_mu"][..., None, None]  # [B,1,1,1]
                    y = (torch.log(torch.clamp(y + self.log_eps, min=self.log_eps)) - vmu) / vscale
                else:  # "zscore"
                    vmu = stats["v_mu"][..., None, None]  # [B,1,1,1]
                    y = (y - vmu) / vscale

        if self.t_method == "max":
            tscale = stats["t_s"][..., None, None]  # [B,1,1,1]
            t = t / tscale
        return y, t

    # ------------------------------------------------------------ #
    def inverse(self, y, t, stats):
        if self.v_method != "none":
            if self.v_method == "log":
                y = (torch.exp(y) - self.log_eps).clamp_min(0.0)
            else:
                vscale = stats["v_sigma"][..., None, None]
                if self.v_method == "max":
                    y = y * vscale
                elif self.v_method == "log_and_max":
                    y = (torch.exp(y * vscale) - self.log_eps).clamp_min(0.0)
                elif self.v_method == "log_and_z":
                    vmu = stats["v_mu"][..., None, None]
                    y = (torch.exp(y * vscale + vmu) - self.log_eps).clamp_min(0.0)
                else:  # "zscore"
                    vmu = stats["v_mu"][..., None, None]
                    y = y * vscale + vmu

        if self.t_method == "max":
            tscale = stats["t_s"][..., None, None]
            t = t * tscale
        return y, t

    # ------------------------------------------------------------ #
    def scale_dosing_amounts(self, dosing: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
        """Scale dosing amounts with the same statistics used for observations."""

        if self.v_method == "none":
            return dosing

        if self.v_method == "log":
            return torch.log(torch.clamp(dosing + self.log_eps, min=self.log_eps))

        vscale = stats["v_sigma"]
        # Keep stats rank aligned with dosing rank to avoid accidental
        # broadcasting from [B, 1] against [B] -> [B, B].
        while vscale.ndim > dosing.ndim and vscale.shape[-1] == 1:
            vscale = vscale.squeeze(-1)
        while vscale.ndim < dosing.ndim:
            vscale = vscale.unsqueeze(-1)

        if self.v_method in {"max", "log_and_max"}:
            return dosing / vscale

        vmu = stats["v_mu"]
        while vmu.ndim > dosing.ndim and vmu.shape[-1] == 1:
            vmu = vmu.squeeze(-1)
        while vmu.ndim < dosing.ndim:
            vmu = vmu.unsqueeze(-1)

        if self.v_method == "log_and_z":
            dosing = torch.log(torch.clamp(dosing + self.log_eps, min=self.log_eps))

        return (dosing - vmu) / vscale
