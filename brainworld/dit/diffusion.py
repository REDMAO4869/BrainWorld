from __future__ import annotations

import math
from typing import Tuple

import torch


def normalize_prediction_type(prediction_type: str) -> str:
    key = str(prediction_type).strip().lower()
    if key in {"epsilon", "eps"}:
        return "epsilon"
    if key in {"v", "velocity", "v_prediction", "v-pred", "vpred"}:
        return "v"
    raise ValueError(f"Unsupported prediction_type: {prediction_type}")


def normalize_schedule_type(schedule: str) -> str:
    key = str(schedule).strip().lower()
    if key in {"linear", "cosine"}:
        return key
    raise ValueError(f"Unsupported diffusion.schedule: {schedule}")


class GaussianDiffusion:
    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 1.0e-4,
        beta_end: float = 2.0e-2,
        schedule: str = "linear",
        cosine_s: float = 0.008,
    ) -> None:
        if int(num_steps) <= 1:
            raise ValueError(f"num_steps must be > 1, got {num_steps}")

        self.num_steps = int(num_steps)
        self.schedule = normalize_schedule_type(schedule)
        self.cosine_s = float(cosine_s)

        if self.schedule == "linear":
            betas = torch.linspace(float(beta_start), float(beta_end), self.num_steps, dtype=torch.float32)
        else:
            betas = self._build_cosine_betas(self.num_steps, s=self.cosine_s)

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_bars[:-1]], dim=0)

        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        posterior_log_variance = torch.log(torch.clamp(posterior_variance, min=1.0e-20))
        posterior_mean_coef1 = betas * torch.sqrt(alpha_bars_prev) / (1.0 - alpha_bars)
        posterior_mean_coef2 = (1.0 - alpha_bars_prev) * torch.sqrt(alphas) / (1.0 - alpha_bars)

        self.registered = {
            "betas": betas,
            "alphas": alphas,
            "alpha_bars": alpha_bars,
            "alpha_bars_prev": alpha_bars_prev,
            "sqrt_alpha_bars": torch.sqrt(alpha_bars),
            "sqrt_one_minus_alpha_bars": torch.sqrt(1.0 - alpha_bars),
            "sqrt_recip_alpha_bars": torch.sqrt(1.0 / alpha_bars),
            "sqrt_recipm1_alpha_bars": torch.sqrt(torch.clamp(1.0 / alpha_bars - 1.0, min=0.0)),
            "posterior_variance": posterior_variance,
            "posterior_log_variance": posterior_log_variance,
            "posterior_mean_coef1": posterior_mean_coef1,
            "posterior_mean_coef2": posterior_mean_coef2,
        }

    def to(self, device: torch.device) -> "GaussianDiffusion":
        for k, v in self.registered.items():
            self.registered[k] = v.to(device)
        return self

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_steps, (int(batch_size),), device=device, dtype=torch.long)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        s1 = self._extract(self.registered["sqrt_alpha_bars"], t, x0.shape)
        s2 = self._extract(self.registered["sqrt_one_minus_alpha_bars"], t, x0.shape)
        xt = s1 * x0 + s2 * noise
        return xt, noise

    def target_v(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha = self._extract(self.registered["sqrt_alpha_bars"], t, x0.shape)
        sigma = self._extract(self.registered["sqrt_one_minus_alpha_bars"], t, x0.shape)
        return alpha * noise - sigma * x0

    def get_training_target(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
        prediction_type: str,
    ) -> torch.Tensor:
        pred_type = normalize_prediction_type(prediction_type)
        if pred_type == "epsilon":
            return noise
        return self.target_v(x0, t, noise)

    def predict_x0_from_eps(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        c1 = self._extract(self.registered["sqrt_recip_alpha_bars"], t, xt.shape)
        c2 = self._extract(self.registered["sqrt_recipm1_alpha_bars"], t, xt.shape)
        return c1 * xt - c2 * eps

    def predict_eps_from_v(self, xt: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        alpha = self._extract(self.registered["sqrt_alpha_bars"], t, xt.shape)
        sigma = self._extract(self.registered["sqrt_one_minus_alpha_bars"], t, xt.shape)
        return sigma * xt + alpha * v

    def predict_x0_from_v(self, xt: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        alpha = self._extract(self.registered["sqrt_alpha_bars"], t, xt.shape)
        sigma = self._extract(self.registered["sqrt_one_minus_alpha_bars"], t, xt.shape)
        return alpha * xt - sigma * v

    def predict_eps(self, xt: torch.Tensor, t: torch.Tensor, model_pred: torch.Tensor, prediction_type: str) -> torch.Tensor:
        pred_type = normalize_prediction_type(prediction_type)
        if pred_type == "epsilon":
            return model_pred
        return self.predict_eps_from_v(xt, t, model_pred)

    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, model_pred: torch.Tensor, prediction_type: str) -> torch.Tensor:
        pred_type = normalize_prediction_type(prediction_type)
        if pred_type == "epsilon":
            return self.predict_x0_from_eps(xt, t, model_pred)
        return self.predict_x0_from_v(xt, t, model_pred)

    def q_posterior(self, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = (
            self._extract(self.registered["posterior_mean_coef1"], t, xt.shape) * x0
            + self._extract(self.registered["posterior_mean_coef2"], t, xt.shape) * xt
        )
        var = self._extract(self.registered["posterior_variance"], t, xt.shape)
        log_var = self._extract(self.registered["posterior_log_variance"], t, xt.shape)
        return mean, var, log_var

    def p_sample(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        model_pred: torch.Tensor,
        prediction_type: str,
        *,
        clip_x0: float | None = None,
    ) -> torch.Tensor:
        x0 = self.predict_x0(xt, t, model_pred, prediction_type)
        if clip_x0 is not None:
            clip = float(clip_x0)
            x0 = torch.clamp(x0, min=-clip, max=clip)

        mean, _, log_var = self.q_posterior(x0, xt, t)
        nonzero = (t > 0).float()
        while nonzero.ndim < xt.ndim:
            nonzero = nonzero.unsqueeze(-1)
        noise = torch.randn_like(xt)
        return mean + nonzero * torch.exp(0.5 * log_var) * noise

    def p_sample_from_eps(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        *,
        clip_x0: float | None = None,
    ) -> torch.Tensor:
        return self.p_sample(xt, t, eps, "epsilon", clip_x0=clip_x0)

    @staticmethod
    def _extract(arr: torch.Tensor, t: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        out = arr.gather(0, t)
        while out.ndim < len(shape):
            out = out.unsqueeze(-1)
        return out

    @staticmethod
    def _build_cosine_betas(num_steps: int, s: float = 0.008) -> torch.Tensor:
        steps = torch.arange(0, num_steps + 1, dtype=torch.float32)
        t = steps / float(num_steps)
        alpha_bar = torch.cos(((t + float(s)) / (1.0 + float(s))) * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
        return torch.clamp(betas, min=1.0e-8, max=0.999)
