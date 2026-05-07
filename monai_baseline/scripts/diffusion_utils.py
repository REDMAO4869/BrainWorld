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
    aliases = {
        "linear_beta": "linear",
        "linear": "linear",
        "scaled_linear_beta": "scaled_linear",
        "scaled_linear": "scaled_linear",
        "cosine": "cosine",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported diffusion.schedule: {schedule}")
    return aliases[key]


class Stage2GaussianDiffusion:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1.5e-3,
        beta_end: float = 1.95e-2,
        schedule: str = "scaled_linear_beta",
        prediction_type: str = "v",
        clip_sample: bool = False,
    ) -> None:
        self.num_train_timesteps = int(num_train_timesteps)
        if self.num_train_timesteps <= 1:
            raise ValueError("num_train_timesteps must be > 1")
        self.schedule = normalize_schedule_type(schedule)
        self.prediction_type = normalize_prediction_type(prediction_type)
        self.clip_sample = bool(clip_sample)

        if self.schedule == "linear":
            betas = torch.linspace(float(beta_start), float(beta_end), self.num_train_timesteps, dtype=torch.float32)
        elif self.schedule == "scaled_linear":
            betas = torch.linspace(math.sqrt(float(beta_start)), math.sqrt(float(beta_end)), self.num_train_timesteps, dtype=torch.float32) ** 2
        else:
            betas = self._build_cosine_betas(self.num_train_timesteps)

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
        self.timesteps = torch.arange(self.num_train_timesteps - 1, -1, -1, dtype=torch.long)

    def to(self, device: torch.device) -> "Stage2GaussianDiffusion":
        for key, value in self.registered.items():
            self.registered[key] = value.to(device)
        self.timesteps = self.timesteps.to(device)
        return self

    def set_timesteps(self, num_inference_steps: int, device: torch.device | None = None) -> None:
        num_inference_steps = max(1, int(num_inference_steps))
        steps = torch.linspace(self.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.float32)
        timesteps = torch.round(steps).to(dtype=torch.long)
        if device is not None:
            timesteps = timesteps.to(device)
        self.timesteps = timesteps

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_train_timesteps, (int(batch_size),), device=device, dtype=torch.long)

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self._extract(self.registered["sqrt_alpha_bars"], timesteps, original_samples.shape)
        sqrt_sigma = self._extract(self.registered["sqrt_one_minus_alpha_bars"], timesteps, original_samples.shape)
        return sqrt_alpha * original_samples + sqrt_sigma * noise

    def target_v(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha = self._extract(self.registered["sqrt_alpha_bars"], t, x0.shape)
        sigma = self._extract(self.registered["sqrt_one_minus_alpha_bars"], t, x0.shape)
        return alpha * noise - sigma * x0

    def get_training_target(
        self,
        *,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        prediction_type: str | None = None,
    ) -> torch.Tensor:
        pred_type = normalize_prediction_type(prediction_type or self.prediction_type)
        if pred_type == "epsilon":
            return noise
        return self.target_v(original_samples, timesteps, noise)

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

    def predict_eps(self, xt: torch.Tensor, t: torch.Tensor, model_pred: torch.Tensor, prediction_type: str | None = None) -> torch.Tensor:
        pred_type = normalize_prediction_type(prediction_type or self.prediction_type)
        if pred_type == "epsilon":
            return model_pred
        return self.predict_eps_from_v(xt, t, model_pred)

    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, model_pred: torch.Tensor, prediction_type: str | None = None) -> torch.Tensor:
        pred_type = normalize_prediction_type(prediction_type or self.prediction_type)
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

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor | int, sample: torch.Tensor):
        if isinstance(timestep, int):
            t = torch.full((sample.shape[0],), int(timestep), device=sample.device, dtype=torch.long)
        elif timestep.ndim == 0:
            t = torch.full((sample.shape[0],), int(timestep.item()), device=sample.device, dtype=torch.long)
        else:
            t = timestep.to(device=sample.device, dtype=torch.long)
        x0 = self.predict_x0(sample, t, model_output)
        if self.clip_sample:
            x0 = torch.clamp(x0, -1.0, 1.0)
        mean, _, log_var = self.q_posterior(x0, sample, t)
        nonzero = (t > 0).float()
        while nonzero.ndim < sample.ndim:
            nonzero = nonzero.unsqueeze(-1)
        noise = torch.randn_like(sample)
        prev_sample = mean + nonzero * torch.exp(0.5 * log_var) * noise
        return _StepOutput(prev_sample=prev_sample, pred_original_sample=x0)

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


class _StepOutput:
    def __init__(self, *, prev_sample: torch.Tensor, pred_original_sample: torch.Tensor) -> None:
        self.prev_sample = prev_sample
        self.pred_original_sample = pred_original_sample
