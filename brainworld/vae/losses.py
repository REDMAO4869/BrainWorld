from __future__ import annotations

from typing import Dict

import torch

from .wavelet import Haar3DSpatial, wavelet_volume_coeffs


def _build_fg_mask(x: torch.Tensor, mask: torch.Tensor | None, fg_threshold: float) -> torch.Tensor:
    if mask is not None:
        if mask.ndim != 5:
            raise ValueError(f"mask must be [B,1,D,H,W], got {tuple(mask.shape)}")
        m = (mask > 0).float().unsqueeze(2)
        return m.expand(-1, -1, x.shape[2], -1, -1, -1)
    return (x.abs() > float(fg_threshold)).float()


def _kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    lv = torch.clamp(logvar, min=-30.0, max=20.0)
    return torch.mean(-0.5 * (1.0 + lv - mu.pow(2) - lv.exp()))


def compute_vae_losses(
    *,
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    mask: torch.Tensor | None,
    loss_cfg: Dict,
    fg_threshold: float,
    kl_weight: float,
    haar: Haar3DSpatial | None = None,
) -> Dict[str, torch.Tensor]:
    fg_weight = float(loss_cfg.get("fg_weight", 1.0))
    bg_weight = float(loss_cfg.get("bg_weight", 0.05))
    bg_zero_weight = float(loss_cfg.get("bg_zero_weight", 0.02))
    wavelet_weight = float(loss_cfg.get("wavelet_weight", 0.0))
    temporal_weight = float(loss_cfg.get("temporal_weight", 0.05))

    fg = _build_fg_mask(x, mask=mask, fg_threshold=fg_threshold)
    bg = 1.0 - fg

    err_abs = torch.abs(x_hat - x)
    w = fg_weight * fg + bg_weight * bg
    recon = (err_abs * w).sum() / w.sum().clamp_min(1.0)

    bg_zero = (torch.abs(x_hat) * bg).sum() / bg.sum().clamp_min(1.0)

    if wavelet_weight > 0:
        if haar is None:
            haar = Haar3DSpatial().to(x.device)
        coeff_gt = wavelet_volume_coeffs(x, haar=haar)
        coeff_recon = wavelet_volume_coeffs(x_hat, haar=haar)
        wavelet = torch.mean(torch.abs(coeff_recon - coeff_gt))
    else:
        wavelet = torch.zeros((), device=x.device, dtype=x.dtype)

    if temporal_weight > 0 and x.shape[2] > 1:
        d_gt = x[:, :, 1:] - x[:, :, :-1]
        d_pred = x_hat[:, :, 1:] - x_hat[:, :, :-1]
        temporal = torch.mean(torch.abs(d_pred - d_gt))
    else:
        temporal = torch.zeros((), device=x.device, dtype=x.dtype)

    kl = _kl_loss(mu, logvar)
    total = recon + bg_zero_weight * bg_zero + wavelet_weight * wavelet + temporal_weight * temporal + float(kl_weight) * kl

    return {
        "loss": total,
        "recon_loss": recon,
        "bg_zero_loss": bg_zero,
        "wavelet_loss": wavelet,
        "temporal_loss": temporal,
        "kl_loss": kl,
        "kl_weight": torch.tensor(float(kl_weight), device=x.device, dtype=x.dtype),
        "fg_ratio": fg.mean(),
        "bg_ratio": bg.mean(),
        "xhat_abs_mean": torch.mean(torch.abs(x_hat)),
        "xhat_abs_max": torch.amax(torch.abs(x_hat)),
    }
