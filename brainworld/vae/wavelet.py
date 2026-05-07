from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Haar3DSpatial(nn.Module):
    """Fixed 3D Haar transform on spatial dims only, applied frame by frame."""

    def __init__(self) -> None:
        super().__init__()
        kernels = self._build_kernels()
        self.register_buffer("kernels", kernels)

    @staticmethod
    def _build_kernels() -> torch.Tensor:
        h = torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=torch.float32) * 0.3535533906
        g = torch.tensor([[[1, -1], [1, -1]], [[1, -1], [1, -1]]], dtype=torch.float32) * 0.3535533906
        hh = torch.tensor([[[1, 1], [-1, -1]], [[1, 1], [-1, -1]]], dtype=torch.float32) * 0.3535533906
        gh = torch.tensor([[[1, -1], [-1, 1]], [[1, -1], [-1, 1]]], dtype=torch.float32) * 0.3535533906
        h_v = torch.tensor([[[1, 1], [1, 1]], [[-1, -1], [-1, -1]]], dtype=torch.float32) * 0.3535533906
        g_v = torch.tensor([[[1, -1], [1, -1]], [[-1, 1], [-1, 1]]], dtype=torch.float32) * 0.3535533906
        hh_v = torch.tensor([[[1, 1], [-1, -1]], [[-1, -1], [1, 1]]], dtype=torch.float32) * 0.3535533906
        gh_v = torch.tensor([[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]], dtype=torch.float32) * 0.3535533906
        bank = torch.stack([h, g, hh, gh, h_v, g_v, hh_v, gh_v], dim=0)
        return bank.unsqueeze(1)

    @staticmethod
    def _pad_even(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        _, _, d, h, w = x.shape
        pd = d % 2
        ph = h % 2
        pw = w % 2
        if pd or ph or pw:
            x = F.pad(x, (0, pw, 0, ph, 0, pd), mode="replicate")
        return x, (pd, ph, pw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != 1:
            raise ValueError(f"Haar3DSpatial expects [N,1,D,H,W], got {tuple(x.shape)}")
        x, _ = self._pad_even(x)
        k = self.kernels.to(dtype=x.dtype, device=x.device)
        return F.conv3d(x, k, stride=2, padding=0)


def wavelet_volume_coeffs(x: torch.Tensor, haar: Haar3DSpatial | None = None) -> torch.Tensor:
    """x: [B,1,T,D,H,W] -> coeffs [B,8,T,D2,H2,W2]."""
    if x.ndim != 6 or x.shape[1] != 1:
        raise ValueError(f"Expected [B,1,T,D,H,W], got {tuple(x.shape)}")
    if haar is None:
        haar = Haar3DSpatial().to(x.device)

    b, _, t, d, h, w = x.shape
    x_bt = x.permute(0, 2, 1, 3, 4, 5).reshape(b * t, 1, d, h, w)
    c_bt = haar(x_bt)
    d2, h2, w2 = c_bt.shape[-3:]
    c = c_bt.reshape(b, t, 8, d2, h2, w2).permute(0, 2, 1, 3, 4, 5).contiguous()
    return c
