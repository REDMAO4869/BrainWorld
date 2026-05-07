from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavelet import Haar3DSpatial


def _to_list_int(x: Sequence[int] | int, n: int) -> list[int]:
    if isinstance(x, int):
        return [int(x)] * n
    out = [int(v) for v in x]
    if len(out) != n:
        raise ValueError(f"Expected list length {n}, got {len(out)}")
    return out


def _pick_gn_groups(channels: int, max_groups: int = 8) -> int:
    g = min(max_groups, int(channels))
    while g > 1:
        if channels % g == 0:
            return g
        g -= 1
    return 1


def _check_odd_kernel(k: int, name: str) -> int:
    k = int(k)
    if k < 1:
        raise ValueError(f"{name} must be >= 1, got {k}")
    if (k % 2) == 0:
        raise ValueError(f"{name} must be odd to keep shape stable, got {k}")
    return k


def _merge_time(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    b, c, t, d, h, w = x.shape
    xt = x.permute(0, 2, 1, 3, 4, 5).reshape(b * t, c, d, h, w)
    return xt, b, t


def _restore_time(x_bt: torch.Tensor, b: int, t: int) -> torch.Tensor:
    _, c, d, h, w = x_bt.shape
    return x_bt.reshape(b, t, c, d, h, w).permute(0, 2, 1, 3, 4, 5).contiguous()


class SpatialResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int, dropout: float) -> None:
        super().__init__()
        ks = _check_odd_kernel(kernel_size, "spatial_kernel_size")
        pd = ks // 2
        g1 = _pick_gn_groups(in_channels)
        g2 = _pick_gn_groups(out_channels)
        self.norm1 = nn.GroupNorm(g1, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=ks, padding=pd)
        self.norm2 = nn.GroupNorm(g2, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=ks, padding=pd)
        self.dropout = nn.Dropout3d(dropout) if float(dropout) > 0 else nn.Identity()
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt, b, t = _merge_time(x)
        h = self.conv1(F.silu(self.norm1(xt)))
        h = self.dropout(h)
        h = self.conv2(F.silu(self.norm2(h)))
        out = self.skip(xt) + h
        return _restore_time(out, b, t)


class WaveletFusionLite(nn.Module):
    """Lightweight spatial wavelet-assisted residual fusion."""

    def __init__(self, channels: int, energy_channels: int) -> None:
        super().__init__()
        self.to_scalar = nn.Conv3d(channels, 1, kernel_size=1)
        self.haar = Haar3DSpatial()
        self.to_energy = nn.Conv3d(8, energy_channels, kernel_size=1)
        self.energy_to_c = nn.Conv3d(energy_channels, channels, kernel_size=1)
        self.gate = nn.Conv3d(channels, channels, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt, b, t = _merge_time(x)
        scalar = self.to_scalar(xt)
        coeff = self.haar(scalar)
        coeff_up = F.interpolate(coeff, size=xt.shape[-3:], mode="trilinear", align_corners=False)
        energy = self.to_energy(coeff_up)
        delta = torch.sigmoid(self.gate(xt)) * self.energy_to_c(energy)
        out = xt + torch.tanh(self.alpha) * delta
        return _restore_time(out, b, t)


class SpatialDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        res_blocks: int,
        dropout: float,
        use_wavelet: bool,
        wavelet_energy_channels: int,
    ) -> None:
        super().__init__()
        ks = _check_odd_kernel(kernel_size, "spatial_kernel_size")
        self.down = nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.blocks = nn.ModuleList(
            [SpatialResBlock(out_channels, out_channels, kernel_size=ks, dropout=dropout) for _ in range(int(res_blocks))]
        )
        self.wavelet = WaveletFusionLite(out_channels, wavelet_energy_channels) if use_wavelet else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt, b, t = _merge_time(x)
        x = _restore_time(self.down(xt), b, t)
        for blk in self.blocks:
            x = blk(x)
        return self.wavelet(x)


class SpatialUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        res_blocks: int,
        dropout: float,
        use_wavelet: bool,
        wavelet_energy_channels: int,
    ) -> None:
        super().__init__()
        ks = _check_odd_kernel(kernel_size, "spatial_kernel_size")
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.blocks = nn.ModuleList(
            [SpatialResBlock(out_channels, out_channels, kernel_size=ks, dropout=dropout) for _ in range(int(res_blocks))]
        )
        self.wavelet = WaveletFusionLite(out_channels, wavelet_energy_channels) if use_wavelet else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt, b, t = _merge_time(x)
        x = _restore_time(self.up(xt), b, t)
        for blk in self.blocks:
            x = blk(x)
        return self.wavelet(x)


class LocalTemporalBlock(nn.Module):
    """Per-latent-voxel temporal mixing without global spatial pooling."""

    def __init__(self, channels: int, *, kernel_size: int, dropout: float, residual_scale: float) -> None:
        super().__init__()
        ks = _check_odd_kernel(kernel_size, "temporal_kernel_size")
        pd = ks // 2
        g = _pick_gn_groups(channels)
        self.norm1 = nn.GroupNorm(g, channels)
        self.dw1 = nn.Conv1d(channels, channels, kernel_size=ks, padding=pd, groups=channels)
        self.pw1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm2 = nn.GroupNorm(g, channels)
        self.dw2 = nn.Conv1d(channels, channels, kernel_size=ks, padding=pd, groups=channels)
        self.pw2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout) if float(dropout) > 0 else nn.Identity()
        self.scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, d, h, w = x.shape
        seq = x.permute(0, 3, 4, 5, 1, 2).reshape(b * d * h * w, c, t)
        h1 = self.pw1(self.dw1(F.silu(self.norm1(seq))))
        h1 = self.dropout(h1)
        h2 = self.pw2(self.dw2(F.silu(self.norm2(h1))))
        seq = seq + self.scale * h2
        return seq.reshape(b, d, h, w, c, t).permute(0, 4, 5, 1, 2, 3).contiguous()


class TemporalDownsampleBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        g = _pick_gn_groups(channels)
        self.norm = nn.GroupNorm(g, channels)
        self.dw = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1, groups=channels)
        self.pw = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, d, h, w = x.shape
        seq = x.permute(0, 3, 4, 5, 1, 2).reshape(b * d * h * w, c, t)
        seq = self.pw(self.dw(F.silu(self.norm(seq))))
        t2 = seq.shape[-1]
        return seq.reshape(b, d, h, w, c, t2).permute(0, 4, 5, 1, 2, 3).contiguous()


class TemporalUpsampleBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        g = _pick_gn_groups(channels)
        self.norm = nn.GroupNorm(g, channels)
        self.up = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1, groups=channels)
        self.pw = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, target_t: Optional[int] = None) -> torch.Tensor:
        b, c, t, d, h, w = x.shape
        seq = x.permute(0, 3, 4, 5, 1, 2).reshape(b * d * h * w, c, t)
        seq = self.pw(self.up(F.silu(self.norm(seq))))
        if target_t is not None:
            target_t = int(target_t)
            if seq.shape[-1] < target_t:
                seq = F.pad(seq, (0, target_t - seq.shape[-1]), mode="replicate")
            seq = seq[:, :, :target_t]
        t2 = seq.shape[-1]
        return seq.reshape(b, d, h, w, c, t2).permute(0, 4, 5, 1, 2, 3).contiguous()


class TemporalPassThrough(nn.Module):
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x


@dataclass
class FMRIWaveletVAEConfig:
    in_channels: int = 1
    stage_channels: Tuple[int, int, int, int] = (16, 32, 64, 96)
    latent_dim: int = 16
    res_blocks_per_stage: int = 2
    dropout: float = 0.0
    spatial_kernel_size: int = 3
    temporal_kernel_size: int = 3
    temporal_residual_scale: float = 0.1
    logvar_min: float = -6.0
    logvar_max: float = 4.0
    wavelet_fusion_enabled: bool = True
    wavelet_energy_channels: int = 8
    wavelet_stages: Tuple[str, ...] = ("enc2", "enc3", "dec1", "dec2")
    num_spatial_downsamples: int = 4
    temporal_mode: str = "full"
    spatial_only_frame_chunk_size: int = 0


class FMRIWaveletVAE(nn.Module):
    def __init__(self, cfg: FMRIWaveletVAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if int(cfg.num_spatial_downsamples) not in (3, 4):
            raise ValueError(f"num_spatial_downsamples must be 3 or 4, got {cfg.num_spatial_downsamples}")
        self.num_spatial_downsamples = int(cfg.num_spatial_downsamples)
        self.temporal_mode = str(cfg.temporal_mode).strip().lower()
        if self.temporal_mode not in {"full", "spatial_only"}:
            raise ValueError(f"temporal_mode must be full or spatial_only, got {cfg.temporal_mode}")
        self.spatial_only = self.temporal_mode == "spatial_only"
        self.spatial_only_frame_chunk_size = max(0, int(cfg.spatial_only_frame_chunk_size))
        c0, c1, c2, c3 = cfg.stage_channels
        ks = _check_odd_kernel(cfg.spatial_kernel_size, "spatial_kernel_size")
        wavelet_stages = {str(x) for x in cfg.wavelet_stages}

        self.stem = nn.Conv3d(cfg.in_channels, c0, kernel_size=4, stride=2, padding=1)
        self.enc1 = SpatialDownBlock(
            c0,
            c1,
            kernel_size=ks,
            res_blocks=cfg.res_blocks_per_stage,
            dropout=cfg.dropout,
            use_wavelet=cfg.wavelet_fusion_enabled and "enc1" in wavelet_stages,
            wavelet_energy_channels=cfg.wavelet_energy_channels,
        )
        if self.spatial_only:
            self.temporal_block1 = TemporalPassThrough()
            self.temporal_down1 = TemporalPassThrough()
        else:
            self.temporal_block1 = LocalTemporalBlock(
                c1,
                kernel_size=cfg.temporal_kernel_size,
                dropout=cfg.dropout,
                residual_scale=cfg.temporal_residual_scale,
            )
            self.temporal_down1 = TemporalDownsampleBlock(c1)
        self.enc2 = SpatialDownBlock(
            c1,
            c2,
            kernel_size=ks,
            res_blocks=cfg.res_blocks_per_stage,
            dropout=cfg.dropout,
            use_wavelet=cfg.wavelet_fusion_enabled and "enc2" in wavelet_stages,
            wavelet_energy_channels=cfg.wavelet_energy_channels,
        )
        if self.spatial_only:
            self.temporal_block2 = TemporalPassThrough()
            self.temporal_down2 = TemporalPassThrough()
        else:
            self.temporal_block2 = LocalTemporalBlock(
                c2,
                kernel_size=cfg.temporal_kernel_size,
                dropout=cfg.dropout,
                residual_scale=cfg.temporal_residual_scale,
            )
            self.temporal_down2 = TemporalDownsampleBlock(c2)
        if self.num_spatial_downsamples == 4:
            self.enc3 = SpatialDownBlock(
                c2,
                c3,
                kernel_size=ks,
                res_blocks=cfg.res_blocks_per_stage,
                dropout=cfg.dropout,
                use_wavelet=cfg.wavelet_fusion_enabled and "enc3" in wavelet_stages,
                wavelet_energy_channels=cfg.wavelet_energy_channels,
            )
        else:
            self.enc3 = nn.Identity()

        latent_channels = c3 if self.num_spatial_downsamples == 4 else c2
        self.to_mu = nn.Conv3d(latent_channels, cfg.latent_dim, kernel_size=1)
        self.to_logvar = nn.Conv3d(latent_channels, cfg.latent_dim, kernel_size=1)
        if self.to_logvar.bias is not None:
            nn.init.constant_(self.to_logvar.bias, -2.0)

        self.dec_in = nn.Conv3d(cfg.latent_dim, latent_channels, kernel_size=ks, padding=ks // 2)
        if self.num_spatial_downsamples == 4:
            self.dec1 = SpatialUpBlock(
                c3,
                c2,
                kernel_size=ks,
                res_blocks=cfg.res_blocks_per_stage,
                dropout=cfg.dropout,
                use_wavelet=cfg.wavelet_fusion_enabled and "dec1" in wavelet_stages,
                wavelet_energy_channels=cfg.wavelet_energy_channels,
            )
        else:
            self.dec1 = nn.Identity()
        if self.spatial_only:
            self.temporal_up1 = TemporalPassThrough()
        else:
            self.temporal_up1 = TemporalUpsampleBlock(c2)
        self.dec2 = SpatialUpBlock(
            c2,
            c1,
            kernel_size=ks,
            res_blocks=cfg.res_blocks_per_stage,
            dropout=cfg.dropout,
            use_wavelet=cfg.wavelet_fusion_enabled and "dec2" in wavelet_stages,
            wavelet_energy_channels=cfg.wavelet_energy_channels,
        )
        if self.spatial_only:
            self.temporal_up2 = TemporalPassThrough()
        else:
            self.temporal_up2 = TemporalUpsampleBlock(c1)
        self.dec3 = SpatialUpBlock(
            c1,
            c0,
            kernel_size=ks,
            res_blocks=cfg.res_blocks_per_stage,
            dropout=cfg.dropout,
            use_wavelet=cfg.wavelet_fusion_enabled and "dec3" in wavelet_stages,
            wavelet_energy_channels=cfg.wavelet_energy_channels,
        )
        self.dec4 = SpatialUpBlock(
            c0,
            max(8, c0 // 2),
            kernel_size=ks,
            res_blocks=cfg.res_blocks_per_stage,
            dropout=cfg.dropout,
            use_wavelet=cfg.wavelet_fusion_enabled and "dec4" in wavelet_stages,
            wavelet_energy_channels=cfg.wavelet_energy_channels,
        )
        self.dec_out = nn.Conv3d(max(8, c0 // 2), cfg.in_channels, kernel_size=ks, padding=ks // 2)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, sample_posterior: bool) -> torch.Tensor:
        if not sample_posterior:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _align_spatial_to_target(x: torch.Tensor, target_shape_dhw: Tuple[int, int, int]) -> torch.Tensor:
        td, th, tw = target_shape_dhw
        if x.shape[-3:] == (td, th, tw):
            return x
        b, c, t, d, h, w = x.shape
        xt = x.permute(0, 2, 1, 3, 4, 5).reshape(b * t, c, d, h, w)
        xt = F.interpolate(xt, size=(td, th, tw), mode="trilinear", align_corners=False)
        return xt.reshape(b, t, c, td, th, tw).permute(0, 2, 1, 3, 4, 5).contiguous()

    def _run_bt_in_chunks(self, x_bt: torch.Tensor, fn) -> torch.Tensor:
        chunk = int(self.spatial_only_frame_chunk_size)
        if chunk <= 0 or x_bt.shape[0] <= chunk:
            return fn(x_bt)
        outs = []
        for start in range(0, x_bt.shape[0], chunk):
            outs.append(fn(x_bt[start : start + chunk]))
        return torch.cat(outs, dim=0)

    def _encode_spatial_bt(self, x_bt: torch.Tensor) -> torch.Tensor:
        n = int(x_bt.shape[0])
        h = _restore_time(self.stem(x_bt), n, 1)
        h = self.enc1(h)
        h = self.enc2(h)
        h = self.enc3(h)
        h_bt, _, _ = _merge_time(h)
        return h_bt

    def _decode_spatial_bt(self, z_bt: torch.Tensor) -> torch.Tensor:
        n = int(z_bt.shape[0])
        h = _restore_time(self.dec_in(z_bt), n, 1)
        h = self.dec1(h)
        h = self.dec2(h)
        h = self.dec3(h)
        h = self.dec4(h)
        h_bt, _, _ = _merge_time(h)
        return self.dec_out(h_bt)

    def encode(self, x: torch.Tensor, sample_posterior: bool = True) -> Dict[str, torch.Tensor]:
        if x.ndim != 6:
            raise ValueError(f"Expected [B,1,T,D,H,W], got {tuple(x.shape)}")
        _, _, t0, _, _, _ = x.shape

        if self.spatial_only:
            xt, b, t = _merge_time(x)
            h_bt = self._run_bt_in_chunks(xt, self._encode_spatial_bt)
            mu_bt = self.to_mu(h_bt)
            logvar_bt = self.to_logvar(h_bt)
            mu = _restore_time(mu_bt, b, t)
            logvar = _restore_time(logvar_bt, b, t)
        else:
            xt, b, t = _merge_time(x)
            h = _restore_time(self.stem(xt), b, t)
            h = self.enc1(h)
            h = self.temporal_block1(h)
            h = self.temporal_down1(h)
            h = self.enc2(h)
            h = self.temporal_block2(h)
            h = self.temporal_down2(h)
            h = self.enc3(h)

            h_bt, b2, tz = _merge_time(h)
            mu_bt = self.to_mu(h_bt)
            logvar_bt = self.to_logvar(h_bt)
            mu = _restore_time(mu_bt, b2, tz)
            logvar = _restore_time(logvar_bt, b2, tz)

        lv_min = float(self.cfg.logvar_min)
        lv_max = float(self.cfg.logvar_max)
        if lv_max < lv_min:
            lv_min, lv_max = lv_max, lv_min
        logvar = torch.clamp(logvar, min=lv_min, max=lv_max)
        z = self.reparameterize(mu, logvar, sample_posterior=sample_posterior)

        return {
            "mu": mu.permute(0, 2, 1, 3, 4, 5).contiguous(),
            "logvar": logvar.permute(0, 2, 1, 3, 4, 5).contiguous(),
            "z": z.permute(0, 2, 1, 3, 4, 5).contiguous(),
            "t_orig": torch.tensor([t0], device=x.device, dtype=torch.long),
        }

    def decode(self, z: torch.Tensor, target_t: Optional[int] = None) -> torch.Tensor:
        if z.ndim != 6:
            raise ValueError(f"Expected z [B,T,L,D,H,W], got {tuple(z.shape)}")

        if self.spatial_only:
            b, t, l, dz, hz, wz = z.shape
            z_bt = z.reshape(b * t, l, dz, hz, wz)
            x_bt = self._run_bt_in_chunks(z_bt, self._decode_spatial_bt)
            _, c, d, h, w = x_bt.shape
            x = x_bt.reshape(b, t, c, d, h, w).permute(0, 2, 1, 3, 4, 5).contiguous()
            if target_t is not None:
                target_t = int(target_t)
                if x.shape[2] < target_t:
                    x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, target_t - x.shape[2]), mode="replicate")
                x = x[:, :, :target_t]
            return x

        z = z.permute(0, 2, 1, 3, 4, 5).contiguous()
        z_bt, b, t = _merge_time(z)
        h = _restore_time(self.dec_in(z_bt), b, t)
        h = self.dec1(h)
        half_t = None if target_t is None else max(1, int(target_t) // 2)
        h = self.temporal_up1(h, target_t=half_t)
        h = self.dec2(h)
        h = self.temporal_up2(h, target_t=target_t)
        h = self.dec3(h)
        h = self.dec4(h)

        h_bt, b2, t2 = _merge_time(h)
        x_bt = self.dec_out(h_bt)
        x = _restore_time(x_bt, b2, t2)
        if target_t is not None:
            target_t = int(target_t)
            if x.shape[2] < target_t:
                x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, target_t - x.shape[2]), mode="replicate")
            x = x[:, :, :target_t]
        return x

    def forward(self, x: torch.Tensor, sample_posterior: bool = True) -> Dict[str, torch.Tensor]:
        enc = self.encode(x, sample_posterior=sample_posterior)
        x_hat = self.decode(enc["z"], target_t=int(x.shape[2]))
        x_hat = self._align_spatial_to_target(x_hat, target_shape_dhw=(x.shape[3], x.shape[4], x.shape[5]))
        return {
            "x_hat": x_hat,
            "mu": enc["mu"],
            "logvar": enc["logvar"],
            "z": enc["z"],
            "t_orig": enc["t_orig"],
        }


def build_model_from_config(cfg: Dict) -> FMRIWaveletVAE:
    model_cfg = cfg.get("model", cfg)
    stage_channels = tuple(_to_list_int(model_cfg.get("stage_channels", [16, 32, 64, 96]), 4))

    wavelet_stages_raw = model_cfg.get("wavelet_stages", ("enc2", "enc3", "dec1", "dec2"))
    if isinstance(wavelet_stages_raw, str):
        wavelet_stages = tuple(x.strip() for x in wavelet_stages_raw.replace(",", " ").split() if x.strip())
    else:
        wavelet_stages = tuple(str(x).strip() for x in wavelet_stages_raw if str(x).strip())

    cfg_obj = FMRIWaveletVAEConfig(
        in_channels=int(model_cfg.get("in_channels", 1)),
        stage_channels=stage_channels,
        latent_dim=int(model_cfg.get("latent_dim", 16)),
        res_blocks_per_stage=int(model_cfg.get("res_blocks_per_stage", 2)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        spatial_kernel_size=int(model_cfg.get("spatial_kernel_size", model_cfg.get("kernel_size", 3))),
        temporal_kernel_size=int(model_cfg.get("temporal_kernel_size", model_cfg.get("temporal_block_kernel_size", 3))),
        temporal_residual_scale=float(model_cfg.get("temporal_residual_scale", 0.1)),
        logvar_min=float(model_cfg.get("logvar_min", -6.0)),
        logvar_max=float(model_cfg.get("logvar_max", 4.0)),
        wavelet_fusion_enabled=bool(model_cfg.get("wavelet_fusion_enabled", True)),
        wavelet_energy_channels=int(model_cfg.get("wavelet_energy_channels", 8)),
        wavelet_stages=wavelet_stages,
        num_spatial_downsamples=int(model_cfg.get("num_spatial_downsamples", 4)),
        temporal_mode=str(model_cfg.get("temporal_mode", "full")),
        spatial_only_frame_chunk_size=int(model_cfg.get("spatial_only_frame_chunk_size", 0)),
    )
    return FMRIWaveletVAE(cfg_obj)
