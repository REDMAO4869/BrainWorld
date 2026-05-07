from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sincos_1d(pos: torch.Tensor, dim: int, *, temperature: float = 10_000.0) -> torch.Tensor:
    if dim <= 0:
        return torch.zeros((pos.shape[0], 0), device=pos.device, dtype=torch.float32)
    half = dim // 2
    if half == 0:
        return torch.zeros((pos.shape[0], dim), device=pos.device, dtype=torch.float32)
    omega = torch.arange(half, device=pos.device, dtype=torch.float32) / float(half)
    omega = 1.0 / (temperature**omega)
    out = pos.float().unsqueeze(1) * omega.unsqueeze(0)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=emb.device, dtype=emb.dtype)], dim=1)
    if emb.shape[1] != dim:
        emb = F.pad(emb, (0, max(0, dim - emb.shape[1])))[:, :dim]
    return emb


def build_3d_sincos_pos_embed(d: int, h: int, w: int, dim: int, *, device: torch.device) -> torch.Tensor:
    dz = dim // 3
    dy = dim // 3
    dx = dim - dz - dy
    zz, yy, xx = torch.meshgrid(
        torch.arange(d, device=device),
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    emb = torch.cat(
        [
            _sincos_1d(zz.reshape(-1), dz),
            _sincos_1d(yy.reshape(-1), dy),
            _sincos_1d(xx.reshape(-1), dx),
        ],
        dim=1,
    )
    if emb.shape[1] != dim:
        emb = F.pad(emb, (0, max(0, dim - emb.shape[1])))[:, :dim]
    return emb.unsqueeze(0)


class PatchEmbed3D(nn.Module):
    def __init__(self, in_ch: int, d_model: int, patch_size: int) -> None:
        super().__init__()
        p = int(patch_size)
        self.patch_size = p
        self.proj = nn.Conv3d(in_ch, d_model, kernel_size=p, stride=p)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int, int]:
        y = self.proj(x)
        b, c, dp, hp, wp = y.shape
        return y.permute(0, 2, 3, 4, 1).reshape(b, dp * hp * wp, c), dp, hp, wp


class TimestepEmbedder(nn.Module):
    def __init__(self, d_cond: int, hidden: Optional[int] = None) -> None:
        super().__init__()
        self.d_cond = int(d_cond)
        hidden_dim = int(hidden) if hidden is not None else int(d_cond * 4)
        self.mlp = nn.Sequential(
            nn.Linear(self.d_cond, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.d_cond),
        )

    @staticmethod
    def sincos(t: torch.Tensor, dim: int) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError("timesteps must be (B,)")
        half = dim // 2
        if half == 0:
            return torch.zeros((t.shape[0], dim), device=t.device, dtype=torch.float32)
        freqs = torch.exp(-math.log(10_000.0) * torch.arange(0, half, device=t.device, dtype=torch.float32) / float(half))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros((t.shape[0], 1), device=t.device, dtype=emb.dtype)], dim=1)
        if emb.shape[1] != dim:
            emb = F.pad(emb, (0, max(0, dim - emb.shape[1])))[:, :dim]
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sincos(t, self.d_cond))


class AdaLNZero(nn.Module):
    def __init__(self, d_model: int, d_cond: int) -> None:
        super().__init__()
        hidden = int(d_cond * 4)
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_cond, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 6 * d_model),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, y: torch.Tensor):
        out = self.net(y).view(y.shape[0], 6, -1)
        return out.unbind(dim=1)


class DiTBlock(nn.Module):
    def __init__(self, *, d_model: int, num_heads: int, mlp_ratio: float, dropout: float, d_cond: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )
        self.drop = nn.Dropout(dropout)
        self.mod = AdaLNZero(d_model=d_model, d_cond=d_cond)

    @staticmethod
    def _adaln(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.mod(y)
        h = self._adaln(self.ln1(x), shift1, scale1)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(attn_out) * gate1.unsqueeze(1)

        h = self._adaln(self.ln2(x), shift2, scale2)
        x = x + self.drop(self.mlp(h)) * gate2.unsqueeze(1)
        return x


class DiT3DDenoiser(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_tasks: int,
        patch_size: int = 4,
        d_model: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        d_cond: Optional[int] = None,
        pos_embed: str = "sincos",
        max_d: int = 8,
        max_h: int = 8,
        max_w: int = 8,
        fc_dim: int = 0,
        use_anchor_condition: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.patch_size = int(patch_size)
        self.d_model = int(d_model)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout = float(dropout)
        self.d_cond = int(d_cond) if d_cond is not None else int(d_model)
        self.pos_embed_type = str(pos_embed).strip().lower()
        self.fc_dim = int(fc_dim)
        self.use_anchor_condition = bool(use_anchor_condition)

        self.patch = PatchEmbed3D(self.in_channels, self.d_model, self.patch_size)
        if self.pos_embed_type == "learned":
            self.max_d = int(max_d)
            self.max_h = int(max_h)
            self.max_w = int(max_w)
            self.pos_embed = nn.Parameter(torch.zeros(1, self.max_d * self.max_h * self.max_w, self.d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif self.pos_embed_type == "sincos":
            self.max_d = self.max_h = self.max_w = 0
            self.pos_embed = None
        else:
            raise ValueError("pos_embed must be one of: sincos, learned")

        self.time_embed = TimestepEmbedder(self.d_cond)
        self.task_embed = nn.Embedding(int(num_tasks) + 1, self.d_cond)
        fc_input_dim = max(1, self.fc_dim if self.fc_dim > 0 else self.d_cond)
        self.fc_input_dim = fc_input_dim
        self.fc_mlp = nn.Sequential(
            nn.Linear(fc_input_dim, self.d_cond),
            nn.SiLU(),
            nn.Linear(self.d_cond, self.d_cond),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    dropout=self.dropout,
                    d_cond=self.d_cond,
                )
                for _ in range(self.depth)
            ]
        )
        self.final_ln = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.out_channels * (self.patch_size**3))

    def _pos_embed(self, dp: int, hp: int, wp: int, device: torch.device) -> torch.Tensor:
        if self.pos_embed_type == "learned":
            if dp > self.max_d or hp > self.max_h or wp > self.max_w:
                raise ValueError(
                    f"learned pos embed grid exceeded: got (d,h,w)=({dp},{hp},{wp}) max=({self.max_d},{self.max_h},{self.max_w})"
                )
            return self.pos_embed[:, : dp * hp * wp, :].to(device=device)
        return build_3d_sincos_pos_embed(dp, hp, wp, self.d_model, device=device)

    def _prepare_fc(self, fc_cond: Optional[torch.Tensor], has_fc: Optional[torch.Tensor], batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if fc_cond is None:
            fc = torch.zeros((batch_size, self.fc_input_dim), device=device, dtype=dtype)
        else:
            fc = fc_cond.float()
            if fc.ndim == 1:
                fc = fc.unsqueeze(0)
            if fc.ndim > 2:
                fc = fc.flatten(start_dim=1)
            if fc.shape[0] != batch_size:
                raise ValueError(f"fc_cond batch mismatch: {fc.shape[0]} vs {batch_size}")
            if int(fc.shape[1]) != self.fc_input_dim:
                fc = F.adaptive_avg_pool1d(fc.unsqueeze(1), self.fc_input_dim).squeeze(1)
            fc = fc.to(device=device, dtype=dtype)
        if has_fc is not None:
            exists = has_fc.to(device=device, dtype=dtype).view(-1, 1)
            fc = fc * (exists > 0.5)
        return fc

    def _condition_input(self, x: torch.Tensor, anchor_latent: Optional[torch.Tensor]) -> torch.Tensor:
        if self.use_anchor_condition and anchor_latent is not None:
            x = x + 0.1 * anchor_latent.to(device=x.device, dtype=x.dtype)
        return x

    def unpatchify(self, tokens: torch.Tensor, dp: int, hp: int, wp: int) -> torch.Tensor:
        b, l, d = tokens.shape
        p = self.patch_size
        c = self.out_channels
        expected_d = c * (p**3)
        if l != dp * hp * wp:
            raise ValueError(f"token length mismatch: {l} vs {dp*hp*wp}")
        if d != expected_d:
            raise ValueError(f"token dim mismatch: {d} vs {expected_d}")
        x = tokens.view(b, dp, hp, wp, c, p, p, p)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return x.view(b, c, dp * p, hp * p, wp * p)

    def forward(
        self,
        *,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: torch.Tensor,
        fc_cond: Optional[torch.Tensor] = None,
        anchor_latent: Optional[torch.Tensor] = None,
        has_fc: Optional[torch.Tensor] = None,
        return_hiddens: bool = False,
        capture_layers: Optional[Sequence[int]] = None,
    ) -> torch.Tensor | dict[str, Any]:
        if x.ndim != 5:
            raise ValueError(f"Expected x as (B,C,D,H,W), got {tuple(x.shape)}")
        b, c, d, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"in_channels mismatch: {c} vs {self.in_channels}")
        if timesteps.ndim != 1 or timesteps.shape[0] != b:
            raise ValueError("timesteps must be (B,)")
        if class_labels.ndim != 1 or class_labels.shape[0] != b:
            raise ValueError("class_labels must be (B,)")
        if (d % self.patch_size) != 0 or (h % self.patch_size) != 0 or (w % self.patch_size) != 0:
            raise ValueError(
                f"Input spatial shape {(d, h, w)} must be divisible by patch_size={self.patch_size}"
            )

        x = self._condition_input(x, anchor_latent)
        tokens, dp, hp, wp = self.patch(x)
        tokens = tokens + self._pos_embed(dp, hp, wp, x.device).to(device=x.device, dtype=tokens.dtype)

        y = self.time_embed(timesteps).to(dtype=tokens.dtype)
        labels = class_labels.clamp(min=0, max=self.task_embed.num_embeddings - 1)
        y = y + self.task_embed(labels).to(dtype=tokens.dtype)
        y = y + self.fc_mlp(self._prepare_fc(fc_cond, has_fc, b, x.device, tokens.dtype))

        capture = set(int(v) for v in capture_layers) if capture_layers is not None else set()
        hiddens: list[torch.Tensor] = []
        for idx, block in enumerate(self.blocks):
            tokens = block(tokens, y)
            if return_hiddens and idx in capture:
                hiddens.append(tokens)

        tokens = self.head(self.final_ln(tokens))
        sample = self.unpatchify(tokens, dp, hp, wp)
        if not return_hiddens:
            return sample
        return {"sample": sample, "hiddens": hiddens}


def build_diffusion_dit3d(config: dict[str, Any], num_tasks: int, fc_dim: int = 0) -> DiT3DDenoiser:
    model_cfg = config.get("model", {})
    dit_cfg = config.get("dit", {})
    latent_channels = int(model_cfg["latent_channels"])
    num_frames = int(model_cfg["num_frames"])
    stacked_channels = latent_channels * num_frames
    d_model = int(dit_cfg.get("d_model", 768))
    return DiT3DDenoiser(
        in_channels=stacked_channels,
        out_channels=stacked_channels,
        num_tasks=int(num_tasks),
        patch_size=int(dit_cfg.get("patch_size", 4)),
        d_model=d_model,
        depth=int(dit_cfg.get("depth", 12)),
        num_heads=int(dit_cfg.get("num_heads", 12)),
        mlp_ratio=float(dit_cfg.get("mlp_ratio", 4.0)),
        dropout=float(dit_cfg.get("dropout", 0.0)),
        d_cond=d_model,
        pos_embed=str(dit_cfg.get("pos_embed", "sincos")),
        fc_dim=int(fc_dim),
        use_anchor_condition=bool(config.get("conditioning", {}).get("use_anchor_fc_condition", True)),
    )
