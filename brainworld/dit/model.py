from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / max(1, half))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def compute_patch_audit(target_shape: Sequence[int], patch_size: Sequence[int]) -> Dict[str, object]:
    if len(target_shape) != 5:
        raise ValueError(f"target_shape must be [Tz,L,D,H,W], got {tuple(target_shape)}")
    if len(patch_size) != 3:
        raise ValueError(f"patch_size must be [pd,ph,pw], got {tuple(patch_size)}")
    tz, latent_channels, d, h, w = [int(v) for v in target_shape]
    pd, ph, pw = [int(v) for v in patch_size]
    if d % pd != 0 or h % ph != 0 or w % pw != 0:
        raise ValueError(
            f"latent spatial shape {(d, h, w)} must be divisible by patch size {(pd, ph, pw)}"
        )
    gd, gh, gw = d // pd, h // ph, w // pw
    return {
        "target_shape": (tz, latent_channels, d, h, w),
        "patch_size": (pd, ph, pw),
        "grid_shape": (gd, gh, gw),
        "tokens_per_t": int(gd * gh * gw),
        "patch_num": int(tz * gd * gh * gw),
        "patch_dim": int(latent_channels * pd * ph * pw),
        "latent_channels": int(latent_channels),
        "latent_tz": int(tz),
    }


class ConditionEncoderBase(nn.Module):
    def __init__(self, hidden_dim: int, num_tokens: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_tokens = max(0, int(num_tokens))
        self.null_global = nn.Parameter(torch.zeros(1, hidden_dim))
        self.null_tokens = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_dim))

    def _apply_mask(self, tokens: torch.Tensor, global_emb: torch.Tensor, exists: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if exists is None:
            return tokens, global_emb
        mask = (exists > 0.5).view(-1, 1, 1)
        gmask = (exists > 0.5).view(-1, 1)
        if tokens.numel() > 0:
            tokens = torch.where(mask, tokens, self.null_tokens.expand(tokens.shape[0], -1, -1))
        global_emb = torch.where(gmask, global_emb, self.null_global.expand(global_emb.shape[0], -1))
        return tokens, global_emb


class VectorConditionEncoder(ConditionEncoderBase):
    def __init__(self, input_dim: int, hidden_dim: int, num_tokens: int) -> None:
        super().__init__(hidden_dim=hidden_dim, num_tokens=num_tokens)
        self.in_norm = nn.LayerNorm(int(input_dim))
        self.global_mlp = nn.Sequential(
            nn.Linear(int(input_dim), hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.token_mlp = None
        if self.num_tokens > 0:
            self.token_mlp = nn.Sequential(
                nn.Linear(int(input_dim), hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim * self.num_tokens),
            )

    def forward(self, x: torch.Tensor, exists: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.in_norm(x)
        global_emb = self.global_mlp(h)
        if self.num_tokens > 0 and self.token_mlp is not None:
            tokens = self.token_mlp(h).view(x.shape[0], self.num_tokens, self.hidden_dim)
        else:
            tokens = x.new_zeros((x.shape[0], 0, self.hidden_dim))
        return self._apply_mask(tokens, global_emb, exists)


class SequenceConditionEncoder(ConditionEncoderBase):
    def __init__(self, feature_dim: int, hidden_dim: int, num_tokens: int) -> None:
        super().__init__(hidden_dim=hidden_dim, num_tokens=num_tokens)
        self.proj = nn.Linear(int(feature_dim), hidden_dim)
        self.global_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, exists: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.proj(x)
        if self.num_tokens > 0:
            tokens = F.adaptive_avg_pool1d(tokens.transpose(1, 2), self.num_tokens).transpose(1, 2)
        else:
            tokens = x.new_zeros((x.shape[0], 0, self.hidden_dim))
        if tokens.shape[1] > 0:
            global_emb = self.global_proj(tokens.mean(dim=1))
        else:
            global_emb = self.global_proj(self.proj(x).mean(dim=1))
        return self._apply_mask(tokens, global_emb, exists)


class SpatialConditionEncoder(ConditionEncoderBase):
    def __init__(self, input_shape: Sequence[int], hidden_dim: int, patch_size: Sequence[int]) -> None:
        c, d, h, w = [int(v) for v in input_shape]
        pd, ph, pw = [int(v) for v in patch_size]
        if d % pd != 0 or h % ph != 0 or w % pw != 0:
            raise ValueError(f"Spatial condition shape {(c, d, h, w)} not divisible by patch size {(pd, ph, pw)}")
        gd, gh, gw = d // pd, h // ph, w // pw
        super().__init__(hidden_dim=hidden_dim, num_tokens=gd * gh * gw)
        self.patch_embed = nn.Conv3d(c, hidden_dim, kernel_size=(pd, ph, pw), stride=(pd, ph, pw))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_dim))
        self.global_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, exists: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed
        global_emb = self.global_proj(tokens.mean(dim=1))
        return self._apply_mask(tokens, global_emb, exists)


class TemporalSpatialConditionEncoder(ConditionEncoderBase):
    def __init__(self, input_shape: Sequence[int], hidden_dim: int, patch_size: Sequence[int]) -> None:
        tz, c, d, h, w = [int(v) for v in input_shape]
        pd, ph, pw = [int(v) for v in patch_size]
        if d % pd != 0 or h % ph != 0 or w % pw != 0:
            raise ValueError(f"Temporal condition shape {(tz, c, d, h, w)} not divisible by patch size {(pd, ph, pw)}")
        gd, gh, gw = d // pd, h // ph, w // pw
        super().__init__(hidden_dim=hidden_dim, num_tokens=tz * gd * gh * gw)
        self.tz = tz
        self.tokens_per_t = gd * gh * gw
        self.patch_embed = nn.Conv3d(c, hidden_dim, kernel_size=(pd, ph, pw), stride=(pd, ph, pw))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.tokens_per_t, hidden_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, tz, hidden_dim))
        self.global_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.time_embed, std=0.02)

    def forward(self, x: torch.Tensor, exists: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        b, tz, c, d, h, w = x.shape
        if tz != self.tz:
            raise ValueError(f"Temporal condition T mismatch: got {tz}, expected {self.tz}")
        h_bt = self.patch_embed(x.reshape(b * tz, c, d, h, w)).flatten(2).transpose(1, 2)
        tokens = h_bt.view(b, tz, self.tokens_per_t, self.hidden_dim)
        tokens = tokens + self.pos_embed.unsqueeze(1) + self.time_embed.unsqueeze(2)
        tokens = tokens.view(b, self.num_tokens, self.hidden_dim)
        global_emb = self.global_proj(tokens.mean(dim=1))
        return self._apply_mask(tokens, global_emb, exists)


def build_condition_encoder(name: str, input_shape: Sequence[int], cfg: Dict, hidden_dim: int) -> ConditionEncoderBase:
    mode = str(cfg.get("mode", "auto")).strip().lower()
    num_tokens = int(cfg.get("num_tokens", 4))
    shape = tuple(int(v) for v in input_shape)
    ndim = len(shape)

    if mode == "auto":
        if ndim == 1:
            mode = "vector"
        elif ndim == 2:
            mode = "sequence"
        elif ndim == 4:
            mode = "spatial"
        elif ndim == 5:
            mode = "temporal_spatial"
        else:
            raise ValueError(f"Unsupported auto mode for {name} shape={shape}")

    if mode == "vector":
        if ndim != 1:
            raise ValueError(f"{name} mode=vector expects ndim=1, got shape={shape}")
        return VectorConditionEncoder(input_dim=shape[0], hidden_dim=hidden_dim, num_tokens=num_tokens)

    if mode == "sequence":
        if ndim != 2:
            raise ValueError(f"{name} mode=sequence expects ndim=2, got shape={shape}")
        return SequenceConditionEncoder(feature_dim=shape[-1], hidden_dim=hidden_dim, num_tokens=num_tokens)

    if mode == "spatial":
        if ndim != 4:
            raise ValueError(f"{name} mode=spatial expects ndim=4, got shape={shape}")
        patch = cfg.get("patch_size", [1, 1, 1])
        return SpatialConditionEncoder(input_shape=shape, hidden_dim=hidden_dim, patch_size=patch)

    if mode == "temporal_spatial":
        if ndim != 5:
            raise ValueError(f"{name} mode=temporal_spatial expects ndim=5, got shape={shape}")
        patch = cfg.get("patch_size", [1, 1, 1])
        return TemporalSpatialConditionEncoder(input_shape=shape, hidden_dim=hidden_dim, patch_size=patch)

    raise ValueError(f"Unsupported condition encoder mode={mode} for {name}")


class DiTConditionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.modulator = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 9),
        )
        nn.init.zeros_(self.modulator[-1].weight)
        nn.init.zeros_(self.modulator[-1].bias)

    def forward(self, x: torch.Tensor, cond_tokens: Optional[torch.Tensor], global_cond: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2, shift3, scale3, gate3 = self.modulator(global_cond).chunk(9, dim=1)
        y = modulate(self.norm1(x), shift1, scale1)
        y, _ = self.self_attn(y, y, y, need_weights=False)
        x = x + torch.tanh(gate1).unsqueeze(1) * y

        if cond_tokens is not None and cond_tokens.shape[1] > 0:
            y = modulate(self.norm2(x), shift2, scale2)
            y, _ = self.cross_attn(y, cond_tokens, cond_tokens, need_weights=False)
            x = x + torch.tanh(gate2).unsqueeze(1) * y

        y = self.mlp(modulate(self.norm3(x), shift3, scale3))
        x = x + torch.tanh(gate3).unsqueeze(1) * y
        return x


class ConditionalLatentDiT(nn.Module):
    def __init__(
        self,
        *,
        target_shape: Sequence[int],
        patch_size: Sequence[int],
        hidden_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        condition_shapes: Dict[str, Optional[Sequence[int]]],
        condition_cfg: Dict,
        diversity_cfg: Dict,
        max_time_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.target_shape = tuple(int(v) for v in target_shape)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.max_time_steps = int(max_time_steps)
        self.condition_cfg = condition_cfg
        self.diversity_cfg = diversity_cfg

        audit = compute_patch_audit(self.target_shape, self.patch_size)
        self.latent_tz = int(audit["latent_tz"])
        self.latent_channels = int(audit["latent_channels"])
        self.grid_shape = tuple(int(v) for v in audit["grid_shape"])
        self.tokens_per_t = int(audit["tokens_per_t"])
        self.num_tokens = int(audit["patch_num"])

        pd, ph, pw = self.patch_size
        self.patch_embed = nn.Conv3d(self.latent_channels, hidden_dim, kernel_size=(pd, ph, pw), stride=(pd, ph, pw))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.tokens_per_t, hidden_dim))
        self.time_axis_embed = nn.Parameter(torch.zeros(1, self.latent_tz, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.time_axis_embed, std=0.02)

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.direction_embed = nn.Embedding(2, hidden_dim)
        self.global_cond_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        fusion_cfg = condition_cfg.get("global_fusion", {})
        if not isinstance(fusion_cfg, dict):
            fusion_cfg = {}
        self.global_fusion_mode = str(fusion_cfg.get("mode", "sum")).strip().lower()
        if self.global_fusion_mode not in {"sum", "attn"}:
            raise ValueError(f"Unsupported global_fusion.mode={self.global_fusion_mode}, expected one of ['sum','attn']")
        if self.global_fusion_mode == "attn":
            attn_heads = int(fusion_cfg.get("num_heads", 4))
            attn_dropout = float(fusion_cfg.get("dropout", 0.0))
            if attn_heads <= 0 or hidden_dim % attn_heads != 0:
                raise ValueError(
                    f"Invalid global_fusion.num_heads={attn_heads} for hidden_dim={hidden_dim}; must be >0 and divisible"
                )
            self.global_fusion_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.global_fusion_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=attn_heads, dropout=attn_dropout, batch_first=True
            )
            self.global_fusion_norm = nn.LayerNorm(hidden_dim)

        self.fc_encoder = None
        fc_shape = condition_shapes.get("fc", None)
        if fc_shape is not None and bool(condition_cfg.get("fc", {}).get("enabled", False)):
            self.fc_encoder = build_condition_encoder("fc", fc_shape, condition_cfg.get("fc", {}), hidden_dim)

        self.mri_encoder = None
        mri_shape = condition_shapes.get("mri", None)
        if mri_shape is not None and bool(condition_cfg.get("mri", {}).get("enabled", False)):
            self.mri_encoder = build_condition_encoder("mri", mri_shape, condition_cfg.get("mri", {}), hidden_dim)

        self.video_encoder = None
        video_shape = condition_shapes.get("video", None)
        if video_shape is not None and bool(condition_cfg.get("video", {}).get("enabled", False)):
            self.video_encoder = build_condition_encoder("video", video_shape, condition_cfg.get("video", {}), hidden_dim)

        self.audio_encoder = None
        audio_shape = condition_shapes.get("audio", None)
        if audio_shape is not None and bool(condition_cfg.get("audio", {}).get("enabled", False)):
            self.audio_encoder = build_condition_encoder("audio", audio_shape, condition_cfg.get("audio", {}), hidden_dim)

        self.meta_encoder = None
        meta_shape = condition_shapes.get("metadata", None)
        if meta_shape is not None and bool(condition_cfg.get("metadata", {}).get("enabled", False)):
            self.meta_encoder = build_condition_encoder("metadata", meta_shape, condition_cfg.get("metadata", {}), hidden_dim)

        self.blocks = nn.ModuleList(
            [DiTConditionBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, self.latent_channels * pd * ph * pw)

        self.long_residual = bool(diversity_cfg.get("enabled", False) and diversity_cfg.get("long_residual", False))
        self.long_residual_norms = nn.ModuleDict()
        self.long_residual_projs = nn.ModuleDict()
        self.long_residual_gates = nn.ParameterDict()
        if self.long_residual:
            for li in range(depth // 2, depth):
                key = str(li)
                self.long_residual_norms[key] = nn.LayerNorm(hidden_dim * 2)
                self.long_residual_projs[key] = nn.Linear(hidden_dim * 2, hidden_dim)
                self.long_residual_gates[key] = nn.Parameter(torch.tensor(0.0))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        if self.global_fusion_mode == "attn":
            nn.init.trunc_normal_(self.global_fusion_query, std=0.02)

    def _fuse_globals(self, globals_: list[torch.Tensor], bsz: int) -> torch.Tensor:
        if len(globals_) == 0:
            base = torch.zeros((bsz, self.hidden_dim), device=self.pos_embed.device)
            return self.global_cond_proj(base)

        if self.global_fusion_mode == "attn":
            # Per-sample modality-level attention over [fc, mri, metadata] global vectors.
            g = torch.stack(globals_, dim=1)  # (B, M, D)
            q = self.global_fusion_query.expand(bsz, -1, -1)  # (B, 1, D)
            fused, _ = self.global_fusion_attn(q, g, g, need_weights=False)
            base = self.global_fusion_norm(fused.squeeze(1))
            return self.global_cond_proj(base)

        return self.global_cond_proj(torch.stack(globals_, dim=0).sum(dim=0))

    def _patchify_target(self, x: torch.Tensor) -> torch.Tensor:
        b, tz, c, d, h, w = x.shape
        if (tz, c, d, h, w) != self.target_shape:
            raise ValueError(f"Target latent shape mismatch: got {(tz, c, d, h, w)}, expected {self.target_shape}")
        y = self.patch_embed(x.reshape(b * tz, c, d, h, w)).flatten(2).transpose(1, 2)
        y = y.reshape(b, tz, self.tokens_per_t, self.hidden_dim)
        y = y + self.pos_embed.unsqueeze(1) + self.time_axis_embed.unsqueeze(2)
        return y.reshape(b, self.num_tokens, self.hidden_dim)

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        b = tokens.shape[0]
        gd, gh, gw = self.grid_shape
        pd, ph, pw = self.patch_size
        x = tokens.view(b, self.latent_tz, self.tokens_per_t, self.latent_channels * pd * ph * pw)
        x = x.view(b, self.latent_tz, gd, gh, gw, self.latent_channels, pd, ph, pw)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4, 8).contiguous()
        return x.view(b, self.latent_tz, self.latent_channels, gd * pd, gh * ph, gw * pw)

    def _encode_conditions(
        self,
        *,
        fc_cond: Optional[torch.Tensor],
        has_fc: Optional[torch.Tensor],
        mri_cond: Optional[torch.Tensor],
        has_mri: Optional[torch.Tensor],
        video_cond: Optional[torch.Tensor],
        has_video: Optional[torch.Tensor],
        audio_cond: Optional[torch.Tensor],
        has_audio: Optional[torch.Tensor],
        meta_cond: Optional[torch.Tensor],
        has_meta: Optional[torch.Tensor],
        batch_size: Optional[int] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        tokens: list[torch.Tensor] = []
        globals_: list[torch.Tensor] = []
        bsz = 0

        if fc_cond is not None:
            bsz = int(fc_cond.shape[0])
        elif mri_cond is not None:
            bsz = int(mri_cond.shape[0])
        elif video_cond is not None:
            bsz = int(video_cond.shape[0])
        elif audio_cond is not None:
            bsz = int(audio_cond.shape[0])
        elif meta_cond is not None:
            bsz = int(meta_cond.shape[0])
        elif has_fc is not None:
            bsz = int(has_fc.shape[0])
        elif has_mri is not None:
            bsz = int(has_mri.shape[0])
        elif has_video is not None:
            bsz = int(has_video.shape[0])
        elif has_audio is not None:
            bsz = int(has_audio.shape[0])
        elif has_meta is not None:
            bsz = int(has_meta.shape[0])
        elif batch_size is not None:
            bsz = int(batch_size)

        if self.fc_encoder is not None and fc_cond is not None:
            fc_tokens, fc_global = self.fc_encoder(fc_cond, has_fc)
            if fc_tokens.shape[1] > 0:
                tokens.append(fc_tokens)
            globals_.append(fc_global)

        if self.mri_encoder is not None and mri_cond is not None:
            mri_tokens, mri_global = self.mri_encoder(mri_cond, has_mri)
            if mri_tokens.shape[1] > 0:
                tokens.append(mri_tokens)
            globals_.append(mri_global)

        if self.video_encoder is not None and video_cond is not None:
            video_tokens, video_global = self.video_encoder(video_cond, has_video)
            if video_tokens.shape[1] > 0:
                tokens.append(video_tokens)
            globals_.append(video_global)

        if self.audio_encoder is not None and audio_cond is not None:
            audio_tokens, audio_global = self.audio_encoder(audio_cond, has_audio)
            if audio_tokens.shape[1] > 0:
                tokens.append(audio_tokens)
            globals_.append(audio_global)

        if self.meta_encoder is not None and meta_cond is not None:
            meta_tokens, meta_global = self.meta_encoder(meta_cond, has_meta)
            if meta_tokens.shape[1] > 0:
                tokens.append(meta_tokens)
            globals_.append(meta_global)

        cond_tokens = None if len(tokens) == 0 else torch.cat(tokens, dim=1)
        if bsz <= 0:
            raise ValueError("Unable to infer batch size for condition encoding")
        global_cond = self._fuse_globals(globals_, bsz)
        return cond_tokens, global_cond

    @torch.inference_mode()
    def debug_condition_encoding(
        self,
        *,
        fc_cond: Optional[torch.Tensor] = None,
        has_fc: Optional[torch.Tensor] = None,
        mri_cond: Optional[torch.Tensor] = None,
        has_mri: Optional[torch.Tensor] = None,
        video_cond: Optional[torch.Tensor] = None,
        has_video: Optional[torch.Tensor] = None,
        audio_cond: Optional[torch.Tensor] = None,
        has_audio: Optional[torch.Tensor] = None,
        meta_cond: Optional[torch.Tensor] = None,
        has_meta: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, object]:
        bsz = 0
        for candidate in (fc_cond, mri_cond, video_cond, audio_cond, meta_cond, has_fc, has_mri, has_video, has_audio, has_meta):
            if candidate is not None:
                bsz = int(candidate.shape[0])
                break
        if bsz <= 0 and batch_size is not None:
            bsz = int(batch_size)
        if bsz <= 0:
            raise ValueError("Unable to infer batch size for debug_condition_encoding")

        per_modality: Dict[str, Dict[str, object]] = {}
        tokens: list[torch.Tensor] = []
        globals_: list[torch.Tensor] = []
        specs = (
            ("fc", self.fc_encoder, fc_cond, has_fc),
            ("mri", self.mri_encoder, mri_cond, has_mri),
            ("video", self.video_encoder, video_cond, has_video),
            ("audio", self.audio_encoder, audio_cond, has_audio),
            ("metadata", self.meta_encoder, meta_cond, has_meta),
        )
        for name, encoder, cond, exists in specs:
            if encoder is None or cond is None:
                per_modality[name] = {
                    "enabled": bool(encoder is not None),
                    "present": False,
                    "token_shape": None,
                    "global_shape": None,
                    "tokens": None,
                    "global": None,
                    "has": (None if exists is None else exists.detach().cpu()),
                }
                continue
            modal_tokens, modal_global = encoder(cond, exists)
            if modal_tokens.shape[1] > 0:
                tokens.append(modal_tokens)
            globals_.append(modal_global)
            per_modality[name] = {
                "enabled": True,
                "present": True,
                "token_shape": [int(v) for v in modal_tokens.shape],
                "global_shape": [int(v) for v in modal_global.shape],
                "tokens": modal_tokens.detach().cpu(),
                "global": modal_global.detach().cpu(),
                "has": (None if exists is None else exists.detach().cpu()),
            }

        cond_tokens = None if len(tokens) == 0 else torch.cat(tokens, dim=1)
        global_cond = self._fuse_globals(globals_, bsz)
        return {
            "cond_tokens": (None if cond_tokens is None else cond_tokens.detach().cpu()),
            "global_cond": global_cond.detach().cpu(),
            "per_modality": per_modality,
        }

    def _apply_long_residual(self, layer_idx: int, x: torch.Tensor, early_cache: Dict[int, torch.Tensor]) -> torch.Tensor:
        if not self.long_residual:
            return x
        pair_idx = self.depth - 1 - int(layer_idx)
        if pair_idx not in early_cache:
            return x
        key = str(layer_idx)
        cat = torch.cat([x, early_cache[pair_idx]], dim=-1)
        delta = self.long_residual_projs[key](self.long_residual_norms[key](cat))
        return x + torch.tanh(self.long_residual_gates[key]) * delta

    def forward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        direction_id: torch.Tensor,
        *,
        fc_cond: Optional[torch.Tensor] = None,
        has_fc: Optional[torch.Tensor] = None,
        mri_cond: Optional[torch.Tensor] = None,
        has_mri: Optional[torch.Tensor] = None,
        video_cond: Optional[torch.Tensor] = None,
        has_video: Optional[torch.Tensor] = None,
        audio_cond: Optional[torch.Tensor] = None,
        has_audio: Optional[torch.Tensor] = None,
        meta_cond: Optional[torch.Tensor] = None,
        has_meta: Optional[torch.Tensor] = None,
        return_hiddens: bool = False,
        capture_layers: Optional[Sequence[int]] = None,
    ) -> torch.Tensor | Dict[str, object]:
        y = self._patchify_target(xt)
        cond_tokens, cond_global = self._encode_conditions(
            fc_cond=fc_cond,
            has_fc=has_fc,
            mri_cond=mri_cond,
            has_mri=has_mri,
            video_cond=video_cond,
            has_video=has_video,
            audio_cond=audio_cond,
            has_audio=has_audio,
            meta_cond=meta_cond,
            has_meta=has_meta,
            batch_size=int(xt.shape[0]),
        )

        global_cond = cond_global + self.time_mlp(timestep_embedding(t, self.hidden_dim)) + self.direction_embed(direction_id)

        if capture_layers is None:
            capture_set = {int(v) for v in self.diversity_cfg.get("capture_layers", [])}
        else:
            capture_set = {int(v) for v in capture_layers}
        early_cache: Dict[int, torch.Tensor] = {}
        hiddens: Dict[int, torch.Tensor] = {}
        for li, blk in enumerate(self.blocks):
            y = self._apply_long_residual(li, y, early_cache)
            y = blk(y, cond_tokens, global_cond)
            if li < self.depth // 2:
                early_cache[li] = y
            if return_hiddens and li in capture_set:
                hiddens[int(li)] = y

        pred = self._unpatchify(self.head(self.norm(y)))
        if return_hiddens:
            return {"pred": pred, "hiddens": hiddens}
        return pred


def _pair_hidden_layers(hiddens: Dict[int, torch.Tensor], pair_strategy: str) -> list[tuple[int, int]]:
    layers = sorted(int(k) for k in hiddens.keys())
    if len(layers) < 2:
        return []
    if str(pair_strategy).lower() == "symmetric":
        pairs: list[tuple[int, int]] = []
        for i in range(len(layers) // 2):
            a = layers[i]
            b = layers[-1 - i]
            if a != b:
                pairs.append((a, b))
        return pairs
    out: list[tuple[int, int]] = []
    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            out.append((layers[i], layers[j]))
    return out


def compute_diversity_terms(hiddens: Dict[int, torch.Tensor], cfg: Dict) -> Dict[str, torch.Tensor]:
    if len(hiddens) < 2:
        zero = next(iter(hiddens.values())).new_zeros(()) if len(hiddens) > 0 else torch.tensor(0.0)
        return {"orth": zero, "mi": zero, "disp": zero, "total": zero}

    pair_strategy = str(cfg.get("pair_strategy", "symmetric"))
    pairs = _pair_hidden_layers(hiddens, pair_strategy)
    if len(pairs) == 0:
        zero = next(iter(hiddens.values())).new_zeros(())
        return {"orth": zero, "mi": zero, "disp": zero, "total": zero}

    orth_terms = []
    mi_terms = []
    disp_terms = []
    for a, b in pairs:
        ha = hiddens[a]
        hb = hiddens[b]
        ma = F.normalize(ha.mean(dim=1), dim=-1)
        mb = F.normalize(hb.mean(dim=1), dim=-1)
        orth_terms.append((ma * mb).sum(dim=-1).abs().mean())

        ta = F.normalize(ha, dim=-1)
        tb = F.normalize(hb, dim=-1)
        mi_terms.append((ta * tb).sum(dim=-1).abs().mean())

    for h in hiddens.values():
        act = h.abs().mean(dim=(0, 1))
        act = act / act.max().clamp_min(1.0e-6)
        disp_terms.append(-act.var(unbiased=False))

    l_orth = torch.stack(orth_terms).mean()
    l_mi = torch.stack(mi_terms).mean()
    l_disp = torch.stack(disp_terms).mean()

    loss_terms = {str(x).lower() for x in cfg.get("loss_terms", ["orth", "mi", "disp"])}
    total = l_orth.new_zeros(())
    if "orth" in loss_terms:
        total = total + l_orth
    if "mi" in loss_terms:
        total = total + l_mi
    if "disp" in loss_terms:
        total = total + l_disp
    return {"orth": l_orth, "mi": l_mi, "disp": l_disp, "total": total}
