from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .diffusion import GaussianDiffusion
from .noise_utils import NoiseMode, noise_for_batch


def _unwrap_module(m: nn.Module) -> nn.Module:
    return m.module if hasattr(m, "module") else m


def resolve_capture_layers(depth: int, capture_layers: Sequence[int]) -> List[int]:
    d = int(depth)
    out: List[int] = []
    for lid in capture_layers:
        i = int(lid)
        if i < 0:
            i = d + i
        if i < 0 or i >= d:
            raise ValueError(f"capture_layer out of range: got {lid} -> {i}, depth={d}")
        out.append(i)
    return out


@dataclass(frozen=True)
class FeatureProtocolCondConfig:
    timestep: int
    capture_layers: List[int]
    noise_mode: NoiseMode = "per_subject"
    noise_seed: int = 0


@dataclass(frozen=True)
class FeatureProtocolOutput:
    tokens_list: List[torch.Tensor]
    meta: Dict[str, Any]


class FeatureProtocolCond:
    def __init__(
        self,
        *,
        model: nn.Module,
        diffusion: GaussianDiffusion,
        cfg: FeatureProtocolCondConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.diffusion = diffusion
        self.cfg = cfg
        self.device = device

    def tokens_from_batch(
        self,
        x0: torch.Tensor,
        *,
        direction_id: torch.Tensor,
        cond_inputs: Dict[str, Optional[torch.Tensor]],
        subjects: Sequence[str],
        sample_indices: Optional[Sequence[int]] = None,
        enable_grad: bool = False,
        timestep: Optional[int] = None,
    ) -> FeatureProtocolOutput:
        x0 = x0.to(device=self.device)
        direction_id = direction_id.to(device=self.device, dtype=torch.long)

        bsz = int(x0.shape[0])
        t = int(self.cfg.timestep if timestep is None else timestep)
        t_vec = torch.full((bsz,), t, device=self.device, dtype=torch.int64)

        if len(subjects) != bsz:
            raise ValueError(f"subjects length must match batch size: got {len(subjects)} vs {bsz}")

        idx_list = None
        if sample_indices is not None:
            if len(sample_indices) != bsz:
                raise ValueError("sample_indices length must match batch size")
            idx_list = [int(i) for i in sample_indices]

        eps = noise_for_batch(
            mode=self.cfg.noise_mode,
            subjects=[str(s) for s in subjects],
            sample_indices=idx_list,
            global_seed=int(self.cfg.noise_seed),
            shape_per_sample=tuple(x0.shape[1:]),
            device=self.device,
            dtype=x0.dtype,
        )
        xt, _ = self.diffusion.q_sample(x0, t_vec, noise=eps)

        core = _unwrap_module(self.model)
        layers = resolve_capture_layers(int(getattr(core, "depth")), self.cfg.capture_layers)

        if enable_grad:
            out = self.model(
                xt,
                t_vec,
                direction_id,
                fc_cond=cond_inputs.get("fc_cond", None),
                has_fc=cond_inputs.get("has_fc", None),
                mri_cond=cond_inputs.get("mri_cond", None),
                has_mri=cond_inputs.get("has_mri", None),
                meta_cond=cond_inputs.get("meta_cond", None),
                has_meta=cond_inputs.get("has_meta", None),
                return_hiddens=True,
                capture_layers=layers,
            )
        else:
            with torch.no_grad():
                out = self.model(
                    xt,
                    t_vec,
                    direction_id,
                    fc_cond=cond_inputs.get("fc_cond", None),
                    has_fc=cond_inputs.get("has_fc", None),
                    mri_cond=cond_inputs.get("mri_cond", None),
                    has_mri=cond_inputs.get("has_mri", None),
                    meta_cond=cond_inputs.get("meta_cond", None),
                    has_meta=cond_inputs.get("has_meta", None),
                    return_hiddens=True,
                    capture_layers=layers,
                )

        hiddens = out.get("hiddens", None) if isinstance(out, dict) else getattr(out, "hiddens", None)
        if hiddens is None:
            raise RuntimeError("Expected hiddens when return_hiddens=True")

        tokens_list: List[torch.Tensor] = []
        for lid in layers:
            if lid not in hiddens:
                raise RuntimeError(f"Requested hidden layer not captured: {lid}")
            tokens_list.append(hiddens[lid])

        meta: Dict[str, Any] = {
            "timestep": int(t),
            "capture_layers_resolved": layers,
            "noise_mode": str(self.cfg.noise_mode),
            "noise_seed": int(self.cfg.noise_seed),
        }
        return FeatureProtocolOutput(tokens_list=tokens_list, meta=meta)
