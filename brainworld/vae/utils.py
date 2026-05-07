from __future__ import annotations

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def _repo_root() -> str:
    return str(Path(__file__).resolve().parents[2])


def _expand_env_tree(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand_env_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_tree(v) for v in obj]
    if isinstance(obj, str):
        v = obj.replace('__REPO_ROOT__', _repo_root())
        return os.path.expandvars(v)
    return obj



def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return _expand_env_tree(json.load(f))


def save_json(obj: Dict, path: str) -> None:
    parent = str(Path(path).parent)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def mse(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.mean((x - y) ** 2).item())


def mae(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(x - y)).item())


def psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1.0e-8) -> float:
    m = torch.mean((x - y) ** 2)
    if float(m.item()) < eps:
        return 99.0
    vmax = torch.max(torch.abs(y)).clamp_min(1.0)
    p = 20.0 * torch.log10(vmax) - 10.0 * torch.log10(m.clamp_min(eps))
    return float(p.item())


def fg_bg_metrics(pred: torch.Tensor, gt: torch.Tensor, fg_mask: torch.Tensor) -> Dict[str, float]:
    fg = fg_mask.float()
    bg = 1.0 - fg

    fg_n = fg.sum().clamp_min(1.0)
    bg_n = bg.sum().clamp_min(1.0)

    err2 = (pred - gt) ** 2
    fg_mse = float((err2 * fg).sum().item() / float(fg_n.item()))
    bg_mse = float((err2 * bg).sum().item() / float(bg_n.item()))
    bg_abs_mean = float((torch.abs(pred) * bg).sum().item() / float(bg_n.item()))

    return {
        "fg_mse": fg_mse,
        "bg_mse": bg_mse,
        "bg_abs_mean": bg_abs_mean,
    }
