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
import time


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def make_timestamped_dir(base_dir: str) -> str:
    ensure_dir(base_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(base_dir, stamp)
    if not os.path.exists(out):
        return ensure_dir(out)
    idx = 1
    while True:
        cand = os.path.join(base_dir, f"{stamp}_{idx:02d}")
        if not os.path.exists(cand):
            return ensure_dir(cand)
        idx += 1


def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return _expand_env_tree(json.load(f))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(cfg: Dict[str, Any]) -> torch.device:
    dev = str(cfg.get("device", "auto"))
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(dev)
