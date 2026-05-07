from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return sum(parameter.numel() for parameter in module.parameters())


def cycle(loader, sampler=None):
    epoch = 0
    while True:
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def extract_state_dict(payload: Any, preferred_key: str | None = None) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        if preferred_key and preferred_key in payload and isinstance(payload[preferred_key], dict):
            return payload[preferred_key]
        for key in (
            "state_dict",
            "model_state_dict",
            "model",
            "network",
            "autoencoder",
            "vqvae",
            "unet",
            "ema_model_state_dict",
        ):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint payload must be a dict-like state dict, got {type(payload)!r}")
    return payload


def _strip_prefix(key: str) -> str:
    for prefix in (
        "module.",
        "model.",
        "network.",
        "vqvae.",
        "autoencoder.",
        "unet.",
    ):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def load_partial_weights(
    module: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = False,
    preferred_key: str | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(payload, preferred_key=preferred_key)
    current_state = module.state_dict()

    matched: dict[str, torch.Tensor] = {}
    missing_in_checkpoint: list[str] = []
    unexpected_in_checkpoint: list[str] = []
    shape_mismatch: list[str] = []

    for raw_key, value in state_dict.items():
        key = _strip_prefix(raw_key)
        if key not in current_state:
            unexpected_in_checkpoint.append(key)
            continue
        if current_state[key].shape != value.shape:
            shape_mismatch.append(
                f"{key}: checkpoint {tuple(value.shape)} != model {tuple(current_state[key].shape)}"
            )
            continue
        matched[key] = value

    for key in current_state:
        if key not in matched:
            missing_in_checkpoint.append(key)

    merged = current_state.copy()
    merged.update(matched)
    module.load_state_dict(merged, strict=False)

    report = {
        "checkpoint_path": str(checkpoint_path),
        "matched_keys": len(matched),
        "missing_keys": missing_in_checkpoint,
        "unexpected_keys": unexpected_in_checkpoint,
        "shape_mismatch": shape_mismatch,
    }

    if strict and (missing_in_checkpoint or unexpected_in_checkpoint or shape_mismatch):
        raise RuntimeError(f"Strict checkpoint loading failed: {report}")
    return report


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
