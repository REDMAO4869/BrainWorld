from __future__ import annotations

import hashlib
from typing import List, Literal, Optional, Tuple

import torch

NoiseMode = Literal["fixed", "per_subject", "per_sample_index"]


def _stable_hash_to_u64(text: str) -> int:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="little", signed=False)


def _make_generator(*, device: torch.device, seed: int) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed) & 0x7FFFFFFF)
    return g


def noise_for_sample(
    *,
    mode: NoiseMode,
    subject_id: str,
    sample_index: Optional[int],
    global_seed: int,
    shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    fixed_cache: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mode = str(mode)
    if mode not in ("fixed", "per_subject", "per_sample_index"):
        raise ValueError("noise_mode must be one of: fixed|per_subject|per_sample_index")

    if mode == "fixed":
        if fixed_cache is not None:
            return fixed_cache
        g = _make_generator(device=device, seed=int(global_seed))
        return torch.randn(shape, generator=g, device=device, dtype=dtype)

    if mode == "per_sample_index":
        if sample_index is None:
            raise ValueError("sample_index is required for noise_mode=per_sample_index")
        seed = int(global_seed) + int(sample_index)
        g = _make_generator(device=device, seed=seed)
        return torch.randn(shape, generator=g, device=device, dtype=dtype)

    seed = _stable_hash_to_u64(str(subject_id)) ^ int(global_seed)
    g = _make_generator(device=device, seed=int(seed))
    return torch.randn(shape, generator=g, device=device, dtype=dtype)


def noise_for_batch(
    *,
    mode: NoiseMode,
    subjects: List[str],
    sample_indices: Optional[List[int]],
    global_seed: int,
    shape_per_sample: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    fixed_cache = None
    if mode == "fixed":
        fixed_cache = noise_for_sample(
            mode=mode,
            subject_id="fixed",
            sample_index=0,
            global_seed=global_seed,
            shape=shape_per_sample,
            device=device,
            dtype=dtype,
            fixed_cache=None,
        )

    eps_list: List[torch.Tensor] = []
    for i, sid in enumerate(subjects):
        idx = None if sample_indices is None else int(sample_indices[i])
        eps = noise_for_sample(
            mode=mode,
            subject_id=str(sid),
            sample_index=idx,
            global_seed=global_seed,
            shape=shape_per_sample,
            device=device,
            dtype=dtype,
            fixed_cache=fixed_cache,
        )
        eps_list.append(eps)

    return torch.stack(eps_list, dim=0)
