from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from brainworld.dit.data import audit_to_dict, build_splits_from_config, collate_batch
from brainworld.dit.diffusion import GaussianDiffusion, normalize_schedule_type
from brainworld.dit.downstream_aggregators import build_layer_aggregator
from brainworld.dit.downstream_heads import build_head
from brainworld.dit.downstream_metrics import evaluate_classification, evaluate_regression
from brainworld.dit.downstream_protocol import FeatureProtocolCond, FeatureProtocolCondConfig, resolve_capture_layers
from brainworld.dit.model import ConditionalLatentDiT, compute_patch_audit
from brainworld.dit.utils import ensure_dir, load_json, resolve_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BrainWorld conditional diffusion downstream")
    p.add_argument("--config", required=True, help="Path to JSON config")
    return p.parse_args()


@dataclass(frozen=True)
class DistContext:
    enabled: bool
    world_size: int
    rank: int
    local_rank: int

    @property
    def is_main(self) -> bool:
        return int(self.rank) == 0


def _setup_dist_context(cfg: Dict[str, Any], gpu_ids: List[int]) -> Tuple[DistContext, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ddp_env = world_size > 1

    if ddp_env:
        if not torch.cuda.is_available():
            raise RuntimeError("WORLD_SIZE>1 but CUDA is not available")
        cuda_count = int(torch.cuda.device_count())
        if local_rank < 0 or local_rank >= cuda_count:
            raise ValueError(
                f"LOCAL_RANK out of range for visible CUDA devices ({cuda_count}): {local_rank}"
            )

        torch.cuda.set_device(local_rank)
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available")
        if not dist.is_initialized():
            ddp_cfg = cfg.get("ddp", {}) if isinstance(cfg.get("ddp", {}), dict) else {}
            timeout_sec = int(ddp_cfg.get("timeout_sec", 180))
            if timeout_sec <= 0:
                timeout_sec = 180
            timeout = timedelta(seconds=int(timeout_sec))
            try:
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    device_id=local_rank,
                    timeout=timeout,
                )
            except TypeError:
                try:
                    dist.init_process_group(backend="nccl", init_method="env://", timeout=timeout)
                except TypeError:
                    dist.init_process_group(backend="nccl", init_method="env://")

        return DistContext(enabled=True, world_size=world_size, rank=rank, local_rank=local_rank), torch.device(
            f"cuda:{local_rank}"
        )

    cuda_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if len(gpu_ids) > 0:
        if not torch.cuda.is_available():
            raise RuntimeError("gpu_ids is provided but CUDA is not available")
        bad_ids = [int(g) for g in gpu_ids if int(g) < 0 or int(g) >= int(cuda_count)]
        if len(bad_ids) > 0:
            raise ValueError(f"gpu_ids out of range for visible CUDA devices ({cuda_count}): {bad_ids}")
        device = torch.device(f"cuda:{int(gpu_ids[0])}")
    else:
        device = resolve_device(cfg)

    return DistContext(enabled=False, world_size=1, rank=0, local_rank=0), device


def _distributed_sampler(
    ds: Dataset,
    *,
    dist_ctx: DistContext,
    shuffle: bool,
) -> Optional[DistributedSampler]:
    if not dist_ctx.enabled:
        return None
    return DistributedSampler(
        ds,
        num_replicas=int(dist_ctx.world_size),
        rank=int(dist_ctx.rank),
        shuffle=bool(shuffle),
        drop_last=False,
    )


def _all_reduce_sum_scalar(value: float, *, device: torch.device, dist_ctx: DistContext) -> float:
    if not dist_ctx.enabled:
        return float(value)
    t = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def _dist_barrier(dist_ctx: DistContext) -> None:
    if not dist_ctx.enabled:
        return
    if torch.cuda.is_available():
        try:
            dist.barrier(device_ids=[int(dist_ctx.local_rank)])
            return
        except TypeError:
            pass
    dist.barrier()


def _abort_dist_process_group(dist_ctx: DistContext, *, reason: str = "") -> None:
    if not dist_ctx.enabled:
        return
    if (not dist.is_available()) or (not dist.is_initialized()):
        return

    if dist_ctx.is_main and str(reason).strip() != "":
        print(f"[fatal][ddp] aborting process group: {reason}")

    # Best-effort fast abort path; fall back to destroy if abort API is unavailable.
    try:
        abort_fn = getattr(dist, "abort", None)
        if callable(abort_fn):
            abort_fn()
            return
    except Exception:
        pass

    try:
        pg = getattr(dist, "distributed_c10d", None)
        if pg is not None:
            get_default_group = getattr(pg, "_get_default_group", None)
            if callable(get_default_group):
                group = get_default_group()
                group_abort = getattr(group, "abort", None)
                if callable(group_abort):
                    group_abort()
                    return
    except Exception:
        pass

    try:
        dist.destroy_process_group()
    except Exception:
        pass


def _norm_col_name(x: str) -> str:
    return str(x).replace("﻿", "").replace("`", "").strip().lower()


def _resolve_csv_col(fieldnames: List[str], wanted: str) -> str:
    norm2raw = {_norm_col_name(c): c for c in fieldnames}
    raw = norm2raw.get(_norm_col_name(wanted), None)
    if raw is None:
        raise ValueError(f"Column '{wanted}' not found in CSV header: {fieldnames}")
    return raw


def _clean_subject(s: str) -> str:
    return str(s).strip()


def _parse_t_list(emb_cfg: Dict[str, Any]) -> List[int]:
    t_list_cfg = emb_cfg.get("t_list", None)
    if t_list_cfg is None:
        t_list = [int(emb_cfg.get("timestep", 10))]
    elif isinstance(t_list_cfg, str):
        t_list = [int(x) for x in t_list_cfg.split(",") if x.strip() != ""]
    elif isinstance(t_list_cfg, (int, float)):
        t_list = [int(t_list_cfg)]
    else:
        t_list = [int(x) for x in list(t_list_cfg)]
    if len(t_list) == 0:
        t_list = [int(emb_cfg.get("timestep", 10))]
    return t_list


def _subject_id_from_base_ds(base_ds: Dataset, idx: int) -> str:
    # Fast path: avoid touching heavy __getitem__ when dataset stores pair metadata.
    pair_samples = getattr(base_ds, "pair_samples", None)
    if pair_samples is not None:
        try:
            ps = pair_samples[int(idx)]
            sid = getattr(ps, "subject_id", None)
            if sid is not None:
                return str(sid)
        except Exception:
            pass

    item = base_ds[int(idx)]
    return str(item["subject_id"])


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k, None), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _safe_name(x: str) -> str:
    s = str(x).strip()
    out = []
    for c in s:
        if c.isalnum() or c in {"-", "_"}:
            out.append(c)
        else:
            out.append("_")
    s2 = "".join(out).strip("_")
    return s2 if s2 else "task"


def _short_sha1_text(text: str, n: int = 8) -> str:
    h = hashlib.sha1(str(text).encode("utf-8")).hexdigest()
    return str(h[: max(4, int(n))])


def _config_tag(config_path: str) -> str:
    base = os.path.basename(str(config_path))
    stem, _ = os.path.splitext(base)
    return _safe_name(stem)


def _checkpoint_tag(ckpt_path: str) -> str:
    p = os.path.abspath(str(ckpt_path))
    b1 = os.path.basename(os.path.dirname(p))
    b2 = os.path.basename(os.path.dirname(os.path.dirname(p)))
    # Typical structure: .../<run_stamp>/checkpoints/best.pt
    if str(b1).lower() == "checkpoints" and b2 != "":
        return _safe_name(b2)
    stem, _ = os.path.splitext(os.path.basename(p))
    return _safe_name(stem)


def _make_run_output_dir(*, out_root: str, config_path: str, ckpt_path: str, run_name: Optional[str] = None) -> str:
    ensure_dir(out_root)
    if run_name is not None and str(run_name).strip() != "":
        stem = _safe_name(str(run_name))
    else:
        cfg_tag = _config_tag(config_path)
        ckpt_tag = _checkpoint_tag(ckpt_path)
        stem = _safe_name(f"{cfg_tag}__{ckpt_tag}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(out_root, f"{stem}__{stamp}")
    if not os.path.exists(base):
        return ensure_dir(base)

    idx = 1
    while True:
        cand = os.path.join(out_root, f"{stem}__{stamp}_{idx:02d}")
        if not os.path.exists(cand):
            return ensure_dir(cand)
        idx += 1


def _default_cache_root(*, cfg: Dict[str, Any], ckpt_path: str) -> str:
    out_cfg = cfg.get("output", {}) if isinstance(cfg.get("output", {}), dict) else {}
    explicit = str(out_cfg.get("cache_root", "")).strip()
    if explicit != "":
        return os.path.abspath(explicit)

    cfg_path = str(cfg.get("_config_path", "config.json"))
    tag = _safe_name(f"{_config_tag(cfg_path)}__{_checkpoint_tag(ckpt_path)}")
    return os.path.abspath(os.path.join("outputs", "downstream_feature_cache", tag))


def _resolve_tasks(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    tasks = cfg.get("tasks", None)
    if tasks is None:
        tasks = cfg.get("task_list", None)
    if tasks is None:
        base = {
            "name": str(cfg.get("name", "default_task")),
            "label": copy.deepcopy(cfg.get("label", {})),
            "task": copy.deepcopy(cfg.get("task_head", cfg.get("task", {}))),
        }
        tasks = [base]

    if not isinstance(tasks, list) or len(tasks) == 0:
        raise ValueError("tasks/task_list must be a non-empty list")

    out: List[Dict[str, Any]] = []
    for t in tasks:
        if not isinstance(t, dict):
            raise ValueError("each task must be a dict")
        enabled_raw = t.get("enabled", True)
        if isinstance(enabled_raw, str):
            enabled = enabled_raw.strip().lower() not in {"0", "false", "no", "off"}
        else:
            enabled = bool(enabled_raw)
        if enabled:
            out.append(t)

    if len(out) == 0:
        raise ValueError("all tasks are disabled")
    return out


def _read_label_map(
    *,
    path: str,
    subject_col: str,
    target_col: str,
    task_type: str,
    label_map: Optional[Dict[str, Any]],
    duplicate_policy_cls: str,
    duplicate_policy_reg: str,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    values: Dict[str, List[float]] = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"label csv has no header: {path}")

        sub_raw = _resolve_csv_col(reader.fieldnames, subject_col)
        tgt_raw = _resolve_csv_col(reader.fieldnames, target_col)

        for row in reader:
            sid = _clean_subject(row.get(sub_raw, ""))
            tv_raw = _clean_subject(row.get(tgt_raw, ""))
            if sid == "" or tv_raw == "":
                continue

            tv = tv_raw
            if label_map is not None and tv in label_map:
                tv = str(label_map[tv])

            try:
                v = float(tv)
            except Exception:
                continue
            values[sid].append(v)

    out: Dict[str, float] = {}
    dup_subjects = 0
    for sid, arr in values.items():
        if len(arr) > 1:
            dup_subjects += 1
        if task_type == "classification":
            ints = [int(round(x)) for x in arr]
            if duplicate_policy_cls == "first":
                out[sid] = float(ints[0])
            else:
                out[sid] = float(Counter(ints).most_common(1)[0][0])
        else:
            if duplicate_policy_reg == "first":
                out[sid] = float(arr[0])
            else:
                out[sid] = float(np.mean(arr))

    audit = {
        "label_subjects": len(out),
        "duplicate_subjects": dup_subjects,
        "duplicate_policy_cls": duplicate_policy_cls,
        "duplicate_policy_reg": duplicate_policy_reg,
    }
    return out, audit


class LabeledPairDataset(Dataset):
    def __init__(
        self,
        *,
        base_ds: Dataset,
        split_name: str,
        target_map: Dict[str, float],
        task_type: str,
        target_norm: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.base_ds = base_ds
        self.split_name = str(split_name)
        self.task_type = str(task_type)
        self.target_norm = target_norm

        keep: List[int] = []
        yvals: List[float] = []
        missing = 0
        for idx in range(len(self.base_ds)):
            sid = _subject_id_from_base_ds(self.base_ds, idx)
            if sid not in target_map:
                missing += 1
                continue
            y = float(target_map[sid])
            if self.task_type == "regression" and self.target_norm is not None:
                mu, sd = self.target_norm
                y = (y - float(mu)) / float(sd)
            keep.append(int(idx))
            yvals.append(float(y))

        if len(keep) == 0:
            raise RuntimeError(f"No labeled samples for split={self.split_name}")

        self.keep_indices = keep
        self.yvals = np.asarray(yvals, dtype=np.float32)
        self.audit = {
            "split": self.split_name,
            "requested_samples": int(len(self.base_ds)),
            "used_samples": int(len(self.keep_indices)),
            "missing_label_samples": int(missing),
        }

    def __len__(self) -> int:
        return len(self.keep_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base_idx = int(self.keep_indices[int(idx)])
        item = dict(self.base_ds[base_idx])
        item["index"] = base_idx
        item["y"] = float(self.yvals[int(idx)])
        return item


def collate_labeled(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out = collate_batch(batch)
    out["index"] = torch.tensor([int(b["index"]) for b in batch], dtype=torch.long)
    out["y"] = torch.tensor([float(b["y"]) for b in batch], dtype=torch.float32)
    return out


def _prepare_condition_inputs(batch: Dict[str, Any], device: torch.device) -> Dict[str, Optional[torch.Tensor]]:
    out: Dict[str, Optional[torch.Tensor]] = {}
    fc = batch.get("fc_cond", None)
    out["fc_cond"] = None if fc is None else fc.to(device=device, dtype=torch.float32, non_blocking=True)
    out["has_fc"] = batch.get("has_fc", None)
    if out["has_fc"] is not None:
        out["has_fc"] = out["has_fc"].to(device=device, dtype=torch.float32, non_blocking=True)

    mri = batch.get("mri_cond", None)
    out["mri_cond"] = None if mri is None else mri.to(device=device, dtype=torch.float32, non_blocking=True)
    out["has_mri"] = batch.get("has_mri", None)
    if out["has_mri"] is not None:
        out["has_mri"] = out["has_mri"].to(device=device, dtype=torch.float32, non_blocking=True)

    meta = batch.get("meta_cond", None)
    out["meta_cond"] = None if meta is None else meta.to(device=device, dtype=torch.float32, non_blocking=True)
    out["has_meta"] = batch.get("has_meta", None)
    if out["has_meta"] is not None:
        out["has_meta"] = out["has_meta"].to(device=device, dtype=torch.float32, non_blocking=True)
    return out


def _build_model_from_dataset(cfg: Dict[str, Any], ds) -> ConditionalLatentDiT:
    mcfg = cfg.get("model", {})
    condition_shapes = {
        "fc": ds.fc_shape,
        "mri": ds.mri_shape,
        "metadata": ((ds.meta_dim,) if ds.meta_dim > 0 else None),
    }
    return ConditionalLatentDiT(
        target_shape=ds.target_shape,
        patch_size=mcfg.get("patch_size", [1, 2, 1]),
        hidden_dim=int(mcfg.get("hidden_dim", 512)),
        depth=int(mcfg.get("depth", 12)),
        num_heads=int(mcfg.get("num_heads", 8)),
        mlp_ratio=float(mcfg.get("mlp_ratio", 4.0)),
        dropout=float(mcfg.get("dropout", 0.0)),
        condition_shapes=condition_shapes,
        condition_cfg=cfg.get("data", {}).get("conditions", {}),
        diversity_cfg=cfg.get("diversity", {}),
        max_time_steps=int(cfg.get("diffusion", {}).get("num_steps", 1000)),
    )


def _weights_for_log(w: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if w is None:
        return None
    if w.ndim == 2:
        return w.mean(dim=0)
    if w.ndim == 1:
        return w
    if w.ndim == 3:
        return w.mean(dim=(0, 1))
    return None


def _format_weights_for_tqdm(w: Optional[torch.Tensor], max_len: int = 6) -> str:
    ww = _weights_for_log(w)
    if ww is None:
        return "[]"
    arr = ww.detach().cpu().float().tolist()
    s = ",".join(f"{x:.2f}" for x in arr[:max_len])
    if len(arr) > max_len:
        s += ",..."
    return f"[{s}]"


def _count_params(module: nn.Module) -> Tuple[int, int]:
    total = int(sum(p.numel() for p in module.parameters()))
    trainable = int(sum(p.numel() for p in module.parameters() if p.requires_grad))
    return total, trainable


def _fmt_million(n: int) -> str:
    return f"{(float(n) / 1e6):.2f}M"


def _unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


def _state_dict_cpu(module: nn.Module) -> Dict[str, torch.Tensor]:
    core = _unwrap_module(module)
    return {k: v.detach().cpu() for k, v in core.state_dict().items()}


def _is_condition_encoder_key(k: str) -> bool:
    key = str(k)
    return (
        key.startswith("fc_encoder.")
        or key.startswith("mri_encoder.")
        or key.startswith("video_encoder.")
        or key.startswith("audio_encoder.")
        or key.startswith("meta_encoder.")
    )


def _load_state_dict(
    module: nn.Module,
    state: Dict[str, torch.Tensor],
    *,
    allow_condition_mismatch: bool = False,
) -> None:
    core = _unwrap_module(module)
    if not allow_condition_mismatch:
        core.load_state_dict(state, strict=True)
        return

    model_state = core.state_dict()
    model_keys = set(model_state.keys())
    ckpt_keys = set(state.keys())

    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)

    bad_missing = [k for k in missing if not _is_condition_encoder_key(k)]
    bad_unexpected = [k for k in unexpected if not _is_condition_encoder_key(k)]
    if len(bad_missing) > 0 or len(bad_unexpected) > 0:
        miss_preview = bad_missing[:12]
        unexp_preview = bad_unexpected[:12]
        raise RuntimeError(
            "Checkpoint/model mismatch outside condition encoders.\n"
            f"missing(non-cond)={len(bad_missing)} preview={miss_preview}\n"
            f"unexpected(non-cond)={len(bad_unexpected)} preview={unexp_preview}"
        )

    filtered = {k: v for k, v in state.items() if k in model_keys}
    incompatible = core.load_state_dict(filtered, strict=False)
    miss2 = [k for k in incompatible.missing_keys if not _is_condition_encoder_key(k)]
    unexp2 = [k for k in incompatible.unexpected_keys if not _is_condition_encoder_key(k)]
    if len(miss2) > 0 or len(unexp2) > 0:
        miss_preview = miss2[:12]
        unexp_preview = unexp2[:12]
        raise RuntimeError(
            "State_dict load mismatch outside condition encoders after filtering.\n"
            f"missing(non-cond)={len(miss2)} preview={miss_preview}\n"
            f"unexpected(non-cond)={len(unexp2)} preview={unexp_preview}"
        )

    if len(missing) > 0 or len(unexpected) > 0:
        print(
            "[audit] checkpoint condition-key mismatch ignored: "
            f"missing_cond={len([k for k in missing if _is_condition_encoder_key(k)])} "
            f"unexpected_cond={len([k for k in unexpected if _is_condition_encoder_key(k)])}"
        )


def _resolve_gpu_ids(cfg: Dict[str, Any]) -> List[int]:
    cand: List[Any] = []
    if isinstance(cfg.get("train", None), dict):
        cand.append(cfg.get("train", {}).get("gpu_ids", None))
    if isinstance(cfg.get("training", None), dict):
        cand.append(cfg.get("training", {}).get("gpu_ids", None))
    cand.append(cfg.get("gpu_ids", None))

    picked = None
    for v in cand:
        if v is not None:
            picked = v
            break
    if picked is None:
        return []

    if not isinstance(picked, list):
        raise ValueError("gpu_ids must be a list of CUDA indices")

    out: List[int] = []
    for x in picked:
        i = int(x)
        if i < 0:
            raise ValueError(f"gpu_ids must be >= 0, got {i}")
        out.append(i)
    # dedupe while preserving order
    seen = set()
    uniq: List[int] = []
    for i in out:
        if i in seen:
            continue
        seen.add(i)
        uniq.append(i)
    return uniq


def _extract_cache_feature_batch(
    *,
    batch: Dict[str, Any],
    protocol: FeatureProtocolCond,
    t_list: List[int],
    device: torch.device,
    noise_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    x0 = batch["target_latent"].to(device=device, dtype=torch.float32)
    y = batch["y"].to(device=device, dtype=torch.float32)
    direction_id = batch["direction_id"].to(device=device, dtype=torch.long)
    cond_inputs = _prepare_condition_inputs(batch, device=device)
    subjects = [str(s) for s in batch["subject_id"]]
    indices = [int(x) for x in batch["index"]]

    tokens_list: List[torch.Tensor] = []
    for t in t_list:
        out = protocol.tokens_from_batch(
            x0,
            direction_id=direction_id,
            cond_inputs=cond_inputs,
            subjects=subjects,
            sample_indices=(indices if noise_mode == "per_sample_index" else None),
            enable_grad=False,
            timestep=int(t),
        )
        tokens_list.extend(out.tokens_list)

    pooled = [tok.mean(dim=1) for tok in tokens_list]
    feat = torch.cat(pooled, dim=1)  # (B, L*D)

    rows: List[Dict[str, Any]] = []
    for i in range(int(y.shape[0])):
        rows.append(
            {
                "subject_id": str(batch["subject_id"][i]),
                "sequence_id": str(batch["sequence_id"][i]),
                "direction": str(batch["direction"][i]),
                "anchor_chunk_id": int(batch["anchor_chunk_id"][i].item()),
                "target_chunk_id": int(batch["target_chunk_id"][i].item()),
            }
        )
    return feat, y, rows


def _dataset_subset_signature(ds: Dataset) -> Dict[str, Any]:
    keep_indices = getattr(ds, "keep_indices", None)
    if keep_indices is None:
        arr = np.arange(int(len(ds)), dtype=np.int64)
        mode = "full"
    else:
        arr = np.asarray(list(keep_indices), dtype=np.int64)
        mode = "subset"
    h = hashlib.sha1(arr.tobytes()).hexdigest()
    return {
        "mode": mode,
        "count": int(arr.shape[0]),
        "indices_sha1": str(h),
    }


def _cache_paths(cache_root: str, task_name: str, split_name: str, cache_scope: str) -> Tuple[str, str]:
    scope = str(cache_scope).strip().lower()
    if scope == "task":
        base = os.path.join(cache_root, _safe_name(task_name), split_name)
    elif scope == "shared":
        base = os.path.join(cache_root, "__shared__", split_name)
    else:
        raise ValueError(f"linear_probe_cache.scope must be shared|task, got {cache_scope}")
    return base + ".npz", base + ".meta.json"


def _cache_signature(
    *,
    cfg: Dict[str, Any],
    ckpt_path: str,
    split_name: str,
    ds: Dataset,
) -> Dict[str, Any]:
    emb = cfg.get("embedding", {}) if isinstance(cfg.get("embedding", {}), dict) else {}
    data = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    target = data.get("target", {}) if isinstance(data.get("target", {}), dict) else {}
    cond = data.get("conditions", {}) if isinstance(data.get("conditions", {}), dict) else {}
    return {
        "checkpoint": os.path.abspath(str(ckpt_path)),
        "split": str(split_name),
        "num_samples": int(len(ds)),
        "subset": _dataset_subset_signature(ds),
        "t_list": [int(x) for x in _parse_t_list(emb)],
        "capture_layers": [int(x) for x in list(emb.get("capture_layers", [-1]))],
        "noise_mode": str(emb.get("noise_mode", "per_subject")),
        "noise_seed": int(emb.get("noise_seed", 0)),
        "target_latent_field": str(target.get("latent_field", "mu")),
        "target_dtype": str(target.get("dataset_dtype", "float32")),
        "fc_enabled": bool(cond.get("fc", {}).get("enabled", False)) if isinstance(cond.get("fc", {}), dict) else False,
        "mri_enabled": bool(cond.get("mri", {}).get("enabled", False)) if isinstance(cond.get("mri", {}), dict) else False,
        "meta_enabled": bool(cond.get("metadata", {}).get("enabled", False)) if isinstance(cond.get("metadata", {}), dict) else False,
    }


def _collect_or_load_cached_split(
    *,
    task_name: str,
    split_name: str,
    ds: Dataset,
    protocol: FeatureProtocolCond,
    cfg: Dict[str, Any],
    ckpt_path: str,
    device: torch.device,
    cache_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    cache_root = os.path.abspath(str(cache_cfg.get("root", "outputs/cond_dit_feature_cache")))
    ensure_dir(cache_root)
    cache_scope = str(cache_cfg.get("scope", "shared")).strip().lower()
    npz_path, meta_path = _cache_paths(cache_root, task_name, split_name, cache_scope)

    sig = _cache_signature(cfg=cfg, ckpt_path=ckpt_path, split_name=split_name, ds=ds)
    reuse = bool(cache_cfg.get("reuse_if_exists", True))

    if reuse and os.path.isfile(npz_path) and os.path.isfile(meta_path):
        try:
            old_meta = load_json(meta_path)
            old_sig = old_meta.get("signature", {}) if isinstance(old_meta, dict) else {}
            if old_sig == sig:
                with np.load(npz_path, allow_pickle=True) as d:
                    X = np.asarray(d["X"], dtype=np.float32)
                    rows = []
                    n = int(X.shape[0])
                    sub = d["subject_id"].tolist()
                    seq = d["sequence_id"].tolist()
                    dire = d["direction"].tolist()
                    anc = d["anchor_chunk_id"].astype(np.int32).tolist()
                    tgt = d["target_chunk_id"].astype(np.int32).tolist()
                    for i in range(n):
                        rows.append(
                            {
                                "subject_id": str(sub[i]),
                                "sequence_id": str(seq[i]),
                                "direction": str(dire[i]),
                                "anchor_chunk_id": int(anc[i]),
                                "target_chunk_id": int(tgt[i]),
                            }
                        )

                y = np.asarray(getattr(ds, "yvals", []), dtype=np.float32)
                if int(y.shape[0]) != int(X.shape[0]):
                    raise RuntimeError(
                        f"cache label/sample mismatch for split={split_name}: "
                        f"cache_n={int(X.shape[0])} ds_n={int(y.shape[0])}"
                    )

                audit = {
                    "split": split_name,
                    "cache_hit": True,
                    "cache_scope": cache_scope,
                    "cache_npz": npz_path,
                    "num_samples": int(X.shape[0]),
                    "feature_dim": int(X.shape[1]) if X.ndim == 2 else -1,
                }
                return X, y, rows, audit
        except Exception:
            pass

    batch_size = int(cache_cfg.get("extract_batch_size", train_cfg.get("batch_size", 32)))
    num_workers = int(cache_cfg.get("extract_num_workers", train_cfg.get("num_workers", 0)))
    noise_mode = str(cfg.get("embedding", {}).get("noise_mode", "per_subject"))
    t_list = _parse_t_list(cfg.get("embedding", {}))

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_labeled,
    )

    feat_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        it = tqdm(dl, desc=f"cache[{task_name}:{split_name}]", ncols=120)
        for batch in it:
            feat, y_batch, r = _extract_cache_feature_batch(
                batch=batch,
                protocol=protocol,
                t_list=t_list,
                device=device,
                noise_mode=noise_mode,
            )
            feat_list.append(feat.detach().cpu().numpy().astype(np.float32))
            y_list.append(y_batch.detach().cpu().numpy().astype(np.float32))
            rows.extend(r)

    X = np.concatenate(feat_list, axis=0)
    y = np.asarray(getattr(ds, "yvals", []), dtype=np.float32)
    if int(y.shape[0]) != int(X.shape[0]):
        if len(y_list) > 0:
            y = np.concatenate(y_list, axis=0)
        if int(y.shape[0]) != int(X.shape[0]):
            raise RuntimeError(
                f"cache label/sample mismatch after extraction for split={split_name}: "
                f"X_n={int(X.shape[0])} y_n={int(y.shape[0])}"
            )

    save_dtype = str(cache_cfg.get("dtype", "float16")).strip().lower()
    if save_dtype == "float32":
        X_save = X.astype(np.float32)
    else:
        X_save = X.astype(np.float16)

    subj = np.array([r["subject_id"] for r in rows], dtype=object)
    seq = np.array([r["sequence_id"] for r in rows], dtype=object)
    dire = np.array([r["direction"] for r in rows], dtype=object)
    anc = np.array([int(r["anchor_chunk_id"]) for r in rows], dtype=np.int32)
    tgt = np.array([int(r["target_chunk_id"]) for r in rows], dtype=np.int32)

    ensure_dir(os.path.dirname(npz_path))
    np.savez(npz_path, X=X_save, y=y.astype(np.float32), subject_id=subj, sequence_id=seq, direction=dire, anchor_chunk_id=anc, target_chunk_id=tgt)
    save_json({"signature": sig, "saved_dtype": str(X_save.dtype), "num_samples": int(X.shape[0]), "feature_dim": int(X.shape[1])}, meta_path)

    audit = {
        "split": split_name,
        "cache_hit": False,
        "cache_scope": cache_scope,
        "cache_npz": npz_path,
        "num_samples": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
    }
    return X.astype(np.float32), y.astype(np.float32), rows, audit


def _build_task_cfg(base_cfg: Dict[str, Any], task_raw: Dict[str, Any], idx: int) -> Tuple[str, Dict[str, Any]]:
    cfg = copy.deepcopy(base_cfg)
    cfg.pop("tasks", None)
    cfg.pop("task_list", None)

    name = _safe_name(task_raw.get("name", f"task_{idx:02d}"))

    for key in ("label", "task_head", "head", "train", "training", "optim", "output", "data", "embedding", "aggregator"):
        if isinstance(task_raw.get(key, None), dict):
            if key not in cfg or not isinstance(cfg.get(key, None), dict):
                cfg[key] = {}
            _deep_update(cfg[key], task_raw[key])

    if "task" in task_raw and isinstance(task_raw["task"], dict):
        if "task_head" not in cfg or not isinstance(cfg.get("task_head", None), dict):
            cfg["task_head"] = {}
        _deep_update(cfg["task_head"], task_raw["task"])

    if "mode" in task_raw:
        cfg["mode"] = task_raw["mode"]

    if "output_dir" in task_raw:
        cfg["_task_output_dir"] = str(task_raw["output_dir"])
    elif "out_subdir" in task_raw:
        cfg["_task_output_dir"] = str(task_raw["out_subdir"])

    return name, cfg


def _task_from_cfg(cfg: Dict[str, Any], y_train: np.ndarray) -> Tuple[str, int]:
    tcfg = cfg.get("task_head", cfg.get("task", {}))
    task_type = str(tcfg.get("task_type", tcfg.get("type", "classification"))).strip().lower()
    if task_type not in ("classification", "regression"):
        raise ValueError("task.task_type/type must be classification or regression")
    if task_type == "regression":
        return task_type, 1

    if "num_classes" in tcfg:
        nc = int(tcfg.get("num_classes", 2))
    else:
        nc = int(np.max(y_train).item()) + 1
    if nc <= 1:
        raise ValueError("num_classes must be >= 2 for classification")
    return task_type, nc


def _primary_metric(task_type: str, metric_name: str, metrics: Dict[str, float]) -> float:
    name = str(metric_name)
    if name in metrics:
        return float(metrics[name])
    if task_type == "regression":
        return float(metrics.get("pearson", 0.0))
    return float(metrics.get("f1_weighted", 0.0))


def _build_pred_rows(
    *,
    batch: Dict[str, Any],
    pred: torch.Tensor,
    y: torch.Tensor,
    task_type: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    bsz = int(y.shape[0])
    if task_type == "classification":
        probs = torch.softmax(pred, dim=1).detach().cpu()
        yv = y.long().view(-1).detach().cpu()
        pv = torch.argmax(pred, dim=1).detach().cpu()
        for i in range(bsz):
            row = {
                "subject_id": str(batch["subject_id"][i]),
                "sequence_id": str(batch["sequence_id"][i]),
                "direction": str(batch["direction"][i]),
                "anchor_chunk_id": int(batch["anchor_chunk_id"][i].item()),
                "target_chunk_id": int(batch["target_chunk_id"][i].item()),
                "target": int(yv[i].item()),
                "pred": int(pv[i].item()),
            }
            if int(probs.shape[1]) > 1:
                row["prob_pos"] = float(probs[i, 1].item())
            rows.append(row)
    else:
        yv = y.view(-1).detach().cpu()
        pv = pred.view(-1).detach().cpu()
        for i in range(bsz):
            rows.append(
                {
                    "subject_id": str(batch["subject_id"][i]),
                    "sequence_id": str(batch["sequence_id"][i]),
                    "direction": str(batch["direction"][i]),
                    "anchor_chunk_id": int(batch["anchor_chunk_id"][i].item()),
                    "target_chunk_id": int(batch["target_chunk_id"][i].item()),
                    "target": float(yv[i].item()),
                    "pred": float(pv[i].item()),
                }
            )
    return rows


def _write_rows_csv(rows: List[Dict[str, Any]], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    if len(rows) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["subject_id", "target", "pred"])
            w.writeheader()
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _forward_batch(
    *,
    batch: Dict[str, Any],
    protocol: FeatureProtocolCond,
    t_list: List[int],
    agg_type: str,
    pool_mode: str,
    model_trainable: bool,
    aggregator: nn.Module,
    head: nn.Module,
    device: torch.device,
    noise_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    x0 = batch["target_latent"].to(device=device, dtype=torch.float32)
    y = batch["y"].to(device=device, dtype=torch.float32)
    direction_id = batch["direction_id"].to(device=device, dtype=torch.long)
    cond_inputs = _prepare_condition_inputs(batch, device=device)
    subjects = [str(s) for s in batch["subject_id"]]
    indices = [int(x) for x in batch["index"]]

    tokens_list: List[torch.Tensor] = []
    for t in t_list:
        out = protocol.tokens_from_batch(
            x0,
            direction_id=direction_id,
            cond_inputs=cond_inputs,
            subjects=subjects,
            sample_indices=(indices if noise_mode == "per_sample_index" else None),
            enable_grad=bool(model_trainable),
            timestep=int(t),
        )
        tokens_list.extend(out.tokens_list)

    if agg_type == "token_attn":
        E = torch.cat(tokens_list, dim=1)
    else:
        if pool_mode != "mean":
            raise ValueError(f"Unsupported embedding.pool={pool_mode}; currently only 'mean' is implemented")
        pooled = [tok.mean(dim=1) for tok in tokens_list]
        E = torch.stack(pooled, dim=1)

    emb, weights = aggregator(E)
    pred = head(emb)
    return y, pred, weights


def _eval_loader(
    *,
    loader: DataLoader,
    task_type: str,
    num_classes: int,
    loss_fn: nn.Module,
    protocol: FeatureProtocolCond,
    t_list: List[int],
    agg_type: str,
    pool_mode: str,
    model_trainable: bool,
    aggregator: nn.Module,
    head: nn.Module,
    device: torch.device,
    noise_mode: str,
    dist_ctx: DistContext,
    desc: Optional[str] = None,
    return_rows: bool = True,
) -> Tuple[float, Dict[str, float], List[Dict[str, Any]]]:
    if model_trainable:
        protocol.model.eval()
    aggregator.eval()
    head.eval()

    result_obj: Optional[Tuple[float, Dict[str, float], List[Dict[str, Any]]]] = None

    if (not dist_ctx.enabled) or dist_ctx.is_main:
        all_y: List[torch.Tensor] = []
        all_out: List[torch.Tensor] = []
        rows: List[Dict[str, Any]] = []
        loss_sum = 0.0
        n_obs = 0

        it = loader
        if desc is not None:
            it = tqdm(loader, desc=desc, ncols=120, leave=False)

        with torch.no_grad():
            for batch in it:
                y, out, _w = _forward_batch(
                    batch=batch,
                    protocol=protocol,
                    t_list=t_list,
                    agg_type=agg_type,
                    pool_mode=pool_mode,
                    model_trainable=False,
                    aggregator=aggregator,
                    head=head,
                    device=device,
                    noise_mode=noise_mode,
                )
                if task_type == "classification":
                    loss = loss_fn(out, y.long().view(-1))
                else:
                    loss = loss_fn(out.view(-1), y.view(-1))

                bsz = int(y.shape[0])
                loss_sum += float(loss.item()) * max(1, bsz)
                n_obs += bsz
                all_y.append(y.detach().cpu())
                all_out.append(out.detach().cpu())
                if return_rows:
                    rows.extend(_build_pred_rows(batch=batch, pred=out, y=y, task_type=task_type))

        y_cat = torch.cat(all_y, dim=0)
        out_cat = torch.cat(all_out, dim=0)
        if task_type == "classification":
            metrics = evaluate_classification(y_cat.long().view(-1), out_cat, num_classes=int(num_classes))
        else:
            metrics = evaluate_regression(y_cat.view(-1), out_cat.view(-1))

        result_obj = (loss_sum / max(1, n_obs), metrics, rows)

    if dist_ctx.enabled:
        if dist_ctx.is_main:
            if result_obj is None:
                raise RuntimeError("evaluation result is unavailable on rank0")
            to_send = (float(result_obj[0]), dict(result_obj[1]), [])
        else:
            to_send = None
        payload: List[Optional[Tuple[float, Dict[str, float], List[Dict[str, Any]]]]] = [to_send]
        dist.broadcast_object_list(payload, src=0)
        if not dist_ctx.is_main:
            result_obj = payload[0]

    if result_obj is None:
        raise RuntimeError("evaluation result is unavailable")
    if not return_rows:
        return float(result_obj[0]), dict(result_obj[1]), []
    return float(result_obj[0]), dict(result_obj[1]), list(result_obj[2])


def _reload_model_from_ckpt(model: nn.Module, base_state: Dict[str, torch.Tensor]) -> None:
    _load_state_dict(model, base_state)



def _build_cache_pred_rows(
    *,
    rows_meta: List[Dict[str, Any]],
    batch_indices: torch.Tensor,
    pred: torch.Tensor,
    y: torch.Tensor,
    task_type: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    idxv = batch_indices.long().view(-1).detach().cpu().tolist()

    if task_type == "classification":
        probs = torch.softmax(pred, dim=1).detach().cpu()
        yv = y.long().view(-1).detach().cpu()
        pv = torch.argmax(pred, dim=1).detach().cpu()
        for i, ridx in enumerate(idxv):
            base = dict(rows_meta[int(ridx)])
            base["target"] = int(yv[i].item())
            base["pred"] = int(pv[i].item())
            if int(probs.shape[1]) > 1:
                base["prob_pos"] = float(probs[i, 1].item())
            rows.append(base)
        return rows

    yv = y.view(-1).detach().cpu()
    pv = pred.view(-1).detach().cpu()
    for i, ridx in enumerate(idxv):
        base = dict(rows_meta[int(ridx)])
        base["target"] = float(yv[i].item())
        base["pred"] = float(pv[i].item())
        rows.append(base)
    return rows


def _eval_cache_head(
    *,
    X: torch.Tensor,
    y: torch.Tensor,
    rows_meta: List[Dict[str, Any]],
    task_type: str,
    num_classes: int,
    loss_fn: nn.Module,
    head: nn.Module,
    device: torch.device,
    batch_size: int,
    desc: Optional[str] = None,
) -> Tuple[float, Dict[str, float], List[Dict[str, Any]]]:
    head.eval()
    idx_all = torch.arange(int(X.shape[0]), dtype=torch.long)
    ds = TensorDataset(X, y, idx_all)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    loss_sum = 0.0
    n_obs = 0
    all_y: List[torch.Tensor] = []
    all_out: List[torch.Tensor] = []
    pred_rows: List[Dict[str, Any]] = []

    it = dl
    if desc is not None:
        it = tqdm(dl, desc=desc, ncols=120, leave=False)

    with torch.no_grad():
        for xb, yb, ib in it:
            xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
            yb = yb.to(device=device, dtype=torch.float32, non_blocking=True)
            out = head(xb)
            if task_type == "classification":
                loss = loss_fn(out, yb.long().view(-1))
            else:
                loss = loss_fn(out.view(-1), yb.view(-1))

            bsz = int(yb.shape[0])
            loss_sum += float(loss.item()) * max(1, bsz)
            n_obs += bsz
            all_y.append(yb.detach().cpu())
            all_out.append(out.detach().cpu())
            pred_rows.extend(_build_cache_pred_rows(rows_meta=rows_meta, batch_indices=ib, pred=out, y=yb, task_type=task_type))

    y_cat = torch.cat(all_y, dim=0)
    out_cat = torch.cat(all_out, dim=0)
    if task_type == "classification":
        metrics = evaluate_classification(y_cat.long().view(-1), out_cat, num_classes=int(num_classes))
    else:
        metrics = evaluate_regression(y_cat.view(-1), out_cat.view(-1))
    return loss_sum / max(1, n_obs), metrics, pred_rows


def _run_cached_linear_probe(
    *,
    task_name: str,
    cfg: Dict[str, Any],
    model: nn.Module,
    base_model_state: Dict[str, torch.Tensor],
    protocol: FeatureProtocolCond,
    train_ds: LabeledPairDataset,
    val_ds: LabeledPairDataset,
    test_ds: LabeledPairDataset,
    task_type: str,
    num_classes: int,
    target_col: str,
    label_audit: Dict[str, Any],
    target_norm: Optional[Tuple[float, float]],
    output_dir: str,
    device: torch.device,
) -> Dict[str, Any]:
    train_cfg = cfg.get("train", cfg.get("training", {}))
    head_cfg = cfg.get("head", {})
    emb_cfg = cfg.get("embedding", {})
    opt_cfg = cfg.get("optim", {})
    cache_cfg_raw = cfg.get("linear_probe_cache", {}) if isinstance(cfg.get("linear_probe_cache", {}), dict) else {}
    cache_cfg = dict(cache_cfg_raw)

    _reload_model_from_ckpt(model, base_model_state)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    ckpt_info = cfg.get("ckpt", cfg.get("checkpoints", {}))
    ckpt_path_for_sig = os.path.abspath(
        str(ckpt_info.get("checkpoint", ckpt_info.get("cond_dit_ckpt", ckpt_info.get("stage1_ckpt", ""))))
    )
    if str(cache_cfg.get("root", "")).strip() == "":
        cache_cfg["root"] = _default_cache_root(cfg=cfg, ckpt_path=ckpt_path_for_sig)

    X_tr_np, y_tr_np, rows_tr_meta, cache_train_audit = _collect_or_load_cached_split(
        task_name=task_name,
        split_name="train",
        ds=train_ds,
        protocol=protocol,
        cfg=cfg,
        ckpt_path=ckpt_path_for_sig,
        device=device,
        cache_cfg=cache_cfg,
        train_cfg=train_cfg,
    )
    X_va_np, y_va_np, rows_va_meta, cache_val_audit = _collect_or_load_cached_split(
        task_name=task_name,
        split_name="val",
        ds=val_ds,
        protocol=protocol,
        cfg=cfg,
        ckpt_path=ckpt_path_for_sig,
        device=device,
        cache_cfg=cache_cfg,
        train_cfg=train_cfg,
    )
    X_te_np, y_te_np, rows_te_meta, cache_test_audit = _collect_or_load_cached_split(
        task_name=task_name,
        split_name="test",
        ds=test_ds,
        protocol=protocol,
        cfg=cfg,
        ckpt_path=ckpt_path_for_sig,
        device=device,
        cache_cfg=cache_cfg,
        train_cfg=train_cfg,
    )

    standardize = bool(cache_cfg.get("standardize", True))
    feat_mu = None
    feat_sd = None
    if standardize:
        feat_mu = np.mean(X_tr_np, axis=0, keepdims=True).astype(np.float32)
        feat_sd = np.std(X_tr_np, axis=0, keepdims=True).astype(np.float32)
        feat_sd[feat_sd < 1.0e-6] = 1.0
        X_tr_np = (X_tr_np - feat_mu) / feat_sd
        X_va_np = (X_va_np - feat_mu) / feat_sd
        X_te_np = (X_te_np - feat_mu) / feat_sd

    X_tr = torch.from_numpy(X_tr_np.astype(np.float32))
    X_va = torch.from_numpy(X_va_np.astype(np.float32))
    X_te = torch.from_numpy(X_te_np.astype(np.float32))
    y_tr = torch.from_numpy(y_tr_np.astype(np.float32))
    y_va = torch.from_numpy(y_va_np.astype(np.float32))
    y_te = torch.from_numpy(y_te_np.astype(np.float32))

    feature_dim = int(X_tr.shape[1])
    hidden_dim = int(head_cfg.get("mlp_hidden", 0))
    if hidden_dim <= 0:
        hidden_dim = None
    head = build_head(
        d_model=feature_dim,
        task_type=task_type,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=float(head_cfg.get("mlp_dropout", 0.1)),
    ).to(device)

    noise_mode = str(emb_cfg.get("noise_mode", "per_subject"))
    pool_mode = str(emb_cfg.get("pool", "mean"))

    model_total, model_train = _count_params(model)
    head_total, head_train = _count_params(head)
    print("-" * 100)
    print(
        f"[audit][{task_name}] mode=linear_probe cache_enabled=True model_trainable=False "
        f"task_type={task_type} target_col={target_col}"
    )
    print(
        f"[audit][{task_name}] cache feature_dim={feature_dim} standardize={standardize} "
        f"cache_hit(train/val/test)={cache_train_audit['cache_hit']}/{cache_val_audit['cache_hit']}/{cache_test_audit['cache_hit']}"
    )
    print(
        f"[audit][{task_name}] trainable_params total={head_train}({_fmt_million(head_train)}) "
        f"/ all_params={head_total + model_total}({_fmt_million(head_total + model_total)})"
    )
    print(
        f"[audit][{task_name}] model trainable/all={model_train}/{model_total} "
        f"head trainable/all={head_train}/{head_total}"
    )
    print("-" * 100)

    if task_type == "classification":
        use_class_weights = bool(train_cfg.get("use_class_weights", True))
        if use_class_weights:
            y_cls = y_tr.long().view(-1)
            counts = torch.bincount(y_cls, minlength=int(num_classes)).float()
            total = float(torch.sum(counts).item())
            denom = (counts * float(num_classes)).clamp_min(1.0)
            w = total / denom
            w[counts == 0] = 0.0
            loss_fn = nn.CrossEntropyLoss(weight=w.to(device))
        else:
            loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()

    lr_head = float(opt_cfg.get("lr_head", train_cfg.get("lr_head", 1.0e-4)))
    wd_head = float(opt_cfg.get("weight_decay_head", train_cfg.get("weight_decay_head", 0.0)))
    optimizer = torch.optim.AdamW(
        [{"params": head.parameters(), "lr": lr_head, "weight_decay": wd_head}],
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        eps=float(opt_cfg.get("eps", 1.0e-8)),
    )

    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    batch_size = int(train_cfg.get("batch_size", 64))
    epochs = int(train_cfg.get("epochs", 20))
    patience = int(train_cfg.get("patience", 10))
    best_metric_name = str(train_cfg.get("best_metric", "f1_weighted" if task_type == "classification" else "pearson"))
    eval_test_each_epoch = bool(train_cfg.get("eval_test_each_epoch", False))
    history_flush_each_epoch = bool(train_cfg.get("history_flush_each_epoch", True))
    print_epoch_metrics = bool(train_cfg.get("print_epoch_metrics", True))
    metric_name_norm = str(best_metric_name).strip().lower()
    higher_is_better = metric_name_norm not in {"loss", "val_loss", "mse", "mae"}
    best_val = -1.0e18 if higher_is_better else 1.0e18
    bad = 0
    history: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, Any]] = None

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    ensure_dir(output_dir)

    for ep in range(1, epochs + 1):
        head.train()
        loss_sum = 0.0
        n_obs = 0

        pbar = tqdm(train_loader, desc=f"{task_name} train_cache[{ep}/{epochs}]", ncols=120)
        for xb, yb in pbar:
            xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
            yb = yb.to(device=device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                out = head(xb)
                if task_type == "classification":
                    loss = loss_fn(out, yb.long().view(-1))
                else:
                    loss = loss_fn(out.view(-1), yb.view(-1))

            if use_amp:
                scaler.scale(loss).backward()
                if float(train_cfg.get("grad_clip_norm", 0.0)) > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=float(train_cfg.get("grad_clip_norm", 1.0)))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if float(train_cfg.get("grad_clip_norm", 0.0)) > 0.0:
                    torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=float(train_cfg.get("grad_clip_norm", 1.0)))
                optimizer.step()

            bsz = int(yb.shape[0])
            loss_sum += float(loss.item()) * max(1, bsz)
            n_obs += bsz
            pbar.set_postfix(loss=f"{loss_sum/max(1, n_obs):.5f}")

        train_loss = loss_sum / max(1, n_obs)
        tr_eval_loss, tr_eval_metrics, _ = _eval_cache_head(
            X=X_tr,
            y=y_tr,
            rows_meta=rows_tr_meta,
            task_type=task_type,
            num_classes=num_classes,
            loss_fn=loss_fn,
            head=head,
            device=device,
            batch_size=batch_size,
            desc=None,
        )
        val_loss, val_metrics, _ = _eval_cache_head(
            X=X_va,
            y=y_va,
            rows_meta=rows_va_meta,
            task_type=task_type,
            num_classes=num_classes,
            loss_fn=loss_fn,
            head=head,
            device=device,
            batch_size=batch_size,
            desc=None,
        )

        test_loss = None
        test_metrics = None
        if eval_test_each_epoch:
            test_loss, test_metrics, _ = _eval_cache_head(
                X=X_te,
                y=y_te,
                rows_meta=rows_te_meta,
                task_type=task_type,
                num_classes=num_classes,
                loss_fn=loss_fn,
                head=head,
                device=device,
                batch_size=batch_size,
                desc=None,
            )

        rec = {
            "epoch": ep,
            "train_loss": float(train_loss),
            "train_eval_loss": float(tr_eval_loss),
            "val_loss": float(val_loss),
        }
        rec.update({f"train_{k}": float(v) for k, v in tr_eval_metrics.items()})
        rec.update({f"val_{k}": float(v) for k, v in val_metrics.items()})
        if test_loss is not None and test_metrics is not None:
            rec["test_loss"] = float(test_loss)
            rec.update({f"test_{k}": float(v) for k, v in test_metrics.items()})
        history.append(rec)
        if history_flush_each_epoch:
            save_json({"history": history}, os.path.join(output_dir, "history.json"))
        if print_epoch_metrics:
            msg = (
                f"[epoch {ep}/{epochs}] train_loss={float(train_loss):.5f} "
                f"val_loss={float(val_loss):.5f}"
            )
            key_val_epoch = _primary_metric(task_type, best_metric_name, val_metrics)
            msg += f" val_{best_metric_name}={float(key_val_epoch):.5f}"
            if test_loss is not None and test_metrics is not None:
                test_key = _primary_metric(task_type, best_metric_name, test_metrics)
                msg += f" test_loss={float(test_loss):.5f} test_{best_metric_name}={float(test_key):.5f}"
            print(msg)

        key_val = _primary_metric(task_type, best_metric_name, val_metrics)
        is_better = (key_val > best_val) if higher_is_better else (key_val < best_val)
        if is_better:
            best_val = float(key_val)
            bad = 0
            best_state = {
                "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                "epoch": ep,
                "metric": float(best_val),
            }
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is None:
        raise RuntimeError(f"Task {task_name}: best_state is None")
    head.load_state_dict(best_state["head"], strict=True)

    tr_loss, tr_metrics, tr_rows = _eval_cache_head(
        X=X_tr,
        y=y_tr,
        rows_meta=rows_tr_meta,
        task_type=task_type,
        num_classes=num_classes,
        loss_fn=loss_fn,
        head=head,
        device=device,
        batch_size=batch_size,
        desc="eval_train",
    )
    va_loss, va_metrics, va_rows = _eval_cache_head(
        X=X_va,
        y=y_va,
        rows_meta=rows_va_meta,
        task_type=task_type,
        num_classes=num_classes,
        loss_fn=loss_fn,
        head=head,
        device=device,
        batch_size=batch_size,
        desc="eval_val",
    )
    te_loss, te_metrics, te_rows = _eval_cache_head(
        X=X_te,
        y=y_te,
        rows_meta=rows_te_meta,
        task_type=task_type,
        num_classes=num_classes,
        loss_fn=loss_fn,
        head=head,
        device=device,
        batch_size=batch_size,
        desc="eval_test",
    )

    ensure_dir(output_dir)
    _write_rows_csv(tr_rows, os.path.join(output_dir, "pred_train.csv"))
    _write_rows_csv(va_rows, os.path.join(output_dir, "pred_val.csv"))
    _write_rows_csv(te_rows, os.path.join(output_dir, "pred_test.csv"))

    capture_layers = resolve_capture_layers(int(getattr(_unwrap_module(model), "depth")), list(emb_cfg.get("capture_layers", [-1])))
    t_list = _parse_t_list(emb_cfg)
    noise_mode = str(emb_cfg.get("noise_mode", "per_subject"))
    pool_mode = str(emb_cfg.get("pool", "mean"))

    ckpt_path = os.path.join(output_dir, "downstream_best.pt")
    torch.save(
        {
            "task_name": task_name,
            "task_type": task_type,
            "target_col": target_col,
            "num_classes": int(num_classes),
            "mode": "linear_probe",
            "cache_mode": True,
            "best_state": best_state,
            "feature_dim": int(feature_dim),
            "config": cfg,
            "capture_layers": capture_layers,
            "t_list": t_list,
        },
        ckpt_path,
    )

    save_json({"history": history}, os.path.join(output_dir, "history.json"))

    cache_summary = {
        "enabled": True,
        "standardize": standardize,
        "train": cache_train_audit,
        "val": cache_val_audit,
        "test": cache_test_audit,
    }
    if standardize and feat_mu is not None and feat_sd is not None:
        cache_summary["zscore"] = {
            "mu_mean": float(np.mean(feat_mu)),
            "mu_std": float(np.std(feat_mu)),
            "sd_mean": float(np.mean(feat_sd)),
            "sd_std": float(np.std(feat_sd)),
        }

    summary = {
        "task_name": task_name,
        "mode": "linear_probe",
        "task_type": task_type,
        "target_col": target_col,
        "num_classes": int(num_classes),
        "label_audit": label_audit,
        "label_zscore": (
            None
            if target_norm is None
            else {"mean": float(target_norm[0]), "std": float(target_norm[1])}
        ),
        "split_audit": {
            "train": train_ds.audit,
            "val": val_ds.audit,
            "test": test_ds.audit,
        },
        "embedding": {
            "t_list": [int(x) for x in t_list],
            "capture_layers": [int(x) for x in capture_layers],
            "pool": pool_mode,
            "noise_mode": noise_mode,
            "noise_seed": int(emb_cfg.get("noise_seed", 0)),
        },
        "cache": cache_summary,
        "metrics": {
            "train_loss": float(tr_loss),
            "val_loss": float(va_loss),
            "test_loss": float(te_loss),
            "train": {k: float(v) for k, v in tr_metrics.items()},
            "val": {k: float(v) for k, v in va_metrics.items()},
            "test": {k: float(v) for k, v in te_metrics.items()},
            "best_val": float(best_val),
            "best_metric_name": best_metric_name,
        },
        "checkpoint": ckpt_path,
    }
    save_json(summary, os.path.join(output_dir, "summary.json"))
    return summary


def _run_one_task(
    *,
    task_name: str,
    cfg: Dict[str, Any],
    model: nn.Module,
    base_model_state: Dict[str, torch.Tensor],
    protocol: FeatureProtocolCond,
    train_ds_base: Dataset,
    val_ds_base: Dataset,
    test_ds_base: Dataset,
    output_dir: str,
    device: torch.device,
    dist_ctx: DistContext,
) -> Dict[str, Any]:
    label_cfg = cfg.get("label", {})
    head_task_cfg = cfg.get("task_head", cfg.get("task", {}))
    train_cfg = cfg.get("train", cfg.get("training", {}))
    head_cfg = cfg.get("head", {})
    agg_cfg = cfg.get("aggregator", {})
    emb_cfg = cfg.get("embedding", {})
    opt_cfg = cfg.get("optim", {})
    cache_cfg = cfg.get("linear_probe_cache", {}) if isinstance(cfg.get("linear_probe_cache", {}), dict) else {}

    label_csv = str(label_cfg.get("label_csv_path", label_cfg.get("label_csv", ""))).strip()
    if label_csv == "":
        raise ValueError(f"Task {task_name}: missing label_csv_path")
    subject_col = str(label_cfg.get("subject_col", "Subject"))
    target_col = str(label_cfg.get("target_col", head_task_cfg.get("target_col", ""))).strip()
    if target_col == "":
        raise ValueError(f"Task {task_name}: missing target_col")

    mode = str(cfg.get("mode", "linear_probe")).strip().lower()
    if mode not in ("linear_probe", "full_finetune"):
        raise ValueError("mode must be one of: linear_probe|full_finetune")
    use_cache = (mode == "linear_probe") and bool(cache_cfg.get("enabled", False))

    task_type_hint = str(head_task_cfg.get("task_type", head_task_cfg.get("type", "classification"))).strip().lower()
    if task_type_hint not in ("classification", "regression"):
        raise ValueError("task_type must be classification or regression")

    labels, label_audit = _read_label_map(
        path=label_csv,
        subject_col=subject_col,
        target_col=target_col,
        task_type=task_type_hint,
        label_map=(label_cfg.get("label_map", None) if isinstance(label_cfg.get("label_map", None), dict) else None),
        duplicate_policy_cls=str(label_cfg.get("duplicate_policy_cls", "majority")),
        duplicate_policy_reg=str(label_cfg.get("duplicate_policy_reg", "mean")),
    )

    target_norm = None
    if task_type_hint == "regression" and bool(label_cfg.get("zscore", False)):
        vals = []
        for idx in range(len(train_ds_base)):
            sid = _subject_id_from_base_ds(train_ds_base, idx)
            if sid in labels:
                vals.append(float(labels[sid]))
        if len(vals) == 0:
            raise RuntimeError(f"Task {task_name}: no train labels available for zscore")
        mu = float(np.mean(vals))
        sd = float(np.std(vals))
        if sd < 1.0e-8:
            sd = 1.0
        target_norm = (mu, sd)

    train_ds = LabeledPairDataset(base_ds=train_ds_base, split_name="train", target_map=labels, task_type=task_type_hint, target_norm=target_norm)
    val_ds = LabeledPairDataset(base_ds=val_ds_base, split_name="val", target_map=labels, task_type=task_type_hint, target_norm=target_norm)
    test_ds = LabeledPairDataset(base_ds=test_ds_base, split_name="test", target_map=labels, task_type=task_type_hint, target_norm=target_norm)

    y_train_np = train_ds.yvals
    task_type, num_classes = _task_from_cfg(cfg, y_train_np)

    if use_cache:
        if dist_ctx.enabled:
            packet: Optional[Dict[str, Any]] = None
            if dist_ctx.is_main:
                try:
                    summary_obj = _run_cached_linear_probe(
                        task_name=task_name,
                        cfg=cfg,
                        model=model,
                        base_model_state=base_model_state,
                        protocol=protocol,
                        train_ds=train_ds,
                        val_ds=val_ds,
                        test_ds=test_ds,
                        task_type=task_type,
                        num_classes=num_classes,
                        target_col=target_col,
                        label_audit=label_audit,
                        target_norm=target_norm,
                        output_dir=output_dir,
                        device=device,
                    )
                    packet = {"ok": True, "summary": summary_obj}
                except Exception as e:
                    packet = {"ok": False, "error": str(e)}

            payload: List[Optional[Dict[str, Any]]] = [packet]
            dist.broadcast_object_list(payload, src=0)
            out = payload[0]
            if out is None:
                raise RuntimeError("linear_probe summary broadcast failed")
            if not bool(out.get("ok", False)):
                raise RuntimeError(str(out.get("error", "linear_probe rank0 failed")))
            return dict(out["summary"])

        return _run_cached_linear_probe(
            task_name=task_name,
            cfg=cfg,
            model=model,
            base_model_state=base_model_state,
            protocol=protocol,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            task_type=task_type,
            num_classes=num_classes,
            target_col=target_col,
            label_audit=label_audit,
            target_norm=target_norm,
            output_dir=output_dir,
            device=device,
        )

    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 0))

    train_sampler = _distributed_sampler(train_ds, dist_ctx=dist_ctx, shuffle=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_labeled,
    )
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_labeled,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_labeled,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_labeled,
    )

    if dist_ctx.is_main:
        ensure_dir(output_dir)

    _reload_model_from_ckpt(model, base_model_state)

    model_trainable = mode == "full_finetune"
    for p in model.parameters():
        p.requires_grad = bool(model_trainable)
    model.train(model_trainable)

    agg_type = str(agg_cfg.get("type", "lws_scalar"))
    t_list = _parse_t_list(emb_cfg)
    core_model = _unwrap_module(model)
    capture_layers = resolve_capture_layers(int(getattr(core_model, "depth")), list(emb_cfg.get("capture_layers", [-1])))
    num_agg_layers = int(len(t_list) * len(capture_layers))
    d_model = int(getattr(core_model, "hidden_dim"))

    aggregator = build_layer_aggregator(agg_cfg, d_model=d_model, num_layers=num_agg_layers).to(device)
    hidden_dim = int(head_cfg.get("mlp_hidden", 0))
    if hidden_dim <= 0:
        hidden_dim = None
    head = build_head(
        d_model=d_model,
        task_type=task_type,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=float(head_cfg.get("mlp_dropout", 0.1)),
    ).to(device)

    if dist_ctx.enabled:
        ddp_cfg = cfg.get("ddp", {}) if isinstance(cfg.get("ddp", {}), dict) else {}
        ddp_find_unused_aux = bool(ddp_cfg.get("find_unused_parameters_aux", False))
        aggregator = DDP(
            aggregator,
            device_ids=[int(dist_ctx.local_rank)],
            output_device=int(dist_ctx.local_rank),
            find_unused_parameters=ddp_find_unused_aux,
        )
        head = DDP(
            head,
            device_ids=[int(dist_ctx.local_rank)],
            output_device=int(dist_ctx.local_rank),
            find_unused_parameters=False,
        )

    noise_mode = str(emb_cfg.get("noise_mode", "per_subject"))
    pool_mode = str(emb_cfg.get("pool", "mean"))

    model_total, model_train = _count_params(model)
    agg_total, agg_train = _count_params(aggregator)
    head_total, head_train = _count_params(head)
    total_trainable = int(model_train + agg_train + head_train)
    total_params = int(model_total + agg_total + head_total)
    if dist_ctx.is_main:
        print("-" * 100)
        print(
            f"[audit][{task_name}] mode={mode} model_trainable={model_trainable} "
            f"task_type={task_type} target_col={target_col}"
        )
        print(
            f"[audit][{task_name}] trainable_params total={total_trainable}({_fmt_million(total_trainable)}) "
            f"/ all_params={total_params}({_fmt_million(total_params)})"
        )
        print(
            f"[audit][{task_name}] model trainable/all={model_train}/{model_total} "
            f"agg trainable/all={agg_train}/{agg_total} head trainable/all={head_train}/{head_total}"
        )
        print(
            f"[audit][{task_name}] embedding t_list={t_list} capture_layers={capture_layers} "
            f"aggregator={agg_type} pool={pool_mode}"
        )
        print("-" * 100)

    if task_type == "classification":
        use_class_weights = bool(train_cfg.get("use_class_weights", True))
        if use_class_weights:
            y_cls = np.asarray(y_train_np, dtype=np.int64)
            counts_np = np.bincount(y_cls, minlength=int(num_classes)).astype(np.float32)
            counts = torch.from_numpy(counts_np)
            total = float(np.sum(counts_np))
            denom = (counts * float(num_classes)).clamp_min(1.0)
            w = total / denom
            w[counts == 0] = 0.0
            loss_fn = nn.CrossEntropyLoss(weight=w.to(device))
        else:
            loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()

    params: List[Dict[str, Any]] = []
    lr_dit = float(opt_cfg.get("lr_dit", train_cfg.get("lr_dit", 1.0e-5)))
    lr_head = float(opt_cfg.get("lr_head", train_cfg.get("lr_head", 1.0e-4)))
    lr_agg = float(opt_cfg.get("lr_agg", train_cfg.get("lr_agg", 1.0e-4)))
    wd_dit = float(opt_cfg.get("weight_decay_dit", train_cfg.get("weight_decay_dit", 1.0e-4)))
    wd_head = float(opt_cfg.get("weight_decay_head", train_cfg.get("weight_decay_head", 0.0)))
    wd_agg = float(opt_cfg.get("weight_decay_agg", train_cfg.get("weight_decay_agg", 0.0)))

    if model_trainable:
        dit_params = [p for p in model.parameters() if p.requires_grad]
        if len(dit_params) > 0:
            params.append({"params": dit_params, "lr": lr_dit, "weight_decay": wd_dit})
    params.append({"params": aggregator.parameters(), "lr": lr_agg, "weight_decay": wd_agg})
    params.append({"params": head.parameters(), "lr": lr_head, "weight_decay": wd_head})

    optimizer = torch.optim.AdamW(
        params,
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        eps=float(opt_cfg.get("eps", 1.0e-8)),
    )

    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epochs = int(train_cfg.get("epochs", 20))
    patience = int(train_cfg.get("patience", 10))
    best_metric_name = str(train_cfg.get("best_metric", "f1_weighted" if task_type == "classification" else "pearson"))
    eval_test_each_epoch = bool(train_cfg.get("eval_test_each_epoch", False))
    history_flush_each_epoch = bool(train_cfg.get("history_flush_each_epoch", True))
    print_epoch_metrics = bool(train_cfg.get("print_epoch_metrics", True))

    history: List[Dict[str, Any]] = []
    metric_name_norm = str(best_metric_name).strip().lower()
    higher_is_better = metric_name_norm not in {"loss", "val_loss", "mse", "mae"}
    best_val = -1.0e18 if higher_is_better else 1.0e18
    bad = 0
    best_state: Optional[Dict[str, Any]] = None

    for ep in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(ep)

        if model_trainable:
            model.train()
        else:
            model.eval()
        aggregator.train()
        head.train()

        loss_sum = 0.0
        n_obs = 0
        last_w = None

        iterator = train_loader
        if dist_ctx.is_main:
            iterator = tqdm(train_loader, desc=f"{task_name} train[{ep}/{epochs}]", ncols=120)

        for batch in iterator:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                y, pred, weights = _forward_batch(
                    batch=batch,
                    protocol=protocol,
                    t_list=t_list,
                    agg_type=agg_type,
                    pool_mode=pool_mode,
                    model_trainable=model_trainable,
                    aggregator=aggregator,
                    head=head,
                    device=device,
                    noise_mode=noise_mode,
                )
                if task_type == "classification":
                    loss = loss_fn(pred, y.long().view(-1))
                else:
                    loss = loss_fn(pred.view(-1), y.view(-1))

            if use_amp:
                scaler.scale(loss).backward()
                if float(train_cfg.get("grad_clip_norm", 0.0)) > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None],
                        max_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if float(train_cfg.get("grad_clip_norm", 0.0)) > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None],
                        max_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
                    )
                optimizer.step()

            bsz = int(y.shape[0])
            loss_sum += float(loss.item()) * max(1, bsz)
            n_obs += bsz
            last_w = weights
            if dist_ctx.is_main and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(loss=f"{loss_sum/max(1,n_obs):.5f}", w=_format_weights_for_tqdm(last_w))

        total_loss_sum = _all_reduce_sum_scalar(loss_sum, device=device, dist_ctx=dist_ctx)
        total_n_obs = _all_reduce_sum_scalar(float(n_obs), device=device, dist_ctx=dist_ctx)
        train_loss = float(total_loss_sum / max(1.0, total_n_obs))

        tr_eval_loss, tr_eval_metrics, _tr_rows = _eval_loader(
            loader=train_eval_loader,
            task_type=task_type,
            num_classes=num_classes,
            loss_fn=loss_fn,
            protocol=protocol,
            t_list=t_list,
            agg_type=agg_type,
            pool_mode=pool_mode,
            model_trainable=model_trainable,
            aggregator=aggregator,
            head=head,
            device=device,
            noise_mode=noise_mode,
            dist_ctx=dist_ctx,
            desc=None,
            return_rows=False,
        )

        val_loss, val_metrics, _val_rows = _eval_loader(
            loader=val_loader,
            task_type=task_type,
            num_classes=num_classes,
            loss_fn=loss_fn,
            protocol=protocol,
            t_list=t_list,
            agg_type=agg_type,
            pool_mode=pool_mode,
            model_trainable=model_trainable,
            aggregator=aggregator,
            head=head,
            device=device,
            noise_mode=noise_mode,
            dist_ctx=dist_ctx,
            desc=None,
            return_rows=False,
        )

        test_loss = None
        test_metrics = None
        if eval_test_each_epoch:
            test_loss, test_metrics, _test_rows = _eval_loader(
                loader=test_loader,
                task_type=task_type,
                num_classes=num_classes,
                loss_fn=loss_fn,
                protocol=protocol,
                t_list=t_list,
                agg_type=agg_type,
                pool_mode=pool_mode,
                model_trainable=model_trainable,
                aggregator=aggregator,
                head=head,
                device=device,
                noise_mode=noise_mode,
                dist_ctx=dist_ctx,
                desc=None,
                return_rows=False,
            )

        rec = {
            "epoch": ep,
            "train_loss": float(train_loss),
            "train_eval_loss": float(tr_eval_loss),
            "val_loss": float(val_loss),
        }
        rec.update({f"train_{k}": float(v) for k, v in tr_eval_metrics.items()})
        rec.update({f"val_{k}": float(v) for k, v in val_metrics.items()})
        if test_loss is not None and test_metrics is not None:
            rec["test_loss"] = float(test_loss)
            rec.update({f"test_{k}": float(v) for k, v in test_metrics.items()})
        if dist_ctx.is_main:
            history.append(rec)
            if history_flush_each_epoch:
                save_json({"history": history}, os.path.join(output_dir, "history.json"))
            if print_epoch_metrics:
                msg = (
                    f"[epoch {ep}/{epochs}] train_loss={float(train_loss):.5f} "
                    f"val_loss={float(val_loss):.5f}"
                )
                key_val_epoch = _primary_metric(task_type, best_metric_name, val_metrics)
                msg += f" val_{best_metric_name}={float(key_val_epoch):.5f}"
                if test_loss is not None and test_metrics is not None:
                    test_key = _primary_metric(task_type, best_metric_name, test_metrics)
                    msg += f" test_loss={float(test_loss):.5f} test_{best_metric_name}={float(test_key):.5f}"
                print(msg)

        key_val = _primary_metric(task_type, best_metric_name, val_metrics)
        is_better = (key_val > best_val) if higher_is_better else (key_val < best_val)

        if is_better:
            best_val = float(key_val)
            bad = 0
            best_state = {
                "model": _state_dict_cpu(model) if model_trainable else None,
                "aggregator": {k: v.detach().cpu() for k, v in _unwrap_module(aggregator).state_dict().items()},
                "head": {k: v.detach().cpu() for k, v in _unwrap_module(head).state_dict().items()},
                "epoch": ep,
                "metric": float(best_val),
            }
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is None:
        raise RuntimeError(f"Task {task_name}: best_state is None")

    if model_trainable and best_state.get("model", None) is not None:
        _load_state_dict(model, best_state["model"])
    _unwrap_module(aggregator).load_state_dict(best_state["aggregator"], strict=True)
    _unwrap_module(head).load_state_dict(best_state["head"], strict=True)

    tr_loss, tr_metrics, tr_rows = _eval_loader(
        loader=train_eval_loader,
        task_type=task_type,
        num_classes=num_classes,
        loss_fn=loss_fn,
        protocol=protocol,
        t_list=t_list,
        agg_type=agg_type,
        pool_mode=pool_mode,
        model_trainable=model_trainable,
        aggregator=aggregator,
        head=head,
        device=device,
        noise_mode=noise_mode,
        dist_ctx=dist_ctx,
        desc=("eval_train" if dist_ctx.is_main else None),
        return_rows=True,
    )
    va_loss, va_metrics, va_rows = _eval_loader(
        loader=val_loader,
        task_type=task_type,
        num_classes=num_classes,
        loss_fn=loss_fn,
        protocol=protocol,
        t_list=t_list,
        agg_type=agg_type,
        pool_mode=pool_mode,
        model_trainable=model_trainable,
        aggregator=aggregator,
        head=head,
        device=device,
        noise_mode=noise_mode,
        dist_ctx=dist_ctx,
        desc=("eval_val" if dist_ctx.is_main else None),
        return_rows=True,
    )
    te_loss, te_metrics, te_rows = _eval_loader(
        loader=test_loader,
        task_type=task_type,
        num_classes=num_classes,
        loss_fn=loss_fn,
        protocol=protocol,
        t_list=t_list,
        agg_type=agg_type,
        pool_mode=pool_mode,
        model_trainable=model_trainable,
        aggregator=aggregator,
        head=head,
        device=device,
        noise_mode=noise_mode,
        dist_ctx=dist_ctx,
        desc=("eval_test" if dist_ctx.is_main else None),
        return_rows=True,
    )

    ckpt_path = os.path.join(output_dir, "downstream_best.pt")
    if dist_ctx.is_main:
        ensure_dir(output_dir)
        _write_rows_csv(tr_rows, os.path.join(output_dir, "pred_train.csv"))
        _write_rows_csv(va_rows, os.path.join(output_dir, "pred_val.csv"))
        _write_rows_csv(te_rows, os.path.join(output_dir, "pred_test.csv"))

        torch.save(
            {
                "task_name": task_name,
                "task_type": task_type,
                "target_col": target_col,
                "num_classes": int(num_classes),
                "mode": mode,
                "best_state": best_state,
                "config": cfg,
                "capture_layers": capture_layers,
                "t_list": t_list,
            },
            ckpt_path,
        )

        save_json({"history": history}, os.path.join(output_dir, "history.json"))

    summary = {
        "task_name": task_name,
        "mode": mode,
        "task_type": task_type,
        "target_col": target_col,
        "num_classes": int(num_classes),
        "label_audit": label_audit,
        "label_zscore": (
            None
            if target_norm is None
            else {"mean": float(target_norm[0]), "std": float(target_norm[1])}
        ),
        "split_audit": {
            "train": train_ds.audit,
            "val": val_ds.audit,
            "test": test_ds.audit,
        },
        "embedding": {
            "t_list": [int(x) for x in t_list],
            "capture_layers": [int(x) for x in capture_layers],
            "pool": pool_mode,
            "noise_mode": noise_mode,
            "noise_seed": int(emb_cfg.get("noise_seed", 0)),
        },
        "metrics": {
            "train_loss": float(tr_loss),
            "val_loss": float(va_loss),
            "test_loss": float(te_loss),
            "train": {k: float(v) for k, v in tr_metrics.items()},
            "val": {k: float(v) for k, v in va_metrics.items()},
            "test": {k: float(v) for k, v in te_metrics.items()},
            "best_val": float(best_val),
            "best_metric_name": best_metric_name,
        },
        "checkpoint": ckpt_path,
    }
    if dist_ctx.is_main:
        save_json(summary, os.path.join(output_dir, "summary.json"))

    return summary


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)
    cfg["_config_path"] = os.path.abspath(args.config)

    seed = int(cfg.get("seed", cfg.get("train", {}).get("seed", 0)))
    set_seed(seed)

    gpu_ids = _resolve_gpu_ids(cfg)
    dist_ctx, device = _setup_dist_context(cfg, gpu_ids)
    cuda_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0

    ckpt_cfg = cfg.get("ckpt", cfg.get("checkpoints", {}))
    ckpt_path = str(
        ckpt_cfg.get("checkpoint", ckpt_cfg.get("cond_dit_ckpt", ckpt_cfg.get("stage1_ckpt", "")))
    ).strip()
    if ckpt_path == "":
        raise ValueError("ckpt.checkpoint (or cond_dit_ckpt) is required")
    ckpt_path = os.path.abspath(ckpt_path)

    if bool(ckpt_cfg.get("auto_load_model_config", True)):
        ckpt_tmp = torch.load(ckpt_path, map_location="cpu")
        mcfg = ckpt_tmp.get("model_config", None)
        if isinstance(mcfg, dict):
            for key in ("model", "diffusion", "conditioning", "diversity", "loss"):
                if key in mcfg and key not in cfg:
                    cfg[key] = copy.deepcopy(mcfg[key])

            ckpt_data = mcfg.get("data", None)
            if isinstance(ckpt_data, dict):
                ckpt_cond = ckpt_data.get("conditions", None)
                if isinstance(ckpt_cond, dict):
                    if not isinstance(cfg.get("data", None), dict):
                        cfg["data"] = {}
                    if not isinstance(cfg["data"].get("conditions", None), dict):
                        cfg["data"]["conditions"] = {}
                    for k, v in ckpt_cond.items():
                        if k not in cfg["data"]["conditions"]:
                            cfg["data"]["conditions"][k] = copy.deepcopy(v)

    schedule = normalize_schedule_type(cfg.get("diffusion", {}).get("schedule", "linear"))

    train_ds, val_ds, test_ds, audits = build_splits_from_config(cfg)
    model = _build_model_from_dataset(cfg, train_ds)

    diffusion = GaussianDiffusion(
        num_steps=int(cfg.get("diffusion", {}).get("num_steps", 1000)),
        beta_start=float(cfg.get("diffusion", {}).get("beta_start", 1.0e-4)),
        beta_end=float(cfg.get("diffusion", {}).get("beta_end", 2.0e-2)),
        schedule=schedule,
        cosine_s=float(cfg.get("diffusion", {}).get("cosine_s", 0.008)),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model" not in ckpt:
        raise KeyError(f"checkpoint missing model key: {ckpt_path}")
    _load_state_dict(
        model,
        ckpt["model"],
        allow_condition_mismatch=bool(ckpt_cfg.get("allow_condition_mismatch", False)),
    )
    model = model.to(device)

    use_ddp = bool(dist_ctx.enabled)
    use_data_parallel = bool((not use_ddp) and device.type == "cuda" and len(gpu_ids) > 1)
    if use_ddp:
        ddp_cfg = cfg.get("ddp", {}) if isinstance(cfg.get("ddp", {}), dict) else {}
        ddp_find_unused = bool(ddp_cfg.get("find_unused_parameters", True))
        model = DDP(
            model,
            device_ids=[int(dist_ctx.local_rank)],
            output_device=int(dist_ctx.local_rank),
            find_unused_parameters=ddp_find_unused,
        )
    elif use_data_parallel:
        model = nn.DataParallel(model, device_ids=[int(g) for g in gpu_ids], output_device=int(gpu_ids[0])).to(device)
    model.eval()

    emb_cfg = cfg.get("embedding", {})
    t_list = _parse_t_list(emb_cfg)
    core_model = _unwrap_module(model)
    capture_layers = resolve_capture_layers(int(getattr(core_model, "depth")), list(emb_cfg.get("capture_layers", [-1])))

    protocol_cfg = FeatureProtocolCondConfig(
        timestep=int(emb_cfg.get("timestep", (t_list[0] if len(t_list) > 0 else 10))),
        capture_layers=[int(x) for x in capture_layers],
        noise_mode=str(emb_cfg.get("noise_mode", "per_subject")),
        noise_seed=int(emb_cfg.get("noise_seed", 0)),
    )
    protocol = FeatureProtocolCond(model=model, diffusion=diffusion, cfg=protocol_cfg, device=device)

    patch_info = compute_patch_audit(train_ds.target_shape, cfg.get("model", {}).get("patch_size", [1, 2, 1]))

    out_cfg = cfg.get("output", {}) if isinstance(cfg.get("output", {}), dict) else {}
    out_root = os.path.abspath(str(out_cfg.get("out_root", "outputs/cond_dit_downstream_general")))
    run_name = str(out_cfg.get("run_name", "")).strip()

    if dist_ctx.is_main:
        run_out = _make_run_output_dir(
            out_root=out_root,
            config_path=os.path.abspath(args.config),
            ckpt_path=ckpt_path,
            run_name=(run_name if run_name != "" else None),
        )
    else:
        run_out = ""
    if dist_ctx.enabled:
        payload: List[str] = [run_out]
        dist.broadcast_object_list(payload, src=0)
        run_out = str(payload[0])
        ensure_dir(run_out)

    run_context = {
        "config": os.path.abspath(args.config),
        "config_tag": _config_tag(os.path.abspath(args.config)),
        "config_fingerprint": _short_sha1_text(f"{os.path.abspath(args.config)}::{ckpt_path}"),
        "checkpoint": ckpt_path,
        "checkpoint_tag": _checkpoint_tag(ckpt_path),
        "parallel_mode": ("ddp" if use_ddp else ("data_parallel" if use_data_parallel else "single_device")),
        "world_size": int(dist_ctx.world_size),
        "rank": int(dist_ctx.rank),
        "local_rank": int(dist_ctx.local_rank),
        "run_output_dir": run_out,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if dist_ctx.is_main:
        save_json(run_context, os.path.join(run_out, "run_context.json"))

    active_gpu_ids: List[int] = []
    if use_ddp:
        active_gpu_ids = [int(dist_ctx.local_rank)]
    elif device.type == "cuda":
        if len(gpu_ids) > 0:
            active_gpu_ids = [int(g) for g in gpu_ids]
        elif device.index is not None:
            active_gpu_ids = [int(device.index)]
        else:
            active_gpu_ids = [0]

    if use_ddp:
        parallel_mode = "ddp"
    elif isinstance(model, nn.DataParallel):
        parallel_mode = "data_parallel"
    else:
        parallel_mode = "single_device"

    if dist_ctx.is_main:
        print("=" * 100)
        print(f"[audit] config={os.path.abspath(args.config)}")
        print(f"[audit] checkpoint={ckpt_path}")
        print(f"[audit] device={device}")
        print(f"[audit] world_size={dist_ctx.world_size} rank={dist_ctx.rank} local_rank={dist_ctx.local_rank}")
        print(f"[audit] cuda.device_count={cuda_count}")
        print(f"[audit] requested_gpu_ids={gpu_ids}")
        print(f"[audit] active_gpu_ids={active_gpu_ids}")
        print(f"[audit] parallel_mode={parallel_mode}")
        print(f"[audit] target_shape={tuple(int(v) for v in train_ds.target_shape)}")
        print(f"[audit] condition_shapes fc={train_ds.fc_shape} mri={train_ds.mri_shape} metadata={(train_ds.meta_dim if train_ds.meta_dim > 0 else None)}")
        print(f"[audit] diffusion.schedule={schedule}")
        print(f"[audit] embedding.t_list={t_list} capture_layers={capture_layers} noise_mode={protocol_cfg.noise_mode}")
        print(f"[audit] patch_num={patch_info['patch_num']} patch_dim={patch_info['patch_dim']} grid={patch_info['grid_shape']}")
        print(f"[audit] split train={audit_to_dict(audits['train'])}")
        print(f"[audit] split val={audit_to_dict(audits['val'])}")
        print(f"[audit] split test={audit_to_dict(audits['test'])}")
        print("=" * 100)

    tasks = _resolve_tasks(cfg)
    run_mode_filter = str(os.environ.get("DOWNSTREAM_RUN_MODE", "all")).strip().lower()
    if run_mode_filter in {"full_finetune", "linear_probe"}:
        filtered: List[Dict[str, Any]] = []
        for t in tasks:
            mode_t = str(t.get("mode", cfg.get("mode", "linear_probe"))).strip().lower()
            if mode_t == run_mode_filter:
                filtered.append(t)
        if len(filtered) == 0:
            raise ValueError(f"No enabled tasks match DOWNSTREAM_RUN_MODE={run_mode_filter}")
        tasks = filtered
    elif run_mode_filter not in {"", "all"}:
        raise ValueError("DOWNSTREAM_RUN_MODE must be one of: all|full_finetune|linear_probe")

    if dist_ctx.is_main:
        print(f"[audit] task_filter={run_mode_filter if run_mode_filter != '' else 'all'} selected={len(tasks)}")

    results: List[Dict[str, Any]] = []

    base_model_state = _state_dict_cpu(model)

    for i, task_raw in enumerate(tasks, start=1):
        task_name, task_cfg = _build_task_cfg(cfg, task_raw, i)
        task_out_cfg = task_cfg.get("_task_output_dir", None)
        if task_out_cfg is None:
            task_out_dir = os.path.join(run_out, task_name)
        else:
            if os.path.isabs(str(task_out_cfg)):
                task_out_dir = str(task_out_cfg)
            else:
                task_out_dir = os.path.join(run_out, str(task_out_cfg))

        if dist_ctx.is_main:
            print("-" * 100)
            print(f"[task {i}/{len(tasks)}] name={task_name}")
            print(f"[task {i}/{len(tasks)}] output={task_out_dir}")
            print("-" * 100)

        try:
            summary = _run_one_task(
                task_name=task_name,
                cfg=task_cfg,
                model=model,
                base_model_state=base_model_state,
                protocol=protocol,
                train_ds_base=train_ds,
                val_ds_base=val_ds,
                test_ds_base=test_ds,
                output_dir=task_out_dir,
                device=device,
                dist_ctx=dist_ctx,
            )
            results.append({"task_name": task_name, "ok": True, "output_dir": task_out_dir, "summary": summary})
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            results.append({"task_name": task_name, "ok": False, "output_dir": task_out_dir, "error": err_msg})
            if dist_ctx.enabled:
                try:
                    ensure_dir(run_out)
                    with open(os.path.join(run_out, f"rank{dist_ctx.rank}_fatal.txt"), "w", encoding="utf-8") as f:
                        f.write(f"task={task_name}\n")
                        f.write(f"rank={dist_ctx.rank} local_rank={dist_ctx.local_rank}\n")
                        f.write(f"error={err_msg}\n")
                except Exception:
                    pass

                if dist_ctx.is_main:
                    partial = {
                        "config": os.path.abspath(args.config),
                        "checkpoint": ckpt_path,
                        "task_filter": run_mode_filter if run_mode_filter != "" else "all",
                        "run_output_dir": run_out,
                        "run_context": os.path.join(run_out, "run_context.json"),
                        "num_tasks": len(tasks),
                        "results": results,
                        "aborted": True,
                    }
                    save_json(partial, os.path.join(run_out, "multi_task_summary.json"))

                _abort_dist_process_group(
                    dist_ctx,
                    reason=f"task={task_name} failed on rank={dist_ctx.rank}: {err_msg}",
                )
                raise
            if bool(cfg.get("stop_on_error", True)):
                break

    final = {
        "config": os.path.abspath(args.config),
        "checkpoint": ckpt_path,
        "task_filter": run_mode_filter if run_mode_filter != "" else "all",
        "run_output_dir": run_out,
        "run_context": os.path.join(run_out, "run_context.json"),
        "num_tasks": len(tasks),
        "results": results,
    }
    if dist_ctx.is_main:
        save_json(final, os.path.join(run_out, "multi_task_summary.json"))

        ok_cnt = sum(1 for r in results if bool(r.get("ok", False)))
        print("=" * 100)
        print(f"[done] completed_tasks={ok_cnt}/{len(tasks)}")
        print(f"[done] summary={os.path.join(run_out, 'multi_task_summary.json')}")

    if dist_ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
