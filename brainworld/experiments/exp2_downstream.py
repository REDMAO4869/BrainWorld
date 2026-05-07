from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import shutil
import sys
import time
import warnings
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

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from brainworld.dit.data import SampleLoadError, audit_to_dict, build_splits_from_config, collate_batch
from brainworld.dit.diffusion import GaussianDiffusion, normalize_schedule_type
from brainworld.dit.downstream_aggregators import build_layer_aggregator
from brainworld.dit.downstream_heads import build_head
from brainworld.dit.downstream_metrics import evaluate_classification, evaluate_regression
from brainworld.dit.downstream_protocol import FeatureProtocolCond, FeatureProtocolCondConfig, resolve_capture_layers
from brainworld.dit.model import ConditionalLatentDiT, compute_patch_audit
from brainworld.dit.utils import ensure_dir, load_json, resolve_device, save_json, set_seed

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only=False.*")


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
                    device_id=torch.device(f"cuda:{local_rank}"),
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
        logical_gpu_ids = _logical_gpu_ids_for_current_process(gpu_ids)
        bad_ids = [int(g) for g in logical_gpu_ids if int(g) < 0 or int(g) >= int(cuda_count)]
        if len(bad_ids) > 0:
            raise ValueError(f"logical gpu ids out of range for visible CUDA devices ({cuda_count}): {bad_ids}")
        device = torch.device(f"cuda:{int(logical_gpu_ids[0])}")
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


def _norm_subject_id(s: str) -> str:
    return _clean_subject(s).lower()


def _read_subject_split(path: str, subject_col: str = "Subject") -> set[str]:
    path = os.path.abspath(str(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"subject split csv not found: {path}")

    out: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        has_header = "," in first_line and _norm_col_name(subject_col) in [
            _norm_col_name(x) for x in first_line.split(",")
        ]
        if has_header:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return out
            raw_col = _resolve_csv_col(reader.fieldnames, subject_col)
            for row in reader:
                sid = _norm_subject_id(row.get(raw_col, ""))
                if sid != "":
                    out.add(sid)
        else:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 0:
                    continue
                sid = _norm_subject_id(row[0])
                if sid != "":
                    out.add(sid)
    return out


def _ensure_str_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if str(x).strip() != ""]
    return [str(v)]


def _split_subject_path(data_cfg: Dict[str, Any], split: str) -> str:
    splits = data_cfg.get("subject_splits", data_cfg.get("splits", {}))
    if isinstance(splits, dict):
        raw = splits.get(split, None)
        if isinstance(raw, dict):
            raw = raw.get("csv", raw.get("path", None))
        if raw is not None and str(raw).strip() != "":
            return os.path.abspath(str(raw))

    root = str(data_cfg.get("subject_split_root", "")).strip()
    datasets = _ensure_str_list(data_cfg.get("datasets", data_cfg.get("dataset", [])))
    if root != "" and len(datasets) == 1:
        return os.path.abspath(os.path.join(root, datasets[0], f"{split}.csv"))
    return ""


def _infer_sample_dataset(sample: Any, candidates: Sequence[str]) -> str:
    haystacks = [
        str(getattr(sample, "target_npz_path", "")),
        str(getattr(sample, "target_source_path", "")),
        str(getattr(sample, "fc_array_path", "")),
        str(getattr(sample, "mri_array_path", "")),
        str(getattr(sample, "sequence_id", "")),
    ]
    for name in sorted([str(x) for x in candidates], key=len, reverse=True):
        if name == "":
            continue
        needle = name.lower()
        for h in haystacks:
            hp = h.replace("\\", "/").lower()
            parts = [p for p in hp.split("/") if p != ""]
            if needle in parts or f"/{needle}/" in hp:
                return name
    return ""


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


class FilteredPairDataset(Dataset):
    def __init__(
        self,
        *,
        base_ds: Dataset,
        keep_indices: Sequence[int],
        split_name: str,
        audit: Dict[str, Any],
    ) -> None:
        if len(keep_indices) == 0:
            raise RuntimeError(f"No samples left after exp2 subject/dataset filtering for split={split_name}")
        self.base_ds = base_ds
        self.keep_indices = [int(x) for x in keep_indices]
        self.split_name = str(split_name)
        pair_samples = getattr(base_ds, "pair_samples", None)
        self.pair_samples = [pair_samples[i] for i in self.keep_indices] if pair_samples is not None else None
        self.subject_ids = [_norm_subject_id(_subject_id_from_base_ds(base_ds, i)) for i in self.keep_indices]
        self.audit = dict(audit)

        for attr in ("target_shape", "fc_shape", "mri_shape", "meta_dim"):
            if hasattr(base_ds, attr):
                setattr(self, attr, getattr(base_ds, attr))

    def __len__(self) -> int:
        return len(self.keep_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.pair_samples is None or not hasattr(self.base_ds, "_build_item"):
            return self.base_ds[self.keep_indices[int(idx)]]

        num_samples = len(self.pair_samples)
        start_idx = int(idx) % num_samples
        last_err: Optional[Exception] = None
        build_item = getattr(self.base_ds, "_build_item")

        for offset in range(num_samples):
            cur_idx = (start_idx + offset) % num_samples
            sample = self.pair_samples[cur_idx]
            try:
                return build_item(sample)
            except SampleLoadError as exc:
                last_err = exc
                warnings.warn(
                    f"skipping bad filtered sample split={self.split_name} idx={cur_idx} "
                    f"subject={getattr(sample, 'subject_id', '')} "
                    f"sequence={getattr(sample, 'sequence_id', '')} "
                    f"anchor={getattr(sample, 'anchor_chunk_id', '')}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

        raise RuntimeError(
            f"Unable to find a readable sample inside filtered split={self.split_name}, "
            f"start_idx={start_idx}; last_error={last_err}"
        )


def _condition_path_counts(ds: Dataset) -> Dict[str, int]:
    pair_samples = getattr(ds, "pair_samples", None)
    if pair_samples is None:
        base = getattr(ds, "base_ds", None)
        keep = getattr(ds, "keep_indices", None)
        base_pairs = getattr(base, "pair_samples", None)
        if base_pairs is not None and keep is not None:
            pair_samples = [base_pairs[int(i)] for i in keep]
    if pair_samples is None:
        return {"samples": int(len(ds)), "fc_path": -1, "mri_path": -1}

    fc = 0
    mri = 0
    for sample in pair_samples:
        if str(getattr(sample, "fc_array_path", "")).strip() != "":
            fc += 1
        if str(getattr(sample, "mri_array_path", "")).strip() != "":
            mri += 1
    return {"samples": int(len(pair_samples)), "fc_path": int(fc), "mri_path": int(mri)}


def _condition_enabled_summary(cfg: Dict[str, Any]) -> Dict[str, Dict[str, bool]]:
    cond = cfg.get("data", {}).get("conditions", {})
    out: Dict[str, Dict[str, bool]] = {}
    for name in ("fc", "mri", "metadata"):
        spec = cond.get(name, {}) if isinstance(cond, dict) else {}
        if not isinstance(spec, dict):
            spec = {}
        out[name] = {
            "enabled": bool(spec.get("enabled", False)),
            "required": bool(spec.get("required", False)),
        }
    return out


def _rank_print(dist_ctx: DistContext, msg: str) -> None:
    if dist_ctx.is_main:
        print(msg, flush=True)


def _show_tqdm(cfg: Dict[str, Any]) -> bool:
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg.get("runtime", {}), dict) else {}
    out_cfg = cfg.get("output", {}) if isinstance(cfg.get("output", {}), dict) else {}
    raw = runtime_cfg.get("show_tqdm", out_cfg.get("show_tqdm", False))
    if isinstance(raw, str):
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    return bool(raw)


def _filter_base_dataset_for_task(base_ds: Dataset, cfg: Dict[str, Any], split_name: str) -> Dataset:
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    datasets = _ensure_str_list(data_cfg.get("datasets", data_cfg.get("dataset", [])))
    dataset_candidates = _ensure_str_list(
        data_cfg.get("all_datasets", cfg.get("data", {}).get("target", {}).get("dataset_list", []))
    )
    if len(dataset_candidates) == 0:
        dataset_candidates = _ensure_str_list(cfg.get("data", {}).get("target", {}).get("dataset_list", []))
    if len(dataset_candidates) == 0:
        dataset_candidates = datasets

    wanted_datasets = {str(x).lower() for x in datasets if str(x).strip() != ""}
    subject_path = _split_subject_path(data_cfg, split_name)
    wanted_subjects: Optional[set[str]] = None
    if subject_path != "":
        wanted_subjects = _read_subject_split(subject_path, subject_col=str(data_cfg.get("subject_col", "Subject")))

    keep: List[int] = []
    dataset_counts: Dict[str, int] = defaultdict(int)
    missing_dataset = 0
    subject_filtered = 0
    dataset_filtered = 0

    pair_samples = getattr(base_ds, "pair_samples", None)
    for idx in range(len(base_ds)):
        sid = _norm_subject_id(_subject_id_from_base_ds(base_ds, idx))
        sample = pair_samples[idx] if pair_samples is not None else None
        ds_name = _infer_sample_dataset(sample, dataset_candidates) if sample is not None else ""
        if ds_name == "" and len(wanted_datasets) > 0:
            missing_dataset += 1
            dataset_filtered += 1
            continue
        if len(wanted_datasets) > 0 and ds_name.lower() not in wanted_datasets:
            dataset_filtered += 1
            continue
        if wanted_subjects is not None and sid not in wanted_subjects:
            subject_filtered += 1
            continue
        keep.append(idx)
        dataset_counts[ds_name if ds_name != "" else "unknown"] += 1

    max_subjects = int(data_cfg.get("max_subjects", 0) or 0)
    if max_subjects > 0:
        seen_subjects: set[str] = set()
        limited: List[int] = []
        for idx in keep:
            sid = _norm_subject_id(_subject_id_from_base_ds(base_ds, idx))
            if sid not in seen_subjects and len(seen_subjects) >= max_subjects:
                continue
            seen_subjects.add(sid)
            limited.append(idx)
        keep = limited

    max_samples_cfg = data_cfg.get("max_samples_per_split", data_cfg.get("max_samples", 0))
    max_samples = int(max_samples_cfg or 0)
    if max_samples > 0:
        keep = keep[:max_samples]

    audit = {
        "split": split_name,
        "source_samples": int(len(base_ds)),
        "kept_samples": int(len(keep)),
        "datasets": datasets,
        "subject_split_csv": subject_path,
        "subject_split_subjects": None if wanted_subjects is None else int(len(wanted_subjects)),
        "dataset_filtered_samples": int(dataset_filtered),
        "subject_filtered_samples": int(subject_filtered),
        "missing_dataset_samples": int(missing_dataset),
        "dataset_counts": {k: int(v) for k, v in sorted(dataset_counts.items())},
        "max_subjects": int(max_subjects),
        "max_samples_per_split": int(max_samples),
    }
    return FilteredPairDataset(base_ds=base_ds, keep_indices=keep, split_name=split_name, audit=audit)


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k, None), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _target_manifest_signature(cfg: Dict[str, Any]) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    target_cfg = data_cfg.get("target", {}) if isinstance(data_cfg.get("target", {}), dict) else {}
    parts: List[Tuple[str, Tuple[str, ...]]] = []
    for split in ("train", "val", "test"):
        parts.append((split, tuple(os.path.abspath(str(x)) for x in _ensure_str_list(target_cfg.get(split, [])))))
    return tuple(parts)


def _task_overrides_target_manifests(task_cfg: Dict[str, Any], root_cfg: Dict[str, Any]) -> bool:
    return _target_manifest_signature(task_cfg) != _target_manifest_signature(root_cfg)


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


def _contains_summary_json(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for _, _, files in os.walk(path):
        if "summary.json" in files:
            return True
    return False


def _remove_dir_if_no_summary(path: str) -> bool:
    path = os.path.abspath(str(path))
    if not os.path.isdir(path):
        return False
    if _contains_summary_json(path):
        return False
    shutil.rmtree(path)
    return True


def _remove_task_output_dir(path: str) -> bool:
    path = os.path.abspath(str(path))
    if not os.path.isdir(path):
        return False
    shutil.rmtree(path)
    return True


def _archive_good_output(
    *,
    src_dir: str,
    good_root: str,
    run_tag: str,
    task_name: str,
    seed: int,
    metric: str,
    value: float,
    pass_rank: int,
    candidate_name: str = "",
    keep_top_k: int = 3,
) -> str:
    src = os.path.abspath(str(src_dir))
    if not os.path.isdir(src):
        return ""
    keep_top_k = max(1, int(keep_top_k))
    metric_tag = _safe_name(str(metric))
    metric_prefers_smaller = _lower_bound_prefers_smaller(str(metric))
    cand_tag = _safe_name(str(candidate_name)) if str(candidate_name).strip() != "" else ""
    task_dir = os.path.abspath(os.path.join(str(good_root), _safe_name(str(task_name))))
    ensure_dir(task_dir)

    index_path = os.path.join(task_dir, "index.json")
    existing_records: List[Dict[str, Any]] = []
    if os.path.isfile(index_path):
        try:
            old_index = load_json(index_path)
            raw_records = old_index.get("saved_passes", []) if isinstance(old_index, dict) else []
            if isinstance(raw_records, list):
                for rec in raw_records:
                    if isinstance(rec, dict) and str(rec.get("saved_dir", "")).strip() != "":
                        existing_records.append(dict(rec))
        except Exception:
            existing_records = []

    candidate_pieces = [
        "candidate",
        _safe_name(str(run_tag)),
        f"seed_{int(seed):04d}",
        f"{metric_tag}_{float(value):.6f}",
    ]
    if cand_tag != "":
        candidate_pieces.append(cand_tag)
    candidate_dst = os.path.abspath(os.path.join(task_dir, "__".join(candidate_pieces)))
    if os.path.isdir(candidate_dst):
        shutil.rmtree(candidate_dst)
    shutil.copytree(src, candidate_dst)

    new_record = {
        "seed": int(seed),
        "metric": str(metric),
        "metric_value": float(value),
        "source_dir": src,
        "saved_dir": candidate_dst,
        "run_tag": str(run_tag),
        "candidate_name": str(candidate_name),
    }
    records = existing_records + [new_record]

    dedup: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        saved = os.path.abspath(str(rec.get("saved_dir", "")))
        if saved == "" or not os.path.isdir(saved):
            continue
        key = f"{rec.get('source_dir', saved)}::{rec.get('seed', '')}::{rec.get('candidate_name', '')}"
        old = dedup.get(key)
        missing_value = float("inf") if metric_prefers_smaller else -float("inf")
        rec_value = float(rec.get("metric_value", missing_value))
        old_value = missing_value if old is None else float(old.get("metric_value", missing_value))
        if old is None or (rec_value < old_value if metric_prefers_smaller else rec_value > old_value):
            dedup[key] = rec
    missing_value = float("inf") if metric_prefers_smaller else -float("inf")
    records = sorted(
        dedup.values(),
        key=lambda r: float(r.get("metric_value", missing_value)),
        reverse=not metric_prefers_smaller,
    )

    kept_records: List[Dict[str, Any]] = []
    kept_source_dirs: set[str] = set()
    for rank, rec in enumerate(records[:keep_top_k], start=1):
        rec_metric = str(rec.get("metric", metric))
        rec_metric_tag = _safe_name(rec_metric)
        rec_cand = _safe_name(str(rec.get("candidate_name", "")))
        pieces = [
            f"rank_{int(rank):02d}",
            _safe_name(str(rec.get("run_tag", run_tag))),
            f"seed_{int(rec.get('seed', 0)):04d}",
            f"{rec_metric_tag}_{float(rec.get('metric_value', 0.0)):.6f}",
        ]
        if rec_cand != "":
            pieces.append(rec_cand)
        ranked_dst = os.path.abspath(os.path.join(task_dir, "__".join(pieces)))
        cur_saved = os.path.abspath(str(rec.get("saved_dir", "")))
        if cur_saved != ranked_dst:
            if os.path.isdir(ranked_dst):
                shutil.rmtree(ranked_dst)
            shutil.move(cur_saved, ranked_dst)
        rec = dict(rec)
        rec["rank"] = int(rank)
        rec["saved_dir"] = ranked_dst
        kept_records.append(rec)
        kept_source_dirs.add(ranked_dst)

    for rec in records[keep_top_k:]:
        saved = os.path.abspath(str(rec.get("saved_dir", "")))
        if saved.startswith(task_dir) and os.path.isdir(saved) and saved not in kept_source_dirs:
            shutil.rmtree(saved)

    index = {
        "task_name": str(task_name),
        "metric": str(metric),
        "threshold": None,
        "prefers_smaller": bool(metric_prefers_smaller),
        "keep_top_k": int(keep_top_k),
        "saved_passes": kept_records,
    }
    save_json(index, index_path)

    for rec in kept_records:
        if os.path.abspath(str(rec.get("source_dir", ""))) == src and int(rec.get("seed", -1)) == int(seed):
            return str(rec.get("saved_dir", ""))
    return str(kept_records[0].get("saved_dir", "")) if len(kept_records) > 0 else ""


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


def _resolve_seed_list(cfg: Dict[str, Any], task_raw: Optional[Dict[str, Any]] = None) -> Tuple[List[int], bool]:
    raw: Any = None
    explicit = False
    if isinstance(task_raw, dict):
        if "seed_list" in task_raw:
            raw = task_raw.get("seed_list")
            explicit = True
        elif "seeds" in task_raw:
            raw = task_raw.get("seeds")
            explicit = True
        elif "seed" in task_raw:
            raw = [task_raw.get("seed")]
            explicit = True

    if raw is None:
        if "seed_list" in cfg:
            raw = cfg.get("seed_list")
            explicit = True
        elif "seeds" in cfg:
            raw = cfg.get("seeds")
            explicit = True
        else:
            raw = [cfg.get("seed", cfg.get("train", {}).get("seed", 0))]

    if isinstance(raw, (int, float, str)):
        raw_list = [raw]
    elif isinstance(raw, (list, tuple)):
        raw_list = list(raw)
    else:
        raise ValueError("seed_list/seeds must be an int or a list of ints")

    seeds: List[int] = []
    for x in raw_list:
        sx = str(x).strip()
        if sx == "":
            continue
        seeds.append(int(float(sx)))

    if len(seeds) == 0:
        raise ValueError("seed_list/seeds resolved to an empty list")
    return seeds, explicit


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
            sid = _norm_subject_id(row.get(sub_raw, ""))
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
        subject_ids: List[str] = []
        missing = 0
        for idx in range(len(self.base_ds)):
            sid = _norm_subject_id(_subject_id_from_base_ds(self.base_ds, idx))
            if sid not in target_map:
                missing += 1
                continue
            y = float(target_map[sid])
            if self.task_type == "regression" and self.target_norm is not None:
                mu, sd = self.target_norm
                y = (y - float(mu)) / float(sd)
            keep.append(int(idx))
            yvals.append(float(y))
            subject_ids.append(sid)

        if len(keep) == 0:
            raise RuntimeError(f"No labeled samples for split={self.split_name}")

        self.keep_indices = keep
        self.subject_ids = subject_ids
        self.yvals = np.asarray(yvals, dtype=np.float32)
        self.audit = {
            "split": self.split_name,
            "requested_samples": int(len(self.base_ds)),
            "used_samples": int(len(self.keep_indices)),
            "missing_label_samples": int(missing),
            "base_filter_audit": getattr(self.base_ds, "audit", None),
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


def _prepare_condition_inputs(
    batch: Dict[str, Any],
    device: torch.device,
    cond_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[torch.Tensor]]:
    out: Dict[str, Optional[torch.Tensor]] = {}
    cond_cfg = cond_cfg if isinstance(cond_cfg, dict) else {}
    fc_enabled = bool(cond_cfg.get("fc", {}).get("enabled", True)) if isinstance(cond_cfg.get("fc", {}), dict) else True
    mri_enabled = bool(cond_cfg.get("mri", {}).get("enabled", True)) if isinstance(cond_cfg.get("mri", {}), dict) else True
    meta_enabled = (
        bool(cond_cfg.get("metadata", {}).get("enabled", True))
        if isinstance(cond_cfg.get("metadata", {}), dict)
        else True
    )

    fc = batch.get("fc_cond", None)
    out["fc_cond"] = None if (fc is None or not fc_enabled) else fc.to(device=device, dtype=torch.float32, non_blocking=True)
    out["has_fc"] = None if not fc_enabled else batch.get("has_fc", None)
    if out["has_fc"] is not None:
        out["has_fc"] = out["has_fc"].to(device=device, dtype=torch.float32, non_blocking=True)

    mri = batch.get("mri_cond", None)
    out["mri_cond"] = None if (mri is None or not mri_enabled) else mri.to(device=device, dtype=torch.float32, non_blocking=True)
    out["has_mri"] = None if not mri_enabled else batch.get("has_mri", None)
    if out["has_mri"] is not None:
        out["has_mri"] = out["has_mri"].to(device=device, dtype=torch.float32, non_blocking=True)

    meta = batch.get("meta_cond", None)
    out["meta_cond"] = None if (meta is None or not meta_enabled) else meta.to(device=device, dtype=torch.float32, non_blocking=True)
    out["has_meta"] = None if not meta_enabled else batch.get("has_meta", None)
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
    if isinstance(cfg.get("runtime", None), dict):
        cand.append(cfg.get("runtime", {}).get("gpu_id", None))
        cand.append(cfg.get("runtime", {}).get("gpu_ids", None))
    if isinstance(cfg.get("train", None), dict):
        cand.append(cfg.get("train", {}).get("gpu_id", None))
        cand.append(cfg.get("train", {}).get("gpu_ids", None))
    if isinstance(cfg.get("training", None), dict):
        cand.append(cfg.get("training", {}).get("gpu_id", None))
        cand.append(cfg.get("training", {}).get("gpu_ids", None))
    cand.append(cfg.get("gpu_id", None))
    cand.append(cfg.get("gpu_ids", None))

    picked = None
    for v in cand:
        if v is not None:
            picked = v
            break
    if picked is None:
        return []

    if isinstance(picked, str):
        picked = [x.strip() for x in picked.split(",") if x.strip() != ""]
    elif isinstance(picked, (int, float)):
        picked = [int(picked)]
    elif not isinstance(picked, list):
        raise ValueError("gpu_id/gpu_ids must be a list, comma-separated string, or integer CUDA index")

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


def _logical_gpu_ids_for_current_process(gpu_ids: List[int]) -> List[int]:
    if not torch.cuda.is_available():
        return list(gpu_ids)
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if visible == "":
        return list(gpu_ids)
    cuda_count = int(torch.cuda.device_count())
    if cuda_count <= 0:
        return []
    if len(gpu_ids) <= 0:
        return list(range(cuda_count))
    return list(range(min(len(gpu_ids), cuda_count)))


def _extract_cache_feature_batch(
    *,
    batch: Dict[str, Any],
    protocol: FeatureProtocolCond,
    t_list: List[int],
    device: torch.device,
    noise_mode: str,
    cond_cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    x0 = batch["target_latent"].to(device=device, dtype=torch.float32)
    y = batch["y"].to(device=device, dtype=torch.float32)
    direction_id = batch["direction_id"].to(device=device, dtype=torch.long)
    cond_inputs = _prepare_condition_inputs(batch, device=device, cond_cfg=cond_cfg)
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
        "subjects_sha1": hashlib.sha1(
            "\n".join([str(x) for x in getattr(ds, "subject_ids", [])]).encode("utf-8")
        ).hexdigest(),
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
    sig = _cache_signature(cfg=cfg, ckpt_path=ckpt_path, split_name=split_name, ds=ds)
    cache_split_name = str(split_name)
    if cache_scope == "shared":
        sig_tag = _short_sha1_text(json.dumps(sig, sort_keys=True), n=12)
        cache_split_name = f"{split_name}__{sig_tag}"
    npz_path, meta_path = _cache_paths(cache_root, task_name, cache_split_name, cache_scope)

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
        it = tqdm(dl, desc=f"cache[{task_name}:{split_name}]", ncols=120) if _show_tqdm(cfg) else dl
        for batch in it:
            feat, y_batch, r = _extract_cache_feature_batch(
                batch=batch,
                protocol=protocol,
                t_list=t_list,
                device=device,
                noise_mode=noise_mode,
                cond_cfg=cfg.get("data", {}).get("conditions", {}),
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

    for key in ("label", "task_head", "head", "train", "training", "optim", "output", "data", "embedding", "aggregator", "eval_aggregation"):
        if isinstance(task_raw.get(key, None), dict):
            if key not in cfg or not isinstance(cfg.get(key, None), dict):
                cfg[key] = {}
            _deep_update(cfg[key], task_raw[key])

    if "task" in task_raw and isinstance(task_raw["task"], dict):
        if "task_head" not in cfg or not isinstance(cfg.get("task_head", None), dict):
            cfg["task_head"] = {}
        _deep_update(cfg["task_head"], task_raw["task"])

    if "datasets" in task_raw or "dataset" in task_raw or "subject_splits" in task_raw or "split_dir" in task_raw:
        if "data" not in cfg or not isinstance(cfg.get("data", None), dict):
            cfg["data"] = {}
        if "datasets" in task_raw:
            cfg["data"]["datasets"] = copy.deepcopy(task_raw["datasets"])
        if "dataset" in task_raw:
            cfg["data"]["datasets"] = [str(task_raw["dataset"])]
        if "subject_splits" in task_raw:
            cfg["data"]["subject_splits"] = copy.deepcopy(task_raw["subject_splits"])
        if "split_dir" in task_raw:
            split_dir = str(task_raw["split_dir"])
            cfg["data"]["subject_splits"] = {
                "train": os.path.join(split_dir, "train.csv"),
                "val": os.path.join(split_dir, "val.csv"),
                "test": os.path.join(split_dir, "test.csv"),
            }

    for k in ("max_subjects", "max_samples_per_split", "max_samples"):
        if k in task_raw:
            if "data" not in cfg or not isinstance(cfg.get("data", None), dict):
                cfg["data"] = {}
            cfg["data"][k] = task_raw[k]

    if "mode" in task_raw:
        cfg["mode"] = task_raw["mode"]

    if "output_dir" in task_raw:
        cfg["_task_output_dir"] = str(task_raw["output_dir"])
    elif "out_subdir" in task_raw:
        cfg["_task_output_dir"] = str(task_raw["out_subdir"])

    return name, cfg


def _materialize_exp2_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    data = cfg.setdefault("data", {})
    if not isinstance(data, dict):
        raise ValueError("data must be a dict")

    manifest_root = str(data.get("manifest_root", "")).strip()
    datasets = _ensure_str_list(data.get("datasets", data.get("all_datasets", [])))
    target = data.setdefault("target", {})
    if not isinstance(target, dict):
        raise ValueError("data.target must be a dict")

    if manifest_root != "" and len(datasets) > 0:
        target.setdefault("direct_pairing", True)
        target.setdefault("prefilter_bad_samples", False)
        target.setdefault("latent_field", "mu")
        target.setdefault("dataset_dtype", "float32")
        target.setdefault("dataset_list", datasets)
        for split in ("train", "val", "test"):
            if split not in target or len(_ensure_str_list(target.get(split, []))) == 0:
                target[split] = [os.path.join(manifest_root, ds, f"{split}.csv") for ds in datasets]
        target.setdefault(
            "csv_map",
            {
                "path": "vae_latent_path",
                "target_path": "target_latent_path",
                "target_source_path": "target_voxel_data_path",
                "subject": "Subject",
                "sequence": "sequence_id",
                "anchor_chunk": "anchor_chunk_id",
                "target_chunk": "target_chunk_id",
                "direction": "pair_direction",
                "source_path": "voxel_data_path",
                "fc_path": "fc_embedding_path",
                "mri_path": "MRI_embedding_path",
            },
        )

        cond = data.setdefault("conditions", {})
        if isinstance(cond, dict):
            for name, path_col, default_required in (
                ("fc", "fc_embedding_path", True),
                ("mri", "MRI_embedding_path", False),
            ):
                spec = cond.setdefault(name, {})
                if not isinstance(spec, dict):
                    continue
                spec.setdefault("enabled", True)
                spec.setdefault("required", default_required)
                spec.setdefault("alignment", "subject_chunk" if name == "fc" else "subject")
                spec.setdefault("dtype", "float32")
                spec.setdefault("array_field", "")
                spec.setdefault("mode", "vector")
                spec.setdefault("num_tokens", 1)
                spec.setdefault("csv_map", {"path": path_col, "subject": "Subject", "chunk": ["chunk_id", "Chunk"]})
                for split in ("train", "val", "test"):
                    if split not in spec or len(_ensure_str_list(spec.get(split, []))) == 0:
                        spec[split] = list(target[split])

    label_root = str(data.get("label_root", "")).strip()
    subject_split_root = str(data.get("subject_split_root", "")).strip()
    tasks = cfg.get("tasks", cfg.get("task_list", []))
    if isinstance(tasks, list):
        for task in tasks:
            if not isinstance(task, dict):
                continue
            label = task.setdefault("label", {})
            if not isinstance(label, dict):
                continue
            raw_label = str(label.get("label_csv_path", label.get("label_csv", ""))).strip()
            if raw_label != "" and label_root != "" and not os.path.isabs(raw_label):
                label["label_csv_path"] = os.path.join(label_root, raw_label)
            task_datasets = _ensure_str_list(task.get("datasets", task.get("dataset", [])))
            if len(task_datasets) == 1 and subject_split_root != "":
                task_data = task.setdefault("data", {})
                if isinstance(task_data, dict):
                    task_data.setdefault("subject_split_root", subject_split_root)
                    task_data.setdefault("datasets", task_datasets)

    data.setdefault("missing_required_policy", "drop")
    data.setdefault("verify_paths", False)
    return cfg


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


def _resolve_task_lower_bound(task_raw: Dict[str, Any], task_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = None
    for src in (task_raw, task_raw.get("train", {}), task_cfg, task_cfg.get("train", {})):
        if not isinstance(src, dict):
            continue
        if "lower_bound" in src:
            raw = src.get("lower_bound")
            break
        if "performance_lower_bound" in src:
            raw = src.get("performance_lower_bound")
            break

    if raw is None or raw == "":
        return None

    if isinstance(raw, dict):
        enabled = raw.get("enabled", True)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() not in {"0", "false", "no", "off"}
        if not bool(enabled):
            return None
        value = raw.get("value", raw.get("min", raw.get("threshold", raw.get("lower_bound", None))))
        if value is None or str(value).strip() == "":
            raise ValueError("lower_bound dict must contain value/min/threshold/lower_bound")
        metric = str(raw.get("metric", "")).strip()
        required_passes = int(raw.get("required_passes", raw.get("num_passes", raw.get("passes", raw.get("max_attempts", 3)))) or 0)
    else:
        value = raw
        metric = ""
        required_passes = 3

    train_cfg = task_cfg.get("train", {}) if isinstance(task_cfg.get("train", {}), dict) else {}
    task_head_cfg = task_cfg.get("task_head", task_cfg.get("task", {}))
    if not isinstance(task_head_cfg, dict):
        task_head_cfg = {}
    task_type = str(task_head_cfg.get("task_type", task_head_cfg.get("type", "classification"))).strip().lower()
    best_metric_name = str(train_cfg.get("best_metric", "f1_weighted" if task_type == "classification" else "pearson"))
    if metric == "":
        metric = f"test_{best_metric_name}"

    return {
        "value": float(value),
        "metric": metric,
        "best_metric_name": best_metric_name,
        "required_passes": int(required_passes),
    }


def _metric_from_summary_for_lower_bound(summary: Dict[str, Any], lower_bound_cfg: Dict[str, Any]) -> Optional[float]:
    metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics", {}), dict) else {}
    metric = str(lower_bound_cfg.get("metric", "")).strip()
    best_metric_name = str(lower_bound_cfg.get("best_metric_name", metrics.get("best_metric_name", ""))).strip()

    if metric in {"best", "best_val"}:
        val = metrics.get("best_val", None)
        return None if val is None else float(val)

    if metric == "":
        metric = f"test_{best_metric_name}" if best_metric_name != "" else "test"

    if metric in metrics and isinstance(metrics.get(metric, None), (int, float)):
        return float(metrics[metric])

    if "_" in metric:
        split, key = metric.split("_", 1)
        split_metrics = metrics.get(split, None)
        if isinstance(split_metrics, dict) and key in split_metrics:
            return float(split_metrics[key])

    for split in ("val", "test", "train"):
        split_metrics = metrics.get(split, None)
        if isinstance(split_metrics, dict) and metric in split_metrics:
            return float(split_metrics[metric])

    if best_metric_name != "":
        split_metrics = metrics.get(metric, None)
        if isinstance(split_metrics, dict) and best_metric_name in split_metrics:
            return float(split_metrics[best_metric_name])

    return None


def _lower_bound_prefers_smaller(metric_name: str) -> bool:
    metric_norm = str(metric_name).strip().lower()
    if metric_norm == "":
        return False
    smaller_tokens = ("loss", "mse", "mae", "rmse", "error")
    return any(token in metric_norm for token in smaller_tokens)


def _evaluate_lower_bound(summary: Dict[str, Any], lower_bound_cfg: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if lower_bound_cfg is None:
        return None
    value = _metric_from_summary_for_lower_bound(summary, lower_bound_cfg)
    if value is None:
        raise KeyError(f"lower_bound metric not found in summary: {lower_bound_cfg.get('metric', '')}")
    metric_name = str(lower_bound_cfg.get("metric", "best_val"))
    threshold = float(lower_bound_cfg["value"])
    if _lower_bound_prefers_smaller(metric_name):
        passed = bool(float(value) <= threshold)
    else:
        passed = bool(float(value) >= threshold)
    return {
        "metric": metric_name,
        "value": float(value),
        "threshold": threshold,
        "passed": passed,
    }


def _load_existing_good_passes(
    *,
    good_root: str,
    task_name: str,
    lower_bound_cfg: Optional[Dict[str, Any]],
    required_passes: int,
) -> List[Dict[str, Any]]:
    if lower_bound_cfg is None:
        return []
    task_dir = os.path.abspath(os.path.join(str(good_root), _safe_name(str(task_name))))
    if not os.path.isdir(task_dir):
        return []

    records: List[Dict[str, Any]] = []
    seen_summary_paths: set[str] = set()
    for root, _dirs, files in os.walk(task_dir):
        if "summary.json" not in files:
            continue
        summary_path = os.path.abspath(os.path.join(root, "summary.json"))
        if summary_path in seen_summary_paths:
            continue
        seen_summary_paths.add(summary_path)
        try:
            summary = load_json(summary_path)
            lb_result = _evaluate_lower_bound(summary, lower_bound_cfg)
        except Exception:
            continue
        if lb_result is None or not bool(lb_result.get("passed", False)):
            continue
        seed_raw = summary.get("seed", None) if isinstance(summary, dict) else None
        try:
            seed_val = int(seed_raw)
        except Exception:
            seed_val = -1
        records.append(
            {
                "task_name": str(task_name),
                "seed": int(seed_val),
                "metric": str(lb_result.get("metric", lower_bound_cfg.get("metric", ""))),
                "value": float(lb_result["value"]),
                "threshold": float(lb_result["threshold"]),
                "passed": True,
                "output_dir": os.path.abspath(root),
                "summary_path": summary_path,
                "source": "existing_good",
            }
        )

    metric_name = str(lower_bound_cfg.get("metric", ""))
    prefers_smaller = _lower_bound_prefers_smaller(metric_name)
    missing_value = float("inf") if prefers_smaller else -float("inf")
    records = sorted(
        records,
        key=lambda r: float(r.get("value", missing_value)),
        reverse=not prefers_smaller,
    )
    return records[: max(0, int(required_passes))]


def _format_lower_bound_top3(records: Sequence[Dict[str, Any]]) -> str:
    if len(records) == 0:
        return "top3=none"
    metric_name = ""
    for rec in records:
        metric_name = str(rec.get("metric", "")).strip()
        if metric_name != "":
            break
    prefers_smaller = _lower_bound_prefers_smaller(metric_name)
    missing_value = float("inf") if prefers_smaller else -float("inf")
    top = sorted(records, key=lambda r: float(r.get("value", missing_value)), reverse=not prefers_smaller)[:3]
    parts = []
    for i, rec in enumerate(top, start=1):
        seed_s = str(rec.get("seed", "?"))
        value_s = f"{float(rec.get('value', 0.0)):.6g}"
        status_s = "PASS" if bool(rec.get("passed", False)) else "below"
        cand_s = str(rec.get("candidate", "")).strip()
        suffix = f",{cand_s}" if cand_s != "" else ""
        parts.append(f"#{i}:{value_s}({status_s},seed={seed_s}{suffix})")
    return "top3=" + " ".join(parts)


def _resolve_task_lower_bound_search(cfg: Dict[str, Any], task_raw: Dict[str, Any]) -> Dict[str, Any]:
    raw = None
    for key in ("lower_bound_search", "sweep"):
        if isinstance(task_raw.get(key, None), dict):
            raw = task_raw.get(key)
            break
    if raw is None:
        for key in ("lower_bound_search", "sweep"):
            if isinstance(cfg.get(key, None), dict):
                raw = cfg.get(key)
                break

    if not isinstance(raw, dict):
        return {"enabled": False, "candidates": [], "keep_top_k": 0}

    enabled = raw.get("enabled", True)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() not in {"0", "false", "no", "off"}
    if not bool(enabled):
        return {"enabled": False, "candidates": [], "keep_top_k": int(raw.get("keep_top_k", 0) or 0)}

    raw_candidates = raw.get("candidates", raw.get("trials", []))
    if not isinstance(raw_candidates, list):
        raise ValueError("lower_bound_search.candidates/trials must be a list")

    candidates: List[Dict[str, Any]] = []
    for idx, cand in enumerate(raw_candidates):
        if not isinstance(cand, dict):
            raise ValueError("each lower_bound_search candidate must be a dict")
        name = _safe_name(cand.get("name", f"sweep_{idx:02d}"))
        if isinstance(cand.get("overrides", None), dict):
            overrides = copy.deepcopy(cand["overrides"])
        else:
            overrides = {k: copy.deepcopy(v) for k, v in cand.items() if k not in {"name", "enabled", "description"}}
        candidates.append({"name": name, "overrides": overrides})

    return {
        "enabled": bool(len(candidates) > 0),
        "candidates": candidates,
        "keep_top_k": int(raw.get("keep_top_k", 0) or 0),
    }


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
            for c in range(int(probs.shape[1])):
                row[f"prob_{c}"] = float(probs[i, c].item())
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


def _resolve_eval_aggregation_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    raw = cfg.get("eval_aggregation", None)
    if not isinstance(raw, dict):
        return {"enabled": False}

    enabled = raw.get("enabled", True)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() not in {"0", "false", "no", "off"}
    if not bool(enabled):
        return {"enabled": False}

    group_by = str(raw.get("group_by", raw.get("key", "subject_id"))).strip() or "subject_id"
    metric_prefix = str(raw.get("metric_prefix", "")).strip()
    if metric_prefix == "":
        metric_prefix = "subject" if group_by == "subject_id" else _safe_name(group_by)

    return {
        "enabled": True,
        "group_by": group_by,
        "metric_prefix": metric_prefix,
        "classification_reduce": str(raw.get("classification_reduce", "mean_prob")).strip().lower() or "mean_prob",
        "classification_threshold": str(
            raw.get("classification_threshold", raw.get("binary_threshold", "argmax"))
        ).strip().lower() or "argmax",
        "regression_reduce": str(raw.get("regression_reduce", "mean_pred")).strip().lower() or "mean_pred",
    }


def _evaluate_classification_from_pred(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    num_classes: int,
) -> Dict[str, float]:
    y = torch.as_tensor(list(y_true), dtype=torch.long).view(-1)
    pred = torch.as_tensor(list(y_pred), dtype=torch.long).view(-1)
    if int(y.numel()) == 0:
        return {"acc": 0.0, "balanced_acc": 0.0, "f1_weighted": 0.0}

    acc = float((pred == y).float().mean().item())

    f1_sum = 0.0
    support_sum = 0
    for c in range(int(num_classes)):
        yc = (y == c).long()
        pc = (pred == c).long()
        tp = ((yc == 1) & (pc == 1)).sum().item()
        fp = ((yc == 0) & (pc == 1)).sum().item()
        fn = ((yc == 1) & (pc == 0)).sum().item()
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1c = float(2.0 * prec * rec / max(1.0e-12, prec + rec))
        sup = int(yc.sum().item())
        f1_sum += f1c * sup
        support_sum += sup

    bal_acc = 0.0
    cls_present = 0
    for c in range(int(num_classes)):
        yc = (y == c)
        sup = int(yc.sum().item())
        if sup <= 0:
            continue
        cls_present += 1
        bal_acc += float((pred[yc] == c).float().mean().item())
    bal_acc = float(bal_acc / max(1, cls_present))

    return {
        "acc": acc,
        "balanced_acc": bal_acc,
        "f1_weighted": float(f1_sum / max(1, support_sum)),
    }


def _grouped_metrics_from_rows(
    *,
    rows: List[Dict[str, Any]],
    task_type: str,
    num_classes: int,
    eval_agg_cfg: Dict[str, Any],
) -> Dict[str, float]:
    if not bool(eval_agg_cfg.get("enabled", False)) or len(rows) == 0:
        return {}

    group_by = str(eval_agg_cfg.get("group_by", "subject_id")).strip() or "subject_id"
    metric_prefix = str(eval_agg_cfg.get("metric_prefix", "subject")).strip() or "subject"

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(group_by, "")).strip()
        if key == "":
            return {}
        grouped[key].append(row)

    if len(grouped) == 0:
        return {}

    if task_type == "classification":
        reduce_mode = str(eval_agg_cfg.get("classification_reduce", "mean_prob")).strip().lower()
        use_prob_mean = reduce_mode in {"mean_prob", "mean_probs", "avg_prob", "avg_probs"}
        y_group: List[int] = []
        pred_group: List[int] = []
        conflict_groups = 0
        for items in grouped.values():
            target_counts = Counter(int(float(r["target"])) for r in items)
            if len(target_counts) > 1:
                conflict_groups += 1
            y_group.append(int(target_counts.most_common(1)[0][0]))

            if use_prob_mean:
                prob_cols = [f"prob_{c}" for c in range(int(num_classes))]
                has_all_prob_cols = all(all(col in r for col in prob_cols) for r in items)
                if has_all_prob_cols:
                    mean_probs = [
                        float(np.mean([float(r[f"prob_{c}"]) for r in items]))
                        for c in range(int(num_classes))
                    ]
                    pred_group.append(int(np.argmax(np.asarray(mean_probs, dtype=np.float32))))
                    continue

            pred_counts = Counter(int(float(r["pred"])) for r in items)
            pred_group.append(int(pred_counts.most_common(1)[0][0]))

        metrics = _evaluate_classification_from_pred(y_group, pred_group, num_classes=int(num_classes))
        metrics["groups"] = float(len(grouped))
        metrics["target_conflicts"] = float(conflict_groups)
        return {f"{metric_prefix}_{k}": float(v) for k, v in metrics.items()}

    reduce_mode = str(eval_agg_cfg.get("regression_reduce", "mean_pred")).strip().lower()
    target_reduce = np.mean if reduce_mode == "mean_target" else (lambda arr: arr[0])
    y_group = []
    pred_group = []
    for items in grouped.values():
        y_vals = [float(r["target"]) for r in items]
        p_vals = [float(r["pred"]) for r in items]
        y_group.append(float(target_reduce(y_vals)))
        pred_group.append(float(np.mean(p_vals)))

    metrics = evaluate_regression(
        torch.as_tensor(y_group, dtype=torch.float32),
        torch.as_tensor(pred_group, dtype=torch.float32),
    )
    metrics["groups"] = float(len(grouped))
    return {f"{metric_prefix}_{k}": float(v) for k, v in metrics.items()}


def _binary_group_probs_targets(
    rows: List[Dict[str, Any]],
    *,
    group_by: str,
) -> Tuple[List[float], List[int]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(group_by, "")).strip()
        if key == "" or "prob_1" not in row:
            continue
        grouped[key].append(row)

    probs: List[float] = []
    targets: List[int] = []
    for items in grouped.values():
        if len(items) == 0:
            continue
        target_counts = Counter(int(float(r["target"])) for r in items)
        targets.append(int(target_counts.most_common(1)[0][0]))
        probs.append(float(np.mean([float(r["prob_1"]) for r in items])))
    return probs, targets


def _best_binary_threshold(
    rows: List[Dict[str, Any]],
    *,
    group_by: str,
) -> Optional[float]:
    probs, targets = _binary_group_probs_targets(rows, group_by=group_by)
    if len(probs) == 0 or len(set(targets)) < 2:
        return None

    values = sorted(set(float(p) for p in probs))
    candidates: List[float] = [0.0, 1.0]
    candidates.extend(values)
    candidates.extend(float((a + b) * 0.5) for a, b in zip(values[:-1], values[1:]))

    best_key: Optional[Tuple[float, float, float]] = None
    best_threshold: Optional[float] = None
    for threshold in candidates:
        pred = [1 if p >= threshold else 0 for p in probs]
        metrics = _evaluate_classification_from_pred(targets, pred, num_classes=2)
        key = (
            float(metrics.get("f1_weighted", 0.0)),
            float(metrics.get("balanced_acc", 0.0)),
            float(metrics.get("acc", 0.0)),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def _binary_threshold_grouped_metrics(
    rows: List[Dict[str, Any]],
    *,
    group_by: str,
    metric_prefix: str,
    threshold: float,
) -> Dict[str, float]:
    probs, targets = _binary_group_probs_targets(rows, group_by=group_by)
    if len(probs) == 0:
        return {}
    pred = [1 if p >= float(threshold) else 0 for p in probs]
    metrics = _evaluate_classification_from_pred(targets, pred, num_classes=2)
    metrics["groups"] = float(len(probs))
    metrics["threshold"] = float(threshold)
    return {f"{metric_prefix}_{k}": float(v) for k, v in metrics.items()}


def _apply_eval_threshold_calibration(
    *,
    task_type: str,
    num_classes: int,
    eval_agg_cfg: Dict[str, Any],
    train_rows: List[Dict[str, Any]],
    val_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
) -> None:
    mode = str(eval_agg_cfg.get("classification_threshold", "argmax")).strip().lower()
    if mode not in {"val_f1", "val_f1_weighted", "validation_f1", "validation_f1_weighted"}:
        return
    if task_type != "classification" or int(num_classes) != 2:
        return
    if not bool(eval_agg_cfg.get("enabled", False)):
        return
    reduce_mode = str(eval_agg_cfg.get("classification_reduce", "mean_prob")).strip().lower()
    if reduce_mode not in {"mean_prob", "mean_probs", "avg_prob", "avg_probs"}:
        return

    group_by = str(eval_agg_cfg.get("group_by", "subject_id")).strip() or "subject_id"
    metric_prefix = str(eval_agg_cfg.get("metric_prefix", "subject")).strip() or "subject"
    threshold = _best_binary_threshold(val_rows, group_by=group_by)
    if threshold is None:
        return

    train_metrics.update(
        _binary_threshold_grouped_metrics(
            train_rows, group_by=group_by, metric_prefix=metric_prefix, threshold=float(threshold)
        )
    )
    val_metrics.update(
        _binary_threshold_grouped_metrics(
            val_rows, group_by=group_by, metric_prefix=metric_prefix, threshold=float(threshold)
        )
    )
    test_metrics.update(
        _binary_threshold_grouped_metrics(
            test_rows, group_by=group_by, metric_prefix=metric_prefix, threshold=float(threshold)
        )
    )


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
    cond_cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    x0 = batch["target_latent"].to(device=device, dtype=torch.float32)
    y = batch["y"].to(device=device, dtype=torch.float32)
    direction_id = batch["direction_id"].to(device=device, dtype=torch.long)
    cond_inputs = _prepare_condition_inputs(batch, device=device, cond_cfg=cond_cfg)
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
    cond_cfg: Dict[str, Any],
    eval_agg_cfg: Dict[str, Any],
    desc: Optional[str] = None,
    return_rows: bool = True,
) -> Tuple[float, Dict[str, float], List[Dict[str, Any]]]:
    if model_trainable:
        protocol.model.eval()
    aggregator.eval()
    head.eval()

    all_y: List[torch.Tensor] = []
    all_out: List[torch.Tensor] = []
    rows: List[Dict[str, Any]] = []
    loss_sum = 0.0
    n_obs = 0

    it = loader
    if desc is not None and ((not dist_ctx.enabled) or dist_ctx.is_main):
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
                cond_cfg=cond_cfg,
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

    if len(all_y) > 0:
        y_local = torch.cat(all_y, dim=0)
        out_local = torch.cat(all_out, dim=0)
    else:
        y_local = torch.empty((0,), dtype=torch.float32)
        out_dim = int(num_classes) if task_type == "classification" else 1
        out_local = torch.empty((0, out_dim), dtype=torch.float32)

    if dist_ctx.enabled:
        packed: Dict[str, Any] = {
            "loss_sum": float(loss_sum),
            "n_obs": int(n_obs),
            "y": y_local,
            "out": out_local,
            "rows": rows if return_rows else [],
        }
        gathered: List[Optional[Dict[str, Any]]] = [None for _ in range(int(dist_ctx.world_size))]
        dist.all_gather_object(gathered, packed)
        result_obj: Optional[Tuple[float, Dict[str, float], List[Dict[str, Any]]]] = None
        if dist_ctx.is_main:
            y_parts: List[torch.Tensor] = []
            out_parts: List[torch.Tensor] = []
            all_rows: List[Dict[str, Any]] = []
            total_loss = 0.0
            total_n = 0
            for item in gathered:
                if item is None:
                    continue
                total_loss += float(item.get("loss_sum", 0.0))
                total_n += int(item.get("n_obs", 0))
                yp = item.get("y", None)
                op = item.get("out", None)
                if isinstance(yp, torch.Tensor) and int(yp.shape[0]) > 0:
                    y_parts.append(yp)
                if isinstance(op, torch.Tensor) and int(op.shape[0]) > 0:
                    out_parts.append(op)
                if return_rows:
                    all_rows.extend(list(item.get("rows", [])))
            y_cat = torch.cat(y_parts, dim=0)
            out_cat = torch.cat(out_parts, dim=0)
            if task_type == "classification":
                metrics = evaluate_classification(y_cat.long().view(-1), out_cat, num_classes=int(num_classes))
            else:
                metrics = evaluate_regression(y_cat.view(-1), out_cat.view(-1))
            metrics.update(
                _grouped_metrics_from_rows(
                    rows=all_rows,
                    task_type=task_type,
                    num_classes=num_classes,
                    eval_agg_cfg=eval_agg_cfg,
                )
            )
            result_obj = (total_loss / max(1, total_n), metrics, all_rows)

        payload: List[Optional[Tuple[float, Dict[str, float], List[Dict[str, Any]]]]] = [result_obj]
        dist.broadcast_object_list(payload, src=0)
        if payload[0] is None:
            raise RuntimeError("distributed evaluation result broadcast failed")
        out_result = payload[0]
        if not return_rows:
            return float(out_result[0]), dict(out_result[1]), []
        return float(out_result[0]), dict(out_result[1]), list(out_result[2])

    if task_type == "classification":
        metrics = evaluate_classification(y_local.long().view(-1), out_local, num_classes=int(num_classes))
    else:
        metrics = evaluate_regression(y_local.view(-1), out_local.view(-1))
    metrics.update(
        _grouped_metrics_from_rows(
            rows=rows,
            task_type=task_type,
            num_classes=num_classes,
            eval_agg_cfg=eval_agg_cfg,
        )
    )
    if not return_rows:
        return float(loss_sum / max(1, n_obs)), metrics, []
    return float(loss_sum / max(1, n_obs)), metrics, rows


def _reload_model_from_ckpt(model: nn.Module, base_state: Dict[str, torch.Tensor]) -> None:
    _load_state_dict(model, base_state)


def _rank0_plain_protocol(protocol: FeatureProtocolCond) -> FeatureProtocolCond:
    model = protocol.model
    if hasattr(model, "module"):
        return FeatureProtocolCond(
            model=model.module,
            diffusion=protocol.diffusion,
            cfg=protocol.cfg,
            device=protocol.device,
        )
    return protocol


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
            for c in range(int(probs.shape[1])):
                base[f"prob_{c}"] = float(probs[i, c].item())
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
    eval_agg_cfg: Dict[str, Any],
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
    metrics.update(
        _grouped_metrics_from_rows(
            rows=pred_rows,
            task_type=task_type,
            num_classes=num_classes,
            eval_agg_cfg=eval_agg_cfg,
        )
    )
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
    eval_agg_cfg = _resolve_eval_aggregation_cfg(cfg)
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
    extract_protocol = _rank0_plain_protocol(protocol)

    X_tr_np, y_tr_np, rows_tr_meta, cache_train_audit = _collect_or_load_cached_split(
        task_name=task_name,
        split_name="train",
        ds=train_ds,
        protocol=extract_protocol,
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
        protocol=extract_protocol,
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
        protocol=extract_protocol,
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
    eval_train_each_epoch = bool(train_cfg.get("eval_train_each_epoch", False))
    eval_test_each_epoch = bool(train_cfg.get("eval_test_each_epoch", False))
    history_flush_each_epoch = bool(train_cfg.get("history_flush_each_epoch", True))
    print_epoch_metrics = bool(train_cfg.get("print_epoch_metrics", True))
    show_tqdm = _show_tqdm(cfg)
    metric_name_norm = str(best_metric_name).strip().lower()
    higher_is_better = metric_name_norm not in {"loss", "val_loss", "mse", "mae"}
    best_val = -1.0e18 if higher_is_better else 1.0e18
    bad = 0
    history: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, Any]] = None
    start_epoch = 1

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    ensure_dir(output_dir)
    latest_path = os.path.join(output_dir, "latest.pt")
    resume_cfg = cfg.get("resume", {}) if isinstance(cfg.get("resume", {}), dict) else {}
    resume_path = str(resume_cfg.get("from", "") or resume_cfg.get("resume_from", "") or "").strip()
    if resume_path == "" and bool(resume_cfg.get("auto", True)) and os.path.isfile(latest_path):
        resume_path = latest_path
    if resume_path != "":
        ck = torch.load(os.path.abspath(resume_path), map_location=device)
        if ck.get("mode", "") == "linear_probe" and "head" in ck:
            head.load_state_dict(ck["head"], strict=True)
            if "optimizer" in ck:
                optimizer.load_state_dict(ck["optimizer"])
            if "scaler" in ck and ck["scaler"] is not None:
                scaler.load_state_dict(ck["scaler"])
            history = list(ck.get("history", []))
            best_state = ck.get("best_state", None)
            best_val = float(ck.get("best_val", best_val))
            bad = int(ck.get("bad_epochs", bad))
            start_epoch = int(ck.get("epoch", 0)) + 1
            print(f"[resume][{task_name}] linear_probe from {resume_path} start_epoch={start_epoch}")

    for ep in range(start_epoch, epochs + 1):
        head.train()
        loss_sum = 0.0
        n_obs = 0

        pbar = tqdm(train_loader, desc=f"{task_name} train_cache[{ep}/{epochs}]", ncols=120) if show_tqdm else train_loader
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
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(loss=f"{loss_sum/max(1, n_obs):.5f}")

        train_loss = loss_sum / max(1, n_obs)
        tr_eval_loss = None
        tr_eval_metrics = None
        if eval_train_each_epoch:
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
                eval_agg_cfg=eval_agg_cfg,
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
            eval_agg_cfg=eval_agg_cfg,
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
                eval_agg_cfg=eval_agg_cfg,
                desc=None,
            )

        rec = {
            "epoch": ep,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        }
        if tr_eval_loss is not None and tr_eval_metrics is not None:
            rec["train_eval_loss"] = float(tr_eval_loss)
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

        torch.save(
            {
                "task_name": task_name,
                "seed": int(cfg.get("seed", cfg.get("train", {}).get("seed", 0))),
                "task_type": task_type,
                "target_col": target_col,
                "num_classes": int(num_classes),
                "mode": "linear_probe",
                "epoch": int(ep),
                "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if use_amp else None,
                "best_state": best_state,
                "best_val": float(best_val),
                "bad_epochs": int(bad),
                "history": history,
                "config": cfg,
            },
            latest_path,
        )
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
        eval_agg_cfg=eval_agg_cfg,
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
        eval_agg_cfg=eval_agg_cfg,
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
        eval_agg_cfg=eval_agg_cfg,
        desc="eval_test",
    )
    _apply_eval_threshold_calibration(
        task_type=task_type,
        num_classes=num_classes,
        eval_agg_cfg=eval_agg_cfg,
        train_rows=tr_rows,
        val_rows=va_rows,
        test_rows=te_rows,
        train_metrics=tr_metrics,
        val_metrics=va_metrics,
        test_metrics=te_metrics,
    )

    ensure_dir(output_dir)
    _write_rows_csv(tr_rows, os.path.join(output_dir, "pred_train.csv"))
    _write_rows_csv(va_rows, os.path.join(output_dir, "pred_val.csv"))
    _write_rows_csv(te_rows, os.path.join(output_dir, "pred_test.csv"))

    capture_layers = resolve_capture_layers(int(getattr(_unwrap_module(model), "depth")), list(emb_cfg.get("capture_layers", [-1])))
    t_list = _parse_t_list(emb_cfg)
    noise_mode = str(emb_cfg.get("noise_mode", "per_subject"))
    pool_mode = str(emb_cfg.get("pool", "mean"))

    ckpt_path = os.path.join(output_dir, "best.pt")
    torch.save(
        {
            "task_name": task_name,
            "seed": int(cfg.get("seed", cfg.get("train", {}).get("seed", 0))),
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

    observed_train_num_classes = None
    observed_train_class_counts = None
    if task_type == "classification":
        y_tr_int = y_tr_np.astype(np.int64).reshape(-1)
        if y_tr_int.size > 0:
            uniq, cnt = np.unique(y_tr_int, return_counts=True)
            observed_train_num_classes = int(np.max(y_tr_int).item()) + 1
            observed_train_class_counts = {str(int(u)): int(c) for u, c in zip(uniq.tolist(), cnt.tolist())}

    summary = {
        "task_name": task_name,
        "seed": int(cfg.get("seed", cfg.get("train", {}).get("seed", 0))),
        "mode": "linear_probe",
        "task_type": task_type,
        "target_col": target_col,
        "num_classes": int(num_classes),
        "observed_train_num_classes": observed_train_num_classes,
        "observed_train_class_counts": observed_train_class_counts,
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
        "eval_aggregation": eval_agg_cfg,
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
    eval_agg_cfg = _resolve_eval_aggregation_cfg(cfg)

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
    run_seed = int(cfg.get("seed", cfg.get("train", {}).get("seed", 0)))

    labels, label_audit = _read_label_map(
        path=label_csv,
        subject_col=subject_col,
        target_col=target_col,
        task_type=task_type_hint,
        label_map=(label_cfg.get("label_map", None) if isinstance(label_cfg.get("label_map", None), dict) else None),
        duplicate_policy_cls=str(label_cfg.get("duplicate_policy_cls", "majority")),
        duplicate_policy_reg=str(label_cfg.get("duplicate_policy_reg", "mean")),
    )

    train_ds_base = _filter_base_dataset_for_task(train_ds_base, cfg, "train")
    val_ds_base = _filter_base_dataset_for_task(val_ds_base, cfg, "val")
    test_ds_base = _filter_base_dataset_for_task(test_ds_base, cfg, "test")

    target_norm = None
    if task_type_hint == "regression":
        vals = []
        seen_label_subjects: set[str] = set()
        for idx in range(len(train_ds_base)):
            sid = _norm_subject_id(_subject_id_from_base_ds(train_ds_base, idx))
            if sid in labels and sid not in seen_label_subjects:
                seen_label_subjects.add(sid)
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
    observed_train_num_classes = None
    observed_train_class_counts = None
    configured_num_classes = None
    if task_type == "classification":
        y_train_int = np.asarray(y_train_np, dtype=np.int64).reshape(-1)
        if y_train_int.size > 0:
            uniq, cnt = np.unique(y_train_int, return_counts=True)
            observed_train_num_classes = int(np.max(y_train_int).item()) + 1
            observed_train_class_counts = {str(int(u)): int(c) for u, c in zip(uniq.tolist(), cnt.tolist())}
        if "num_classes" in head_task_cfg:
            configured_num_classes = int(head_task_cfg.get("num_classes", observed_train_num_classes or num_classes))

    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg.get("runtime", {}), dict) else {}
    preflight_only = bool(runtime_cfg.get("preflight_only", cfg.get("preflight_only", False)))
    if dist_ctx.is_main:
        zmsg = "none" if target_norm is None else f"mean={target_norm[0]:.6g} std={target_norm[1]:.6g}"
        cond_status = _condition_enabled_summary(cfg)
        cond_counts = {
            "train": _condition_path_counts(train_ds),
            "val": _condition_path_counts(val_ds),
            "test": _condition_path_counts(test_ds),
        }
        print(
            f"[preflight][{task_name}] seed={run_seed} mode={mode} task_type={task_type} target={target_col} "
            f"train/val/test={len(train_ds)}/{len(val_ds)}/{len(test_ds)} label_zscore={zmsg}"
        )
        print(
            f"[preflight][{task_name}] conditions "
            f"fc enabled={cond_status['fc']['enabled']} required={cond_status['fc']['required']} "
            f"paths(train/val/test)={cond_counts['train']['fc_path']}/{cond_counts['val']['fc_path']}/{cond_counts['test']['fc_path']} "
            f"mri enabled={cond_status['mri']['enabled']} required={cond_status['mri']['required']} "
            f"paths(train/val/test)={cond_counts['train']['mri_path']}/{cond_counts['val']['mri_path']}/{cond_counts['test']['mri_path']}"
        )
        if task_type == "classification":
            print(
                f"[preflight][{task_name}] class_counts train={observed_train_class_counts} "
                f"configured_num_classes={configured_num_classes if configured_num_classes is not None else int(num_classes)} "
                f"observed_num_classes={observed_train_num_classes if observed_train_num_classes is not None else int(num_classes)}"
            )
            if (
                configured_num_classes is not None
                and observed_train_num_classes is not None
                and int(configured_num_classes) != int(observed_train_num_classes)
            ):
                print(
                    f"[warn][{task_name}] configured num_classes={int(configured_num_classes)} "
                    f"but observed train labels imply num_classes={int(observed_train_num_classes)}"
                )
        preflight = {
            "task_name": task_name,
            "seed": int(run_seed),
            "mode": mode,
            "task_type": task_type,
            "target_col": target_col,
            "num_classes": int(num_classes),
            "configured_num_classes": configured_num_classes,
            "observed_train_num_classes": observed_train_num_classes,
            "observed_train_class_counts": observed_train_class_counts,
            "label_audit": label_audit,
            "label_zscore": None if target_norm is None else {"mean": float(target_norm[0]), "std": float(target_norm[1])},
            "split_audit": {"train": train_ds.audit, "val": val_ds.audit, "test": test_ds.audit},
            "condition_status": cond_status,
            "condition_path_counts": cond_counts,
            "conditions": cfg.get("data", {}).get("conditions", {}),
            "embedding": cfg.get("embedding", {}),
            "eval_aggregation": eval_agg_cfg,
        }
        ensure_dir(output_dir)
        save_json(preflight, os.path.join(output_dir, "preflight.json"))
    if preflight_only:
        return {
            "task_name": task_name,
            "seed": int(run_seed),
            "mode": mode,
            "task_type": task_type,
            "target_col": target_col,
            "preflight_only": True,
            "configured_num_classes": configured_num_classes,
            "observed_train_num_classes": observed_train_num_classes,
            "observed_train_class_counts": observed_train_class_counts,
            "label_zscore": None if target_norm is None else {"mean": float(target_norm[0]), "std": float(target_norm[1])},
            "split_audit": {"train": train_ds.audit, "val": val_ds.audit, "test": test_ds.audit},
        }

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
    train_eval_sampler = _distributed_sampler(train_ds, dist_ctx=dist_ctx, shuffle=False)
    val_sampler = _distributed_sampler(val_ds, dist_ctx=dist_ctx, shuffle=False)
    test_sampler = _distributed_sampler(test_ds, dist_ctx=dist_ctx, shuffle=False)

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
        sampler=train_eval_sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_labeled,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_labeled,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
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
    eval_train_each_epoch = bool(train_cfg.get("eval_train_each_epoch", False))
    eval_test_each_epoch = bool(train_cfg.get("eval_test_each_epoch", False))
    history_flush_each_epoch = bool(train_cfg.get("history_flush_each_epoch", True))
    print_epoch_metrics = bool(train_cfg.get("print_epoch_metrics", True))
    show_tqdm = _show_tqdm(cfg)

    history: List[Dict[str, Any]] = []
    metric_name_norm = str(best_metric_name).strip().lower()
    higher_is_better = metric_name_norm not in {"loss", "val_loss", "mse", "mae"}
    best_val = -1.0e18 if higher_is_better else 1.0e18
    bad = 0
    best_state: Optional[Dict[str, Any]] = None
    start_epoch = 1

    latest_path = os.path.join(output_dir, "latest.pt")
    resume_cfg = cfg.get("resume", {}) if isinstance(cfg.get("resume", {}), dict) else {}
    resume_path = str(resume_cfg.get("from", "") or resume_cfg.get("resume_from", "") or "").strip()
    if resume_path == "" and bool(resume_cfg.get("auto", True)) and os.path.isfile(latest_path):
        resume_path = latest_path
    if resume_path != "":
        ck = torch.load(os.path.abspath(resume_path), map_location=device)
        if ck.get("mode", "") == mode:
            if model_trainable and ck.get("model", None) is not None:
                _load_state_dict(model, ck["model"])
            if "aggregator" in ck:
                _unwrap_module(aggregator).load_state_dict(ck["aggregator"], strict=True)
            if "head" in ck:
                _unwrap_module(head).load_state_dict(ck["head"], strict=True)
            if "optimizer" in ck:
                optimizer.load_state_dict(ck["optimizer"])
            if "scaler" in ck and ck["scaler"] is not None:
                scaler.load_state_dict(ck["scaler"])
            history = list(ck.get("history", []))
            best_state = ck.get("best_state", None)
            best_val = float(ck.get("best_val", best_val))
            bad = int(ck.get("bad_epochs", bad))
            start_epoch = int(ck.get("epoch", 0)) + 1
            if dist_ctx.is_main:
                print(f"[resume][{task_name}] {mode} from {resume_path} start_epoch={start_epoch}")

    for ep in range(start_epoch, epochs + 1):
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
        if dist_ctx.is_main and show_tqdm:
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
                    cond_cfg=cfg.get("data", {}).get("conditions", {}),
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

        tr_eval_loss = None
        tr_eval_metrics = None
        if eval_train_each_epoch:
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
                cond_cfg=cfg.get("data", {}).get("conditions", {}),
                eval_agg_cfg=eval_agg_cfg,
                desc=(f"{task_name} eval_train[{ep}/{epochs}]" if dist_ctx.is_main and show_tqdm else None),
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
            cond_cfg=cfg.get("data", {}).get("conditions", {}),
            eval_agg_cfg=eval_agg_cfg,
            desc=(f"{task_name} eval_val[{ep}/{epochs}]" if dist_ctx.is_main and show_tqdm else None),
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
                cond_cfg=cfg.get("data", {}).get("conditions", {}),
                eval_agg_cfg=eval_agg_cfg,
                desc=(f"{task_name} eval_test[{ep}/{epochs}]" if dist_ctx.is_main and show_tqdm else None),
                return_rows=False,
            )

        rec = {
            "epoch": ep,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        }
        if tr_eval_loss is not None and tr_eval_metrics is not None:
            rec["train_eval_loss"] = float(tr_eval_loss)
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

        if dist_ctx.is_main:
            torch.save(
                {
                    "task_name": task_name,
                    "seed": int(cfg.get("seed", cfg.get("train", {}).get("seed", 0))),
                    "task_type": task_type,
                    "target_col": target_col,
                    "num_classes": int(num_classes),
                    "mode": mode,
                    "epoch": int(ep),
                    "model": _state_dict_cpu(model) if model_trainable else None,
                    "aggregator": {k: v.detach().cpu() for k, v in _unwrap_module(aggregator).state_dict().items()},
                    "head": {k: v.detach().cpu() for k, v in _unwrap_module(head).state_dict().items()},
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if use_amp else None,
                    "best_state": best_state,
                    "best_val": float(best_val),
                    "bad_epochs": int(bad),
                    "history": history,
                    "config": cfg,
                },
                latest_path,
            )
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
        cond_cfg=cfg.get("data", {}).get("conditions", {}),
        eval_agg_cfg=eval_agg_cfg,
        desc=("eval_train" if dist_ctx.is_main and show_tqdm else None),
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
        cond_cfg=cfg.get("data", {}).get("conditions", {}),
        eval_agg_cfg=eval_agg_cfg,
        desc=("eval_val" if dist_ctx.is_main and show_tqdm else None),
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
        cond_cfg=cfg.get("data", {}).get("conditions", {}),
        eval_agg_cfg=eval_agg_cfg,
        desc=("eval_test" if dist_ctx.is_main and show_tqdm else None),
        return_rows=True,
    )
    _apply_eval_threshold_calibration(
        task_type=task_type,
        num_classes=num_classes,
        eval_agg_cfg=eval_agg_cfg,
        train_rows=tr_rows,
        val_rows=va_rows,
        test_rows=te_rows,
        train_metrics=tr_metrics,
        val_metrics=va_metrics,
        test_metrics=te_metrics,
    )

    ckpt_path = os.path.join(output_dir, "best.pt")
    if dist_ctx.is_main:
        ensure_dir(output_dir)
        _write_rows_csv(tr_rows, os.path.join(output_dir, "pred_train.csv"))
        _write_rows_csv(va_rows, os.path.join(output_dir, "pred_val.csv"))
        _write_rows_csv(te_rows, os.path.join(output_dir, "pred_test.csv"))

        torch.save(
            {
                "task_name": task_name,
                "seed": int(cfg.get("seed", cfg.get("train", {}).get("seed", 0))),
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
        "seed": int(cfg.get("seed", cfg.get("train", {}).get("seed", 0))),
        "mode": mode,
        "task_type": task_type,
        "target_col": target_col,
        "num_classes": int(num_classes),
        "observed_train_num_classes": observed_train_num_classes,
        "observed_train_class_counts": observed_train_class_counts,
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
        "eval_aggregation": eval_agg_cfg,
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
    cfg = _materialize_exp2_config(load_json(args.config))
    cfg["_config_path"] = os.path.abspath(args.config)

    root_seed_list, root_seed_list_explicit = _resolve_seed_list(cfg)
    seed = int(root_seed_list[0])
    cfg["seed"] = int(seed)
    set_seed(seed)

    gpu_ids = _resolve_gpu_ids(cfg)
    dist_ctx, device = _setup_dist_context(cfg, gpu_ids)
    cuda_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    logical_gpu_ids = _logical_gpu_ids_for_current_process(gpu_ids)

    ckpt_cfg = cfg.get("ckpt", cfg.get("checkpoints", {}))
    ckpt_path = str(
        ckpt_cfg.get("checkpoint", ckpt_cfg.get("cond_dit_ckpt", ckpt_cfg.get("stage1_ckpt", "")))
    ).strip()
    if ckpt_path == "":
        raise ValueError("ckpt.checkpoint (or cond_dit_ckpt) is required")
    ckpt_path = os.path.abspath(ckpt_path)

    _rank_print(dist_ctx, f"[load] checkpoint metadata start: {ckpt_path}")
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
    _rank_print(dist_ctx, "[load] checkpoint metadata done")

    schedule = normalize_schedule_type(cfg.get("diffusion", {}).get("schedule", "linear"))

    _rank_print(dist_ctx, "[data] building train/val/test splits start")
    train_ds, val_ds, test_ds, audits = build_splits_from_config(cfg)
    _rank_print(dist_ctx, "[data] building train/val/test splits done")
    model = _build_model_from_dataset(cfg, train_ds)

    diffusion = GaussianDiffusion(
        num_steps=int(cfg.get("diffusion", {}).get("num_steps", 1000)),
        beta_start=float(cfg.get("diffusion", {}).get("beta_start", 1.0e-4)),
        beta_end=float(cfg.get("diffusion", {}).get("beta_end", 2.0e-2)),
        schedule=schedule,
        cosine_s=float(cfg.get("diffusion", {}).get("cosine_s", 0.008)),
    ).to(device)

    _rank_print(dist_ctx, f"[load] checkpoint weights start: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model" not in ckpt:
        raise KeyError(f"checkpoint missing model key: {ckpt_path}")
    _load_state_dict(
        model,
        ckpt["model"],
        allow_condition_mismatch=bool(ckpt_cfg.get("allow_condition_mismatch", False)),
    )
    _rank_print(dist_ctx, "[load] checkpoint weights loaded into model")
    model = model.to(device)
    _rank_print(dist_ctx, f"[load] model moved to device={device}")

    use_ddp = bool(dist_ctx.enabled)
    use_data_parallel = bool((not use_ddp) and device.type == "cuda" and len(logical_gpu_ids) > 1)
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
        model = nn.DataParallel(
            model,
            device_ids=[int(g) for g in logical_gpu_ids],
            output_device=int(logical_gpu_ids[0]),
        ).to(device)
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
    out_root = os.path.abspath(str(out_cfg.get("out_root", "exp2/outputs/downstream")))
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
        "seed": int(seed),
        "seed_list": [int(x) for x in root_seed_list],
        "seed_list_explicit": bool(root_seed_list_explicit),
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
            active_gpu_ids = [int(g) for g in logical_gpu_ids]
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
        print(f"[audit] logical_gpu_ids={logical_gpu_ids}")
        print(f"[audit] active_gpu_ids={active_gpu_ids}")
        print(f"[audit] parallel_mode={parallel_mode}")
        print(f"[audit] seed_list={root_seed_list} seed_list_explicit={root_seed_list_explicit}")
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
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg.get("runtime", {}), dict) else {}
    run_mode_filter = str(os.environ.get("DOWNSTREAM_RUN_MODE", runtime_cfg.get("mode_filter", "all"))).strip().lower()
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

    task_seed_plan: List[Tuple[str, List[int], bool]] = []
    for i, task_raw in enumerate(tasks, start=1):
        name_i = _safe_name(task_raw.get("name", f"task_{i:02d}"))
        seeds_i, seeds_explicit_i = _resolve_seed_list(cfg, task_raw)
        task_seed_plan.append((name_i, seeds_i, seeds_explicit_i))

    total_task_runs = int(sum(len(seeds_i) for _, seeds_i, _ in task_seed_plan))
    task_seed_sets = [{int(s) for s in seeds_i} for _, seeds_i, _ in task_seed_plan]
    seed_order: List[int] = []
    seen_seed_order: set[int] = set()
    for s in root_seed_list:
        si = int(s)
        if si in seen_seed_order:
            continue
        if any(si in seed_set for seed_set in task_seed_sets):
            seed_order.append(si)
            seen_seed_order.add(si)
    for _, seeds_i, _ in task_seed_plan:
        for s in seeds_i:
            si = int(s)
            if si not in seen_seed_order:
                seed_order.append(si)
                seen_seed_order.add(si)

    if dist_ctx.is_main:
        print(f"[audit] task_runs={total_task_runs} seed_plan={[(n, s) for n, s, _ in task_seed_plan]}")
        print(f"[audit] execution_order=seed_major seed_order={seed_order}")

    results: List[Dict[str, Any]] = []

    base_model_state = _state_dict_cpu(model)
    task_run_idx = 0
    stop_all = False

    prepared_tasks: List[Dict[str, Any]] = []
    for i, task_raw in enumerate(tasks, start=1):
        task_name, task_cfg_base = _build_task_cfg(cfg, task_raw, i)
        seed_list, seed_list_explicit = _resolve_seed_list(cfg, task_raw)
        task_out_cfg = task_cfg_base.get("_task_output_dir", None)
        if task_out_cfg is None:
            task_base_out_dir = os.path.join(run_out, task_name)
        else:
            if os.path.isabs(str(task_out_cfg)):
                task_base_out_dir = str(task_out_cfg)
            else:
                task_base_out_dir = os.path.join(run_out, str(task_out_cfg))

        lower_bound_cfg = _resolve_task_lower_bound(task_raw, task_cfg_base)
        lower_bound_search_cfg = _resolve_task_lower_bound_search(cfg, task_raw) if lower_bound_cfg is not None else {"enabled": False, "candidates": [], "keep_top_k": 0}
        required_passes = int(lower_bound_cfg.get("required_passes", 3) or 3) if lower_bound_cfg is not None else 0
        existing_good_records: List[Dict[str, Any]] = []
        if lower_bound_cfg is not None:
            out_cfg_i = cfg.get("output", {}) if isinstance(cfg.get("output", {}), dict) else {}
            mode_i = str(task_cfg_base.get("mode", task_raw.get("mode", cfg.get("mode", "linear_probe")))).strip().lower()
            good_key = "good_ft_root" if mode_i == "full_finetune" else "good_lp_root"
            good_default = os.path.join(out_root, "good_ft" if mode_i == "full_finetune" else "good_lp")
            good_root_i = os.path.abspath(str(out_cfg_i.get(good_key, good_default)))
            existing_good_records = _load_existing_good_passes(
                good_root=good_root_i,
                task_name=task_name,
                lower_bound_cfg=lower_bound_cfg,
                required_passes=required_passes,
            )
        use_seed_subdir = bool(len(seed_list) > 1 or lower_bound_cfg is not None)
        prepared_tasks.append(
            {
                "idx": int(i),
                "task_name": task_name,
                "task_cfg_base": task_cfg_base,
                "task_base_out_dir": task_base_out_dir,
                "seed_list": [int(s) for s in seed_list],
                "seed_pos": {int(s): int(j) for j, s in enumerate(seed_list, start=1)},
                "use_seed_subdir": bool(use_seed_subdir),
                "lower_bound": lower_bound_cfg,
                "lower_bound_search": lower_bound_search_cfg,
                "lower_bound_attempts": 0,
                "lower_bound_passes": int(len(existing_good_records)),
                "lower_bound_top": existing_good_records,
                "satisfied": bool(lower_bound_cfg is not None and int(len(existing_good_records)) >= required_passes),
                "existing_good_passes": existing_good_records,
            }
        )

    if dist_ctx.is_main:
        print("=" * 100)
        print("[progress] task queue")
        for p in prepared_tasks:
            lbp = p.get("lower_bound", None)
            if lbp is None:
                print(f"[progress] - {p['task_name']}: seeds={p['seed_list']}")
            else:
                search_cfg_i = p.get("lower_bound_search", {})
                cand_n = len(search_cfg_i.get("candidates", [])) if isinstance(search_cfg_i, dict) else 0
                print(
                    f"[progress] - {p['task_name']}: target {lbp['metric']}>={float(lbp['value']):.6g} "
                    f"passes={int(p.get('lower_bound_passes', 0))}/{int(lbp.get('required_passes', 3) or 3)} "
                    f"existing_good={len(p.get('existing_good_passes', []))} sweep_candidates={cand_n}"
                )
        print("=" * 100)

    executed_seed_order: List[int] = []
    seed_cursor = 0
    next_auto_seed = int(max(seed_order) + 1) if len(seed_order) > 0 else int(seed) + 1

    while not stop_all:
        if seed_cursor >= len(seed_order):
            has_pending_lower_bound = any(
                p.get("lower_bound", None) is not None
                and int(p.get("lower_bound_passes", 0)) < int(p.get("lower_bound", {}).get("required_passes", 3) or 3)
                for p in prepared_tasks
            )
            if not has_pending_lower_bound:
                break
            while int(next_auto_seed) in seen_seed_order:
                next_auto_seed += 1
            seed_order.append(int(next_auto_seed))
            seen_seed_order.add(int(next_auto_seed))
            next_auto_seed += 1

        run_seed = int(seed_order[seed_cursor])
        seed_idx = int(seed_cursor + 1)
        seed_cursor += 1
        executed_seed_order.append(int(run_seed))

        if stop_all:
            break
        for prepared in prepared_tasks:
            if stop_all:
                break
            seed_list = [int(s) for s in prepared["seed_list"]]
            if bool(prepared.get("satisfied", False)):
                continue
            lower_bound_cfg = prepared.get("lower_bound", None)
            if lower_bound_cfg is None and int(run_seed) not in set(seed_list):
                continue
            if lower_bound_cfg is not None:
                required_passes = int(lower_bound_cfg.get("required_passes", 3) or 3)
                if int(prepared.get("lower_bound_passes", 0)) >= required_passes:
                    prepared["satisfied"] = True
                    continue
            task_run_idx += 1
            i = int(prepared["idx"])
            task_name = str(prepared["task_name"])
            task_cfg_base = prepared["task_cfg_base"]
            task_base_out_dir = str(prepared["task_base_out_dir"])
            use_seed_subdir = bool(prepared["use_seed_subdir"])
            seed_j = int(prepared["seed_pos"].get(int(run_seed), int(prepared.get("lower_bound_attempts", 0)) + 1))
            task_cfg = copy.deepcopy(task_cfg_base)
            task_cfg["seed"] = int(run_seed)
            if isinstance(task_cfg.get("train", None), dict):
                task_cfg["train"]["seed"] = int(run_seed)
            else:
                task_cfg["train"] = {"seed": int(run_seed)}

            search_candidate = None
            search_cfg = prepared.get("lower_bound_search", {})
            search_candidates = search_cfg.get("candidates", []) if isinstance(search_cfg, dict) else []
            if lower_bound_cfg is not None and isinstance(search_candidates, list) and len(search_candidates) > 0:
                cand_idx = int(prepared.get("lower_bound_attempts", 0)) % len(search_candidates)
                search_candidate = copy.deepcopy(search_candidates[cand_idx])
                _deep_update(task_cfg, copy.deepcopy(search_candidate.get("overrides", {})))
                task_cfg["_lower_bound_search"] = {
                    "candidate_index": int(cand_idx),
                    "candidate_name": str(search_candidate.get("name", f"sweep_{cand_idx:02d}")),
                }

            if use_seed_subdir:
                if search_candidate is not None:
                    task_out_dir = os.path.join(
                        task_base_out_dir,
                        f"seed_{int(run_seed):04d}__{_safe_name(str(search_candidate.get('name', 'sweep')))}",
                    )
                else:
                    task_out_dir = os.path.join(task_base_out_dir, f"seed_{int(run_seed):04d}")
            else:
                task_out_dir = task_base_out_dir

            set_seed(int(run_seed))

            if dist_ctx.is_main:
                print("-" * 100)
                print(
                    f"[seed {seed_idx}/{len(seed_order)} task {i}/{len(tasks)} seed_run {seed_j}/{len(seed_list)} "
                    f"run {task_run_idx}/{total_task_runs}] "
                    f"name={task_name} seed={int(run_seed)}"
                )
                if lower_bound_cfg is not None:
                    print(
                        f"[task {i}/{len(tasks)}] lower_bound metric={lower_bound_cfg['metric']} "
                        f"threshold={float(lower_bound_cfg['value']):.6g} "
                        f"passes={int(prepared.get('lower_bound_passes', 0))}/{int(lower_bound_cfg.get('required_passes', 3) or 3)} "
                        f"attempt={int(prepared.get('lower_bound_attempts', 0)) + 1}"
                    )
                    if search_candidate is not None:
                        print(
                            f"[task {i}/{len(tasks)}] lower_bound_search candidate={search_candidate.get('name', '')}"
                        )
                print(f"[task {i}/{len(tasks)}] output={task_out_dir}")
                print("-" * 100)

            try:
                task_train_ds_base = train_ds
                task_val_ds_base = val_ds
                task_test_ds_base = test_ds
                if _task_overrides_target_manifests(task_cfg, cfg):
                    manifest_sig = _target_manifest_signature(task_cfg)
                    cached_sig = prepared.get("_task_dataset_cache_signature", None)
                    cached_splits = prepared.get("_task_dataset_cache_splits", None)
                    if cached_sig != manifest_sig or cached_splits is None:
                        if dist_ctx.is_main:
                            print(f"[data][{task_name}] task-specific target manifests detected; rebuilding splits")
                        rebuilt_train_ds, rebuilt_val_ds, rebuilt_test_ds, _ = build_splits_from_config(task_cfg)
                        prepared["_task_dataset_cache_signature"] = manifest_sig
                        prepared["_task_dataset_cache_splits"] = (rebuilt_train_ds, rebuilt_val_ds, rebuilt_test_ds)
                        cached_splits = prepared["_task_dataset_cache_splits"]
                    task_train_ds_base, task_val_ds_base, task_test_ds_base = cached_splits

                summary = _run_one_task(
                    task_name=task_name,
                    cfg=task_cfg,
                    model=model,
                    base_model_state=base_model_state,
                    protocol=protocol,
                    train_ds_base=task_train_ds_base,
                    val_ds_base=task_val_ds_base,
                    test_ds_base=task_test_ds_base,
                    output_dir=task_out_dir,
                    device=device,
                    dist_ctx=dist_ctx,
                )
                if lower_bound_cfg is not None:
                    prepared["lower_bound_attempts"] = int(prepared.get("lower_bound_attempts", 0)) + 1
                lower_bound_result = _evaluate_lower_bound(summary, lower_bound_cfg)
                if lower_bound_result is not None:
                    top_records = prepared.get("lower_bound_top", [])
                    if not isinstance(top_records, list):
                        top_records = []
                    top_records.append(
                        {
                            "seed": int(run_seed),
                            "value": float(lower_bound_result["value"]),
                            "passed": bool(lower_bound_result["passed"]),
                            "candidate": ("" if search_candidate is None else str(search_candidate.get("name", ""))),
                            "output_dir": task_out_dir,
                        }
                    )
                    prefers_smaller = _lower_bound_prefers_smaller(str(lower_bound_result.get("metric", "")))
                    missing_value = float("inf") if prefers_smaller else -float("inf")
                    prepared["lower_bound_top"] = sorted(
                        top_records,
                        key=lambda r: float(r.get("value", missing_value)),
                        reverse=not prefers_smaller,
                    )[:3]
                top_archived_dir = ""
                if lower_bound_result is not None and dist_ctx.is_main:
                    out_cfg = cfg.get("output", {}) if isinstance(cfg.get("output", {}), dict) else {}
                    task_mode_for_archive = str(task_cfg.get("mode", "")).strip().lower()
                    if task_mode_for_archive in {"linear_probe", "full_finetune"}:
                        top_key = "top_ft_root" if task_mode_for_archive == "full_finetune" else "top_lp_root"
                        top_default = os.path.join(out_root, "top_ft" if task_mode_for_archive == "full_finetune" else "top_lp")
                        top_root = os.path.abspath(str(out_cfg.get(top_key, top_default)))
                        top_archived_dir = _archive_good_output(
                            src_dir=task_out_dir,
                            good_root=top_root,
                            run_tag=os.path.basename(run_out),
                            task_name=task_name,
                            seed=int(run_seed),
                            metric=str(lower_bound_result["metric"]),
                            value=float(lower_bound_result["value"]),
                            pass_rank=0,
                            candidate_name=("" if search_candidate is None else str(search_candidate.get("name", ""))),
                            keep_top_k=3,
                        )
                        if top_archived_dir != "":
                            print(f"[lower_bound] archived top3 {task_mode_for_archive} output={top_archived_dir}")
                if lower_bound_result is not None and bool(lower_bound_result.get("passed", False)):
                    prepared["lower_bound_passes"] = int(prepared.get("lower_bound_passes", 0)) + 1
                    if int(prepared.get("lower_bound_passes", 0)) >= int(lower_bound_cfg.get("required_passes", 3) or 3):
                        prepared["satisfied"] = True
                archived_dir = ""
                if lower_bound_result is not None and bool(lower_bound_result.get("passed", False)) and dist_ctx.is_main:
                    out_cfg = cfg.get("output", {}) if isinstance(cfg.get("output", {}), dict) else {}
                    task_mode_for_archive = str(task_cfg.get("mode", "")).strip().lower()
                    if task_mode_for_archive in {"linear_probe", "full_finetune"}:
                        good_key = "good_ft_root" if task_mode_for_archive == "full_finetune" else "good_lp_root"
                        good_default = os.path.join(out_root, "good_ft" if task_mode_for_archive == "full_finetune" else "good_lp")
                        good_root = os.path.abspath(str(out_cfg.get(good_key, good_default)))
                        archived_dir = _archive_good_output(
                            src_dir=task_out_dir,
                            good_root=good_root,
                            run_tag=os.path.basename(run_out),
                            task_name=task_name,
                            seed=int(run_seed),
                            metric=str(lower_bound_result["metric"]),
                            value=float(lower_bound_result["value"]),
                            pass_rank=int(prepared.get("lower_bound_passes", 0)),
                            candidate_name=("" if search_candidate is None else str(search_candidate.get("name", ""))),
                            keep_top_k=int(lower_bound_cfg.get("required_passes", 3) or 3),
                        )
                        if archived_dir != "":
                            print(f"[lower_bound] archived passing {task_mode_for_archive} output={archived_dir}")
                output_removed = False
                if lower_bound_result is not None and not bool(lower_bound_result.get("passed", False)):
                    if dist_ctx.is_main:
                        output_removed = _remove_task_output_dir(task_out_dir)
                        if output_removed:
                            print(f"[lower_bound] removed below-bound output={task_out_dir}")
                result = {
                    "task_name": task_name,
                    "seed": int(run_seed),
                    "ok": True,
                    "output_dir": task_out_dir,
                    "summary": summary,
                }
                if lower_bound_result is not None:
                    result["lower_bound"] = lower_bound_result
                    result["output_removed"] = bool(output_removed)
                    if archived_dir != "":
                        result["archived_dir"] = archived_dir
                    if search_candidate is not None:
                        result["lower_bound_search"] = {
                            "candidate_name": str(search_candidate.get("name", "")),
                            "overrides": search_candidate.get("overrides", {}),
                        }
                    if dist_ctx.is_main:
                        status = "passed" if bool(lower_bound_result["passed"]) else "below"
                        print(
                            f"[lower_bound] task={task_name} seed={int(run_seed)} "
                            f"metric={lower_bound_result['metric']} value={float(lower_bound_result['value']):.6g} "
                            f"threshold={float(lower_bound_result['threshold']):.6g} status={status} "
                            f"passes={int(prepared.get('lower_bound_passes', 0))}/{int(lower_bound_cfg.get('required_passes', 3) or 3)}"
                        )
                        done_tasks = sum(1 for p in prepared_tasks if bool(p.get("satisfied", False)))
                        active_status = []
                        for p in prepared_tasks:
                            lbp = p.get("lower_bound", None)
                            if lbp is None:
                                continue
                            active_status.append(
                                f"{p['task_name']}={int(p.get('lower_bound_passes', 0))}/{int(lbp.get('required_passes', 3) or 3)}"
                            )
                        print(
                            f"[progress] tasks_satisfied={done_tasks}/{len(prepared_tasks)} "
                            f"attempts={len(results) + 1} current={task_name} "
                            f"{_format_lower_bound_top3(prepared.get('lower_bound_top', []))}"
                        )
                        if len(active_status) > 0:
                            print("[progress] pass_counts " + " ".join(active_status))
                        if bool(lower_bound_result["passed"]):
                            if bool(prepared.get("satisfied", False)):
                                print(f"[lower_bound] task={task_name} satisfied; collected required passing seeds")
                            else:
                                print(f"[lower_bound] task={task_name} passed; continuing until required passes are collected")
                results.append(result)
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                if dist_ctx.is_main:
                    removed = _remove_dir_if_no_summary(task_out_dir)
                    if removed:
                        print(f"[cleanup] removed incomplete output={task_out_dir}")
                results.append({"task_name": task_name, "seed": int(run_seed), "ok": False, "output_dir": task_out_dir, "error": err_msg})
                if dist_ctx.enabled:
                    try:
                        ensure_dir(run_out)
                        with open(os.path.join(run_out, f"rank{dist_ctx.rank}_fatal.txt"), "w", encoding="utf-8") as f:
                            f.write(f"task={task_name}\n")
                            f.write(f"seed={int(run_seed)}\n")
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
                            "num_task_runs": int(total_task_runs),
                            "actual_num_task_runs": int(len(results)),
                            "execution_order": "seed_major",
                            "seed_order": [int(x) for x in seed_order],
                            "executed_seed_order": [int(x) for x in executed_seed_order],
                            "seed_plan": [{"task_name": n, "seed_list": [int(x) for x in s]} for n, s, _ in task_seed_plan],
                            "task_status": [
                                {
                                    "task_name": str(p["task_name"]),
                                    "lower_bound": p.get("lower_bound", None),
                                    "lower_bound_search": p.get("lower_bound_search", None),
                                    "lower_bound_attempts": int(p.get("lower_bound_attempts", 0)),
                                    "lower_bound_passes": int(p.get("lower_bound_passes", 0)),
                                    "lower_bound_top": p.get("lower_bound_top", []),
                                    "satisfied": bool(p.get("satisfied", False)),
                                }
                                for p in prepared_tasks
                            ],
                            "results": results,
                            "aborted": True,
                        }
                        save_json(partial, os.path.join(run_out, "multi_task_summary.json"))

                    _abort_dist_process_group(
                        dist_ctx,
                        reason=f"task={task_name} seed={int(run_seed)} failed on rank={dist_ctx.rank}: {err_msg}",
                    )
                    raise
                if bool(cfg.get("stop_on_error", True)):
                    stop_all = True
                    break

    final = {
        "config": os.path.abspath(args.config),
        "checkpoint": ckpt_path,
        "task_filter": run_mode_filter if run_mode_filter != "" else "all",
        "run_output_dir": run_out,
        "run_context": os.path.join(run_out, "run_context.json"),
        "num_tasks": len(tasks),
        "num_task_runs": int(total_task_runs),
        "actual_num_task_runs": int(len(results)),
        "execution_order": "seed_major",
        "seed_order": [int(x) for x in seed_order],
        "executed_seed_order": [int(x) for x in executed_seed_order],
        "seed_plan": [{"task_name": n, "seed_list": [int(x) for x in s]} for n, s, _ in task_seed_plan],
        "task_status": [
            {
                "task_name": str(p["task_name"]),
                "lower_bound": p.get("lower_bound", None),
                "lower_bound_search": p.get("lower_bound_search", None),
                "lower_bound_attempts": int(p.get("lower_bound_attempts", 0)),
                "lower_bound_passes": int(p.get("lower_bound_passes", 0)),
                "lower_bound_top": p.get("lower_bound_top", []),
                "satisfied": bool(p.get("satisfied", False)),
            }
            for p in prepared_tasks
        ],
        "results": results,
    }
    if dist_ctx.is_main:
        ok_cnt = sum(1 for r in results if bool(r.get("ok", False)))
        out_cfg = cfg.get("output", {}) if isinstance(cfg.get("output", {}), dict) else {}
        keep_failed_runs = bool(out_cfg.get("keep_failed_runs", False))

        print("=" * 100)
        print(f"[done] completed_runs={ok_cnt} planned_initial_runs={total_task_runs}")
        if ok_cnt > 0 or keep_failed_runs:
            save_json(final, os.path.join(run_out, "multi_task_summary.json"))
            print(f"[done] summary={os.path.join(run_out, 'multi_task_summary.json')}")
        else:
            removed = _remove_dir_if_no_summary(run_out)
            if removed:
                print(f"[cleanup] removed run with no completed result={run_out}")
            else:
                print(f"[done] no completed result; kept run_output_dir={run_out}")

    if dist_ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
