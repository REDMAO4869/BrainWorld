#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable


EXP1_DOWNSTREAM_DIR = Path(__file__).resolve().parent
DDIT_ROOT = EXP1_DOWNSTREAM_DIR.parent
if str(DDIT_ROOT) not in sys.path:
    sys.path.insert(0, str(DDIT_ROOT))

from brainworld.dit.diffusion import GaussianDiffusion, normalize_schedule_type  # noqa: E402
from brainworld.dit.downstream_metrics import evaluate_classification, evaluate_regression  # noqa: E402
from brainworld.dit.downstream_protocol import FeatureProtocolCond, FeatureProtocolCondConfig, resolve_capture_layers  # noqa: E402
from brainworld.dit.model import ConditionalLatentDiT  # noqa: E402

Json = Dict[str, Any]


@dataclass(frozen=True)
class BlockSpec:
    dataset: str
    subject: str
    source: str
    start: int
    latent_path: str
    latent_key: str
    fc_path: str
    mri_path: str
    sequence_id: str


@dataclass(frozen=True)
class SubjectSample:
    task_name: str
    dataset: str
    split: str
    source: str
    horizon: int
    subject: str
    label: float
    blocks: Tuple[BlockSpec, ...]


@dataclass(frozen=True)
class BlockSample:
    task_name: str
    dataset: str
    split: str
    source: str
    horizon: int
    subject: str
    label: float
    block: BlockSpec


@dataclass(frozen=True)
class ExpSubject:
    subject: str
    sequence_id: str
    rows_by_target: Dict[int, Dict[str, str]]


@dataclass
class ManifestBundle:
    samples: Dict[str, Dict[str, Dict[str, List[SubjectSample]]]]
    audits: Dict[str, Any]


@dataclass(frozen=True)
class DistEnv:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Exp1 GT/GE augmentation downstream")
    p.add_argument("--config", required=True, help="Path to JSON config")
    p.add_argument("--preflight-only", action="store_true", help="Only build manifests and audits")
    p.add_argument("--extract-only", action="store_true", help="Only extract/cache DDIT block features")
    p.add_argument("--train-only", action="store_true", help="Train from existing feature caches")
    p.add_argument("--tasks", default="", help="Comma-separated task names to run")
    p.add_argument(
        "--max-subjects-per-split",
        type=int,
        default=0,
        help="Optional smoke cap applied per task/split/source after manifest construction",
    )
    p.add_argument("--ddp", action="store_true", help="Enable distributed sweep sharding under torchrun")
    p.add_argument("--ddp-backend", default="nccl", help="torch.distributed backend (default: nccl)")
    return p.parse_args()


def load_json(path: str | Path) -> Json:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, cls=JsonEncoder), encoding="utf-8")
    os.replace(tmp, p)


class JsonEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def ensure_dir(path: str | Path) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def safe_name(x: str) -> str:
    out = []
    for c in str(x):
        if c.isalnum() or c in {"-", "_"}:
            out.append(c)
        else:
            out.append("_")
    s = "".join(out).strip("_")
    return s or "item"


def norm_subject(s: str) -> str:
    return str(s).strip().lower()


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def stable_int(*parts: object, base: int = 0) -> int:
    text = "|".join(str(x) for x in parts)
    return (int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16) + int(base)) % (2**31 - 1)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def resolve_device(cfg: Json) -> torch.device:
    raw = str(cfg.get("device", "auto")).strip().lower()
    if raw == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def resolve_lp_protocol(cfg: Json) -> str:
    raw = str(cfg.get("lp_protocol", "")).strip().lower()
    if raw in {"", "legacy", "subject_concat"}:
        return "subject_concat"
    if raw in {"aligned", "exp2_cached_sample", "sample_level"}:
        return "exp2_cached_sample"
    if raw in {"aligned_subject_mean", "subject_mean", "subject_mean_pool", "exp2_cached_subject_mean"}:
        return "aligned_subject_mean"
    raise ValueError(f"unsupported lp_protocol: {raw}")


def _aligned_root_name(name: str, prefix: str) -> str:
    n = str(name).strip()
    if n.startswith(prefix):
        return f"{prefix}_aligned_lp{n[len(prefix):]}"
    return f"{n}_aligned_lp"


def apply_protocol_path_overrides(cfg: Json) -> None:
    protocol = resolve_lp_protocol(cfg)
    if protocol == "subject_concat":
        return
    paths = cfg.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("cfg.paths must be a dict")
    out_root = Path(str(paths["output_root"]))
    report_root = Path(str(paths["report_root"]))
    if protocol == "exp2_cached_sample":
        protocol_out = str(paths.get("aligned_lp_output_root", "")).strip()
        protocol_report = str(paths.get("aligned_lp_report_root", "")).strip()
        if protocol_out == "":
            protocol_out = str(out_root.parent / _aligned_root_name(out_root.name, "outputs"))
        if protocol_report == "":
            protocol_report = str(report_root.parent / _aligned_root_name(report_root.name, "reports"))
    else:
        protocol_out = str(paths.get("aligned_subject_mean_lp_output_root", "")).strip()
        protocol_report = str(paths.get("aligned_subject_mean_lp_report_root", "")).strip()
        if protocol_out == "":
            protocol_out = str(out_root.parent / _aligned_root_name(out_root.name, "outputs").replace("_aligned_lp", "_aligned_subject_mean_lp"))
        if protocol_report == "":
            protocol_report = str(report_root.parent / _aligned_root_name(report_root.name, "reports").replace("_aligned_lp", "_aligned_subject_mean_lp"))
    paths["output_root"] = protocol_out
    paths["report_root"] = protocol_report
    cfg["paths"] = paths


def effective_training_cfg(cfg: Json) -> Json:
    train_cfg = dict(cfg.get("training", {}))
    protocol = resolve_lp_protocol(cfg)
    if protocol in {"exp2_cached_sample", "aligned_subject_mean"}:
        train_cfg.update(
            {
                "epochs": 30,
                "patience": 10,
                "batch_size": 3,
                "lr": 1.0e-4,
                "weight_decay": 0.0,
                "standardize_features": True,
            }
        )
        aligned_cfg = cfg.get("aligned_lp", {})
        if isinstance(aligned_cfg, dict):
            extra_train = aligned_cfg.get("training", {})
            if isinstance(extra_train, dict):
                train_cfg.update(extra_train)
        if protocol == "aligned_subject_mean":
            subject_mean_cfg = cfg.get("aligned_subject_mean_lp", {})
            if isinstance(subject_mean_cfg, dict):
                extra_train = subject_mean_cfg.get("training", {})
                if isinstance(extra_train, dict):
                    train_cfg.update(extra_train)
    return train_cfg


def show_progress(cfg: Optional[Json] = None) -> bool:
    if int(os.environ.get("RANK", "0")) != 0:
        return False
    if not isinstance(cfg, dict):
        return True
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg.get("runtime", {}), dict) else {}
    return bool(runtime_cfg.get("show_tqdm", True))


def init_dist_env(args: argparse.Namespace) -> DistEnv:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = bool(args.ddp or world_size > 1)
    if not enabled:
        return DistEnv(enabled=False, rank=0, world_size=1, local_rank=0)
    if torch.cuda.is_available():
        # Explicitly pin each rank to its local GPU before NCCL init to avoid
        # "Guessing device_id" warnings and potential barrier hangs.
        torch.cuda.set_device(local_rank)
    if world_size <= 1:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not dist.is_initialized():
        backend = str(args.ddp_backend).strip().lower() or "nccl"
        if backend == "nccl" and not torch.cuda.is_available():
            backend = "gloo"
        dist.init_process_group(backend=backend)
    return DistEnv(enabled=True, rank=rank, world_size=world_size, local_rank=local_rank)


def dist_barrier_if_needed(denv: DistEnv) -> None:
    if denv.enabled and dist.is_initialized():
        dist.barrier()


def select_npz_key(keys: Sequence[str], preferred: str) -> str:
    if preferred and preferred in keys:
        return preferred
    for key in ("pred_latent", "mu", "z", "arr_0", "arr", "data", "x"):
        if key in keys:
            return key
    if not keys:
        raise ValueError("empty npz")
    return str(keys[0])


def load_array(path: str, preferred_key: str = "", dtype: str = "float32") -> np.ndarray:
    p = Path(path)
    dt = np.float16 if str(dtype).lower() == "float16" else np.float32
    if p.suffix.lower() == ".npy":
        return np.asarray(np.load(str(p), allow_pickle=False), dtype=dt)
    if p.suffix.lower() == ".npz":
        with np.load(str(p), allow_pickle=False) as d:
            key = select_npz_key(list(d.keys()), preferred_key)
            return np.asarray(d[key], dtype=dt)
    raise ValueError(f"unsupported array path: {path}")


def read_split_subjects(path: str | Path) -> List[str]:
    subjects: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return subjects
    start = 0
    head0 = rows[0][0].strip().lower() if rows[0] else ""
    if head0 in {"subject", "sub", "subject_id", "id"}:
        start = 1
    seen = set()
    for row in rows[start:]:
        if not row:
            continue
        sid = str(row[0]).strip()
        if sid and sid not in seen:
            seen.add(sid)
            subjects.append(sid)
    return subjects


def norm_col(x: str) -> str:
    return str(x).replace("﻿", "").replace("`", "").strip().lower()


def resolve_csv_col(fieldnames: Sequence[str], wanted: str) -> str:
    norm2raw = {norm_col(c): c for c in fieldnames}
    got = norm2raw.get(norm_col(wanted))
    if got is None:
        raise ValueError(f"column {wanted!r} not found in {list(fieldnames)}")
    return got


def read_label_map(path: str | Path, subject_col: str, target_col: str, task_type: str) -> Tuple[Dict[str, float], Json]:
    values: Dict[str, List[float]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"label csv has no header: {path}")
        sub_col = resolve_csv_col(reader.fieldnames, subject_col)
        tgt_col = resolve_csv_col(reader.fieldnames, target_col)
        for row in reader:
            sid = norm_subject(row.get(sub_col, ""))
            raw = str(row.get(tgt_col, "")).strip()
            if not sid or raw == "":
                continue
            try:
                values[sid].append(float(raw))
            except Exception:
                continue

    out: Dict[str, float] = {}
    duplicate_subjects = 0
    for sid, arr in values.items():
        if len(arr) > 1:
            duplicate_subjects += 1
        if task_type == "classification":
            vals = [int(round(v)) for v in arr]
            out[sid] = float(Counter(vals).most_common(1)[0][0])
        else:
            out[sid] = float(np.mean(arr))
    return out, {"label_subjects": len(out), "duplicate_subjects": duplicate_subjects}


def regression_dataset_cfg(regression_cfg: Json) -> Json:
    if isinstance(regression_cfg.get("dataset", None), dict):
        return regression_cfg["dataset"]
    if isinstance(regression_cfg.get("hcp", None), dict):
        out = dict(regression_cfg["hcp"])
        out.setdefault("name", "HCP")
        out.setdefault("mode", "target_pairs")
        return out
    return {}


def infer_chunk_start(row: Dict[str, str], ds_cfg: Json, block_size: int) -> int:
    import re

    pattern = str(ds_cfg.get("chunk_index_regex", r"part[_-]?(\d+)")).strip() or r"part[_-]?(\d+)"
    regex = re.compile(pattern, flags=re.IGNORECASE)
    keys = [
        str(ds_cfg.get("chunk_path_key", "voxel_data_path")).strip() or "voxel_data_path",
        "voxel_data_path",
        "vae_latent_path",
        "target_voxel_data_path",
        "target_latent_path",
    ]
    for key in keys:
        raw = str(row.get(key, "")).strip()
        if not raw:
            continue
        match = regex.search(raw)
        if match:
            return int(match.group(1)) * int(block_size)
    raise ValueError(f"could not infer chunk start from row")


def insert_subject_subdir_no_stat(path_str: str, subject: str, ds_cfg: Json) -> str:
    raw = str(path_str).strip()
    if not raw or not bool(ds_cfg.get("insert_subject_subdir_if_missing", False)):
        return raw
    p = Path(raw)
    subject_clean = str(subject).strip()
    fmt = str(ds_cfg.get("subject_format", "raw")).strip().lower() or "raw"
    if fmt == "zfill6":
        subject_dir = subject_clean.zfill(6)
    elif fmt == "sub-zfill6":
        subject_dir = f"sub-{subject_clean.zfill(6)}"
    else:
        subject_dir = subject_clean
    if p.parent.name == subject_dir:
        return raw
    return str(p.parent / subject_dir / p.name)


def load_subject_records(regression_cfg: Json, horizons: Sequence[int], *, show_tqdm: bool = True) -> List[ExpSubject]:
    ds_cfg = regression_dataset_cfg(regression_cfg)
    dataset_name = str(ds_cfg.get("name", ds_cfg.get("dataset_name", "HCP"))).strip() or "HCP"
    dataset_mode = str(ds_cfg.get("mode", "auto")).strip().lower() or "auto"
    csv_paths = [str(x) for x in ds_cfg.get("csv_paths", [])]
    if not csv_paths:
        raise ValueError("dataset/hcp csv_paths must be non-empty")

    sequence_id_filter = str(ds_cfg.get("sequence_id", "REST1_LR_hp2000_clean")).strip()
    subject_key = str(ds_cfg.get("subject_key", "Subject")).strip() or "Subject"
    sequence_key = str(ds_cfg.get("sequence_key", "sequence_id")).strip() or "sequence_id"
    pair_direction_key = str(ds_cfg.get("pair_direction_key", "pair_direction")).strip() or "pair_direction"
    pair_direction_value = str(ds_cfg.get("pair_direction_value", "next")).strip().lower()
    target_chunk_key = str(ds_cfg.get("target_chunk_key", "target_chunk_id")).strip() or "target_chunk_id"
    target_chunk_scale = int(ds_cfg.get("target_chunk_scale", 1) or 1)
    target_chunk_offset = int(ds_cfg.get("target_chunk_offset", 0) or 0)
    group_by_subject_sequence = bool(ds_cfg.get("group_by_subject_sequence", False))
    fc_embedding_key = str(ds_cfg.get("fc_embedding_key", "fc_embedding_path")).strip() or "fc_embedding_path"
    mri_embedding_key = str(ds_cfg.get("mri_embedding_key", "MRI_embedding_path")).strip() or "MRI_embedding_path"
    target_voxel_key = str(ds_cfg.get("target_voxel_key", "target_voxel_data_path")).strip() or "target_voxel_data_path"
    target_latent_key = str(ds_cfg.get("target_latent_key", "target_latent_path")).strip() or "target_latent_path"

    block_size = int(regression_cfg.get("regression", {}).get("block_size", 40))
    start_frame = int(regression_cfg.get("regression", {}).get("start_frame", 160))
    condition_start = int(regression_cfg.get("regression", {}).get("condition_frame_start", 120))
    max_blocks = max(int(math.ceil(int(h) / float(block_size))) for h in horizons)
    needed_targets = [start_frame + i * block_size for i in range(max_blocks)]

    by_subject: Dict[str, Dict[str, Any]] = {}
    detected_mode: Optional[str] = None
    csv_iter: Iterable[str] = csv_paths
    if show_tqdm:
        csv_iter = tqdm(csv_paths, desc=f"{dataset_name} subject csvs", ncols=120, leave=False)
    for csv_path in csv_iter:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for raw_row in reader:
                row = {str(k): str(v) for k, v in raw_row.items()}
                if sequence_id_filter and str(row.get(sequence_key, "")).strip() != sequence_id_filter:
                    continue
                if pair_direction_value and str(row.get(pair_direction_key, pair_direction_value)).strip().lower() != pair_direction_value:
                    continue
                subject = str(row.get(subject_key, "")).strip()
                if not subject:
                    continue
                seq_value = str(row.get(sequence_key, "")).strip()
                subject_group_key = f"{subject}__{seq_value}" if group_by_subject_sequence and seq_value else subject
                raw_target = str(row.get(target_chunk_key, "")).strip()
                use_target_pairs = dataset_mode == "target_pairs" or (dataset_mode == "auto" and bool(raw_target))
                if use_target_pairs:
                    detected_mode = detected_mode or "target_pairs"
                    try:
                        target_start = int(raw_target) * target_chunk_scale + target_chunk_offset
                    except Exception:
                        continue
                    if target_start not in needed_targets:
                        continue
                    entry = by_subject.setdefault(subject_group_key, {"rows": {}})
                    entry["rows"].setdefault(target_start, row)
                else:
                    detected_mode = detected_mode or "sequential_chunks"
                    try:
                        chunk_start = infer_chunk_start(row, ds_cfg, block_size)
                    except Exception:
                        continue
                    entry = by_subject.setdefault(subject_group_key, {"rows": {}})
                    entry["rows"].setdefault(chunk_start, row)

    out: List[ExpSubject] = []
    subject_items = sorted(by_subject.items())
    subject_iter: Iterable[Tuple[str, Dict[str, Any]]] = subject_items
    if show_tqdm:
        subject_iter = tqdm(subject_items, desc=f"{dataset_name} subjects", ncols=120, leave=False)
    for subject, entry in subject_iter:
        rows = entry["rows"]
        if detected_mode == "sequential_chunks":
            required_starts = [condition_start] + needed_targets
            if any(t not in rows for t in required_starts):
                continue
            init_condition = rows[condition_start]
            normalized: Dict[int, Dict[str, str]] = {}
            for target_start in needed_targets:
                target_row = dict(rows[target_start])
                prev_row = rows.get(target_start - block_size, init_condition)
                target_row[target_chunk_key] = str(target_start)
                target_row[target_voxel_key] = insert_subject_subdir_no_stat(
                    str(target_row.get(target_voxel_key, target_row.get("voxel_data_path", ""))),
                    subject,
                    ds_cfg,
                )
                target_row[target_latent_key] = insert_subject_subdir_no_stat(
                    str(target_row.get(target_latent_key, target_row.get("vae_latent_path", ""))),
                    subject,
                    ds_cfg,
                )
                target_row[fc_embedding_key] = insert_subject_subdir_no_stat(
                    str(prev_row.get(fc_embedding_key, init_condition.get(fc_embedding_key, ""))),
                    subject,
                    ds_cfg,
                )
                target_row[mri_embedding_key] = str(target_row.get(mri_embedding_key, init_condition.get(mri_embedding_key, "")))
                target_row[sequence_key] = str(target_row.get(sequence_key, sequence_id_filter or dataset_name)).strip() or dataset_name
                normalized[target_start] = target_row
            rows = normalized

        if any(t not in rows for t in needed_targets):
            continue
        init = rows[start_frame]
        out.append(
            ExpSubject(
                subject=subject,
                sequence_id=str(init.get(sequence_key, sequence_id_filter or dataset_name)).strip() or dataset_name,
                rows_by_target={int(k): dict(v) for k, v in rows.items()},
            )
        )
    return out


def generated_block_dir(run_root: Path, subject: str, start: int) -> Path:
    return run_root / "subjects" / safe_name(subject) / "shared_trajectory_0400" / "blocks" / f"{start:04d}_{start + 40:04d}"


def _row_path(row: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        v = str(row.get(key, "")).strip()
        if v:
            return v
    return ""


def path_is_file(path: str, cache: Dict[str, bool]) -> bool:
    key = str(path)
    if key not in cache:
        cache[key] = Path(key).is_file()
    return bool(cache[key])


def build_block_specs(
    *,
    dataset: str,
    subject_record: Any,
    source: str,
    horizon: int,
    run_root: Path,
    verify_gt_paths: bool,
    verify_ge_paths: bool,
    verify_optional_mri_paths: bool,
    exists_cache: Dict[str, bool],
) -> Tuple[List[BlockSpec], List[str]]:
    starts = [160 + i * 40 for i in range(int(math.ceil(int(horizon) / 40.0)))]
    blocks: List[BlockSpec] = []
    missing: List[str] = []
    rows = subject_record.rows_by_target
    for start in starts:
        row = rows.get(int(start), {})
        if source == "gt":
            latent_path = _row_path(row, "target_latent_path", "vae_latent_path")
            latent_key = "mu"
            fc_path = _row_path(row, "fc_embedding_path")
        else:
            block_dir = generated_block_dir(run_root, subject_record.subject, int(start))
            latent_path = str(block_dir / "pred_latent.npz")
            latent_key = "pred_latent"
            if int(start) == 160:
                fc_path = _row_path(rows.get(160, {}), "fc_embedding_path")
            else:
                fc_path = str(generated_block_dir(run_root, subject_record.subject, int(start) - 40) / "fc_embedding.npy")
        mri_path = _row_path(row, "MRI_embedding_path") or _row_path(rows.get(160, {}), "MRI_embedding_path")

        verify_source = bool(verify_gt_paths if source == "gt" else verify_ge_paths)
        if (not latent_path) or (verify_source and not path_is_file(latent_path, exists_cache)):
            missing.append(f"latent:{start}:{latent_path}")
        if (not fc_path) or (verify_source and not path_is_file(fc_path, exists_cache)):
            missing.append(f"fc:{start}:{fc_path}")
        if verify_optional_mri_paths and mri_path and not path_is_file(mri_path, exists_cache):
            # MRI is optional for the checkpoint, but record it for the audit.
            missing.append(f"mri_optional:{start}:{mri_path}")

        blocks.append(
            BlockSpec(
                dataset=str(dataset),
                subject=str(subject_record.subject),
                source=str(source),
                start=int(start),
                latent_path=latent_path,
                latent_key=latent_key,
                fc_path=fc_path,
                mri_path=mri_path,
                sequence_id=str(subject_record.sequence_id),
            )
        )
    hard_missing = [x for x in missing if not x.startswith("mri_optional:")]
    return blocks, hard_missing


def selected_task_names(cfg: Json, raw: str) -> List[str]:
    all_names = [str(t["name"]) for t in cfg.get("tasks", [])]
    if str(raw).strip() == "":
        return all_names
    wanted = [x.strip() for x in str(raw).split(",") if x.strip()]
    bad = [x for x in wanted if x not in set(all_names)]
    if bad:
        raise ValueError(f"unknown task(s): {bad}; available={all_names}")
    return wanted


def cap_samples(samples: List[SubjectSample], cap: int) -> List[SubjectSample]:
    if int(cap) <= 0:
        return samples
    return samples[: int(cap)]


def build_manifest(cfg: Json, task_names: Sequence[str], max_subjects_per_split: int = 0) -> ManifestBundle:
    paths = cfg["paths"]
    split_root = Path(paths["split_root"])
    label_root = Path(paths["label_root"])
    exp1_output_root = Path(paths["exp1_output_root"])

    tasks_by_name = {str(t["name"]): dict(t) for t in cfg.get("tasks", [])}
    samples: Dict[str, Dict[str, Dict[str, List[SubjectSample]]]] = {}
    audits: Dict[str, Any] = {
        "note": "Existing split CSVs are used as source of truth; observed proportions are about 7:1:2.",
        "datasets": {},
        "tasks": {},
    }

    dataset_cache: Dict[str, Tuple[Dict[str, Any], Dict[str, float], Json]] = {}
    exists_cache: Dict[str, bool] = {}
    manifest_cfg = cfg.get("manifest", {}) if isinstance(cfg.get("manifest", {}), dict) else {}
    verify_gt_paths = bool(manifest_cfg.get("verify_gt_paths", False))
    verify_ge_paths = bool(manifest_cfg.get("verify_ge_paths", True))
    verify_optional_mri_paths = bool(manifest_cfg.get("verify_optional_mri_paths", False))
    use_tqdm = show_progress(cfg)

    task_iter: Iterable[str] = task_names
    if use_tqdm:
        task_iter = tqdm(task_names, desc="manifest tasks", ncols=120, leave=False)
    for task_name in task_iter:
        task = tasks_by_name[task_name]
        dataset = str(task["dataset"])
        horizon = int(task["horizon"])
        run_cfg = cfg["runs"][dataset]

        if dataset not in dataset_cache:
            if use_tqdm:
                print(f"[manifest] dataset={dataset} loading subject records", flush=True)
            regression_cfg = load_json(run_cfg["regression_config"])
            subjects = load_subject_records(regression_cfg, [200, 400], show_tqdm=use_tqdm)
            subject_index = {norm_subject(s.subject): s for s in subjects}
            label_path = label_root / str(run_cfg["label_csv"])
            label_map, label_audit = read_label_map(
                label_path,
                str(run_cfg.get("label_subject_col", "Subject")),
                str(run_cfg["target_col"]),
                str(run_cfg["task_type"]),
            )
            dataset_cache[dataset] = (subject_index, label_map, label_audit)
            split_counts = {}
            for split in ("train", "val", "test"):
                split_subjects = read_split_subjects(split_root / dataset / f"{split}.csv")
                split_counts[split] = len(split_subjects)
            audits["datasets"][dataset] = {
                "regression_config": str(run_cfg["regression_config"]),
                "generated_run": str(run_cfg["generated_run"]),
                "loaded_generation_subjects": int(len(subject_index)),
                "label_audit": label_audit,
                "split_counts": split_counts,
            }

        subject_index, label_map, _ = dataset_cache[dataset]
        run_root = exp1_output_root / str(run_cfg["generated_run"])
        task_samples: Dict[str, Dict[str, List[SubjectSample]]] = {
            "gt": {"train": [], "val": [], "test": []},
            "ge": {"train": [], "val": [], "test": []},
        }
        task_audit: Dict[str, Any] = {
            "dataset": dataset,
            "horizon": horizon,
            "source": {"gt": {}, "ge": {}},
            "missing_examples": [],
        }

        for split in ("train", "val", "test"):
            split_subjects = read_split_subjects(split_root / dataset / f"{split}.csv")
            split_requested = len(split_subjects)
            for source in ("gt", "ge"):
                missing_label = 0
                missing_subject = 0
                missing_latent = 0
                usable = 0
                subject_iter: Iterable[str] = split_subjects
                if use_tqdm:
                    subject_iter = tqdm(
                        split_subjects,
                        desc=f"{task_name} {split} {source}",
                        ncols=120,
                        leave=False,
                    )
                for raw_sid in subject_iter:
                    sid = norm_subject(raw_sid)
                    y = label_map.get(sid)
                    if y is None:
                        missing_label += 1
                        continue
                    rec = subject_index.get(sid)
                    if rec is None:
                        missing_subject += 1
                        continue
                    blocks, missing = build_block_specs(
                        dataset=dataset,
                        subject_record=rec,
                        source=source,
                        horizon=horizon,
                        run_root=run_root,
                        verify_gt_paths=verify_gt_paths,
                        verify_ge_paths=verify_ge_paths,
                        verify_optional_mri_paths=verify_optional_mri_paths,
                        exists_cache=exists_cache,
                    )
                    if missing:
                        missing_latent += 1
                        if len(task_audit["missing_examples"]) < 20:
                            task_audit["missing_examples"].append(
                                {"split": split, "source": source, "subject": raw_sid, "missing": missing[:4]}
                            )
                        continue
                    task_samples[source][split].append(
                        SubjectSample(
                            task_name=task_name,
                            dataset=dataset,
                            split=split,
                            source=source,
                            horizon=horizon,
                            subject=str(rec.subject),
                            label=float(y),
                            blocks=tuple(blocks),
                        )
                    )
                    usable += 1
                task_audit["source"][source][split] = {
                    "requested_subjects": int(split_requested),
                    "usable_subjects": int(usable),
                    "missing_label": int(missing_label),
                    "missing_generation_subject": int(missing_subject),
                    "missing_required_latent_or_fc": int(missing_latent),
                }
                task_samples[source][split] = cap_samples(task_samples[source][split], max_subjects_per_split)
                if max_subjects_per_split > 0:
                    task_audit["source"][source][split]["after_smoke_cap"] = len(task_samples[source][split])

        samples[task_name] = task_samples
        audits["tasks"][task_name] = task_audit

    return ManifestBundle(samples=samples, audits=audits)


def print_preflight(bundle: ManifestBundle) -> None:
    print("=" * 96)
    print("[preflight] Exp1 downstream manifest")
    print("[preflight] using existing split CSVs as source of truth (observed about 7:1:2)")
    for task_name, task_audit in bundle.audits["tasks"].items():
        print("-" * 96)
        print(f"[task] {task_name} dataset={task_audit['dataset']} horizon={task_audit['horizon']}")
        for source in ("gt", "ge"):
            parts = []
            for split in ("train", "val", "test"):
                a = task_audit["source"][source][split]
                parts.append(
                    f"{split}: usable={a['usable_subjects']}/{a['requested_subjects']} "
                    f"missing_label={a['missing_label']} missing_subject={a['missing_generation_subject']} "
                    f"missing_latent={a['missing_required_latent_or_fc']}"
                )
            print(f"[{source}] " + " | ".join(parts))
    print("=" * 96)


def condition_enabled(cfg: Json, name: str) -> bool:
    cond = cfg.get("data", {}).get("conditions", {})
    spec = cond.get(name, {}) if isinstance(cond, dict) else {}
    return isinstance(spec, dict) and bool(spec.get("enabled", False))


def load_checkpoint_config(cfg: Json) -> Json:
    ckpt_path = str(cfg["ckpt"]["checkpoint"])
    ckpt = torch.load(ckpt_path, map_location="cpu")
    out = dict(cfg)
    model_cfg = ckpt.get("model_config", {})
    if isinstance(model_cfg, dict) and bool(cfg.get("ckpt", {}).get("auto_load_model_config", True)):
        for key in ("model", "diffusion", "conditioning", "diversity", "loss"):
            if key in model_cfg:
                out[key] = model_cfg[key]
        ckpt_data = model_cfg.get("data", {})
        if isinstance(ckpt_data, dict):
            out.setdefault("data", {})
            out["data"].setdefault("conditions", {})
            for key, value in ckpt_data.get("conditions", {}).items():
                out["data"]["conditions"].setdefault(key, value)
    out["_checkpoint_obj"] = ckpt
    return out


def is_condition_encoder_key(key: str) -> bool:
    return (
        key.startswith("fc_encoder.")
        or key.startswith("mri_encoder.")
        or key.startswith("meta_encoder.")
        or key.startswith("video_encoder.")
        or key.startswith("audio_encoder.")
    )


def load_state_dict(model: nn.Module, state: Dict[str, torch.Tensor], allow_condition_mismatch: bool) -> None:
    if not allow_condition_mismatch:
        model.load_state_dict(state, strict=True)
        return
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    ckpt_keys = set(state.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    bad_missing = [k for k in missing if not is_condition_encoder_key(k)]
    bad_unexpected = [k for k in unexpected if not is_condition_encoder_key(k)]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "checkpoint/model mismatch outside condition encoders: "
            f"missing={bad_missing[:12]} unexpected={bad_unexpected[:12]}"
        )
    filtered = {k: v for k, v in state.items() if k in model_keys}
    model.load_state_dict(filtered, strict=False)


class FeatureCache:
    def __init__(self, cfg: Json) -> None:
        self.raw_cfg = cfg
        self.cfg = cfg
        self.device = resolve_device(cfg)
        self.cache_root = Path(cfg["paths"]["feature_cache_root"])
        self.embedding_cfg = cfg.get("embedding", {})
        self.t_list = [int(x) for x in self.embedding_cfg.get("t_list", [100])]
        self.cache_dtype = str(self.embedding_cfg.get("cache_dtype", "float16")).lower()
        self.show_tqdm = bool(cfg.get("runtime", {}).get("show_tqdm", False))
        self._model: Optional[nn.Module] = None
        self._protocol: Optional[FeatureProtocolCond] = None
        self._target_shape = tuple(int(v) for v in self.cfg.get("model", {}).get("target_shape", [40, 16, 10, 12, 10]))
        cond_shapes = self.cfg.get("model", {}).get("condition_shapes", {})
        self._fc_shape = tuple(int(v) for v in cond_shapes.get("fc", [800]))
        self._mri_shape = tuple(int(v) for v in cond_shapes.get("mri", [768]))

    def cache_signature(self, block: BlockSpec) -> Json:
        return {
            "checkpoint": os.path.abspath(str(self.raw_cfg["ckpt"]["checkpoint"])),
            "dataset": block.dataset,
            "subject": block.subject,
            "source": block.source,
            "start": int(block.start),
            "latent_path": os.path.abspath(block.latent_path),
            "latent_key": block.latent_key,
            "fc_path": os.path.abspath(block.fc_path),
            "mri_path": os.path.abspath(block.mri_path) if block.mri_path else "",
            "t_list": self.t_list,
            "capture_layers": list(self.embedding_cfg.get("capture_layers", [-1])),
            "noise_mode": str(self.embedding_cfg.get("noise_mode", "per_sample_index")),
            "noise_seed": int(self.embedding_cfg.get("noise_seed", 0)),
        }

    def cache_paths(self, block: BlockSpec) -> Tuple[Path, Path]:
        sig = self.cache_signature(block)
        tag = short_hash(json.dumps(sig, sort_keys=True), 16)
        base = (
            self.cache_root
            / safe_name(block.dataset)
            / safe_name(block.source)
            / safe_name(block.subject)
            / f"start_{int(block.start):04d}__{tag}"
        )
        return base.with_suffix(".npz"), base.with_suffix(".meta.json")

    def feature_exists(self, block: BlockSpec) -> bool:
        npz_path, meta_path = self.cache_paths(block)
        if not npz_path.is_file() or not meta_path.is_file():
            return False
        try:
            meta = load_json(meta_path)
            return meta.get("signature") == self.cache_signature(block)
        except Exception:
            return False

    def load_feature(self, block: BlockSpec) -> np.ndarray:
        npz_path, meta_path = self.cache_paths(block)
        if not npz_path.is_file():
            raise FileNotFoundError(f"missing block feature cache: {npz_path}")
        meta = load_json(meta_path)
        if meta.get("signature") != self.cache_signature(block):
            raise RuntimeError(f"stale block feature cache: {npz_path}")
        with np.load(npz_path, allow_pickle=False) as d:
            return np.asarray(d["feature"], dtype=np.float32)

    def _ensure_model(self) -> FeatureProtocolCond:
        if self._protocol is not None:
            return self._protocol

        self.cfg = load_checkpoint_config(self.raw_cfg)
        self._target_shape = tuple(int(v) for v in self.cfg.get("model", {}).get("target_shape", [40, 16, 10, 12, 10]))
        cond_shapes = self.cfg.get("model", {}).get("condition_shapes", {})
        self._fc_shape = tuple(int(v) for v in cond_shapes.get("fc", [800]))
        self._mri_shape = tuple(int(v) for v in cond_shapes.get("mri", [768]))

        torch.backends.cuda.matmul.allow_tf32 = True
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        mcfg = self.cfg.get("model", {})
        condition_shapes = {
            "fc": self._fc_shape if condition_enabled(self.cfg, "fc") else None,
            "mri": self._mri_shape if condition_enabled(self.cfg, "mri") else None,
            "metadata": None,
        }
        model = ConditionalLatentDiT(
            target_shape=self._target_shape,
            patch_size=mcfg.get("patch_size", [2, 4, 2]),
            hidden_dim=int(mcfg.get("hidden_dim", 512)),
            depth=int(mcfg.get("depth", 16)),
            num_heads=int(mcfg.get("num_heads", 8)),
            mlp_ratio=float(mcfg.get("mlp_ratio", 4.0)),
            dropout=float(mcfg.get("dropout", 0.1)),
            condition_shapes=condition_shapes,
            condition_cfg=self.cfg.get("data", {}).get("conditions", {}),
            diversity_cfg=self.cfg.get("diversity", {}),
            max_time_steps=int(self.cfg.get("diffusion", {}).get("num_steps", 1000)),
        )
        ckpt = self.cfg["_checkpoint_obj"]
        load_state_dict(
            model,
            ckpt["model"],
            allow_condition_mismatch=bool(self.raw_cfg.get("ckpt", {}).get("allow_condition_mismatch", False)),
        )
        model = model.to(self.device).eval()

        dcfg = self.cfg.get("diffusion", {})
        diffusion = GaussianDiffusion(
            num_steps=int(dcfg.get("num_steps", 1000)),
            beta_start=float(dcfg.get("beta_start", 1.0e-4)),
            beta_end=float(dcfg.get("beta_end", 2.0e-2)),
            schedule=normalize_schedule_type(dcfg.get("schedule", "linear")),
            cosine_s=float(dcfg.get("cosine_s", 0.008)),
        ).to(self.device)
        layers = resolve_capture_layers(int(getattr(model, "depth")), list(self.embedding_cfg.get("capture_layers", [-1])))
        protocol_cfg = FeatureProtocolCondConfig(
            timestep=int(self.t_list[0] if self.t_list else 100),
            capture_layers=[int(x) for x in layers],
            noise_mode=str(self.embedding_cfg.get("noise_mode", "per_sample_index")),
            noise_seed=int(self.embedding_cfg.get("noise_seed", 0)),
        )
        self._model = model
        self._protocol = FeatureProtocolCond(model=model, diffusion=diffusion, cfg=protocol_cfg, device=self.device)
        print(f"[extract] loaded DDIT checkpoint on {self.device}; block feature layers={layers} t_list={self.t_list}", flush=True)
        return self._protocol

    def _load_block_batch(self, blocks: Sequence[BlockSpec]) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]], List[str], List[int]]:
        latents = []
        fc_list = []
        mri_list = []
        has_fc = []
        has_mri = []
        subjects = []
        indices = []
        for block in blocks:
            x = load_array(block.latent_path, block.latent_key, dtype="float32")
            if tuple(int(v) for v in x.shape) != self._target_shape:
                raise ValueError(f"target latent shape mismatch for {block.latent_path}: got {x.shape}, expected {self._target_shape}")
            latents.append(torch.from_numpy(np.asarray(x, dtype=np.float32)))

            if block.fc_path and Path(block.fc_path).is_file():
                fc = load_array(block.fc_path, dtype="float32")
                has_fc.append(1.0)
            else:
                fc = np.zeros(self._fc_shape, dtype=np.float32)
                has_fc.append(0.0)
            fc_list.append(torch.from_numpy(np.asarray(fc, dtype=np.float32).reshape(self._fc_shape)))

            if block.mri_path and Path(block.mri_path).is_file():
                mri = load_array(block.mri_path, dtype="float32")
                has_mri.append(1.0)
            else:
                mri = np.zeros(self._mri_shape, dtype=np.float32)
                has_mri.append(0.0)
            mri_list.append(torch.from_numpy(np.asarray(mri, dtype=np.float32).reshape(self._mri_shape)))

            subjects.append(str(block.subject))
            indices.append(stable_int(block.dataset, block.source, block.subject, block.start))

        x0 = torch.stack(latents, dim=0).to(self.device, dtype=torch.float32)
        cond_inputs: Dict[str, Optional[torch.Tensor]] = {
            "fc_cond": torch.stack(fc_list, dim=0).to(self.device, dtype=torch.float32),
            "has_fc": torch.tensor(has_fc, device=self.device, dtype=torch.float32),
            "mri_cond": torch.stack(mri_list, dim=0).to(self.device, dtype=torch.float32),
            "has_mri": torch.tensor(has_mri, device=self.device, dtype=torch.float32),
            "meta_cond": None,
            "has_meta": None,
        }
        return x0, cond_inputs, subjects, indices

    @torch.inference_mode()
    def extract_missing(self, blocks: Sequence[BlockSpec]) -> Json:
        unique: Dict[str, BlockSpec] = {}
        for block in blocks:
            sig = json.dumps(self.cache_signature(block), sort_keys=True)
            unique[sig] = block
        todo = [b for b in unique.values() if not self.feature_exists(b)]
        audit = {"requested_blocks": len(blocks), "unique_blocks": len(unique), "cache_hits": len(unique) - len(todo), "extracted": len(todo)}
        if not todo:
            return audit

        protocol = self._ensure_model()
        bs = int(self.embedding_cfg.get("extract_batch_size", 32))
        iterator: Iterable[int] = range(0, len(todo), bs)
        if self.show_tqdm:
            iterator = tqdm(iterator, desc="extract block features", ncols=120)
        for start_idx in iterator:
            batch_blocks = todo[start_idx : start_idx + bs]
            x0, cond_inputs, subjects, indices = self._load_block_batch(batch_blocks)
            direction_id = torch.ones((len(batch_blocks),), device=self.device, dtype=torch.long)
            pooled: List[torch.Tensor] = []
            for t in self.t_list:
                out = protocol.tokens_from_batch(
                    x0,
                    direction_id=direction_id,
                    cond_inputs=cond_inputs,
                    subjects=subjects,
                    sample_indices=indices,
                    enable_grad=False,
                    timestep=int(t),
                )
                pooled.extend([tok.mean(dim=1) for tok in out.tokens_list])
            feat = torch.cat(pooled, dim=1).detach().cpu().numpy().astype(np.float32)
            for i, block in enumerate(batch_blocks):
                npz_path, meta_path = self.cache_paths(block)
                ensure_dir(npz_path.parent)
                arr = feat[i].astype(np.float32 if self.cache_dtype == "float32" else np.float16)
                np.savez_compressed(npz_path, feature=arr)
                save_json(
                    {
                        "signature": self.cache_signature(block),
                        "feature_dim": int(feat.shape[1]),
                        "saved_dtype": str(arr.dtype),
                        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    },
                    meta_path,
                )
        return audit

    def sample_feature(self, sample: SubjectSample) -> np.ndarray:
        feats = [self.load_feature(block) for block in sample.blocks]
        return np.concatenate(feats, axis=0).astype(np.float32, copy=False)

    def sample_feature_mean(self, sample: SubjectSample) -> np.ndarray:
        feats = [self.load_feature(block) for block in sample.blocks]
        return np.stack(feats, axis=0).mean(axis=0).astype(np.float32, copy=False)


def iter_task_blocks(bundle: ManifestBundle, task_names: Sequence[str]) -> List[BlockSpec]:
    blocks: List[BlockSpec] = []
    seen = set()
    for task_name in task_names:
        for source in ("gt", "ge"):
            for split in ("train", "val", "test"):
                for sample in bundle.samples[task_name][source][split]:
                    for block in sample.blocks:
                        key = (block.dataset, block.source, block.subject, block.start, block.latent_path, block.fc_path)
                        if key not in seen:
                            seen.add(key)
                            blocks.append(block)
    return blocks


def feature_matrix(
    cache: FeatureCache,
    samples: Sequence[SubjectSample],
    *,
    show_tqdm: bool = False,
    desc: str = "build feature matrix",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X = []
    y = []
    subjects = []
    iterator: Iterable[SubjectSample] = samples
    if show_tqdm:
        iterator = tqdm(samples, desc=desc, ncols=120, leave=False)
    for sample in iterator:
        X.append(cache.sample_feature(sample))
        y.append(float(sample.label))
        subjects.append(str(sample.subject))
    if not X:
        raise RuntimeError("empty feature matrix")
    return np.stack(X, axis=0).astype(np.float32), np.asarray(y, dtype=np.float32), subjects


def subject_mean_feature_matrix(
    cache: FeatureCache,
    samples: Sequence[SubjectSample],
    *,
    show_tqdm: bool = False,
    desc: str = "build subject-mean feature matrix",
) -> Tuple[np.ndarray, np.ndarray, List[Json]]:
    X = []
    y = []
    rows_meta: List[Json] = []
    iterator: Iterable[SubjectSample] = samples
    if show_tqdm:
        iterator = tqdm(samples, desc=desc, ncols=120, leave=False)
    for sample in iterator:
        X.append(cache.sample_feature_mean(sample))
        y.append(float(sample.label))
        rows_meta.append(
            {
                "subject": str(sample.subject),
                "source": str(sample.source),
                "dataset": str(sample.dataset),
                "num_blocks": int(len(sample.blocks)),
            }
        )
    if not X:
        raise RuntimeError("empty subject-mean feature matrix")
    return np.stack(X, axis=0).astype(np.float32), np.asarray(y, dtype=np.float32), rows_meta


def expand_samples_to_blocks(samples: Sequence[SubjectSample]) -> List[BlockSample]:
    out: List[BlockSample] = []
    for sample in samples:
        for block in sample.blocks:
            out.append(
                BlockSample(
                    task_name=str(sample.task_name),
                    dataset=str(sample.dataset),
                    split=str(sample.split),
                    source=str(sample.source),
                    horizon=int(sample.horizon),
                    subject=str(sample.subject),
                    label=float(sample.label),
                    block=block,
                )
            )
    return out


def block_feature_matrix(
    cache: FeatureCache,
    samples: Sequence[BlockSample],
    *,
    show_tqdm: bool = False,
    desc: str = "build block feature matrix",
) -> Tuple[np.ndarray, np.ndarray, List[Json]]:
    X = []
    y = []
    rows_meta: List[Json] = []
    iterator: Iterable[BlockSample] = samples
    if show_tqdm:
        iterator = tqdm(samples, desc=desc, ncols=120, leave=False)
    for sample in iterator:
        X.append(cache.load_feature(sample.block))
        y.append(float(sample.label))
        rows_meta.append(
            {
                "subject": str(sample.subject),
                "source": str(sample.source),
                "start": int(sample.block.start),
                "dataset": str(sample.dataset),
                "sequence_id": str(sample.block.sequence_id),
            }
        )
    if not X:
        raise RuntimeError("empty block feature matrix")
    return np.stack(X, axis=0).astype(np.float32), np.asarray(y, dtype=np.float32), rows_meta


def stratified_order(samples: Sequence[SubjectSample], task_cfg: Json, seed: int) -> List[int]:
    rng = random.Random(int(seed))
    labels = np.asarray([s.label for s in samples], dtype=np.float64)
    if not len(samples):
        return []
    mode = str(task_cfg.get("selection_strata", "label")).lower()
    strata: Dict[int, List[int]] = defaultdict(list)
    if mode == "quantile":
        bins = int(task_cfg.get("age_bins", 5))
        qs = np.unique(np.quantile(labels, np.linspace(0.0, 1.0, bins + 1)))
        if len(qs) <= 2:
            keys = np.zeros_like(labels, dtype=np.int64)
        else:
            keys = np.digitize(labels, qs[1:-1], right=True)
        for idx, key in enumerate(keys):
            strata[int(key)].append(idx)
    else:
        for idx, value in enumerate(labels):
            strata[int(round(float(value)))].append(idx)

    scored: List[Tuple[float, float, int]] = []
    for key, idxs in strata.items():
        rng.shuffle(idxs)
        n = max(1, len(idxs))
        for rank, idx in enumerate(idxs):
            scored.append(((rank + rng.random()) / n, rng.random() + key * 1.0e-6, idx))
    scored.sort()
    return [idx for _, _, idx in scored]


def weighted_f1_from_logits(y_true: torch.Tensor, logits: torch.Tensor, num_classes: int) -> float:
    return float(evaluate_classification(y_true.long().view(-1), logits, num_classes=num_classes)["f1_weighted"])


def class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y.astype(np.int64), minlength=int(num_classes)).astype(np.float64)
    weights = counts.sum() / np.maximum(1.0, counts * float(num_classes))
    return torch.from_numpy(weights.astype(np.float32))


def inverse_regression(pred_norm: np.ndarray, y_norm: np.ndarray, target_norm: Tuple[float, float]) -> Json:
    mu, sd = target_norm
    pred = pred_norm * float(sd) + float(mu)
    y = y_norm * float(sd) + float(mu)
    mae = float(np.mean(np.abs(pred - y)))
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return {"mae_years": mae, "rmse_years": rmse}


def predict_head_outputs(
    model: nn.Module,
    X: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    X_t = torch.from_numpy(X.astype(np.float32))
    dl = DataLoader(TensorDataset(X_t), batch_size=int(batch_size), shuffle=False)
    outs: List[torch.Tensor] = []
    with torch.no_grad():
        for (xb,) in dl:
            out = model(xb.to(device=device, dtype=torch.float32)).detach().cpu()
            outs.append(out)
    return torch.cat(outs, dim=0)


def metrics_rows_from_outputs(
    out_cat: torch.Tensor,
    y_cat: torch.Tensor,
    *,
    task_type: str,
    num_classes: int,
    target_norm: Optional[Tuple[float, float]] = None,
) -> Tuple[Json, List[Json]]:
    rows: List[Json] = []
    if task_type == "classification":
        metrics = evaluate_classification(y_cat.long().view(-1), out_cat, num_classes=int(num_classes))
        probs = torch.softmax(out_cat, dim=1)
        pred = torch.argmax(out_cat, dim=1)
        for i in range(int(y_cat.shape[0])):
            row = {"index": int(i), "target": int(y_cat[i].item()), "pred": int(pred[i].item())}
            if probs.shape[1] > 1:
                row["prob_pos"] = float(probs[i, 1].item())
            rows.append(row)
        metrics["accuracy"] = float(metrics.pop("acc"))
        metrics["f1_weighted"] = float(metrics["f1_weighted"])
        metrics["balanced_accuracy"] = float(metrics["balanced_acc"])
        metrics.pop("balanced_acc", None)
        return metrics, rows

    metrics = evaluate_regression(y_cat.view(-1), out_cat.view(-1))
    metrics["std_mse"] = float(metrics.pop("mse"))
    pred_np = out_cat.view(-1).numpy()
    y_np = y_cat.view(-1).numpy()
    if target_norm is not None:
        metrics.update(inverse_regression(pred_np, y_np, target_norm))
    for i in range(int(y_cat.shape[0])):
        row = {"index": int(i), "target": float(y_np[i]), "pred": float(pred_np[i])}
        if target_norm is not None:
            mu, sd = target_norm
            row["target_years"] = float(y_np[i] * sd + mu)
            row["pred_years"] = float(pred_np[i] * sd + mu)
        rows.append(row)
    return metrics, rows


def eval_head(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    task_type: str,
    num_classes: int,
    batch_size: int,
    device: torch.device,
    target_norm: Optional[Tuple[float, float]] = None,
) -> Tuple[Json, List[Json]]:
    out_cat = predict_head_outputs(model, X, batch_size=batch_size, device=device)
    y_cat = torch.from_numpy(y.astype(np.float32))
    return metrics_rows_from_outputs(
        out_cat,
        y_cat,
        task_type=task_type,
        num_classes=num_classes,
        target_norm=target_norm,
    )


def write_predictions(path: Path, rows: Sequence[Json], subjects: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for row in rows:
        r = dict(row)
        idx = int(r.pop("index"))
        r = {"subject": subjects[idx], **r}
        out_rows.append(r)
    fieldnames = list(out_rows[0].keys()) if out_rows else ["subject", "target", "pred"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def write_prediction_rows(path: Path, rows: Sequence[Json], meta_rows: Sequence[Json]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for row in rows:
        r = dict(row)
        idx = int(r.pop("index"))
        meta = dict(meta_rows[idx]) if 0 <= idx < len(meta_rows) else {}
        out_rows.append({**meta, **r})
    fieldnames = list(out_rows[0].keys()) if out_rows else ["subject", "target", "pred"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def aggregate_subject_outputs(
    outputs: torch.Tensor,
    y: np.ndarray,
    rows_meta: Sequence[Json],
    *,
    task_type: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[Json]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for idx, meta in enumerate(rows_meta):
        subject = str(meta.get("subject", ""))
        if subject not in grouped:
            grouped[subject] = {"outputs": [], "targets": [], "count": 0}
            order.append(subject)
        grouped[subject]["outputs"].append(outputs[idx])
        grouped[subject]["targets"].append(float(y[idx]))
        grouped[subject]["count"] += 1

    agg_outputs: List[torch.Tensor] = []
    agg_targets: List[float] = []
    agg_meta: List[Json] = []
    for subject in order:
        info = grouped[subject]
        subject_outputs = torch.stack(info["outputs"], dim=0)
        if task_type == "classification":
            agg_out = subject_outputs.mean(dim=0)
        else:
            agg_out = subject_outputs.view(-1).mean().view(1)
        agg_outputs.append(agg_out)
        agg_targets.append(float(np.mean(np.asarray(info["targets"], dtype=np.float32))))
        agg_meta.append({"subject": subject, "num_blocks": int(info["count"])})
    return torch.stack(agg_outputs, dim=0), torch.tensor(agg_targets, dtype=torch.float32), agg_meta


def train_one_run_aligned_lp(
    *,
    cfg: Json,
    cache: FeatureCache,
    task_name: str,
    task_cfg: Json,
    run_cfg: Json,
    ratio: float,
    seed: int,
    samples: Dict[str, Dict[str, List[SubjectSample]]],
) -> Json:
    set_seed(seed)
    train_cfg = effective_training_cfg(cfg)
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg.get("runtime", {}), dict) else {}
    show_tqdm = bool(runtime_cfg.get("show_tqdm", True))
    device = resolve_device(cfg)
    out_root = Path(cfg["paths"]["output_root"])
    run_dir = out_root / task_name / f"ratio_{float(ratio):.2f}" / f"seed_{int(seed)}"
    ensure_dir(run_dir)

    gt_train = list(samples["gt"]["train"])
    ge_train_all = list(samples["ge"]["train"])
    order = stratified_order(ge_train_all, run_cfg, seed)
    requested_ge = int(round(float(ratio) * len(gt_train)))
    selected_indices = order[: min(requested_ge, len(order))]
    ge_train = [ge_train_all[i] for i in selected_indices]
    train_subject_samples = gt_train + ge_train
    val_subject_samples = list(samples["gt"]["val"])
    test_subject_samples = list(samples["gt"]["test"])

    train_blocks = expand_samples_to_blocks(train_subject_samples)
    val_blocks = expand_samples_to_blocks(val_subject_samples)
    test_blocks = expand_samples_to_blocks(test_subject_samples)
    selected_ge_subjects = [str(s.subject) for s in ge_train]

    Xtr, ytr_raw, tr_meta = block_feature_matrix(
        cache,
        train_blocks,
        show_tqdm=show_tqdm,
        desc=f"{task_name} r={float(ratio):.2f} s={int(seed)} train block features",
    )
    Xva, yva_raw, va_meta = block_feature_matrix(
        cache,
        val_blocks,
        show_tqdm=show_tqdm,
        desc=f"{task_name} r={float(ratio):.2f} s={int(seed)} val block features",
    )
    Xte, yte_raw, te_meta = block_feature_matrix(
        cache,
        test_blocks,
        show_tqdm=show_tqdm,
        desc=f"{task_name} r={float(ratio):.2f} s={int(seed)} test block features",
    )

    standardize = bool(train_cfg.get("standardize_features", True))
    if standardize:
        mean = Xtr.mean(axis=0, keepdims=True)
        std = Xtr.std(axis=0, keepdims=True)
        std = np.where(std < 1.0e-6, 1.0, std)
        Xtr = (Xtr - mean) / std
        Xva = (Xva - mean) / std
        Xte = (Xte - mean) / std
    else:
        mean = np.zeros((1, Xtr.shape[1]), dtype=np.float32)
        std = np.ones((1, Xtr.shape[1]), dtype=np.float32)

    task_type = str(run_cfg["task_type"])
    num_classes = int(run_cfg.get("num_classes", 1))
    target_norm: Optional[Tuple[float, float]] = None
    if task_type == "regression":
        seen_subjects = set()
        gt_train_labels: List[float] = []
        for sample in gt_train:
            sid = str(sample.subject)
            if sid in seen_subjects:
                continue
            seen_subjects.add(sid)
            gt_train_labels.append(float(sample.label))
        gt_train_labels_np = np.asarray(gt_train_labels, dtype=np.float32)
        mu = float(np.mean(gt_train_labels_np))
        sd = float(np.std(gt_train_labels_np))
        sd = 1.0 if sd < 1.0e-12 else sd
        target_norm = (mu, sd)
        ytr = (ytr_raw - mu) / sd
        yva = (yva_raw - mu) / sd
        yte = (yte_raw - mu) / sd
        head = nn.Linear(int(Xtr.shape[1]), 1).to(device)
        loss_fn = nn.MSELoss()
        best_key = "pearson"
        maximize = True
    else:
        ytr = ytr_raw.astype(np.int64)
        yva = yva_raw.astype(np.int64)
        yte = yte_raw.astype(np.int64)
        head = nn.Linear(int(Xtr.shape[1]), int(num_classes)).to(device)
        if bool(train_cfg.get("use_class_weights", True)):
            loss_fn = nn.CrossEntropyLoss(weight=class_weights(ytr, num_classes).to(device))
        else:
            loss_fn = nn.CrossEntropyLoss()
        best_key = "f1_weighted"
        maximize = True

    batch_size = int(train_cfg.get("batch_size", 3))
    Xtr_t = torch.from_numpy(Xtr.astype(np.float32))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    train_dl = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(
        head.parameters(),
        lr=float(train_cfg.get("lr", 1.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    epochs = int(train_cfg.get("epochs", 30))
    patience = int(train_cfg.get("patience", 10))
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_metric = -float("inf") if maximize else float("inf")
    bad_epochs = 0
    history: List[Json] = []

    epoch_range: Iterable[int] = range(1, epochs + 1)
    epoch_pbar = None
    if show_tqdm:
        epoch_pbar = tqdm(
            epoch_range,
            desc=f"{task_name} r={float(ratio):.2f} s={int(seed)} aligned epochs",
            ncols=120,
            leave=False,
        )
        epoch_range = epoch_pbar

    for epoch in epoch_range:
        head.train()
        loss_sum = 0.0
        n_obs = 0
        for xb, yb in train_dl:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device)
            opt.zero_grad(set_to_none=True)
            out = head(xb)
            if task_type == "classification":
                loss = loss_fn(out, yb.long().view(-1))
            else:
                loss = loss_fn(out.view(-1), yb.float().view(-1))
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * int(xb.shape[0])
            n_obs += int(xb.shape[0])

        val_metrics, _ = eval_head(
            head,
            Xva,
            yva,
            task_type=task_type,
            num_classes=num_classes,
            batch_size=batch_size,
            device=device,
            target_norm=target_norm,
        )
        metric_value = float(val_metrics[best_key])
        is_better = metric_value > best_metric if maximize else metric_value < best_metric
        history.append({"epoch": epoch, "train_loss": loss_sum / max(1, n_obs), **{f"val_{k}": v for k, v in val_metrics.items()}})
        if is_better:
            best_metric = metric_value
            best_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
        if epoch_pbar is not None and hasattr(epoch_pbar, "set_postfix"):
            epoch_pbar.set_postfix(
                {
                    f"val_{best_key}": f"{metric_value:.4f}",
                    "best": f"{best_metric:.4f}",
                    "bad": int(bad_epochs),
                }
            )

    if epoch_pbar is not None and hasattr(epoch_pbar, "close"):
        epoch_pbar.close()

    if best_state is None:
        raise RuntimeError("no best aligned linear-probe state was recorded")
    head.load_state_dict(best_state)

    tr_out = predict_head_outputs(head, Xtr, batch_size=batch_size, device=device)
    va_out = predict_head_outputs(head, Xva, batch_size=batch_size, device=device)
    te_out = predict_head_outputs(head, Xte, batch_size=batch_size, device=device)

    ytr_t_eval = torch.from_numpy(ytr.astype(np.float32))
    yva_t_eval = torch.from_numpy(yva.astype(np.float32))
    yte_t_eval = torch.from_numpy(yte.astype(np.float32))

    train_sample_metrics, train_sample_rows = metrics_rows_from_outputs(
        tr_out,
        ytr_t_eval,
        task_type=task_type,
        num_classes=num_classes,
        target_norm=target_norm,
    )
    val_sample_metrics, val_sample_rows = metrics_rows_from_outputs(
        va_out,
        yva_t_eval,
        task_type=task_type,
        num_classes=num_classes,
        target_norm=target_norm,
    )
    test_sample_metrics, test_sample_rows = metrics_rows_from_outputs(
        te_out,
        yte_t_eval,
        task_type=task_type,
        num_classes=num_classes,
        target_norm=target_norm,
    )

    tr_sub_out, tr_sub_y, tr_sub_meta = aggregate_subject_outputs(tr_out, ytr, tr_meta, task_type=task_type)
    va_sub_out, va_sub_y, va_sub_meta = aggregate_subject_outputs(va_out, yva, va_meta, task_type=task_type)
    te_sub_out, te_sub_y, te_sub_meta = aggregate_subject_outputs(te_out, yte, te_meta, task_type=task_type)

    train_subject_metrics, train_subject_rows = metrics_rows_from_outputs(
        tr_sub_out,
        tr_sub_y,
        task_type=task_type,
        num_classes=num_classes,
        target_norm=target_norm,
    )
    val_subject_metrics, val_subject_rows = metrics_rows_from_outputs(
        va_sub_out,
        va_sub_y,
        task_type=task_type,
        num_classes=num_classes,
        target_norm=target_norm,
    )
    test_subject_metrics, test_subject_rows = metrics_rows_from_outputs(
        te_sub_out,
        te_sub_y,
        task_type=task_type,
        num_classes=num_classes,
        target_norm=target_norm,
    )

    write_prediction_rows(run_dir / "pred_train.csv", train_sample_rows, tr_meta)
    write_prediction_rows(run_dir / "pred_val.csv", val_sample_rows, va_meta)
    write_prediction_rows(run_dir / "pred_test.csv", test_sample_rows, te_meta)
    write_prediction_rows(run_dir / "pred_train_subject.csv", train_subject_rows, tr_sub_meta)
    write_prediction_rows(run_dir / "pred_val_subject.csv", val_subject_rows, va_sub_meta)
    write_prediction_rows(run_dir / "pred_test_subject.csv", test_subject_rows, te_sub_meta)

    torch.save(
        {
            "head_state": head.state_dict(),
            "feature_mean": mean.astype(np.float32),
            "feature_std": std.astype(np.float32),
            "target_norm": target_norm,
            "task_name": task_name,
            "ratio": float(ratio),
            "seed": int(seed),
            "feature_dim": int(Xtr.shape[1]),
            "lp_protocol": "exp2_cached_sample",
        },
        run_dir / "linear_probe.pt",
    )
    save_json({"history": history}, run_dir / "history.json")

    summary = {
        "task_name": task_name,
        "dataset": task_cfg["dataset"],
        "horizon": int(task_cfg["horizon"]),
        "ratio": float(ratio),
        "seed": int(seed),
        "task_type": task_type,
        "lp_protocol": "exp2_cached_sample",
        "feature_dim": int(Xtr.shape[1]),
        "target_norm": None if target_norm is None else {"mean": target_norm[0], "std": target_norm[1], "scope": "gt_train_unique_subjects"},
        "num_items": {"train": int(Xtr.shape[0]), "val": int(Xva.shape[0]), "test": int(Xte.shape[0])},
        "num_subjects": {
            "train": int(len({str(m["subject"]) for m in tr_meta})),
            "val": int(len({str(m["subject"]) for m in va_meta})),
            "test": int(len({str(m["subject"]) for m in te_meta})),
        },
        "train_block_count": int(Xtr.shape[0]),
        "val_block_count": int(Xva.shape[0]),
        "test_block_count": int(Xte.shape[0]),
        "train_subject_count": int(len({str(m["subject"]) for m in tr_meta})),
        "val_subject_count": int(len({str(m["subject"]) for m in va_meta})),
        "test_subject_count": int(len({str(m["subject"]) for m in te_meta})),
        "selected_ge_subjects": selected_ge_subjects,
        "augmentation": {
            "gt_train": int(len(gt_train)),
            "ge_train_available": int(len(ge_train_all)),
            "ge_requested": int(requested_ge),
            "ge_used": int(len(ge_train)),
            "shortfall": int(max(0, requested_ge - len(ge_train))),
        },
        "sample_metrics": {
            "train": train_sample_metrics,
            "val": val_sample_metrics,
            "test": test_sample_metrics,
            "best_val": float(best_metric),
            "best_key": best_key,
        },
        "subject_metrics": {
            "train": train_subject_metrics,
            "val": val_subject_metrics,
            "test": test_subject_metrics,
            "best_val": float(val_subject_metrics.get(best_key, 0.0)),
            "best_key": best_key,
        },
        "output_dir": str(run_dir),
    }
    save_json(summary, run_dir / "summary.json")
    return summary


def train_one_run_aligned_subject_mean_lp(
    *,
    cfg: Json,
    cache: FeatureCache,
    task_name: str,
    task_cfg: Json,
    run_cfg: Json,
    ratio: float,
    seed: int,
    samples: Dict[str, Dict[str, List[SubjectSample]]],
) -> Json:
    set_seed(seed)
    train_cfg = effective_training_cfg(cfg)
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg.get("runtime", {}), dict) else {}
    show_tqdm = bool(runtime_cfg.get("show_tqdm", True))
    device = resolve_device(cfg)
    out_root = Path(cfg["paths"]["output_root"])
    run_dir = out_root / task_name / f"ratio_{float(ratio):.2f}" / f"seed_{int(seed)}"
    ensure_dir(run_dir)

    gt_train = list(samples["gt"]["train"])
    ge_train_all = list(samples["ge"]["train"])
    order = stratified_order(ge_train_all, run_cfg, seed)
    requested_ge = int(round(float(ratio) * len(gt_train)))
    selected_indices = order[: min(requested_ge, len(order))]
    ge_train = [ge_train_all[i] for i in selected_indices]
    train_subject_samples = gt_train + ge_train
    val_subject_samples = list(samples["gt"]["val"])
    test_subject_samples = list(samples["gt"]["test"])
    selected_ge_subjects = [str(s.subject) for s in ge_train]

    Xtr, ytr_raw, tr_meta = subject_mean_feature_matrix(
        cache,
        train_subject_samples,
        show_tqdm=show_tqdm,
        desc=f"{task_name} r={float(ratio):.2f} s={int(seed)} train subject-mean features",
    )
    Xva, yva_raw, va_meta = subject_mean_feature_matrix(
        cache,
        val_subject_samples,
        show_tqdm=show_tqdm,
        desc=f"{task_name} r={float(ratio):.2f} s={int(seed)} val subject-mean features",
    )
    Xte, yte_raw, te_meta = subject_mean_feature_matrix(
        cache,
        test_subject_samples,
        show_tqdm=show_tqdm,
        desc=f"{task_name} r={float(ratio):.2f} s={int(seed)} test subject-mean features",
    )

    standardize = bool(train_cfg.get("standardize_features", True))
    if standardize:
        mean = Xtr.mean(axis=0, keepdims=True)
        std = Xtr.std(axis=0, keepdims=True)
        std = np.where(std < 1.0e-6, 1.0, std)
        Xtr = (Xtr - mean) / std
        Xva = (Xva - mean) / std
        Xte = (Xte - mean) / std
    else:
        mean = np.zeros((1, Xtr.shape[1]), dtype=np.float32)
        std = np.ones((1, Xtr.shape[1]), dtype=np.float32)

    task_type = str(run_cfg["task_type"])
    num_classes = int(run_cfg.get("num_classes", 1))
    target_norm: Optional[Tuple[float, float]] = None
    if task_type == "regression":
        seen_subjects = set()
        gt_train_labels: List[float] = []
        for sample in gt_train:
            sid = str(sample.subject)
            if sid in seen_subjects:
                continue
            seen_subjects.add(sid)
            gt_train_labels.append(float(sample.label))
        gt_train_labels_np = np.asarray(gt_train_labels, dtype=np.float32)
        mu = float(np.mean(gt_train_labels_np))
        sd = float(np.std(gt_train_labels_np))
        sd = 1.0 if sd < 1.0e-12 else sd
        target_norm = (mu, sd)
        ytr = (ytr_raw - mu) / sd
        yva = (yva_raw - mu) / sd
        yte = (yte_raw - mu) / sd
        head = nn.Linear(int(Xtr.shape[1]), 1).to(device)
        loss_fn = nn.MSELoss()
        best_key = "pearson"
        maximize = True
    else:
        ytr = ytr_raw.astype(np.int64)
        yva = yva_raw.astype(np.int64)
        yte = yte_raw.astype(np.int64)
        head = nn.Linear(int(Xtr.shape[1]), int(num_classes)).to(device)
        if bool(train_cfg.get("use_class_weights", True)):
            loss_fn = nn.CrossEntropyLoss(weight=class_weights(ytr, num_classes).to(device))
        else:
            loss_fn = nn.CrossEntropyLoss()
        best_key = "f1_weighted"
        maximize = True

    batch_size = int(train_cfg.get("batch_size", 3))
    Xtr_t = torch.from_numpy(Xtr.astype(np.float32))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    train_dl = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(
        head.parameters(),
        lr=float(train_cfg.get("lr", 1.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    epochs = int(train_cfg.get("epochs", 30))
    patience = int(train_cfg.get("patience", 10))
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_metric = -float("inf") if maximize else float("inf")
    bad_epochs = 0
    history: List[Json] = []

    epoch_range: Iterable[int] = range(1, epochs + 1)
    epoch_pbar = None
    if show_tqdm:
        epoch_pbar = tqdm(
            epoch_range,
            desc=f"{task_name} r={float(ratio):.2f} s={int(seed)} subject-mean epochs",
            ncols=120,
            leave=False,
        )
        epoch_range = epoch_pbar

    for epoch in epoch_range:
        head.train()
        loss_sum = 0.0
        n_obs = 0
        for xb, yb in train_dl:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device)
            opt.zero_grad(set_to_none=True)
            out = head(xb)
            if task_type == "classification":
                loss = loss_fn(out, yb.long().view(-1))
            else:
                loss = loss_fn(out.view(-1), yb.float().view(-1))
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * int(xb.shape[0])
            n_obs += int(xb.shape[0])

        val_metrics, _ = eval_head(
            head,
            Xva,
            yva,
            task_type=task_type,
            num_classes=num_classes,
            batch_size=batch_size,
            device=device,
            target_norm=target_norm,
        )
        metric_value = float(val_metrics[best_key])
        is_better = metric_value > best_metric if maximize else metric_value < best_metric
        history.append({"epoch": epoch, "train_loss": loss_sum / max(1, n_obs), **{f"val_{k}": v for k, v in val_metrics.items()}})
        if is_better:
            best_metric = metric_value
            best_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
        if epoch_pbar is not None and hasattr(epoch_pbar, "set_postfix"):
            epoch_pbar.set_postfix(
                {
                    f"val_{best_key}": f"{metric_value:.4f}",
                    "best": f"{best_metric:.4f}",
                    "bad": int(bad_epochs),
                }
            )

    if epoch_pbar is not None and hasattr(epoch_pbar, "close"):
        epoch_pbar.close()

    if best_state is None:
        raise RuntimeError("no best subject-mean aligned linear-probe state was recorded")
    head.load_state_dict(best_state)

    train_metrics, train_rows = eval_head(head, Xtr, ytr, task_type=task_type, num_classes=num_classes, batch_size=batch_size, device=device, target_norm=target_norm)
    val_metrics, val_rows = eval_head(head, Xva, yva, task_type=task_type, num_classes=num_classes, batch_size=batch_size, device=device, target_norm=target_norm)
    test_metrics, test_rows = eval_head(head, Xte, yte, task_type=task_type, num_classes=num_classes, batch_size=batch_size, device=device, target_norm=target_norm)

    write_prediction_rows(run_dir / "pred_train.csv", train_rows, tr_meta)
    write_prediction_rows(run_dir / "pred_val.csv", val_rows, va_meta)
    write_prediction_rows(run_dir / "pred_test.csv", test_rows, te_meta)
    write_prediction_rows(run_dir / "pred_train_subject.csv", train_rows, tr_meta)
    write_prediction_rows(run_dir / "pred_val_subject.csv", val_rows, va_meta)
    write_prediction_rows(run_dir / "pred_test_subject.csv", test_rows, te_meta)

    torch.save(
        {
            "head_state": head.state_dict(),
            "feature_mean": mean.astype(np.float32),
            "feature_std": std.astype(np.float32),
            "target_norm": target_norm,
            "task_name": task_name,
            "ratio": float(ratio),
            "seed": int(seed),
            "feature_dim": int(Xtr.shape[1]),
            "lp_protocol": "aligned_subject_mean",
            "subject_pool": "mean",
        },
        run_dir / "linear_probe.pt",
    )
    save_json({"history": history}, run_dir / "history.json")

    summary = {
        "task_name": task_name,
        "dataset": task_cfg["dataset"],
        "horizon": int(task_cfg["horizon"]),
        "ratio": float(ratio),
        "seed": int(seed),
        "task_type": task_type,
        "lp_protocol": "aligned_subject_mean",
        "feature_dim": int(Xtr.shape[1]),
        "subject_pool": "mean",
        "target_norm": None if target_norm is None else {"mean": target_norm[0], "std": target_norm[1], "scope": "gt_train_unique_subjects"},
        "num_items": {"train": int(Xtr.shape[0]), "val": int(Xva.shape[0]), "test": int(Xte.shape[0])},
        "num_subjects": {"train": int(len(tr_meta)), "val": int(len(va_meta)), "test": int(len(te_meta))},
        "train_subject_count": int(len(tr_meta)),
        "val_subject_count": int(len(va_meta)),
        "test_subject_count": int(len(te_meta)),
        "train_block_count": int(sum(int(m.get("num_blocks", 0)) for m in tr_meta)),
        "val_block_count": int(sum(int(m.get("num_blocks", 0)) for m in va_meta)),
        "test_block_count": int(sum(int(m.get("num_blocks", 0)) for m in te_meta)),
        "selected_ge_subjects": selected_ge_subjects,
        "augmentation": {
            "gt_train": int(len(gt_train)),
            "ge_train_available": int(len(ge_train_all)),
            "ge_requested": int(requested_ge),
            "ge_used": int(len(ge_train)),
            "shortfall": int(max(0, requested_ge - len(ge_train))),
        },
        "sample_metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "best_val": float(best_metric),
            "best_key": best_key,
        },
        "subject_metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "best_val": float(best_metric),
            "best_key": best_key,
        },
        "output_dir": str(run_dir),
    }
    save_json(summary, run_dir / "summary.json")
    return summary


def train_one_run(
    *,
    cfg: Json,
    cache: FeatureCache,
    task_name: str,
    task_cfg: Json,
    run_cfg: Json,
    ratio: float,
    seed: int,
    samples: Dict[str, Dict[str, List[SubjectSample]]],
) -> Json:
    protocol = resolve_lp_protocol(cfg)
    if protocol == "exp2_cached_sample":
        return train_one_run_aligned_lp(
            cfg=cfg,
            cache=cache,
            task_name=task_name,
            task_cfg=task_cfg,
            run_cfg=run_cfg,
            ratio=ratio,
            seed=seed,
            samples=samples,
        )
    if protocol == "aligned_subject_mean":
        return train_one_run_aligned_subject_mean_lp(
            cfg=cfg,
            cache=cache,
            task_name=task_name,
            task_cfg=task_cfg,
            run_cfg=run_cfg,
            ratio=ratio,
            seed=seed,
            samples=samples,
        )
    set_seed(seed)
    train_cfg = effective_training_cfg(cfg)
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg.get("runtime", {}), dict) else {}
    show_tqdm = bool(runtime_cfg.get("show_tqdm", True))
    device = resolve_device(cfg)
    out_root = Path(cfg["paths"]["output_root"])
    run_dir = out_root / task_name / f"ratio_{float(ratio):.2f}" / f"seed_{int(seed)}"
    ensure_dir(run_dir)

    gt_train = list(samples["gt"]["train"])
    ge_train_all = list(samples["ge"]["train"])
    order = stratified_order(ge_train_all, run_cfg, seed)
    requested_ge = int(round(float(ratio) * len(gt_train)))
    selected_indices = order[: min(requested_ge, len(order))]
    ge_train = [ge_train_all[i] for i in selected_indices]
    train_samples = gt_train + ge_train
    val_samples = list(samples["gt"]["val"])
    test_samples = list(samples["gt"]["test"])

    Xtr, ytr_raw, tr_subjects = feature_matrix(
        cache,
        train_samples,
        show_tqdm=show_tqdm,
        desc=f"{task_name} r={float(ratio):.2f} s={int(seed)} train features",
    )
    Xva, yva_raw, va_subjects = feature_matrix(
        cache,
        val_samples,
        show_tqdm=show_tqdm,
        desc=f"{task_name} r={float(ratio):.2f} s={int(seed)} val features",
    )
    Xte, yte_raw, te_subjects = feature_matrix(
        cache,
        test_samples,
        show_tqdm=show_tqdm,
        desc=f"{task_name} r={float(ratio):.2f} s={int(seed)} test features",
    )

    standardize = bool(train_cfg.get("standardize_features", True))
    if standardize:
        mean = Xtr.mean(axis=0, keepdims=True)
        std = Xtr.std(axis=0, keepdims=True)
        std = np.where(std < 1.0e-6, 1.0, std)
        Xtr = (Xtr - mean) / std
        Xva = (Xva - mean) / std
        Xte = (Xte - mean) / std
    else:
        mean = np.zeros((1, Xtr.shape[1]), dtype=np.float32)
        std = np.ones((1, Xtr.shape[1]), dtype=np.float32)

    task_type = str(run_cfg["task_type"])
    num_classes = int(run_cfg.get("num_classes", 1))
    target_norm: Optional[Tuple[float, float]] = None
    if task_type == "regression":
        gt_train_labels = np.asarray([s.label for s in gt_train], dtype=np.float32)
        mu = float(np.mean(gt_train_labels))
        sd = float(np.std(gt_train_labels))
        sd = 1.0 if sd < 1.0e-12 else sd
        target_norm = (mu, sd)
        ytr = (ytr_raw - mu) / sd
        yva = (yva_raw - mu) / sd
        yte = (yte_raw - mu) / sd
        head = nn.Linear(Xtr.shape[1], 1).to(device)
        loss_fn: nn.Module = nn.MSELoss()
        best_key = "pearson"
        maximize = True
    else:
        ytr = ytr_raw.astype(np.int64)
        yva = yva_raw.astype(np.int64)
        yte = yte_raw.astype(np.int64)
        head = nn.Linear(Xtr.shape[1], int(num_classes)).to(device)
        if bool(train_cfg.get("use_class_weights", True)):
            loss_fn = nn.CrossEntropyLoss(weight=class_weights(ytr, num_classes).to(device))
        else:
            loss_fn = nn.CrossEntropyLoss()
        best_key = "f1_weighted"
        maximize = True

    batch_size = int(train_cfg.get("batch_size", 64))
    Xtr_t = torch.from_numpy(Xtr.astype(np.float32))
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    train_dl = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(head.parameters(), lr=float(train_cfg.get("lr", 1.0e-3)), weight_decay=float(train_cfg.get("weight_decay", 1.0e-4)))
    epochs = int(train_cfg.get("epochs", 80))
    patience = int(train_cfg.get("patience", 15))
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_metric = -float("inf") if maximize else float("inf")
    bad_epochs = 0
    history: List[Json] = []

    epoch_range: Iterable[int] = range(1, epochs + 1)
    epoch_pbar = None
    if show_tqdm:
        epoch_pbar = tqdm(
            epoch_range,
            desc=f"{task_name} r={float(ratio):.2f} s={int(seed)} epochs",
            ncols=120,
            leave=False,
        )
        epoch_range = epoch_pbar

    for epoch in epoch_range:
        head.train()
        loss_sum = 0.0
        n_obs = 0
        for xb, yb in train_dl:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device)
            opt.zero_grad(set_to_none=True)
            out = head(xb)
            if task_type == "classification":
                loss = loss_fn(out, yb.long().view(-1))
            else:
                loss = loss_fn(out.view(-1), yb.float().view(-1))
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * int(xb.shape[0])
            n_obs += int(xb.shape[0])

        val_metrics, _ = eval_head(
            head,
            Xva,
            yva,
            task_type=task_type,
            num_classes=num_classes,
            batch_size=batch_size,
            device=device,
            target_norm=target_norm,
        )
        metric_value = float(val_metrics[best_key])
        is_better = metric_value > best_metric if maximize else metric_value < best_metric
        history.append({"epoch": epoch, "train_loss": loss_sum / max(1, n_obs), **{f"val_{k}": v for k, v in val_metrics.items()}})
        if is_better:
            best_metric = metric_value
            best_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
        if epoch_pbar is not None and hasattr(epoch_pbar, "set_postfix"):
            epoch_pbar.set_postfix(
                {
                    f"val_{best_key}": f"{metric_value:.4f}",
                    "best": f"{best_metric:.4f}",
                    "bad": int(bad_epochs),
                }
            )

    if epoch_pbar is not None and hasattr(epoch_pbar, "close"):
        epoch_pbar.close()

    if best_state is None:
        raise RuntimeError("no best linear-probe state was recorded")
    head.load_state_dict(best_state)

    train_metrics, train_rows = eval_head(head, Xtr, ytr, task_type=task_type, num_classes=num_classes, batch_size=batch_size, device=device, target_norm=target_norm)
    val_metrics, val_rows = eval_head(head, Xva, yva, task_type=task_type, num_classes=num_classes, batch_size=batch_size, device=device, target_norm=target_norm)
    test_metrics, test_rows = eval_head(head, Xte, yte, task_type=task_type, num_classes=num_classes, batch_size=batch_size, device=device, target_norm=target_norm)

    write_predictions(run_dir / "pred_train.csv", train_rows, tr_subjects)
    write_predictions(run_dir / "pred_val.csv", val_rows, va_subjects)
    write_predictions(run_dir / "pred_test.csv", test_rows, te_subjects)
    torch.save(
        {
            "head_state": head.state_dict(),
            "feature_mean": mean.astype(np.float32),
            "feature_std": std.astype(np.float32),
            "target_norm": target_norm,
            "task_name": task_name,
            "ratio": float(ratio),
            "seed": int(seed),
            "feature_dim": int(Xtr.shape[1]),
        },
        run_dir / "linear_probe.pt",
    )
    save_json({"history": history}, run_dir / "history.json")

    summary = {
        "task_name": task_name,
        "dataset": task_cfg["dataset"],
        "horizon": int(task_cfg["horizon"]),
        "ratio": float(ratio),
        "seed": int(seed),
        "task_type": task_type,
        "feature_dim": int(Xtr.shape[1]),
        "target_norm": None if target_norm is None else {"mean": target_norm[0], "std": target_norm[1], "scope": "gt_train"},
        "num_items": {"train": int(Xtr.shape[0]), "val": int(Xva.shape[0]), "test": int(Xte.shape[0])},
        "augmentation": {
            "gt_train": int(len(gt_train)),
            "ge_train_available": int(len(ge_train_all)),
            "ge_requested": int(requested_ge),
            "ge_used": int(len(ge_train)),
            "shortfall": int(max(0, requested_ge - len(ge_train))),
        },
        "metrics": {"train": train_metrics, "val": val_metrics, "test": test_metrics, "best_val": float(best_metric), "best_key": best_key},
        "output_dir": str(run_dir),
    }
    save_json(summary, run_dir / "summary.json")
    return summary


def flatten_result(summary: Json) -> Json:
    if "sample_metrics" in summary and "subject_metrics" in summary:
        row = {
            "task_name": summary["task_name"],
            "dataset": summary["dataset"],
            "horizon": int(summary["horizon"]),
            "ratio": float(summary["ratio"]),
            "seed": int(summary["seed"]),
            "lp_protocol": str(summary.get("lp_protocol", "")),
            "train_n": int(summary["num_items"]["train"]),
            "val_n": int(summary["num_items"]["val"]),
            "test_n": int(summary["num_items"]["test"]),
            "train_subject_count": int(summary.get("train_subject_count", summary.get("num_subjects", {}).get("train", 0))),
            "val_subject_count": int(summary.get("val_subject_count", summary.get("num_subjects", {}).get("val", 0))),
            "test_subject_count": int(summary.get("test_subject_count", summary.get("num_subjects", {}).get("test", 0))),
            "gt_train": int(summary["augmentation"]["gt_train"]),
            "ge_requested": int(summary["augmentation"]["ge_requested"]),
            "ge_used": int(summary["augmentation"]["ge_used"]),
            "ge_shortfall": int(summary["augmentation"]["shortfall"]),
        }
        for prefix, metric_group in (("sample", summary["sample_metrics"]), ("subject", summary["subject_metrics"])):
            for split in ("train", "val", "test"):
                for k, v in metric_group[split].items():
                    row[f"{prefix}_{split}_{k}"] = float(v)
        return row
    row: Json = {
        "task_name": summary["task_name"],
        "dataset": summary["dataset"],
        "horizon": int(summary["horizon"]),
        "ratio": float(summary["ratio"]),
        "seed": int(summary["seed"]),
        "train_n": int(summary["num_items"]["train"]),
        "val_n": int(summary["num_items"]["val"]),
        "test_n": int(summary["num_items"]["test"]),
        "gt_train": int(summary["augmentation"]["gt_train"]),
        "ge_requested": int(summary["augmentation"]["ge_requested"]),
        "ge_used": int(summary["augmentation"]["ge_used"]),
        "ge_shortfall": int(summary["augmentation"]["shortfall"]),
    }
    for split in ("train", "val", "test"):
        for k, v in summary["metrics"][split].items():
            row[f"{split}_{k}"] = float(v)
    return row


def write_csv(path: Path, rows: Sequence[Json]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_results(rows: Sequence[Json]) -> Tuple[List[Json], str]:
    grouped: Dict[Tuple[str, float], List[Json]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task_name"]), float(row["ratio"]))].append(row)
    agg_rows: List[Json] = []
    for (task_name, ratio), items in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        base = {
            "task_name": task_name,
            "dataset": items[0]["dataset"],
            "horizon": int(items[0]["horizon"]),
            "ratio": float(ratio),
            "seeds": len(items),
            "gt_train": int(items[0]["gt_train"]),
            "ge_used_mean": float(np.mean([x["ge_used"] for x in items])),
            "ge_shortfall_mean": float(np.mean([x["ge_shortfall"] for x in items])),
        }
        metric_keys = sorted(k for k in items[0].keys() if k.startswith("test_"))
        for key in metric_keys:
            vals = np.asarray([float(x[key]) for x in items if key in x], dtype=np.float64)
            if vals.size:
                base[f"{key}_mean"] = float(np.mean(vals))
                base[f"{key}_std"] = float(np.std(vals))
        agg_rows.append(base)

    md = ["# Exp1 Downstream GT/GE Augmentation", "", "Existing split CSVs are used as source of truth; observed proportions are about 7:1:2.", ""]
    for task_name in sorted(set(r["task_name"] for r in agg_rows)):
        md.append(f"## {task_name}")
        task_rows = [r for r in agg_rows if r["task_name"] == task_name]
        if "subject_test_pearson_mean" in task_rows[0] or "subject_test_f1_weighted_mean" in task_rows[0]:
            if str(task_rows[0]["dataset"]) == "HCP":
                preferred = [
                    "sample_test_f1_weighted_mean",
                    "subject_test_f1_weighted_mean",
                    "sample_test_accuracy_mean",
                    "subject_test_accuracy_mean",
                    "sample_test_balanced_accuracy_mean",
                    "subject_test_balanced_accuracy_mean",
                ]
            else:
                preferred = [
                    "sample_test_pearson_mean",
                    "subject_test_pearson_mean",
                    "sample_test_mae_years_mean",
                    "subject_test_mae_years_mean",
                    "sample_test_rmse_years_mean",
                    "subject_test_rmse_years_mean",
                    "subject_test_r2_mean",
                ]
            cols = ["ratio", "seeds", "ge_used_mean"] + [x for x in preferred if x in task_rows[0]]
        else:
            primary = "test_f1_weighted" if str(task_rows[0]["dataset"]) == "HCP" else "test_pearson"
            secondary = ["test_accuracy", "test_balanced_accuracy", "test_auroc"] if str(task_rows[0]["dataset"]) == "HCP" else ["test_mae_years", "test_rmse_years", "test_r2", "test_std_mse"]
            cols = ["ratio", "seeds", "ge_used_mean", f"{primary}_mean", f"{primary}_std"] + [f"{x}_mean" for x in secondary if f"{x}_mean" in task_rows[0]]
        md.append("| " + " | ".join(cols) + " |")
        md.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for row in task_rows:
            vals = []
            for col in cols:
                v = row.get(col, "")
                vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
            md.append("| " + " | ".join(vals) + " |")
        md.append("")
    return agg_rows, "\n".join(md)


def main() -> None:
    args = parse_args()
    denv = init_dist_env(args)
    cfg = load_json(args.config)
    apply_protocol_path_overrides(cfg)
    if denv.enabled and str(cfg.get("device", "auto")).strip().lower() == "auto" and torch.cuda.is_available():
        cfg["device"] = f"cuda:{int(denv.local_rank)}"
    task_names = selected_task_names(cfg, args.tasks)
    ensure_dir(cfg["paths"]["output_root"])
    ensure_dir(cfg["paths"]["feature_cache_root"])
    ensure_dir(cfg["paths"]["report_root"])

    bundle = build_manifest(cfg, task_names, max_subjects_per_split=int(args.max_subjects_per_split))
    report_root = Path(cfg["paths"]["report_root"])
    if denv.rank == 0:
        save_json(bundle.audits, report_root / "manifest_audit.json")
    dist_barrier_if_needed(denv)
    if denv.rank == 0:
        print_preflight(bundle)
    if args.preflight_only:
        if denv.rank == 0:
            print(f"[done] preflight audit: {report_root / 'manifest_audit.json'}")
        if denv.enabled and dist.is_initialized():
            dist.destroy_process_group()
        return

    cache = FeatureCache(cfg)
    if not args.train_only:
        if denv.rank == 0:
            blocks = iter_task_blocks(bundle, task_names)
            audit = cache.extract_missing(blocks)
            save_json(audit, report_root / "feature_cache_audit.json")
            print(f"[extract] {audit}", flush=True)
        dist_barrier_if_needed(denv)
    if args.extract_only:
        if denv.rank == 0:
            print(f"[done] feature cache audit: {report_root / 'feature_cache_audit.json'}")
        if denv.enabled and dist.is_initialized():
            dist.destroy_process_group()
        return

    results: List[Json] = []
    tasks_by_name = {str(t["name"]): dict(t) for t in cfg.get("tasks", [])}
    jobs: List[Tuple[str, float, int]] = []
    for task_name in task_names:
        for ratio in [float(x) for x in cfg.get("ratios", [0.0])]:
            for seed in [int(x) for x in cfg.get("seeds", [cfg.get("seed", 42)])]:
                jobs.append((task_name, ratio, seed))
    local_jobs = [job for i, job in enumerate(jobs) if (i % denv.world_size) == denv.rank]
    print(
        f"[rank {denv.rank}/{denv.world_size}] device={resolve_device(cfg)} local_jobs={len(local_jobs)} total_jobs={len(jobs)}",
        flush=True,
    )
    for task_name, ratio, seed in local_jobs:
        task_cfg = tasks_by_name[task_name]
        run_cfg = cfg["runs"][str(task_cfg["dataset"])]
        print(f"[train] task={task_name} ratio={ratio:.2f} seed={seed}", flush=True)
        summary = train_one_run(
            cfg=cfg,
            cache=cache,
            task_name=task_name,
            task_cfg=task_cfg,
            run_cfg=run_cfg,
            ratio=ratio,
            seed=seed,
            samples=bundle.samples[task_name],
        )
        flat = flatten_result(summary)
        results.append(flat)
        if "subject_metrics" in summary and "sample_metrics" in summary:
            metric = "subject_test_f1_weighted" if str(run_cfg["task_type"]) == "classification" else "subject_test_pearson"
        else:
            metric = "test_f1_weighted" if str(run_cfg["task_type"]) == "classification" else "test_pearson"
        print(f"[done] {task_name} ratio={ratio:.2f} seed={seed} {metric}={flat.get(metric, float('nan')):.4f}", flush=True)

    ddp_parts_root = report_root / "ddp_parts"
    ensure_dir(ddp_parts_root)
    part_csv = ddp_parts_root / f"all_results.rank{denv.rank}.csv"
    part_json = ddp_parts_root / f"all_results.rank{denv.rank}.json"
    write_csv(part_csv, results)
    save_json({"rank": denv.rank, "world_size": denv.world_size, "results": results}, part_json)
    dist_barrier_if_needed(denv)

    if denv.rank == 0:
        merged_results: List[Json] = []
        for r in range(denv.world_size):
            part = ddp_parts_root / f"all_results.rank{r}.json"
            if not part.exists():
                continue
            payload = load_json(part)
            merged_results.extend(list(payload.get("results", [])))
        write_csv(report_root / "all_results.csv", merged_results)
        agg_rows, md = aggregate_results(merged_results)
        write_csv(report_root / "aggregate_results.csv", agg_rows)
        save_json({"results": merged_results, "aggregate": agg_rows}, report_root / "aggregate_results.json")
        (report_root / "aggregate_report.md").write_text(md, encoding="utf-8")
        print(f"[done] aggregate report: {report_root / 'aggregate_report.md'}")

    dist_barrier_if_needed(denv)
    if denv.enabled and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
