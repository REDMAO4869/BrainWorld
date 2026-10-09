from __future__ import annotations

import csv
import json
import os
import re
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import get_worker_info


CHUNK_RE = re.compile(r"chunk[_-]?(\d+)", re.IGNORECASE)
FRAME_RANGE_RE = re.compile(r"_(\d+)-(\d+)(?:__\d+)?$", re.IGNORECASE)


def _is_rank0() -> bool:
    return str(os.environ.get("RANK", "0")).strip() in {"", "0"}


def _progress(msg: str) -> None:
    if _is_rank0():
        print(f"[data] {msg}", flush=True)


_WARNED_SAMPLE_ERRORS = 0
_MAX_WARNED_SAMPLE_ERRORS = 20


def _warn_sample_error(msg: str) -> None:
    global _WARNED_SAMPLE_ERRORS
    if not _is_rank0():
        return
    worker = get_worker_info()
    if worker is not None and int(worker.id) != 0:
        return
    if _WARNED_SAMPLE_ERRORS >= _MAX_WARNED_SAMPLE_ERRORS:
        return
    _WARNED_SAMPLE_ERRORS += 1
    print(f"[data][warn] {msg}", flush=True)


@dataclass(frozen=True)
class TargetLatentRecord:
    subject_id: str
    sequence_id: str
    chunk_id: int
    npz_path: str
    source_path: str


@dataclass(frozen=True)
class FeatureRecord:
    subject_id: str
    sequence_id: Optional[str]
    chunk_id: Optional[int]
    array_path: str


@dataclass(frozen=True)
class PairSample:
    subject_id: str
    sequence_id: str
    anchor_chunk_id: int
    target_chunk_id: int
    direction: str
    target_npz_path: str
    target_source_path: str
    fc_array_path: str = ""
    mri_array_path: str = ""
    video_array_path: str = ""
    audio_array_path: str = ""


@dataclass
class MetadataTable:
    vectors: Dict[str, np.ndarray]
    dim: int
    categorical_cols: List[str]
    category_sizes: Dict[str, int]

    def lookup(self, subject_id: str) -> tuple[Optional[np.ndarray], bool]:
        v = self.vectors.get(str(subject_id), None)
        return v, v is not None


@dataclass
class ConditionStore:
    name: str
    enabled: bool
    required: bool
    alignment: str
    array_field: str
    dtype: str
    records: List[FeatureRecord]
    sample_shape: Optional[Tuple[int, ...]]
    by_subject: Dict[str, FeatureRecord]
    by_subject_sequence_chunk: Dict[Tuple[str, str, int], FeatureRecord]
    by_subject_chunk: Dict[Tuple[str, int], FeatureRecord]

    def lookup(self, subject_id: str, anchor_chunk_id: int, sequence_id: str = "") -> tuple[Optional[FeatureRecord], bool]:
        if not self.enabled:
            return None, False
        if self.alignment == "subject":
            rec = self.by_subject.get(str(subject_id), None)
            return rec, rec is not None
        if str(sequence_id) != "":
            rec = self.by_subject_sequence_chunk.get((str(subject_id), str(sequence_id), int(anchor_chunk_id)), None)
            if rec is not None:
                return rec, True
        rec = self.by_subject_chunk.get((str(subject_id), int(anchor_chunk_id)), None)
        return rec, rec is not None


@dataclass
class SplitAudit:
    split: str
    num_target_latents: int
    num_pair_samples: int
    num_subjects: int
    direction_counts: Dict[str, int]
    target_preview: List[str]
    pair_preview: List[Dict[str, object]]
    condition_shape: Dict[str, Optional[Tuple[int, ...]]]
    condition_available: Dict[str, int]
    condition_required_missing: Dict[str, int]
    dropped_required_pairs: Dict[str, int]


class SampleLoadError(RuntimeError):
    pass


def _ensure_list(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return [str(x) for x in v]


def _select_npz_key(keys: List[str], preferred: str) -> str:
    if preferred and preferred in keys:
        return preferred
    for key in ("mu", "z", "arr_0", "arr", "data", "x"):
        if key in keys:
            return key
    return keys[0]


def _load_array(path: str, preferred_key: str = "", dtype: str = "float32") -> np.ndarray:
    p = Path(path)
    dt = np.float16 if str(dtype).lower() == "float16" else np.float32
    if p.suffix.lower() == ".npy":
        arr = np.load(str(p))
        return np.asarray(arr, dtype=dt)
    if p.suffix.lower() == ".npz":
        with np.load(str(p)) as data:
            keys = list(data.keys())
            if len(keys) == 0:
                raise ValueError(f"Empty npz file: {path}")
            key = _select_npz_key(keys, preferred_key)
            return np.asarray(data[key], dtype=dt)
    raise ValueError(f"Unsupported array file: {path}")


def _load_array_spec(path_spec: str, preferred_key: str = "", dtype: str = "float32") -> np.ndarray:
    spec = str(path_spec).strip()
    if spec.startswith("["):
        paths = json.loads(spec)
        if not isinstance(paths, list) or len(paths) == 0:
            raise ValueError(f"Invalid path-list spec: {path_spec}")
        arrays = []
        first_shape = None
        for raw in paths:
            arr = _load_array(str(raw), preferred_key=preferred_key, dtype=dtype)
            if first_shape is None:
                first_shape = tuple(int(v) for v in arr.shape)
            elif tuple(int(v) for v in arr.shape) != first_shape:
                raise ValueError(f"Path-list arrays have mismatched shapes in {path_spec}")
            arrays.append(np.asarray(arr))
        return np.stack(arrays, axis=0)
    return _load_array(spec, preferred_key=preferred_key, dtype=dtype)


def _raise_sample_load_error(kind: str, path: str, exc: Exception) -> None:
    raise SampleLoadError(f"{kind} load failed: {path} ({type(exc).__name__}: {exc})") from exc


def _probe_array_readable(path: str, preferred_key: str = "", dtype: str = "float32") -> tuple[bool, str]:
    try:
        if str(path).strip().startswith("["):
            arr = _load_array_spec(str(path), preferred_key=preferred_key, dtype=dtype)
            _ = tuple(int(v) for v in arr.shape)
            return True, ""
        p = Path(path)
        if not p.is_file():
            return False, "missing"
        if p.suffix.lower() == ".npy":
            arr = np.load(str(p), mmap_mode="r")
            _ = tuple(int(v) for v in arr.shape)
            return True, ""
        if p.suffix.lower() == ".npz":
            with np.load(str(p)) as data:
                keys = list(data.keys())
                if len(keys) == 0:
                    return False, "empty npz"
                key = _select_npz_key(keys, preferred_key)
                arr = data[key]
                _ = tuple(int(v) for v in arr.shape)
            return True, ""
        # Fallback to normal loader for any supported edge case.
        _ = _load_array(str(p), preferred_key=preferred_key, dtype=dtype)
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _infer_subject_id(raw_subject: str, path_hint: str) -> str:
    s = str(raw_subject).strip()
    if s != "":
        return s
    stem = _normalized_stem(path_hint)
    if "__" in stem:
        return stem.split("__", 1)[0]
    if "_" in stem:
        return stem.split("_", 1)[0]
    parent = os.path.basename(os.path.dirname(path_hint))
    return parent if parent != "" else stem


def _normalized_stem(path_hint: str) -> str:
    stem = os.path.basename(str(path_hint).strip())
    while True:
        lowered = stem.lower()
        if lowered.endswith(".nii.gz"):
            stem = stem[:-7]
            continue
        ext = os.path.splitext(stem)[1]
        if ext.lower() in {".npy", ".npz", ".nii", ".gz"} and ext != "":
            stem = stem[:-len(ext)]
            continue
        break
    return stem


def _infer_sequence_id(subject_id: str, *values: str) -> str:
    def _canonicalize(prefix: str) -> str:
        sid = str(subject_id).strip()
        dup = f"{sid}__{sid}__"
        if sid != "" and prefix.startswith(dup):
            return prefix[len(sid) + 2 :]
        return prefix

    for value in values:
        s = str(value).strip()
        if s == "":
            continue
        stem = _normalized_stem(s)
        m = FRAME_RANGE_RE.search(stem)
        if m is not None:
            prefix = _canonicalize(stem[: m.start()].rstrip("_-"))
            if prefix != "":
                return prefix
        m = CHUNK_RE.search(stem)
        if m is not None:
            prefix = _canonicalize(stem[: m.start()].rstrip("_-"))
            if prefix != "":
                return prefix
    return str(subject_id)


def _infer_chunk_id(*values: str) -> Optional[int]:
    for value in values:
        s = str(value).strip()
        if s == "":
            continue
        stem = _normalized_stem(s)
        m = FRAME_RANGE_RE.search(stem)
        if m is not None:
            return int(m.group(1))
    for value in values:
        s = str(value)
        m = CHUNK_RE.search(s)
        if m is not None:
            return int(m.group(1))
    for value in values:
        stem = _normalized_stem(str(value))
        nums = re.findall(r"(\d+)", stem)
        if len(nums) > 0:
            return int(nums[-1])
    return None


def _iter_csv_rows(csv_path: str) -> Iterable[dict]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {str(k).strip(): str(v).strip() for k, v in row.items() if k is not None}


def _resolve_path(path_value: str, base_dir: str) -> str:
    if path_value == "":
        return ""
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(base_dir, path_value))


def _csv_map_candidates(csv_map: Dict | None, key: str, defaults: Sequence[str]) -> List[str]:
    out: List[str] = []
    if isinstance(csv_map, dict):
        raw = csv_map.get(key, None)
        if isinstance(raw, str):
            v = raw.strip()
            if v != "":
                out.append(v)
        elif isinstance(raw, (list, tuple)):
            for x in raw:
                v = str(x).strip()
                if v != "":
                    out.append(v)
    for d in defaults:
        v = str(d).strip()
        if v != "":
            out.append(v)

    dedup: List[str] = []
    seen: set[str] = set()
    for v in out:
        k = v.lower()
        if k in seen:
            continue
        seen.add(k)
        dedup.append(v)
    return dedup


def _row_get_first(row: Dict[str, str], keys: Sequence[str]) -> str:
    for k in keys:
        if k in row:
            v = str(row.get(k, "")).strip()
            if v != "":
                return v
    lower_map = {str(k).strip().lower(): str(v).strip() for k, v in row.items()}
    for k in keys:
        v = lower_map.get(str(k).strip().lower(), "")
        if v != "":
            return v
    return ""


def _discover_target_records(
    roots: Sequence[str],
    csv_map: Dict | None = None,
    verify_exists: bool = True,
) -> List[TargetLatentRecord]:
    path_cols = _csv_map_candidates(csv_map, "path", ("npz_path", "Path", "path"))
    source_cols = _csv_map_candidates(csv_map, "source_path", ("source_path",))
    subject_cols = _csv_map_candidates(csv_map, "subject", ("subject_id", "Subject"))
    sequence_cols = _csv_map_candidates(csv_map, "sequence", ("sequence_id", "run_id", "session_id"))
    chunk_cols = _csv_map_candidates(csv_map, "chunk", ("chunk_id", "Chunk"))

    out: Dict[str, TargetLatentRecord] = {}
    for raw in roots:
        root = os.path.abspath(str(raw))
        if not os.path.exists(root):
            continue

        if os.path.isfile(root) and root.lower().endswith(".csv"):
            base = os.path.dirname(root)
            for row in _iter_csv_rows(root):
                npz_path = _row_get_first(row, path_cols)
                npz_path = _resolve_path(npz_path, base)
                if npz_path == "":
                    continue
                if verify_exists and (not os.path.isfile(npz_path)):
                    continue
                source_path = _resolve_path(_row_get_first(row, source_cols), base)
                subject_id = _infer_subject_id(_row_get_first(row, subject_cols), npz_path)
                sequence_id = str(_row_get_first(row, sequence_cols)).strip()
                if sequence_id == "":
                    sequence_id = _infer_sequence_id(subject_id, source_path, npz_path)
                chunk_id = _infer_chunk_id(_row_get_first(row, chunk_cols), source_path, npz_path)
                if chunk_id is None:
                    continue
                out[npz_path] = TargetLatentRecord(
                    subject_id=subject_id,
                    sequence_id=sequence_id,
                    chunk_id=int(chunk_id),
                    npz_path=npz_path,
                    source_path=source_path,
                )
            continue

        if os.path.isdir(root):
            manifest = os.path.join(root, "manifest.csv")
            if os.path.isfile(manifest):
                for rec in _discover_target_records([manifest], csv_map=csv_map):
                    out[rec.npz_path] = rec
                continue

            for rr, _, files in os.walk(root):
                for fn in files:
                    if not fn.endswith(".npz"):
                        continue
                    npz_path = os.path.join(rr, fn)
                    subject_id = _infer_subject_id("", npz_path)
                    sequence_id = _infer_sequence_id(subject_id, npz_path)
                    chunk_id = _infer_chunk_id(npz_path)
                    if chunk_id is None:
                        continue
                    out[npz_path] = TargetLatentRecord(
                        subject_id=subject_id,
                        sequence_id=sequence_id,
                        chunk_id=int(chunk_id),
                        npz_path=npz_path,
                        source_path="",
                    )
    return sorted(out.values(), key=lambda x: (x.subject_id, x.sequence_id, x.chunk_id, x.npz_path))


def _resolve_feature_path_value(path_value: str, base_dir: str) -> str:
    raw = str(path_value).strip()
    if raw.startswith("["):
        return raw
    return _resolve_path(raw, base_dir)


def _discover_direct_pair_samples(
    roots: Sequence[str],
    csv_map: Dict | None = None,
    verify_exists: bool = True,
) -> List[PairSample]:
    anchor_path_cols = _csv_map_candidates(csv_map, "path", ("vae_latent_path", "npz_path", "Path", "path"))
    target_path_cols = _csv_map_candidates(csv_map, "target_path", ("target_latent_path",))
    source_cols = _csv_map_candidates(csv_map, "source_path", ("voxel_data_path", "source_path"))
    target_source_cols = _csv_map_candidates(csv_map, "target_source_path", ("target_voxel_data_path",))
    subject_cols = _csv_map_candidates(csv_map, "subject", ("subject_id", "Subject"))
    sequence_cols = _csv_map_candidates(csv_map, "sequence", ("sequence_id", "run_id", "session_id"))
    anchor_chunk_cols = _csv_map_candidates(csv_map, "anchor_chunk", ("anchor_chunk_id", "chunk_id", "Chunk"))
    target_chunk_cols = _csv_map_candidates(csv_map, "target_chunk", ("target_chunk_id",))
    direction_cols = _csv_map_candidates(csv_map, "direction", ("pair_direction", "direction"))
    fc_path_cols = _csv_map_candidates(csv_map, "fc_path", ("fc_embedding_path",))
    mri_path_cols = _csv_map_candidates(csv_map, "mri_path", ("MRI_embedding_path",))
    video_path_cols = _csv_map_candidates(csv_map, "video_path", ("video_embedding_path",))
    audio_path_cols = _csv_map_candidates(csv_map, "audio_path", ("audio_embedding_path",))

    out: List[PairSample] = []
    for raw in roots:
        root = os.path.abspath(str(raw))
        if (not os.path.exists(root)) or (not os.path.isfile(root)) or (not root.lower().endswith(".csv")):
            continue

        base = os.path.dirname(root)
        for row in _iter_csv_rows(root):
            anchor_path = _resolve_path(_row_get_first(row, anchor_path_cols), base)
            source_path = _resolve_path(_row_get_first(row, source_cols), base)
            target_npz_path = _resolve_path(_row_get_first(row, target_path_cols), base)
            if target_npz_path == "":
                continue
            if verify_exists and (not os.path.isfile(target_npz_path)):
                continue

            target_source_path = _resolve_path(_row_get_first(row, target_source_cols), base)
            subject_id = _infer_subject_id(_row_get_first(row, subject_cols), anchor_path if anchor_path != "" else target_npz_path)
            sequence_id = str(_row_get_first(row, sequence_cols)).strip()
            if sequence_id == "":
                sequence_id = _infer_sequence_id(subject_id, source_path, anchor_path, target_npz_path)

            anchor_chunk_id = _infer_chunk_id(_row_get_first(row, anchor_chunk_cols), source_path, anchor_path)
            if anchor_chunk_id is None:
                continue
            target_chunk_id = _infer_chunk_id(_row_get_first(row, target_chunk_cols), target_source_path, target_npz_path)
            if target_chunk_id is None:
                continue

            direction = str(_row_get_first(row, direction_cols)).strip().lower()
            if direction == "":
                direction = "next"
            fc_array_path = _resolve_feature_path_value(_row_get_first(row, fc_path_cols), base)
            mri_array_path = _resolve_feature_path_value(_row_get_first(row, mri_path_cols), base)
            video_array_path = _resolve_feature_path_value(_row_get_first(row, video_path_cols), base)
            audio_array_path = _resolve_feature_path_value(_row_get_first(row, audio_path_cols), base)

            out.append(
                PairSample(
                    subject_id=subject_id,
                    sequence_id=sequence_id,
                    anchor_chunk_id=int(anchor_chunk_id),
                    target_chunk_id=int(target_chunk_id),
                    direction=direction,
                    target_npz_path=target_npz_path,
                    target_source_path=target_source_path,
                    fc_array_path=fc_array_path,
                    mri_array_path=mri_array_path,
                    video_array_path=video_array_path,
                    audio_array_path=audio_array_path,
                )
            )

    return out


def _discover_feature_records(
    roots: Sequence[str],
    alignment: str,
    csv_map: Dict | None = None,
    verify_exists: bool = True,
) -> List[FeatureRecord]:
    path_cols = _csv_map_candidates(csv_map, "path", ("path", "Path", "npz_path"))
    subject_cols = _csv_map_candidates(csv_map, "subject", ("subject_id", "Subject"))
    sequence_cols = _csv_map_candidates(csv_map, "sequence", ("sequence_id", "run_id", "session_id"))
    chunk_cols = _csv_map_candidates(csv_map, "chunk", ("chunk_id", "Chunk"))

    out: Dict[str, FeatureRecord] = {}
    for raw in roots:
        root = os.path.abspath(str(raw))
        if not os.path.exists(root):
            continue

        if os.path.isfile(root) and root.lower().endswith(".csv"):
            base = os.path.dirname(root)
            for row in _iter_csv_rows(root):
                path_value = _row_get_first(row, path_cols)
                arr_path = _resolve_path(path_value, base)
                if arr_path == "":
                    continue
                if verify_exists and (not os.path.isfile(arr_path)):
                    continue
                subject_id = _infer_subject_id(_row_get_first(row, subject_cols), arr_path)
                sequence_id = None
                if alignment != "subject":
                    sequence_id = str(_row_get_first(row, sequence_cols)).strip()
                    if sequence_id == "":
                        sequence_id = _infer_sequence_id(subject_id, arr_path)
                chunk_id = _infer_chunk_id(_row_get_first(row, chunk_cols), arr_path)
                if alignment == "subject_chunk" and chunk_id is None:
                    continue
                rec = FeatureRecord(
                    subject_id=subject_id,
                    sequence_id=(None if alignment == "subject" else str(sequence_id)),
                    chunk_id=(None if alignment == "subject" else int(chunk_id)),
                    array_path=arr_path,
                )
                out[arr_path] = rec
            continue

        if os.path.isdir(root):
            for rr, _, files in os.walk(root):
                for fn in files:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext not in {".npy", ".npz"}:
                        continue
                    arr_path = os.path.join(rr, fn)
                    subject_id = _infer_subject_id("", arr_path)
                    sequence_id = None if alignment == "subject" else _infer_sequence_id(subject_id, arr_path)
                    chunk_id = _infer_chunk_id(arr_path)
                    if alignment == "subject_chunk" and chunk_id is None:
                        continue
                    out[arr_path] = FeatureRecord(
                        subject_id=subject_id,
                        sequence_id=sequence_id,
                        chunk_id=(None if alignment == "subject" else int(chunk_id)),
                        array_path=arr_path,
                    )
    return sorted(
        out.values(),
        key=lambda x: (x.subject_id, "" if x.sequence_id is None else x.sequence_id, -1 if x.chunk_id is None else x.chunk_id, x.array_path),
    )


def _condition_roots_for_split(spec_cfg: Dict, split: str) -> List[str]:
    if split in spec_cfg:
        return _ensure_list(spec_cfg.get(split, []))
    return _ensure_list(spec_cfg.get("roots", []))


def _probe_feature_shape(records: Sequence[FeatureRecord], array_field: str, dtype: str) -> Optional[Tuple[int, ...]]:
    for rec in records:
        try:
            arr = _load_array(rec.array_path, preferred_key=array_field, dtype=dtype)
            return tuple(int(v) for v in arr.shape)
        except Exception:
            continue
    return None


def _build_condition_store(name: str, spec_cfg: Dict, split: str, verify_exists: bool = True) -> ConditionStore:
    enabled = bool(spec_cfg.get("enabled", False))
    required = bool(spec_cfg.get("required", False))
    alignment = str(spec_cfg.get("alignment", "subject_chunk")).strip().lower()
    if alignment not in {"subject", "subject_chunk"}:
        raise ValueError(f"conditions.{name}.alignment must be subject or subject_chunk, got {alignment}")

    roots = _condition_roots_for_split(spec_cfg, split)
    records = (
        _discover_feature_records(
            roots,
            alignment=alignment,
            csv_map=spec_cfg.get("csv_map", {}),
            verify_exists=verify_exists,
        )
        if enabled
        else []
    )
    sample_shape = _probe_feature_shape(records, str(spec_cfg.get("array_field", "")), str(spec_cfg.get("dtype", "float32")))

    by_subject: Dict[str, FeatureRecord] = {}
    by_subject_sequence_chunk: Dict[Tuple[str, str, int], FeatureRecord] = {}
    by_subject_chunk: Dict[Tuple[str, int], FeatureRecord] = {}
    for rec in records:
        if rec.chunk_id is None:
            by_subject[str(rec.subject_id)] = rec
        else:
            if rec.sequence_id is not None:
                by_subject_sequence_chunk[(str(rec.subject_id), str(rec.sequence_id), int(rec.chunk_id))] = rec
            by_subject_chunk[(str(rec.subject_id), int(rec.chunk_id))] = rec

    return ConditionStore(
        name=name,
        enabled=enabled,
        required=required,
        alignment=alignment,
        array_field=str(spec_cfg.get("array_field", "")),
        dtype=str(spec_cfg.get("dtype", "float32")),
        records=records,
        sample_shape=sample_shape,
        by_subject=by_subject,
        by_subject_sequence_chunk=by_subject_sequence_chunk,
        by_subject_chunk=by_subject_chunk,
    )


def _empty_condition_store(name: str, spec_cfg: Dict) -> ConditionStore:
    enabled = bool(spec_cfg.get("enabled", False))
    required = bool(spec_cfg.get("required", False))
    alignment = str(spec_cfg.get("alignment", "subject_chunk")).strip().lower()
    return ConditionStore(
        name=name,
        enabled=enabled,
        required=required,
        alignment=alignment,
        array_field=str(spec_cfg.get("array_field", "")),
        dtype=str(spec_cfg.get("dtype", "float32")),
        records=[],
        sample_shape=None,
        by_subject={},
        by_subject_sequence_chunk={},
        by_subject_chunk={},
    )


def _probe_feature_shape_from_pair_samples(
    samples: Sequence[PairSample],
    attr: str,
    array_field: str,
    dtype: str,
) -> Optional[Tuple[int, ...]]:
    for sample in samples:
        path = str(getattr(sample, attr, "")).strip()
        if path == "":
            continue
        try:
            arr = _load_array_spec(path, preferred_key=array_field, dtype=dtype)
            return tuple(int(v) for v in arr.shape)
        except Exception:
            continue
    return None


def _sample_condition_attr(name: str) -> str:
    mapping = {
        "fc": "fc_array_path",
        "mri": "mri_array_path",
        "video": "video_array_path",
        "audio": "audio_array_path",
    }
    if name not in mapping:
        raise KeyError(f"Unsupported condition name={name}")
    return mapping[name]


def _sample_condition_path(sample: PairSample, name: str) -> str:
    return str(getattr(sample, _sample_condition_attr(name), "")).strip()


def _probe_optional_condition_sample(sample: PairSample, store: ConditionStore) -> tuple[bool, PairSample]:
    path_attr = _sample_condition_attr(store.name)
    path_value = str(getattr(sample, path_attr, "")).strip()
    if path_value == "":
        return (not store.required), sample
    ok, _ = _probe_array_readable(path_value, preferred_key=store.array_field, dtype=store.dtype)
    if ok:
        return True, sample
    if store.required:
        return False, sample
    return True, replace(sample, **{path_attr: ""})


def _build_metadata_table(spec_cfg: Dict) -> Optional[MetadataTable]:
    if not bool(spec_cfg.get("enabled", False)):
        return None

    csv_path = str(spec_cfg.get("path", "")).strip()
    if csv_path == "":
        return None
    csv_path = os.path.abspath(csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"metadata csv not found: {csv_path}")

    rows = list(_iter_csv_rows(csv_path))
    if len(rows) == 0:
        raise ValueError(f"metadata csv has no rows: {csv_path}")

    header = list(rows[0].keys())
    subject_col = str(spec_cfg.get("subject_col", "")).strip()
    if subject_col == "":
        for cand in ("subject_id", "Subject", "subject", "participant_id"):
            if cand in header:
                subject_col = cand
                break
    if subject_col == "" or subject_col not in header:
        raise ValueError(f"Unable to find metadata subject column in {csv_path}")

    categorical_cols = [str(x) for x in spec_cfg.get("categorical_cols", [])]
    if len(categorical_cols) == 0:
        categorical_cols = [c for c in header if c != subject_col]
    if len(categorical_cols) == 0:
        raise ValueError("metadata requires at least one categorical column")

    missing_token = str(spec_cfg.get("missing_token", "__missing__"))
    category_values: Dict[str, List[str]] = {}
    for col in categorical_cols:
        vals = set()
        for row in rows:
            raw = row.get(col, "")
            vals.add(raw if raw != "" else missing_token)
        category_values[col] = sorted(vals)

    vectors: Dict[str, np.ndarray] = {}
    category_sizes = {k: len(v) for k, v in category_values.items()}
    total_dim = sum(category_sizes.values())
    offsets: Dict[str, int] = {}
    cursor = 0
    for col in categorical_cols:
        offsets[col] = cursor
        cursor += category_sizes[col]

    for row in rows:
        subject_id = str(row.get(subject_col, "")).strip()
        if subject_id == "":
            continue
        vec = np.zeros((total_dim,), dtype=np.float32)
        for col in categorical_cols:
            raw = row.get(col, "")
            val = raw if raw != "" else missing_token
            vocab = category_values[col]
            idx = vocab.index(val)
            vec[offsets[col] + idx] = 1.0
        vectors[subject_id] = vec

    return MetadataTable(
        vectors=vectors,
        dim=int(total_dim),
        categorical_cols=categorical_cols,
        category_sizes=category_sizes,
    )


def _build_pair_samples(records: Sequence[TargetLatentRecord], cfg: Dict, split: str, seed: int) -> List[PairSample]:
    task_cfg = cfg.get("task", {})
    mode = str(task_cfg.get("mode", "next_only")).strip().lower()
    if mode not in {"next_only", "prev_only", "mixed", "both_per_anchor"}:
        raise ValueError(f"task.mode invalid: {mode}")
    allow_boundary = bool(task_cfg.get("allow_boundary_single_direction", True))
    samp_cfg = task_cfg.get("direction_sampling", {})
    next_prob = float(samp_cfg.get("next_prob", 0.5))
    prev_prob = float(samp_cfg.get("prev_prob", 0.5))
    total = max(1.0e-6, next_prob + prev_prob)
    next_prob = next_prob / total

    grouped: Dict[Tuple[str, str], List[TargetLatentRecord]] = {}
    for rec in records:
        grouped.setdefault((rec.subject_id, rec.sequence_id), []).append(rec)

    split_offset = {"train": 11, "val": 23, "test": 37}.get(str(split), 53)
    rng = random.Random(int(seed) + split_offset)
    pairs: List[PairSample] = []
    for (subject_id, sequence_id), items in grouped.items():
        seq = sorted(items, key=lambda x: x.chunk_id)
        for idx, anchor in enumerate(seq):
            has_prev = idx > 0
            has_next = idx + 1 < len(seq)
            prev_rec = seq[idx - 1] if has_prev else None
            next_rec = seq[idx + 1] if has_next else None

            if mode == "next_only":
                if next_rec is not None:
                    pairs.append(PairSample(subject_id, sequence_id, anchor.chunk_id, next_rec.chunk_id, "next", next_rec.npz_path, next_rec.source_path))
                continue

            if mode == "prev_only":
                if prev_rec is not None:
                    pairs.append(PairSample(subject_id, sequence_id, anchor.chunk_id, prev_rec.chunk_id, "prev", prev_rec.npz_path, prev_rec.source_path))
                continue

            if mode == "both_per_anchor":
                if prev_rec is not None:
                    pairs.append(PairSample(subject_id, sequence_id, anchor.chunk_id, prev_rec.chunk_id, "prev", prev_rec.npz_path, prev_rec.source_path))
                if next_rec is not None:
                    pairs.append(PairSample(subject_id, sequence_id, anchor.chunk_id, next_rec.chunk_id, "next", next_rec.npz_path, next_rec.source_path))
                continue

            options: List[Tuple[str, TargetLatentRecord]] = []
            if prev_rec is not None:
                options.append(("prev", prev_rec))
            if next_rec is not None:
                options.append(("next", next_rec))
            if len(options) == 0:
                continue
            if len(options) == 1:
                if not allow_boundary:
                    continue
                direction, target = options[0]
            else:
                direction, target = options[0]
                if rng.random() < next_prob:
                    direction, target = ("next", next_rec) if next_rec is not None else ("prev", prev_rec)
                else:
                    direction, target = ("prev", prev_rec) if prev_rec is not None else ("next", next_rec)
                if target is None:
                    direction, target = options[0]
            pairs.append(PairSample(subject_id, sequence_id, anchor.chunk_id, target.chunk_id, direction, target.npz_path, target.source_path))

    return pairs


class ConditionalLatentPairDataset(Dataset):
    def __init__(
        self,
        *,
        split: str,
        pair_samples: Sequence[PairSample],
        target_latent_field: str,
        target_dtype: str,
        fc_store: ConditionStore,
        mri_store: ConditionStore,
        video_store: ConditionStore,
        audio_store: ConditionStore,
        metadata_table: Optional[MetadataTable],
    ) -> None:
        if len(pair_samples) == 0:
            raise ValueError(f"{split} dataset has 0 pair samples")
        self.split = str(split)
        self.pair_samples = list(pair_samples)
        self.target_latent_field = str(target_latent_field)
        self.target_dtype = str(target_dtype)
        self.fc_store = fc_store
        self.mri_store = mri_store
        self.video_store = video_store
        self.audio_store = audio_store
        self.metadata_table = metadata_table

        self.target_shape = self._probe_target_shape()
        self.fc_shape = self.fc_store.sample_shape
        if self.fc_shape is None and self.fc_store.enabled:
            self.fc_shape = _probe_feature_shape_from_pair_samples(
                self.pair_samples,
                "fc_array_path",
                self.fc_store.array_field,
                self.fc_store.dtype,
            )
        self.mri_shape = self.mri_store.sample_shape
        if self.mri_shape is None and self.mri_store.enabled:
            self.mri_shape = _probe_feature_shape_from_pair_samples(
                self.pair_samples,
                "mri_array_path",
                self.mri_store.array_field,
                self.mri_store.dtype,
            )
        self.video_shape = self.video_store.sample_shape
        if self.video_shape is None and self.video_store.enabled:
            self.video_shape = _probe_feature_shape_from_pair_samples(
                self.pair_samples,
                "video_array_path",
                self.video_store.array_field,
                self.video_store.dtype,
            )
        self.audio_shape = self.audio_store.sample_shape
        if self.audio_shape is None and self.audio_store.enabled:
            self.audio_shape = _probe_feature_shape_from_pair_samples(
                self.pair_samples,
                "audio_array_path",
                self.audio_store.array_field,
                self.audio_store.dtype,
            )
        self.meta_dim = 0 if self.metadata_table is None else int(self.metadata_table.dim)

    def _probe_target_shape(self) -> Tuple[int, ...]:
        for sample in self.pair_samples:
            try:
                arr = _load_array(sample.target_npz_path, preferred_key=self.target_latent_field, dtype=self.target_dtype)
                return tuple(int(v) for v in arr.shape)
            except Exception:
                continue
        raise RuntimeError(f"Unable to probe target latent shape for split={self.split}")

    def __len__(self) -> int:
        return len(self.pair_samples)

    def _load_feature_or_zero(
        self,
        *,
        sample_path: str,
        shape: Optional[Tuple[int, ...]],
        store: ConditionStore,
        subject_id: str,
        sequence_id: str,
        anchor_chunk_id: int,
    ) -> tuple[torch.Tensor, bool]:
        if not store.enabled or shape is None:
            return torch.zeros((0,), dtype=torch.float32), False
        if str(sample_path).strip() != "":
            try:
                arr = _load_array_spec(str(sample_path).strip(), preferred_key=store.array_field, dtype=store.dtype)
                arr = np.asarray(arr, dtype=np.float32)
                if tuple(int(v) for v in arr.shape) != tuple(int(v) for v in shape):
                    raise ValueError(
                        f"{store.name} feature shape mismatch for {sample_path}: "
                        f"got {tuple(arr.shape)}, expected {shape}"
                    )
                return torch.from_numpy(arr), True
            except Exception as exc:
                if store.required:
                    _raise_sample_load_error(store.name, str(sample_path).strip(), exc)
                _warn_sample_error(f"optional {store.name} feature unavailable, zero-filled: {sample_path} ({type(exc).__name__}: {exc})")
                return torch.zeros(shape, dtype=torch.float32), False
        rec, has_value = store.lookup(subject_id, anchor_chunk_id, sequence_id=sequence_id)
        if not has_value or rec is None:
            return torch.zeros(shape, dtype=torch.float32), False
        try:
            arr = _load_array_spec(rec.array_path, preferred_key=store.array_field, dtype=store.dtype)
            arr = np.asarray(arr, dtype=np.float32)
            if tuple(int(v) for v in arr.shape) != tuple(int(v) for v in shape):
                raise ValueError(
                    f"{store.name} feature shape mismatch for {rec.array_path}: "
                    f"got {tuple(arr.shape)}, expected {shape}"
                )
            return torch.from_numpy(arr), True
        except Exception as exc:
            if store.required:
                _raise_sample_load_error(store.name, rec.array_path, exc)
            _warn_sample_error(f"optional {store.name} feature unavailable, zero-filled: {rec.array_path} ({type(exc).__name__}: {exc})")
            return torch.zeros(shape, dtype=torch.float32), False

    def _load_metadata_or_zero(self, subject_id: str) -> tuple[torch.Tensor, bool]:
        if self.metadata_table is None or self.metadata_table.dim <= 0:
            return torch.zeros((0,), dtype=torch.float32), False
        vec, has_value = self.metadata_table.lookup(subject_id)
        if (not has_value) or vec is None:
            return torch.zeros((self.metadata_table.dim,), dtype=torch.float32), False
        return torch.from_numpy(np.asarray(vec, dtype=np.float32)), True

    def _build_item(self, sample: PairSample) -> Dict[str, object]:
        try:
            target = _load_array(sample.target_npz_path, preferred_key=self.target_latent_field, dtype=self.target_dtype)
        except Exception as exc:
            _raise_sample_load_error("target", sample.target_npz_path, exc)

        fc_tensor, has_fc = self._load_feature_or_zero(
            sample_path=sample.fc_array_path,
            shape=self.fc_shape,
            store=self.fc_store,
            subject_id=sample.subject_id,
            sequence_id=sample.sequence_id,
            anchor_chunk_id=sample.anchor_chunk_id,
        )
        mri_tensor, has_mri = self._load_feature_or_zero(
            sample_path=sample.mri_array_path,
            shape=self.mri_shape,
            store=self.mri_store,
            subject_id=sample.subject_id,
            sequence_id=sample.sequence_id,
            anchor_chunk_id=sample.anchor_chunk_id,
        )
        video_tensor, has_video = self._load_feature_or_zero(
            sample_path=sample.video_array_path,
            shape=self.video_shape,
            store=self.video_store,
            subject_id=sample.subject_id,
            sequence_id=sample.sequence_id,
            anchor_chunk_id=sample.anchor_chunk_id,
        )
        audio_tensor, has_audio = self._load_feature_or_zero(
            sample_path=sample.audio_array_path,
            shape=self.audio_shape,
            store=self.audio_store,
            subject_id=sample.subject_id,
            sequence_id=sample.sequence_id,
            anchor_chunk_id=sample.anchor_chunk_id,
        )
        meta_tensor, has_meta = self._load_metadata_or_zero(sample.subject_id)

        return {
            "target_latent": torch.from_numpy(np.asarray(target, dtype=np.float32)),
            "direction_id": 1 if sample.direction == "next" else 0,
            "direction": sample.direction,
            "subject_id": sample.subject_id,
            "sequence_id": sample.sequence_id,
            "anchor_chunk_id": int(sample.anchor_chunk_id),
            "target_chunk_id": int(sample.target_chunk_id),
            "target_npz_path": sample.target_npz_path,
            "target_source_path": sample.target_source_path,
            "fc_cond": fc_tensor,
            "has_fc": bool(has_fc),
            "mri_cond": mri_tensor,
            "has_mri": bool(has_mri),
            "video_cond": video_tensor,
            "has_video": bool(has_video),
            "audio_cond": audio_tensor,
            "has_audio": bool(has_audio),
            "meta_cond": meta_tensor,
            "has_meta": bool(has_meta),
        }

    def __getitem__(self, idx: int) -> Dict[str, object]:
        num_samples = len(self.pair_samples)
        start_idx = int(idx) % num_samples
        max_tries = num_samples
        last_err: Optional[Exception] = None

        for offset in range(max_tries):
            cur_idx = (start_idx + offset) % num_samples
            sample = self.pair_samples[cur_idx]
            try:
                return self._build_item(sample)
            except SampleLoadError as exc:
                last_err = exc
                _warn_sample_error(
                    f"skipping bad sample split={self.split} idx={cur_idx} subject={sample.subject_id} "
                    f"sequence={sample.sequence_id} anchor={sample.anchor_chunk_id}: {exc}"
                )
                continue

        raise RuntimeError(
            f"Unable to find a readable sample after {max_tries} attempts on split={self.split}, "
            f"start_idx={start_idx}; last_error={last_err}"
        )


def _availability_count(samples: Sequence[PairSample], store: ConditionStore) -> tuple[int, int]:
    if not store.enabled:
        return 0, 0
    avail = 0
    missing_required = 0
    for sample in samples:
        direct_path = _sample_condition_path(sample, store.name)
        ok = str(direct_path).strip() != ""
        if not ok:
            _, ok = store.lookup(sample.subject_id, sample.anchor_chunk_id, sequence_id=sample.sequence_id)
        if ok:
            avail += 1
        elif store.required:
            missing_required += 1
    return avail, missing_required


def _metadata_availability(samples: Sequence[PairSample], table: Optional[MetadataTable], required: bool) -> tuple[int, int]:
    if table is None:
        return 0, len(samples) if required else 0
    avail = 0
    missing = 0
    for sample in samples:
        _, ok = table.lookup(sample.subject_id)
        if ok:
            avail += 1
        elif required:
            missing += 1
    return avail, missing


def _filter_pair_samples_missing_required(
    samples: Sequence[PairSample],
    *,
    fc_store: ConditionStore,
    mri_store: ConditionStore,
    video_store: ConditionStore,
    audio_store: ConditionStore,
    metadata_table: Optional[MetadataTable],
    metadata_required: bool,
) -> tuple[List[PairSample], Dict[str, int]]:
    dropped = {"fc": 0, "mri": 0, "video": 0, "audio": 0, "metadata": 0}
    kept: List[PairSample] = []

    for sample in samples:
        miss_fc = False
        miss_mri = False
        miss_video = False
        miss_audio = False
        miss_meta = False

        if fc_store.enabled and fc_store.required:
            ok = _sample_condition_path(sample, "fc") != ""
            if not ok:
                _, ok = fc_store.lookup(sample.subject_id, sample.anchor_chunk_id, sequence_id=sample.sequence_id)
            miss_fc = not ok

        if mri_store.enabled and mri_store.required:
            ok = _sample_condition_path(sample, "mri") != ""
            if not ok:
                _, ok = mri_store.lookup(sample.subject_id, sample.anchor_chunk_id, sequence_id=sample.sequence_id)
            miss_mri = not ok

        if video_store.enabled and video_store.required:
            ok = _sample_condition_path(sample, "video") != ""
            if not ok:
                _, ok = video_store.lookup(sample.subject_id, sample.anchor_chunk_id, sequence_id=sample.sequence_id)
            miss_video = not ok

        if audio_store.enabled and audio_store.required:
            ok = _sample_condition_path(sample, "audio") != ""
            if not ok:
                _, ok = audio_store.lookup(sample.subject_id, sample.anchor_chunk_id, sequence_id=sample.sequence_id)
            miss_audio = not ok

        if metadata_required:
            if metadata_table is None:
                miss_meta = True
            else:
                _, ok = metadata_table.lookup(sample.subject_id)
                miss_meta = not ok

        if miss_fc or miss_mri or miss_video or miss_audio or miss_meta:
            if miss_fc:
                dropped["fc"] += 1
            if miss_mri:
                dropped["mri"] += 1
            if miss_video:
                dropped["video"] += 1
            if miss_audio:
                dropped["audio"] += 1
            if miss_meta:
                dropped["metadata"] += 1
            continue

        kept.append(sample)

    return kept, dropped


def _prefilter_direct_pair_samples(
    samples: Sequence[PairSample],
    *,
    target_latent_field: str,
    target_dtype: str,
    fc_store: ConditionStore,
    mri_store: ConditionStore,
    video_store: ConditionStore,
    audio_store: ConditionStore,
) -> tuple[List[PairSample], Dict[str, int]]:
    dropped = {"target": 0, "fc": 0, "mri": 0, "video": 0, "audio": 0}
    kept: List[PairSample] = []

    for sample in samples:
        ok, _ = _probe_array_readable(sample.target_npz_path, preferred_key=target_latent_field, dtype=target_dtype)
        if not ok:
            dropped["target"] += 1
            continue

        sample_out = sample

        if fc_store.enabled and fc_store.required:
            fc_path = str(sample.fc_array_path).strip()
            ok, _ = _probe_array_readable(fc_path, preferred_key=fc_store.array_field, dtype=fc_store.dtype)
            if (fc_path == "") or (not ok):
                dropped["fc"] += 1
                continue

        if mri_store.enabled:
            mri_path = str(sample.mri_array_path).strip()
            if mri_path != "":
                ok, _ = _probe_array_readable(mri_path, preferred_key=mri_store.array_field, dtype=mri_store.dtype)
                if not ok:
                    dropped["mri"] += 1
                    sample_out = replace(sample, mri_array_path="")

        ok, sample_out = _probe_optional_condition_sample(sample_out, video_store)
        if not ok:
            dropped["video"] += 1
            continue

        ok, sample_out = _probe_optional_condition_sample(sample_out, audio_store)
        if not ok:
            dropped["audio"] += 1
            continue

        kept.append(sample_out)

    return kept, dropped


def _audit_split(
    *,
    split: str,
    target_records: Sequence[TargetLatentRecord],
    pair_samples: Sequence[PairSample],
    fc_store: ConditionStore,
    mri_store: ConditionStore,
    video_store: ConditionStore,
    audio_store: ConditionStore,
    metadata_table: Optional[MetadataTable],
    metadata_required: bool,
    dropped_required_pairs: Optional[Dict[str, int]] = None,
) -> SplitAudit:
    direction_counts = {"prev": 0, "next": 0}
    for sample in pair_samples:
        direction_counts[sample.direction] = direction_counts.get(sample.direction, 0) + 1

    fc_avail, fc_missing = _availability_count(pair_samples, fc_store)
    mri_avail, mri_missing = _availability_count(pair_samples, mri_store)
    video_avail, video_missing = _availability_count(pair_samples, video_store)
    audio_avail, audio_missing = _availability_count(pair_samples, audio_store)
    meta_avail, meta_missing = _metadata_availability(pair_samples, metadata_table, metadata_required)
    if len(target_records) > 0:
        num_target_latents = len(target_records)
        num_subjects = len(set(x.subject_id for x in target_records))
        target_preview = [x.npz_path for x in target_records[:5]]
    else:
        num_target_latents = len(set(x.target_npz_path for x in pair_samples))
        num_subjects = len(set(x.subject_id for x in pair_samples))
        target_preview = [x.target_npz_path for x in pair_samples[:5]]

    return SplitAudit(
        split=split,
        num_target_latents=num_target_latents,
        num_pair_samples=len(pair_samples),
        num_subjects=num_subjects,
        direction_counts=direction_counts,
        target_preview=target_preview,
        pair_preview=[
            {
                "subject_id": x.subject_id,
                "sequence_id": x.sequence_id,
                "anchor_chunk_id": int(x.anchor_chunk_id),
                "target_chunk_id": int(x.target_chunk_id),
                "direction": x.direction,
            }
            for x in pair_samples[:5]
        ],
        condition_shape={
            "fc": fc_store.sample_shape,
            "mri": mri_store.sample_shape,
            "video": video_store.sample_shape,
            "audio": audio_store.sample_shape,
            "metadata": None if metadata_table is None else (int(metadata_table.dim),),
        },
        condition_available={
            "fc": fc_avail,
            "mri": mri_avail,
            "video": video_avail,
            "audio": audio_avail,
            "metadata": meta_avail,
        },
        condition_required_missing={
            "fc": fc_missing,
            "mri": mri_missing,
            "video": video_missing,
            "audio": audio_missing,
            "metadata": meta_missing,
        },
        dropped_required_pairs={
            "fc": int((dropped_required_pairs or {}).get("fc", 0)),
            "mri": int((dropped_required_pairs or {}).get("mri", 0)),
            "video": int((dropped_required_pairs or {}).get("video", 0)),
            "audio": int((dropped_required_pairs or {}).get("audio", 0)),
            "metadata": int((dropped_required_pairs or {}).get("metadata", 0)),
        },
    )


def build_splits_from_config(
    cfg: Dict,
) -> Tuple[ConditionalLatentPairDataset, ConditionalLatentPairDataset, ConditionalLatentPairDataset, Dict[str, SplitAudit]]:
    data_cfg = cfg.get("data", {})
    target_cfg = data_cfg.get("target", {})
    cond_cfg = data_cfg.get("conditions", {})

    train_roots = _ensure_list(target_cfg.get("train", []))
    val_roots = _ensure_list(target_cfg.get("val", []))
    test_roots = _ensure_list(target_cfg.get("test", []))
    if len(train_roots) == 0 or len(val_roots) == 0 or len(test_roots) == 0:
        raise ValueError("data.target.{train,val,test} must all be non-empty")

    target_latent_field = str(target_cfg.get("latent_field", "mu"))
    target_dtype = str(target_cfg.get("dataset_dtype", "float32"))
    target_csv_map = target_cfg.get("csv_map", {})
    direct_pairing = bool(target_cfg.get("direct_pairing", False))
    direct_pairing_prefilter = bool(target_cfg.get("prefilter_bad_samples", True))
    seed = int(cfg.get("seed", 42))

    metadata_table = _build_metadata_table(cond_cfg.get("metadata", {}))
    metadata_required = bool(cond_cfg.get("metadata", {}).get("required", False))
    missing_required_policy = str(data_cfg.get("missing_required_policy", "drop")).strip().lower()
    if missing_required_policy not in {"drop", "error"}:
        raise ValueError("data.missing_required_policy must be 'drop' or 'error'")
    verify_paths = bool(data_cfg.get("verify_paths", True))

    datasets: Dict[str, ConditionalLatentPairDataset] = {}
    audits: Dict[str, SplitAudit] = {}
    split_roots = {"train": train_roots, "val": val_roots, "test": test_roots}

    _progress(
        f"build_splits direct_pairing={direct_pairing} prefilter_bad_samples={direct_pairing_prefilter} "
        f"verify_paths={verify_paths} missing_required_policy={missing_required_policy} "
        f"train_roots={len(train_roots)} val_roots={len(val_roots)} test_roots={len(test_roots)}"
    )

    for idx, split in enumerate(("train", "val", "test")):
        if direct_pairing:
            _progress(f"{split}: discovering direct pair samples from {len(split_roots[split])} csv roots")
            target_records = []
            raw_pair_samples = _discover_direct_pair_samples(
                split_roots[split],
                csv_map=target_csv_map,
                verify_exists=verify_paths,
            )
            if len(raw_pair_samples) == 0:
                raise ValueError(f"No direct pair samples discovered for split={split}")
            _progress(f"{split}: discovered {len(raw_pair_samples)} direct pair samples")
            fc_store = _empty_condition_store("fc", cond_cfg.get("fc", {}))
            mri_store = _empty_condition_store("mri", cond_cfg.get("mri", {}))
            video_store = _empty_condition_store("video", cond_cfg.get("video", {}))
            audio_store = _empty_condition_store("audio", cond_cfg.get("audio", {}))
            if direct_pairing_prefilter:
                raw_pair_samples, dropped_direct = _prefilter_direct_pair_samples(
                    raw_pair_samples,
                    target_latent_field=target_latent_field,
                    target_dtype=target_dtype,
                    fc_store=fc_store,
                    mri_store=mri_store,
                    video_store=video_store,
                    audio_store=audio_store,
                )
                _progress(f"{split}: prefiltered direct pair samples kept={len(raw_pair_samples)} dropped={dropped_direct}")
            if fc_store.enabled:
                fc_store.sample_shape = _probe_feature_shape_from_pair_samples(
                    raw_pair_samples,
                    "fc_array_path",
                    fc_store.array_field,
                    fc_store.dtype,
                )
            if mri_store.enabled:
                mri_store.sample_shape = _probe_feature_shape_from_pair_samples(
                    raw_pair_samples,
                    "mri_array_path",
                    mri_store.array_field,
                    mri_store.dtype,
                )
            if video_store.enabled:
                video_store.sample_shape = _probe_feature_shape_from_pair_samples(
                    raw_pair_samples,
                    "video_array_path",
                    video_store.array_field,
                    video_store.dtype,
                )
            if audio_store.enabled:
                audio_store.sample_shape = _probe_feature_shape_from_pair_samples(
                    raw_pair_samples,
                    "audio_array_path",
                    audio_store.array_field,
                    audio_store.dtype,
                )
        else:
            _progress(f"{split}: discovering target records from {len(split_roots[split])} csv roots")
            target_records = _discover_target_records(
                split_roots[split],
                csv_map=target_csv_map,
                verify_exists=verify_paths,
            )
            if len(target_records) == 0:
                raise ValueError(f"No target latents discovered for split={split}")
            _progress(f"{split}: discovered {len(target_records)} target records")

            raw_pair_samples = _build_pair_samples(target_records, cfg=cfg, split=split, seed=seed + idx * 97)
            _progress(f"{split}: built {len(raw_pair_samples)} raw pair samples")
            fc_store = _build_condition_store("fc", cond_cfg.get("fc", {}), split, verify_exists=verify_paths)
            mri_store = _build_condition_store("mri", cond_cfg.get("mri", {}), split, verify_exists=verify_paths)
            video_store = _build_condition_store("video", cond_cfg.get("video", {}), split, verify_exists=verify_paths)
            audio_store = _build_condition_store("audio", cond_cfg.get("audio", {}), split, verify_exists=verify_paths)
        _progress(
            f"{split}: feature stores ready fc_records={len(fc_store.records)} mri_records={len(mri_store.records)} "
            f"video_records={len(video_store.records)} audio_records={len(audio_store.records)} "
            f"fc_shape={fc_store.sample_shape} mri_shape={mri_store.sample_shape} "
            f"video_shape={video_store.sample_shape} audio_shape={audio_store.sample_shape}"
        )
        dropped_required_pairs = {"fc": 0, "mri": 0, "video": 0, "audio": 0, "metadata": 0}

        if missing_required_policy == "drop":
            pair_samples, dropped_required_pairs = _filter_pair_samples_missing_required(
                raw_pair_samples,
                fc_store=fc_store,
                mri_store=mri_store,
                video_store=video_store,
                audio_store=audio_store,
                metadata_table=metadata_table,
                metadata_required=metadata_required,
            )
        else:
            pair_samples = list(raw_pair_samples)

        audit = _audit_split(
            split=split,
            target_records=target_records,
            pair_samples=pair_samples,
            fc_store=fc_store,
            mri_store=mri_store,
            video_store=video_store,
            audio_store=audio_store,
            metadata_table=metadata_table,
            metadata_required=metadata_required,
            dropped_required_pairs=dropped_required_pairs,
        )
        if missing_required_policy == "error":
            for name, count in audit.condition_required_missing.items():
                if count > 0:
                    raise ValueError(f"Required condition missing on {count} samples for split={split}, modality={name}")
        elif len(pair_samples) == 0:
            raise ValueError(
                f"All pair samples were dropped due to required-condition missing on split={split}; "
                f"dropped={dropped_required_pairs}, raw_pairs={len(raw_pair_samples)}"
            )

        _progress(
            f"{split}: final pair samples={len(pair_samples)} dropped_required={dropped_required_pairs} "
            f"required_missing={audit.condition_required_missing}"
        )

        datasets[split] = ConditionalLatentPairDataset(
            split=split,
            pair_samples=pair_samples,
            target_latent_field=target_latent_field,
            target_dtype=target_dtype,
            fc_store=fc_store,
            mri_store=mri_store,
            video_store=video_store,
            audio_store=audio_store,
            metadata_table=metadata_table,
        )
        audits[split] = audit

    return datasets["train"], datasets["val"], datasets["test"], audits


def collate_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {
        "target_latent": torch.stack([b["target_latent"] for b in batch], dim=0),
        "direction_id": torch.tensor([int(b["direction_id"]) for b in batch], dtype=torch.long),
        "direction": [str(b["direction"]) for b in batch],
        "subject_id": [str(b["subject_id"]) for b in batch],
        "sequence_id": [str(b["sequence_id"]) for b in batch],
        "anchor_chunk_id": torch.tensor([int(b["anchor_chunk_id"]) for b in batch], dtype=torch.long),
        "target_chunk_id": torch.tensor([int(b["target_chunk_id"]) for b in batch], dtype=torch.long),
        "target_npz_path": [str(b["target_npz_path"]) for b in batch],
        "target_source_path": [str(b["target_source_path"]) for b in batch],
        "has_fc": torch.tensor([1.0 if bool(b["has_fc"]) else 0.0 for b in batch], dtype=torch.float32),
        "has_mri": torch.tensor([1.0 if bool(b["has_mri"]) else 0.0 for b in batch], dtype=torch.float32),
        "has_video": torch.tensor([1.0 if bool(b["has_video"]) else 0.0 for b in batch], dtype=torch.float32),
        "has_audio": torch.tensor([1.0 if bool(b["has_audio"]) else 0.0 for b in batch], dtype=torch.float32),
        "has_meta": torch.tensor([1.0 if bool(b["has_meta"]) else 0.0 for b in batch], dtype=torch.float32),
    }

    fc0 = batch[0]["fc_cond"]
    if isinstance(fc0, torch.Tensor) and fc0.numel() > 0:
        out["fc_cond"] = torch.stack([b["fc_cond"] for b in batch], dim=0)

    mri0 = batch[0]["mri_cond"]
    if isinstance(mri0, torch.Tensor) and mri0.numel() > 0:
        out["mri_cond"] = torch.stack([b["mri_cond"] for b in batch], dim=0)

    video0 = batch[0]["video_cond"]
    if isinstance(video0, torch.Tensor) and video0.numel() > 0:
        out["video_cond"] = torch.stack([b["video_cond"] for b in batch], dim=0)

    audio0 = batch[0]["audio_cond"]
    if isinstance(audio0, torch.Tensor) and audio0.numel() > 0:
        out["audio_cond"] = torch.stack([b["audio_cond"] for b in batch], dim=0)

    meta0 = batch[0]["meta_cond"]
    if isinstance(meta0, torch.Tensor) and meta0.numel() > 0:
        out["meta_cond"] = torch.stack([b["meta_cond"] for b in batch], dim=0)

    return out


def audit_to_dict(a: SplitAudit) -> Dict[str, object]:
    return {
        "split": a.split,
        "num_target_latents": int(a.num_target_latents),
        "num_pair_samples": int(a.num_pair_samples),
        "num_subjects": int(a.num_subjects),
        "direction_counts": {k: int(v) for k, v in a.direction_counts.items()},
        "target_preview": list(a.target_preview),
        "pair_preview": list(a.pair_preview),
        "condition_shape": {
            k: (None if v is None else [int(x) for x in v])
            for k, v in a.condition_shape.items()
        },
        "condition_available": {k: int(v) for k, v in a.condition_available.items()},
        "condition_required_missing": {k: int(v) for k, v in a.condition_required_missing.items()},
        "dropped_required_pairs": {k: int(v) for k, v in a.dropped_required_pairs.items()},
    }
