from __future__ import annotations

import csv
import math
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


VALID_EXTS = {".npy", ".npz"}


@dataclass(frozen=True)
class DatasetAudit:
    split: str
    num_files: int
    roots: List[str]
    sample_preview: List[str]
    requested_ratio: float = 1.0
    num_files_before_subset: int = 0
    num_corrupt_dropped: int = 0


def _sanitize_token(v: str) -> str:
    out = []
    for c in str(v):
        if c.isalnum() or c in {"_", "-"}:
            out.append(c)
        else:
            out.append("_")
    return "".join(out).strip("_") or "unk"


def _dataset_id_from_root(root: str) -> str:
    p = Path(str(root))
    if p.suffix.lower() == ".csv":
        return _sanitize_token(p.parent.name)
    if p.is_dir():
        return _sanitize_token(p.name)
    if p.is_file():
        return _sanitize_token(p.parent.name)
    return "unknown"


def _load_array(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        arr = np.load(path)
        if not isinstance(arr, np.ndarray):
            raise ValueError(f"Expected ndarray from npy, got {type(arr)}: {path}")
        return arr

    if p.suffix == ".npz":
        with np.load(path) as data:
            keys = list(data.keys())
            for k in ("arr_0", "arr", "data", "x"):
                if k in data:
                    return data[k]
            if len(keys) == 0:
                raise ValueError(f"Empty npz file: {path}")
            return data[keys[0]]

    raise ValueError(f"Unsupported file extension: {path}")


def _is_readable_npz(path: str) -> Tuple[bool, str]:
    try:
        if not zipfile.is_zipfile(path):
            return False, "not a zip container"
        with np.load(path) as data:
            _ = list(data.keys())
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _drop_corrupt_npz_files(files: Sequence[str], *, split: str, log_limit: int) -> Tuple[List[str], List[str]]:
    kept: List[str] = []
    dropped: List[str] = []
    lim = max(0, int(log_limit))

    for fp in files:
        if Path(fp).suffix.lower() != ".npz":
            kept.append(fp)
            continue

        ok, reason = _is_readable_npz(fp)
        if ok:
            kept.append(fp)
            continue

        dropped.append(fp)
        if len(dropped) <= lim:
            print(f"[data][warn] split={split} dropped corrupt npz: {fp} ({reason})")

    if len(dropped) > 0:
        print(f"[data][warn] split={split} dropped {len(dropped)} corrupt npz files before dataset build")
    return kept, dropped


def _to_tdhw(arr: np.ndarray, layout: str) -> np.ndarray:
    layout = str(layout).upper()
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D array, got shape={arr.shape}")

    if layout == "DHWT":
        out = np.transpose(arr, (3, 0, 1, 2))
    elif layout == "TDHW":
        out = arr
    elif layout == "THWD":
        out = np.transpose(arr, (0, 3, 1, 2))
    else:
        raise ValueError(f"Unsupported layout={layout}, expected DHWT/TDHW/THWD")

    return out.astype(np.float32, copy=False)


def _crop_time(x: np.ndarray, t_frames: Optional[int], mode: str, rng: random.Random) -> np.ndarray:
    if t_frames is None:
        return x

    t_frames = int(t_frames)
    if x.shape[0] == t_frames:
        return x
    if x.shape[0] < t_frames:
        pad = np.zeros((t_frames - x.shape[0],) + x.shape[1:], dtype=x.dtype)
        return np.concatenate([x, pad], axis=0)

    mode = str(mode).lower()
    if mode == "head":
        return x[:t_frames]
    if mode == "center":
        s = (x.shape[0] - t_frames) // 2
        return x[s : s + t_frames]
    if mode == "random":
        s = rng.randint(0, x.shape[0] - t_frames)
        return x[s : s + t_frames]

    raise ValueError(f"Unsupported temporal crop mode={mode}")


def _normalize(x: np.ndarray, mode: str, fg_thr: float) -> np.ndarray:
    mode = str(mode).lower()
    if mode in ("none", ""):
        return x

    if mode == "zscore":
        m = float(x.mean())
        s = float(x.std())
        s = 1.0 if s < 1.0e-6 else s
        return (x - m) / s

    if mode == "zscore_fg":
        fg = np.abs(x) > float(fg_thr)
        if fg.any():
            vals = x[fg]
            m = float(vals.mean())
            s = float(vals.std())
            s = 1.0 if s < 1.0e-6 else s
            y = (x - m) / s
            y[~fg] = 0.0
            return y
        return x

    if mode == "robust":
        p1 = float(np.percentile(x, 1))
        p99 = float(np.percentile(x, 99))
        if abs(p99 - p1) < 1.0e-6:
            return x - p1
        y = (x - p1) / (p99 - p1)
        return y * 2.0 - 1.0

    raise ValueError(f"Unsupported normalize mode={mode}")


def _parse_bbox_crop(data_cfg: Dict) -> Optional[Tuple[int, int, int, int, int, int]]:
    bc = data_cfg.get("bbox_crop", {})
    if not isinstance(bc, dict):
        return None
    if not bool(bc.get("enabled", False)):
        return None

    def _pair(name: str) -> Tuple[int, int]:
        v = bc.get(name, None)
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ValueError(f"data.bbox_crop.{name} must be [start,end]")
        a, b = int(v[0]), int(v[1])
        if a < 0 or b <= a:
            raise ValueError(f"invalid data.bbox_crop.{name}={v}")
        return a, b

    z0, z1 = _pair("z")
    y0, y1 = _pair("y")
    x0, x1 = _pair("x")
    return (z0, z1, y0, y1, x0, x1)


def _apply_bbox_tdhw(x: np.ndarray, bbox: Optional[Tuple[int, int, int, int, int, int]]) -> np.ndarray:
    if bbox is None:
        return x
    z0, z1, y0, y1, x0, x1 = bbox
    if not (0 <= z0 < z1 <= x.shape[1] and 0 <= y0 < y1 <= x.shape[2] and 0 <= x0 < x1 <= x.shape[3]):
        raise ValueError(f"bbox {bbox} out of range for TDHW shape={tuple(x.shape)}")
    return x[:, z0:z1, y0:y1, x0:x1]


def _apply_bbox_dhw(m: np.ndarray, bbox: Optional[Tuple[int, int, int, int, int, int]]) -> np.ndarray:
    if bbox is None:
        return m
    z0, z1, y0, y1, x0, x1 = bbox
    if not (0 <= z0 < z1 <= m.shape[0] and 0 <= y0 < y1 <= m.shape[1] and 0 <= x0 < x1 <= m.shape[2]):
        raise ValueError(f"bbox {bbox} out of range for DHW shape={tuple(m.shape)}")
    return m[z0:z1, y0:y1, x0:x1]


def _discover_from_csv(csv_path: str, exts: Sequence[str]) -> List[str]:
    allow = {e.lower() for e in exts}
    files: List[str] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) == 0:
        return []

    header = [c.strip().lower() for c in rows[0]]
    start_idx = 0
    path_col = None
    if "path" in header:
        path_col = header.index("path")
        start_idx = 1
    elif len(rows[0]) >= 2 and rows[0][1].strip().lower() == "path":
        path_col = 1
        start_idx = 1

    for row in rows[start_idx:]:
        if len(row) == 0:
            continue
        fp = ""
        if path_col is not None and len(row) > path_col:
            fp = row[path_col].strip()
        elif len(row) >= 2:
            fp = row[1].strip()
        elif len(row) == 1:
            fp = row[0].strip()
        if fp == "":
            continue
        p = Path(fp)
        if not p.exists():
            continue

        if p.is_file() and p.suffix.lower() in allow:
            files.append(str(p))
            continue

        if p.is_dir():
            for q in p.rglob("*"):
                if q.is_file() and q.suffix.lower() in allow:
                    files.append(str(q))
            continue

    return sorted(list(dict.fromkeys(files)))


def _collect_files_from_root(root_i: str, *, recursive: bool, exts: Sequence[str]) -> List[str]:
    allow = {e.lower() for e in exts}
    out: List[str] = []
    root = Path(str(root_i))
    if not root.exists():
        return out

    if root.is_file() and root.suffix.lower() == ".csv":
        return _discover_from_csv(str(root), exts)

    if root.is_file() and root.suffix.lower() in allow:
        return [str(root)]

    it = root.rglob("*") if recursive else root.glob("*")
    for p in it:
        if p.is_file() and p.suffix.lower() in allow:
            out.append(str(p))

    return sorted(list(dict.fromkeys(out)))


def discover_files(roots: Sequence[str], recursive: bool = True, exts: Sequence[str] = (".npy", ".npz")) -> List[str]:
    out: List[str] = []
    for r in roots:
        out.extend(_collect_files_from_root(str(r), recursive=recursive, exts=exts))
    return sorted(list(dict.fromkeys(out)))


def discover_files_with_dataset_ids(
    roots: Sequence[str], recursive: bool = True, exts: Sequence[str] = (".npy", ".npz")
) -> Tuple[List[str], Dict[str, str]]:
    files: List[str] = []
    dataset_id_by_path: Dict[str, str] = {}
    for root_i in roots:
        dataset_id = _dataset_id_from_root(str(root_i))
        for fp in _collect_files_from_root(str(root_i), recursive=recursive, exts=exts):
            fp = str(fp)
            if fp in dataset_id_by_path:
                continue
            files.append(fp)
            dataset_id_by_path[fp] = dataset_id

    ordered = sorted(list(dict.fromkeys(files)))
    return ordered, {fp: dataset_id_by_path[fp] for fp in ordered}


def _subset_files(files: Sequence[str], *, ratio: float, seed: int, split: str) -> List[str]:
    ratio = float(ratio)
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError(f"data.{split}_subset_ratio must be in (0,1], got {ratio}")

    out = list(files)
    if ratio >= 1.0 or len(out) <= 1:
        return out

    n_keep = min(len(out), max(1, int(math.ceil(len(out) * ratio))))
    rng = random.Random(int(seed))
    picked = rng.sample(out, k=n_keep)
    return sorted(picked)


def _load_mask(mask_path: str) -> np.ndarray:
    m = _load_array(mask_path)
    if m.ndim == 4:
        if m.shape[0] == 1:
            m = m[0]
        elif m.shape[-1] == 1:
            m = m[..., 0]
        else:
            raise ValueError(f"Unsupported mask shape={m.shape}")
    if m.ndim != 3:
        raise ValueError(f"Mask must be 3D, got shape={m.shape}")
    return (m > 0).astype(np.float32)


class FMRI4DDataset(Dataset):
    def __init__(
        self,
        files: Sequence[str],
        *,
        split: str,
        layout: str = "DHWT",
        t_frames: Optional[int] = None,
        temporal_crop: str = "center",
        normalize: str = "none",
        fg_threshold: float = 1.0e-6,
        seed: int = 0,
        brain_mask: Optional[np.ndarray] = None,
        bbox_crop: Optional[Tuple[int, int, int, int, int, int]] = None,
        dataset_id_by_path: Optional[Dict[str, str]] = None,
        skip_corrupt_files: bool = False,
        skip_corrupt_retry_limit: int = 64,
        skip_corrupt_log_limit: int = 20,
    ) -> None:
        if len(files) == 0:
            raise ValueError(f"{split} dataset has 0 files")
        self.files = list(files)
        self.split = str(split)
        self.layout = str(layout)
        self.t_frames = t_frames
        self.temporal_crop = str(temporal_crop)
        self.normalize_mode = str(normalize)
        self.fg_threshold = float(fg_threshold)
        self.rng = random.Random(int(seed))
        self.bbox_crop = bbox_crop
        self.brain_mask = _apply_bbox_dhw(brain_mask, bbox_crop) if brain_mask is not None else None
        self.dataset_id_by_path = dict(dataset_id_by_path or {})
        self.skip_corrupt_files = bool(skip_corrupt_files)
        self.skip_corrupt_retry_limit = max(1, int(skip_corrupt_retry_limit))
        self.skip_corrupt_log_limit = max(0, int(skip_corrupt_log_limit))
        self._skip_logged = 0

    def __len__(self) -> int:
        return len(self.files)

    def _build_item(self, path: str, arr: np.ndarray) -> Dict[str, torch.Tensor | str | None]:
        x = _to_tdhw(arr, self.layout)
        x = _apply_bbox_tdhw(x, self.bbox_crop)
        x = _crop_time(x, self.t_frames, self.temporal_crop, self.rng)
        x = _normalize(x, self.normalize_mode, self.fg_threshold)

        x_t = torch.from_numpy(x).float().unsqueeze(0)
        mask_t = None
        if self.brain_mask is not None:
            mask_t = torch.from_numpy(self.brain_mask).float().unsqueeze(0)

        return {"x": x_t, "mask": mask_t, "path": path}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | None]:
        idx = int(idx)
        path = self.files[idx]

        if not self.skip_corrupt_files:
            arr = _load_array(path)
            return self._build_item(path, arr)

        n_files = len(self.files)
        n_try = min(n_files, self.skip_corrupt_retry_limit)
        last_err: Optional[Exception] = None

        for offset in range(n_try):
            path_try = self.files[(idx + offset) % n_files]
            try:
                arr = _load_array(path_try)
            except (zipfile.BadZipFile, EOFError, OSError, ValueError) as e:
                last_err = e
                if self._skip_logged < self.skip_corrupt_log_limit:
                    print(
                        f"[data][warn] split={self.split} skip unreadable sample: {path_try} "
                        f"({type(e).__name__}: {e})"
                    )
                self._skip_logged += 1
                continue
            return self._build_item(path_try, arr)

        raise RuntimeError(
            f"{self.split} failed to load any readable sample near index={idx} after {n_try} attempts; "
            f"last error: {type(last_err).__name__ if last_err is not None else 'unknown'}: {last_err}"
        )


def build_splits_from_config(cfg: Dict) -> Tuple[FMRI4DDataset, FMRI4DDataset, FMRI4DDataset, Dict[str, DatasetAudit]]:
    data_cfg = cfg.get("data", {})

    recursive = bool(data_cfg.get("recursive", True))
    exts = data_cfg.get("extensions", [".npy", ".npz"])
    layout = str(data_cfg.get("layout", "DHWT"))
    t_frames = data_cfg.get("t_frames", None)
    temporal_crop_train = str(data_cfg.get("temporal_crop_train", "random"))
    temporal_crop_eval = str(data_cfg.get("temporal_crop_eval", "center"))
    normalize = str(data_cfg.get("normalize", "none"))
    fg_threshold = float(data_cfg.get("fg_threshold", 1.0e-6))
    train_subset_ratio = float(data_cfg.get("train_subset_ratio", 1.0))
    skip_corrupt_files = bool(data_cfg.get("skip_corrupt_files", False))
    precheck_corrupt_files = bool(data_cfg.get("precheck_corrupt_files", True))
    skip_corrupt_retry_limit = int(data_cfg.get("skip_corrupt_retry_limit", 64))
    skip_corrupt_log_limit = int(data_cfg.get("skip_corrupt_log_limit", 20))
    seed = int(cfg.get("seed", 0))
    bbox_crop = _parse_bbox_crop(data_cfg)

    train_roots = list(data_cfg.get("train", []))
    val_roots = list(data_cfg.get("val", []))
    test_roots = list(data_cfg.get("test", []))

    train_files_all = discover_files(train_roots, recursive=recursive, exts=exts)
    val_files = discover_files(val_roots, recursive=recursive, exts=exts)
    test_files = discover_files(test_roots, recursive=recursive, exts=exts)
    train_files = _subset_files(train_files_all, ratio=train_subset_ratio, seed=seed + 101, split="train")

    train_corrupt_dropped = 0
    val_corrupt_dropped = 0
    test_corrupt_dropped = 0
    if skip_corrupt_files and precheck_corrupt_files:
        train_files, train_dropped = _drop_corrupt_npz_files(
            train_files, split="train", log_limit=skip_corrupt_log_limit
        )
        val_files, val_dropped = _drop_corrupt_npz_files(val_files, split="val", log_limit=skip_corrupt_log_limit)
        test_files, test_dropped = _drop_corrupt_npz_files(
            test_files, split="test", log_limit=skip_corrupt_log_limit
        )
        train_corrupt_dropped = len(train_dropped)
        val_corrupt_dropped = len(val_dropped)
        test_corrupt_dropped = len(test_dropped)

    if len(train_files) == 0:
        raise ValueError("No train files discovered. Check data.train in config JSON.")
    if len(val_files) == 0:
        raise ValueError("No val files discovered. Check data.val in config JSON.")
    if len(test_files) == 0:
        raise ValueError("No test files discovered. Check data.test in config JSON.")

    brain_mask = None
    mask_path = str(data_cfg.get("brain_mask_path", "")).strip()
    if mask_path:
        brain_mask = _load_mask(mask_path)

    ds_train = FMRI4DDataset(
        train_files,
        split="train",
        layout=layout,
        t_frames=t_frames,
        temporal_crop=temporal_crop_train,
        normalize=normalize,
        fg_threshold=fg_threshold,
        seed=seed + 11,
        brain_mask=brain_mask,
        bbox_crop=bbox_crop,
        skip_corrupt_files=skip_corrupt_files,
        skip_corrupt_retry_limit=skip_corrupt_retry_limit,
        skip_corrupt_log_limit=skip_corrupt_log_limit,
    )
    ds_val = FMRI4DDataset(
        val_files,
        split="val",
        layout=layout,
        t_frames=t_frames,
        temporal_crop=temporal_crop_eval,
        normalize=normalize,
        fg_threshold=fg_threshold,
        seed=seed + 23,
        brain_mask=brain_mask,
        bbox_crop=bbox_crop,
        skip_corrupt_files=skip_corrupt_files,
        skip_corrupt_retry_limit=skip_corrupt_retry_limit,
        skip_corrupt_log_limit=skip_corrupt_log_limit,
    )
    ds_test = FMRI4DDataset(
        test_files,
        split="test",
        layout=layout,
        t_frames=t_frames,
        temporal_crop=temporal_crop_eval,
        normalize=normalize,
        fg_threshold=fg_threshold,
        seed=seed + 37,
        brain_mask=brain_mask,
        bbox_crop=bbox_crop,
        skip_corrupt_files=skip_corrupt_files,
        skip_corrupt_retry_limit=skip_corrupt_retry_limit,
        skip_corrupt_log_limit=skip_corrupt_log_limit,
    )

    preview_k = int(data_cfg.get("audit_preview", 5))
    audits = {
        "train": DatasetAudit(
            split="train",
            num_files=len(train_files),
            roots=train_roots,
            sample_preview=train_files[:preview_k],
            requested_ratio=float(train_subset_ratio),
            num_files_before_subset=len(train_files_all),
            num_corrupt_dropped=train_corrupt_dropped,
        ),
        "val": DatasetAudit(
            split="val",
            num_files=len(val_files),
            roots=val_roots,
            sample_preview=val_files[:preview_k],
            requested_ratio=1.0,
            num_files_before_subset=len(val_files),
            num_corrupt_dropped=val_corrupt_dropped,
        ),
        "test": DatasetAudit(
            split="test",
            num_files=len(test_files),
            roots=test_roots,
            sample_preview=test_files[:preview_k],
            requested_ratio=1.0,
            num_files_before_subset=len(test_files),
            num_corrupt_dropped=test_corrupt_dropped,
        ),
    }
    return ds_train, ds_val, ds_test, audits


def build_split_from_config(cfg: Dict, split: str) -> Tuple[FMRI4DDataset, DatasetAudit]:
    split = str(split).lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split={split}, expected train/val/test")

    data_cfg = cfg.get("data", {})

    recursive = bool(data_cfg.get("recursive", True))
    exts = data_cfg.get("extensions", [".npy", ".npz"])
    layout = str(data_cfg.get("layout", "DHWT"))
    t_frames = data_cfg.get("t_frames", None)
    temporal_crop_train = str(data_cfg.get("temporal_crop_train", "random"))
    temporal_crop_eval = str(data_cfg.get("temporal_crop_eval", "center"))
    normalize = str(data_cfg.get("normalize", "none"))
    fg_threshold = float(data_cfg.get("fg_threshold", 1.0e-6))
    train_subset_ratio = float(data_cfg.get("train_subset_ratio", 1.0))
    skip_corrupt_files = bool(data_cfg.get("skip_corrupt_files", False))
    precheck_corrupt_files = bool(data_cfg.get("precheck_corrupt_files", True))
    skip_corrupt_retry_limit = int(data_cfg.get("skip_corrupt_retry_limit", 64))
    skip_corrupt_log_limit = int(data_cfg.get("skip_corrupt_log_limit", 20))
    seed = int(cfg.get("seed", 0))
    bbox_crop = _parse_bbox_crop(data_cfg)

    roots = list(data_cfg.get(split, []))
    files_all, dataset_id_by_path = discover_files_with_dataset_ids(roots, recursive=recursive, exts=exts)

    requested_ratio = 1.0
    files = list(files_all)
    if split == "train":
        requested_ratio = float(train_subset_ratio)
        files = _subset_files(files_all, ratio=requested_ratio, seed=seed + 101, split="train")

    num_corrupt_dropped = 0
    if skip_corrupt_files and precheck_corrupt_files:
        files, dropped = _drop_corrupt_npz_files(files, split=split, log_limit=skip_corrupt_log_limit)
        num_corrupt_dropped = len(dropped)

    dataset_id_by_path = {fp: dataset_id_by_path[fp] for fp in files if fp in dataset_id_by_path}

    if len(files) == 0:
        raise ValueError(f"No {split} files discovered. Check data.{split} in config JSON.")

    brain_mask = None
    mask_path = str(data_cfg.get("brain_mask_path", "")).strip()
    if mask_path:
        brain_mask = _load_mask(mask_path)

    temporal_crop = temporal_crop_train if split == "train" else temporal_crop_eval
    split_seed = {"train": seed + 11, "val": seed + 23, "test": seed + 37}[split]
    dataset = FMRI4DDataset(
        files,
        split=split,
        layout=layout,
        t_frames=t_frames,
        temporal_crop=temporal_crop,
        normalize=normalize,
        fg_threshold=fg_threshold,
        seed=split_seed,
        brain_mask=brain_mask,
        bbox_crop=bbox_crop,
        dataset_id_by_path=dataset_id_by_path,
        skip_corrupt_files=skip_corrupt_files,
        skip_corrupt_retry_limit=skip_corrupt_retry_limit,
        skip_corrupt_log_limit=skip_corrupt_log_limit,
    )

    preview_k = int(data_cfg.get("audit_preview", 5))
    audit = DatasetAudit(
        split=split,
        num_files=len(files),
        roots=roots,
        sample_preview=files[:preview_k],
        requested_ratio=float(requested_ratio),
        num_files_before_subset=len(files_all),
        num_corrupt_dropped=num_corrupt_dropped,
    )
    return dataset, audit


def collate_batch(batch: Sequence[Dict[str, torch.Tensor | str | None]]) -> Dict[str, torch.Tensor | List[str] | None]:
    x = torch.stack([b["x"] for b in batch], dim=0)
    paths = [str(b["path"]) for b in batch]

    masks = [b["mask"] for b in batch]
    if all(m is not None for m in masks):
        mask_t = torch.stack([m for m in masks if m is not None], dim=0)
    else:
        mask_t = None

    return {"x": x, "mask": mask_t, "paths": paths}


def audit_to_dict(a: DatasetAudit) -> Dict:
    return {
        "split": a.split,
        "num_files": int(a.num_files),
        "num_files_before_subset": int(a.num_files_before_subset),
        "num_corrupt_dropped": int(a.num_corrupt_dropped),
        "requested_ratio": float(a.requested_ratio),
        "roots": list(a.roots),
        "sample_preview": list(a.sample_preview),
    }
