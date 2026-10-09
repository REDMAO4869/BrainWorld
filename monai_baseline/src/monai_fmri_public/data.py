from __future__ import annotations

import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def load_latent_array(path: str | Path, *, npz_preferred_keys: tuple[str, ...] = ("mu", "arr_0", "arr", "data", "x")) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"latent file not found: {p}")

    if p.suffix == ".npy":
        arr = np.asarray(np.load(p), dtype=np.float32)
        return arr

    if p.suffix == ".npz":
        with np.load(p) as data:
            picked = None
            for key in npz_preferred_keys:
                if key in data:
                    picked = key
                    break
            if picked is None:
                keys = list(data.keys())
                if len(keys) == 0:
                    raise ValueError(f"Empty npz latent file: {p}")
                picked = keys[0]
            arr = np.asarray(data[picked], dtype=np.float32)
        return arr

    raise ValueError(f"Unsupported latent file type: {p}")


def load_clip_array(path: str | Path, image_key: str = "data", layout: str = "xyzt") -> np.ndarray:
    path = Path(path)
    suffix = "".join(path.suffixes)
    if suffix.endswith(".nii") or suffix.endswith(".nii.gz"):
        import nibabel as nib

        array = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
    elif path.suffix == ".npz":
        with np.load(path) as data:
            if image_key not in (None, "", "auto") and image_key in data:
                array = np.asarray(data[image_key], dtype=np.float32)
            else:
                picked_key = None
                for candidate in ("data", "arr", "arr_0", "x"):
                    if candidate in data:
                        picked_key = candidate
                        break
                if picked_key is None:
                    keys = list(data.keys())
                    if not keys:
                        raise ValueError(f"Empty npz file: {path}")
                    picked_key = keys[0]
                array = np.asarray(data[picked_key], dtype=np.float32)
    elif path.suffix == ".npy":
        array = np.asarray(np.load(path), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    if array.ndim != 4:
        raise ValueError(f"Expected 4D clip, got shape {array.shape} from {path}")

    if layout == "xyzt":
        array = np.transpose(array, (3, 0, 1, 2))
    elif layout == "txyz":
        pass
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    return array


def normalize_clip(clip: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return clip
    if mode == "zscore":
        mean = float(clip.mean())
        std = float(clip.std())
        std = std if std > 1e-8 else 1.0
        return (clip - mean) / std
    if mode == "zscore_nonzero":
        mask = clip != 0
        values = clip[mask] if bool(mask.any()) else clip.reshape(-1)
        mean = float(values.mean())
        std = float(values.std())
        std = std if std > 1e-8 else 1.0
        return (clip - mean) / std
    raise ValueError(f"Unsupported normalization: {mode}")


def maybe_resize_clip(clip: np.ndarray, target_spatial_size: list[int] | tuple[int, int, int] | None) -> np.ndarray:
    if target_spatial_size is None:
        return clip
    target_spatial_size = tuple(int(value) for value in target_spatial_size)
    if clip.shape[1:] == target_spatial_size:
        return clip
    tensor = torch.from_numpy(clip[:, None]).float()
    tensor = F.interpolate(tensor, size=target_spatial_size, mode="trilinear", align_corners=False)
    return tensor[:, 0].cpu().numpy()


def maybe_trim_or_pad_frames(clip: np.ndarray, target_num_frames: int | None) -> np.ndarray:
    if target_num_frames is None:
        return clip
    current = clip.shape[0]
    if current == target_num_frames:
        return clip
    if current > target_num_frames:
        start = max((current - target_num_frames) // 2, 0)
        return clip[start : start + target_num_frames]
    pad = np.repeat(clip[-1:, ...], target_num_frames - current, axis=0)
    return np.concatenate([clip, pad], axis=0)


def prepare_clip_from_record(record: dict[str, Any], data_config: dict[str, Any]) -> np.ndarray:
    clip = load_clip_array(
        record["image"],
        image_key=record.get("image_key", data_config.get("image_key", "data")),
        layout=record.get("layout", data_config.get("layout", "xyzt")),
    )
    clip = maybe_trim_or_pad_frames(clip, data_config.get("target_num_frames"))
    clip = maybe_resize_clip(clip, data_config.get("target_spatial_size"))
    clip = normalize_clip(clip, data_config.get("normalization", "zscore_nonzero"))
    return clip.astype(np.float32, copy=False)


def build_task_vocab(records: list[dict[str, Any]]) -> dict[str, int]:
    tasks = sorted({record["task"] for record in records})
    return {task: index for index, task in enumerate(tasks)}


class ClipDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], data_config: dict[str, Any]) -> None:
        if len(records) == 0:
            raise ValueError("ClipDataset received 0 records")
        self.records = records
        self.data_config = data_config
        self.skip_corrupt_files = bool(data_config.get("skip_corrupt_files", False))
        self.skip_corrupt_retry_limit = max(1, int(data_config.get("skip_corrupt_retry_limit", 64)))
        self.skip_corrupt_log_limit = max(0, int(data_config.get("skip_corrupt_log_limit", 20)))
        self._skip_logged = 0

    def __len__(self) -> int:
        return len(self.records)

    def _build_item_from_index(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        clip = prepare_clip_from_record(record, self.data_config)
        tensor = torch.from_numpy(clip[:, None]).float()
        return {
            "clip": tensor,
            "task": record["task"],
            "subject": record.get("subject", "unknown"),
            "session": record.get("session", "unknown"),
            "segment": int(record.get("segment", 0)),
            "path": record["image"],
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        if not self.skip_corrupt_files:
            return self._build_item_from_index(index)

        num_records = len(self.records)
        max_tries = min(num_records, self.skip_corrupt_retry_limit)
        last_error = None
        for offset in range(max_tries):
            probe_index = (index + offset) % num_records
            try:
                return self._build_item_from_index(probe_index)
            except (zipfile.BadZipFile, EOFError, OSError, ValueError) as error:
                last_error = error
                if self._skip_logged < self.skip_corrupt_log_limit:
                    bad_path = self.records[probe_index].get("image", "<unknown>")
                    print(
                        f"[data][warn] skip unreadable sample: {bad_path} "
                        f"({type(error).__name__}: {error})"
                    )
                self._skip_logged += 1
        raise RuntimeError(
            f"Failed to load a readable sample near index={index} after {max_tries} attempts; "
            f"last_error={type(last_error).__name__ if last_error else 'unknown'}: {last_error}"
        )


class RandomFrameDataset(Dataset):
    def __init__(self, clip_dataset: ClipDataset, seed: int = 42) -> None:
        self.clip_dataset = clip_dataset
        self.random = random.Random(seed)

    def __len__(self) -> int:
        return len(self.clip_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        batch = self.clip_dataset[index]
        clip = batch["clip"]
        frame_index = self.random.randrange(clip.shape[0])
        return {
            "image": clip[frame_index],
            "frame_index": frame_index,
            "task": batch["task"],
            "subject": batch["subject"],
            "session": batch["session"],
            "segment": batch["segment"],
            "path": batch["path"],
        }


class LatentCacheDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        task_vocab: dict[str, int],
        *,
        expected_num_frames: int,
        expected_latent_channels: int,
    ) -> None:
        self.records = records
        self.task_vocab = task_vocab
        self.expected_num_frames = expected_num_frames
        self.expected_latent_channels = expected_latent_channels

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        latents = load_latent_array(record["latent_path"])
        if latents.ndim != 5:
            raise ValueError(f"Expected latent tensor [T,C,D,H,W], got {latents.shape}")
        num_frames, latent_channels = latents.shape[:2]
        if num_frames != self.expected_num_frames:
            raise ValueError(
                f"Latent frame mismatch for {record['latent_path']}: {num_frames} != {self.expected_num_frames}"
            )
        if latent_channels != self.expected_latent_channels:
            raise ValueError(
                f"Latent channel mismatch for {record['latent_path']}: {latent_channels} != {self.expected_latent_channels}"
            )
        stacked = latents.reshape(num_frames * latent_channels, *latents.shape[2:])
        task = record["task"]
        return {
            "latent_stacked": torch.from_numpy(stacked).float(),
            "task": task,
            "task_id": torch.tensor(self.task_vocab[task], dtype=torch.long),
            "subject": record.get("subject", "unknown"),
            "session": record.get("session", "unknown"),
            "segment": int(record.get("segment", 0)),
            "path": record["latent_path"],
        }


_SEGMENT_RANGE_RE = re.compile(r"_(\d+)-(\d+)$")


def _source_stem_without_range(source_path: str) -> str:
    path = Path(source_path)
    name = path.name
    if name.endswith('.nii.gz'):
        stem = name[:-7]
    elif path.suffix in {'.npz', '.npy', '.nii'}:
        stem = path.stem
    else:
        stem = name
    m = _SEGMENT_RANGE_RE.search(stem)
    if m is not None:
        stem = stem[: m.start()]
    return stem


class PairLatentCacheDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        task_vocab: dict[str, int],
        *,
        expected_num_frames: int,
        expected_latent_channels: int,
        pair_mode: str = 'next_only',
        split: str = 'train',
        fc_config: dict[str, Any] | None = None,
    ) -> None:
        if pair_mode != 'next_only':
            raise ValueError(f'Only pair_mode=next_only is supported, got {pair_mode}')
        self.records = records
        self.task_vocab = task_vocab
        self.expected_num_frames = int(expected_num_frames)
        self.expected_latent_channels = int(expected_latent_channels)
        self.pair_mode = str(pair_mode)
        self.split = str(split)

        fc_cfg = fc_config or {}
        self.fc_required = bool(fc_cfg.get('required', True))
        self.fc_path_field = str(fc_cfg.get('path_field', 'fc_path'))
        self.fc_npz_key = str(fc_cfg.get('npz_key', 'auto'))
        self.fc_dim_hint = int(fc_cfg.get('dim_hint', 0) or 0)

        self.pairs: list[tuple[dict[str, Any], dict[str, Any]]] = self._build_pairs(records)
        if len(self.pairs) == 0:
            raise ValueError(f'PairLatentCacheDataset split={self.split} has 0 pairs')

        self.fc_dim = self._probe_fc_dim()
        if self.fc_required and self.fc_dim <= 0:
            raise ValueError(
                f"PairLatentCacheDataset split={self.split} requires external fc features, "
                f"but no valid fc vectors were found via field '{self.fc_path_field}'"
            )

    def _build_pairs(self, records: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in records:
            dataset = str(row.get('dataset', 'unknown'))
            subject = str(row.get('subject', 'unknown'))
            session = str(row.get('session', 'unknown'))
            source_image = str(row.get('source_image', row.get('latent_path', '')))
            sequence = _source_stem_without_range(source_image)
            key = (dataset, subject, session, sequence)
            grouped[key].append(row)

        out: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for _, items in grouped.items():
            seq = sorted(
                items,
                key=lambda r: (
                    int(r.get('segment', 0)),
                    str(r.get('source_image', r.get('latent_path', ''))),
                ),
            )
            for idx in range(len(seq) - 1):
                anchor = seq[idx]
                target = seq[idx + 1]
                if int(target.get('segment', 0)) <= int(anchor.get('segment', 0)):
                    continue
                out.append((anchor, target))
        return out

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_stacked(self, record: dict[str, Any]) -> np.ndarray:
        latents = load_latent_array(record["latent_path"])
        if latents.ndim != 5:
            raise ValueError(f"Expected latent tensor [T,C,D,H,W], got {latents.shape}")
        num_frames, latent_channels = latents.shape[:2]
        if num_frames != self.expected_num_frames:
            raise ValueError(
                f"Latent frame mismatch for {record['latent_path']}: {num_frames} != {self.expected_num_frames}"
            )
        if latent_channels != self.expected_latent_channels:
            raise ValueError(
                f"Latent channel mismatch for {record['latent_path']}: {latent_channels} != {self.expected_latent_channels}"
            )
        return latents.reshape(num_frames * latent_channels, *latents.shape[2:])

    def _resolve_fc_path(self, anchor_record: dict[str, Any], target_record: dict[str, Any]) -> str:
        # Prefer anchor record field; fallback to target for compatibility.
        anchor_value = str(anchor_record.get(self.fc_path_field, '')).strip()
        if anchor_value:
            return anchor_value
        target_value = str(target_record.get(self.fc_path_field, '')).strip()
        if target_value:
            return target_value
        return ''

    def _load_fc_vector_from_path(self, fc_path: str) -> np.ndarray:
        path = Path(fc_path)
        if not path.exists():
            raise FileNotFoundError(f"fc file not found: {fc_path}")
        if path.suffix == '.npy':
            arr = np.asarray(np.load(path), dtype=np.float32)
        elif path.suffix == '.npz':
            with np.load(path) as data:
                if self.fc_npz_key not in {'', 'auto'} and self.fc_npz_key in data:
                    arr = np.asarray(data[self.fc_npz_key], dtype=np.float32)
                else:
                    keys = list(data.keys())
                    if len(keys) == 0:
                        raise ValueError(f"Empty npz fc file: {fc_path}")
                    arr = np.asarray(data[keys[0]], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported fc file type: {fc_path}")

        if arr.ndim == 0:
            arr = arr.reshape(1)
        arr = arr.reshape(-1).astype(np.float32, copy=False)
        return arr

    def _probe_fc_dim(self) -> int:
        if self.fc_dim_hint > 0:
            return int(self.fc_dim_hint)
        for anchor_record, target_record in self.pairs:
            fc_path = self._resolve_fc_path(anchor_record, target_record)
            if not fc_path:
                continue
            try:
                return int(self._load_fc_vector_from_path(fc_path).shape[0])
            except Exception:
                continue
        return 0

    def _load_fc_vector(self, anchor_record: dict[str, Any], target_record: dict[str, Any]) -> tuple[np.ndarray, bool]:
        fc_path = self._resolve_fc_path(anchor_record, target_record)
        if not fc_path:
            if self.fc_required:
                raise ValueError(
                    f"Missing external fc path field '{self.fc_path_field}' for anchor={anchor_record.get('latent_path')}"
                )
            return np.zeros((self.fc_dim,), dtype=np.float32), False

        vec = self._load_fc_vector_from_path(fc_path)
        if self.fc_dim <= 0:
            self.fc_dim = int(vec.shape[0])
        if int(vec.shape[0]) != int(self.fc_dim):
            raise ValueError(
                f"Inconsistent fc dim for {fc_path}: got {vec.shape[0]} expected {self.fc_dim}"
            )
        return vec, True

    def __getitem__(self, index: int) -> dict[str, Any]:
        anchor_record, target_record = self.pairs[int(index)]
        anchor_stacked = self._load_stacked(anchor_record)
        target_stacked = self._load_stacked(target_record)
        fc_vec, has_fc = self._load_fc_vector(anchor_record, target_record)
        task = str(target_record['task'])
        return {
            'anchor_latent_stacked': torch.from_numpy(anchor_stacked).float(),
            'target_latent_stacked': torch.from_numpy(target_stacked).float(),
            'fc_cond': torch.from_numpy(fc_vec).float(),
            'has_fc': bool(has_fc),
            'task': task,
            'task_id': torch.tensor(self.task_vocab[task], dtype=torch.long),
            'subject': target_record.get('subject', 'unknown'),
            'session': target_record.get('session', 'unknown'),
            'anchor_segment': int(anchor_record.get('segment', 0)),
            'target_segment': int(target_record.get('segment', 0)),
            'anchor_path': anchor_record['latent_path'],
            'target_path': target_record['latent_path'],
        }


class UniversalSplitPairDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        task_vocab: dict[str, int],
        *,
        expected_num_frames: int,
        expected_latent_channels: int,
        fc_config: dict[str, Any] | None = None,
        fc_missing_policy: str = 'drop',
        split: str = 'train',
    ) -> None:
        self.rows = rows
        self.task_vocab = task_vocab
        self.expected_num_frames = int(expected_num_frames)
        self.expected_latent_channels = int(expected_latent_channels)
        self.split = str(split)

        fc_cfg = fc_config or {}
        self.fc_required = bool(fc_cfg.get('required', True))
        self.fc_npz_key = str(fc_cfg.get('npz_key', 'auto'))
        self.fc_dim_hint = int(fc_cfg.get('dim_hint', 0) or 0)
        self.fc_missing_policy = str(fc_missing_policy).strip().lower()
        if self.fc_missing_policy not in {'drop', 'zero', 'error'}:
            raise ValueError(f'Unsupported fc_missing_policy: {self.fc_missing_policy}')

        filtered = []
        dropped_missing_fc = 0
        dropped_missing_target = 0
        for row in self.rows:
            latent_path = str(row.get('latent_path', '')).strip()
            target_path = str(row.get('target_latent_path', '')).strip()
            if not latent_path or not target_path:
                dropped_missing_target += 1
                continue
            fc_path = str(row.get('fc_embedding_path', '')).strip()
            if self.fc_missing_policy == 'drop' and not fc_path:
                dropped_missing_fc += 1
                continue
            filtered.append({
                'subject': str(row.get('Subject', row.get('subject', 'unknown'))),
                'task': str(row.get('task', 'rest')),
                'session': str(row.get('session', 'unknown')),
                'dataset': str(row.get('dataset', 'unknown')),
                'anchor_path': latent_path,
                'target_path': target_path,
                'fc_path': fc_path,
            })

        self.samples = filtered
        if len(self.samples) == 0:
            raise ValueError(
                f'UniversalSplitPairDataset split={self.split} has 0 samples after filtering '
                f'(drop_missing_target={dropped_missing_target}, drop_missing_fc={dropped_missing_fc})'
            )

        self.fc_dim = self._probe_fc_dim()
        if self.fc_required and self.fc_missing_policy == 'error' and self.fc_dim <= 0:
            raise ValueError(f'split={self.split} requires external fc but no valid fc vectors were found')

    def __len__(self) -> int:
        return len(self.samples)

    def _load_stacked(self, latent_path: str) -> np.ndarray:
        latents = load_latent_array(latent_path)
        if latents.ndim != 5:
            raise ValueError(f'Expected latent tensor [T,C,D,H,W], got {latents.shape} from {latent_path}')
        num_frames, latent_channels = latents.shape[:2]
        if num_frames != self.expected_num_frames:
            raise ValueError(f'Latent frame mismatch for {latent_path}: {num_frames} != {self.expected_num_frames}')
        if latent_channels != self.expected_latent_channels:
            raise ValueError(f'Latent channel mismatch for {latent_path}: {latent_channels} != {self.expected_latent_channels}')
        return latents.reshape(num_frames * latent_channels, *latents.shape[2:])

    def _load_fc_vector_from_path(self, fc_path: str) -> np.ndarray:
        path = Path(fc_path)
        if not path.exists():
            raise FileNotFoundError(f'fc file not found: {fc_path}')
        if path.suffix == '.npy':
            arr = np.asarray(np.load(path), dtype=np.float32)
        elif path.suffix == '.npz':
            with np.load(path) as data:
                if self.fc_npz_key not in {'', 'auto'} and self.fc_npz_key in data:
                    arr = np.asarray(data[self.fc_npz_key], dtype=np.float32)
                else:
                    keys = list(data.keys())
                    if len(keys) == 0:
                        raise ValueError(f'Empty npz fc file: {fc_path}')
                    arr = np.asarray(data[keys[0]], dtype=np.float32)
        else:
            raise ValueError(f'Unsupported fc file type: {fc_path}')
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return arr.reshape(-1).astype(np.float32, copy=False)

    def _probe_fc_dim(self) -> int:
        if self.fc_dim_hint > 0:
            return int(self.fc_dim_hint)
        for sample in self.samples:
            fc_path = sample['fc_path']
            if not fc_path:
                continue
            try:
                return int(self._load_fc_vector_from_path(fc_path).shape[0])
            except Exception:
                continue
        return 0

    def _load_fc_vector(self, sample: dict[str, Any]) -> tuple[np.ndarray, bool]:
        fc_path = str(sample.get('fc_path', '')).strip()
        if not fc_path:
            if self.fc_missing_policy == 'error' and self.fc_required:
                raise ValueError(f"Missing external fc path for anchor={sample.get('anchor_path')}")
            return np.zeros((self.fc_dim,), dtype=np.float32), False

        vec = self._load_fc_vector_from_path(fc_path)
        if self.fc_dim <= 0:
            self.fc_dim = int(vec.shape[0])
        if int(vec.shape[0]) != int(self.fc_dim):
            raise ValueError(f'Inconsistent fc dim for {fc_path}: got {vec.shape[0]} expected {self.fc_dim}')
        return vec, True

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[int(index)]
        anchor_stacked = self._load_stacked(sample['anchor_path'])
        target_stacked = self._load_stacked(sample['target_path'])
        fc_vec, has_fc = self._load_fc_vector(sample)
        task = str(sample['task'])
        return {
            'anchor_latent_stacked': torch.from_numpy(anchor_stacked).float(),
            'target_latent_stacked': torch.from_numpy(target_stacked).float(),
            'fc_cond': torch.from_numpy(fc_vec).float(),
            'has_fc': bool(has_fc),
            'task': task,
            'task_id': torch.tensor(self.task_vocab[task], dtype=torch.long),
            'subject': sample.get('subject', 'unknown'),
            'session': sample.get('session', 'unknown'),
            'anchor_path': sample['anchor_path'],
            'target_path': sample['target_path'],
            'dataset': sample.get('dataset', 'unknown'),
        }
