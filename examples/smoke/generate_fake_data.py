from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np


def _write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def generate_fake_voxels(runtime_root: Path) -> dict:
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)

    data_root = runtime_root / "data"
    splits_root = data_root / "splits"
    labels_path = data_root / "labels.csv"

    plan = {
        "train": [("sub-001", 0), ("sub-001", 1), ("sub-002", 0), ("sub-002", 1), ("sub-003", 0), ("sub-003", 1), ("sub-004", 0), ("sub-004", 1)],
        "val": [("sub-101", 0), ("sub-101", 1), ("sub-102", 0), ("sub-102", 1)],
        "test": [("sub-201", 0), ("sub-201", 1), ("sub-202", 0), ("sub-202", 1)],
    }
    label_rows = []
    for subject, label in [("sub-001", 0), ("sub-002", 1), ("sub-003", 0), ("sub-004", 1), ("sub-101", 0), ("sub-102", 1), ("sub-201", 0), ("sub-202", 1)]:
        label_rows.append({"Subject": subject, "Label": label})

    rng = np.random.default_rng(42)
    manifest_paths = {}
    for split, items in plan.items():
        rows = []
        split_dir = data_root / "voxels" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for subject, chunk in items:
            arr = rng.normal(loc=float(chunk), scale=0.1, size=(8, 8, 8, 4)).astype(np.float32)
            path = split_dir / f"{subject}__seq-rest__chunk_{chunk:03d}.npy"
            np.save(path, arr)
            rows.append({"path": str(path)})
        csv_path = splits_root / f"{split}.csv"
        _write_csv(csv_path, ["path"], rows)
        manifest_paths[split] = csv_path

    _write_csv(labels_path, ["Subject", "Label"], label_rows)
    return {"data_root": data_root, "split_csvs": manifest_paths, "labels_path": labels_path}


def build_pairs_from_manifests(manifest_csv: Path, pairs_csv: Path, split_name: str):
    with manifest_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    by_subject = {}
    for row in rows:
        src = Path(row["source_path"]).stem
        parts = src.split("__")
        subject = parts[0]
        chunk_token = next((p for p in parts if p.startswith("chunk_")), "chunk_000")
        chunk_id = int(chunk_token.split("_")[-1])
        by_subject.setdefault(subject, []).append((chunk_id, row))
    out_rows = []
    emb_root = pairs_csv.parent.parent / "embeddings" / split_name
    emb_root.mkdir(parents=True, exist_ok=True)
    for subject, items in by_subject.items():
        items = sorted(items, key=lambda x: x[0])
        if len(items) < 2:
            continue
        fc = emb_root / f"{subject}_fc.npy"
        mri = emb_root / f"{subject}_mri.npy"
        np.save(fc, np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
        np.save(mri, np.asarray([0.5, 0.6, 0.7], dtype=np.float32))
        for idx in range(len(items) - 1):
            anchor_chunk, anchor = items[idx]
            target_chunk, target = items[idx + 1]
            out_rows.append(
                {
                    "Subject": subject,
                    "sequence_id": "seq-rest",
                    "anchor_chunk_id": anchor_chunk,
                    "target_chunk_id": target_chunk,
                    "pair_direction": "next",
                    "vae_latent_path": anchor["npz_path"],
                    "target_latent_path": target["npz_path"],
                    "voxel_data_path": anchor["source_path"],
                    "target_voxel_data_path": target["source_path"],
                    "fc_embedding_path": str(fc),
                    "MRI_embedding_path": str(mri),
                }
            )
    _write_csv(
        pairs_csv,
        [
            "Subject",
            "sequence_id",
            "anchor_chunk_id",
            "target_chunk_id",
            "pair_direction",
            "vae_latent_path",
            "target_latent_path",
            "voxel_data_path",
            "target_voxel_data_path",
            "fc_embedding_path",
            "MRI_embedding_path",
        ],
        out_rows,
    )
    return out_rows
