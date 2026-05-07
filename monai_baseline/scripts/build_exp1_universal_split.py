#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from typing import Dict, List, Tuple

CSV_HEADER = [
    "Subject",
    "latent_path",
    "fc_embedding_path",
    "target_latent_path",
    "task",
    "session",
]

_RANGE_RE = re.compile(r"_(\d+)-(\d+)$")
_SEG_RE = re.compile(r"_seg(\d+)$", re.IGNORECASE)

# Explicit latent-dataset -> FC-dataset mapping (strict mapping policy)
DATASET_FC_MAPPING: dict[str, str] = {
    "ABCD_all_session": "ABCD",
    "ADNI_NEW_resampled": "ADNI_NEW_resampled",
    "CHCP_all_session": "CHCP",
    "CineBrain": "CineBrain",
    "FCON": "FCON",
    "HCP_all_session": "HCP",
    "HCP_movie": "HCP_movie",
    "ICB": "ICB",
    "ISYB": "ISYB",
    "NKI": "NKI_40",
    "PIOP1": "PIOP1",
    "PIOP2": "PIOP2",
    "RHBC": "BHRC_40",
    "SALD": "SALD",
    "fwhm6": "fwhm6",
}

# Explicitly excluded this round due to ambiguous mapping policy.
EXCLUDED_DATASETS: dict[str, str] = {
    "emo_film": "excluded_by_policy_ambiguous_mapping",
}


def stem_without_ext(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if path.suffix.lower() in {".npy", ".npz", ".nii"}:
        return path.stem
    return name


def parse_sample(stem: str, dataset: str) -> tuple[str, str, int, str]:
    m = _RANGE_RE.search(stem)
    if m:
        start = int(m.group(1))
        seq = stem[: m.start()]
    else:
        m2 = _SEG_RE.search(stem)
        if m2:
            start = int(m2.group(1))
            seq = stem[: m2.start()]
        else:
            start = -1
            seq = stem

    if dataset == "HCP_all_session" and "__" in seq:
        subject, rest = seq.split("__", 1)
        session = rest
    else:
        tokens = seq.split("_")
        subject = tokens[0]
        session = "_".join(tokens[1:]) if len(tokens) > 1 else "unknown"

    return subject, seq, start, session


def build_subject_split_map(subjects: list[str], train_ratio: float, val_ratio: float) -> dict[str, str]:
    scored = sorted(subjects, key=lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())
    n = len(scored)
    if n == 0:
        return {}
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val
    if n >= 3:
        if n_val <= 0:
            n_val = 1
        if n_test <= 0:
            n_test = 1
        n_train = max(1, n - n_val - n_test)
    split_map = {}
    for i, sub in enumerate(scored):
        if i < n_train:
            split_map[sub] = "train"
        elif i < n_train + n_val:
            split_map[sub] = "val"
        else:
            split_map[sub] = "test"
    return split_map


def _parse_name_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def build_explicit_split_map(
    *,
    train_subjects: list[str],
    val_subjects: list[str],
    test_subjects: list[str],
) -> dict[str, str]:
    split_map: dict[str, str] = {}
    for split_name, subjects in [
        ("train", train_subjects),
        ("val", val_subjects),
        ("test", test_subjects),
    ]:
        for subject in subjects:
            if subject in split_map:
                raise ValueError(f"Duplicate subject {subject!r} across explicit split assignments")
            split_map[subject] = split_name
    return split_map


def build_fc_index(fc_root: Path) -> Dict[str, str]:
    index: Dict[str, str] = {}
    if not fc_root.exists():
        return index
    for p in sorted(fc_root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in {".npy", ".npz"}:
            continue
        s = stem_without_ext(p)
        if s in index:
            continue
        index[s] = str(p)
    return index


def collect_latent_files(latent_root: Path, include_npz: bool) -> list[Path]:
    patterns: list[str] = ["*.npy"]
    if include_npz:
        patterns.append("*.npz")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(latent_root.rglob(pattern))
    files = sorted({str(p.resolve()): p for p in files}.values(), key=lambda p: str(p))
    return files


def split_for_subject(
    dataset: str,
    subject: str,
    split_map: dict[str, str],
    *,
    cinebrain_split_mode: str,
    all_train_datasets: set[str],
) -> str:
    if dataset in all_train_datasets:
        return "train"
    if dataset == "CineBrain" and cinebrain_split_mode == "all_train":
        return "train"
    return split_map.get(subject, "train")


def main() -> None:
    t0 = time.time()
    ap = argparse.ArgumentParser(description="Build public universal-split CSVs for the MONAI baseline")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--cinebrain-audit-topk", type=int, default=100)
    ap.add_argument(
        "--latent-root",
        default="artifacts/vqvae_latents",
        help="Monai stage1 latent root; dataset paths default to <latent-root>/<dataset>.",
    )
    ap.add_argument(
        "--fc-root",
        dest="fc_root",
        default="data/fc_embeddings",
        help="FC embedding root.",
    )
    ap.add_argument(
        "--cinebrain-split-mode",
        default="all_train",
        choices=["all_train", "hash_split", "subject_fixed"],
        help="How CineBrain rows are assigned to train/val/test.",
    )
    ap.add_argument(
        "--cinebrain-train-subjects",
        default="",
        help="Comma-separated CineBrain subjects assigned to train when --cinebrain-split-mode=subject_fixed.",
    )
    ap.add_argument(
        "--cinebrain-val-subjects",
        default="",
        help="Comma-separated CineBrain subjects assigned to val when --cinebrain-split-mode=subject_fixed.",
    )
    ap.add_argument(
        "--cinebrain-test-subjects",
        default="",
        help="Comma-separated CineBrain subjects assigned to test when --cinebrain-split-mode=subject_fixed.",
    )
    ap.add_argument(
        "--selected-datasets",
        default="",
        help="Optional comma-separated subset of datasets to regenerate. Default regenerates all selected datasets.",
    )
    ap.add_argument(
        "--all-train-datasets",
        default="",
        help="Comma-separated datasets forced into train split only (e.g., HCP_movie).",
    )
    ap.add_argument(
        "--include-npz",
        action="store_true",
        help="Also include *.npz latent files for backward compatibility.",
    )
    args = ap.parse_args()

    def _resolve_repo_path(value: str) -> Path:
        path = Path(str(value))
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return path

    args.output_root = str(_resolve_repo_path(args.output_root))
    args.latent_root = str(_resolve_repo_path(args.latent_root))
    args.fc_root = str(_resolve_repo_path(args.fc_root))

    latent_root = Path(args.latent_root)
    fc_root = Path(args.fc_root)
    if not latent_root.exists():
        raise FileNotFoundError(f"latent root not found: {latent_root}")
    if not fc_root.exists():
        raise FileNotFoundError(f"fc root not found: {fc_root}")

    available_latent_datasets = sorted([p.name for p in latent_root.iterdir() if p.is_dir()])
    selected_dataset_filter = set(_parse_name_list(args.selected_datasets))
    all_train_datasets = set(_parse_name_list(args.all_train_datasets))

    dropped_datasets: list[dict[str, str]] = []
    selected_datasets: list[str] = []
    for ds in available_latent_datasets:
        if selected_dataset_filter and ds not in selected_dataset_filter:
            continue
        if ds in EXCLUDED_DATASETS:
            dropped_datasets.append({"dataset": ds, "reason": EXCLUDED_DATASETS[ds]})
            continue
        if ds not in DATASET_FC_MAPPING:
            dropped_datasets.append({"dataset": ds, "reason": "missing_fc_mapping"})
            continue
        selected_datasets.append(ds)

    if selected_dataset_filter:
        missing_selected = sorted(selected_dataset_filter - set(selected_datasets) - set(EXCLUDED_DATASETS.keys()))
        if missing_selected:
            raise ValueError(f"Requested datasets not available or unsupported: {missing_selected}")

    cinebrain_explicit_split_map = build_explicit_split_map(
        train_subjects=_parse_name_list(args.cinebrain_train_subjects),
        val_subjects=_parse_name_list(args.cinebrain_val_subjects),
        test_subjects=_parse_name_list(args.cinebrain_test_subjects),
    )
    if str(args.cinebrain_split_mode) == "subject_fixed" and not cinebrain_explicit_split_map:
        raise ValueError("cinebrain_split_mode=subject_fixed requires explicit train/val/test subject lists")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "output_root": str(output_root),
        "latent_root": str(latent_root),
        "fc_root": str(fc_root),
        "cinebrain_split_mode": str(args.cinebrain_split_mode),
        "cinebrain_explicit_split_map": cinebrain_explicit_split_map,
        "include_npz": bool(args.include_npz),
        "dataset_fc_mapping": dict(DATASET_FC_MAPPING),
        "selected_datasets": selected_datasets,
        "all_train_datasets": sorted(all_train_datasets),
        "dropped_datasets": dropped_datasets,
        "datasets": {},
    }

    print(
        json.dumps(
            {
                "stage": "prepare_split_start",
                "latent_root": str(latent_root),
                "fc_root": str(fc_root),
                "selected_dataset_count": len(selected_datasets),
                "excluded_count": len(dropped_datasets),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for dataset in selected_datasets:
        ds_t0 = time.time()
        fc_dataset = DATASET_FC_MAPPING[dataset]
        ds_latent_root = latent_root / dataset
        ds_fc_root = fc_root / fc_dataset

        print(
            json.dumps(
                {
                    "stage": "dataset_start",
                    "dataset": dataset,
                    "fc_dataset": fc_dataset,
                    "latent_root": str(ds_latent_root),
                    "fc_root": str(ds_fc_root),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        if not ds_latent_root.exists():
            dropped_datasets.append({"dataset": dataset, "reason": "latent_root_missing"})
            continue
        if not ds_fc_root.exists():
            dropped_datasets.append({"dataset": dataset, "reason": f"fc_root_missing:{ds_fc_root}"})
            continue

        fc_index = build_fc_index(ds_fc_root)
        files = collect_latent_files(ds_latent_root, include_npz=bool(args.include_npz))

        grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
        parse_errors = 0
        for p in files:
            stem = stem_without_ext(p)
            try:
                subject, seq, seg, session = parse_sample(stem, dataset)
            except Exception:
                parse_errors += 1
                continue
            grouped[(subject, seq)].append(
                {
                    "Subject": subject,
                    "stem": stem,
                    "segment": seg,
                    "session": session,
                    "latent_path": str(p),
                }
            )

        rows_by_split = {"train": [], "val": [], "test": []}
        fc_matched = 0
        fc_missing = 0
        pair_count = 0
        dropped_non_monotonic = 0
        subjects_by_split = {"train": set(), "val": set(), "test": set()}
        cine_pairs_preview = []

        dataset_subjects = sorted({k[0] for k in grouped.keys()})
        if dataset == "CineBrain" and str(args.cinebrain_split_mode) == "subject_fixed":
            unknown_subjects = sorted(set(cinebrain_explicit_split_map.keys()) - set(dataset_subjects))
            if unknown_subjects:
                raise ValueError(f"CineBrain explicit split contains unknown subjects: {unknown_subjects}")
            missing_subjects = sorted(set(dataset_subjects) - set(cinebrain_explicit_split_map.keys()))
            if missing_subjects:
                raise ValueError(f"CineBrain explicit split is missing subjects: {missing_subjects}")
            split_map = dict(cinebrain_explicit_split_map)
        else:
            split_map = build_subject_split_map(dataset_subjects, args.train_ratio, args.val_ratio)

        for (subject, _seq), items in grouped.items():
            items_sorted = sorted(items, key=lambda x: (int(x["segment"]), x["stem"]))
            split = split_for_subject(
                dataset,
                subject,
                split_map,
                cinebrain_split_mode=str(args.cinebrain_split_mode),
                all_train_datasets=all_train_datasets,
            )
            subjects_by_split[split].add(subject)
            for i in range(len(items_sorted) - 1):
                anchor = items_sorted[i]
                target = items_sorted[i + 1]
                if int(anchor["segment"]) >= 0 and int(target["segment"]) >= 0:
                    if int(target["segment"]) <= int(anchor["segment"]):
                        dropped_non_monotonic += 1
                        continue

                fc_path = fc_index.get(anchor["stem"], "")
                if fc_path:
                    fc_matched += 1
                else:
                    fc_missing += 1

                row = {
                    "Subject": anchor["Subject"],
                    "latent_path": anchor["latent_path"],
                    "fc_embedding_path": fc_path,
                    "target_latent_path": target["latent_path"],
                    "task": "rest",
                    "session": anchor["session"],
                }
                rows_by_split[split].append(row)
                pair_count += 1

                if dataset == "CineBrain" and len(cine_pairs_preview) < int(args.cinebrain_audit_topk):
                    cine_pairs_preview.append(
                        {
                            "split": split,
                            "subject": anchor["Subject"],
                            "anchor_stem": anchor["stem"],
                            "target_stem": target["stem"],
                            "fc_stem_matched": bool(fc_path),
                            "fc_path": fc_path,
                        }
                    )

        ds_dir = output_root / dataset
        ds_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            out_csv = ds_dir / f"{split}.csv"
            with out_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=CSV_HEADER)
                w.writeheader()
                w.writerows(rows_by_split[split])

        summary["datasets"][dataset] = {
            "latent_root": str(ds_latent_root),
            "fc_root": str(ds_fc_root),
            "fc_dataset": str(fc_dataset),
            "latent_files": len(files),
            "latent_files_npy": int(sum(1 for p in files if p.suffix.lower() == ".npy")),
            "latent_files_npz": int(sum(1 for p in files if p.suffix.lower() == ".npz")),
            "pairs_total": pair_count,
            "rows_kept": pair_count,
            "fc_matched": fc_matched,
            "fc_missing": fc_missing,
            "dropped_unmapped": 0,
            "dropped_non_monotonic": dropped_non_monotonic,
            "parse_errors": parse_errors,
            "rows_per_split": {k: len(v) for k, v in rows_by_split.items()},
            "subjects_per_split": {k: len(v) for k, v in subjects_by_split.items()},
        }

        print(
            json.dumps(
                {
                    "stage": "dataset_done",
                    "dataset": dataset,
                    "elapsed_sec": round(time.time() - ds_t0, 3),
                    "latent_files": len(files),
                    "pairs_total": pair_count,
                    "fc_missing": fc_missing,
                    "rows_per_split": {k: len(v) for k, v in rows_by_split.items()},
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        if dataset == "CineBrain":
            audit_path = output_root / "cinebrain_pairing_audit_top100.json"
            audit_payload = {
                "dataset": dataset,
                "pairs_preview": cine_pairs_preview,
                "summary": summary["datasets"][dataset],
            }
            audit_path.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "summary": str(summary_path),
                "selected_datasets": len(selected_datasets),
                "elapsed_sec": round(time.time() - t0, 3),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
