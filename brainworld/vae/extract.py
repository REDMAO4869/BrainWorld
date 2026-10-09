from __future__ import annotations

import argparse
import csv
import os
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from brainworld.vae.data import build_split_from_config, collate_batch
from brainworld.vae.model import build_model_from_config
from brainworld.vae.utils import ensure_dir, load_json, save_json, set_seed


MANIFEST_FIELDS = [
    "sample_id",
    "dataset_id",
    "subject_id",
    "source_path",
    "relative_npz_path",
    "npz_path",
    "saved_fields",
    "mu_shape",
    "mu_mean",
    "mu_std",
    "status",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract latent features from BrainWorld VAE")
    p.add_argument("--config", required=True, help="Path to extraction JSON config")
    return p.parse_args()


def _device_from_cfg(cfg: Dict) -> torch.device:
    dev = str(cfg.get("device", "auto"))
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(dev)


def _pick_loader(cfg: Dict):
    split = str(cfg.get("extraction", {}).get("split", "train")).lower()
    ds, audit = build_split_from_config(cfg, split)
    audits = {split: audit}

    ex_cfg = cfg.get("extraction", {})
    bs = int(ex_cfg.get("batch_size", 1))
    nw = int(ex_cfg.get("num_workers", 2))
    shard_index = int(ex_cfg.get("shard_index", 0))
    num_shards = int(ex_cfg.get("num_shards", 1))
    if num_shards <= 0:
        raise ValueError(f"extraction.num_shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"extraction.shard_index must be in [0, {num_shards}), got {shard_index}")

    full_files = list(ds.files)
    dataset_id_by_path = dict(getattr(ds, "dataset_id_by_path", {}))
    files_before = len(full_files)
    if num_shards > 1:
        ds.files = list(full_files[shard_index::num_shards])
        if len(ds.files) == 0:
            raise RuntimeError(
                f"Shard {shard_index}/{num_shards} for split={split} has 0 files; "
                f"split only has {files_before} files"
            )
        preview_k = max(1, len(audits[split].sample_preview))
        audits[split] = replace(audits[split], num_files=len(ds.files), sample_preview=list(ds.files[:preview_k]))

    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True, collate_fn=collate_batch)
    shard_meta = {
        "shard_index": shard_index,
        "num_shards": num_shards,
        "files_before_shard": files_before,
        "files_after_shard": len(ds.files),
        "all_files": full_files,
        "dataset_id_by_path": dataset_id_by_path,
    }
    return split, loader, audits, shard_meta


def _to_np(x: torch.Tensor, dtype_name: str) -> np.ndarray:
    x_cpu = x.detach().cpu()
    if x_cpu.dtype == torch.bfloat16:
        x_cpu = x_cpu.to(torch.float32)
    a = x_cpu.numpy()
    dtype_name = str(dtype_name).lower()
    if dtype_name == "float16":
        return a.astype(np.float16, copy=False)
    if dtype_name == "float32":
        return a.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported save dtype={dtype_name}, expected float16/float32")


def _load_model_cfg_from_ckpt(ckpt_path: str) -> Dict | None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_cfg = ckpt.get("model_config", None)
    if isinstance(model_cfg, dict) and "model" in model_cfg:
        return model_cfg
    return None


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


def _subject_and_stem_from_path(path_i: str) -> tuple[str, str]:
    base = os.path.basename(path_i)
    stem = os.path.splitext(base)[0]
    if "__" in stem:
        subj = stem.split("__", 1)[0]
    else:
        subj = os.path.basename(os.path.dirname(path_i))
    return _sanitize_token(subj), _sanitize_token(stem)


def _latent_filename_from_source(path_i: str) -> str:
    base = os.path.basename(path_i)
    stem, ext = os.path.splitext(base)
    if ext.lower() == ".npz":
        return base
    return f"{stem}.npz"


def _candidate_relpath_within_dataset(path_i: str, dataset_id: str) -> str:
    path = Path(str(path_i))
    filename = _latent_filename_from_source(path_i)
    parts = list(path.parts)

    anchor_idx = None
    for idx, part in enumerate(parts[:-1]):
        if _sanitize_token(part) == dataset_id:
            anchor_idx = idx
            break

    if anchor_idx is not None:
        rel_parts = [part for part in parts[anchor_idx + 1 : -1] if part not in {"", os.sep}]
        if rel_parts:
            return os.path.join(dataset_id, *rel_parts, filename)
    return os.path.join(dataset_id, filename)


def _build_output_relpath_map(
    files: Sequence[str],
    dataset_id_by_path: Dict[str, str],
    *,
    output_dataset_id: str = "",
    flatten_output_paths: bool = False,
) -> Tuple[Dict[str, Dict[str, str]], int]:
    ordered_files = [str(path_i) for path_i in dict.fromkeys(str(path_i) for path_i in files)]
    grouped: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for fp in ordered_files:
        dataset_id = _sanitize_token(output_dataset_id or dataset_id_by_path.get(fp, "unknown"))
        if flatten_output_paths:
            relpath = os.path.join(dataset_id, _latent_filename_from_source(fp))
        else:
            relpath = _candidate_relpath_within_dataset(fp, dataset_id)
        grouped[dataset_id].append((fp, relpath))

    resolved: Dict[str, Dict[str, str]] = {}
    collision_count = 0
    for dataset_id, items in grouped.items():
        relpath_counts = Counter(relpath for _, relpath in items)
        used_relpaths = set()
        for fp, relpath in items:
            final_relpath = relpath
            if relpath_counts[relpath] > 1 or final_relpath in used_relpaths:
                collision_count += 1
                filename = _latent_filename_from_source(fp)
                parent_parts = [part for part in Path(fp).parent.parts if part not in {"", os.sep}]
                for depth in range(1, len(parent_parts) + 1):
                    final_relpath = os.path.join(dataset_id, *parent_parts[-depth:], filename)
                    if final_relpath not in used_relpaths:
                        break
                else:
                    stem = os.path.splitext(filename)[0]
                    suffix = 1
                    final_relpath = os.path.join(dataset_id, f"{stem}__dup{suffix:03d}.npz")
                    while final_relpath in used_relpaths:
                        suffix += 1
                        final_relpath = os.path.join(dataset_id, f"{stem}__dup{suffix:03d}.npz")

            used_relpaths.add(final_relpath)
            resolved[fp] = {"dataset_id": dataset_id, "relative_npz_path": final_relpath}

    return resolved, collision_count


def _output_meta_for_path(path_i: str, output_relpath_map: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    meta = output_relpath_map.get(str(path_i))
    if meta is not None:
        return meta
    return {
        "dataset_id": "unknown",
        "relative_npz_path": os.path.join("unknown", _latent_filename_from_source(path_i)),
    }


def _task_state_dir(output_base: str, split: str, shard_meta: Dict, ex_cfg: Dict) -> str:
    custom_state_dir = str(ex_cfg.get("state_dir", "")).strip()
    if custom_state_dir:
        return ensure_dir(custom_state_dir)

    parts = [output_base, "_state", split]
    if int(shard_meta.get("num_shards", 1)) > 1:
        parts.append(
            f"shard_{int(shard_meta.get('shard_index', 0)):02d}_of_{int(shard_meta.get('num_shards', 1)):02d}"
        )
    return ensure_dir(os.path.join(*parts))


def _write_manifest(path: str, rows: List[Dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    ex_cfg = cfg.get("extraction", {})
    ckpt_path = str(ex_cfg.get("checkpoint", "")).strip()
    if ckpt_path == "":
        raise ValueError("extraction.checkpoint is required")

    if not os.path.isabs(ckpt_path):
        cfg_dir = os.path.dirname(os.path.abspath(args.config))
        ckpt_path = os.path.abspath(os.path.join(cfg_dir, ckpt_path))

    if bool(ex_cfg.get("auto_load_model_config", True)):
        mcfg = _load_model_cfg_from_ckpt(ckpt_path)
        if mcfg is not None:
            cfg["model"] = mcfg["model"]
            data = cfg.get("data", {})
            for key, value in mcfg.get("data", {}).items():
                data[key] = value
            cfg["data"] = data

    device = _device_from_cfg(cfg)
    split, loader, audits, shard_meta = _pick_loader(cfg)

    output_base = ensure_dir(str(ex_cfg.get("output_dir", "outputs/wf_vae2_latents")))
    state_root = _task_state_dir(output_base, split, shard_meta, ex_cfg)
    summary_path = os.path.join(state_root, "summary.json")
    manifest_path = os.path.join(state_root, "manifest.csv")
    state_path = os.path.join(state_root, "resume_state.json")

    output_relpath_map, collision_count = _build_output_relpath_map(
        shard_meta.get("all_files", []),
        shard_meta.get("dataset_id_by_path", {}),
        output_dataset_id=str(ex_cfg.get("output_dataset_id", "")).strip(),
        flatten_output_paths=bool(ex_cfg.get("flatten_output_paths", False)),
    )
    dataset_count = len({meta["dataset_id"] for meta in output_relpath_map.values()})

    model = build_model_from_config(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    save_dtype = str(ex_cfg.get("save_dtype", "float16"))
    posterior_mode = str(ex_cfg.get("posterior_mode", "mu")).lower()
    sample_posterior = posterior_mode == "sample"
    max_samples = int(ex_cfg.get("max_samples", 0))
    save_compressed = bool(ex_cfg.get("save_compressed", False))
    save_fields = [str(x).lower() for x in ex_cfg.get("save_fields", ["mu", "logvar", "z"])]
    save_fields = [x for x in save_fields if x in {"mu", "logvar", "z"}]
    resume_enabled = bool(ex_cfg.get("resume", True))
    if len(save_fields) == 0:
        raise ValueError("extraction.save_fields must contain at least one of mu/logvar/z")

    use_amp = bool(ex_cfg.get("use_amp", True) and device.type == "cuda")
    amp_dtype_name = str(ex_cfg.get("amp_dtype", "bf16")).lower()
    amp_dtype = torch.float16 if amp_dtype_name == "fp16" else torch.bfloat16
    requested_timestamped_output = bool(ex_cfg.get("use_timestamped_output", False))

    print("=" * 88)
    print(f"[audit] split={split}")
    print(f"[audit] files={audits[split].num_files}")
    if int(shard_meta["num_shards"]) > 1:
        print(
            f"[audit] shard={shard_meta['shard_index']}/{shard_meta['num_shards']} "
            f"files_before_shard={shard_meta['files_before_shard']} files_after_shard={shard_meta['files_after_shard']}"
        )
    print(f"[audit] roots={audits[split].roots}")
    print(f"[audit] preview={audits[split].sample_preview}")
    print(f"[audit] checkpoint={ckpt_path}")
    print(f"[audit] output_base={output_base}")
    print(f"[audit] state_dir={state_root}")
    print(f"[audit] dataset_count={dataset_count} collision_fallbacks={collision_count}")
    print(f"[audit] save_fields={save_fields} save_compressed={save_compressed} save_dtype={save_dtype}")
    print(f"[audit] resume_enabled={resume_enabled}")
    print(f"[audit] use_amp={use_amp} amp_dtype={amp_dtype_name}")
    print(f"[audit] requested_use_timestamped_output={requested_timestamped_output} effective_use_timestamped_output=False")
    print("[audit] naming=<dataset_id>/<original_relative_path>/<original_filename>.npz")
    print("=" * 88)
    save_json(
        {
            "status": "running",
            "split": split,
            "checkpoint": ckpt_path,
            "resume_enabled": resume_enabled,
            "output_base": output_base,
            "state_dir": state_root,
            "shard_index": int(shard_meta["shard_index"]),
            "num_shards": int(shard_meta["num_shards"]),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        state_path,
    )

    dataset = loader.dataset
    dataset_files = list(getattr(dataset, "files", []))
    if len(dataset_files) == 0:
        raise RuntimeError("No files assigned to the current extraction task")

    selected_records: List[Dict] = []
    for sample_idx, path_i in enumerate(dataset_files):
        if max_samples > 0 and len(selected_records) >= max_samples:
            break
        path_i = str(path_i)
        meta = _output_meta_for_path(path_i, output_relpath_map)
        npz_path = os.path.join(output_base, meta["relative_npz_path"])
        selected_records.append(
            {
                "sample_id": int(sample_idx),
                "source_path": path_i,
                "dataset_id": str(meta["dataset_id"]),
                "relative_npz_path": str(meta["relative_npz_path"]),
                "npz_path": npz_path,
                "dataset_local_index": int(sample_idx),
            }
        )

    rows: List[Dict] = []
    pending_indices: List[int] = []
    pending_records: List[Dict] = []
    n_done = 0
    n_new = 0
    n_skipped_existing = 0
    mu_mean_sum = 0.0
    mu_std_sum = 0.0
    printed_latent_shape = False

    for record in selected_records:
        if resume_enabled and os.path.isfile(record["npz_path"]):
            subject_id, _ = _subject_and_stem_from_path(record["source_path"])
            rows.append(
                {
                    "sample_id": int(record["sample_id"]),
                    "dataset_id": record["dataset_id"],
                    "subject_id": subject_id,
                    "source_path": record["source_path"],
                    "relative_npz_path": record["relative_npz_path"],
                    "npz_path": record["npz_path"],
                    "saved_fields": ",".join(save_fields),
                    "mu_shape": "",
                    "mu_mean": "",
                    "mu_std": "",
                    "status": "existing",
                }
            )
            n_done += 1
            n_skipped_existing += 1
        else:
            pending_indices.append(int(record["dataset_local_index"]))
            pending_records.append(record)

    print(
        f"[audit] selected_samples={len(selected_records)} pending_new={len(pending_records)} "
        f"existing_skip={n_skipped_existing}"
    )

    pending_loader = None
    if pending_indices:
        pending_loader = DataLoader(
            Subset(dataset, pending_indices),
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            collate_fn=loader.collate_fn,
        )

    pbar = tqdm(total=len(selected_records), desc=f"extract[{split}]", ncols=120)
    if n_skipped_existing > 0:
        pbar.update(n_skipped_existing)
        pbar.set_postfix(done=n_done, new=n_new, resumed=n_skipped_existing)

    if pending_loader is not None:
        pending_cursor = 0
        with torch.inference_mode():
            for batch in pending_loader:
                x = batch["x"].to(device=device, dtype=torch.float32)
                amp_ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()
                with amp_ctx:
                    enc = model.encode(x, sample_posterior=sample_posterior)

                if not printed_latent_shape:
                    print("=" * 88)
                    print(f"[audit] input_batch_shape={tuple(x.shape)}")
                    print(
                        f"[audit] latent_mu_shape={tuple(enc['mu'].shape)} "
                        f"latent_logvar_shape={tuple(enc['logvar'].shape)} latent_z_shape={tuple(enc['z'].shape)}"
                    )
                    per_sample_numel = int(enc["mu"][0].numel())
                    print(f"[audit] latent_numel_per_sample={per_sample_numel}")
                    print("=" * 88)
                    printed_latent_shape = True

                batch_paths = [str(path_i) for path_i in batch["paths"]]
                batch_records = pending_records[pending_cursor : pending_cursor + len(batch_paths)]
                pending_cursor += len(batch_paths)

                for i, path_i in enumerate(batch_paths):
                    record = batch_records[i]
                    meta = _output_meta_for_path(path_i, output_relpath_map)
                    npz_path = os.path.join(output_base, meta["relative_npz_path"])
                    subject_id, _ = _subject_and_stem_from_path(path_i)

                    if resume_enabled and os.path.isfile(npz_path):
                        rows.append(
                            {
                                "sample_id": int(record["sample_id"]),
                                "dataset_id": meta["dataset_id"],
                                "subject_id": subject_id,
                                "source_path": path_i,
                                "relative_npz_path": meta["relative_npz_path"],
                                "npz_path": npz_path,
                                "saved_fields": ",".join(save_fields),
                                "mu_shape": "",
                                "mu_mean": "",
                                "mu_std": "",
                                "status": "existing",
                            }
                        )
                        n_done += 1
                        n_skipped_existing += 1
                        continue

                    mu_i = enc["mu"][i]
                    logvar_i = enc["logvar"][i]
                    z_i = enc["z"][i]

                    payload = {}
                    if "mu" in save_fields:
                        payload["mu"] = _to_np(mu_i, save_dtype)
                    if "logvar" in save_fields:
                        payload["logvar"] = _to_np(logvar_i, save_dtype)
                    if "z" in save_fields:
                        payload["z"] = _to_np(z_i, save_dtype)

                    ensure_dir(os.path.dirname(npz_path))
                    if save_compressed:
                        np.savez_compressed(npz_path, **payload)
                    else:
                        np.savez(npz_path, **payload)

                    mu_mean = float(mu_i.mean().item())
                    mu_std = float(mu_i.std().item())
                    mu_mean_sum += mu_mean
                    mu_std_sum += mu_std

                    rows.append(
                        {
                            "sample_id": int(record["sample_id"]),
                            "dataset_id": meta["dataset_id"],
                            "subject_id": subject_id,
                            "source_path": path_i,
                            "relative_npz_path": meta["relative_npz_path"],
                            "npz_path": npz_path,
                            "saved_fields": ",".join(save_fields),
                            "mu_shape": str(tuple(mu_i.shape)),
                            "mu_mean": mu_mean,
                            "mu_std": mu_std,
                            "status": "new",
                        }
                    )
                    n_done += 1
                    n_new += 1

                pbar.update(len(batch_paths))
                pbar.set_postfix(done=n_done, new=n_new, resumed=n_skipped_existing)
    pbar.close()

    if len(rows) == 0:
        raise RuntimeError("No samples extracted or resumed")

    rows.sort(key=lambda row: (int(row["sample_id"]), str(row["dataset_id"]), str(row["source_path"])))
    _write_manifest(manifest_path, rows)

    summary = {
        "shard_index": int(shard_meta["shard_index"]),
        "num_shards": int(shard_meta["num_shards"]),
        "files_before_shard": int(shard_meta["files_before_shard"]),
        "files_after_shard": int(shard_meta["files_after_shard"]),
        "num_samples": len(rows),
        "num_newly_written": int(n_new),
        "num_existing_skipped": int(n_skipped_existing),
        "split": split,
        "checkpoint": ckpt_path,
        "output_base": output_base,
        "state_dir": state_root,
        "dataset_count": int(dataset_count),
        "collision_fallbacks": int(collision_count),
        "save_dtype": save_dtype,
        "save_fields": save_fields,
        "save_compressed": save_compressed,
        "posterior_mode": posterior_mode,
        "mean_mu_mean": float(mu_mean_sum / max(1, n_new)),
        "mean_mu_std": float(mu_std_sum / max(1, n_new)),
        "manifest": manifest_path,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(summary, summary_path)
    save_json(
        {
            "status": "completed",
            **summary,
        },
        state_path,
    )
    print(f"[done] latent extraction output: {output_base}")
    print(f"[done] state directory: {state_root}")


if __name__ == "__main__":
    main()
