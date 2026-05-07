from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from monai_fmri_public.config import load_json, load_jsonl, resolve_paths, save_json, save_jsonl, save_resolved_config
from monai_fmri_public.data import prepare_clip_from_record
from monai_fmri_public.distributed import DistributedContext, barrier, cleanup_distributed, print_main, setup_distributed
from monai_fmri_public.models import build_vqvae
from monai_fmri_public.utils import ensure_dir, seed_everything


def _limit_records(records: list[dict], max_items: int | None) -> list[dict]:
    if max_items is None:
        return records
    return records[: int(max_items)]


def _shard_records(records: list[dict], dist_ctx: DistributedContext) -> list[tuple[int, dict]]:
    if not dist_ctx.enabled:
        return list(enumerate(records))
    return [(global_index, records[global_index]) for global_index in range(dist_ctx.rank, len(records), dist_ctx.world_size)]


def _all_reduce_scalar(value: float | int, dist_ctx: DistributedContext, *, dtype: torch.dtype) -> float | int:
    tensor = torch.tensor([value], device=dist_ctx.device, dtype=dtype)
    if dist_ctx.enabled:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    reduced = tensor.item()
    if dtype == torch.long:
        return int(reduced)
    return float(reduced)


def _dataset_name_for_record(record: dict) -> str:
    if record.get("dataset"):
        return str(record["dataset"])
    return Path(record["image"]).parent.name


def _latent_output_name(record: dict) -> str:
    source_path = Path(record["image"])
    if source_path.name.endswith(".nii.gz"):
        base_name = source_path.name[: -len(".nii.gz")]
    elif source_path.suffix in {".npz", ".npy", ".nii"}:
        base_name = source_path.stem
    else:
        base_name = source_path.name
    return f"{base_name}.npy"


def _normalize_output_relpath(output_relpath: str) -> Path:
    rel = Path(output_relpath)
    parts = [part for part in rel.parts if part not in ("", ".", "..")]
    if not parts:
        raise ValueError(f"Invalid output_relpath: {output_relpath}")
    rel = Path(*parts)
    suffix_text = "".join(rel.suffixes)
    if suffix_text.endswith(".nii.gz"):
        stem = rel.name[: -len(".nii.gz")]
    elif rel.suffix in {".npz", ".npy", ".nii"}:
        stem = rel.stem
    else:
        stem = rel.name
    return rel.with_name(f"{stem}.npy")


def _latent_output_relpath(record: dict) -> Path:
    output_relpath = record.get("output_relpath")
    if output_relpath:
        return _normalize_output_relpath(str(output_relpath))
    return Path(_latent_output_name(record))


def _split_manifest_path(output_root: Path, split_name: str, dist_ctx: DistributedContext) -> Path:
    if dist_ctx.enabled:
        return output_root / f"{split_name}_latents.rank{dist_ctx.rank:04d}.jsonl"
    return output_root / f"{split_name}_latents.jsonl"


def _append_jsonl_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _latent_stats_from_array(latents_np: np.ndarray) -> tuple[float, float, int]:
    latents64 = latents_np.astype(np.float64, copy=False)
    return float(latents64.sum()), float(np.square(latents64).sum()), int(latents64.size)


def _read_latent_file(path: Path) -> np.ndarray:
    latents = np.load(path)
    if latents.ndim != 5:
        raise ValueError(f"Expected latent tensor [T,C,D,H,W], got {latents.shape} from {path}")
    return latents


def _cached_record_from_latents(record: dict, out_path: Path, latents_shape: tuple[int, ...]) -> dict:
    dataset_name = _dataset_name_for_record(record)
    return {
        "latent_path": str(out_path),
        "dataset": dataset_name,
        "task": record["task"],
        "subject": record.get("subject", "unknown"),
        "session": record.get("session", "unknown"),
        "segment": int(record.get("segment", 0)),
        "num_frames": int(latents_shape[0]),
        "latent_channels": int(latents_shape[1]),
        "latent_spatial_shape": list(latents_shape[2:]),
        "source_image": str(record["image"]),
    }


def _load_existing_records(shard_manifest_path: Path) -> tuple[list[dict], dict[str, dict], dict[str, str], int]:
    if not shard_manifest_path.exists():
        return [], {}, {}, 0

    rows = load_jsonl(shard_manifest_path)
    valid_rows: list[dict] = []
    rows_by_source: dict[str, dict] = {}
    output_to_source: dict[str, str] = {}
    stale_rows = 0

    for row in rows:
        source_image = row.get("source_image")
        latent_path = row.get("latent_path")
        if not source_image or not latent_path:
            stale_rows += 1
            continue
        latent_file = Path(latent_path)
        if not latent_file.exists():
            stale_rows += 1
            continue
        collision_key = str(latent_file)
        previous_source = output_to_source.get(collision_key)
        if previous_source is not None and previous_source != source_image:
            raise RuntimeError(
                f"Manifest collision detected for {collision_key}: {previous_source} vs {source_image}"
            )
        output_to_source[collision_key] = str(source_image)
        rows_by_source[str(source_image)] = row
        valid_rows.append(row)

    return valid_rows, rows_by_source, output_to_source, stale_rows


def _merge_rank_jsonl(output_root: Path, split_name: str, world_size: int) -> None:
    merged_path = output_root / f"{split_name}_latents.jsonl"
    seen_sources: set[str] = set()
    with merged_path.open("w", encoding="utf-8") as merged_handle:
        for rank in range(world_size):
            shard_path = output_root / f"{split_name}_latents.rank{rank:04d}.jsonl"
            if not shard_path.exists():
                continue
            with shard_path.open("r", encoding="utf-8") as shard_handle:
                for line in shard_handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    source_image = str(row.get("source_image", ""))
                    if source_image in seen_sources:
                        continue
                    seen_sources.add(source_image)
                    merged_handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_progress(dist_ctx: DistributedContext, split_name: str, records: list[dict]) -> tqdm:
    return tqdm(
        records,
        total=len(records),
        desc=f"cache:{split_name}:r{dist_ctx.rank}",
        dynamic_ncols=True,
        position=dist_ctx.local_rank if dist_ctx.enabled else 0,
        disable=len(records) == 0,
    )




def _load_manifest_records(manifest_path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(manifest_path)
    rows = load_jsonl(manifest_path)
    base_dir = manifest_path.parent
    resolved_rows: list[dict[str, Any]] = []
    for row in rows:
        row_out = dict(row)
        image = str(row_out.get("image", "")).strip()
        if image:
            image_path = Path(image)
            if not image_path.is_absolute():
                row_out["image"] = str((base_dir / image_path).resolve())
        resolved_rows.append(row_out)
    return resolved_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode 4D clips into cached VQ-VAE latents.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_json(args.config)
    config = resolve_paths(
        config,
        Path(args.config).resolve().parent,
        [
            ("vqvae_checkpoint",),
            ("manifest_paths", "train"),
            ("manifest_paths", "val"),
            ("manifest_paths", "test"),
            ("cache", "output_root"),
        ],
    )
    dist_ctx = setup_distributed(config)
    try:
        seed_everything(int(config.get("seed", 42)) + dist_ctx.rank)
        output_root = ensure_dir(config["cache"]["output_root"])
        if dist_ctx.is_main:
            save_resolved_config(config, output_root)
        barrier(dist_ctx)

        device = dist_ctx.device
        amp_enabled = bool(config.get("amp", True)) and device.type == "cuda"

        model = build_vqvae(config["model"]).to(device)
        checkpoint = torch.load(config["vqvae_checkpoint"], map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        dtype_name = config["cache"].get("dtype", "float16")
        cache_dtype = np.float16 if dtype_name == "float16" else np.float32
        max_items = config["cache"].get("max_items")
        quantized = bool(config["cache"].get("quantized", True))
        stats_split = str(config["cache"].get("stats_split", "train"))

        data_cfg = config.get("data", {})
        skip_corrupt_files = bool(data_cfg.get("skip_corrupt_files", False))
        skip_corrupt_log_limit = int(data_cfg.get("skip_corrupt_log_limit", 20))

        total_sum = 0.0
        total_sq_sum = 0.0
        total_count = 0
        total_skipped_corrupt = 0
        total_resumed = 0
        total_stale_rows = 0

        for split_name, manifest_path in config["manifest_paths"].items():
            records = _limit_records(_load_manifest_records(manifest_path), max_items)
            shard_items = _shard_records(records, dist_ctx)
            shard_manifest_path = _split_manifest_path(output_root, split_name, dist_ctx)
            existing_rows, rows_by_source, seen_output_paths, stale_rows = _load_existing_records(shard_manifest_path)
            if stale_rows > 0:
                save_jsonl(shard_manifest_path, existing_rows)
            completed_local = len(existing_rows)
            resumed_local = 0
            skipped_corrupt_local = 0
            total_stale_rows += stale_rows

            print_main(
                dist_ctx,
                f"[cache] split={split_name} total_records={len(records)} shard_mode=strided rank0_shard={len(shard_items) if dist_ctx.rank == 0 else 'see rank logs'}",
            )

            progress = _build_progress(dist_ctx, split_name, shard_items)
            for global_index, record in progress:
                dataset_name = _dataset_name_for_record(record)
                dataset_dir = ensure_dir(output_root / dataset_name)
                out_path = dataset_dir / _latent_output_relpath(record)
                source_image = str(record["image"])
                ensure_dir(out_path.parent)
                collision_key = str(out_path)
                previous_source = seen_output_paths.get(collision_key)
                if previous_source is not None and previous_source != source_image:
                    raise RuntimeError(
                        f"Output collision detected for {collision_key}: {previous_source} vs {source_image}"
                    )

                try:
                    if args.dry_run:
                        clip = prepare_clip_from_record(record, config["data"])
                        frames = torch.from_numpy(clip[:, None]).to(device)
                        with torch.no_grad():
                            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                                latents = model.encode_stage_2_inputs(frames, quantized=quantized)
                        latents_np = latents.float().cpu().numpy()
                        if dist_ctx.is_main:
                            print(
                                {
                                    "split": split_name,
                                    "global_records_considered": len(records),
                                    "rank": dist_ctx.rank,
                                    "world_size": dist_ctx.world_size,
                                    "global_index": global_index,
                                    "clip_shape": tuple(frames.shape),
                                    "latent_shape": tuple(latents_np.shape),
                                    "task": record["task"],
                                    "source_image": source_image,
                                    "latent_output_path": str(out_path),
                                }
                            )
                        barrier(dist_ctx)
                        return

                    existing_row = rows_by_source.get(source_image)
                    if existing_row is not None and Path(existing_row["latent_path"]).exists():
                        if split_name == stats_split:
                            existing_latents = _read_latent_file(Path(existing_row["latent_path"]))
                            stat_sum, stat_sq_sum, stat_count = _latent_stats_from_array(existing_latents)
                            total_sum += stat_sum
                            total_sq_sum += stat_sq_sum
                            total_count += stat_count
                        resumed_local += 1
                        continue

                    if out_path.exists():
                        existing_latents = _read_latent_file(out_path)
                        cached_record = _cached_record_from_latents(record, out_path, tuple(existing_latents.shape))
                        rows_by_source[source_image] = cached_record
                        seen_output_paths[collision_key] = source_image
                        _append_jsonl_row(shard_manifest_path, cached_record)
                        completed_local += 1
                        resumed_local += 1
                        if split_name == stats_split:
                            stat_sum, stat_sq_sum, stat_count = _latent_stats_from_array(existing_latents)
                            total_sum += stat_sum
                            total_sq_sum += stat_sq_sum
                            total_count += stat_count
                        continue

                    clip = prepare_clip_from_record(record, config["data"])
                    frames = torch.from_numpy(clip[:, None]).to(device)
                    with torch.no_grad():
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                            latents = model.encode_stage_2_inputs(frames, quantized=quantized)
                    latents_np = latents.float().cpu().numpy()
                    with out_path.open("wb") as handle:
                        np.save(handle, latents_np.astype(cache_dtype))

                    cached_record = _cached_record_from_latents(record, out_path, tuple(latents_np.shape))
                    rows_by_source[source_image] = cached_record
                    seen_output_paths[collision_key] = source_image
                    _append_jsonl_row(shard_manifest_path, cached_record)
                    completed_local += 1

                    if split_name == stats_split:
                        stat_sum, stat_sq_sum, stat_count = _latent_stats_from_array(latents_np)
                        total_sum += stat_sum
                        total_sq_sum += stat_sq_sum
                        total_count += stat_count
                except (zipfile.BadZipFile, EOFError, OSError, ValueError) as exc:
                    if not skip_corrupt_files:
                        raise
                    skipped_corrupt_local += 1
                    if skipped_corrupt_local <= skip_corrupt_log_limit:
                        progress.write(
                            f"[cache][rank {dist_ctx.rank}][warn] skip unreadable sample: "
                            f"{record.get('image', '<missing>')} | {type(exc).__name__}: {exc}"
                        )
                    continue

            progress.close()

            total_skipped_corrupt += skipped_corrupt_local
            total_resumed += resumed_local
            global_written = _all_reduce_scalar(completed_local, dist_ctx, dtype=torch.long)
            global_skipped = _all_reduce_scalar(skipped_corrupt_local, dist_ctx, dtype=torch.long)
            global_resumed = _all_reduce_scalar(resumed_local, dist_ctx, dtype=torch.long)
            global_stale_rows = _all_reduce_scalar(stale_rows, dist_ctx, dtype=torch.long)
            barrier(dist_ctx)
            if dist_ctx.is_main and dist_ctx.enabled:
                _merge_rank_jsonl(output_root, split_name, dist_ctx.world_size)
            barrier(dist_ctx)
            print_main(dist_ctx, f"[cache] wrote {global_written} records for split={split_name}")
            if global_resumed > 0:
                print_main(dist_ctx, f"[cache] resumed_existing split={split_name} count={global_resumed}")
            if global_skipped > 0:
                print_main(dist_ctx, f"[cache] skipped_corrupt_total split={split_name} count={global_skipped}")
            if global_stale_rows > 0:
                print_main(dist_ctx, f"[cache] compacted_stale_manifest_rows split={split_name} count={global_stale_rows}")

        global_sum = _all_reduce_scalar(total_sum, dist_ctx, dtype=torch.float64)
        global_sq_sum = _all_reduce_scalar(total_sq_sum, dist_ctx, dtype=torch.float64)
        global_count = _all_reduce_scalar(total_count, dist_ctx, dtype=torch.long)
        global_skipped_total = _all_reduce_scalar(total_skipped_corrupt, dist_ctx, dtype=torch.long)
        global_resumed_total = _all_reduce_scalar(total_resumed, dist_ctx, dtype=torch.long)
        global_stale_rows_total = _all_reduce_scalar(total_stale_rows, dist_ctx, dtype=torch.long)

        mean = float(global_sum) / max(int(global_count), 1)
        variance = max(float(global_sq_sum) / max(int(global_count), 1) - mean * mean, 1e-12)
        std = math.sqrt(variance)
        scale_factor = 1.0 / std
        stats = {
            "stats_split": stats_split,
            "mean": mean,
            "std": std,
            "scale_factor": scale_factor,
            "count": int(global_count),
            "skipped_corrupt_total": int(global_skipped_total),
            "resumed_existing_total": int(global_resumed_total),
            "compacted_stale_manifest_rows_total": int(global_stale_rows_total),
        }
        if dist_ctx.is_main:
            save_json(output_root / "latent_stats.json", stats)
            print(f"[cache] latent std={std:.6f} scale_factor={scale_factor:.6f}")
    finally:
        cleanup_distributed(dist_ctx)


if __name__ == "__main__":
    main()
