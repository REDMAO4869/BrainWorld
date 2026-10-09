from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for _src in [SRC_ROOT, Path(__file__).resolve().parent]:
    if _src.exists() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from monai_fmri_public.config import load_json, resolve_paths, save_json
from monai_fmri_public.data import load_latent_array
from monai_fmri_public.distributed import (
    DistributedContext,
    barrier,
    cleanup_distributed,
    print_main,
    reduce_mean_scalar,
    setup_distributed,
)
from monai_fmri_public.models import build_diffusion_unet
from monai_fmri_public.utils import ensure_dir, seed_everything
from diffusion_utils import Stage2GaussianDiffusion, normalize_prediction_type
from dit3d_model import build_diffusion_dit3d


DIT_MODEL_TYPES = {"dit", "dit3d"}


def _normalize_model_type(config: dict[str, Any]) -> str:
    model_type = str(config.get("model_type", "dit3d")).strip().lower()
    if model_type in DIT_MODEL_TYPES:
        return "dit3d"
    if model_type == "unet":
        return "unet"
    raise ValueError(f"Unsupported model_type={model_type!r}; expected one of dit3d/dit/unet")


def _normalize_prediction_type_from_config(config: dict[str, Any]) -> str:
    diffusion_cfg = config.get("diffusion", {}) if isinstance(config.get("diffusion", {}), dict) else {}
    return normalize_prediction_type(diffusion_cfg.get("prediction_type", "v"))


def _build_conditioned_noisy(
    noisy: torch.Tensor,
    *,
    fc_cond: torch.Tensor | None,
    anchor_latent: torch.Tensor | None,
    use_anchor_fc_condition: bool,
    anchor_fc_scale: float,
) -> torch.Tensor:
    x = noisy
    if use_anchor_fc_condition and fc_cond is not None:
        fc = fc_cond.float()
        if fc.ndim == 1:
            fc = fc.unsqueeze(0)
        if fc.ndim > 2:
            fc = fc.flatten(start_dim=1)
        channels = int(noisy.shape[1])
        if int(fc.shape[1]) != channels:
            fc = torch.nn.functional.adaptive_avg_pool1d(fc.unsqueeze(1), channels).squeeze(1)
        x = x + float(anchor_fc_scale) * fc[:, :, None, None, None]
    if use_anchor_fc_condition and anchor_latent is not None:
        x = x + 0.1 * anchor_latent.float()
    return x


def _forward_diffusion_model(
    model: torch.nn.Module,
    *,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    class_labels: torch.Tensor,
    fc_cond: torch.Tensor | None,
    anchor_latent: torch.Tensor | None,
    has_fc: torch.Tensor | None,
    use_anchor_fc_condition: bool,
    anchor_fc_scale: float,
    model_type: str,
) -> torch.Tensor:
    model_type_norm = str(model_type).strip().lower()
    if model_type_norm in DIT_MODEL_TYPES:
        return model(
            x=noisy,
            timesteps=timesteps,
            class_labels=class_labels,
            fc_cond=(fc_cond if use_anchor_fc_condition else None),
            anchor_latent=(anchor_latent if use_anchor_fc_condition else None),
            has_fc=(has_fc if use_anchor_fc_condition else None),
        )

    model_input = _build_conditioned_noisy(
        noisy,
        fc_cond=fc_cond,
        anchor_latent=anchor_latent,
        use_anchor_fc_condition=use_anchor_fc_condition,
        anchor_fc_scale=anchor_fc_scale,
    )
    return model(x=model_input, timesteps=timesteps, class_labels=class_labels)


def _load_universal_split_rows(
    split_root: Path,
    datasets: list[str],
    split: str,
    *,
    default_task: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        csv_path = split_root / dataset / f"{split}.csv"
        if not csv_path.exists():
            print(f"[infer][warn] missing split csv: {csv_path}")
            continue
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r = dict(row)
                for path_field in ("latent_path", "target_latent_path", "fc_embedding_path"):
                    value = str(r.get(path_field, "")).strip()
                    if value:
                        path_value = Path(value)
                        if not path_value.is_absolute():
                            r[path_field] = str((csv_path.parent / path_value).resolve())
                r["dataset"] = dataset
                r.setdefault("task", default_task)
                rows.append(r)
    return rows


def _resolve_universal_datasets(split_root: Path, configured: list[str] | None) -> list[str]:
    if configured:
        return [str(x) for x in configured]
    out = [p.name for p in split_root.iterdir() if p.is_dir()]
    out.sort()
    return out


def _load_fc_vector(fc_path: str, *, npz_key: str) -> np.ndarray:
    path = Path(fc_path)
    if not path.exists():
        raise FileNotFoundError(f"fc file not found: {fc_path}")
    if path.suffix == ".npy":
        arr = np.asarray(np.load(path), dtype=np.float32)
    elif path.suffix == ".npz":
        with np.load(path) as data:
            if npz_key not in {"", "auto"} and npz_key in data:
                arr = np.asarray(data[npz_key], dtype=np.float32)
            else:
                keys = list(data.keys())
                if len(keys) == 0:
                    raise ValueError(f"Empty npz fc file: {fc_path}")
                arr = np.asarray(data[keys[0]], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported fc file type: {fc_path}")
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1).astype(np.float32, copy=False)


def _infer_latent_spatial_size(records: list[dict[str, Any]]) -> list[int]:
    for row in records:
        latent_path = str(row.get("latent_path", "")).strip()
        if not latent_path:
            continue
        p = Path(latent_path)
        if not p.exists():
            continue
        arr = load_latent_array(p)
        if arr.ndim == 5:
            return [int(arr.shape[2]), int(arr.shape[3]), int(arr.shape[4])]
    raise RuntimeError("Unable to infer latent spatial size from input rows")


class NextWindowInferDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        task_vocab: dict[str, int],
        *,
        output_root: Path,
        skip_existing: bool,
        fc_missing_policy: str,
        fc_npz_key: str,
        default_task: str,
        fc_dim_hint: int,
    ) -> None:
        self.task_vocab = task_vocab
        self.output_root = output_root
        self.skip_existing = bool(skip_existing)
        self.fc_missing_policy = str(fc_missing_policy).strip().lower()
        self.fc_npz_key = str(fc_npz_key)
        self.default_task = str(default_task)
        self.fc_dim = int(fc_dim_hint)
        if self.fc_missing_policy not in {"drop", "zero", "error"}:
            raise ValueError(f"Unsupported fc_missing_policy: {self.fc_missing_policy}")

        samples: list[dict[str, Any]] = []
        skipped_existing = 0
        dropped_no_target = 0
        dropped_no_fc = 0
        for row in rows:
            target_path = str(row.get("target_latent_path", "")).strip()
            if not target_path:
                dropped_no_target += 1
                continue
            dataset = str(row.get("dataset", "unknown"))
            out_path = output_root / dataset / Path(target_path).name
            if self.skip_existing and out_path.exists():
                skipped_existing += 1
                continue
            fc_path = str(row.get("fc_embedding_path", "")).strip()
            if self.fc_missing_policy == "drop" and not fc_path:
                dropped_no_fc += 1
                continue
            task = str(row.get("task", self.default_task))
            if task not in self.task_vocab:
                task = self.default_task
            samples.append(
                {
                    "subject": str(row.get("Subject", row.get("subject", "unknown"))),
                    "dataset": dataset,
                    "task": task,
                    "latent_path": str(row.get("latent_path", "")).strip(),
                    "target_path": target_path,
                    "fc_path": fc_path,
                    "out_path": str(out_path),
                }
            )

        self.samples = samples
        self.stats = {
            "rows_total": len(rows),
            "rows_kept": len(samples),
            "rows_skipped_existing": skipped_existing,
            "rows_dropped_no_target": dropped_no_target,
            "rows_dropped_no_fc": dropped_no_fc,
        }
        if len(self.samples) == 0:
            raise ValueError(f"No samples to infer after filtering: {self.stats}")

        if self.fc_dim <= 0:
            for s in self.samples:
                if not s["fc_path"]:
                    continue
                try:
                    self.fc_dim = int(_load_fc_vector(s["fc_path"], npz_key=self.fc_npz_key).shape[0])
                    break
                except Exception:
                    continue
        if self.fc_dim <= 0 and self.fc_missing_policy != "drop":
            raise ValueError("Unable to determine fc dim (all fc missing?)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[int(idx)]
        fc_path = str(s["fc_path"]).strip()
        has_fc = bool(fc_path)
        if has_fc:
            fc_vec = _load_fc_vector(fc_path, npz_key=self.fc_npz_key)
            if int(fc_vec.shape[0]) != int(self.fc_dim):
                raise ValueError(
                    f"Inconsistent fc dim for {fc_path}: got {fc_vec.shape[0]} expected {self.fc_dim}"
                )
        else:
            if self.fc_missing_policy == "error":
                raise ValueError(f"Missing fc path for sample target={s['target_path']}")
            fc_vec = np.zeros((self.fc_dim,), dtype=np.float32)

        return {
            "fc_cond": torch.from_numpy(fc_vec).float(),
            "has_fc": torch.tensor(bool(has_fc), dtype=torch.bool),
            "task_id": torch.tensor(int(self.task_vocab[s["task"]]), dtype=torch.long),
            "dataset": s["dataset"],
            "subject": s["subject"],
            "latent_path": s["latent_path"],
            "target_path": s["target_path"],
            "out_path": s["out_path"],
        }


def _scheduler_prev_sample(step_output):
    if hasattr(step_output, "prev_sample"):
        return step_output.prev_sample
    if isinstance(step_output, (tuple, list)) and len(step_output) > 0:
        return step_output[0]
    if isinstance(step_output, torch.Tensor):
        return step_output
    raise TypeError(f"Unsupported scheduler.step output type: {type(step_output)}")


def build_loader(dataset: Dataset, loader_config: dict[str, Any], dist_ctx: DistributedContext) -> tuple[DataLoader, DistributedSampler | None]:
    num_workers = int(loader_config.get("num_workers", 0))
    sampler = None
    if dist_ctx.enabled:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist_ctx.world_size,
            rank=dist_ctx.rank,
            shuffle=False,
            drop_last=False,
            seed=0,
        )

    kwargs = {
        "batch_size": int(loader_config.get("batch_size", 1)),
        "shuffle": False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": bool(loader_config.get("pin_memory", True)),
        "drop_last": False,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(loader_config.get("persistent_workers", True))
    return DataLoader(dataset, **kwargs), sampler


@torch.no_grad()
def run_infer(
    model: torch.nn.Module,
    scheduler,
    loader: DataLoader,
    *,
    device: torch.device,
    dist_ctx: DistributedContext,
    use_anchor_fc_condition: bool,
    anchor_fc_scale: float,
    model_type: str,
    null_class: int,
    num_frames: int,
    latent_channels: int,
    latent_spatial_size: list[int],
    scale_factor: float,
    num_inference_steps: int,
    base_seed: int,
) -> dict[str, Any]:
    model.eval()
    scheduler.set_timesteps(num_inference_steps, device=device)

    saved = 0
    skipped = 0
    dropped_no_fc = 0

    progress = tqdm(total=len(loader), desc=f"infer-r{dist_ctx.rank}", dynamic_ncols=True, disable=not dist_ctx.is_main)
    stacked_channels = int(num_frames * latent_channels)

    for batch in loader:
        fc_cond = batch["fc_cond"].to(device, non_blocking=True)
        has_fc = batch["has_fc"].to(device, non_blocking=True).bool()
        labels = batch["task_id"].to(device, non_blocking=True)
        conditioned_labels = labels.clone()
        conditioned_labels[~has_fc] = null_class
        dropped_no_fc += int((~has_fc).sum().item())

        bs = int(fc_cond.shape[0])
        shape = (
            bs,
            stacked_channels,
            int(latent_spatial_size[0]),
            int(latent_spatial_size[1]),
            int(latent_spatial_size[2]),
        )

        first_out = str(batch["out_path"][0]) if bs > 0 else "none"
        h = int(hashlib.md5(first_out.encode("utf-8")).hexdigest()[:8], 16)
        gen = torch.Generator(device=device)
        gen.manual_seed(int(base_seed + h + dist_ctx.rank * 1000003))
        sample = torch.randn(shape, generator=gen, device=device, dtype=torch.float32)

        anchor_latent = None
        if use_anchor_fc_condition and batch.get("latent_path") is not None:
            anchor_list = []
            for latent_path in batch["latent_path"]:
                anchor = load_latent_array(str(latent_path)).astype(np.float32, copy=False)
                anchor = anchor.reshape(anchor.shape[0] * anchor.shape[1], *anchor.shape[2:])
                anchor_list.append(anchor)
            anchor_latent = torch.from_numpy(np.stack(anchor_list, axis=0)).to(device=device, dtype=torch.float32) * float(scale_factor)

        for t in scheduler.timesteps:
            timesteps = torch.full((bs,), int(t), device=device, dtype=torch.long)
            noise_pred = _forward_diffusion_model(
                model,
                noisy=sample,
                timesteps=timesteps,
                class_labels=conditioned_labels,
                fc_cond=fc_cond,
                anchor_latent=anchor_latent,
                has_fc=has_fc,
                use_anchor_fc_condition=use_anchor_fc_condition,
                anchor_fc_scale=anchor_fc_scale,
                model_type=model_type,
            )
            sample = _scheduler_prev_sample(scheduler.step(noise_pred, t, sample))

        sample = sample / float(scale_factor)
        sample = sample.reshape(
            bs,
            int(num_frames),
            int(latent_channels),
            int(latent_spatial_size[0]),
            int(latent_spatial_size[1]),
            int(latent_spatial_size[2]),
        )

        sample_np = sample.detach().cpu().numpy().astype(np.float32)
        out_paths = batch["out_path"]
        for i in range(bs):
            out_path = Path(out_paths[i])
            ensure_dir(out_path.parent)
            if out_path.exists():
                skipped += 1
                continue
            if out_path.suffix.lower() == ".npz":
                np.savez_compressed(out_path, sample_np[i])
            elif out_path.suffix.lower() == ".npy":
                np.save(out_path, sample_np[i])
            else:
                np.save(out_path.with_suffix(out_path.suffix + ".npy"), sample_np[i])
            saved += 1

        if dist_ctx.is_main:
            progress.update(1)
            progress.set_postfix(saved=saved, skipped=skipped)

    if dist_ctx.is_main:
        progress.close()

    return {
        "saved": int(saved),
        "skipped_existing_runtime": int(skipped),
        "missing_fc_runtime": int(dropped_no_fc),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer next-window latents from fc embeddings using stage2 DiT/UNet.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path. If omitted, use infer.checkpoint_path or output_dir/diffusion_last.pt")
    parser.add_argument("--use-ema", action="store_true", help="Load ema_model_state_dict from checkpoint for sampling.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_json(args.config)
    config = resolve_paths(
        config,
        Path(args.config).resolve().parent,
        [
            ("manifest_paths", "train"),
            ("manifest_paths", "val"),
            ("task_vocab_path",),
            ("latent_stats_path",),
            ("output_dir",),
            ("data", "universal_split_root"),
            ("infer", "checkpoint_path"),
            ("infer", "output_root"),
        ],
    )
    infer_cfg = config.get("infer", {})

    dist_ctx = setup_distributed(config)
    try:
        seed_everything(int(config.get("seed", 42)) + dist_ctx.rank)
        model_type = _normalize_model_type(config)

        task_vocab = load_json(config["task_vocab_path"])
        latent_stats = load_json(config["latent_stats_path"])
        scale_factor = float(latent_stats.get("scale_factor", 1.0))

        conditioning_cfg = config.get("conditioning", {})
        fc_cfg = conditioning_cfg.get("fc", {}) if isinstance(conditioning_cfg.get("fc", {}), dict) else {}
        use_anchor_fc_condition = bool(conditioning_cfg.get("use_anchor_fc_condition", True))
        anchor_fc_scale = float(conditioning_cfg.get("anchor_fc_scale", 1.0))
        fc_missing_policy = str(conditioning_cfg.get("fc_missing_policy", "zero")).strip().lower()

        data_cfg = config.get("data", {})
        if not bool(data_cfg.get("use_universal_split_csv", False)):
            raise ValueError("infer currently requires data.use_universal_split_csv=true")
        split_root = Path(data_cfg["universal_split_root"])
        datasets = _resolve_universal_datasets(split_root, data_cfg.get("datasets"))
        default_task = str(data_cfg.get("default_task", "rest"))

        infer_splits = infer_cfg.get("splits", ["test"])
        if isinstance(infer_splits, str):
            infer_splits = [infer_splits]
        infer_splits = [str(s) for s in infer_splits]

        rows: list[dict[str, Any]] = []
        for split in infer_splits:
            rows.extend(_load_universal_split_rows(split_root, datasets, split, default_task=default_task))

        output_root = ensure_dir(infer_cfg.get("output_root", str(Path(config["output_dir"]) / "infer_next_latents")))
        skip_existing = bool(infer_cfg.get("skip_existing", True))

        dataset = NextWindowInferDataset(
            rows,
            task_vocab,
            output_root=output_root,
            skip_existing=skip_existing,
            fc_missing_policy=fc_missing_policy,
            fc_npz_key=str(fc_cfg.get("npz_key", "auto")),
            default_task=default_task,
            fc_dim_hint=int(fc_cfg.get("dim_hint", 0) or 0),
        )

        latent_spatial_size = infer_cfg.get("latent_spatial_size")
        if latent_spatial_size is None:
            latent_spatial_size = _infer_latent_spatial_size(rows)
        latent_spatial_size = [int(x) for x in latent_spatial_size]

        device = dist_ctx.device
        if model_type in DIT_MODEL_TYPES:
            model = build_diffusion_dit3d(
                config=config,
                num_tasks=len(task_vocab),
                fc_dim=int(fc_cfg.get("dim_hint", 0) or 0),
            ).to(device)
        elif model_type == "unet":
            model = build_diffusion_unet(config["model"], num_tasks=len(task_vocab)).to(device)
        else:
            raise ValueError(f"Unsupported model_type={model_type!r}")
        prediction_type = _normalize_prediction_type_from_config(config)
        scheduler = Stage2GaussianDiffusion(**config["diffusion"]).to(device)

        ckpt_path = args.checkpoint or infer_cfg.get("checkpoint_path")
        if not ckpt_path:
            ckpt_path = str(Path(config["output_dir"]) / "diffusion_last.pt")
        ckpt_path = str(ckpt_path)
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        use_ema = bool(args.use_ema or infer_cfg.get("use_ema", False))
        if use_ema:
            ema_state = checkpoint.get("ema_model_state_dict")
            if not isinstance(ema_state, dict):
                raise ValueError(f"checkpoint missing ema_model_state_dict while use_ema=true: {ckpt_path}")
            model.load_state_dict(ema_state, strict=True)
        else:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)

        if dist_ctx.is_main:
            print("=" * 88)
            print("[stage2-infer] startup summary")
            print(f"[stage2-infer] ddp_enabled={dist_ctx.enabled} world_size={dist_ctx.world_size} device={device}")
            print(f"[stage2-infer] model_type={model_type}")
            print(f"[stage2-infer] datasets={','.join(datasets)} splits={','.join(infer_splits)}")
            print(f"[stage2-infer] rows_total={len(rows)}")
            print(f"[stage2-infer] rows_kept={len(dataset)}")
            print(f"[stage2-infer] filter_stats={dataset.stats}")
            print(f"[stage2-infer] output_root={output_root}")
            print(f"[stage2-infer] checkpoint={ckpt_path} use_ema={use_ema} prediction_type={prediction_type}")
            print(f"[stage2-infer] latent_spatial_size={latent_spatial_size} scale_factor={scale_factor:.6f}")
            print("=" * 88)

        loader_cfg = infer_cfg.get("loader", config.get("loader", {}))
        loader, _ = build_loader(dataset, loader_cfg, dist_ctx)

        if args.dry_run:
            batch = next(iter(loader))
            print_main(dist_ctx, {
                "batch_size": int(batch["fc_cond"].shape[0]),
                "fc_cond_shape": tuple(batch["fc_cond"].shape),
                "has_fc_true": int(batch["has_fc"].sum().item()),
                "example_out": str(batch["out_path"][0]),
            })
            return

        num_inference_steps = int(infer_cfg.get("num_inference_steps", 1000))
        model.eval()
        null_class = len(task_vocab)
        runtime_stats = run_infer(
            model,
            scheduler,
            loader,
            device=device,
            dist_ctx=dist_ctx,
            use_anchor_fc_condition=use_anchor_fc_condition,
            anchor_fc_scale=anchor_fc_scale,
            model_type=model_type,
            null_class=null_class,
            num_frames=int(config["model"]["num_frames"]),
            latent_channels=int(config["model"]["latent_channels"]),
            latent_spatial_size=latent_spatial_size,
            scale_factor=scale_factor,
            num_inference_steps=num_inference_steps,
            base_seed=int(infer_cfg.get("sampling_seed", int(config.get("seed", 42)))),
        )

        summary_local = {
            **dataset.stats,
            **runtime_stats,
            "rank": int(dist_ctx.rank),
        }
        mean_saved = reduce_mean_scalar(float(runtime_stats["saved"]), dist_ctx)
        mean_missing = reduce_mean_scalar(float(runtime_stats["missing_fc_runtime"]), dist_ctx)

        if dist_ctx.is_main:
            summary = {
                "config": str(args.config),
                "checkpoint": ckpt_path,
                "use_ema": bool(use_ema),
                "output_root": str(output_root),
                "datasets": datasets,
                "splits": infer_splits,
                "ddp_world_size": int(dist_ctx.world_size),
                "stats_rank0_local": summary_local,
                "stats_mean_saved_per_rank": float(mean_saved),
                "stats_mean_missing_fc_per_rank": float(mean_missing),
            }
            save_json(Path(output_root) / "infer_summary.json", summary)
            print(f"[stage2-infer] done. summary -> {Path(output_root) / 'infer_summary.json'}")

        barrier(dist_ctx)
    finally:
        cleanup_distributed(dist_ctx)


if __name__ == "__main__":
    main()
