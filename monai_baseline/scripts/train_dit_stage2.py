from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for _src in [SRC_ROOT, Path(__file__).resolve().parent]:
    if _src.exists() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from monai_fmri_public.config import load_json, load_jsonl, resolve_paths, save_json, save_resolved_config
from monai_fmri_public.data import (
    LatentCacheDataset,
    PairLatentCacheDataset,
    UniversalSplitPairDataset,
)
from monai_fmri_public.distributed import (
    DistributedContext,
    barrier,
    cleanup_distributed,
    print_main,
    reduce_mean_scalar,
    setup_distributed,
    unwrap_model,
    wrap_ddp,
)
from monai_fmri_public.ema import ExponentialMovingAverage
from monai_fmri_public.models import build_diffusion_unet, build_vqvae
from monai_fmri_public.utils import (
    count_parameters,
    cycle,
    ensure_dir,
    load_partial_weights,
    save_checkpoint,
    seed_everything,
)
from diffusion_utils import Stage2GaussianDiffusion, normalize_prediction_type
from dit3d_model import build_diffusion_dit3d

try:
    import nibabel as nib
except Exception:
    nib = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


NIFTI_AFFINE = np.array(
    [[-2, 0, 0, 96], [0, 2, 0, -112], [0, 0, 2, -90], [0, 0, 0, 1]],
    dtype=np.float32,
)

warnings.filterwarnings(
    "once",
    message=r"Grad strides do not match bucket view strides.*",
    category=UserWarning,
)


DIT_MODEL_TYPES = {"dit", "dit3d"}


def _normalize_model_type(config: dict) -> str:
    model_type = str(config.get("model_type", "dit3d")).strip().lower()
    if model_type in DIT_MODEL_TYPES:
        return "dit3d"
    if model_type == "unet":
        return "unet"
    raise ValueError(f"Unsupported model_type={model_type!r}; expected one of dit3d/dit/unet")


def _normalize_prediction_type_from_config(config: dict) -> str:
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
            fc = F.adaptive_avg_pool1d(fc.unsqueeze(1), channels).squeeze(1)
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

    noisy_input = _build_conditioned_noisy(
        noisy,
        fc_cond=fc_cond,
        anchor_latent=anchor_latent,
        use_anchor_fc_condition=use_anchor_fc_condition,
        anchor_fc_scale=anchor_fc_scale,
    )
    return model(x=noisy_input, timesteps=timesteps, class_labels=class_labels)



def _extract_batch_tensors(batch: dict, *, device: torch.device, scale_factor: float, use_anchor_fc_condition: bool):
    if use_anchor_fc_condition:
        target_latent = batch["target_latent_stacked"].to(device, non_blocking=True).float() * scale_factor
        anchor_latent = batch["anchor_latent_stacked"].to(device, non_blocking=True).float() * scale_factor
        fc_cond = batch["fc_cond"].to(device, non_blocking=True).float()
        has_fc = batch.get("has_fc")
        if has_fc is not None:
            has_fc = has_fc.to(device, non_blocking=True).bool()
    else:
        target_latent = batch["latent_stacked"].to(device, non_blocking=True).float() * scale_factor
        anchor_latent = None
        fc_cond = None
        has_fc = None
    labels = batch["task_id"].to(device, non_blocking=True)
    return target_latent, anchor_latent, fc_cond, has_fc, labels



def _compute_task_histogram(labels: torch.Tensor) -> dict[int, int]:
    if labels.numel() == 0:
        return {}
    unique, counts = torch.unique(labels.detach().cpu(), return_counts=True)
    return {int(u.item()): int(c.item()) for u, c in zip(unique, counts)}


def _load_universal_split_rows(
    split_root: Path,
    datasets: list[str],
    split: str,
    *,
    default_task: str,
) -> list[dict]:
    rows: list[dict] = []
    for dataset in datasets:
        csv_path = split_root / dataset / f"{split}.csv"
        if not csv_path.exists():
            print(f"[stage2][warn] missing split csv: {csv_path}")
            continue
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_out = dict(row)
                for path_field in ("latent_path", "target_latent_path", "fc_embedding_path"):
                    value = str(row_out.get(path_field, "")).strip()
                    if value:
                        path_value = Path(value)
                        if not path_value.is_absolute():
                            row_out[path_field] = str((csv_path.parent / path_value).resolve())
                row_out["dataset"] = dataset
                row_out.setdefault("task", default_task)
                rows.append(row_out)
    return rows


def _resolve_universal_datasets(split_root: Path, configured: list[str] | None) -> list[str]:
    if configured:
        return [str(x) for x in configured]
    if not split_root.exists():
        raise FileNotFoundError(f"universal split root not found: {split_root}")
    out = [p.name for p in split_root.iterdir() if p.is_dir()]
    out.sort()
    return out


def _summarize_pair_rows(
    rows: list[dict],
    *,
    split_name: str,
    use_anchor_fc_condition: bool,
    fc_missing_policy: str,
) -> dict[str, int | str]:
    total = len(rows)
    missing_anchor = 0
    missing_target = 0
    missing_fc = 0
    for row in rows:
        anchor_path = str(row.get("latent_path", "")).strip()
        target_path = str(row.get("target_latent_path", "")).strip()
        fc_path = str(row.get("fc_embedding_path", "")).strip()
        if not anchor_path:
            missing_anchor += 1
        if use_anchor_fc_condition and not target_path:
            missing_target += 1
        if use_anchor_fc_condition and not fc_path:
            missing_fc += 1

    estimated_trainable = max(total - missing_anchor - missing_target, 0)
    if use_anchor_fc_condition and str(fc_missing_policy).strip().lower() == "drop":
        estimated_trainable = max(estimated_trainable - missing_fc, 0)

    return {
        "split": split_name,
        "rows_total": int(total),
        "missing_anchor": int(missing_anchor),
        "missing_target": int(missing_target),
        "missing_fc": int(missing_fc),
        "estimated_trainable": int(estimated_trainable),
        "fc_missing_policy": str(fc_missing_policy),
    }


def _resolve_training_schedule(
    *,
    train_size: int,
    loader_cfg: dict,
    training_cfg: dict,
    world_size: int,
) -> dict[str, int | float | str | None]:
    per_rank_batch = max(1, int(loader_cfg.get("batch_size", 1)))
    global_batch = max(1, int(per_rank_batch * max(1, world_size)))
    steps_per_epoch = max(1, int(math.ceil(max(int(train_size), 1) / float(global_batch))))

    max_steps_raw = training_cfg.get("max_steps")
    has_max_steps = max_steps_raw is not None and str(max_steps_raw).strip() not in {"", "none", "null"}
    max_epochs_raw = training_cfg.get("max_epochs")
    has_max_epochs = max_epochs_raw is not None and str(max_epochs_raw).strip() not in {"", "none", "null"}

    if has_max_steps:
        max_steps = max(1, int(max_steps_raw))
        approx_epochs = float(max_steps) / float(steps_per_epoch)
        return {
            "mode": "max_steps",
            "max_epochs": (int(max_epochs_raw) if has_max_epochs else None),
            "max_steps": int(max_steps),
            "steps_per_epoch": int(steps_per_epoch),
            "per_rank_batch": int(per_rank_batch),
            "global_batch": int(global_batch),
            "explicit_max_steps": int(max_steps),
            "approx_epochs": float(approx_epochs),
        }

    if has_max_epochs:
        max_epochs = max(1, int(max_epochs_raw))
        max_steps = max(1, int(max_epochs * steps_per_epoch))
        return {
            "mode": "max_epochs",
            "max_epochs": int(max_epochs),
            "max_steps": int(max_steps),
            "steps_per_epoch": int(steps_per_epoch),
            "per_rank_batch": int(per_rank_batch),
            "global_batch": int(global_batch),
            "explicit_max_steps": None,
            "approx_epochs": float(max_epochs),
        }

    max_steps = 10000
    approx_epochs = float(max_steps) / float(steps_per_epoch)
    return {
        "mode": "max_steps",
        "max_epochs": None,
        "max_steps": int(max_steps),
        "steps_per_epoch": int(steps_per_epoch),
        "per_rank_batch": int(per_rank_batch),
        "global_batch": int(global_batch),
        "explicit_max_steps": int(max_steps),
        "approx_epochs": float(approx_epochs),
    }



def _format_seconds(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "n/a"
    total = int(round(seconds))
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _broadcast_main_scalar(value: float, dist_ctx: DistributedContext) -> float:
    tensor = torch.tensor([float(value)], device=dist_ctx.device, dtype=torch.float32)
    if dist_ctx.enabled:
        torch.distributed.broadcast(tensor, src=0)
    return float(tensor.item())


def build_loader(dataset, loader_config: dict, shuffle: bool, dist_ctx: DistributedContext) -> tuple[DataLoader, DistributedSampler | None]:
    num_workers = int(loader_config.get("num_workers", 0))
    drop_last = bool(loader_config.get("drop_last", False))
    sampler = None
    if dist_ctx.enabled:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist_ctx.world_size,
            rank=dist_ctx.rank,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=0,
        )
        shuffle = False

    kwargs = {
        "batch_size": int(loader_config.get("batch_size", 1)),
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": bool(loader_config.get("pin_memory", True)),
        "drop_last": drop_last,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(loader_config.get("persistent_workers", True))
    return DataLoader(dataset, **kwargs), sampler


@torch.no_grad()
def evaluate(
    model,
    loader,
    diffusion,
    device,
    amp_enabled: bool,
    max_batches: int,
    scale_factor: float,
    dist_ctx: DistributedContext,
    *,
    use_anchor_fc_condition: bool,
    anchor_fc_scale: float,
    null_class: int,
    model_type: str,
) -> float:
    model.eval()
    local_sum = 0.0
    local_count = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        latents, anchor_latent, fc_cond, has_fc, labels = _extract_batch_tensors(
            batch,
            device=device,
            scale_factor=scale_factor,
            use_anchor_fc_condition=use_anchor_fc_condition,
        )
        conditioned_labels = labels.clone()
        if has_fc is not None:
            conditioned_labels[~has_fc] = null_class
        noise = torch.randn_like(latents)
        timesteps = diffusion.sample_timesteps(latents.shape[0], device)
        noisy = diffusion.add_noise(original_samples=latents, noise=noise, timesteps=timesteps)
        target = diffusion.get_training_target(original_samples=latents, noise=noise, timesteps=timesteps)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            prediction = _forward_diffusion_model(
                model,
                noisy=noisy,
                timesteps=timesteps,
                class_labels=conditioned_labels,
                fc_cond=fc_cond,
                anchor_latent=anchor_latent,
                has_fc=has_fc,
                use_anchor_fc_condition=use_anchor_fc_condition,
                anchor_fc_scale=anchor_fc_scale,
                model_type=model_type,
            )
            loss = F.mse_loss(prediction.float(), target.float())
        local_sum += float(loss.item())
        local_count += 1
    model.train()

    local_avg = local_sum / max(local_count, 1)
    return reduce_mean_scalar(local_avg, dist_ctx)


def _safe_meta_value(values, index: int, default):
    if values is None:
        return default
    if isinstance(values, torch.Tensor):
        if values.ndim == 0:
            return values.item()
        if index < values.shape[0]:
            item = values[index]
            return item.item() if hasattr(item, "item") else item
        return default
    if isinstance(values, (list, tuple)):
        return values[index] if index < len(values) else default
    return values


def _safe_token(value) -> str:
    text = str(value)
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out) or "unknown"


def _slice_for_plane(volume: np.ndarray, axis: int) -> np.ndarray:
    center = volume.shape[axis] // 2
    if axis == 0:
        x = volume[center, :, :]
    elif axis == 1:
        x = volume[:, center, :]
    elif axis == 2:
        x = volume[:, :, center]
    else:
        raise ValueError(f"Unsupported axis: {axis}")
    return np.rot90(x)


def _finite_clean(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _robust_limits(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    merged = np.concatenate([a.reshape(-1), b.reshape(-1)])
    finite = merged[np.isfinite(merged)]
    if finite.size == 0:
        return -1.0, 1.0
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.0))
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
        c = float(np.mean(finite))
        s = float(np.std(finite))
        s = s if s > 1e-6 else 1.0
        return c - s, c + s
    return lo, hi


def _save_nii_tdhw(x_tdhw: np.ndarray, out_nii: Path) -> None:
    if nib is None:
        raise RuntimeError("stage2_visualization enabled but nibabel is unavailable.")
    out_nii.parent.mkdir(parents=True, exist_ok=True)
    dhwt = np.transpose(_finite_clean(x_tdhw), (1, 2, 3, 0))
    nib.save(nib.Nifti1Image(dhwt.astype(np.float32), NIFTI_AFFINE), str(out_nii))


def _metrics(pred_tdhw: np.ndarray, gt_tdhw: np.ndarray) -> dict[str, float]:
    pred = _finite_clean(pred_tdhw).reshape(-1).astype(np.float64)
    gt = _finite_clean(gt_tdhw).reshape(-1).astype(np.float64)
    mse = float(np.mean((pred - gt) ** 2))
    rmse = float(np.sqrt(max(mse, 0.0)))
    pred_m = pred - pred.mean()
    gt_m = gt - gt.mean()
    den = float(np.sqrt((pred_m * pred_m).sum() * (gt_m * gt_m).sum()))
    corr = float((pred_m * gt_m).sum() / den) if den > 1e-12 else 0.0
    ss_res = float(((gt - pred) ** 2).sum())
    ss_tot = float(((gt - gt.mean()) ** 2).sum())
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    gt_std = float(np.std(gt))
    pred_std = float(np.std(pred))
    std_ratio = float(pred_std / gt_std) if gt_std > 1e-12 else 0.0
    return {
        "mse": mse,
        "rmse": rmse,
        "corr": corr,
        "r2": r2,
        "std_ratio": std_ratio,
    }


def _latent_metrics(pred_latent: torch.Tensor, gt_latent: torch.Tensor) -> dict[str, float]:
    pred = pred_latent.detach().float().reshape(-1)
    gt = gt_latent.detach().float().reshape(-1)
    mse = float(F.mse_loss(pred, gt).item())
    pred_centered = pred - pred.mean()
    gt_centered = gt - gt.mean()
    denom = float(torch.linalg.norm(pred_centered).item() * torch.linalg.norm(gt_centered).item())
    corr = float(torch.dot(pred_centered, gt_centered).item() / denom) if denom > 1.0e-12 else 0.0
    cosine = float(F.cosine_similarity(pred.unsqueeze(0), gt.unsqueeze(0), dim=1).item())
    return {
        "latent_mse": mse,
        "latent_corr": corr,
        "latent_cosine": cosine,
    }


def _save_preview(pred_tdhw: np.ndarray, gt_tdhw: np.ndarray, out_png: Path, *, dpi: int, step: int) -> None:
    if plt is None:
        raise RuntimeError("stage2_visualization enabled but matplotlib is unavailable.")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    t = int(pred_tdhw.shape[0] // 2)
    pred = _finite_clean(pred_tdhw[t])
    gt = _finite_clean(gt_tdhw[t])
    err = np.abs(pred - gt)
    vmin, vmax = _robust_limits(pred, gt)
    emax = float(np.percentile(err[np.isfinite(err)], 99.0)) if np.isfinite(err).any() else 1.0
    emax = emax if emax > 1e-8 else 1.0

    fig, axes = plt.subplots(3, 3, figsize=(9.5, 9.5), constrained_layout=True)
    planes = ["sagittal", "coronal", "axial"]
    for row, axis in enumerate([0, 1, 2]):
        gt_s = _slice_for_plane(gt, axis)
        pred_s = _slice_for_plane(pred, axis)
        err_s = _slice_for_plane(err, axis)
        axes[row, 0].imshow(gt_s, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row, 1].imshow(pred_s, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row, 2].imshow(err_s, cmap="magma", vmin=0.0, vmax=emax)
        axes[row, 0].set_ylabel(planes[row])
        if row == 0:
            axes[row, 0].set_title("GT")
            axes[row, 1].set_title("Pred")
            axes[row, 2].set_title("Abs Error")
        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.suptitle(f"step={int(step)} middle_frame={t}", fontsize=10)
    fig.savefig(out_png, dpi=int(dpi))
    plt.close(fig)


def _load_vqvae_model_cfg(config_json: Path) -> dict:
    cfg = load_json(config_json)
    if isinstance(cfg, dict) and isinstance(cfg.get("model"), dict):
        return dict(cfg["model"])
    if isinstance(cfg, dict) and "num_channels" in cfg and "downsample_parameters" in cfg and "upsample_parameters" in cfg:
        return dict(cfg)
    raise ValueError(f"Cannot find VQ-VAE model config in: {config_json}")


@torch.no_grad()
def _decode_tdhw(vqvae: torch.nn.Module, latent_tcdhw: torch.Tensor, decode_batch_size: int) -> np.ndarray:
    if latent_tcdhw.ndim != 5:
        raise ValueError(f"Expected latent [T,C,D,H,W], got {tuple(latent_tcdhw.shape)}")
    outs = []
    for chunk in torch.split(latent_tcdhw.float(), max(1, int(decode_batch_size)), dim=0):
        outs.append(vqvae.decode_stage_2_outputs(chunk))
    decoded = torch.cat(outs, dim=0)
    if decoded.ndim != 5 or int(decoded.shape[1]) != 1:
        raise ValueError(f"Expected decoded [T,1,D,H,W], got {tuple(decoded.shape)}")
    return decoded[:, 0].detach().cpu().numpy().astype(np.float32)


def _scheduler_prev_sample(step_output):
    if hasattr(step_output, "prev_sample"):
        return step_output.prev_sample
    if isinstance(step_output, (tuple, list)) and len(step_output) > 0:
        return step_output[0]
    if isinstance(step_output, torch.Tensor):
        return step_output
    raise TypeError(f"Unsupported scheduler.step output type: {type(step_output)}")


@torch.no_grad()
def _pred_to_quantized_residual(vqvae: torch.nn.Module, latent_tcdhw: torch.Tensor) -> float:
    if not hasattr(vqvae, "quantize"):
        return 0.0
    quantized, _ = vqvae.quantize(latent_tcdhw.float())
    return float(F.mse_loss(latent_tcdhw.float(), quantized.float()).item())


@torch.no_grad()
def _run_stage2_visualization(
    *,
    model: torch.nn.Module,
    scheduler,
    vqvae: torch.nn.Module,
    batch: dict,
    device: torch.device,
    step: int,
    output_dir: Path,
    metrics_csv_path: Path,
    num_samples: int,
    num_inference_steps: int,
    decode_batch_size: int,
    num_frames: int,
    latent_channels: int,
    scale_factor: float,
    use_anchor_fc_condition: bool,
    anchor_fc_scale: float,
    null_class: int,
    model_type: str,
    base_seed: int,
    dpi: int,
    amp_enabled: bool,
) -> None:
    target_key = "target_latent_stacked" if "target_latent_stacked" in batch else "latent_stacked"
    target_stacked = batch[target_key]
    if int(target_stacked.shape[0]) <= 0:
        return
    bs = min(int(target_stacked.shape[0]), int(num_samples))
    gt_stacked = target_stacked[:bs].to(device, non_blocking=True).float()
    stacked_channels = int(gt_stacked.shape[1])
    depth, height, width = int(gt_stacked.shape[2]), int(gt_stacked.shape[3]), int(gt_stacked.shape[4])
    expected_channels = int(num_frames * latent_channels)
    if stacked_channels != expected_channels:
        raise ValueError(f"stacked channels mismatch: got {stacked_channels} expected {expected_channels}")

    if use_anchor_fc_condition and "fc_cond" in batch:
        fc_cond = batch["fc_cond"][:bs].to(device, non_blocking=True).float()
    else:
        fc_cond = None
    labels = batch["task_id"][:bs].to(device, non_blocking=True)
    conditioned_labels = labels.clone()
    if use_anchor_fc_condition and "has_fc" in batch:
        has_fc = batch["has_fc"][:bs].to(device, non_blocking=True).bool()
        conditioned_labels[~has_fc] = null_class

    shape = (bs, stacked_channels, depth, height, width)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(base_seed + int(step) * 10007))
    sample = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)

    was_training = model.training
    model.eval()
    scheduler.set_timesteps(int(num_inference_steps), device=device)
    vis_start = time.monotonic()
    vis_total_steps = max(int(len(scheduler.timesteps)), 1)
    vis_log_every = max(1, min(100, vis_total_steps // 10 if vis_total_steps >= 10 else 1))
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    for vis_idx, t in enumerate(scheduler.timesteps, start=1):
        timesteps = torch.full((bs,), int(t), device=device, dtype=torch.long)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
            noise_pred = _forward_diffusion_model(
                model,
                noisy=sample,
                timesteps=timesteps,
                class_labels=conditioned_labels,
                fc_cond=fc_cond,
                anchor_latent=batch["anchor_latent_stacked"][:bs].to(device, non_blocking=True).float() * scale_factor if use_anchor_fc_condition and "anchor_latent_stacked" in batch else None,
                has_fc=batch["has_fc"][:bs].to(device, non_blocking=True).bool() if use_anchor_fc_condition and "has_fc" in batch else None,
                use_anchor_fc_condition=use_anchor_fc_condition,
                anchor_fc_scale=anchor_fc_scale,
                model_type=model_type,
            )
        sample = _scheduler_prev_sample(scheduler.step(noise_pred, t, sample))
        if vis_idx == 1 or vis_idx % vis_log_every == 0 or vis_idx == vis_total_steps:
            elapsed = max(time.monotonic() - vis_start, 1e-6)
            step_per_sec = float(vis_idx) / float(elapsed)
            remaining = max(vis_total_steps - vis_idx, 0)
            eta_seconds = float(remaining) / float(step_per_sec) if step_per_sec > 0 else float("inf")
            print(
                "[stage2][visualization] "
                f"sampling_step={vis_idx}/{vis_total_steps} "
                f"elapsed={_format_seconds(elapsed)} "
                f"eta={_format_seconds(eta_seconds)} "
                f"speed={step_per_sec:.2f}it/s"
            )
    if was_training:
        model.train()

    pred_stacked = sample / float(scale_factor)
    pred_latents = pred_stacked.reshape(bs, num_frames, latent_channels, depth, height, width)
    gt_latents = gt_stacked.reshape(bs, num_frames, latent_channels, depth, height, width)

    rows: list[dict[str, str | int | float]] = []
    for i in range(bs):
        subject = _safe_meta_value(batch.get("subject"), i, "unknown")
        session = _safe_meta_value(batch.get("session"), i, "unknown")
        dataset = _safe_meta_value(batch.get("dataset"), i, "unknown")
        task = _safe_meta_value(batch.get("task"), i, "unknown")
        prefix = (
            f"step_{int(step):06d}"
            f"_sample_{int(i):02d}"
            f"_sub-{_safe_token(subject)}"
            f"_ses-{_safe_token(session)}"
            f"_ds-{_safe_token(dataset)}"
            f"_task-{_safe_token(task)}"
        )

        pred_tdhw = _decode_tdhw(vqvae, pred_latents[i], decode_batch_size)
        gt_tdhw = _decode_tdhw(vqvae, gt_latents[i], decode_batch_size)
        m = _metrics(pred_tdhw, gt_tdhw)
        latent_m = _latent_metrics(pred_latents[i], gt_latents[i])
        pred_to_quantized_residual = _pred_to_quantized_residual(vqvae, pred_latents[i])

        pred_nii = output_dir / "pred_nii" / f"{prefix}.nii.gz"
        gt_nii = output_dir / "gt_nii" / f"{prefix}.nii.gz"
        preview_png = output_dir / "preview" / f"{prefix}.png"
        _save_nii_tdhw(pred_tdhw, pred_nii)
        _save_nii_tdhw(gt_tdhw, gt_nii)
        _save_preview(pred_tdhw, gt_tdhw, preview_png, dpi=dpi, step=step)

        rows.append(
            {
                "step": int(step),
                "sample_index": int(i),
                "subject": str(subject),
                "session": str(session),
                "dataset": str(dataset),
                "task": str(task),
                "corr_decoded": float(m["corr"]),
                "r2_decoded": float(m["r2"]),
                "rmse_decoded": float(m["rmse"]),
                "std_ratio_decoded": float(m["std_ratio"]),
                "latent_mse": float(latent_m["latent_mse"]),
                "latent_corr": float(latent_m["latent_corr"]),
                "latent_cosine": float(latent_m["latent_cosine"]),
                "pred_to_quantized_residual": float(pred_to_quantized_residual),
                "pred_nii": str(pred_nii),
                "gt_nii": str(gt_nii),
                "preview_png": str(preview_png),
            }
        )

    with metrics_csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "sample_index",
                "subject",
                "session",
                "dataset",
                "task",
                "corr_decoded",
                "r2_decoded",
                "rmse_decoded",
                "std_ratio_decoded",
                "latent_mse",
                "latent_corr",
                "latent_cosine",
                "pred_to_quantized_residual",
                "pred_nii",
                "gt_nii",
                "preview_png",
            ],
        )
        for row in rows:
            writer.writerow(row)


def _print_startup_summary(
    config: dict,
    dist_ctx: DistributedContext,
    model: torch.nn.Module,
    model_type: str,
    train_size: int,
    val_size: int,
    num_tasks: int,
    scale_factor: float,
    schedule_info: dict[str, int | float | str | None],
) -> None:
    loader_cfg = config.get("loader", {})
    training_cfg = config.get("training", {})
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    frozen_params = total_params - trainable_params
    per_rank_batch = int(schedule_info["per_rank_batch"])
    global_batch = int(schedule_info["global_batch"])
    print_main(dist_ctx, "=" * 88)
    print_main(dist_ctx, "[stage2] startup summary")
    print_main(dist_ctx, f"[stage2] ddp_enabled={dist_ctx.enabled} world_size={dist_ctx.world_size} device={dist_ctx.device}")
    print_main(dist_ctx, f"[stage2] model_type={model_type}")
    print_main(dist_ctx, f"[stage2] train_size={train_size} val_size={val_size} num_tasks={num_tasks}")
    print_main(dist_ctx, f"[stage2] latent_scale_factor={scale_factor:.6f}")
    print_main(dist_ctx, f"[stage2] batch_size_per_rank={per_rank_batch} global_batch_size={global_batch}")
    print_main(
        dist_ctx,
        f"[stage2] params_total={total_params:,} params_trainable={trainable_params:,} params_frozen={frozen_params:,}",
    )
    if str(schedule_info.get("mode")) == "max_epochs":
        print_main(
            dist_ctx,
            f"[stage2] schedule=max_epochs max_epochs={int(schedule_info['max_epochs'])} "
            f"steps_per_epoch={int(schedule_info['steps_per_epoch'])} "
            f"computed_max_steps={int(schedule_info['max_steps'])}",
        )
    else:
        max_epochs_note = ""
        if schedule_info.get("max_epochs") is not None:
            max_epochs_note = f" config_max_epochs={int(schedule_info['max_epochs'])}"
        print_main(
            dist_ctx,
            f"[stage2] schedule=max_steps max_steps={int(schedule_info['max_steps'])} "
            f"approx_epochs={float(schedule_info['approx_epochs']):.2f} "
            f"steps_per_epoch={int(schedule_info['steps_per_epoch'])}{max_epochs_note}",
        )
    print_main(
        dist_ctx,
        f"[stage2] log_every={int(training_cfg.get('log_every', 10))} "
        f"val_every={int(training_cfg.get('val_every', 500))} "
        f"save_every={int(training_cfg.get('save_every', 1000))} "
        f"class_dropout_prob={float(training_cfg.get('class_dropout_prob', 0.05))} "
        f"ema_decay={float(training_cfg.get('ema_decay', 0.9999))}",
    )
    if str(schedule_info.get("mode")) == "max_epochs":
        print_main(
            dist_ctx,
            f"[stage2] epoch_maintenance "
            f"val_every_epochs={int(training_cfg.get('val_every_epochs', 1))} "
            f"save_every_epochs={int(training_cfg.get('save_every_epochs', 1))}",
        )
    print_main(dist_ctx, "=" * 88)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train stage2 diffusion denoiser (DiT3D or UNet) on stacked latents.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-from", default=None, help="Optional path to a stage2 checkpoint (diffusion_last.pt) for true resume.")
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
            ("resume_from",),
            ("warm_start", "checkpoint_path"),
            ("data", "universal_split_root"),
            ("stage2_visualization", "vqvae_model_config_json"),
            ("stage2_visualization", "vqvae_checkpoint"),
        ],
    )
    dist_ctx = setup_distributed(config)
    try:
        seed_everything(int(config.get("seed", 42)) + dist_ctx.rank)

        output_dir = ensure_dir(config["output_dir"])
        if dist_ctx.is_main:
            save_resolved_config(config, output_dir)
        barrier(dist_ctx)

        task_vocab = load_json(config["task_vocab_path"])
        latent_stats = load_json(config["latent_stats_path"])
        if dist_ctx.is_main:
            save_json(output_dir / "task_vocab.json", task_vocab)
        barrier(dist_ctx)
        scale_factor = float(latent_stats.get("scale_factor", 1.0))

        device = dist_ctx.device
        amp_enabled = bool(config.get("amp", True)) and device.type == "cuda"

        model_config = config["model"]
        model_type = _normalize_model_type(config)
        data_cfg = config.get("data", {})
        use_universal_split_csv = bool(data_cfg.get("use_universal_split_csv", False))
        if use_universal_split_csv:
            split_root = Path(data_cfg["universal_split_root"])
            datasets = _resolve_universal_datasets(split_root, data_cfg.get("datasets"))
            default_task = str(data_cfg.get("default_task", "rest"))
            train_records = _load_universal_split_rows(split_root, datasets, "train", default_task=default_task)
            val_records = _load_universal_split_rows(split_root, datasets, "val", default_task=default_task)
            print_main(
                dist_ctx,
                f"[stage2] universal_split enabled root={split_root} "
                f"datasets={','.join(datasets)} train_rows={len(train_records)} val_rows={len(val_records)}",
            )
        else:
            train_records = load_jsonl(config["manifest_paths"]["train"])
            val_records = load_jsonl(config["manifest_paths"]["val"])
        conditioning_cfg = config.get("conditioning", {})
        use_anchor_fc_condition = bool(conditioning_cfg.get("use_anchor_fc_condition", True))
        anchor_fc_scale = float(conditioning_cfg.get("anchor_fc_scale", 1.0))
        pair_mode = str(conditioning_cfg.get("pair_mode", "next_only"))
        fc_cfg = dict(conditioning_cfg.get("fc", {})) if isinstance(conditioning_cfg.get("fc", {}), dict) else {}
        fc_missing_policy = str(conditioning_cfg.get("fc_missing_policy", "drop")).strip().lower()

        train_row_stats = _summarize_pair_rows(
            train_records,
            split_name="train",
            use_anchor_fc_condition=use_anchor_fc_condition,
            fc_missing_policy=fc_missing_policy,
        )
        val_row_stats = _summarize_pair_rows(
            val_records,
            split_name="val",
            use_anchor_fc_condition=use_anchor_fc_condition,
            fc_missing_policy=fc_missing_policy,
        )
        print_main(
            dist_ctx,
            f"[stage2] precheck split=train rows_total={train_row_stats['rows_total']} "
            f"estimated_trainable={train_row_stats['estimated_trainable']} "
            f"missing_anchor={train_row_stats['missing_anchor']} "
            f"missing_target={train_row_stats['missing_target']} "
            f"missing_fc={train_row_stats['missing_fc']} "
            f"fc_missing_policy={train_row_stats['fc_missing_policy']}",
        )
        print_main(
            dist_ctx,
            f"[stage2] precheck split=val rows_total={val_row_stats['rows_total']} "
            f"estimated_trainable={val_row_stats['estimated_trainable']} "
            f"missing_anchor={val_row_stats['missing_anchor']} "
            f"missing_target={val_row_stats['missing_target']} "
            f"missing_fc={val_row_stats['missing_fc']} "
            f"fc_missing_policy={val_row_stats['fc_missing_policy']}",
        )

        if use_anchor_fc_condition and use_universal_split_csv:
            train_dataset = UniversalSplitPairDataset(
                train_records,
                task_vocab,
                expected_num_frames=int(model_config["num_frames"]),
                expected_latent_channels=int(model_config["latent_channels"]),
                fc_config=fc_cfg,
                fc_missing_policy=fc_missing_policy,
                split="train",
            )
            val_dataset = UniversalSplitPairDataset(
                val_records,
                task_vocab,
                expected_num_frames=int(model_config["num_frames"]),
                expected_latent_channels=int(model_config["latent_channels"]),
                fc_config=fc_cfg,
                fc_missing_policy=fc_missing_policy,
                split="val",
            )
        elif use_anchor_fc_condition:
            fc_cfg = dict(conditioning_cfg.get("fc", {})) if isinstance(conditioning_cfg.get("fc", {}), dict) else {}
            train_dataset = PairLatentCacheDataset(
                train_records,
                task_vocab,
                expected_num_frames=int(model_config["num_frames"]),
                expected_latent_channels=int(model_config["latent_channels"]),
                pair_mode=pair_mode,
                split="train",
                fc_config=fc_cfg,
            )
            val_dataset = PairLatentCacheDataset(
                val_records,
                task_vocab,
                expected_num_frames=int(model_config["num_frames"]),
                expected_latent_channels=int(model_config["latent_channels"]),
                pair_mode=pair_mode,
                split="val",
                fc_config=fc_cfg,
            )
        else:
            train_dataset = LatentCacheDataset(
                train_records,
                task_vocab,
                expected_num_frames=int(model_config["num_frames"]),
                expected_latent_channels=int(model_config["latent_channels"]),
            )
            val_dataset = LatentCacheDataset(
                val_records,
                task_vocab,
                expected_num_frames=int(model_config["num_frames"]),
                expected_latent_channels=int(model_config["latent_channels"]),
            )
        train_loader, train_sampler = build_loader(train_dataset, config["loader"], shuffle=True, dist_ctx=dist_ctx)
        val_loader, _ = build_loader(val_dataset, config["loader"], shuffle=False, dist_ctx=dist_ctx)
        schedule_info = _resolve_training_schedule(
            train_size=len(train_dataset),
            loader_cfg=config.get("loader", {}),
            training_cfg=config.get("training", {}),
            world_size=dist_ctx.world_size,
        )

        if model_type in DIT_MODEL_TYPES:
            base_model = build_diffusion_dit3d(
                config=config,
                num_tasks=len(task_vocab),
                fc_dim=int(fc_cfg.get("dim_hint", 0) or 0),
            ).to(device)
        elif model_type == "unet":
            base_model = build_diffusion_unet(model_config, num_tasks=len(task_vocab)).to(device)
        else:
            raise ValueError(f"Unsupported model_type={model_type!r}")
        prediction_type = _normalize_prediction_type_from_config(config)
        _print_startup_summary(
            config,
            dist_ctx,
            base_model,
            model_type,
            len(train_dataset),
            len(val_dataset),
            len(task_vocab),
            scale_factor,
            schedule_info,
        )

        if dist_ctx.is_main:
            model_audit = {
                "model_type": model_type,
                "prediction_type": prediction_type,
                "uses_anchor_latent_condition": bool(use_anchor_fc_condition),
                "fc_missing_policy": fc_missing_policy,
                "train_row_stats": train_row_stats,
                "val_row_stats": val_row_stats,
            }
            if hasattr(base_model, "get_model_audit"):
                model_audit.update(base_model.get_model_audit())
            save_json(output_dir / "model_audit.json", model_audit)

        warm_start = config.get("warm_start", {})
        if warm_start.get("checkpoint_path"):
            report = load_partial_weights(
                base_model,
                warm_start["checkpoint_path"],
                strict=bool(warm_start.get("strict", False)),
            )
            if dist_ctx.is_main:
                save_json(output_dir / "warm_start_report.json", report)
                print(f"Loaded warm start: {report['matched_keys']} matched keys")
        barrier(dist_ctx)

        model = wrap_ddp(
            base_model,
            dist_ctx,
            find_unused_parameters=bool(config.get("training", {}).get("ddp_find_unused_parameters", False)),
        )

        scheduler = Stage2GaussianDiffusion(**config["diffusion"]).to(device)
        optimizer_config = config["optimizer"]
        optimizer = torch.optim.AdamW(
            unwrap_model(model).parameters(),
            lr=float(optimizer_config.get("lr", 1e-4)),
            betas=tuple(optimizer_config.get("betas", [0.9, 0.99])),
            weight_decay=float(optimizer_config.get("weight_decay", 0.01)),
        )
        ema = ExponentialMovingAverage(unwrap_model(model), decay=float(config["training"].get("ema_decay", 0.9999)))
        null_class = len(task_vocab)

        resume_from = args.resume_from or config.get("resume_from")
        start_step = 0
        best_val_loss = float("inf")
        if resume_from:
            ckpt_path = Path(resume_from)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"resume checkpoint not found: {ckpt_path}")
            checkpoint = torch.load(str(ckpt_path), map_location="cpu")
            unwrap_model(model).load_state_dict(checkpoint["model_state_dict"], strict=True)
            if checkpoint.get("ema_model_state_dict") is not None:
                ema.load_state_dict(checkpoint["ema_model_state_dict"])
                ema.to(device)
            if checkpoint.get("optimizer_state_dict") is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_step = int(checkpoint.get("step", 0))
            best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
            print_main(dist_ctx, f"[stage2] resumed from {ckpt_path} @ step={start_step} best_val_loss={best_val_loss:.6f}")
        barrier(dist_ctx)

        if args.dry_run:
            batch = next(iter(train_loader))
            latents, anchor_latent, fc_cond, has_fc, labels = _extract_batch_tensors(
                batch,
                device=device,
                scale_factor=scale_factor,
                use_anchor_fc_condition=use_anchor_fc_condition,
            )
            conditioned_labels = labels.clone()
            if has_fc is not None:
                conditioned_labels[~has_fc] = null_class
            noise = torch.randn_like(latents)
            timesteps = scheduler.sample_timesteps(latents.shape[0], device)
            noisy = scheduler.add_noise(original_samples=latents, noise=noise, timesteps=timesteps)
            target = scheduler.get_training_target(original_samples=latents, noise=noise, timesteps=timesteps)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                prediction = _forward_diffusion_model(
                    model,
                    noisy=noisy,
                    timesteps=timesteps,
                    class_labels=conditioned_labels,
                    fc_cond=fc_cond,
                    anchor_latent=anchor_latent,
                    has_fc=has_fc,
                    use_anchor_fc_condition=use_anchor_fc_condition,
                    anchor_fc_scale=anchor_fc_scale,
                    model_type=model_type,
                )
            if dist_ctx.is_main:
                print({
                    "device": str(device),
                    "ddp_enabled": dist_ctx.enabled,
                    "world_size": dist_ctx.world_size,
                    "latent_shape": tuple(latents.shape),
                    "fc_cond_shape": (None if fc_cond is None else tuple(fc_cond.shape)),
                    "anchor_shape": (None if anchor_latent is None else tuple(anchor_latent.shape)),
                    "prediction_shape": tuple(prediction.shape),
                    "prediction_type": prediction_type,
                    "task_histogram": _compute_task_histogram(labels),
                    "num_tasks": len(task_vocab),
                    "scale_factor": scale_factor,
                    "use_anchor_fc_condition": use_anchor_fc_condition,
                    "anchor_fc_scale": anchor_fc_scale,
                    "model_type": model_type,
                })
            return

        training = config["training"]
        max_steps = int(schedule_info["max_steps"])
        steps_per_epoch = int(schedule_info["steps_per_epoch"])
        epoch_mode = str(schedule_info.get("mode")) == "max_epochs"
        log_every = int(training.get("log_every", 10))
        val_every = int(training.get("val_every", 500))
        save_every = int(training.get("save_every", 1000))
        val_every_epochs = max(1, int(training.get("val_every_epochs", 1)))
        save_every_epochs = max(1, int(training.get("save_every_epochs", 1)))
        class_dropout_prob = float(training.get("class_dropout_prob", 0.05))
        max_val_batches = int(training.get("max_val_batches", 10))
        stage2_vis_cfg = config.get("stage2_visualization", {})
        stage2_vis_enabled = bool(stage2_vis_cfg.get("enabled", False))
        stage2_vis_every = max(1, int(stage2_vis_cfg.get("every_n_steps", save_every)))
        stage2_vis_every_epochs = max(1, int(stage2_vis_cfg.get("every_n_epochs", 1)))
        stage2_vis_trigger_mode_raw = str(stage2_vis_cfg.get("trigger_mode", "auto")).strip().lower()
        if stage2_vis_trigger_mode_raw not in {"auto", "steps", "epochs"}:
            raise ValueError(
                f"Invalid stage2_visualization.trigger_mode={stage2_vis_trigger_mode_raw!r}; expected one of auto/steps/epochs"
            )
        if stage2_vis_trigger_mode_raw == "auto":
            stage2_vis_trigger_mode = "epochs" if epoch_mode else "steps"
        else:
            stage2_vis_trigger_mode = stage2_vis_trigger_mode_raw
        stage2_vis_use_steps = stage2_vis_trigger_mode == "steps"
        stage2_vis_num_samples = max(1, int(stage2_vis_cfg.get("num_samples", 2)))
        stage2_vis_num_inference_steps = max(1, int(stage2_vis_cfg.get("num_inference_steps", 200)))
        stage2_vis_decode_batch_size = max(1, int(stage2_vis_cfg.get("decode_batch_size", 8)))
        stage2_vis_dpi = int(stage2_vis_cfg.get("dpi", 130))
        stage2_vis_seed = int(stage2_vis_cfg.get("seed", int(config.get("seed", 42))))
        stage2_vis_use_ema_for_sampling = bool(stage2_vis_cfg.get("use_ema_for_sampling", True))
        stage2_vis_output_dir = output_dir / str(
            stage2_vis_cfg.get("output_subdir", "visualizations/stage2_next_window")
        )
        stage2_vis_metrics_csv = stage2_vis_output_dir / "metrics.csv"

        stage2_vis_scheduler = None
        stage2_vis_vqvae = None
        if stage2_vis_enabled:
            if nib is None:
                raise RuntimeError("stage2_visualization.enabled=true requires nibabel, but import failed.")
            if plt is None:
                raise RuntimeError("stage2_visualization.enabled=true requires matplotlib, but import failed.")
            vqvae_model_cfg_json_raw = str(stage2_vis_cfg.get("vqvae_model_config_json", "")).strip()
            vqvae_checkpoint_raw = str(stage2_vis_cfg.get("vqvae_checkpoint", "")).strip()
            if not vqvae_model_cfg_json_raw:
                raise ValueError("Missing stage2_visualization.vqvae_model_config_json")
            if not vqvae_checkpoint_raw:
                raise ValueError("Missing stage2_visualization.vqvae_checkpoint")
            vqvae_model_cfg_json = Path(vqvae_model_cfg_json_raw)
            vqvae_checkpoint = Path(vqvae_checkpoint_raw)
            if not vqvae_model_cfg_json.exists():
                raise FileNotFoundError(f"VQ-VAE model config not found: {vqvae_model_cfg_json}")
            if not vqvae_checkpoint.exists():
                raise FileNotFoundError(f"VQ-VAE checkpoint not found: {vqvae_checkpoint}")
            vqvae_model_cfg = _load_vqvae_model_cfg(vqvae_model_cfg_json)
            stage2_vis_vqvae = build_vqvae(vqvae_model_cfg).to(device)
            stage1_ckpt = torch.load(str(vqvae_checkpoint), map_location="cpu")
            stage1_state = stage1_ckpt.get("model_state_dict")
            if not isinstance(stage1_state, dict):
                raise ValueError(f"Invalid stage1 checkpoint, missing model_state_dict: {vqvae_checkpoint}")
            stage2_vis_vqvae.load_state_dict(stage1_state, strict=True)
            stage2_vis_vqvae.eval()
            stage2_vis_vqvae.requires_grad_(False)
            stage2_vis_scheduler = Stage2GaussianDiffusion(**config["diffusion"]).to(device)

            if dist_ctx.is_main:
                ensure_dir(stage2_vis_output_dir / "pred_nii")
                ensure_dir(stage2_vis_output_dir / "gt_nii")
                ensure_dir(stage2_vis_output_dir / "preview")
                if not stage2_vis_metrics_csv.exists():
                    with stage2_vis_metrics_csv.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=[
                                "step",
                                "sample_index",
                                "subject",
                                "session",
                                "dataset",
                                "task",
                                "corr_decoded",
                                "r2_decoded",
                                "rmse_decoded",
                                "std_ratio_decoded",
                                "latent_mse",
                                "latent_corr",
                                "latent_cosine",
                                "pred_to_quantized_residual",
                                "pred_nii",
                                "gt_nii",
                                "preview_png",
                            ],
                        )
                        writer.writeheader()
            barrier(dist_ctx)
            print_main(
                dist_ctx,
                f"[stage2] visualization enabled trigger={stage2_vis_trigger_mode} "
                f"every_n_steps={stage2_vis_every} every_n_epochs={stage2_vis_every_epochs} "
                f"num_samples={stage2_vis_num_samples} infer_steps={stage2_vis_num_inference_steps} "
                f"use_ema_for_sampling={stage2_vis_use_ema_for_sampling} "
                f"out={stage2_vis_output_dir}",
            )

        train_iterator = cycle(train_loader, train_sampler)
        if start_step >= max_steps:
            print_main(dist_ctx, f"[stage2] checkpoint step ({start_step}) >= max_steps ({max_steps}); nothing to do.")
            return

        model.train()
        progress_total = int(schedule_info["max_epochs"]) if epoch_mode else int(max_steps)
        progress_initial = (float(start_step) / float(steps_per_epoch)) if epoch_mode else float(start_step)
        progress_desc = "stage2-epoch" if epoch_mode else "stage2"
        progress = tqdm(
            total=progress_total,
            initial=progress_initial,
            desc=progress_desc,
            dynamic_ncols=True,
            disable=not dist_ctx.is_main,
        )
        train_start_monotonic = time.monotonic()
        for step in range(start_step + 1, max_steps + 1):
            batch = next(train_iterator)
            current_epoch = int((step - 1) // steps_per_epoch + 1)
            step_in_epoch = int((step - 1) % steps_per_epoch + 1)
            epoch_end = bool((step % steps_per_epoch == 0) or (step == max_steps))
            latents, anchor_latent, fc_cond, has_fc, labels = _extract_batch_tensors(
                batch,
                device=device,
                scale_factor=scale_factor,
                use_anchor_fc_condition=use_anchor_fc_condition,
            )
            drop_mask = torch.rand(labels.shape[0], device=device) < class_dropout_prob
            if has_fc is not None:
                drop_mask = drop_mask | (~has_fc)
            conditioned_labels = labels.clone()
            conditioned_labels[drop_mask] = null_class

            noise = torch.randn_like(latents)
            timesteps = scheduler.sample_timesteps(latents.shape[0], device)
            noisy = scheduler.add_noise(original_samples=latents, noise=noise, timesteps=timesteps)
            target = scheduler.get_training_target(original_samples=latents, noise=noise, timesteps=timesteps)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                prediction = _forward_diffusion_model(
                    model,
                    noisy=noisy,
                    timesteps=timesteps,
                    class_labels=conditioned_labels,
                    fc_cond=fc_cond,
                    anchor_latent=anchor_latent,
                    has_fc=has_fc,
                    use_anchor_fc_condition=use_anchor_fc_condition,
                    anchor_fc_scale=anchor_fc_scale,
                    model_type=model_type,
                )
                loss = F.mse_loss(prediction.float(), target.float())
            loss.backward()
            optimizer.step()
            ema.update(unwrap_model(model))

            mean_loss = reduce_mean_scalar(loss, dist_ctx)
            mean_dropped = reduce_mean_scalar(float(drop_mask.sum().item()), dist_ctx)
            timestep_mean = reduce_mean_scalar(float(timesteps.float().mean().item()), dist_ctx)
            progress_log_trigger = bool(step % log_every == 0 or step == 1 or epoch_end)

            if dist_ctx.is_main:
                progress.set_postfix(
                    loss=f"{mean_loss:.6f}",
                    null_drop=f"{mean_dropped:.2f}",
                    t_mean=f"{timestep_mean:.1f}",
                    epoch=f"{current_epoch}",
                    step_in_epoch=f"{step_in_epoch}/{steps_per_epoch}",
                    best_val=f"{best_val_loss:.6f}" if best_val_loss < float("inf") else "n/a",
                    refresh=progress_log_trigger,
                )
                if progress_log_trigger:
                    elapsed = max(time.monotonic() - train_start_monotonic, 1e-6)
                    finished_steps = max(step - start_step, 1)
                    steps_per_sec = float(finished_steps) / float(elapsed)
                    remaining_steps = max(max_steps - step, 0)
                    eta_seconds = (float(remaining_steps) / steps_per_sec) if steps_per_sec > 0 else float("inf")
                    total_epochs = float(schedule_info["max_epochs"]) if epoch_mode else float(schedule_info["approx_epochs"])
                    progress_epoch = float(step) / float(steps_per_epoch)
                    progress.write(
                        "[stage2][progress] "
                        f"epoch={progress_epoch:.2f}/{total_epochs:.2f} "
                        f"step={step}/{max_steps} "
                        f"loss={mean_loss:.6f} "
                        f"speed={steps_per_sec:.2f}step/s "
                        f"elapsed={_format_seconds(elapsed)} "
                        f"eta={_format_seconds(eta_seconds)}"
                    )

            if epoch_mode:
                run_val = bool((epoch_end and (current_epoch % val_every_epochs == 0)) or (step == max_steps))
            else:
                run_val = bool(step % val_every == 0 or step == max_steps)

            if run_val:
                val_loss = evaluate(
                    model,
                    val_loader,
                    scheduler,
                    device,
                    amp_enabled,
                    max_val_batches,
                    scale_factor,
                    dist_ctx,
                    use_anchor_fc_condition=use_anchor_fc_condition,
                    anchor_fc_scale=anchor_fc_scale,
                    null_class=null_class,
                    model_type=model_type,
                )
                if dist_ctx.is_main:
                    progress.write(f"[stage2] step={step} epoch={current_epoch} val_loss={val_loss:.6f}")
                if dist_ctx.is_main and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        output_dir / "diffusion_best.pt",
                        {
                            "model_state_dict": unwrap_model(model).state_dict(),
                            "ema_model_state_dict": ema.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "step": step,
                            "epoch": current_epoch,
                            "best_val_loss": best_val_loss,
                            "task_vocab": task_vocab,
                            "scale_factor": scale_factor,
                            "config": config,
                        },
                    )
                best_val_sync = best_val_loss if dist_ctx.is_main else float("inf")
                best_val_loss = _broadcast_main_scalar(best_val_sync, dist_ctx)
                barrier(dist_ctx)

            if epoch_mode:
                # Keep "last" checkpoint semantics stable: update every epoch end (and final step),
                # while archiving periodic epoch snapshots separately.
                run_last_save = bool(epoch_end or step == max_steps)
                run_periodic_save = bool(epoch_end and (current_epoch % save_every_epochs == 0))
            else:
                run_last_save = bool(step % save_every == 0 or step == max_steps)
                run_periodic_save = bool(step % save_every == 0)

            if run_last_save or run_periodic_save:
                if dist_ctx.is_main:
                    ckpt_payload = {
                        "model_state_dict": unwrap_model(model).state_dict(),
                        "ema_model_state_dict": ema.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step,
                        "epoch": current_epoch,
                        "best_val_loss": best_val_loss,
                        "task_vocab": task_vocab,
                        "scale_factor": scale_factor,
                        "config": config,
                    }

                    if run_last_save:
                        save_checkpoint(output_dir / "diffusion_last.pt", ckpt_payload)

                    if run_periodic_save:
                        if epoch_mode and epoch_end:
                            save_checkpoint(output_dir / f"diffusion_ep{current_epoch:04d}.pt", ckpt_payload)
                        elif not epoch_mode:
                            save_checkpoint(output_dir / f"diffusion_step{step:07d}.pt", ckpt_payload)
                barrier(dist_ctx)

            if stage2_vis_use_steps:
                run_vis = bool(stage2_vis_enabled and (step % stage2_vis_every == 0 or step == max_steps))
            else:
                run_vis = bool(
                    stage2_vis_enabled
                    and ((epoch_end and (current_epoch % stage2_vis_every_epochs == 0)) or (step == max_steps))
                )

            if run_vis:
                barrier(dist_ctx)
                if dist_ctx.is_main:
                    vis_model = unwrap_model(model)
                    model_state_before_ema = None
                    if stage2_vis_use_ema_for_sampling:
                        model_state_before_ema = {k: v.detach().clone() for k, v in vis_model.state_dict().items()}
                        ema.copy_to(vis_model)
                    try:
                        _run_stage2_visualization(
                            model=vis_model,
                            scheduler=stage2_vis_scheduler,
                            vqvae=stage2_vis_vqvae,
                            batch=batch,
                            device=device,
                            step=step,
                            output_dir=stage2_vis_output_dir,
                            metrics_csv_path=stage2_vis_metrics_csv,
                            num_samples=stage2_vis_num_samples,
                            num_inference_steps=stage2_vis_num_inference_steps,
                            decode_batch_size=stage2_vis_decode_batch_size,
                            num_frames=int(model_config["num_frames"]),
                            latent_channels=int(model_config["latent_channels"]),
                            scale_factor=scale_factor,
                            use_anchor_fc_condition=use_anchor_fc_condition,
                            anchor_fc_scale=anchor_fc_scale,
                            null_class=null_class,
                            model_type=model_type,
                            base_seed=stage2_vis_seed,
                            dpi=stage2_vis_dpi,
                            amp_enabled=amp_enabled,
                        )
                    finally:
                        if model_state_before_ema is not None:
                            vis_model.load_state_dict(model_state_before_ema, strict=True)
                barrier(dist_ctx)

            if dist_ctx.is_main:
                if epoch_mode:
                    progress.update(1.0 / float(steps_per_epoch))
                else:
                    progress.update(1)

        if dist_ctx.is_main:
            progress.close()
        print_main(dist_ctx, f"Stage-2 training finished. Best validation loss: {best_val_loss:.6f}")
    finally:
        cleanup_distributed(dist_ctx)


if __name__ == "__main__":
    main()
