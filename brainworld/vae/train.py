from __future__ import annotations

import argparse
import copy
import csv
import os
import random
import time
from contextlib import nullcontext
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from brainworld.vae.data import audit_to_dict, build_splits_from_config, collate_batch
from brainworld.vae.losses import compute_vae_losses
from brainworld.vae.model import build_model_from_config
from brainworld.vae.utils import ensure_dir, fg_bg_metrics, load_json, mae, mse, psnr, save_json, set_seed
from brainworld.vae.wavelet import Haar3DSpatial


def _make_timestamped_run_dir(base_dir: str) -> str:
    base_dir = str(base_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    cand = os.path.join(base_dir, stamp)
    if not os.path.exists(cand):
        return ensure_dir(cand)
    i = 1
    while True:
        c = os.path.join(base_dir, f"{stamp}_{i:02d}")
        if not os.path.exists(c):
            return ensure_dir(c)
        i += 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BrainWorld VAE for 4D fMRI")
    p.add_argument("--config", required=True, help="Path to training JSON config")
    return p.parse_args()


def _dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _dist_rank() -> int:
    if not _dist_is_initialized():
        return 0
    return int(dist.get_rank())


def _is_main_process() -> bool:
    return _dist_rank() == 0


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        t = v.strip().lower()
        if t in {"1", "true", "yes", "y", "on"}:
            return True
        if t in {"0", "false", "no", "n", "off", "", "none", "null"}:
            return False
    return bool(v)


def _device_from_cfg(cfg: Dict) -> torch.device:
    dev = str(cfg.get("device", "auto"))
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(dev)


def _resolve_gpu_ids(cfg: Dict) -> List[int]:
    tr_cfg = cfg.get("training", {})
    gpu_ids_cfg = tr_cfg.get("gpu_ids", None)
    n_cuda = int(torch.cuda.device_count())
    if n_cuda <= 0:
        return []

    if gpu_ids_cfg is None:
        return list(range(n_cuda))
    if isinstance(gpu_ids_cfg, int):
        gpu_ids = [int(gpu_ids_cfg)]
    else:
        gpu_ids = [int(x) for x in gpu_ids_cfg]

    gpu_ids_in_range = [g for g in gpu_ids if 0 <= g < n_cuda]
    if len(gpu_ids_in_range) == 0 and len(gpu_ids) > 0:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cvd.strip() != "":
            gpu_ids_in_range = list(range(n_cuda))
    if len(gpu_ids_in_range) == 0:
        raise ValueError(f"training.gpu_ids is empty or invalid. Available GPU count: {n_cuda}")
    return gpu_ids_in_range


def build_loaders(cfg: Dict, *, ddp: bool, rank: int, world_size: int):
    ds_train, ds_val, ds_test, audits = build_splits_from_config(cfg)

    tr_cfg = cfg.get("training", {})
    bs = int(tr_cfg.get("batch_size", 1))
    nw = int(tr_cfg.get("num_workers", 2))

    train_sampler = DistributedSampler(ds_train, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(ds_val, num_replicas=world_size, rank=rank, shuffle=False) if ddp else None
    test_sampler = DistributedSampler(ds_test, num_replicas=world_size, rank=rank, shuffle=False) if ddp else None

    train_loader = DataLoader(
        ds_train,
        batch_size=bs,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=nw,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=bs,
        shuffle=False,
        sampler=val_sampler,
        num_workers=nw,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=bs,
        shuffle=False,
        sampler=test_sampler,
        num_workers=nw,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    samplers = {"train": train_sampler, "val": val_sampler, "test": test_sampler}
    return train_loader, val_loader, test_loader, audits, samplers


def _get_core_model(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else m


def _to_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


def _count_params(model: nn.Module) -> tuple[int, int]:
    core = _get_core_model(model)
    total = sum(p.numel() for p in core.parameters())
    trainable = sum(p.numel() for p in core.parameters() if p.requires_grad)
    return int(total), int(trainable)


def _dp_chunk_count(batch_size: int, n_devices: int) -> int:
    if batch_size <= 0 or n_devices <= 0:
        return 0
    chunk_size = (batch_size + n_devices - 1) // n_devices
    return (batch_size + chunk_size - 1) // chunk_size


def _should_bypass_dp(model: nn.Module, batch_size: int) -> bool:
    if not isinstance(model, nn.DataParallel):
        return False
    n_devices = len(model.device_ids)
    if n_devices <= 1:
        return False
    return _dp_chunk_count(int(batch_size), n_devices) < n_devices


def _extract_views(vol_dhw: np.ndarray) -> Dict[str, np.ndarray]:
    d, h, w = vol_dhw.shape
    return {
        "axial(z-mid)": vol_dhw[d // 2, :, :],
        "coronal(y-mid)": vol_dhw[:, h // 2, :],
        "sagittal(x-mid)": vol_dhw[:, :, w // 2],
    }


def _middle_frames(t: int, n_frames: int) -> List[int]:
    t = max(1, int(t))
    n_frames = max(1, min(int(n_frames), t))
    mid = t // 2
    left = max(0, mid - (n_frames // 2))
    right = min(t, left + n_frames)
    left = max(0, right - n_frames)
    return list(range(left, right))


def _save_recon_monitor_png(out_png: str, x_tdhw: torch.Tensor, recon_tdhw: torch.Tensor, frame_ids: List[int]) -> None:
    gt_frames = [x_tdhw[t].detach().cpu().numpy() for t in frame_ids]
    rc_frames = [recon_tdhw[t].detach().cpu().numpy() for t in frame_ids]

    gt_views = [_extract_views(v) for v in gt_frames]
    rc_views = [_extract_views(v) for v in rc_frames]

    gt_concat = np.concatenate([vv.reshape(-1) for vf in gt_views for vv in vf.values()], axis=0)
    vmin = float(np.percentile(gt_concat, 1.0))
    vmax = float(np.percentile(gt_concat, 99.0))
    if vmax <= vmin:
        vmax = vmin + 1.0e-6

    err_views = []
    for i in range(len(gt_views)):
        err_views.append({k: np.abs(rc_views[i][k] - gt_views[i][k]) for k in gt_views[i].keys()})
    err_concat = np.concatenate([vv.reshape(-1) for vf in err_views for vv in vf.values()], axis=0)
    emax = float(np.percentile(err_concat, 99.0))
    emax = max(emax, 1.0e-8)

    view_names = list(gt_views[0].keys())
    n_rows = len(frame_ids) * 3
    n_cols = len(view_names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.8 * n_rows))
    if n_cols == 1:
        axes = np.array(axes).reshape(n_rows, 1)

    row = 0
    for i, t in enumerate(frame_ids):
        row_defs = [
            (f"GT t={t}", gt_views[i], "gray", (vmin, vmax)),
            (f"Recon t={t}", rc_views[i], "gray", (vmin, vmax)),
            (f"|Recon-GT| t={t}", err_views[i], "magma", (0.0, emax)),
        ]
        for row_name, row_views, cmap, (lo, hi) in row_defs:
            for j, vn in enumerate(view_names):
                ax = axes[row, j]
                ax.imshow(row_views[vn], cmap=cmap, vmin=lo, vmax=hi)
                if row == 0:
                    ax.set_title(vn)
                if j == 0:
                    ax.set_ylabel(row_name)
                ax.axis("off")
            row += 1

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def _init_recon_monitor(cfg: Dict, output_root: str) -> Dict:
    mc = cfg.get("training", {}).get("recon_monitor", {})
    enabled = bool(mc.get("enabled", False))
    out = {
        "enabled": enabled,
        "every_n_steps": max(1, int(mc.get("every_n_steps", 200))),
        "num_middle_frames": max(1, int(mc.get("num_middle_frames", 3))),
        "max_events_per_epoch": max(1, int(mc.get("max_events_per_epoch", 8))),
        "save_npz": bool(mc.get("save_npz", False)),
        "events_this_epoch": 0,
        "global_event_id": 0,
    }
    if not enabled:
        return out

    monitor_root = ensure_dir(os.path.join(output_root, str(mc.get("output_dir", "train_recon_monitor"))))
    vis_dir = ensure_dir(os.path.join(monitor_root, "visuals"))
    npz_dir = ensure_dir(os.path.join(monitor_root, "npz"))
    csv_path = os.path.join(monitor_root, "metrics.csv")

    out["monitor_root"] = monitor_root
    out["vis_dir"] = vis_dir
    out["npz_dir"] = npz_dir
    out["csv_path"] = csv_path
    out["csv_header_written"] = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    return out


def _write_monitor_row(row: Dict, csv_path: str, header_written: bool) -> bool:
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not header_written:
            w.writeheader()
        w.writerow(row)
    return True


def _maybe_save_recon_monitor(
    *,
    monitor: Dict,
    epoch: int,
    step_in_epoch: int,
    x: torch.Tensor,
    recon: torch.Tensor,
    fg_thr: float,
    source_path: str,
) -> None:
    if not monitor.get("enabled", False):
        return
    if (step_in_epoch % int(monitor["every_n_steps"])) != 0:
        return
    if int(monitor["events_this_epoch"]) >= int(monitor["max_events_per_epoch"]):
        return
    if x.shape[0] <= 0:
        return

    sid = int(monitor["global_event_id"])
    xi = x[0, 0].detach().float()
    ri = recon[0, 0].detach().float()

    frame_ids = _middle_frames(int(xi.shape[0]), int(monitor["num_middle_frames"]))
    png_path = os.path.join(monitor["vis_dir"], f"ep{epoch:03d}_step{step_in_epoch:06d}_id{sid:06d}.png")
    _save_recon_monitor_png(png_path, xi, ri, frame_ids)

    fg = (xi.abs() > float(fg_thr)).float().unsqueeze(0).unsqueeze(0)
    fg = fg.expand(1, 1, xi.shape[0], xi.shape[1], xi.shape[2], xi.shape[3])
    fb = fg_bg_metrics(ri.unsqueeze(0).unsqueeze(0), xi.unsqueeze(0).unsqueeze(0), fg)

    row = {
        "event_id": sid,
        "epoch": int(epoch),
        "step_in_epoch": int(step_in_epoch),
        "source_path": source_path,
        "mse": mse(ri, xi),
        "mae": mae(ri, xi),
        "psnr": psnr(ri, xi),
        "fg_mse": fb["fg_mse"],
        "bg_mse": fb["bg_mse"],
        "bg_abs_mean": fb["bg_abs_mean"],
        "visual_path": png_path,
    }

    if bool(monitor.get("save_npz", False)):
        npz_path = os.path.join(monitor["npz_dir"], f"ep{epoch:03d}_step{step_in_epoch:06d}_id{sid:06d}.npz")
        np.savez_compressed(
            npz_path,
            gt=xi.cpu().numpy().astype(np.float32),
            recon=ri.cpu().numpy().astype(np.float32),
            frame_ids=np.array(frame_ids, dtype=np.int32),
        )
        row["npz_path"] = npz_path

    monitor["csv_header_written"] = _write_monitor_row(row, monitor["csv_path"], bool(monitor.get("csv_header_written", False)))
    monitor["events_this_epoch"] = int(monitor["events_this_epoch"]) + 1
    monitor["global_event_id"] = sid + 1


def pretrain_data_audit(loader: DataLoader, cfg: Dict, device: torch.device) -> Dict:
    n_pick = int(cfg.get("data", {}).get("pretrain_audit_samples", 5))
    n_pick = max(1, n_pick)
    ds = loader.dataset
    n_ds = len(ds)
    rng = random.Random(int(cfg.get("seed", 42)) + 9001)
    pick = list(range(n_ds)) if n_pick >= n_ds else rng.sample(range(n_ds), k=n_pick)

    fg_thr = float(cfg.get("data", {}).get("fg_threshold", 1.0e-6))
    fg_ratio_sum = 0.0
    bg_ratio_sum = 0.0

    for idx in pick:
        item = ds[int(idx)]
        x = item["x"].to(device=device, dtype=torch.float32)
        mask = item.get("mask", None)
        if mask is not None:
            mask = mask.to(device=device, dtype=torch.float32).unsqueeze(1)
            fg = mask.unsqueeze(2).expand(-1, -1, x.shape[1], -1, -1, -1)
        else:
            fg = (x.abs() > fg_thr).float().unsqueeze(0)

        fg_ratio_sum += float(fg.mean().item())
        bg_ratio_sum += float((1.0 - fg).mean().item())

    out = {
        "samples": len(pick),
        "fg_ratio_mean": fg_ratio_sum / max(1, len(pick)),
        "bg_ratio_mean": bg_ratio_sum / max(1, len(pick)),
    }
    print("=" * 88)
    print(f"[audit] pretrain sample audit samples={out['samples']}")
    print(f"[audit] fg_ratio_mean={out['fg_ratio_mean']:.4f} bg_ratio_mean={out['bg_ratio_mean']:.4f}")
    print("=" * 88)
    return out


def _snapshot_model_cfg(cfg: Dict) -> Dict:
    data = cfg.get("data", {})
    model = copy.deepcopy(cfg.get("model", {}))
    return {
        "model": model,
        "data": {
            "t_frames": int(data.get("t_frames", 40)),
            "fg_threshold": float(data.get("fg_threshold", 1.0e-6)),
            "layout": str(data.get("layout", "DHWT")),
            "normalize": str(data.get("normalize", "none")),
            "bbox_crop": copy.deepcopy(data.get("bbox_crop", {})),
        },
    }


def _load_model_cfg_from_ckpt(ckpt_path: str) -> Dict | None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_cfg = ckpt.get("model_config", None)
    if isinstance(model_cfg, dict) and "model" in model_cfg:
        return model_cfg
    return None


def _scheduled_kl_weight(cfg: Dict, epoch: int) -> float:
    loss_cfg = cfg.get("loss", {})
    tr_cfg = cfg.get("training", {})
    base = float(loss_cfg.get("beta_kl", loss_cfg.get("kl_weight", 1.0e-4)))
    warm_epochs = int(tr_cfg.get("kl_warmup_epochs", 0))
    if warm_epochs <= 0:
        return base
    scale = min(1.0, max(0.0, float(epoch) / float(warm_epochs)))
    return base * scale


def _should_sample_posterior(cfg: Dict, epoch: int) -> bool:
    tr_cfg = cfg.get("training", {})
    det_warm = int(tr_cfg.get("deterministic_warmup_epochs", 0))
    return int(epoch) > det_warm


def _resolve_metric_value(train_m: Dict[str, float], val_m: Dict[str, float], metric_name: str) -> Optional[float]:
    name = str(metric_name).strip()
    if name.startswith("train_"):
        return train_m.get(name[len("train_"):], None)
    if name.startswith("val_"):
        return val_m.get(name[len("val_"):], None)
    if name in val_m:
        return val_m[name]
    if name in train_m:
        return train_m[name]
    return None


def _is_improved(current: float, best: float | None, mode: str, min_delta: float) -> bool:
    if best is None:
        return True
    if mode == "min":
        return current < best - min_delta
    if mode == "max":
        return current > best + min_delta
    raise ValueError(f"Unsupported early stop mode={mode}")


def run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Dict,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    desc: str,
    epoch: int,
    sample_posterior: bool,
    kl_weight: float,
    recon_monitor: Dict | None = None,
    on_train_step: Optional[Callable[[int], None]] = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    tr_cfg = cfg.get("training", {})
    loss_cfg = cfg.get("loss", {})
    fg_thr = float(cfg.get("data", {}).get("fg_threshold", 1.0e-6))
    grad_clip = float(tr_cfg.get("grad_clip", 0.0))

    use_amp = bool(tr_cfg.get("use_amp", True) and device.type == "cuda")
    amp_dtype_name = str(tr_cfg.get("amp_dtype", "bf16")).lower()
    amp_dtype = torch.float16 if amp_dtype_name == "fp16" else torch.bfloat16

    haar = Haar3DSpatial().to(device)
    sums = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "bg_zero_loss": 0.0,
        "wavelet_loss": 0.0,
        "temporal_loss": 0.0,
        "kl_loss": 0.0,
        "kl_weight": 0.0,
        "fg_ratio": 0.0,
        "bg_ratio": 0.0,
        "xhat_abs_mean": 0.0,
        "xhat_abs_max": 0.0,
    }
    n_steps = 0

    show_progress = _is_main_process()
    iterator = tqdm(loader, desc=desc, ncols=120) if show_progress else loader
    for step_idx, batch in enumerate(iterator, start=1):
        x = batch["x"].to(device=device, dtype=torch.float32)
        mask = batch.get("mask", None)
        if isinstance(mask, torch.Tensor):
            mask = mask.to(device=device, dtype=torch.float32)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        amp_ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()
        with amp_ctx:
            if _should_bypass_dp(model, int(x.shape[0])):
                out = _get_core_model(model)(x, sample_posterior=sample_posterior)
            else:
                out = model(x, sample_posterior=sample_posterior)
            loss_dict = compute_vae_losses(
                x=x,
                x_hat=out["x_hat"],
                mu=out["mu"],
                logvar=out["logvar"],
                mask=mask,
                loss_cfg=loss_cfg,
                fg_threshold=fg_thr,
                kl_weight=kl_weight,
                haar=haar,
            )
            loss = loss_dict["loss"]

        if not torch.isfinite(loss):
            if show_progress:
                iterator.set_postfix(loss="nan", recon="nan", kl=f"{sums['kl_loss']/max(1, n_steps):.4f}")
            continue

        if is_train and recon_monitor is not None:
            paths = batch.get("paths", [""])
            src_path = str(paths[0]) if isinstance(paths, list) and len(paths) > 0 else ""
            _maybe_save_recon_monitor(
                monitor=recon_monitor,
                epoch=int(epoch),
                step_in_epoch=int(step_idx),
                x=x,
                recon=out["x_hat"],
                fg_thr=fg_thr,
                source_path=src_path,
            )

        if is_train:
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

        for k in sums.keys():
            sums[k] += _to_float(loss_dict[k])
        n_steps += 1
        if is_train and on_train_step is not None:
            on_train_step(int(step_idx))

        if show_progress:
            iterator.set_postfix(
                loss=f"{sums['loss']/max(1, n_steps):.4f}",
                recon=f"{sums['recon_loss']/max(1, n_steps):.4f}",
                kl=f"{sums['kl_loss']/max(1, n_steps):.4f}",
                beta=f"{sums['kl_weight']/max(1, n_steps):.6f}",
            )

    if _dist_is_initialized():
        keys = list(sums.keys())
        buf = torch.tensor([sums[k] for k in keys] + [float(n_steps)], dtype=torch.float64, device=device)
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        global_steps = max(1.0, float(buf[-1].item()))
        for i, k in enumerate(keys):
            sums[k] = float(buf[i].item()) / global_steps
        return sums

    for k in sums.keys():
        sums[k] /= max(1, n_steps)
    return sums


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    tr_cfg = cfg.get("training", {})
    use_ddp = _as_bool(tr_cfg.get("use_ddp", False))
    resume_from = str(tr_cfg.get("resume_from", "")).strip()
    if resume_from != "" and not os.path.isabs(resume_from):
        cfg_dir = os.path.dirname(os.path.abspath(args.config))
        resume_from = os.path.abspath(os.path.join(cfg_dir, resume_from))
    if resume_from != "":
        if not os.path.isfile(resume_from):
            raise FileNotFoundError(f"resume_from not found: {resume_from}")
        resume_model_cfg = _load_model_cfg_from_ckpt(resume_from)
        if resume_model_cfg is not None:
            cfg["model"] = resume_model_cfg["model"]
            data = cfg.get("data", {})
            for key, value in resume_model_cfg.get("data", {}).items():
                data[key] = value
            cfg["data"] = data

    rank = 0
    local_rank = 0
    world_size = 1
    device = _device_from_cfg(cfg)
    gpu_ids = []

    if use_ddp:
        if device.type != "cuda":
            raise RuntimeError("DDP requires CUDA device")
        required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
        missing = [k for k in required if k not in os.environ]
        if missing:
            visible_gpu_ids = _resolve_gpu_ids(cfg)
            if len(visible_gpu_ids) <= 1:
                print(
                    f"[warn] training.use_ddp=true but torchrun env vars are missing {missing}; "
                    "fall back to single-process training"
                )
                use_ddp = False
                gpu_ids = visible_gpu_ids
            else:
                raise RuntimeError(f"DDP mode requires torchrun env vars, missing: {missing}")

    if use_ddp:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
        gpu_ids = []
    else:
        if device.type == "cuda":
            if not gpu_ids:
                gpu_ids = _resolve_gpu_ids(cfg)
            bs_cfg = max(1, int(tr_cfg.get("batch_size", 1)))
            if len(gpu_ids) > bs_cfg:
                print(f"[warn] gpu_ids({len(gpu_ids)}) > batch_size({bs_cfg}), trim to first {bs_cfg} GPUs")
                gpu_ids = gpu_ids[:bs_cfg]

            n_before = len(gpu_ids)
            while len(gpu_ids) > 1 and _dp_chunk_count(bs_cfg, len(gpu_ids)) < len(gpu_ids):
                gpu_ids = gpu_ids[:-1]
            if len(gpu_ids) != n_before:
                print(
                    f"[warn] batch_size({bs_cfg}) would create empty DP shards on {n_before} GPUs; "
                    f"trim to {len(gpu_ids)} GPUs: {gpu_ids}"
                )

            device = torch.device(f"cuda:{gpu_ids[0]}")
        else:
            gpu_ids = []

    train_loader, val_loader, test_loader, audits, samplers = build_loaders(cfg, ddp=use_ddp, rank=rank, world_size=world_size)

    output_base = ensure_dir(str(tr_cfg.get("output_root", "outputs/wf_vae2_run")))
    if _dist_is_initialized():
        out_holder = [None]
        if _is_main_process():
            out_holder[0] = ensure_dir(os.path.dirname(os.path.dirname(resume_from))) if resume_from != "" else _make_timestamped_run_dir(output_base)
        dist.broadcast_object_list(out_holder, src=0)
        output_root = str(out_holder[0])
        ensure_dir(output_root)
    else:
        output_root = ensure_dir(os.path.dirname(os.path.dirname(resume_from))) if resume_from != "" else _make_timestamped_run_dir(output_base)

    ckpt_dir = ensure_dir(os.path.join(output_root, "checkpoints"))
    recon_monitor = _init_recon_monitor(cfg, output_root) if _is_main_process() else {"enabled": False}

    model = build_model_from_config(cfg).to(device)
    total_params, trainable_params = _count_params(model)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if _is_main_process():
            print(f"[info] DDP enabled world_size={world_size} local_rank={local_rank}")
    elif device.type == "cuda" and len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
        if _is_main_process():
            print(f"[info] DataParallel enabled on GPUs: {gpu_ids}")

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(tr_cfg.get("lr", 2.0e-4)),
        weight_decay=float(tr_cfg.get("weight_decay", 0.01)),
    )

    use_amp = bool(tr_cfg.get("use_amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    if _is_main_process():
        print("=" * 88)
        print(f"[audit] params total={total_params} ({total_params / 1e6:.3f}M) trainable={trainable_params} ({trainable_params / 1e6:.3f}M)")
        print("[audit] dataset usage")
        for k in ("train", "val", "test"):
            a = audits[k]
            if int(a.num_files_before_subset) != int(a.num_files) or float(a.requested_ratio) != 1.0:
                print(
                    f"[audit] split={k} files={a.num_files} "
                    f"(before_subset={a.num_files_before_subset}, ratio={a.requested_ratio:.4f})"
                )
            else:
                print(f"[audit] split={k} files={a.num_files}")
            print(f"[audit] roots={a.roots}")
            print(f"[audit] preview={a.sample_preview}")
        print("=" * 88)
        if bool(recon_monitor.get("enabled", False)):
            print("[audit] recon monitor enabled")
            print(
                f"[audit] recon_monitor every_n_steps={recon_monitor['every_n_steps']} "
                f"num_middle_frames={recon_monitor['num_middle_frames']} "
                f"max_events_per_epoch={recon_monitor['max_events_per_epoch']}"
            )
            print(f"[audit] recon_monitor output={recon_monitor['monitor_root']}")
            print("=" * 88)

    data_audit = pretrain_data_audit(train_loader, cfg, device) if _is_main_process() else {}

    model_cfg = _snapshot_model_cfg(cfg)
    if _is_main_process():
        save_json(model_cfg, os.path.join(output_root, "model_config.json"))
        save_json(cfg, os.path.join(output_root, "train_config.snapshot.json"))
        run_meta = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config_path": os.path.abspath(args.config),
            "output_root": os.path.abspath(output_root),
            "model_config_path": os.path.abspath(os.path.join(output_root, "model_config.json")),
            "train_config_snapshot": os.path.abspath(os.path.join(output_root, "train_config.snapshot.json")),
            "data_audit": data_audit,
            "dataset_audits": {k: audit_to_dict(v) for k, v in audits.items()},
        }
        save_json(run_meta, os.path.join(output_root, "run_meta.json"))

    epochs = int(tr_cfg.get("epochs", 50))
    save_every = int(tr_cfg.get("save_every", 1))
    save_every_steps = max(0, int(tr_cfg.get("save_every_steps", 0)))
    val_every = int(tr_cfg.get("val_every", 1))

    es_cfg = tr_cfg.get("early_stopping", {})
    es_enable = bool(es_cfg.get("enabled", False))
    es_metric = str(es_cfg.get("metric", "val_loss"))
    es_mode = str(es_cfg.get("mode", "min"))
    es_patience = int(es_cfg.get("patience", 5))
    es_min_delta = float(es_cfg.get("min_delta", 0.0))

    history = []
    best_metric = None
    best_epoch = -1
    best_path = os.path.join(ckpt_dir, "best.pt")
    bad_epochs = 0
    start_epoch = 1
    global_train_step = 0

    if resume_from != "":
        resume_ckpt = torch.load(resume_from, map_location=device)
        _get_core_model(model).load_state_dict(resume_ckpt["model"], strict=True)
        if "optimizer" in resume_ckpt and isinstance(resume_ckpt["optimizer"], dict):
            optim.load_state_dict(resume_ckpt["optimizer"])
        if scaler is not None and "scaler" in resume_ckpt and resume_ckpt["scaler"] is not None:
            scaler.load_state_dict(resume_ckpt["scaler"])

        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        if resume_ckpt.get("best_metric", None) is not None:
            best_metric = float(resume_ckpt["best_metric"])
        best_epoch = int(resume_ckpt.get("best_epoch", resume_ckpt.get("epoch", -1)))
        bad_epochs = int(resume_ckpt.get("bad_epochs", 0))
        global_train_step = int(resume_ckpt.get("global_step", 0))
        if global_train_step <= 0:
            global_train_step = max(0, int(resume_ckpt.get("epoch", 0))) * max(1, len(train_loader))

        hist_path = os.path.join(output_root, "history.json")
        if os.path.isfile(hist_path):
            try:
                history = list(load_json(hist_path).get("history", []))
            except Exception:
                history = []

        if _is_main_process():
            print("=" * 88)
            print(f"[resume] checkpoint={resume_from}")
            print(
                f"[resume] start_epoch={start_epoch} epochs={epochs} best_epoch={best_epoch} "
                f"best_metric={best_metric} global_step={global_train_step}"
            )
            print("=" * 88)

    for ep in range(start_epoch, epochs + 1):
        if use_ddp and samplers.get("train", None) is not None:
            samplers["train"].set_epoch(ep)

        kl_weight = _scheduled_kl_weight(cfg, ep)
        sample_posterior = _should_sample_posterior(cfg, ep)
        last_step_in_epoch = 0

        def _build_state(train_metrics: Dict, val_metrics: Dict, monitored_value: Optional[float]) -> Dict:
            return {
                "epoch": ep,
                "step_in_epoch": int(last_step_in_epoch),
                "global_step": int(global_train_step),
                "model": _get_core_model(model).state_dict(),
                "optimizer": optim.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "config": cfg,
                "model_config": model_cfg,
                "train": train_metrics,
                "val": val_metrics,
                "kl_weight": float(kl_weight),
                "sample_posterior": bool(sample_posterior),
                "best_metric": None if best_metric is None else float(best_metric),
                "best_epoch": int(best_epoch),
                "bad_epochs": int(bad_epochs),
                "monitored_metric": es_metric,
                "monitored_value": None if monitored_value is None else float(monitored_value),
            }

        def _on_train_step(step_in_epoch: int) -> None:
            nonlocal global_train_step, last_step_in_epoch
            last_step_in_epoch = int(step_in_epoch)
            global_train_step += 1
            if save_every_steps <= 0:
                return
            if (global_train_step % save_every_steps) != 0:
                return
            step_state = _build_state({}, {}, None)
            step_ckpt = os.path.join(ckpt_dir, f"step_{global_train_step:09d}.pt")
            torch.save(step_state, step_ckpt)
            print(
                f"[ckpt-step] saved {os.path.basename(step_ckpt)} "
                f"(epoch={ep}, step_in_epoch={last_step_in_epoch})"
            )

        recon_monitor["events_this_epoch"] = 0
        train_m = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            cfg=cfg,
            optimizer=optim,
            scaler=scaler,
            desc=f"train[{ep}/{epochs}]",
            epoch=ep,
            sample_posterior=sample_posterior,
            kl_weight=kl_weight,
            recon_monitor=recon_monitor if _is_main_process() else None,
            on_train_step=_on_train_step if _is_main_process() else None,
        )

        if ep % val_every == 0:
            with torch.no_grad():
                val_m = run_epoch(
                    model=model,
                    loader=val_loader,
                    device=device,
                    cfg=cfg,
                    optimizer=None,
                    scaler=None,
                    desc=f"val[{ep}/{epochs}]",
                    epoch=ep,
                    sample_posterior=False,
                    kl_weight=kl_weight,
                    recon_monitor=None,
                )
        else:
            val_m = {}

        current_metric = _resolve_metric_value(train_m, val_m, es_metric)
        improved = False
        if current_metric is None:
            if best_metric is None:
                best_metric = float(train_m.get("loss", 0.0))
                best_epoch = ep
                improved = True
        else:
            improved = _is_improved(float(current_metric), best_metric, mode=es_mode, min_delta=es_min_delta)
            if improved:
                best_metric = float(current_metric)
                best_epoch = ep
                bad_epochs = 0
            elif len(val_m) > 0 or es_metric.startswith("train_"):
                bad_epochs += 1

        if _is_main_process():
            rec = {
                "epoch": ep,
                "global_step": int(global_train_step),
                "kl_weight": float(kl_weight),
                "sample_posterior": bool(sample_posterior),
                "monitored_metric": es_metric,
                "monitored_value": None if current_metric is None else float(current_metric),
                "train": train_m,
                "val": val_m,
            }
            history.append(rec)
            save_json({"history": history}, os.path.join(output_root, "history.json"))

            state = _build_state(train_m, val_m, None if current_metric is None else float(current_metric))

            if ep % save_every == 0:
                torch.save(state, os.path.join(ckpt_dir, f"epoch_{ep:03d}.pt"))
            if improved:
                torch.save(state, best_path)

            best_metric_str = "nan" if best_metric is None else f"{best_metric:.6f}"
            val_loss_str = f"{val_m.get('loss', float('nan')):.6f}" if len(val_m) > 0 else "nan"
            print(
                f"[epoch {ep}] train_loss={train_m['loss']:.6f} val_loss={val_loss_str} "
                f"beta_kl={kl_weight:.6f} sample_posterior={int(sample_posterior)} "
                f"best_epoch={best_epoch} best_metric={best_metric_str}"
            )

        stop_now = False
        if _is_main_process() and es_enable and bad_epochs >= es_patience:
            print(
                f"[early-stop] metric={es_metric} mode={es_mode} "
                f"patience={es_patience} min_delta={es_min_delta}"
            )
            stop_now = True

        if _dist_is_initialized():
            stop_t = torch.tensor([1 if stop_now else 0], dtype=torch.int32, device=device)
            dist.broadcast(stop_t, src=0)
            stop_now = bool(int(stop_t.item()) == 1)

        if stop_now:
            break

    if _dist_is_initialized():
        dist.barrier()

    load_path = best_path if os.path.isfile(best_path) else os.path.join(
        ckpt_dir,
        f"epoch_{max(1, min(epochs, best_epoch if best_epoch > 0 else epochs)):03d}.pt",
    )
    best_ckpt = torch.load(load_path, map_location=device)
    _get_core_model(model).load_state_dict(best_ckpt["model"], strict=True)
    final_kl_weight = _scheduled_kl_weight(cfg, best_epoch if best_epoch > 0 else epochs)
    with torch.no_grad():
        test_m = run_epoch(
            model=model,
            loader=test_loader,
            device=device,
            cfg=cfg,
            optimizer=None,
            scaler=None,
            desc="test[best]",
            epoch=int(best_epoch if best_epoch > 0 else epochs),
            sample_posterior=False,
            kl_weight=final_kl_weight,
            recon_monitor=None,
        )

    if _is_main_process():
        final_summary = {
            "best_epoch": int(best_epoch),
            "best_metric_name": es_metric,
            "best_metric": float(best_metric if best_metric is not None else 0.0),
            "test_metrics": test_m,
            "best_checkpoint": os.path.abspath(load_path),
        }
        save_json(final_summary, os.path.join(output_root, "final_summary.json"))
        print(f"[done] training completed: {output_root}")

    if _dist_is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
