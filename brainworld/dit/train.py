from __future__ import annotations

import argparse
import os
import re
import time
from contextlib import nullcontext
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from brainworld.dit.data import audit_to_dict, build_splits_from_config, collate_batch
from brainworld.dit.diffusion import GaussianDiffusion, normalize_prediction_type, normalize_schedule_type
from brainworld.dit.model import ConditionalLatentDiT, compute_diversity_terms, compute_patch_audit
from brainworld.dit.utils import ensure_dir, load_json, make_timestamped_dir, resolve_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BrainWorld conditional latent diffusion transformer")
    p.add_argument("--config", required=True, help="Path to conditional DiT JSON config")
    return p.parse_args()


def _count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def _resolve_gpu_ids(cfg: Dict) -> List[int]:
    tr_cfg = cfg.get("training", {})
    if not isinstance(tr_cfg, dict):
        tr_cfg = {}

    ids = None
    for key in ("gpu_ids", "gpu_id", "gup_id"):
        if key in tr_cfg and tr_cfg.get(key, None) is not None:
            ids = tr_cfg.get(key)
            break
    if ids is None:
        for key in ("gpu_ids", "gpu_id", "gup_id"):
            if key in cfg and cfg.get(key, None) is not None:
                ids = cfg.get(key)
                break

    if ids is None:
        return []

    if isinstance(ids, str):
        s = ids.strip()
        if s == "":
            return []
        ids = [x.strip() for x in s.split(",") if x.strip() != ""]
    elif isinstance(ids, (int, float)):
        ids = [int(ids)]
    elif isinstance(ids, tuple):
        ids = list(ids)

    if not isinstance(ids, list):
        raise ValueError("gpu_ids/gpu_id must be a list, comma-separated string, or integer CUDA index")
    if len(ids) == 0:
        return []

    out: List[int] = []
    for x in ids:
        i = int(x)
        if i < 0:
            raise ValueError(f"gpu_ids must be >= 0, got {i}")
        out.append(i)
    return out


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, (torch.nn.DataParallel, DDP)):
        return model.module
    return model


def _is_condition_encoder_key(key: str) -> bool:
    return str(key).startswith(
        (
            "fc_encoder.",
            "mri_encoder.",
            "video_encoder.",
            "audio_encoder.",
            "meta_encoder.",
        )
    )


def _load_model_state_dict(module: torch.nn.Module, state: Dict[str, torch.Tensor], allow_condition_mismatch: bool) -> None:
    core = _unwrap_model(module)
    if not allow_condition_mismatch:
        core.load_state_dict(state, strict=True)
        return

    model_state = core.state_dict()
    model_keys = set(model_state.keys())
    ckpt_keys = set(state.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    bad_missing = [k for k in missing if not _is_condition_encoder_key(k)]
    bad_unexpected = [k for k in unexpected if not _is_condition_encoder_key(k)]
    if len(bad_missing) > 0 or len(bad_unexpected) > 0:
        raise RuntimeError(
            "Checkpoint/model mismatch outside condition encoders.\n"
            f"missing(non-cond)={len(bad_missing)} preview={bad_missing[:12]}\n"
            f"unexpected(non-cond)={len(bad_unexpected)} preview={bad_unexpected[:12]}"
        )

    filtered = {k: v for k, v in state.items() if k in model_keys}
    incompatible = core.load_state_dict(filtered, strict=False)
    miss2 = [k for k in incompatible.missing_keys if not _is_condition_encoder_key(k)]
    unexp2 = [k for k in incompatible.unexpected_keys if not _is_condition_encoder_key(k)]
    if len(miss2) > 0 or len(unexp2) > 0:
        raise RuntimeError(
            "State_dict load mismatch outside condition encoders after filtering.\n"
            f"missing(non-cond)={len(miss2)} preview={miss2[:12]}\n"
            f"unexpected(non-cond)={len(unexp2)} preview={unexp2[:12]}"
        )


def _named_trainable_params(module: torch.nn.Module) -> List[tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in _unwrap_model(module).named_parameters() if param.requires_grad]


def _build_optimizer(model: torch.nn.Module, tr_cfg: Dict) -> torch.optim.Optimizer:
    lr = float(tr_cfg.get("lr", 2.0e-4))
    wd = float(tr_cfg.get("weight_decay", 0.01))
    base_lr_scale = float(tr_cfg.get("pretrained_lr_scale", 1.0))
    new_lr_scale = float(tr_cfg.get("new_condition_lr_scale", 1.0))
    new_prefixes = ("video_encoder.", "audio_encoder.")
    named_params = _named_trainable_params(model)
    if base_lr_scale == 1.0 and new_lr_scale == 1.0:
        return torch.optim.AdamW([param for _, param in named_params], lr=lr, weight_decay=wd)

    base_params = [param for name, param in named_params if not name.startswith(new_prefixes)]
    new_params = [param for name, param in named_params if name.startswith(new_prefixes)]
    groups = []
    if len(base_params) > 0:
        groups.append({"params": base_params, "lr": lr * base_lr_scale, "weight_decay": wd})
    if len(new_params) > 0:
        groups.append({"params": new_params, "lr": lr * new_lr_scale, "weight_decay": wd})
    if len(groups) == 0:
        raise RuntimeError("No trainable parameters found for optimizer")
    return torch.optim.AdamW(groups)


def _extract_step_from_checkpoint_name(path: str) -> int:
    m = re.match(r"step_(\d+)\.pt$", os.path.basename(path))
    if m is None:
        return -1
    return int(m.group(1))


def _is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_main() -> bool:
    return _rank() == 0


def _init_distributed_if_needed() -> None:
    if _is_dist() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")


def _destroy_distributed_if_needed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _build_model_from_dataset(cfg: Dict, ds) -> ConditionalLatentDiT:
    mcfg = cfg.get("model", {})
    condition_shapes = {
        "fc": ds.fc_shape,
        "mri": ds.mri_shape,
        "video": ds.video_shape,
        "audio": ds.audio_shape,
        "metadata": ((ds.meta_dim,) if ds.meta_dim > 0 else None),
    }
    return ConditionalLatentDiT(
        target_shape=ds.target_shape,
        patch_size=mcfg.get("patch_size", [1, 2, 1]),
        hidden_dim=int(mcfg.get("hidden_dim", 512)),
        depth=int(mcfg.get("depth", 12)),
        num_heads=int(mcfg.get("num_heads", 8)),
        mlp_ratio=float(mcfg.get("mlp_ratio", 4.0)),
        dropout=float(mcfg.get("dropout", 0.0)),
        condition_shapes=condition_shapes,
        condition_cfg=cfg.get("data", {}).get("conditions", {}),
        diversity_cfg=cfg.get("diversity", {}),
        max_time_steps=int(cfg.get("diffusion", {}).get("num_steps", 1000)),
    )


def _get_model_cfg_snapshot(cfg: Dict, ds) -> Dict:
    m = dict(cfg.get("model", {}))
    m["target_shape"] = [int(v) for v in ds.target_shape]
    m["condition_shapes"] = {
        "fc": None if ds.fc_shape is None else [int(v) for v in ds.fc_shape],
        "mri": None if ds.mri_shape is None else [int(v) for v in ds.mri_shape],
        "video": None if ds.video_shape is None else [int(v) for v in ds.video_shape],
        "audio": None if ds.audio_shape is None else [int(v) for v in ds.audio_shape],
        "metadata": None if ds.meta_dim <= 0 else [int(ds.meta_dim)],
    }
    return {
        "model": m,
        "diffusion": dict(cfg.get("diffusion", {})),
        "task": dict(cfg.get("task", {})),
        "conditioning": dict(cfg.get("conditioning", {})),
        "diversity": dict(cfg.get("diversity", {})),
        "loss": dict(cfg.get("loss", {})),
        "data": {
            "target": {
                "latent_field": str(cfg.get("data", {}).get("target", {}).get("latent_field", "mu")),
                "dataset_dtype": str(cfg.get("data", {}).get("target", {}).get("dataset_dtype", "float32")),
            },
            "conditions": {
                "fc": dict(cfg.get("data", {}).get("conditions", {}).get("fc", {})),
                "mri": dict(cfg.get("data", {}).get("conditions", {}).get("mri", {})),
                "video": dict(cfg.get("data", {}).get("conditions", {}).get("video", {})),
                "audio": dict(cfg.get("data", {}).get("conditions", {}).get("audio", {})),
                "metadata": dict(cfg.get("data", {}).get("conditions", {}).get("metadata", {})),
                "global_fusion": dict(cfg.get("data", {}).get("conditions", {}).get("global_fusion", {})),
            },
        },
    }


def _prepare_condition_inputs(batch: Dict[str, object], device: torch.device, cfg: Dict, training: bool) -> Dict[str, Optional[torch.Tensor]]:
    ccfg = cfg.get("conditioning", {})
    out: Dict[str, Optional[torch.Tensor]] = {}

    fc = batch.get("fc_cond", None)
    out["fc_cond"] = None if fc is None else fc.to(device=device, dtype=torch.float32, non_blocking=True)
    out["has_fc"] = batch.get("has_fc", None)
    if out["has_fc"] is not None:
        out["has_fc"] = out["has_fc"].to(device=device, dtype=torch.float32, non_blocking=True)

    mri = batch.get("mri_cond", None)
    out["mri_cond"] = None if mri is None else mri.to(device=device, dtype=torch.float32, non_blocking=True)
    out["has_mri"] = batch.get("has_mri", None)
    if out["has_mri"] is not None:
        out["has_mri"] = out["has_mri"].to(device=device, dtype=torch.float32, non_blocking=True)

    video = batch.get("video_cond", None)
    out["video_cond"] = None if video is None else video.to(device=device, dtype=torch.float32, non_blocking=True)
    out["has_video"] = batch.get("has_video", None)
    if out["has_video"] is not None:
        out["has_video"] = out["has_video"].to(device=device, dtype=torch.float32, non_blocking=True)

    audio = batch.get("audio_cond", None)
    out["audio_cond"] = None if audio is None else audio.to(device=device, dtype=torch.float32, non_blocking=True)
    out["has_audio"] = batch.get("has_audio", None)
    if out["has_audio"] is not None:
        out["has_audio"] = out["has_audio"].to(device=device, dtype=torch.float32, non_blocking=True)

    meta = batch.get("meta_cond", None)
    out["meta_cond"] = None if meta is None else meta.to(device=device, dtype=torch.float32, non_blocking=True)
    out["has_meta"] = batch.get("has_meta", None)
    if out["has_meta"] is not None:
        out["has_meta"] = out["has_meta"].to(device=device, dtype=torch.float32, non_blocking=True)

    if training:
        if out["has_mri"] is not None and out["mri_cond"] is not None:
            p = float(ccfg.get("drop_prob_mri", 0.0))
            if p > 0.0:
                drop = (torch.rand_like(out["has_mri"]) < p).float()
                out["has_mri"] = out["has_mri"] * (1.0 - drop)
        if out["has_video"] is not None and out["video_cond"] is not None:
            p = float(ccfg.get("drop_prob_video", 0.0))
            if p > 0.0:
                drop = (torch.rand_like(out["has_video"]) < p).float()
                out["has_video"] = out["has_video"] * (1.0 - drop)
                keep = (1.0 - drop).view(-1, 1, 1)
                out["video_cond"] = out["video_cond"] * keep
        if out["has_audio"] is not None and out["audio_cond"] is not None:
            p = float(ccfg.get("drop_prob_audio", 0.0))
            if p > 0.0:
                drop = (torch.rand_like(out["has_audio"]) < p).float()
                out["has_audio"] = out["has_audio"] * (1.0 - drop)
                keep = (1.0 - drop).view(-1, 1, 1)
                out["audio_cond"] = out["audio_cond"] * keep
        if out["has_meta"] is not None and out["meta_cond"] is not None:
            p = float(ccfg.get("drop_prob_meta", 0.0))
            if p > 0.0:
                drop = (torch.rand_like(out["has_meta"]) < p).float()
                out["has_meta"] = out["has_meta"] * (1.0 - drop)
    return out


def _diversity_weight(cfg: Dict, global_step: int) -> float:
    dcfg = cfg.get("diversity", {})
    if not bool(dcfg.get("enabled", False)):
        return 0.0
    base = float(dcfg.get("loss_weight", 0.0))
    warmup = int(dcfg.get("warmup_steps", 0))
    if warmup <= 0:
        return base
    return base * min(1.0, float(global_step) / float(warmup))


def run_epoch(
    *,
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.amp.GradScaler],
    use_amp: bool,
    amp_dtype: torch.dtype,
    cfg: Dict,
    desc: str,
    distributed: bool,
    is_main: bool,
    global_step: int,
    step_checkpoint_hook: Optional[Callable[[int], None]] = None,
) -> tuple[Dict[str, float], int]:
    is_train = optimizer is not None
    model.train(is_train)
    tr_cfg = cfg.get("training", {})
    dcfg = cfg.get("diversity", {})
    x0_aux_weight = float(cfg.get("loss", {}).get("x0_aux_weight", 0.0))
    prediction_type = normalize_prediction_type(cfg.get("loss", {}).get("prediction_type", "epsilon"))
    need_hiddens = bool(dcfg.get("enabled", False))
    grad_clip_norm = float(tr_cfg.get("grad_clip_norm", 0.0))
    skip_nonfinite_batch = bool(tr_cfg.get("skip_nonfinite_batch", True))
    nonfinite_log_limit = int(tr_cfg.get("nonfinite_log_limit", 20))
    nonfinite_seen = 0

    meter = {
        "loss": 0.0,
        "diff_loss": 0.0,
        "x0_aux_loss": 0.0,
        "div_loss": 0.0,
        "pred_mse": 0.0,
        "eps_mse": 0.0,
        "x0_mse": 0.0,
        "n": 0.0,
    }
    pbar = tqdm(loader, desc=desc, ncols=120, disable=not is_main)

    for batch in pbar:
        x0 = batch["target_latent"].to(device=device, dtype=torch.float32, non_blocking=True)
        direction_id = batch["direction_id"].to(device=device, dtype=torch.long, non_blocking=True)
        bsz = int(x0.shape[0])
        t = diffusion.sample_timesteps(bsz, device=device)
        cond_inputs = _prepare_condition_inputs(batch, device, cfg, training=is_train)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        ac = torch.amp.autocast(device_type="cuda", dtype=amp_dtype) if (use_amp and device.type == "cuda") else nullcontext()
        with ac:
            xt, noise = diffusion.q_sample(x0, t)
            target = diffusion.get_training_target(x0, t, noise, prediction_type)
            out = model(
                xt,
                t,
                direction_id,
                fc_cond=cond_inputs["fc_cond"],
                has_fc=cond_inputs["has_fc"],
                mri_cond=cond_inputs["mri_cond"],
                has_mri=cond_inputs["has_mri"],
                video_cond=cond_inputs["video_cond"],
                has_video=cond_inputs["has_video"],
                audio_cond=cond_inputs["audio_cond"],
                has_audio=cond_inputs["has_audio"],
                meta_cond=cond_inputs["meta_cond"],
                has_meta=cond_inputs["has_meta"],
                return_hiddens=need_hiddens,
            )
            pred = out["pred"] if isinstance(out, dict) else out
            diff_loss = F.mse_loss(pred, target)
            pred_eps = diffusion.predict_eps(xt, t, pred, prediction_type)
            x0_hat = diffusion.predict_x0(xt, t, pred, prediction_type)
            x0_aux_loss = F.smooth_l1_loss(x0_hat, x0) if x0_aux_weight > 0.0 else pred.new_zeros(())
            if need_hiddens and isinstance(out, dict) and len(out.get("hiddens", {})) > 0:
                div_stats = compute_diversity_terms(out["hiddens"], dcfg)
                div_weight = _diversity_weight(cfg, global_step + 1)
                div_loss = div_weight * div_stats["total"]
            else:
                div_stats = {"total": pred.new_zeros(())}
                div_loss = pred.new_zeros(())
            loss = diff_loss + x0_aux_weight * x0_aux_loss + div_loss

        finite_loss = bool(torch.isfinite(loss).all().item())
        finite_diff = bool(torch.isfinite(diff_loss).all().item())
        finite_div = bool(torch.isfinite(div_loss).all().item())
        if not (finite_loss and finite_diff and finite_div):
            if is_main and nonfinite_seen < nonfinite_log_limit:
                paths = batch.get("target_npz_path", [])
                head = paths[:3] if isinstance(paths, list) else []
                loss_val = float(loss.detach().cpu()) if finite_loss else float("nan")
                diff_val = float(diff_loss.detach().cpu()) if finite_diff else float("nan")
                div_val = float(div_loss.detach().cpu()) if finite_div else float("nan")
                print(
                    f"[warn] non_finite_batch desc={desc} step={global_step} "
                    f"loss={loss_val:.6f} diff={diff_val:.6f} div={div_val:.6f} "
                    f"sample_paths={head}"
                )
                nonfinite_seen += 1
            if not skip_nonfinite_batch:
                raise RuntimeError(f"Encountered non-finite loss in {desc} at global_step={global_step}")
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            continue

        if is_train:
            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip_norm > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()
            global_step += 1
            if step_checkpoint_hook is not None:
                step_checkpoint_hook(global_step)

        with torch.no_grad():
            pred_mse = F.mse_loss(pred, target)
            eps_mse = F.mse_loss(pred_eps, noise)
            x0_mse = F.mse_loss(x0_hat, x0)

        meter["loss"] += float(loss.item()) * bsz
        meter["diff_loss"] += float(diff_loss.item()) * bsz
        meter["x0_aux_loss"] += float(x0_aux_loss.item()) * bsz
        meter["div_loss"] += float(div_loss.item()) * bsz
        meter["pred_mse"] += float(pred_mse.item()) * bsz
        meter["eps_mse"] += float(eps_mse.item()) * bsz
        meter["x0_mse"] += float(x0_mse.item()) * bsz
        meter["n"] += float(bsz)

        denom = max(1.0, meter["n"])
        pbar.set_postfix(
            loss=f"{meter['loss']/denom:.5f}",
            diff=f"{meter['diff_loss']/denom:.5f}",
            x0=f"{meter['x0_mse']/denom:.5f}",
            div=f"{meter['div_loss']/denom:.5f}",
        )

    if distributed:
        t_reduce = torch.tensor(
            [
                meter["loss"],
                meter["diff_loss"],
                meter["x0_aux_loss"],
                meter["div_loss"],
                meter["pred_mse"],
                meter["eps_mse"],
                meter["x0_mse"],
                meter["n"],
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(t_reduce, op=dist.ReduceOp.SUM)
        meter["loss"] = float(t_reduce[0].item())
        meter["diff_loss"] = float(t_reduce[1].item())
        meter["x0_aux_loss"] = float(t_reduce[2].item())
        meter["div_loss"] = float(t_reduce[3].item())
        meter["pred_mse"] = float(t_reduce[4].item())
        meter["eps_mse"] = float(t_reduce[5].item())
        meter["x0_mse"] = float(t_reduce[6].item())
        meter["n"] = float(t_reduce[7].item())

    denom = max(1.0, meter["n"])
    return (
        {
            "loss": meter["loss"] / denom,
            "diff_loss": meter["diff_loss"] / denom,
            "x0_aux_loss": meter["x0_aux_loss"] / denom,
            "div_loss": meter["div_loss"] / denom,
            "pred_mse": meter["pred_mse"] / denom,
            "eps_mse": meter["eps_mse"] / denom,
            "x0_mse": meter["x0_mse"] / denom,
        },
        global_step,
    )


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)
    prediction_type = normalize_prediction_type(cfg.get("loss", {}).get("prediction_type", "epsilon"))
    schedule_type = normalize_schedule_type(cfg.get("diffusion", {}).get("schedule", "linear"))

    _init_distributed_if_needed()
    distributed = _is_dist()
    rank = _rank()
    local_rank = _local_rank()
    world_size = _world_size()
    is_main = _is_main()

    set_seed(int(cfg.get("seed", 42)) + rank)
    cfg_gpu_ids = _resolve_gpu_ids(cfg)

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA in current setup")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        use_multi_gpu = world_size > 1
        active_gpu_ids = cfg_gpu_ids if len(cfg_gpu_ids) > 0 else list(range(world_size))
    else:
        if len(cfg_gpu_ids) > 0:
            if not torch.cuda.is_available():
                raise RuntimeError("gpu_ids is set but CUDA is not available")
            cuda_count = torch.cuda.device_count()
            bad_ids = [i for i in cfg_gpu_ids if i >= cuda_count]
            if len(bad_ids) > 0:
                raise ValueError(f"gpu_ids contains unavailable device(s): {bad_ids}, cuda_count={cuda_count}")
            device = torch.device(f"cuda:{cfg_gpu_ids[0]}")
        else:
            device = resolve_device(cfg)
        use_multi_gpu = bool(device.type == "cuda" and len(cfg_gpu_ids) > 1)
        active_gpu_ids = cfg_gpu_ids if len(cfg_gpu_ids) > 0 else None

    if is_main:
        print("[prep] starting dataset build", flush=True)
    train_ds, val_ds, test_ds, audits = build_splits_from_config(cfg)
    if is_main:
        print("[prep] dataset build finished", flush=True)
    tr_cfg = cfg.get("training", {})
    bs = int(tr_cfg.get("batch_size", 8))
    nw = int(tr_cfg.get("num_workers", 4))

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        sampler=val_sampler,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=bs,
        shuffle=False,
        sampler=test_sampler,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
    )

    dcfg = cfg.get("diffusion", {})
    model = _build_model_from_dataset(cfg, train_ds).to(device)
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    elif use_multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=cfg_gpu_ids, output_device=cfg_gpu_ids[0])

    diffusion = GaussianDiffusion(
        num_steps=int(dcfg.get("num_steps", 1000)),
        beta_start=float(dcfg.get("beta_start", 1.0e-4)),
        beta_end=float(dcfg.get("beta_end", 2.0e-2)),
        schedule=schedule_type,
        cosine_s=float(dcfg.get("cosine_s", 0.008)),
    ).to(device)

    init_from = str(tr_cfg.get("init_from", tr_cfg.get("init_checkpoint", "")) or "").strip()
    allow_condition_mismatch = bool(tr_cfg.get("allow_condition_mismatch", False))
    if init_from:
        init_path = os.path.abspath(os.path.expanduser(init_from))
        if not os.path.isfile(init_path):
            raise FileNotFoundError(f"init checkpoint not found: {init_path}")
        init_ckpt = torch.load(init_path, map_location="cpu")
        if "model" not in init_ckpt:
            raise KeyError(f"init checkpoint missing 'model': {init_path}")
        _load_model_state_dict(model, init_ckpt["model"], allow_condition_mismatch=allow_condition_mismatch)
        if is_main:
            print(f"[init] loaded model weights from {init_path} allow_condition_mismatch={allow_condition_mismatch}")

    optimizer = _build_optimizer(model, tr_cfg)

    use_amp = bool(tr_cfg.get("use_amp", True) and device.type == "cuda")
    amp_dtype_name = str(tr_cfg.get("amp_dtype", "bf16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16)) if device.type == "cuda" else None

    total_params, trainable_params = _count_params(_unwrap_model(model))
    out_base = ensure_dir(str(tr_cfg.get("output_root", "outputs/cond_dit_run")))
    output_root = make_timestamped_dir(out_base) if is_main else ""
    if distributed:
        obj = [output_root]
        dist.broadcast_object_list(obj, src=0)
        output_root = str(obj[0])
    ckpt_dir = ensure_dir(os.path.join(output_root, "checkpoints"))

    model_cfg_snapshot = _get_model_cfg_snapshot(cfg, train_ds)
    if is_main:
        save_json(model_cfg_snapshot, os.path.join(output_root, "model_config.json"))
        save_json(cfg, os.path.join(output_root, "train_config.snapshot.json"))

    resume_from = str(tr_cfg.get("resume_from", "") or "").strip()

    patch_info = compute_patch_audit(train_ds.target_shape, cfg.get("model", {}).get("patch_size", [1, 2, 1]))
    run_meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": os.path.abspath(args.config),
        "output_root": os.path.abspath(output_root),
        "resume_from": (os.path.abspath(os.path.expanduser(resume_from)) if resume_from else ""),
        "init_from": (os.path.abspath(os.path.expanduser(init_from)) if init_from else ""),
        "device": str(device),
        "gpu_ids": active_gpu_ids,
        "multi_gpu": use_multi_gpu,
        "distributed": distributed,
        "world_size": world_size,
        "rank": rank,
        "params_total": total_params,
        "params_trainable": trainable_params,
        "target_shape": [int(v) for v in train_ds.target_shape],
        "prediction_type": prediction_type,
        "schedule": schedule_type,
        "patch_audit": patch_info,
        "dataset_audits": {k: audit_to_dict(v) for k, v in audits.items()},
    }
    if is_main:
        save_json(run_meta, os.path.join(output_root, "run_meta.json"))

    if is_main:
        print("=" * 100)
        print(f"[audit] device={device} gpu_ids={active_gpu_ids} multi_gpu={use_multi_gpu} distributed={distributed} world_size={world_size}")
        print(f"[audit] params total={total_params} ({total_params/1e6:.3f}M) trainable={trainable_params} ({trainable_params/1e6:.3f}M)")
        print(f"[audit] target_shape={tuple(int(v) for v in train_ds.target_shape)}")
        print(f"[audit] patch_size={tuple(int(v) for v in cfg.get('model', {}).get('patch_size', [1, 2, 1]))}")
        print(f"[audit] diffusion_schedule={schedule_type} prediction_type={prediction_type}")
        print(f"[audit] patch_num={patch_info['patch_num']} patch_dim={patch_info['patch_dim']} grid={patch_info['grid_shape']}")
        for k in ("train", "val", "test"):
            a = audits[k]
            print(f"[audit] split={k} target_latents={a.num_target_latents} pair_samples={a.num_pair_samples} subjects={a.num_subjects}")
            print(f"[audit] direction_counts={a.direction_counts}")
            print(f"[audit] condition_shape={a.condition_shape}")
            print(f"[audit] condition_available={a.condition_available}")
            print(f"[audit] condition_required_missing={a.condition_required_missing}")
            print(f"[audit] target_preview={a.target_preview}")
            print(f"[audit] pair_preview={a.pair_preview}")
        print("=" * 100)

    epochs = int(tr_cfg.get("epochs", 30))
    val_every = int(tr_cfg.get("val_every", 1))
    save_every = int(tr_cfg.get("save_every", 1))
    save_latest_every = int(tr_cfg.get("save_latest_every", 1))
    save_step_every = int(tr_cfg.get("save_step_every", 0))
    keep_recent_step_checkpoints = int(tr_cfg.get("keep_recent_step_checkpoints", 3))
    keep_epoch_checkpoints = bool(tr_cfg.get("keep_epoch_checkpoints", False))

    def _safe_torch_save(obj, path: str, tag: str) -> bool:
        if not is_main:
            return True
        try:
            torch.save(obj, path)
            return True
        except Exception as e:
            print(f"[warn] save_failed tag={tag} path={path} error={e}")
            return False

    def _prune_old_step_checkpoints() -> None:
        if not is_main:
            return
        keep_n = max(0, keep_recent_step_checkpoints)
        step_paths = []
        for name in os.listdir(ckpt_dir):
            step_now = _extract_step_from_checkpoint_name(name)
            if step_now >= 0:
                step_paths.append((step_now, os.path.join(ckpt_dir, name)))
        step_paths.sort()
        if keep_n == 0:
            to_remove = step_paths
        else:
            to_remove = step_paths[:-keep_n]
        for step_now, path in to_remove:
            try:
                os.remove(path)
            except OSError as e:
                print(f"[warn] prune_failed step={step_now} path={path} error={e}")

    start_epoch = 1

    best_val = None
    best_epoch = -1
    history = []
    global_step = 0

    def _build_checkpoint_state(
        *,
        epoch: int,
        global_step_now: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        best_val_now,
        best_epoch_now: int,
    ) -> Dict:
        return {
            "epoch": epoch,
            "global_step": global_step_now,
            "model": _unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": (scaler.state_dict() if scaler is not None else None),
            "model_config": model_cfg_snapshot,
            "train": train_metrics,
            "val": val_metrics,
            "best_val": best_val_now,
            "best_epoch": best_epoch_now,
        }

    if resume_from:
        resume_path = os.path.abspath(os.path.expanduser(resume_from))
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        if "model" not in ckpt:
            raise KeyError(f"resume checkpoint missing 'model': {resume_path}")
        if "epoch" not in ckpt or "global_step" not in ckpt:
            raise KeyError(f"resume checkpoint missing required keys epoch/global_step: {resume_path}")
        _unwrap_model(model).load_state_dict(ckpt["model"], strict=True)
        if "optimizer" not in ckpt:
            raise KeyError(f"resume checkpoint missing 'optimizer': {resume_path}")
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler_state = ckpt.get("scaler", None)
        if scaler is not None and scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        global_step = int(ckpt["global_step"])
        best_val = ckpt.get("best_val", None)
        best_epoch = int(ckpt.get("best_epoch", -1))
        start_epoch = int(ckpt["epoch"]) + 1
        if is_main:
            print(
                f"[resume] checkpoint={resume_path} epoch={ckpt['epoch']} "
                f"global_step={global_step} best_epoch={best_epoch} best_val={best_val} "
                f"start_epoch={start_epoch}"
            )

    for ep in range(start_epoch, epochs + 1):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(ep)

        current_train_metrics: Dict[str, float] = {}

        def _step_checkpoint_hook(step_now: int) -> None:
            if save_step_every <= 0 or step_now % save_step_every != 0:
                return
            state = _build_checkpoint_state(
                epoch=ep,
                global_step_now=step_now,
                train_metrics=current_train_metrics,
                val_metrics={},
                best_val_now=best_val,
                best_epoch_now=best_epoch,
            )
            latest_ok = _safe_torch_save(state, os.path.join(ckpt_dir, "latest.pt"), "latest_step")
            step_tag = f"step_{step_now:07d}"
            step_path = os.path.join(ckpt_dir, f"{step_tag}.pt")
            step_ok = _safe_torch_save(state, step_path, step_tag)
            if step_ok:
                _prune_old_step_checkpoints()
            if is_main and latest_ok and step_ok:
                print(f"[checkpoint] saved step checkpoint at global_step={step_now}: {step_path}")

        tr_m, global_step = run_epoch(
            model=model,
            diffusion=diffusion,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            cfg=cfg,
            desc=f"train[{ep}/{epochs}]",
            distributed=distributed,
            is_main=is_main,
            global_step=global_step,
            step_checkpoint_hook=_step_checkpoint_hook,
        )
        current_train_metrics = dict(tr_m)

        if ep % val_every == 0:
            with torch.no_grad():
                va_m, _ = run_epoch(
                    model=model,
                    diffusion=diffusion,
                    loader=val_loader,
                    device=device,
                    optimizer=None,
                    scaler=None,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    cfg=cfg,
                    desc=f"val[{ep}/{epochs}]",
                    distributed=distributed,
                    is_main=is_main,
                    global_step=global_step,
                )
        else:
            va_m = {}

        if is_main:
            rec = {"epoch": ep, "global_step": global_step, "train": tr_m, "val": va_m}
            history.append(rec)
            save_json({"history": history}, os.path.join(output_root, "history.json"))

        state = _build_checkpoint_state(
            epoch=ep,
            global_step_now=global_step,
            train_metrics=tr_m,
            val_metrics=va_m,
            best_val_now=best_val,
            best_epoch_now=best_epoch,
        )

        if save_latest_every > 0 and ep % save_latest_every == 0:
            _safe_torch_save(state, os.path.join(ckpt_dir, "latest.pt"), "latest")

        if keep_epoch_checkpoints and save_every > 0 and ep % save_every == 0:
            _safe_torch_save(state, os.path.join(ckpt_dir, f"epoch_{ep:03d}.pt"), f"epoch_{ep:03d}")

        if len(va_m) > 0:
            cur = float(va_m["loss"])
            if best_val is None or cur < best_val:
                best_val = cur
                best_epoch = ep
                state["best_val"] = best_val
                state["best_epoch"] = best_epoch
                _safe_torch_save(state, os.path.join(ckpt_dir, "best.pt"), "best")

        if is_main:
            val_msg = f" val_loss={va_m['loss']:.6f}" if len(va_m) > 0 else ""
            print(f"[epoch {ep}] train_loss={tr_m['loss']:.6f}{val_msg} best_epoch={best_epoch} best_val={best_val}")

    best_ckpt = os.path.join(ckpt_dir, "best.pt")
    if distributed:
        dist.barrier()

    if os.path.isfile(best_ckpt):
        if is_main:
            ckpt = torch.load(best_ckpt, map_location=device)
            _unwrap_model(model).load_state_dict(ckpt["model"], strict=True)
        if distributed:
            m = _unwrap_model(model)
            for p in m.parameters():
                dist.broadcast(p.data, src=0)
            for b in m.buffers():
                dist.broadcast(b.data, src=0)

    with torch.no_grad():
        te_m, _ = run_epoch(
            model=model,
            diffusion=diffusion,
            loader=test_loader,
            device=device,
            optimizer=None,
            scaler=None,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            cfg=cfg,
            desc="test[best]",
            distributed=distributed,
            is_main=is_main,
            global_step=global_step,
        )

    if is_main:
        final = {
            "best_epoch": best_epoch,
            "best_val": best_val,
            "test": te_m,
            "best_checkpoint": os.path.abspath(best_ckpt),
            "global_step": global_step,
        }
        save_json(final, os.path.join(output_root, "final_summary.json"))
        print(f"[done] conditional DiT training completed: {output_root}")

    _destroy_distributed_if_needed()


if __name__ == "__main__":
    try:
        main()
    finally:
        _destroy_distributed_if_needed()
