from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from brainworld.dit.data import audit_to_dict, build_splits_from_config, collate_batch
from brainworld.dit.diffusion import GaussianDiffusion, normalize_prediction_type, normalize_schedule_type
from brainworld.dit.model import ConditionalLatentDiT, compute_patch_audit
from brainworld.dit.utils import ensure_dir, load_json, make_timestamped_dir, resolve_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer BrainWorld conditional latent diffusion transformer")
    p.add_argument("--config", required=True, help="Path to conditional DiT inference JSON config")
    return p.parse_args()


def _load_model_cfg_from_ckpt(ckpt_path: str) -> Dict | None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_cfg = ckpt.get("model_config", None)
    return model_cfg if isinstance(model_cfg, dict) else None


def _build_model_from_dataset(cfg: Dict, ds) -> ConditionalLatentDiT:
    mcfg = cfg.get("model", {})
    condition_shapes = {
        "fc": ds.fc_shape,
        "mri": ds.mri_shape,
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


def _prepare_condition_inputs(batch: Dict[str, object], device: torch.device) -> Dict[str, Optional[torch.Tensor]]:
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
    return out


def _slice_cond_inputs(cond_inputs: Dict[str, Optional[torch.Tensor]], index: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
    out: Dict[str, Optional[torch.Tensor]] = {}
    for k, v in cond_inputs.items():
        if torch.is_tensor(v):
            out[k] = v.index_select(0, index)
        else:
            out[k] = v
    return out


def _normalize_modalities(modalities: Sequence[str]) -> Set[str]:
    norm: Set[str] = set()
    for m in modalities:
        key = str(m).strip().lower()
        if key in {"meta", "metadata"}:
            key = "meta"
        if key in {"fc", "mri", "meta"}:
            norm.add(key)
    return norm


def _parse_mode_spec(spec: object) -> Tuple[str, Set[str]]:
    if isinstance(spec, dict):
        name = str(spec.get("name", "mode")).strip()
        enable = spec.get("enable", spec.get("enabled_modalities", []))
        if isinstance(enable, str):
            toks = [t for t in re.split(r"[+,_/\s]+", enable.strip().lower()) if t]
            enabled = _normalize_modalities(toks)
        elif isinstance(enable, (list, tuple)):
            enabled = _normalize_modalities([str(x) for x in enable])
        else:
            enabled = set()
        return (name if name else "mode", enabled)

    raw = str(spec).strip().lower()
    if raw in {"", "full", "all"}:
        return ("full", {"fc", "mri", "meta"})
    if raw in {"none", "null", "uncond", "unconditional"}:
        return ("uncond", set())
    toks = [t for t in re.split(r"[+,_/\s]+", raw) if t]
    enabled = _normalize_modalities(toks)
    name = raw.replace("/", "_").replace("+", "_").replace(" ", "_")
    return (name if name else "mode", enabled)


def _resolve_ablation_modes(ab_cfg: Dict) -> List[Tuple[str, Set[str]]]:
    modes_cfg = ab_cfg.get("modes", ["uncond", "mri", "fc", "fc_mri", "full"])
    if not isinstance(modes_cfg, (list, tuple)):
        modes_cfg = [modes_cfg]
    out: List[Tuple[str, Set[str]]] = []
    seen: Set[str] = set()
    for spec in modes_cfg:
        name, enabled = _parse_mode_spec(spec)
        key = str(name).strip() or "mode"
        if key in seen:
            suffix = 2
            while f"{key}_{suffix}" in seen:
                suffix += 1
            key = f"{key}_{suffix}"
        seen.add(key)
        out.append((key, enabled))
    return out


def _apply_modality_selection(
    cond_inputs: Dict[str, Optional[torch.Tensor]],
    enabled_modalities: Set[str],
) -> Dict[str, Optional[torch.Tensor]]:
    out = dict(cond_inputs)
    mapping = {
        "fc": ("fc_cond", "has_fc"),
        "mri": ("mri_cond", "has_mri"),
        "meta": ("meta_cond", "has_meta"),
    }
    for modal, (cond_key, has_key) in mapping.items():
        if modal not in enabled_modalities:
            out[cond_key] = None
            out[has_key] = None
    return out


def _guided_predict(
    model: torch.nn.Module,
    xt: torch.Tensor,
    t: torch.Tensor,
    direction_id: torch.Tensor,
    cond_inputs: Dict[str, Optional[torch.Tensor]],
    cfg_cfg: Dict,
    has_optional_condition: Optional[bool] = None,
) -> torch.Tensor:
    enabled = bool(cfg_cfg.get("enabled", False))
    scale = float(cfg_cfg.get("scale", 1.0))
    if not enabled or scale == 1.0:
        out = model(xt, t, direction_id, **cond_inputs)
        return out["pred"] if isinstance(out, dict) else out

    if has_optional_condition is None:
        has_optional_condition = False
        for key in ("has_mri", "has_video", "has_audio", "has_meta"):
            val = cond_inputs.get(key, None)
            if val is None:
                continue
            # This branch is a fallback for call sites that don't precompute the
            # flag. In performance-sensitive loops, pass has_optional_condition
            # once before stepping to avoid repeated host-device sync.
            if bool(torch.any(val > 0).item()):
                has_optional_condition = True
                break
    if not has_optional_condition:
        out = model(xt, t, direction_id, **cond_inputs)
        return out["pred"] if isinstance(out, dict) else out

    full = model(xt, t, direction_id, **cond_inputs)
    full_pred = full["pred"] if isinstance(full, dict) else full

    base_inputs = dict(cond_inputs)
    for key in ("has_mri", "has_video", "has_audio", "has_meta"):
        if base_inputs.get(key, None) is not None:
            base_inputs[key] = torch.zeros_like(base_inputs[key])
    base = model(xt, t, direction_id, **base_inputs)
    base_pred = base["pred"] if isinstance(base, dict) else base
    return base_pred + scale * (full_pred - base_pred)


def _load_decoder(dec_cfg: Dict, device: torch.device):
    if not bool(dec_cfg.get("enabled", False)):
        return None

    project_root = str(dec_cfg.get("project_root", "")).strip()
    if project_root != "":
        project_root = os.path.abspath(project_root)
    ckpt_path = str(dec_cfg.get("checkpoint", "")).strip()
    if ckpt_path == "":
        raise ValueError("decoder.checkpoint is required when decoder.enabled=true")
    ckpt_path = os.path.abspath(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"decoder checkpoint not found: {ckpt_path}")

    from brainworld.vae.model import build_model_from_config

    model_cfg = None
    if bool(dec_cfg.get("auto_load_model_config", True)):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_cfg = ckpt.get("model_config", None)
        if not isinstance(model_cfg, dict) or "model" not in model_cfg:
            model_cfg = None
    if model_cfg is None:
        config_path = str(dec_cfg.get("config_path", "")).strip()
        if config_path == "":
            raise ValueError("decoder auto_load_model_config failed and decoder.config_path is empty")
        model_cfg = load_json(config_path)

    model = build_model_from_config(model_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)
    set_seed(int(cfg.get("seed", 42)))

    inf_cfg = cfg.get("inference", {})
    ckpt_path = str(inf_cfg.get("checkpoint", "")).strip()
    if ckpt_path == "":
        raise ValueError("inference.checkpoint is required")
    if not os.path.isabs(ckpt_path):
        cfg_dir = os.path.dirname(os.path.abspath(args.config))
        ckpt_path = os.path.abspath(os.path.join(cfg_dir, ckpt_path))

    if bool(inf_cfg.get("auto_load_model_config", True)):
        mcfg = _load_model_cfg_from_ckpt(ckpt_path)
        if mcfg is not None:
            for key in ("model", "diffusion", "task", "conditioning", "diversity", "loss"):
                if key in mcfg:
                    cfg[key] = mcfg[key]

            ckpt_data = mcfg.get("data", None)
            if isinstance(ckpt_data, dict):
                ckpt_cond = ckpt_data.get("conditions", None)
                if isinstance(ckpt_cond, dict):
                    if not isinstance(cfg.get("data", None), dict):
                        cfg["data"] = {}
                    if not isinstance(cfg["data"].get("conditions", None), dict):
                        cfg["data"]["conditions"] = {}
                    for k, v in ckpt_cond.items():
                        if k not in cfg["data"]["conditions"]:
                            cfg["data"]["conditions"][k] = v

    prediction_type = normalize_prediction_type(cfg.get("loss", {}).get("prediction_type", "epsilon"))
    schedule_type = normalize_schedule_type(cfg.get("diffusion", {}).get("schedule", "linear"))

    device = resolve_device(cfg)
    train_ds, val_ds, test_ds, audits = build_splits_from_config(cfg)
    split = str(inf_cfg.get("split", "test")).lower()
    ds = {"train": train_ds, "val": val_ds, "test": test_ds}[split]
    loader = DataLoader(
        ds,
        batch_size=int(inf_cfg.get("batch_size", 2)),
        shuffle=False,
        num_workers=int(inf_cfg.get("num_workers", 2)),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
    )

    model = _build_model_from_dataset(cfg, ds).to(device)
    dcfg = cfg.get("diffusion", {})
    diffusion = GaussianDiffusion(
        num_steps=int(dcfg.get("num_steps", 1000)),
        beta_start=float(dcfg.get("beta_start", 1.0e-4)),
        beta_end=float(dcfg.get("beta_end", 2.0e-2)),
        schedule=schedule_type,
        cosine_s=float(dcfg.get("cosine_s", 0.008)),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    decoder = _load_decoder(inf_cfg.get("decoder", {}), device=device) if isinstance(inf_cfg.get("decoder", {}), dict) else None
    decoder_target_t = int(inf_cfg.get("decoder", {}).get("target_t", 40)) if decoder is not None else 0

    out_base = ensure_dir(str(inf_cfg.get("output_dir", "outputs/cond_dit_infer")))
    out_root = make_timestamped_dir(out_base)
    npz_dir = ensure_dir(os.path.join(out_root, "npz"))
    decode_dir = ensure_dir(os.path.join(out_root, "decoded_npz")) if decoder is not None else ""

    max_samples = int(inf_cfg.get("max_samples", 16))
    save_dtype = np.float16 if str(inf_cfg.get("save_dtype", "float32")).lower() == "float16" else np.float32
    clip_x0 = inf_cfg.get("clip_x0", None)
    cfg_cfg = inf_cfg.get("cfg", {})
    patch_info = compute_patch_audit(ds.target_shape, cfg.get("model", {}).get("patch_size", [1, 2, 1]))

    print("=" * 100)
    print(f"[audit] split={split} target_shape={tuple(int(v) for v in ds.target_shape)}")
    print(f"[audit] diffusion_schedule={schedule_type} prediction_type={prediction_type}")
    print(f"[audit] patch_num={patch_info['patch_num']} patch_dim={patch_info['patch_dim']} grid={patch_info['grid_shape']}")
    print(f"[audit] condition_shapes fc={ds.fc_shape} mri={ds.mri_shape} metadata={(ds.meta_dim if ds.meta_dim > 0 else None)}")
    print(f"[audit] checkpoint={ckpt_path}")
    print(f"[audit] cfg={cfg_cfg}")
    print(f"[audit] dataset={audit_to_dict(audits[split])}")
    print("=" * 100)

    rows: List[Dict[str, object]] = []
    done = 0
    with torch.inference_mode():
        pbar = tqdm(loader, desc=f"infer[{split}]", ncols=120)
        for batch in pbar:
            x0 = batch["target_latent"].to(device=device, dtype=torch.float32)
            direction_id = batch["direction_id"].to(device=device, dtype=torch.long)
            bsz = int(x0.shape[0])
            cond_inputs = _prepare_condition_inputs(batch, device)
            has_optional_condition = False
            for key in ("has_mri", "has_meta"):
                val = cond_inputs.get(key, None)
                if val is not None and bool(torch.any(val > 0).item()):
                    has_optional_condition = True
                    break
            xt = torch.randn_like(x0)

            for step in reversed(range(diffusion.num_steps)):
                t = torch.full((bsz,), step, device=device, dtype=torch.long)
                pred = _guided_predict(
                    model,
                    xt,
                    t,
                    direction_id,
                    cond_inputs,
                    cfg_cfg,
                    has_optional_condition=has_optional_condition,
                )
                xt = diffusion.p_sample(xt, t, pred, prediction_type, clip_x0=clip_x0)

            pred_latent = xt
            latent_mse = F.mse_loss(pred_latent, x0, reduction="none").flatten(1).mean(dim=1)
            decoded = decoder.decode(pred_latent, target_t=decoder_target_t) if decoder is not None else None

            for i in range(bsz):
                if max_samples > 0 and done >= max_samples:
                    break
                sid = str(batch["subject_id"][i])
                anchor_chunk_id = int(batch["anchor_chunk_id"][i].item())
                target_chunk_id = int(batch["target_chunk_id"][i].item())
                direction = str(batch["direction"][i])
                out_npz = os.path.join(npz_dir, f"{done:06d}__{sid}__{direction}.npz")
                payload = {
                    "pred_latent": pred_latent[i].detach().cpu().numpy().astype(save_dtype),
                    "target_latent": x0[i].detach().cpu().numpy().astype(save_dtype),
                    "direction": np.array([direction]),
                    "anchor_chunk_id": np.array([anchor_chunk_id], dtype=np.int32),
                    "target_chunk_id": np.array([target_chunk_id], dtype=np.int32),
                }
                np.savez(out_npz, **payload)

                decoded_npz = ""
                if decoded is not None:
                    decoded_npz = os.path.join(decode_dir, f"{done:06d}__{sid}__{direction}.npz")
                    np.savez(decoded_npz, x_hat=decoded[i].detach().cpu().numpy().astype(np.float32))

                rows.append(
                    {
                        "idx": done,
                        "subject_id": sid,
                        "direction": direction,
                        "anchor_chunk_id": anchor_chunk_id,
                        "target_chunk_id": target_chunk_id,
                        "target_npz_path": str(batch["target_npz_path"][i]),
                        "pred_npz": out_npz,
                        "decoded_npz": decoded_npz,
                        "latent_mse": float(latent_mse[i].item()),
                    }
                )
                done += 1
            pbar.set_postfix(done=done)
            if max_samples > 0 and done >= max_samples:
                break

    if len(rows) == 0:
        raise RuntimeError("No inference samples produced")

    csv_path = os.path.join(out_root, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "checkpoint": ckpt_path,
        "split": split,
        "num_samples": len(rows),
        "mean_latent_mse": float(np.mean([r["latent_mse"] for r in rows])),
        "output_dir": out_root,
        "metrics_csv": csv_path,
        "dataset_audit": audit_to_dict(audits[split]),
        "patch_audit": patch_info,
        "prediction_type": prediction_type,
        "schedule": schedule_type,
        "cfg": cfg_cfg,
        "decoder_enabled": decoder is not None,
    }
    ab_cfg = inf_cfg.get("condition_ablation", {})
    if isinstance(ab_cfg, dict) and bool(ab_cfg.get("enabled", False)):
        modes = _resolve_ablation_modes(ab_cfg)
        subj_filter = {
            str(x).strip()
            for x in ab_cfg.get("subject_ids", [])
            if str(x).strip() != ""
        }
        max_samples_ab = int(ab_cfg.get("max_samples", max_samples))
        max_per_subject = int(ab_cfg.get("max_per_subject", 0))
        disable_cfg_for_ablation = bool(ab_cfg.get("disable_cfg", True))
        cfg_for_ablation = {"enabled": False, "scale": 1.0} if disable_cfg_for_ablation else cfg_cfg
        mode_rows: List[Dict[str, object]] = []
        per_subject_seen: Dict[str, int] = {}
        done_ab = 0

        with torch.inference_mode():
            pbar = tqdm(loader, desc=f"ablation[{split}]", ncols=120)
            for batch in pbar:
                if max_samples_ab > 0 and done_ab >= max_samples_ab:
                    break

                subject_ids = [str(s) for s in batch["subject_id"]]
                keep: List[int] = []
                for i, sid in enumerate(subject_ids):
                    if len(subj_filter) > 0 and sid not in subj_filter:
                        continue
                    if max_per_subject > 0 and per_subject_seen.get(sid, 0) >= max_per_subject:
                        continue
                    keep.append(i)

                if len(keep) == 0:
                    continue
                if max_samples_ab > 0:
                    remain = max_samples_ab - done_ab
                    if remain <= 0:
                        break
                    keep = keep[:remain]

                idx = torch.as_tensor(keep, device=device, dtype=torch.long)
                x0_full = batch["target_latent"].to(device=device, dtype=torch.float32)
                direction_full = batch["direction_id"].to(device=device, dtype=torch.long)
                x0 = x0_full.index_select(0, idx)
                direction_id = direction_full.index_select(0, idx)
                cond_inputs = _slice_cond_inputs(_prepare_condition_inputs(batch, device), idx)

                xt_seed = torch.randn_like(x0)
                per_mode_mse: Dict[str, torch.Tensor] = {}
                for mode_name, enabled_modalities in modes:
                    mode_inputs = _apply_modality_selection(cond_inputs, enabled_modalities)
                    has_optional_condition = False
                    for key in ("has_mri", "has_meta"):
                        val = mode_inputs.get(key, None)
                        if val is not None and bool(torch.any(val > 0).item()):
                            has_optional_condition = True
                            break
                    xt_mode = xt_seed.clone()
                    bsz = int(xt_mode.shape[0])
                    for step in reversed(range(diffusion.num_steps)):
                        t = torch.full((bsz,), step, device=device, dtype=torch.long)
                        pred = _guided_predict(
                            model,
                            xt_mode,
                            t,
                            direction_id,
                            mode_inputs,
                            cfg_for_ablation,
                            has_optional_condition=has_optional_condition,
                        )
                        xt_mode = diffusion.p_sample(xt_mode, t, pred, prediction_type, clip_x0=clip_x0)
                    per_mode_mse[mode_name] = F.mse_loss(xt_mode, x0, reduction="none").flatten(1).mean(dim=1)

                for local_i, src_i in enumerate(keep):
                    sid = str(batch["subject_id"][src_i])
                    if max_per_subject > 0 and per_subject_seen.get(sid, 0) >= max_per_subject:
                        continue
                    per_subject_seen[sid] = per_subject_seen.get(sid, 0) + 1

                    anchor_chunk_id = int(batch["anchor_chunk_id"][src_i].item())
                    target_chunk_id = int(batch["target_chunk_id"][src_i].item())
                    direction = str(batch["direction"][src_i])
                    target_npz_path = str(batch["target_npz_path"][src_i])
                    for mode_name, _ in modes:
                        mode_rows.append(
                            {
                                "sample_idx": done_ab,
                                "subject_id": sid,
                                "direction": direction,
                                "anchor_chunk_id": anchor_chunk_id,
                                "target_chunk_id": target_chunk_id,
                                "target_npz_path": target_npz_path,
                                "mode": mode_name,
                                "latent_mse": float(per_mode_mse[mode_name][local_i].item()),
                            }
                        )
                    done_ab += 1
                    if max_samples_ab > 0 and done_ab >= max_samples_ab:
                        break
                pbar.set_postfix(done=done_ab)

        if len(mode_rows) > 0:
            mode_csv = os.path.join(out_root, str(ab_cfg.get("metrics_csv", "condition_ablation_metrics.csv")))
            with open(mode_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(mode_rows[0].keys()))
                w.writeheader()
                w.writerows(mode_rows)

            mode_values: Dict[str, List[float]] = {}
            for r in mode_rows:
                mode_values.setdefault(str(r["mode"]), []).append(float(r["latent_mse"]))
            mode_mean = {k: float(np.mean(v)) for k, v in mode_values.items() if len(v) > 0}
            improvement_vs_uncond = None
            if "uncond" in mode_mean:
                base = float(mode_mean["uncond"])
                improvement_vs_uncond = {k: float(base - float(v)) for k, v in mode_mean.items()}

            summary["condition_ablation"] = {
                "enabled": True,
                "subject_filter_count": int(len(subj_filter)),
                "max_samples": int(max_samples_ab),
                "max_per_subject": int(max_per_subject),
                "disable_cfg": bool(disable_cfg_for_ablation),
                "modes": [name for name, _ in modes],
                "num_samples": int(done_ab),
                "mode_mean_latent_mse": mode_mean,
                "improvement_vs_uncond": improvement_vs_uncond,
                "metrics_csv": mode_csv,
            }
        else:
            summary["condition_ablation"] = {
                "enabled": True,
                "num_samples": 0,
                "message": "No samples matched condition_ablation filters",
            }

    save_json(summary, os.path.join(out_root, "summary.json"))
    print(f"[done] conditional DiT inference outputs: {out_root}")


if __name__ == "__main__":
    main()
