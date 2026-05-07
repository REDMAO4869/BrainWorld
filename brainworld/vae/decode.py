from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from brainworld.vae.model import build_model_from_config
from brainworld.vae.utils import ensure_dir, load_json, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Decode BrainWorld VAE latents to NIfTI and compare with source")
    p.add_argument("--config", required=True, help="Path to decode JSON config")
    return p.parse_args()


def _device_from_cfg(cfg: Dict) -> torch.device:
    dev = str(cfg.get("device", "auto"))
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(dev)


def _load_model_cfg_from_ckpt(ckpt_path: str) -> Dict | None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_cfg = ckpt.get("model_config", None)
    if isinstance(model_cfg, dict) and "model" in model_cfg:
        return model_cfg
    return None


def _find_np_array(d: np.lib.npyio.NpzFile) -> np.ndarray:
    keys = list(d.keys())
    if len(keys) == 0:
        raise ValueError("npz has no arrays")
    for key in ("mu", "z", "logvar", "arr_0", "arr", "data", "x"):
        if key in d:
            return d[key]
    return d[keys[0]]


def _load_source_as_tdhw(path: str, layout: str) -> np.ndarray:
    if path.endswith('.npz'):
        with np.load(path) as d:
            arr = _find_np_array(d)
    elif path.endswith('.npy'):
        arr = np.load(path)
    else:
        raise ValueError(f"Unsupported source file: {path}")

    arr = np.asarray(arr, dtype=np.float32)
    layout = str(layout).upper()
    if arr.ndim != 4:
        raise ValueError(f"Expected source 4D, got shape={arr.shape}")

    if layout == 'DHWT':
        arr = np.transpose(arr, (3, 0, 1, 2))
    elif layout == 'TDHW':
        pass
    elif layout == 'THWD':
        arr = np.transpose(arr, (0, 3, 1, 2))
    else:
        raise ValueError(f"Unsupported layout={layout}, expected DHWT/TDHW/THWD")
    return arr.astype(np.float32, copy=False)


def _parse_bbox_crop(data_cfg: Dict) -> Optional[Tuple[int, int, int, int, int, int]]:
    bc = data_cfg.get('bbox_crop', {})
    if not isinstance(bc, dict):
        return None
    if not bool(bc.get('enabled', False)):
        return None

    def _pair(name: str) -> Tuple[int, int]:
        v = bc.get(name, None)
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ValueError(f"data.bbox_crop.{name} must be [start,end]")
        a, b = int(v[0]), int(v[1])
        if a < 0 or b <= a:
            raise ValueError(f"invalid data.bbox_crop.{name}={v}")
        return a, b

    z0, z1 = _pair('z')
    y0, y1 = _pair('y')
    x0, x1 = _pair('x')
    return (z0, z1, y0, y1, x0, x1)


def _apply_bbox_tdhw(x: np.ndarray, bbox: Optional[Tuple[int, int, int, int, int, int]]) -> np.ndarray:
    if bbox is None:
        return x
    z0, z1, y0, y1, x0, x1 = bbox
    if not (0 <= z0 < z1 <= x.shape[1] and 0 <= y0 < y1 <= x.shape[2] and 0 <= x0 < x1 <= x.shape[3]):
        raise ValueError(f"bbox {bbox} out of range for TDHW shape={tuple(x.shape)}")
    return x[:, z0:z1, y0:y1, x0:x1]


def _to_dhwt(x_tdhw: np.ndarray) -> np.ndarray:
    return np.transpose(x_tdhw, (1, 2, 3, 0))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    if den < 1.0e-12:
        return 0.0
    return float((a * b).sum() / den)


def _read_manifest_rows(manifest_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _find_all_latent_npz(npz_dir: str) -> List[str]:
    out: List[str] = []
    for root, _, files in os.walk(npz_dir):
        for fn in files:
            if fn.endswith('.npz'):
                out.append(os.path.join(root, fn))
    out.sort()
    return out


def _rows_from_npz_only(npz_dir: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in _find_all_latent_npz(npz_dir):
        stem = os.path.splitext(os.path.basename(path))[0]
        sid = stem.split('__', 1)[0] if '__' in stem else stem
        rows.append(
            {
                'sample_id': stem,
                'subject_id': sid,
                'npz_path': path,
                'source_path': '',
            }
        )
    return rows


def _build_latent_index(npz_dir: str, latent_run_dir: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    roots = [npz_dir]
    if latent_run_dir != '':
        roots.extend([os.path.join(latent_run_dir, 'npz'), latent_run_dir])
    for root in roots:
        if not os.path.isdir(root):
            continue
        for path in _find_all_latent_npz(root):
            out.setdefault(os.path.basename(path), path)
    return out


def _resolve_latent_npz_path(npz_path: str, npz_dir: str, latent_run_dir: str, latent_index: Dict[str, str]) -> str:
    if npz_path != '' and os.path.isfile(npz_path):
        return npz_path

    base = os.path.basename(npz_path) if npz_path != '' else ''
    candidates: List[str] = []
    if base != '':
        candidates.append(os.path.join(npz_dir, base))
        if latent_run_dir != '':
            candidates.append(os.path.join(latent_run_dir, 'npz', base))
            candidates.append(os.path.join(latent_run_dir, base))
        if base in latent_index:
            candidates.append(latent_index[base])

    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    return npz_path


def _build_source_index(source_roots: List[str]) -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = {}
    for root_dir in source_roots:
        if not os.path.isdir(root_dir):
            continue
        for root, _, files in os.walk(root_dir):
            for fn in files:
                if not (fn.endswith('.npz') or fn.endswith('.npy')):
                    continue
                idx.setdefault(fn, []).append(os.path.join(root, fn))
    for key in idx:
        idx[key].sort()
    return idx


def _guess_source_path(npz_path: str, source_index: Dict[str, List[str]]) -> str:
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    tokens = stem.split('__')

    candidates: List[str] = []
    if len(tokens) >= 3:
        core = '__'.join(tokens[1:-1])
        candidates.append(core + '.npz')
        candidates.append(core + '.npy')
    candidates.append(stem + '.npz')
    candidates.append(stem + '.npy')

    for cand in candidates:
        vals = source_index.get(cand, [])
        if len(vals) > 0:
            return vals[0]
    return ''


def _sanitize_token(v: str) -> str:
    out = []
    for c in str(v):
        if c.isalnum() or c in {'_', '-'}:
            out.append(c)
        else:
            out.append('_')
    return ''.join(out).strip('_') or 'unk'


def _extract_views(vol_dhw: np.ndarray) -> Dict[str, np.ndarray]:
    d, h, w = vol_dhw.shape
    return {
        'axial(z-mid)': vol_dhw[d // 2, :, :],
        'coronal(y-mid)': vol_dhw[:, h // 2, :],
        'sagittal(x-mid)': vol_dhw[:, :, w // 2],
    }


def _save_compare_preview_png(out_png: str, gt_tdhw: np.ndarray, rec_tdhw: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    t = int(gt_tdhw.shape[0])
    frame_ids = sorted(set([max(0, t // 4), max(0, t // 2), min(t - 1, (3 * t) // 4)]))

    gt_concat = gt_tdhw.reshape(-1)
    vmin = float(np.percentile(gt_concat, 1.0))
    vmax = float(np.percentile(gt_concat, 99.0))
    if vmax <= vmin:
        vmax = vmin + 1.0e-6

    diff = np.abs(rec_tdhw - gt_tdhw)
    emax = float(np.percentile(diff.reshape(-1), 99.0))
    emax = max(emax, 1.0e-8)

    view_names = ['axial(z-mid)', 'coronal(y-mid)', 'sagittal(x-mid)']
    n_rows = len(frame_ids) * 3
    fig, axes = plt.subplots(n_rows, len(view_names), figsize=(4 * len(view_names), 2.8 * n_rows))
    axes = np.asarray(axes).reshape(n_rows, len(view_names))

    row = 0
    for t_idx in frame_ids:
        gt_views = _extract_views(gt_tdhw[t_idx])
        rc_views = _extract_views(rec_tdhw[t_idx])
        df_views = {k: np.abs(rc_views[k] - gt_views[k]) for k in view_names}
        row_defs = [
            (f'GT t={t_idx}', gt_views, 'gray', (vmin, vmax)),
            (f'Recon t={t_idx}', rc_views, 'gray', (vmin, vmax)),
            (f'|Recon-GT| t={t_idx}', df_views, 'magma', (0.0, emax)),
        ]
        for row_name, row_views, cmap, (lo, hi) in row_defs:
            for j, vn in enumerate(view_names):
                ax = axes[row, j]
                ax.imshow(row_views[vn], cmap=cmap, vmin=lo, vmax=hi)
                if row == 0:
                    ax.set_title(vn)
                if j == 0:
                    ax.set_ylabel(row_name)
                ax.axis('off')
            row += 1

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def _resize_recon_spatial(x_hat: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
    if tuple(x_hat.shape[-3:]) == tuple(target_shape):
        return x_hat
    bt = x_hat.permute(0, 2, 1, 3, 4, 5).reshape(x_hat.shape[0] * x_hat.shape[2], 1, *x_hat.shape[-3:])
    bt = F.interpolate(bt, size=tuple(target_shape), mode='trilinear', align_corners=False)
    return bt.reshape(x_hat.shape[0], x_hat.shape[2], 1, *target_shape).permute(0, 2, 1, 3, 4, 5).contiguous()


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)
    set_seed(int(cfg.get('seed', 42)))

    try:
        import nibabel as nib
    except Exception as e:
        raise RuntimeError('nibabel is required to save .nii.gz. Please install nibabel in current env.') from e

    dec_cfg = cfg.get('decode', {})
    ckpt_path = str(dec_cfg.get('checkpoint', '')).strip()
    latent_run_dir = str(dec_cfg.get('latent_run_dir', '')).strip()
    raw_npz_dir = str(dec_cfg.get('latent_npz_dir', '')).strip()
    latent_field = str(dec_cfg.get('latent_field', 'mu')).lower()
    max_samples = int(dec_cfg.get('max_samples', 0))
    manifest_required = bool(dec_cfg.get('manifest_required', False))
    allow_missing_gt = bool(dec_cfg.get('allow_missing_gt', True))
    source_search_roots = dec_cfg.get('source_search_roots', [])
    if isinstance(source_search_roots, str):
        source_search_roots = [source_search_roots]
    source_search_roots = [str(x) for x in source_search_roots if str(x).strip() != '']

    if ckpt_path == '':
        raise ValueError('decode.checkpoint is required')
    if latent_run_dir == '' and raw_npz_dir == '':
        raise ValueError('Either decode.latent_run_dir or decode.latent_npz_dir is required')

    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.abspath(ckpt_path)
    if latent_run_dir != '' and not os.path.isabs(latent_run_dir):
        latent_run_dir = os.path.abspath(latent_run_dir)
    if raw_npz_dir != '' and not os.path.isabs(raw_npz_dir):
        raw_npz_dir = os.path.abspath(raw_npz_dir)

    manifest_path = os.path.join(latent_run_dir, 'manifest.csv') if latent_run_dir != '' else ''
    if raw_npz_dir != '':
        npz_dir = raw_npz_dir
    elif latent_run_dir != '':
        npz_dir = os.path.join(latent_run_dir, 'npz')
    else:
        npz_dir = ''

    use_manifest = manifest_path != '' and os.path.isfile(manifest_path)
    if manifest_required and not use_manifest:
        raise FileNotFoundError(f'manifest not found (required): {manifest_path}')

    if use_manifest:
        rows = _read_manifest_rows(manifest_path)
    else:
        rows = _rows_from_npz_only(npz_dir) if npz_dir != '' else []
        if len(rows) == 0 and latent_run_dir != '':
            rows = _rows_from_npz_only(latent_run_dir)

    if len(rows) == 0:
        raise RuntimeError('No latent samples found')
    if max_samples > 0:
        rows = rows[:max_samples]

    model_cfg = dict(cfg)
    if bool(dec_cfg.get('auto_load_model_config', True)):
        mcfg = _load_model_cfg_from_ckpt(ckpt_path)
        if mcfg is not None:
            model_cfg['model'] = mcfg['model']
            data = model_cfg.get('data', {})
            for key, value in mcfg.get('data', {}).items():
                data[key] = value
            model_cfg['data'] = data

    data_cfg = model_cfg.get('data', {})
    data_layout = str(data_cfg.get('layout', 'DHWT'))
    bbox = _parse_bbox_crop(data_cfg)

    device = _device_from_cfg(cfg)
    model = build_model_from_config(model_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()

    output_base = ensure_dir(str(dec_cfg.get('output_dir', 'outputs/wf_vae2_decode_preview')))
    run_name = str(dec_cfg.get('run_name', '')).strip() or time.strftime('%Y%m%d_%H%M%S')
    out_root = ensure_dir(os.path.join(output_base, run_name))

    source_index: Dict[str, List[str]] = {}
    if len(source_search_roots) > 0:
        source_index = _build_source_index(source_search_roots)
    latent_index = _build_latent_index(npz_dir, latent_run_dir)

    metric_rows: List[Dict] = []
    print('=' * 88)
    print(f'[audit] latent_run_dir={latent_run_dir}')
    print(f'[audit] latent_npz_dir={npz_dir}')
    print(f'[audit] use_manifest={use_manifest} manifest_required={manifest_required}')
    print(f'[audit] rows={len(rows)} latent_field={latent_field}')
    print(f'[audit] allow_missing_gt={allow_missing_gt}')
    print(f'[audit] source_search_roots={source_search_roots}')
    print(f'[audit] source_index_keys={len(source_index)} latent_index_keys={len(latent_index)}')
    print(f'[audit] data_layout={data_layout} bbox_crop={data_cfg.get("bbox_crop", {})}')
    print(f'[audit] checkpoint={ckpt_path}')
    print(f'[audit] output_dir={out_root}')
    print('=' * 88)

    with torch.inference_mode():
        for i, row in enumerate(rows):
            raw_npz_path = str(row.get('npz_path', '')).strip()
            src_path = str(row.get('source_path', '')).strip()
            sid = _sanitize_token(str(row.get('subject_id', f'sample_{i:06d}')))

            if raw_npz_path == '':
                print(f'[warn] row has empty npz_path idx={i}')
                continue
            if not os.path.isabs(raw_npz_path):
                raw_npz_path = os.path.abspath(raw_npz_path)
            if src_path != '' and not os.path.isabs(src_path):
                src_path = os.path.abspath(src_path)

            npz_path = _resolve_latent_npz_path(raw_npz_path, npz_dir, latent_run_dir, latent_index)
            if not os.path.isfile(npz_path):
                print(f'[warn] missing latent npz: {raw_npz_path}')
                continue
            if npz_path != raw_npz_path:
                print(f'[fix] latent npz remapped: {raw_npz_path} -> {npz_path}')

            if (src_path == '' or (not os.path.isfile(src_path))) and len(source_index) > 0:
                guess = _guess_source_path(npz_path, source_index)
                if guess != '':
                    src_path = guess

            gt_available = src_path != '' and os.path.isfile(src_path)
            if not gt_available and (not allow_missing_gt):
                print(f'[warn] missing source and allow_missing_gt=false, skip: {npz_path}')
                continue

            with np.load(npz_path) as d:
                if latent_field in d:
                    latent = d[latent_field]
                else:
                    latent = _find_np_array(d)

            latent_np = np.asarray(latent, dtype=np.float32)
            if latent_np.ndim != 5:
                print(f'[warn] invalid latent shape={latent_np.shape}, expect [Tz,L,Dz,Hz,Wz], skip: {npz_path}')
                continue
            z = torch.from_numpy(latent_np).unsqueeze(0).to(device)

            if gt_available:
                src_tdhw = _load_source_as_tdhw(src_path, layout=data_layout)
                src_tdhw = _apply_bbox_tdhw(src_tdhw, bbox)
                t0, d0, h0, w0 = src_tdhw.shape
                x_hat = model.decode(z, target_t=t0)
                x_hat = _resize_recon_spatial(x_hat, target_shape=(d0, h0, w0))
            else:
                t_default = int(data_cfg.get('t_frames', int(z.shape[1])))
                x_hat = model.decode(z, target_t=t_default)
                src_tdhw = None

            rec_tdhw = x_hat[0, 0].detach().cpu().numpy().astype(np.float32)
            rec_dhwt = _to_dhwt(rec_tdhw)

            sample_dir = ensure_dir(os.path.join(out_root, f'{i:03d}_{sid}'))
            rec_path = os.path.join(sample_dir, 'recon.nii.gz')
            nib.save(nib.Nifti1Image(rec_dhwt, affine=np.eye(4, dtype=np.float32)), rec_path)

            mse_v = np.nan
            mae_v = np.nan
            pcc_v = np.nan
            gt_path = ''
            diff_path = ''
            preview_path = ''

            if gt_available and src_tdhw is not None:
                mse_v = float(np.mean((rec_tdhw - src_tdhw) ** 2))
                mae_v = float(np.mean(np.abs(rec_tdhw - src_tdhw)))
                pcc_v = _pearson(rec_tdhw, src_tdhw)

                gt_dhwt = _to_dhwt(src_tdhw)
                gt_path = os.path.join(sample_dir, 'gt.nii.gz')
                nib.save(nib.Nifti1Image(gt_dhwt, affine=np.eye(4, dtype=np.float32)), gt_path)

                diff_dhwt = np.abs(rec_dhwt - gt_dhwt).astype(np.float32)
                diff_path = os.path.join(sample_dir, 'diff_abs.nii.gz')
                nib.save(nib.Nifti1Image(diff_dhwt, affine=np.eye(4, dtype=np.float32)), diff_path)

                np.savez(os.path.join(sample_dir, 'compare.npz'), gt=gt_dhwt, recon=rec_dhwt, diff_abs=diff_dhwt)
                preview_path = os.path.join(sample_dir, 'preview.png')
                _save_compare_preview_png(preview_path, src_tdhw, rec_tdhw)
                print(f'[sample {i}] sid={sid} gt=1 mse={mse_v:.6f} mae={mae_v:.6f} pcc={pcc_v:.4f}')
            else:
                np.savez(os.path.join(sample_dir, 'recon_only.npz'), recon=rec_dhwt)
                print(f'[sample {i}] sid={sid} gt=0 recon_saved')

            metric_rows.append(
                {
                    'sample_idx': i,
                    'subject_id': sid,
                    'source_path': src_path,
                    'gt_found': int(gt_available),
                    'latent_npz': npz_path,
                    'gt_nii': gt_path,
                    'recon_nii': rec_path,
                    'diff_abs_nii': diff_path,
                    'preview_png': preview_path,
                    'mse': mse_v,
                    'mae': mae_v,
                    'pearson': pcc_v,
                }
            )

    if len(metric_rows) == 0:
        raise RuntimeError('No sample decoded')

    csv_path = os.path.join(out_root, 'metrics.csv')
    fieldnames = [
        'sample_idx', 'subject_id', 'source_path', 'gt_found', 'latent_npz', 'gt_nii', 'recon_nii',
        'diff_abs_nii', 'preview_png', 'mse', 'mae', 'pearson'
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    gt_rows = [x for x in metric_rows if int(x['gt_found']) == 1]
    summary = {
        'num_samples': len(metric_rows),
        'num_samples_with_gt': len(gt_rows),
        'latent_run_dir': latent_run_dir,
        'latent_npz_dir': npz_dir,
        'checkpoint': ckpt_path,
        'latent_field': latent_field,
        'use_manifest': use_manifest,
        'manifest_required': manifest_required,
        'allow_missing_gt': allow_missing_gt,
        'source_search_roots': source_search_roots,
        'bbox_crop': data_cfg.get('bbox_crop', {}),
        'output_dir': out_root,
        'metrics_csv': csv_path,
    }
    if len(gt_rows) > 0:
        summary.update(
            {
                'mean_mse': float(np.mean([x['mse'] for x in gt_rows])),
                'mean_mae': float(np.mean([x['mae'] for x in gt_rows])),
                'mean_pearson': float(np.mean([x['pearson'] for x in gt_rows])),
            }
        )

    save_json(summary, os.path.join(out_root, 'summary.json'))
    print('=' * 88)
    print(f'[done] decoded preview saved: {out_root}')


if __name__ == '__main__':
    main()
