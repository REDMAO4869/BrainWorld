from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PLANE_SPECS = (
    ("sagittal", 0),
    ("coronal", 1),
    ("axial", 2),
)


def _to_volume(array: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        array = array.detach().float().cpu().numpy()
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"Expected a single-channel 4D tensor [1,D,H,W], got {array.shape}")
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D volume [D,H,W], got {array.shape}")
    return array


def _slice_for_plane(volume: np.ndarray, axis: int) -> np.ndarray:
    center_index = volume.shape[axis] // 2
    if axis == 0:
        image = volume[center_index, :, :]
    elif axis == 1:
        image = volume[:, center_index, :]
    elif axis == 2:
        image = volume[:, :, center_index]
    else:
        raise ValueError(f"Unsupported axis: {axis}")
    return np.rot90(image)


def _robust_limits(*arrays: np.ndarray) -> tuple[float, float]:
    merged = np.concatenate([arr.reshape(-1) for arr in arrays], axis=0)
    finite = merged[np.isfinite(merged)]
    if finite.size == 0:
        return -1.0, 1.0
    vmin = float(np.percentile(finite, 1.0))
    vmax = float(np.percentile(finite, 99.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        center = float(finite.mean())
        radius = float(finite.std())
        radius = radius if radius > 1e-6 else 1.0
        return center - radius, center + radius
    return vmin, vmax


def _safe_meta_value(values: Any, index: int, default: Any) -> Any:
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


def save_stage1_reconstruction_visuals(
    *,
    images: torch.Tensor,
    reconstructions: torch.Tensor,
    batch: dict[str, Any],
    output_dir: str | Path,
    step: int,
    num_samples: int = 4,
    save_npz: bool = True,
    dpi: int = 130,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = min(int(images.shape[0]), int(reconstructions.shape[0]), int(num_samples))
    for sample_index in range(batch_size):
        gt_volume = _to_volume(images[sample_index])
        recon_volume = _to_volume(reconstructions[sample_index])
        error_volume = np.abs(gt_volume - recon_volume)
        image_vmin, image_vmax = _robust_limits(gt_volume, recon_volume)
        error_vmax = float(np.percentile(error_volume[np.isfinite(error_volume)], 99.0)) if np.isfinite(error_volume).any() else 1.0
        error_vmax = error_vmax if error_vmax > 1e-8 else 1.0

        subject = _safe_meta_value(batch.get("subject"), sample_index, "unknown")
        session = _safe_meta_value(batch.get("session"), sample_index, "unknown")
        task = _safe_meta_value(batch.get("task"), sample_index, "unknown")
        segment = _safe_meta_value(batch.get("segment"), sample_index, -1)
        frame_index = _safe_meta_value(batch.get("frame_index"), sample_index, -1)

        prefix = (
            f"step_{int(step):06d}"
            f"_sample_{sample_index:02d}"
            f"_sub-{subject}"
            f"_ses-{session}"
            f"_task-{task}"
            f"_seg-{segment}"
            f"_frame-{frame_index}"
        )

        fig, axes = plt.subplots(3, 3, figsize=(9.5, 9.5), constrained_layout=True)
        for row_index, (plane_name, axis) in enumerate(PLANE_SPECS):
            gt_slice = _slice_for_plane(gt_volume, axis)
            recon_slice = _slice_for_plane(recon_volume, axis)
            error_slice = _slice_for_plane(error_volume, axis)

            axes[row_index, 0].imshow(gt_slice, cmap="gray", vmin=image_vmin, vmax=image_vmax)
            axes[row_index, 1].imshow(recon_slice, cmap="gray", vmin=image_vmin, vmax=image_vmax)
            axes[row_index, 2].imshow(error_slice, cmap="magma", vmin=0.0, vmax=error_vmax)

            axes[row_index, 0].set_ylabel(plane_name)
            if row_index == 0:
                axes[row_index, 0].set_title("GT")
                axes[row_index, 1].set_title("Recon")
                axes[row_index, 2].set_title("Abs Error")
            for col_index in range(3):
                axes[row_index, col_index].set_xticks([])
                axes[row_index, col_index].set_yticks([])

        fig.suptitle(
            f"step={int(step)} subject={subject} session={session} task={task} segment={segment} frame={frame_index}",
            fontsize=10,
        )
        png_path = output_dir / f"{prefix}.png"
        fig.savefig(png_path, dpi=int(dpi))
        plt.close(fig)

        if save_npz:
            np.savez_compressed(
                output_dir / f"{prefix}.npz",
                gt=gt_volume.astype(np.float32),
                reconstruction=recon_volume.astype(np.float32),
                abs_error=error_volume.astype(np.float32),
                subject=np.asarray(str(subject)),
                session=np.asarray(str(session)),
                task=np.asarray(str(task)),
                segment=np.asarray(int(segment), dtype=np.int64),
                frame_index=np.asarray(int(frame_index), dtype=np.int64),
                step=np.asarray(int(step), dtype=np.int64),
            )
