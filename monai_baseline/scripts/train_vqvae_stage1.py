from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from monai_fmri_public.config import load_json, load_jsonl, resolve_paths, save_json, save_resolved_config
from monai_fmri_public.data import ClipDataset, RandomFrameDataset
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
from monai_fmri_public.models import build_vqvae
from monai_fmri_public.visualization import save_stage1_reconstruction_visuals
from monai_fmri_public.utils import (
    count_parameters,
    cycle,
    ensure_dir,
    load_partial_weights,
    save_checkpoint,
    seed_everything,
)


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
    device,
    amp_enabled: bool,
    max_batches: int,
    recon_loss_name: str,
    quantization_weight: float,
    dist_ctx: DistributedContext,
) -> float:
    model.eval()
    local_sum = 0.0
    local_count = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            reconstruction, quantization_loss = model(images=images)
            if recon_loss_name == "l1":
                recon_loss = F.l1_loss(reconstruction.float(), images.float())
            elif recon_loss_name == "mse":
                recon_loss = F.mse_loss(reconstruction.float(), images.float())
            else:
                raise ValueError(f"Unsupported recon loss: {recon_loss_name}")
            loss = recon_loss + quantization_weight * quantization_loss.float().mean()
        local_sum += float(loss.item())
        local_count += 1
    model.train()

    local_avg = local_sum / max(local_count, 1)
    return reduce_mean_scalar(local_avg, dist_ctx)


def _print_startup_summary(
    config: dict,
    dist_ctx: DistributedContext,
    model: torch.nn.Module,
    train_size: int,
    val_size: int,
) -> None:
    loader_cfg = config.get("loader", {})
    training_cfg = config.get("training", {})
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    frozen_params = total_params - trainable_params
    per_rank_batch = int(loader_cfg.get("batch_size", 1))
    global_batch = per_rank_batch * dist_ctx.world_size
    print_main(dist_ctx, "=" * 88)
    print_main(dist_ctx, "[stage1] startup summary")
    print_main(dist_ctx, f"[stage1] ddp_enabled={dist_ctx.enabled} world_size={dist_ctx.world_size} device={dist_ctx.device}")
    print_main(dist_ctx, f"[stage1] train_size={train_size} val_size={val_size}")
    print_main(dist_ctx, f"[stage1] batch_size_per_rank={per_rank_batch} global_batch_size={global_batch}")
    print_main(
        dist_ctx,
        f"[stage1] params_total={total_params:,} params_trainable={trainable_params:,} params_frozen={frozen_params:,}",
    )
    print_main(
        dist_ctx,
        f"[stage1] max_steps={int(training_cfg.get('max_steps', 1000))} "
        f"log_every={int(training_cfg.get('log_every', 20))} "
        f"val_every={int(training_cfg.get('val_every', 500))} "
        f"save_every={int(training_cfg.get('save_every', 1000))}",
    )
    print_main(dist_ctx, "=" * 88)




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
    parser = argparse.ArgumentParser(description="Train stage-1 MONAI VQ-VAE on 3D frames sampled from 4D clips.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_json(args.config)
    config = resolve_paths(
        config,
        Path(args.config).resolve().parent,
        [
            ("manifest_paths", "train"),
            ("manifest_paths", "val"),
            ("output_dir",),
            ("warm_start", "checkpoint_path"),
        ],
    )
    dist_ctx = setup_distributed(config)
    try:
        seed_everything(int(config.get("seed", 42)) + dist_ctx.rank)

        if dist_ctx.enabled:
            config.setdefault("model", {})["ddp_sync"] = True

        output_dir = ensure_dir(config["output_dir"])
        if dist_ctx.is_main:
            save_resolved_config(config, output_dir)
        barrier(dist_ctx)

        device = dist_ctx.device
        amp_enabled = bool(config.get("amp", True)) and device.type == "cuda"

        train_records = _load_manifest_records(config["manifest_paths"]["train"])
        val_records = _load_manifest_records(config["manifest_paths"]["val"])
        train_dataset = RandomFrameDataset(ClipDataset(train_records, config["data"]), seed=int(config.get("seed", 42)) + dist_ctx.rank)
        val_dataset = RandomFrameDataset(ClipDataset(val_records, config["data"]), seed=int(config.get("seed", 42)) + 1 + dist_ctx.rank)
        train_loader, train_sampler = build_loader(train_dataset, config["loader"], shuffle=True, dist_ctx=dist_ctx)
        val_loader, _ = build_loader(val_dataset, config["loader"], shuffle=False, dist_ctx=dist_ctx)

        base_model = build_vqvae(config["model"]).to(device)
        _print_startup_summary(config, dist_ctx, base_model, len(train_dataset), len(val_dataset))

        warm_start = config.get("warm_start", {})
        if warm_start.get("checkpoint_path"):
            warm_start_report = load_partial_weights(
                base_model,
                warm_start["checkpoint_path"],
                strict=bool(warm_start.get("strict", False)),
            )
            if dist_ctx.is_main:
                save_json(output_dir / "warm_start_report.json", warm_start_report)
                print(f"Loaded warm start: {warm_start_report['matched_keys']} matched keys")
        barrier(dist_ctx)

        model = wrap_ddp(
            base_model,
            dist_ctx,
            find_unused_parameters=bool(config.get("training", {}).get("ddp_find_unused_parameters", False)),
        )

        optimizer_config = config["optimizer"]
        optimizer = torch.optim.AdamW(
            unwrap_model(model).parameters(),
            lr=float(optimizer_config.get("lr", 1e-4)),
            betas=tuple(optimizer_config.get("betas", [0.9, 0.95])),
            weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
        )

        training_config = config["training"]
        max_steps = int(training_config.get("max_steps", 1000))
        log_every = int(training_config.get("log_every", 20))
        val_every = int(training_config.get("val_every", 500))
        save_every = int(training_config.get("save_every", 1000))
        max_val_batches = int(training_config.get("max_val_batches", 10))
        recon_loss_name = str(training_config.get("recon_loss", "l1"))
        quantization_weight = float(training_config.get("quantization_weight", 1.0))

        vis_config = config.get("visualization", {})
        vis_enabled = bool(vis_config.get("enabled", False))
        vis_every = max(1, int(vis_config.get("every_n_steps", val_every)))
        vis_num_samples = max(1, int(vis_config.get("num_samples", 4)))
        vis_save_npz = bool(vis_config.get("save_npz", True))
        vis_dpi = int(vis_config.get("dpi", 130))
        vis_dir = output_dir / str(vis_config.get("output_subdir", "visualizations/stage1_recon"))

        if args.dry_run:
            batch = next(iter(train_loader))
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                reconstruction, quantization_loss = model(images=images)
            if dist_ctx.is_main:
                print({
                    "device": str(device),
                    "ddp_enabled": dist_ctx.enabled,
                    "world_size": dist_ctx.world_size,
                    "image_shape": tuple(images.shape),
                    "reconstruction_shape": tuple(reconstruction.shape),
                    "quantization_loss": float(quantization_loss.float().mean().item()),
                })
            return

        train_iterator = cycle(train_loader, train_sampler)
        best_val_loss = float("inf")
        model.train()
        progress = tqdm(
            range(1, max_steps + 1),
            total=max_steps,
            desc="stage1",
            dynamic_ncols=True,
            disable=not dist_ctx.is_main,
        )
        for step in progress:
            batch = next(train_iterator)
            images = batch["image"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                reconstruction, quantization_loss = model(images=images)
                if recon_loss_name == "l1":
                    recon_loss = F.l1_loss(reconstruction.float(), images.float())
                elif recon_loss_name == "mse":
                    recon_loss = F.mse_loss(reconstruction.float(), images.float())
                else:
                    raise ValueError(f"Unsupported recon loss: {recon_loss_name}")
                loss = recon_loss + quantization_weight * quantization_loss.float().mean()
            loss.backward()
            optimizer.step()

            mean_loss = reduce_mean_scalar(loss, dist_ctx)
            mean_recon = reduce_mean_scalar(recon_loss, dist_ctx)
            mean_quant = reduce_mean_scalar(quantization_loss.float().mean(), dist_ctx)

            if dist_ctx.is_main:
                progress.set_postfix(
                    loss=f"{mean_loss:.6f}",
                    recon=f"{mean_recon:.6f}",
                    quant=f"{mean_quant:.6f}",
                    best_val=f"{best_val_loss:.6f}" if best_val_loss < float("inf") else "n/a",
                    refresh=(step % log_every == 0 or step == 1),
                )

            if dist_ctx.is_main and vis_enabled and (step % vis_every == 0 or step == 1 or step == max_steps):
                save_stage1_reconstruction_visuals(
                    images=images.detach().float().cpu(),
                    reconstructions=reconstruction.detach().float().cpu(),
                    batch=batch,
                    output_dir=vis_dir,
                    step=step,
                    num_samples=vis_num_samples,
                    save_npz=vis_save_npz,
                    dpi=vis_dpi,
                )

            if step % val_every == 0 or step == max_steps:
                val_loss = evaluate(
                    model,
                    val_loader,
                    device,
                    amp_enabled,
                    max_val_batches,
                    recon_loss_name,
                    quantization_weight,
                    dist_ctx,
                )
                if dist_ctx.is_main:
                    progress.write(f"[stage1] step={step} val_loss={val_loss:.6f}")
                if dist_ctx.is_main and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        output_dir / "vqvae_best.pt",
                        {
                            "model_state_dict": unwrap_model(model).state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "step": step,
                            "best_val_loss": best_val_loss,
                            "config": config,
                        },
                    )
                best_val_loss = reduce_mean_scalar(best_val_loss, dist_ctx)
                barrier(dist_ctx)

            if step % save_every == 0 or step == max_steps:
                if dist_ctx.is_main:
                    save_checkpoint(
                        output_dir / "vqvae_last.pt",
                        {
                            "model_state_dict": unwrap_model(model).state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "step": step,
                            "best_val_loss": best_val_loss,
                            "config": config,
                        },
                    )
                barrier(dist_ctx)

        if dist_ctx.is_main:
            progress.close()
        print_main(dist_ctx, f"Stage-1 training finished. Best validation loss: {best_val_loss:.6f}")
    finally:
        cleanup_distributed(dist_ctx)


if __name__ == "__main__":
    main()
