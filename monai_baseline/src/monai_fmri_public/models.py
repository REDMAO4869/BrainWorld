from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch


def _import_generative_modules():
    try:
        from generative.networks.nets import DiffusionModelUNet, VQVAE
        from generative.networks.schedulers import DDPMScheduler
        return DiffusionModelUNet, VQVAE, DDPMScheduler
    except ImportError:
        project_root = Path(__file__).resolve().parents[3]
        env_root = os.environ.get("GENERATIVE_ROOT", "").strip()
        if env_root and env_root not in sys.path:
            sys.path.insert(0, env_root)
        fallback_root = project_root / "third_party" / "GenerativeModels"
        if fallback_root.exists() and str(fallback_root) not in sys.path:
            sys.path.insert(0, str(fallback_root))
        try:
            from generative.networks.nets import DiffusionModelUNet, VQVAE
            from generative.networks.schedulers import DDPMScheduler
            return DiffusionModelUNet, VQVAE, DDPMScheduler
        except ImportError as error:
            raise ImportError(
                "Could not import `generative`. Install MONAI GenerativeModels so `import generative` works, "
                "set GENERATIVE_ROOT to a local GenerativeModels checkout, or vendor it under third_party/GenerativeModels."
            ) from error


def build_vqvae(config: dict[str, Any]):
    _, VQVAE, _ = _import_generative_modules()
    return VQVAE(
        spatial_dims=int(config.get("spatial_dims", 3)),
        in_channels=int(config.get("in_channels", 1)),
        out_channels=int(config.get("out_channels", 1)),
        num_channels=tuple(config["num_channels"]),
        num_res_layers=int(config.get("num_res_layers", 2)),
        num_res_channels=tuple(config["num_res_channels"]),
        downsample_parameters=tuple(tuple(values) for values in config["downsample_parameters"]),
        upsample_parameters=tuple(tuple(values) for values in config["upsample_parameters"]),
        num_embeddings=int(config.get("num_embeddings", 1024)),
        embedding_dim=int(config.get("embedding_dim", 4)),
        commitment_cost=float(config.get("commitment_cost", 0.25)),
        decay=float(config.get("decay", 0.99)),
        epsilon=float(config.get("epsilon", 1e-5)),
        dropout=float(config.get("dropout", 0.0)),
        ddp_sync=bool(config.get("ddp_sync", False)),
        use_checkpointing=bool(config.get("use_checkpointing", False)),
    )


def build_diffusion_unet(config: dict[str, Any], num_tasks: int):
    DiffusionModelUNet, _, _ = _import_generative_modules()
    latent_channels = int(config["latent_channels"])
    num_frames = int(config["num_frames"])
    stacked_channels = latent_channels * num_frames
    return DiffusionModelUNet(
        spatial_dims=int(config.get("spatial_dims", 3)),
        in_channels=stacked_channels,
        out_channels=stacked_channels,
        num_res_blocks=int(config.get("num_res_blocks", 2)),
        num_channels=tuple(config["num_channels"]),
        attention_levels=tuple(bool(value) for value in config["attention_levels"]),
        norm_num_groups=int(config.get("norm_num_groups", 32)),
        norm_eps=float(config.get("norm_eps", 1e-6)),
        resblock_updown=bool(config.get("resblock_updown", True)),
        num_head_channels=tuple(config["num_head_channels"]),
        num_class_embeds=num_tasks + 1,
    )


def build_scheduler(config: dict[str, Any]):
    _, _, DDPMScheduler = _import_generative_modules()
    return DDPMScheduler(
        num_train_timesteps=int(config.get("num_train_timesteps", 1000)),
        schedule=str(config.get("schedule", "scaled_linear_beta")),
        beta_start=float(config.get("beta_start", 0.0015)),
        beta_end=float(config.get("beta_end", 0.0195)),
        clip_sample=bool(config.get("clip_sample", False)),
        prediction_type=str(config.get("prediction_type", "epsilon")),
    )


def flatten_time_to_channels(latents: torch.Tensor) -> torch.Tensor:
    batch, frames, channels, depth, height, width = latents.shape
    return latents.reshape(batch, frames * channels, depth, height, width)


def unflatten_channels_to_time(stacked: torch.Tensor, *, num_frames: int, latent_channels: int) -> torch.Tensor:
    batch, channels, depth, height, width = stacked.shape
    expected_channels = num_frames * latent_channels
    if channels != expected_channels:
        raise ValueError(f"Expected {expected_channels} channels, got {channels}")
    return stacked.reshape(batch, num_frames, latent_channels, depth, height, width)
