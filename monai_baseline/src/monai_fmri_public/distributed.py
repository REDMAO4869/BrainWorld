from __future__ import annotations

import os
from datetime import timedelta
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed(config: dict) -> DistributedContext:
    training_cfg = config.get("training", {})
    requested_ddp = bool(training_cfg.get("use_ddp", False))

    has_env = all(name in os.environ for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if requested_ddp and has_env and torch.cuda.is_available():
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size > 1:
            timeout_hours = float(training_cfg.get("process_group_timeout_hours", 6.0))
            dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(hours=timeout_hours))
            torch.cuda.set_device(local_rank)
            return DistributedContext(
                enabled=True,
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                device=torch.device("cuda", local_rank),
            )

    if requested_ddp and not has_env:
        visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if visible <= 1:
            print("[warn] training.use_ddp=true but <=1 visible GPU or no torchrun env; fallback to single process")
        else:
            print("[warn] training.use_ddp=true but torchrun env vars are missing; fallback to single process")

    device_name = str(config.get("device", "cuda"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    return DistributedContext(enabled=False, rank=0, local_rank=0, world_size=1, device=device)


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_available() and dist.is_initialized():
        if ctx.device.type == "cuda":
            dist.barrier(device_ids=[ctx.local_rank])
        else:
            dist.barrier()


def wrap_ddp(module: torch.nn.Module, ctx: DistributedContext, *, find_unused_parameters: bool = False) -> torch.nn.Module:
    if not ctx.enabled:
        return module
    return DistributedDataParallel(
        module,
        device_ids=[ctx.local_rank],
        output_device=ctx.local_rank,
        find_unused_parameters=bool(find_unused_parameters),
    )


def unwrap_model(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


def reduce_mean_scalar(value: float | torch.Tensor, ctx: DistributedContext) -> float:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().float().reshape(1).to(ctx.device)
    else:
        tensor = torch.tensor([float(value)], device=ctx.device, dtype=torch.float32)
    if ctx.enabled:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= float(ctx.world_size)
    return float(tensor.item())


def print_main(ctx: DistributedContext, *args, **kwargs) -> None:
    if ctx.is_main:
        print(*args, **kwargs)
