from __future__ import annotations

import torch


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow = {name: value.detach().clone() for name, value in model.state_dict().items()}

    def update(self, model: torch.nn.Module) -> None:
        for name, value in model.state_dict().items():
            self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in self.shadow.items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.shadow = {name: value.detach().clone() for name, value in state_dict.items()}

    def to(self, device: torch.device) -> None:
        self.shadow = {name: value.to(device=device) for name, value in self.shadow.items()}

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)
