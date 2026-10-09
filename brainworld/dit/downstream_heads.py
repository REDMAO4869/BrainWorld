from __future__ import annotations

from typing import Optional

import torch.nn as nn


def build_head(*, d_model: int, task_type: str, num_classes: int, hidden_dim: Optional[int] = None, dropout: float = 0.1) -> nn.Module:
    d = int(d_model)
    t = str(task_type).strip().lower()
    if t == "regression":
        return nn.Linear(d, 1)
    if t != "classification":
        raise ValueError("task_type must be classification or regression")
    if int(num_classes) <= 1:
        raise ValueError("num_classes must be >= 2 for classification")
    if hidden_dim is None or int(hidden_dim) <= 0:
        return nn.Linear(d, int(num_classes))
    h = int(hidden_dim)
    return nn.Sequential(
        nn.Linear(d, h),
        nn.ReLU(),
        nn.Dropout(float(dropout)),
        nn.Linear(h, int(num_classes)),
    )
