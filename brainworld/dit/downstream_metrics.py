from __future__ import annotations

from typing import Dict

import torch


def _f1_binary(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return float(2.0 * prec * rec / max(1.0e-12, prec + rec))


def evaluate_classification(y_true: torch.Tensor, logits: torch.Tensor, *, num_classes: int) -> Dict[str, float]:
    y = y_true.long().view(-1)
    pred = torch.argmax(logits, dim=1)
    acc = float((pred == y).float().mean().item())

    f1_sum = 0.0
    support_sum = 0
    for c in range(int(num_classes)):
        yc = (y == c).long()
        pc = (pred == c).long()
        f1c = _f1_binary(yc, pc)
        sup = int(yc.sum().item())
        f1_sum += f1c * sup
        support_sum += sup
    f1_weighted = float(f1_sum / max(1, support_sum))

    bal_acc = 0.0
    cls_present = 0
    for c in range(int(num_classes)):
        yc = (y == c)
        sup = int(yc.sum().item())
        if sup <= 0:
            continue
        cls_present += 1
        bal_acc += float((pred[yc] == c).float().mean().item())
    bal_acc = float(bal_acc / max(1, cls_present))

    out = {
        "acc": acc,
        "balanced_acc": bal_acc,
        "f1_weighted": f1_weighted,
    }

    if int(num_classes) == 2:
        score = torch.softmax(logits, dim=-1)[:, 1]
        # Fast AUC without sklearn; falls back to 0.5 in degenerate cases.
        pos = (y == 1)
        neg = (y == 0)
        n_pos = int(pos.sum().item())
        n_neg = int(neg.sum().item())
        if n_pos > 0 and n_neg > 0:
            order = torch.argsort(score)
            ranks = torch.zeros_like(order, dtype=torch.float32)
            ranks[order] = torch.arange(1, score.numel() + 1, device=score.device, dtype=torch.float32)
            rank_pos = float(ranks[pos].sum().item())
            auc = (rank_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
            out["auroc"] = float(auc)
        else:
            out["auroc"] = 0.5

    return out


def evaluate_regression(y_true: torch.Tensor, pred: torch.Tensor) -> Dict[str, float]:
    yt = y_true.view(-1).float()
    yp = pred.view(-1).float()
    mse = float(torch.mean((yp - yt) ** 2).item())
    mae = float(torch.mean(torch.abs(yp - yt)).item())

    x = yp - yp.mean()
    y = yt - yt.mean()
    denom = float(torch.sqrt(torch.sum(x * x) * torch.sum(y * y)).item())
    pearson = float((torch.sum(x * y).item() / max(1.0e-12, denom)))

    sse = float(torch.sum((yt - yp) ** 2).item())
    sst = float(torch.sum((yt - yt.mean()) ** 2).item())
    r2 = float(1.0 - sse / max(1.0e-12, sst))

    return {"mse": mse, "mae": mae, "pearson": pearson, "r2": r2}
