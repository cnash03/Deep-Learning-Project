"""Confusion-matrix-based segmentation metrics."""
from __future__ import annotations

import torch

from .data.colormaps import IGNORE_INDEX


def predict(logits: torch.Tensor, exclude: list[int] | None = None) -> torch.Tensor:
    """Argmax over classes, never predicting classes in `exclude` (e.g. the ignored void class)."""
    if exclude:
        # Additive mask instead of in-place indexed assignment: the latter hung the Apple MPS backend.
        bias = torch.zeros(logits.shape[1], device=logits.device, dtype=logits.dtype)
        bias[exclude] = torch.finfo(logits.dtype).min
        logits = logits + bias.view(1, -1, 1, 1)
    return logits.argmax(1)


class ConfusionMatrix:
    """Accumulates an NxN confusion matrix (rows = ground truth, cols = prediction)."""

    def __init__(self, num_classes: int, device="cpu"):
        # Always accumulated on CPU: torch.bincount on Apple MPS hung validation. `device` is kept for API compat.
        self.n = num_classes
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.long)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred, target = pred.flatten().cpu(), target.flatten().cpu()
        keep = (target != IGNORE_INDEX) & (target < self.n) & (pred >= 0) & (pred < self.n)
        idx = target[keep] * self.n + pred[keep]
        self.mat += torch.bincount(idx, minlength=self.n**2).reshape(self.n, self.n)

    def iou(self) -> torch.Tensor:
        """Per-class IoU; NaN for classes absent from both GT and prediction."""
        m = self.mat.cpu().double()
        tp = m.diag()
        denom = m.sum(0) + m.sum(1) - tp
        return torch.where(denom > 0, tp / denom.clamp(min=1), torch.full_like(tp, float("nan")))

    def miou(self, exclude: list[int] | None = None) -> float:
        iou = self.iou()
        if exclude:
            mask = torch.ones(self.n, dtype=torch.bool)
            mask[exclude] = False
            iou = iou[mask]
        return float(torch.nanmean(iou))

    def miou_gt_present(self, exclude: list[int] | None = None) -> float:
        """mIoU over classes that occur in the ground truth only (false positives on absent classes ignored)."""
        iou = self.iou()
        present = self.mat.sum(1) > 0
        if exclude:
            present[exclude] = False
        return float(torch.nanmean(iou[present])) if present.any() else float("nan")

    def pixel_acc(self) -> float:
        total = self.mat.sum()
        return float(self.mat.diag().sum().cpu() / total.cpu()) if total > 0 else float("nan")

    def normalized(self) -> torch.Tensor:
        """Row-normalized matrix (recall per GT class), for the confusion-matrix figure."""
        m = self.mat.cpu().double()
        return m / m.sum(1, keepdim=True).clamp(min=1)
