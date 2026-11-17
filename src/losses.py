"""Loss functions: CE, weighted CE, Dice, Focal, CE + Dice. All respect IGNORE_INDEX."""
from __future__ import annotations

import json
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from .data.colormaps import IGNORE_INDEX


class CEDice(nn.Module):
    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
        self.dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=IGNORE_INDEX)
        self.wc, self.wd = ce_weight, dice_weight

    def forward(self, logits, target):
        return self.wc * self.ce(logits, target) + self.wd * self.dice(logits, target)


def load_class_weights(path: str | Path, num_classes: int) -> torch.Tensor:
    """Read weights written by scripts/compute_class_weights.py."""
    w = json.loads(Path(path).read_text())["weights"]
    if len(w) != num_classes:
        raise ValueError(f"{path} has {len(w)} weights, expected {num_classes}")
    return torch.tensor(w, dtype=torch.float32)


def build_loss(cfg: dict, num_classes: int, device) -> nn.Module:
    lc = cfg["loss"]
    name = lc["name"].lower()
    weights = None
    if lc.get("class_weights"):
        weights = load_class_weights(lc["class_weights"], num_classes).to(device)

    if name == "ce":
        return nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    if name == "weighted_ce":
        if weights is None:
            raise ValueError("weighted_ce requires loss.class_weights")
        return nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE_INDEX)
    if name == "dice":
        return smp.losses.DiceLoss(mode="multiclass", ignore_index=IGNORE_INDEX)
    if name == "focal":
        return smp.losses.FocalLoss(mode="multiclass", gamma=lc.get("gamma", 2.0), ignore_index=IGNORE_INDEX)
    if name == "ce_dice":
        return CEDice(lc.get("ce_weight", 1.0), lc.get("dice_weight", 1.0), weights)
    raise ValueError(f"Unknown loss {name}")
