"""Model factory over segmentation_models_pytorch."""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn

ARCHS = {
    "unet": smp.Unet,
    "deeplabv3plus": smp.DeepLabV3Plus,
}


def build_model(cfg: dict, num_classes: int) -> nn.Module:
    m = cfg["model"]
    arch = m["arch"].lower()
    if arch not in ARCHS:
        raise ValueError(f"Unknown arch {arch}; choose from {list(ARCHS)}")
    return ARCHS[arch](
        encoder_name=m["encoder"],
        encoder_weights="imagenet" if m.get("pretrained", True) else None,
        in_channels=3,
        classes=num_classes,
    )
