"""Albumentations pipelines.

Masks are always resized with nearest-neighbor interpolation (albumentations
does this for `mask` targets), so labels are never blended. No vertical flips:
sky is always on top.
"""
from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(cfg: dict, train: bool) -> A.Compose:
    size = cfg["data"]["img_size"]
    aug = cfg.get("augment", {})
    ops: list = []
    if train and aug.get("enabled", True):
        if aug.get("scale_crop", True):
            ops.append(
                A.RandomResizedCrop(
                    size=(size, size),
                    scale=tuple(aug.get("scale", (0.5, 1.0))),
                    ratio=(0.75, 1.33),
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                )
            )
        else:
            ops.append(A.Resize(size, size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST))
        if aug.get("hflip", True):
            ops.append(A.HorizontalFlip(p=0.5))
        if aug.get("color_jitter", True):
            ops.append(A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7))
            ops.append(A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=0.3))
    else:
        # eval_size [H, W] lets cross-dataset eval keep RUGD's aspect squeeze (e.g. RELLIS 1920x1200 -> 384x480).
        h, w = cfg["data"].get("eval_size") or (size, size)
        ops.append(A.Resize(h, w, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST))
    ops += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(ops)
