"""Shared 6-group traversability class map for RUGD and RELLIS-3D.

Used for (a) traversability-group IoU on RUGD and (b) the cross-dataset test:
a RUGD-trained model's predictions are mapped to groups and scored against
RELLIS-3D labels mapped the same way.

Group assignments are a project decision. Revisit them in Week 1 and record
the final choice in the paper's Methods section.
"""
from __future__ import annotations

import numpy as np

from .colormaps import IGNORE_INDEX, RELLIS_CLASSES, RUGD_CLASSES

GROUPS = ["smooth_ground", "rough_ground", "vegetation", "water", "obstacle", "sky"]
_G = {name: i for i, name in enumerate(GROUPS)}

RUGD_TO_GROUP: dict[str, str | None] = {
    "void": None,
    "dirt": "smooth_ground",
    "sand": "smooth_ground",
    "asphalt": "smooth_ground",
    "concrete": "smooth_ground",
    "mulch": "smooth_ground",
    "gravel": "rough_ground",
    "rock-bed": "rough_ground",
    "grass": "vegetation",
    "bush": "vegetation",
    "water": "water",
    "sky": "sky",
    "tree": "obstacle",
    "pole": "obstacle",
    "vehicle": "obstacle",
    "container/generic-object": "obstacle",
    "building": "obstacle",
    "log": "obstacle",
    "bicycle": "obstacle",
    "person": "obstacle",
    "fence": "obstacle",
    "sign": "obstacle",
    "rock": "obstacle",
    "bridge": "obstacle",
    "picnic-table": "obstacle",
}

RELLIS_TO_GROUP: dict[str, str | None] = {
    "void": None,
    "dirt": "smooth_ground",
    "asphalt": "smooth_ground",
    "concrete": "smooth_ground",
    "mud": "rough_ground",
    "rubble": "rough_ground",
    "grass": "vegetation",
    "bush": "vegetation",
    "water": "water",
    "puddle": "water",
    "sky": "sky",
    "tree": "obstacle",
    "pole": "obstacle",
    "vehicle": "obstacle",
    "object": "obstacle",
    "building": "obstacle",
    "log": "obstacle",
    "person": "obstacle",
    "fence": "obstacle",
    "barrier": "obstacle",
}


def _lut(class_names: list[str], mapping: dict[str, str | None]) -> np.ndarray:
    lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for idx, name in enumerate(class_names):
        group = mapping[name]
        if group is not None:
            lut[idx] = _G[group]
    return lut


RUGD_GROUP_LUT = _lut([n for n, _ in RUGD_CLASSES], RUGD_TO_GROUP)
RELLIS_GROUP_LUT = _lut(RELLIS_CLASSES, RELLIS_TO_GROUP)


def to_groups(index_mask, lut: np.ndarray):
    """Map a class-index mask (numpy array or torch tensor) to group indices."""
    try:
        import torch

        if isinstance(index_mask, torch.Tensor):
            return torch.as_tensor(lut, device=index_mask.device).long()[index_mask.long()]
    except ImportError:
        pass
    return lut[index_mask]
