"""RUGD dataset.

Expected layout under `root` (as extracted from the official zips):

    root/RUGD_frames-with-annotations/<sequence>/<sequence>_00001.png
    root/RUGD_annotations/<sequence>/<sequence>_00001.png
    root/RUGD_annotations/RUGD_annotation-colormap.txt   (optional, preferred)

Splits are by video sequence, so near-identical consecutive frames never land
in both train and test. Run `python -m scripts.check_splits` to confirm the
sequence lists below give 4,779 / 733 / 1,924 frames.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .colormaps import IGNORE_INDEX, build_color_lut, load_rugd_colormap, rgb_to_index

RUGD_SPLITS: dict[str, list[str]] = {
    "train": [
        "park-2", "trail", "trail-3", "trail-4", "trail-6", "trail-9",
        "trail-10", "trail-11", "trail-12", "trail-14", "trail-15", "village",
    ],
    "val": ["park-8", "trail-5"],
    "test": ["creek", "park-1", "trail-7", "trail-13"],
}

FRAMES_DIR = "RUGD_frames-with-annotations"
LABELS_DIR = "RUGD_annotations"


def list_rugd_pairs(root: str | Path, split: str) -> list[tuple[Path, Path]]:
    root = Path(root)
    pairs = []
    for seq in RUGD_SPLITS[split]:
        for img in sorted((root / FRAMES_DIR / seq).glob("*.png")):
            lbl = root / LABELS_DIR / seq / img.name
            if lbl.exists():
                pairs.append((img, lbl))
    return pairs


class RUGDDataset(Dataset):
    def __init__(self, root, split: str, transform=None, subset: int | None = None, ignore_void: bool = True):
        self.root = Path(root)
        self.ignore_void = ignore_void
        self.pairs = list_rugd_pairs(self.root, split)
        if not self.pairs:
            raise FileNotFoundError(f"No RUGD {split} frames found under {self.root}")
        if subset:
            # Evenly spaced subset, deterministic across runs.
            idx = np.linspace(0, len(self.pairs) - 1, subset).astype(int)
            self.pairs = [self.pairs[i] for i in idx]
        self.classes = load_rugd_colormap(self.root / LABELS_DIR / "RUGD_annotation-colormap.txt")
        self.lut = build_color_lut(self.classes)
        self.transform = transform

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __len__(self) -> int:
        return len(self.pairs)

    def load_mask(self, i: int) -> np.ndarray:
        mask = rgb_to_index(np.array(Image.open(self.pairs[i][1]).convert("RGB")), self.lut)
        if self.ignore_void:
            mask[mask == 0] = IGNORE_INDEX
        return mask

    def __getitem__(self, i: int):
        image = np.array(Image.open(self.pairs[i][0]).convert("RGB"))
        mask = self.load_mask(i)
        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        return image, torch.as_tensor(np.asarray(mask), dtype=torch.long)
