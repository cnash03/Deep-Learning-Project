"""RELLIS-3D camera-image dataset (stretch goal: cross-dataset evaluation).

Expected layout under `root`:

    root/<seq>/pylon_camera_node/<frame>.jpg
    root/<seq>/pylon_camera_node_label_id/<frame>.png
    root/split/{train,val,test}.lst   # "image_rel_path label_rel_path" per line

The .lst split files come from the RELLIS-3D repository. Labels are raw
ontology IDs and are remapped here to contiguous indices.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .colormaps import IGNORE_INDEX, RELLIS_CLASSES, RELLIS_RAW_IDS, build_id_lut


class RELLISDataset(Dataset):
    def __init__(self, root, split: str, transform=None, subset: int | None = None):
        self.root = Path(root)
        lst = self.root / "split" / f"{split}.lst"
        if not lst.exists():
            raise FileNotFoundError(f"Missing split file {lst}")
        self.pairs = []
        for line in lst.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                self.pairs.append((self.root / parts[0], self.root / parts[1]))
        if subset:
            idx = np.linspace(0, len(self.pairs) - 1, subset).astype(int)
            self.pairs = [self.pairs[i] for i in idx]
        self.classes = RELLIS_CLASSES
        self.lut = build_id_lut(RELLIS_RAW_IDS)
        self.transform = transform

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __len__(self) -> int:
        return len(self.pairs)

    def load_mask(self, i: int) -> np.ndarray:
        mask = self.lut[np.array(Image.open(self.pairs[i][1]))]
        mask[mask == 0] = IGNORE_INDEX  # void
        return mask

    def __getitem__(self, i: int):
        image = np.array(Image.open(self.pairs[i][0]).convert("RGB"))
        mask = self.load_mask(i)
        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        return image, torch.as_tensor(np.asarray(mask), dtype=torch.long)
