"""Class definitions and color maps for RUGD and RELLIS-3D.

RUGD ships its labels as color-coded RGB PNGs, which we decode to integer
class indices. RELLIS-3D ships integer-ID PNGs, whose IDs are not contiguous
(0..34 with gaps), so we remap them to 0..N-1.

The RUGD colors below match RUGD_annotation-colormap.txt from the official
download. If that file is present, `load_rugd_colormap` reads it instead, so
the dataset's own file always wins.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

IGNORE_INDEX = 255

# (name, (R, G, B)); list index == class index. Index 0 is "void".
RUGD_CLASSES: list[tuple[str, tuple[int, int, int]]] = [
    ("void", (0, 0, 0)),
    ("dirt", (108, 64, 20)),
    ("sand", (255, 229, 204)),
    ("grass", (0, 102, 0)),
    ("tree", (0, 255, 0)),
    ("pole", (0, 153, 153)),
    ("water", (0, 128, 255)),
    ("sky", (0, 0, 255)),
    ("vehicle", (255, 255, 0)),
    ("container/generic-object", (255, 0, 127)),
    ("asphalt", (64, 64, 64)),
    ("gravel", (255, 128, 0)),
    ("building", (255, 0, 0)),
    ("mulch", (153, 76, 0)),
    ("rock-bed", (102, 102, 0)),
    ("log", (102, 0, 0)),
    ("bicycle", (0, 255, 128)),
    ("person", (204, 153, 255)),
    ("fence", (102, 0, 204)),
    ("bush", (255, 153, 204)),
    ("sign", (0, 102, 102)),
    ("rock", (153, 204, 255)),
    ("bridge", (102, 255, 255)),
    ("concrete", (101, 101, 11)),
    ("picnic-table", (114, 85, 47)),
]

# RELLIS-3D raw label ID -> name (from the RELLIS-3D ontology).
RELLIS_RAW_IDS: dict[int, str] = {
    0: "void",
    1: "dirt",
    3: "grass",
    4: "tree",
    5: "pole",
    6: "water",
    7: "sky",
    8: "vehicle",
    9: "object",
    10: "asphalt",
    12: "building",
    15: "log",
    17: "person",
    18: "fence",
    19: "bush",
    23: "concrete",
    27: "barrier",
    31: "puddle",
    33: "mud",
    34: "rubble",
}
RELLIS_CLASSES: list[str] = [RELLIS_RAW_IDS[k] for k in sorted(RELLIS_RAW_IDS)]


def load_rugd_colormap(path: str | Path | None = None) -> list[tuple[str, tuple[int, int, int]]]:
    """Read RUGD_annotation-colormap.txt ("idx name R G B" per line) if given, else use the built-in table."""
    if path is None or not Path(path).exists():
        return RUGD_CLASSES
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        idx, name, r, g, b = int(parts[0]), parts[1], *map(int, parts[2:5])
        rows.append((idx, name, (r, g, b)))
    rows.sort()
    if [r[0] for r in rows] != list(range(len(rows))):
        raise ValueError(f"Non-contiguous class indices in {path}")
    return [(name, rgb) for _, name, rgb in rows]


def build_color_lut(classes: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    """24-bit RGB -> class index lookup table; unknown colors map to IGNORE_INDEX."""
    lut = np.full(256**3, IGNORE_INDEX, dtype=np.uint8)
    for idx, (_, (r, g, b)) in enumerate(classes):
        lut[(r << 16) | (g << 8) | b] = idx
    return lut


def rgb_to_index(mask_rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Decode an HxWx3 uint8 color mask to an HxW uint8 class-index mask."""
    m = mask_rgb.astype(np.uint32)
    return lut[(m[..., 0] << 16) | (m[..., 1] << 8) | m[..., 2]]


def build_id_lut(raw_ids: dict[int, str]) -> np.ndarray:
    """Raw label ID -> contiguous index lookup table; unknown IDs map to IGNORE_INDEX."""
    lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for new_idx, raw in enumerate(sorted(raw_ids)):
        lut[raw] = new_idx
    return lut


def colorize(index_mask: np.ndarray, classes: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    """Class-index mask -> RGB image for visualization (ignore pixels shown white)."""
    palette = np.full((256, 3), 255, dtype=np.uint8)
    for idx, (_, rgb) in enumerate(classes):
        palette[idx] = rgb
    return palette[index_mask]
