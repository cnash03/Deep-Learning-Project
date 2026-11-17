import numpy as np
import pytest
from PIL import Image

from src.data.colormaps import RUGD_CLASSES
from src.data.rugd import FRAMES_DIR, LABELS_DIR, RUGD_SPLITS


@pytest.fixture
def fake_rugd(tmp_path):
    """A tiny RUGD-shaped dataset: 4 frames in the first sequence of each split."""
    rng = np.random.default_rng(0)
    colors = np.array([rgb for _, rgb in RUGD_CLASSES], dtype=np.uint8)
    for split in ("train", "val", "test"):
        seq = RUGD_SPLITS[split][0]
        (tmp_path / FRAMES_DIR / seq).mkdir(parents=True)
        (tmp_path / LABELS_DIR / seq).mkdir(parents=True)
        for i in range(4):
            name = f"{seq}_{i:05d}.png"
            Image.fromarray(rng.integers(0, 255, (55, 69, 3), dtype=np.uint8)).save(tmp_path / FRAMES_DIR / seq / name)
            idx = rng.integers(0, len(colors), (55, 69))
            Image.fromarray(colors[idx]).save(tmp_path / LABELS_DIR / seq / name)
    return tmp_path
