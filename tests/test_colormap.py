import numpy as np

from src.data.class_map import GROUPS, RELLIS_GROUP_LUT, RELLIS_TO_GROUP, RUGD_GROUP_LUT, RUGD_TO_GROUP
from src.data.colormaps import (IGNORE_INDEX, RELLIS_CLASSES, RELLIS_RAW_IDS, RUGD_CLASSES, build_color_lut,
                                build_id_lut, colorize, load_rugd_colormap, rgb_to_index)


def test_rugd_colors_unique():
    rgbs = [rgb for _, rgb in RUGD_CLASSES]
    assert len(set(rgbs)) == len(rgbs) == 25


def test_rgb_round_trip():
    lut = build_color_lut(RUGD_CLASSES)
    idx = np.random.default_rng(0).integers(0, len(RUGD_CLASSES), (40, 50)).astype(np.uint8)
    assert np.array_equal(rgb_to_index(colorize(idx, RUGD_CLASSES), lut), idx)


def test_unknown_color_is_ignored():
    lut = build_color_lut(RUGD_CLASSES)
    px = np.array([[[1, 2, 3]]], dtype=np.uint8)
    assert rgb_to_index(px, lut)[0, 0] == IGNORE_INDEX


def test_colormap_file_parsing(tmp_path):
    f = tmp_path / "cmap.txt"
    f.write_text("\n".join(f"{i} {n} {r} {g} {b}" for i, (n, (r, g, b)) in enumerate(RUGD_CLASSES)))
    assert load_rugd_colormap(f) == RUGD_CLASSES


def test_rellis_id_lut_contiguous():
    lut = build_id_lut(RELLIS_RAW_IDS)
    mapped = sorted(lut[k] for k in RELLIS_RAW_IDS)
    assert mapped == list(range(len(RELLIS_CLASSES))) and len(RELLIS_CLASSES) == 20
    assert lut[2] == IGNORE_INDEX  # an unused raw ID


def test_group_maps_cover_every_class():
    assert set(RUGD_TO_GROUP) == {n for n, _ in RUGD_CLASSES}
    assert set(RELLIS_TO_GROUP) == set(RELLIS_CLASSES)
    for lut in (RUGD_GROUP_LUT, RELLIS_GROUP_LUT):
        assert lut[0] == IGNORE_INDEX  # void
        assert set(lut[lut != IGNORE_INDEX]) == set(range(len(GROUPS)))
