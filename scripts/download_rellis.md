# Downloading RELLIS-3D (stretch goal)

RELLIS-3D is hosted on Google Drive, linked from the official repository README
(https://github.com/unmannedlab/RELLIS-3D, "Image Download"). A Baidu Pan mirror
is also listed there. If Drive returns a quota error, wait 24 h (per the README).

## Quick path (test split only, ~1.4 GB on disk)

```
.venv/bin/python scripts/prepare_rellis.py --root data/rellis --splits test
```

The full image archive is 11.6 GB, but Drive serves it with HTTP range support,
so the script reads the zip's central directory remotely and fetches only the
frames named in `split/test.lst` (1,672 JPGs, 1.27 GB). Labels and split lists
are small and downloaded whole; the zips are deleted afterwards (`--keep-zips`
to keep them). It refuses to run if free disk would fall below `--min-free-gb`
(default 10). Add `--splits val` or `train` if you later fine-tune.

## Verified files (sizes from HTTP HEAD, Nov 2026)

| File | Drive ID | Size |
|---|---|---|
| `Rellis_3D_pylon_camera_node.zip` (images, all 13.5k frames incl. unlabeled) | `1F3Leu0H_m6aPVpZITragfreO_SGtL2yV` | 11,610,983,413 B |
| `Rellis_3D_pylon_camera_node_label_id.zip` (ID PNGs, 6,235 frames) | `16URBUQn_VOGvUqfms-0I8HHKMtjPHsu5` | 98,850,316 B |
| `Rellis_3D_image_split.zip` (`train/val/test.lst`) | `1zHmnVaItcYJAWat3Yti1W_5Nfux194WQ` | 44,657 B |
| `Rellis_3D_ontology.zip` (ontology.csv/yaml/png) | `1K8Zf0ju_xI5lnx3NTDLJpVTs59wmGPI6` | 18,235 B |

Direct download URL pattern:
`https://drive.usercontent.google.com/download?id=<ID>&export=download&confirm=t`

Archive members are prefixed `Rellis-3D/` (e.g.
`Rellis-3D/00000/pylon_camera_node/frame000000-1581624652_750.jpg`); the
script strips that prefix. Split sizes: train 3,302 / val 983 / test 1,672.

## Layout (what `src/data/rellis.py` reads)

```
data/rellis/
  00000/pylon_camera_node/*.jpg
  00000/pylon_camera_node_label_id/*.png
  ...
  split/train.lst  split/val.lst  split/test.lst
  ontology.csv
```

The official `.lst` files already use the expected format, one
`<image_rel_path> <label_rel_path>` per line relative to `data/rellis/`, so no
conversion is needed. Images are 1920x1200 JPG; labels are single-channel uint8 PNGs
holding raw ontology IDs.

License: CC BY-NC-SA 3.0 (non-commercial). Cite Jiang et al., arXiv 2011.12954.
