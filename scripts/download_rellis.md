# Downloading RELLIS-3D (stretch goal)

RELLIS-3D is hosted on Google Drive, with download links in the official repository:
https://github.com/unmannedlab/RELLIS-3D (see the "Download" section). If Drive
quota errors appear, the Academic Torrents mirror is listed there too.

We only need the **camera images** (`Rellis_3D_pylon_camera_node.zip`), the
**image annotations** (`Rellis_3D_pylon_camera_node_label_id.zip`), and the
**image split lists** (`train.lst`, `val.lst`, `test.lst`). Skip the LiDAR scans and ROS bags.

Expected layout (what `src/data/rellis.py` reads):

```
data/rellis/
  00000/pylon_camera_node/*.jpg
  00000/pylon_camera_node_label_id/*.png
  ...
  split/train.lst  split/val.lst  split/test.lst
```

Each `.lst` line is `<image_rel_path> <label_rel_path>` relative to `data/rellis/`.
Check that a few lines resolve to real files after extracting.

License: CC BY-NC-SA 3.0 (non-commercial). Cite Jiang et al., arXiv 2011.12954.
