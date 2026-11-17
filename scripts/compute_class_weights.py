"""Count class pixel frequencies on the TRAINING split and write class weights.

    python -m scripts.compute_class_weights --root data/rugd

Outputs:
    results/class_weights_rugd.json   (used by configs/e4_weighted_ce.yaml)
    results/class_freq_rugd.csv       (for the EDA class-frequency figure)

Weights use median-frequency balancing: w_c = median(f) / f_c, where f_c is the
pixel frequency of class c. Ignored/absent classes get weight 0. Weights are
clipped to --max-weight so very rare classes don't destabilize training.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.data.colormaps import IGNORE_INDEX
from src.data.rugd import RUGDDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/rugd")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--max-weight", type=float, default=50.0)
    args = p.parse_args()

    ds = RUGDDataset(args.root, "train", transform=None)
    counts = np.zeros(ds.num_classes, dtype=np.int64)
    ignored = 0
    for i in range(len(ds)):
        m = ds.load_mask(i)
        ignored += int((m == IGNORE_INDEX).sum())
        counts += np.bincount(m[m != IGNORE_INDEX].ravel(), minlength=ds.num_classes)[: ds.num_classes]
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(ds)}")

    total = counts.sum()
    freq = counts / max(total, 1)
    present = counts > 0
    median = np.median(freq[present])
    weights = np.where(present, median / np.maximum(freq, 1e-12), 0.0)
    weights = np.minimum(weights, args.max_weight)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = [n for n, _ in ds.classes]
    (out / "class_weights_rugd.json").write_text(json.dumps(
        {"method": "median_frequency", "max_weight": args.max_weight, "classes": names,
         "weights": [round(float(w), 6) for w in weights]}, indent=2))
    with (out / "class_freq_rugd.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "pixels", "frequency", "weight"])
        for n, c, fr, wt in sorted(zip(names, counts, freq, weights), key=lambda r: -r[1]):
            w.writerow([n, int(c), f"{fr:.6f}", f"{wt:.4f}"])
    print(f"Ignored (void/unknown) pixels: {ignored / (ignored + total):.2%}")
    print(f"Wrote {out / 'class_weights_rugd.json'} and {out / 'class_freq_rugd.csv'}")


if __name__ == "__main__":
    main()
