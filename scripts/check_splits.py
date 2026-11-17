"""Verify the RUGD sequence split: frame counts per split and no sequence in two splits.

    python -m scripts.check_splits --root data/rugd
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.rugd import FRAMES_DIR, LABELS_DIR, RUGD_SPLITS, list_rugd_pairs

EXPECTED = {"train": 4779, "val": 733, "test": 1924}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/rugd")
    root = Path(p.parse_args().root)

    seqs = [s for v in RUGD_SPLITS.values() for s in v]
    dupes = {s for s in seqs if seqs.count(s) > 1}
    on_disk = {d.name for d in (root / FRAMES_DIR).iterdir() if d.is_dir()}
    unassigned = on_disk - set(seqs)
    missing = set(seqs) - on_disk
    print(f"Sequences on disk: {len(on_disk)}  in split lists: {len(seqs)}")

    ok = not dupes and not unassigned and not missing
    for split, names in RUGD_SPLITS.items():
        n = len(list_rugd_pairs(root, split))
        status = "OK" if n == EXPECTED[split] else f"MISMATCH (expected {EXPECTED[split]})"
        ok &= n == EXPECTED[split]
        print(f"{split:5s}: {n:5d} frames from {len(names)} sequences  {status}")
    if dupes:
        print("Sequences in more than one split:", sorted(dupes))
    if unassigned:
        print("Sequences on disk but not in any split:", sorted(unassigned))
    if missing:
        print("Sequences in split lists but missing on disk:", sorted(missing))
    n_frames = sum(1 for _ in (root / FRAMES_DIR).rglob("*.png"))
    n_labels = sum(1 for _ in (root / LABELS_DIR).rglob("*.png"))
    print(f"Frames: {n_frames}  label PNGs: {n_labels}")
    print("ALL CHECKS PASSED" if ok else "CHECKS FAILED -- fix RUGD_SPLITS in src/data/rugd.py")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
