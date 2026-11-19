"""Qualitative figures: image / ground truth / prediction / error map.

    # best, median and worst val frames by per-image mIoU (needs per_image.csv from
    # `python -m src.evaluate --run runs/X --split val --per-image`)
    DEVICE=cpu python -m scripts.gallery --run runs/e3_dlv3p_r50_ce --split val

    # side by side of two runs on rare-class frames (water / log / rock)
    DEVICE=cpu python -m scripts.gallery --run runs/e3_dlv3p_r50_ce --compare runs/e7_ce_dice --split val

Rare-class frames are picked from ground truth only (largest rare-class pixel share, at
most one frame per 30 consecutive frames of a sequence), never from model scores, so the
comparison is not cherry-picked. Inference runs on CPU by default (a handful of frames).
Error maps: wrong pixels vermillion, correct white, void/ignore gray. Writes
results/figures/gallery_<run>_<split>.{pdf,png} and compare_<run>_vs_<run2>_<split>.{pdf,png}.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src import analysis as A
from src.data import build_dataset
from src.data.colormaps import IGNORE_INDEX, load_rugd_colormap
from src.metrics import predict


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--confirm-test", action="store_true", help="required for --split test")
    p.add_argument("--compare", default=None, help="second run for a side-by-side on rare-class frames")
    p.add_argument("--k", type=int, default=1, help="frames per category (best / median / worst)")
    p.add_argument("--n-compare", type=int, default=4, help="rare-class frames in the comparison figure")
    p.add_argument("--rare-classes", nargs="+", default=["water", "log", "rock"])
    p.add_argument("--frames", nargs="+", default=None, help="explicit frame file names (overrides selection)")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default=None, help="figure directory (default: <results>/figures)")
    p.add_argument("--data-root", default=None)
    p.add_argument("--device", default=os.environ.get("DEVICE", "cpu"))
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args(argv)


class Frames:
    """Dataset + model access for a handful of frames."""

    def __init__(self, run: Path, split: str, data_root: str | None, device: str):
        self.model, self.cfg, self.names = A.load_run_model(run, device)
        self.device = device
        cfg = {k: v for k, v in self.cfg.items() if k != "_meta"}
        cfg["data"] = {**cfg["data"], "subset": None, **({"root": data_root} if data_root else {})}
        self.ds = build_dataset(cfg, split, train=False)
        self.size = cfg["data"]["img_size"]
        self.exclude = [0] if cfg["data"].get("ignore_void", True) else []
        self.miou_exclude = self.exclude + [self.names.index(n) for n in cfg.get("eval", {}).get("exclude_from_miou", [])
                                            if n in self.names]
        self.files = [p[0].name for p in self.ds.pairs]

    def index_of(self, file: str) -> int:
        return self.files.index(file)

    def image(self, i: int) -> tuple[np.ndarray, float]:
        """Display image resized like the eval transform, plus the original H/W aspect."""
        im = np.array(Image.open(self.ds.pairs[i][0]).convert("RGB"))
        return cv2.resize(im, (self.size, self.size), interpolation=cv2.INTER_LINEAR), im.shape[0] / im.shape[1]

    def gt(self, i: int) -> np.ndarray:
        return self.ds[i][1].numpy()

    @torch.no_grad()
    def predict(self, i: int, model=None) -> np.ndarray:
        x = self.ds[i][0][None].to(self.device)
        return predict((model or self.model)(x), self.exclude)[0].cpu().numpy()

    def image_miou(self, pred: np.ndarray, gt: np.ndarray) -> float:
        m = A.batch_confusion(torch.as_tensor(pred)[None], torch.as_tensor(gt)[None], len(self.names))[0].numpy()
        return A.miou_from_confusion(m, self.miou_exclude, gt_present_only=True)


def class_colors(frames: Frames) -> list[tuple[int, int, int]]:
    root = Path(frames.ds.root)
    classes = load_rugd_colormap(root / "RUGD_annotations" / "RUGD_annotation-colormap.txt")
    return [rgb for _, rgb in classes]


def select_by_score(per_image: list[dict], k: int) -> list[tuple[str, dict]]:
    rows = sorted([r for r in per_image if r.get("miou")], key=lambda r: float(r["miou"]), reverse=True)
    if not rows:
        return []
    k = max(1, min(k, len(rows) // 3 or 1))
    mid = len(rows) // 2 - k // 2
    out = [("best", r) for r in rows[:k]] + [("median", r) for r in rows[mid:mid + k]] + [("worst", r) for r in rows[-k:]]
    return out


def select_rare(frames: Frames, rare: list[str], n: int, min_gap: int = 30) -> list[int]:
    """Frames with the largest GT share of rare classes; at most one per `min_gap` frames per sequence."""
    ids = [frames.names.index(c) for c in rare if c in frames.names]
    scores = []
    for i in range(len(frames.ds)):
        m = frames.ds.load_mask(i)
        valid = (m != IGNORE_INDEX).sum()
        if valid:
            scores.append((np.isin(m, ids).sum() / valid, i))
    chosen: list[int] = []
    for s, i in sorted(scores, reverse=True):
        if s <= 0 or len(chosen) >= n:
            break
        seq = A.sequence_of(frames.ds.pairs[i][0])
        if all(A.sequence_of(frames.ds.pairs[j][0]) != seq or abs(i - j) >= min_gap for j in chosen):
            chosen.append(i)
    return chosen


def draw(rows: list[dict], col_titles: list[str], pal: np.ndarray, names: list[str], path_stem: Path) -> None:
    """rows: [{"label": str, "aspect": float, "panels": [rgb arrays], "titles": [str|None]}]."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ncol = len(col_titles)
    aspect = rows[0]["aspect"]
    # Fixed margins in inches so the legend never overlaps the panels, whatever the row count.
    left, gap, top = 0.45, 0.05, 0.25
    pw = (A.PAGE_W - left - gap * (ncol - 1)) / ncol
    ph = pw * aspect
    n_leg = len({c for row in rows for c in row.get("classes", set())} - {0, IGNORE_INDEX}) + 2
    leg_h = 0.17 * -(-n_leg // 7) + 0.1
    height = top + len(rows) * ph + (len(rows) - 1) * gap + leg_h
    fig, axes = plt.subplots(len(rows), ncol, figsize=(A.PAGE_W, height), squeeze=False)
    fig.subplots_adjust(left=left / A.PAGE_W, right=1, top=1 - top / height, bottom=leg_h / height,
                        wspace=gap / pw, hspace=gap / ph)
    used: set[int] = set()
    for r, row in enumerate(rows):
        used |= row.get("classes", set())
        for c in range(ncol):
            ax = axes[r, c]
            ax.imshow(row["panels"][c], aspect=row["aspect"] * row["panels"][c].shape[1] / row["panels"][c].shape[0],
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.spines[:].set_visible(False)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=9)
            if row["titles"][c]:  # metric in the panel corner, so it never collides with the next row
                ax.text(0.02, 0.97, row["titles"][c], transform=ax.transAxes, ha="left", va="top", fontsize=8,
                        color=A.INK, bbox={"boxstyle": "square,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.85})
        axes[r, 0].set_ylabel(row["label"], fontsize=8)
    used -= {0, IGNORE_INDEX}
    handles = [Patch(facecolor=np.array(pal[i]) / 255, edgecolor="#999999", lw=0.3, label=A.short_class(names[i]))
               for i in sorted(used)]
    handles += [Patch(facecolor=np.array(A.ERROR_RGB) / 255, label="error"),
                Patch(facecolor=np.array(A.IGNORE_RGB) / 255, label="void / ignore")]
    fig.legend(handles=handles, loc="lower center", ncol=min(7, len(handles)), fontsize=8, frameon=False,
               handlelength=1.0, columnspacing=0.8, bbox_to_anchor=(0.5, 0.0))
    A.savefig(fig, path_stem.parent, path_stem.name)  # bbox "tight" trims outer whitespace only


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.split == "test" and not args.confirm_test:
        raise SystemExit("Refusing to touch the test split without --confirm-test.")
    torch.set_num_threads(args.threads)
    A.set_style()
    run = Path(args.run)
    out = Path(args.out) if args.out else Path(args.results) / "figures"
    fr = Frames(run, args.split, args.data_root, args.device)
    pal = A.palette_array(class_colors(fr))
    labels = {}

    # ---- best / median / worst gallery
    if args.frames:
        picks = [("frame", {"file": f}) for f in args.frames]
    else:
        per_image = A.read_csv_rows(Path(args.results) / run.name / args.split / "per_image.csv")
        if not per_image and not args.compare:
            raise SystemExit(f"{args.results}/{run.name}/{args.split}/per_image.csv not found. Run:\n"
                             f"  python -m src.evaluate --run {run} --split {args.split} --per-image")
        picks = select_by_score(per_image, args.k)
    rows = []
    for cat, r in picks:
        i = fr.index_of(r["file"])
        img, aspect = fr.image(i)
        gt, pred = fr.gt(i), fr.predict(i)
        miou = fr.image_miou(pred, gt)
        labels[i] = f"{cat}\n{Path(r['file']).stem}"
        rows.append({"label": labels[i], "aspect": aspect, "classes": set(np.unique(gt)) | set(np.unique(pred)),
                     "panels": [img, pal[gt], pal[pred], A.error_map(pred, gt)],
                     "titles": [None, None, f"mIoU {miou * 100:.1f}" if miou == miou else None,
                                f"pixel acc {np.mean(pred[gt != IGNORE_INDEX] == gt[gt != IGNORE_INDEX]) * 100:.1f}"
                                if (gt != IGNORE_INDEX).any() else None]})
    if rows:
        draw(rows, ["image", "ground truth", "prediction", "error map"], pal, fr.names,
             out / f"gallery_{run.name}_{args.split}")
        print(f"[gallery] wrote {out}/gallery_{run.name}_{args.split}.pdf/png ({len(rows)} frames)")

    # ---- two-run comparison on rare-class frames
    if args.compare:
        run2 = Path(args.compare)
        model2, _, names2 = A.load_run_model(run2, args.device)
        if names2 != fr.names:
            raise SystemExit("runs use different class lists; cannot compare")
        idx = [fr.index_of(f) for f in args.frames] if args.frames else select_rare(fr, args.rare_classes, args.n_compare)
        if not idx:
            print(f"[gallery] no frames containing {args.rare_classes} in {args.split}; skipping comparison")
            return
        rows = []
        for i in idx:
            img, aspect = fr.image(i)
            gt = fr.gt(i)
            p1, p2 = fr.predict(i), fr.predict(i, model2)
            m1, m2 = fr.image_miou(p1, gt), fr.image_miou(p2, gt)
            rare_here = [c for c in args.rare_classes if c in fr.names and (gt == fr.names.index(c)).any()]
            rows.append({"label": f"{Path(fr.ds.pairs[i][0]).stem}\n({', '.join(rare_here)})", "aspect": aspect,
                         "classes": set(np.unique(gt)) | set(np.unique(p1)) | set(np.unique(p2)),
                         "panels": [img, pal[gt], pal[p1], pal[p2]],
                         "titles": [None, None, f"mIoU {m1 * 100:.1f}", f"mIoU {m2 * 100:.1f}"]})
        draw(rows, ["image", "ground truth", run.name, run2.name], pal, fr.names,
             out / f"compare_{run.name}_vs_{run2.name}_{args.split}")
        print(f"[gallery] wrote {out}/compare_{run.name}_vs_{run2.name}_{args.split}.pdf/png ({len(rows)} frames)")


if __name__ == "__main__":
    main()
