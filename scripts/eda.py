"""Exploratory data analysis for RUGD: class pixel statistics and paper figures.

    DEVICE=cpu python -m scripts.eda --root data/rugd

CPU only, no torch models. Steps:
  1. Count class pixels for every annotated frame (2 worker processes) and cache
     them to results/eda/class_pixels_by_frame.csv (one row per frame) and
     results/eda/class_pixels_by_sequence.csv (long format). Reruns read the
     cache; pass --recount to rebuild it.
  2. Write results/eda/split_summary.{csv,tex} and results/eda/eda_numbers.json
     (the exact numbers quoted in eda_summary.md).
  3. Draw results/figures/{class_frequency,sample_grid,sequence_class_heatmap}.{pdf,png}.

Counting conventions
  * Masks are decoded with RUGDDataset(..., ignore_void=False).load_mask, so class 0
    is "void" and 255 marks colors not in the RUGD colormap ("unmapped").
  * "Labeled pixels" = all pixels minus void minus unmapped. Every percentage of a
    class below is (class pixels) / (labeled pixels of that split or sequence).
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("DEVICE", "cpu")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

from src.data.colormaps import IGNORE_INDEX, colorize
from src.data.rugd import RUGD_SPLITS, RUGDDataset

SPLITS = ["train", "val", "test"]
# Classes whose misclassification matters most for a ground robot: water, positive
# obstacles (log, rock, rock-bed) and deformable / low-traction ground (dirt, sand).
# RUGD has no "mud" class; dirt and sand are the closest deformable-ground labels.
SAFETY = ["water", "log", "rock", "rock-bed", "dirt", "sand"]
SHORT = {"container/generic-object": "container/object"}

NEUTRAL = "#b4b2ab"
HIGHLIGHT = "#eb6834"
INK = "#1f1f1d"
MUTED = "#6b6a65"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": MUTED, "xtick.color": MUTED, "xtick.labelcolor": INK,
    "ytick.color": INK, "axes.labelcolor": INK, "savefig.dpi": 300,
})


def short(name: str) -> str:
    return SHORT.get(name, name)


# --------------------------------------------------------------------------- counting

_DS: dict[str, RUGDDataset] = {}


def _count(args):
    split, i = args
    if split not in _DS:
        _DS[split] = RUGDDataset(_ROOT, split, transform=None, ignore_void=False)
    ds = _DS[split]
    img_path, lbl_path = ds.pairs[i]
    with Image.open(img_path) as im:
        iw, ih = im.size
    mask = ds.load_mask(i)
    counts = np.bincount(mask.ravel(), minlength=256)
    return split, img_path.parent.name, img_path.name, iw, ih, mask.shape[1], mask.shape[0], counts


def _init(root):
    global _ROOT
    _ROOT = root


def count_pixels(root: Path, classes: list[str], workers: int) -> pd.DataFrame:
    jobs = []
    for split in SPLITS:
        n = len(RUGDDataset(root, split, transform=None, ignore_void=False))
        jobs += [(split, i) for i in range(n)]
    rows = []
    with Pool(workers, initializer=_init, initargs=(str(root),)) as pool:
        for k, (split, seq, frame, iw, ih, mw, mh, c) in enumerate(pool.imap(_count, jobs, chunksize=16)):
            row = {"split": split, "sequence": seq, "frame": frame,
                   "img_w": iw, "img_h": ih, "mask_w": mw, "mask_h": mh}
            row.update({name: int(c[j]) for j, name in enumerate(classes)})
            row["unmapped"] = int(c[IGNORE_INDEX])
            rows.append(row)
            if (k + 1) % 500 == 0:
                print(f"  counted {k + 1}/{len(jobs)} frames", flush=True)
    return pd.DataFrame(rows)


def by_sequence_long(frames: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    cols = classes + ["unmapped"]
    g = frames.groupby(["split", "sequence"], sort=False)
    pix = g[cols].sum()
    present = (frames[cols] > 0).groupby([frames["split"], frames["sequence"]], sort=False).sum()
    nfr = g.size()
    out = []
    for (split, seq) in pix.index:
        for c in cols:
            out.append({"split": split, "sequence": seq, "class": c,
                        "pixels": int(pix.loc[(split, seq), c]),
                        "frames_present": int(present.loc[(split, seq), c]),
                        "n_frames": int(nfr.loc[(split, seq)])})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- statistics

def split_table(seq_long: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Rows = split, cols = class pixels (incl. void and unmapped)."""
    t = seq_long.pivot_table(index="split", columns="class", values="pixels", aggfunc="sum")
    return t.reindex(SPLITS)[classes + ["unmapped"]]


def numbers(frames, seq_long, classes) -> dict:
    labeled_classes = [c for c in classes if c != "void"]
    st = split_table(seq_long, classes)
    lab = st[labeled_classes]
    share = lab.div(lab.sum(axis=1), axis=0) * 100
    total = lab.sum(axis=0)
    share_all = total / total.sum() * 100
    out = {"definitions": "share = class pixels / (all pixels - void - unmapped) within the split, in %"}
    out["frames"] = {s: int((frames.split == s).sum()) for s in SPLITS}
    out["sequences"] = {s: RUGD_SPLITS[s] for s in SPLITS}
    out["frames_per_sequence"] = {s: int(n) for s, n in frames.groupby("sequence").size().items()}
    out["image_sizes"] = sorted({f"{w}x{h}" for w, h in zip(frames.img_w, frames.img_h)})
    out["mask_sizes"] = sorted({f"{w}x{h}" for w, h in zip(frames.mask_w, frames.mask_h)})
    out["image_mask_size_mismatch_frames"] = int(((frames.img_w != frames.mask_w) | (frames.img_h != frames.mask_h)).sum())
    allpix = st.sum(axis=1)
    out["total_pixels"] = {s: int(allpix[s]) for s in SPLITS}
    out["labeled_pixels"] = {s: int(lab.sum(axis=1)[s]) for s in SPLITS}
    out["void_pct_of_all_pixels"] = {s: float(st.loc[s, "void"] / allpix[s] * 100) for s in SPLITS}
    out["void_pct_all_splits"] = float(st["void"].sum() / allpix.sum() * 100)
    out["unmapped_pixels"] = {s: int(st.loc[s, "unmapped"]) for s in SPLITS}
    out["frames_with_void"] = {s: int(((frames.split == s) & (frames.void > 0)).sum()) for s in SPLITS}
    out["share_pct"] = {s: {c: float(share.loc[s, c]) for c in labeled_classes} for s in SPLITS}
    out["share_pct_all_splits"] = {c: float(share_all[c]) for c in labeled_classes}
    out["absent"] = {s: [c for c in labeled_classes if lab.loc[s, c] == 0] for s in SPLITS}
    for s in SPLITS:
        present = share.loc[s][lab.loc[s] > 0].sort_values(ascending=False)
        out.setdefault("imbalance", {})[s] = {
            "most": present.index[0], "most_pixels": int(lab.loc[s, present.index[0]]),
            "least": present.index[-1], "least_pixels": int(lab.loc[s, present.index[-1]]),
            "ratio": float(lab.loc[s, present.index[0]] / lab.loc[s, present.index[-1]]),
        }
        top5 = present.index[:5].tolist()
        out.setdefault("top5", {})[s] = {"classes": top5, "share_pct": float(present.iloc[:5].sum())}
        out.setdefault("tree_plus_grass_pct", {})[s] = float(share.loc[s, "tree"] + share.loc[s, "grass"])
        out.setdefault("safety_pct", {})[s] = {c: float(share.loc[s, c]) for c in SAFETY}
        out.setdefault("safety_total_pct", {})[s] = float(share.loc[s, SAFETY].sum())
        out.setdefault("classes_below_1pct", {})[s] = int(((share.loc[s] < 1) & (lab.loc[s] > 0)).sum())
        out.setdefault("classes_below_0p1pct", {})[s] = int(((share.loc[s] < 0.1) & (lab.loc[s] > 0)).sum())
    out["bridge_pixels"] = {s: int(lab.loc[s, "bridge"]) for s in SPLITS}
    # Frames containing each class (any pixel), per split.
    out["frames_present_pct"] = {
        s: {c: float((frames.loc[frames.split == s, c] > 0).mean() * 100) for c in labeled_classes} for s in SPLITS}
    # Per-sequence dominance.
    sq = seq_long[seq_long["class"].isin(labeled_classes)].pivot_table(
        index="sequence", columns="class", values="pixels", aggfunc="sum")
    sq_share = sq.div(sq.sum(axis=1), axis=0) * 100
    dom = {}
    for seq in sq_share.index:
        r = sq_share.loc[seq].sort_values(ascending=False)
        dom[seq] = {"top": r.index[0], "top_pct": float(r.iloc[0]), "second": r.index[1],
                    "second_pct": float(r.iloc[1]), "n_classes_present": int((sq.loc[seq] > 0).sum())}
    out["sequence_dominant_class"] = dom
    # Which sequences hold each safety class / bridge (pixel share of the class total).
    hold = {}
    for c in SAFETY + ["bridge"]:
        col = sq[c]
        hold[c] = {seq: float(v / col.sum() * 100) for seq, v in col.sort_values(ascending=False).items() if v > 0}
    out["class_pixels_by_sequence_pct_of_class"] = hold
    # Total-variation distance between split class distributions.
    p = share / 100
    out["tv_distance"] = {f"{a}-{b}": float(0.5 * (p.loc[a] - p.loc[b]).abs().sum())
                          for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    return out


def write_split_summary(nums, out_dir: Path, top: list[str]):
    rows = []
    for s in SPLITS:
        sh = nums["share_pct"][s]
        r = {"split": s, "sequences": len(nums["sequences"][s]), "frames": nums["frames"][s],
             "labeled_pixels": nums["labeled_pixels"][s]}
        r.update({f"pct_{c}": round(sh[c], 3) for c in top})
        r["pct_other"] = round(100 - sum(sh[c] for c in top), 3)
        r["pct_safety_classes"] = round(nums["safety_total_pct"][s], 3)
        r["classes_absent"] = ";".join(nums["absent"][s])
        r["sequence_names"] = ";".join(nums["sequences"][s])
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "split_summary.csv", index=False)

    hdr = " & ".join([r"\textbf{Split}", "Seq.", "Frames"] + [c for c in top] + ["other", r"safety$^\dagger$",
                                                                                   "Absent classes"])
    lines = [
        r"% Generated by scripts/eda.py -- do not edit by hand. Requires \usepackage{booktabs}.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{RUGD official sequence split. Class columns give the percentage of each split's labeled "
        r"(non-void) pixels. $^\dagger$Sum of water, log, rock, rock-bed, dirt and sand.}",
        r"\label{tab:split_summary}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrr" + "r" * (len(top) + 2) + r"p{3.2cm}}",
        r"\toprule",
        hdr + r" \\",
        r"\midrule",
    ]
    for r in rows:
        cells = [r["split"], str(r["sequences"]), f"{r['frames']:,}".replace(",", "{,}")]
        cells += [f"{r[f'pct_{c}']:.1f}" for c in top] + [f"{r['pct_other']:.1f}", f"{r['pct_safety_classes']:.2f}"]
        cells.append(", ".join(short(c) for c in r["classes_absent"].split(";")) if r["classes_absent"] else "--")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (out_dir / "split_summary.tex").write_text("\n".join(lines))


# --------------------------------------------------------------------------- figures

def save(fig, path: Path):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


def fig_class_frequency(nums, classes, path: Path):
    labeled = [c for c in classes if c != "void"]
    sh = {s: nums["share_pct"][s] for s in SPLITS}
    order = sorted(labeled, key=lambda c: (sh["train"][c], sh["test"][c]), reverse=True)
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(3.5, 4.1))
    tr = [sh["train"][c] for c in order]
    colors = [HIGHLIGHT if c in SAFETY else NEUTRAL for c in order]
    left = 1e-4
    ax.barh(y, [max(v, left) - left if v > 0 else 0 for v in tr], left=left, height=0.72,
            color=colors, linewidth=0, zorder=2)
    for s, mk, kw in [("val", "o", dict(mfc="white", mec=INK, mew=0.8)),
                      ("test", "D", dict(mfc=INK, mec=INK, mew=0))]:
        xs = [sh[s][c] for c in order]
        yy = [yi for yi, v in zip(y, xs) if v > 0]
        ax.plot([v for v in xs if v > 0], yy, ls="none", marker=mk, ms=3.6 if mk == "o" else 2.8, zorder=3, **kw)
    ax.set_xscale("log")
    ax.set_xlim(left, 100)
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([short(c) for c in order])
    for t, c in zip(ax.get_yticklabels(), order):
        if c in SAFETY:
            t.set_color(HIGHLIGHT)
            t.set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
    ax.set_xticklabels(["0.0001", "0.001", "0.01", "0.1", "1", "10", "100"])
    ax.grid(axis="x", which="major", color="#e6e5e0", linewidth=0.6, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Share of labeled pixels (%, log scale)")
    # Text note for classes absent from a split.
    for yi, c in zip(y, order):
        miss = [s for s in SPLITS if sh[s][c] == 0]
        if miss:
            ax.text(1.4e-4 if sh["train"][c] == 0 else max(sh["train"][c], 2e-4) * 1.6, yi,
                    "absent: " + ", ".join(miss), va="center", ha="left", fontsize=6.5,
                    color=MUTED, style="italic", zorder=4)
    handles = [Patch(color=NEUTRAL, label="train"), Patch(color=HIGHLIGHT, label="train, safety class"),
               Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white", mec=INK, mew=0.8, label="val"),
               Line2D([], [], ls="none", marker="D", ms=2.8, mfc=INK, mec=INK, label="test")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=4, frameon=False,
              handlelength=1.0, borderaxespad=0.2, handletextpad=0.35, columnspacing=0.8)
    fig.tight_layout()
    save(fig, path)


def pick_samples(frames: pd.DataFrame, classes: list[str]):
    """One frame per chosen sequence: the frame with the most classes at >=0.5% of pixels,
    +2 for each safety class present at >=0.5%, ties broken by the earliest frame."""
    choice = [("trail-3", "train"), ("village", "train"), ("trail-15", "train"),
              ("trail-5", "val"), ("creek", "test"), ("park-1", "test")]
    labeled = [c for c in classes if c != "void"]
    picks = []
    for seq, split in choice:
        f = frames[frames.sequence == seq]
        tot = f[labeled].sum(axis=1)
        frac = f[labeled].div(tot, axis=0)
        score = (frac >= 0.005).sum(axis=1) + 2 * (frac[SAFETY] >= 0.005).sum(axis=1)
        picks.append((split, seq, f.loc[score.idxmax(), "frame"]))
    return picks


def fig_sample_grid(root: Path, picks, classes_rgb, path: Path):
    n = len(picks)
    W, gap, title_h, ncol = 7.16, 0.04, 0.17, 8
    pw = (W - gap * (n - 1)) / n
    ph = pw * 550 / 688
    names = [c for c, _ in classes_rgb]
    masks, imgs, seen = [], [], set()
    for split, seq, frame in picks:
        imgs.append(np.array(Image.open(root / "RUGD_frames-with-annotations" / seq / frame).convert("RGB")))
        ds = RUGDDataset(root, split, transform=None, ignore_void=False)
        i = [k for k, p in enumerate(ds.pairs) if p[0].name == frame][0]
        m = ds.load_mask(i)
        masks.append(m)
        seen |= set(np.unique(m).tolist()) - {IGNORE_INDEX, 0}  # void: <25 px per frame, invisible
    leg_rows = int(np.ceil(len(seen) / ncol))
    leg_h = 0.17 * leg_rows + 0.05
    H = title_h + 2 * ph + gap + leg_h
    fig = plt.figure(figsize=(W, H))
    for j, (split, seq, frame) in enumerate(picks):
        x = j * (pw + gap) / W
        for r, arr in enumerate([imgs[j], colorize(masks[j], classes_rgb)]):
            yb = (leg_h + (1 - r) * (ph + gap)) / H
            a = fig.add_axes([x, yb, pw / W, ph / H])
            a.imshow(arr, interpolation="nearest" if r else "antialiased")
            a.set_axis_off()
            if r == 0:
                a.set_title(f"{seq} ({split})", pad=2)
    handles = [Patch(facecolor=np.array(classes_rgb[k][1]) / 255, edgecolor="#8a8984", linewidth=0.4,
                     label=short(names[k])) for k in sorted(seen)]
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.0, handleheight=0.9, handletextpad=0.3, columnspacing=0.9, labelspacing=0.25,
               borderaxespad=0.0)
    save(fig, path)
    return sorted(names[k] for k in seen)


def fig_heatmap(seq_long, nums, classes, path: Path):
    labeled = [c for c in classes if c != "void"]
    order = sorted(labeled, key=lambda c: (nums["share_pct"]["train"][c], nums["share_pct"]["test"][c]),
                   reverse=True)
    sq = seq_long[seq_long["class"].isin(labeled)].pivot_table(
        index="sequence", columns="class", values="pixels", aggfunc="sum")
    rows, labels, split_of = [], [], []
    for s in SPLITS:
        seqs = sorted(RUGD_SPLITS[s], key=lambda q: -nums["frames_per_sequence"][q])
        for q in seqs:
            r = sq.loc[q, order]
            rows.append(r / r.sum() * 100)
            labels.append(f"{q} ({nums['frames_per_sequence'][q]})")
            split_of.append(s)
    for s in SPLITS:  # split aggregates at the bottom
        rows.append(pd.Series({c: nums["share_pct"][s][c] for c in order}))
        labels.append(f"all {s}")
        split_of.append("agg")
    M = np.array([r.values for r in rows], dtype=float)
    Mm = np.ma.masked_where(M <= 0, M)
    cmap = LinearSegmentedColormap.from_list("seq", ["#e3edf9", "#9cc0ea", "#2a78d6", "#123f78", "#06182f"])
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(7.16, 4.3))
    im = ax.imshow(Mm, cmap=cmap, norm=LogNorm(vmin=1e-3, vmax=100), aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([short(c) for c in order], rotation=55, ha="right", rotation_mode="anchor")
    for t, c in zip(ax.get_xticklabels(), order):
        if c in SAFETY:
            t.set_color(HIGHLIGHT)
            t.set_fontweight("bold")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.tick_params(length=0, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # Thin white grid, split separators and split labels.
    ax.set_xticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    # Zero cells: small gray dot so "absent" reads differently from "tiny".
    zy, zx = np.where(M <= 0)
    ax.plot(zx, zy, ls="none", marker="o", ms=1.6, color="#b4b2ab", zorder=3)
    start = 0
    for s in SPLITS + ["agg"]:
        n = split_of.count(s)
        if start > 0:
            ax.axhline(start - 0.5, color=INK, linewidth=1.0 if s != "agg" else 1.4)
        ax.text(-0.012, 1 - (start + n / 2) / len(labels), "" if s == "agg" else s.upper(),
                transform=ax.transAxes, ha="right", va="center", rotation=90, fontsize=8, fontweight="bold",
                color=INK)
        start += n
    # Push split labels left of the sequence names.
    for t in ax.texts:
        t.set_x(-0.155)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, extend="min")
    cb.set_ticks([1e-3, 1e-2, 1e-1, 1, 10, 100])
    cb.set_ticklabels(["0.001", "0.01", "0.1", "1", "10", "100"])
    cb.set_label("Share of the sequence's labeled pixels (%, log)")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2)
    ax.text(1.0, 1.01, "white cell with dot = class absent", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5, color=MUTED, style="italic")
    save(fig, path)


# --------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data/rugd")
    p.add_argument("--eda-dir", default="results/eda")
    p.add_argument("--fig-dir", default="results/figures")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--recount", action="store_true", help="ignore the cached pixel counts")
    args = p.parse_args()
    root, eda_dir, fig_dir = Path(args.root), Path(args.eda_dir), Path(args.fig_dir)
    eda_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ds = RUGDDataset(root, "train", transform=None, ignore_void=False)
    classes_rgb = ds.classes
    classes = [c for c, _ in classes_rgb]

    frame_csv = eda_dir / "class_pixels_by_frame.csv"
    seq_csv = eda_dir / "class_pixels_by_sequence.csv"
    if frame_csv.exists() and not args.recount:
        print(f"Using cached counts in {frame_csv}")
        frames = pd.read_csv(frame_csv)
    else:
        print(f"Counting pixels with {args.workers} workers ...")
        frames = count_pixels(root, classes, args.workers)
        frames.to_csv(frame_csv, index=False)
    seq_long = by_sequence_long(frames, classes)
    seq_long.to_csv(seq_csv, index=False)

    nums = numbers(frames, seq_long, classes)
    top = ["tree", "grass", "mulch", "gravel", "sky", "bush"]
    write_split_summary(nums, eda_dir, top)

    fig_class_frequency(nums, classes, fig_dir / "class_frequency")
    picks = pick_samples(frames, classes)
    nums["sample_grid_frames"] = [f"{seq}/{fr} ({s})" for s, seq, fr in picks]
    nums["sample_grid_classes"] = fig_sample_grid(root, picks, classes_rgb, fig_dir / "sample_grid")
    fig_heatmap(seq_long, nums, classes, fig_dir / "sequence_class_heatmap")
    (eda_dir / "eda_numbers.json").write_text(json.dumps(nums, indent=1))
    print(f"Wrote figures to {fig_dir} and tables to {eda_dir}")


if __name__ == "__main__":
    main()
