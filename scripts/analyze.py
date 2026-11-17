"""Turn finished runs and evaluation outputs into the paper's tables and figures.

    python -m scripts.analyze                                   # everything it can find
    python -m scripts.analyze --cm-run e3_dlv3p_r50_ce --cm-split val

Reads runs/<name>/{config.yaml,history.json} and results/<name>/<split>/* (written by
src.evaluate). Anything missing is skipped with a note. Writes:

    results/tables/  main_results.{csv,md,tex}, per_class_iou_losses.{csv,tex},
                     per_sequence_<split>.{csv,md}, boundary_band.{csv,md,tex}
    results/figures/ training_curves, per_class_iou_losses, arch_comparison,
                     confusion_<run>_<split>, group_confusion_<run>_<split>   (.pdf + .png)

Seed reruns (<name>_s1, <name>_s2) are aggregated with their base run as mean ± sample std.
E0 (overfit sanity check, validated on its training subset) is left out of all tables/figures.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from src import analysis as A
from src.data.class_map import GROUPS, RUGD_TO_GROUP

ARCH_LABEL = {"unet": "U-Net", "deeplabv3plus": "DeepLabV3+"}
ENC_LABEL = {"resnet18": "ResNet-18", "resnet34": "ResNet-34", "resnet50": "ResNet-50", "resnet101": "ResNet-101"}
LOSS_LABEL = {"ce": "CE", "weighted_ce": "Weighted CE", "dice": "Dice", "focal": "Focal", "ce_dice": "CE + Dice"}
GROUP_LABEL = {"smooth_ground": "smooth ground", "rough_ground": "rough ground", "vegetation": "vegetation",
               "water": "water", "obstacle": "obstacle", "sky": "sky"}


def log(msg: str) -> None:
    print(f"[analyze] {msg}")


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------


def discover(runs_dir: Path, results_dir: Path) -> dict[str, dict]:
    """name -> {cfg, history, best, summary{split}, per_class{split}, results_dir}."""
    names = set()
    if runs_dir.exists():
        names |= {p.parent.name for p in runs_dir.glob("*/history.json")}
    if results_dir.exists():
        names |= {p.parent.parent.name for p in results_dir.glob("*/*/summary.json")}
    runs = {}
    for name in sorted(names):
        if name.startswith("e0") or name in ("figures", "tables"):
            continue
        cfg = A.load_run_config(runs_dir / name)
        hist = A.load_history(runs_dir / name)
        info = {"name": name, "cfg": cfg, "history": hist, "best": A.best_epoch(hist),
                "summary": {}, "per_class": {}, "res": results_dir / name}
        for sdir in sorted((results_dir / name).glob("*")) if (results_dir / name).exists() else []:
            s = A.read_json(sdir / "summary.json")
            if s is not None:
                info["summary"][sdir.name] = s
                info["per_class"][sdir.name] = A.read_iou_csv(sdir / "per_class_iou.csv")
        runs[name] = info
    return runs


def exp_num(name: str) -> int:
    m = re.match(r"e(\d+)", name)
    return int(m.group(1)) if m else 99


def groups_of(runs: dict) -> dict[str, list[str]]:
    g: dict[str, list[str]] = {}
    for name in sorted(runs, key=lambda n: (exp_num(n), n)):
        g.setdefault(A.seed_group(name), []).append(name)
    return g


def describe(name: str, runs: dict) -> dict:
    cfg = next((runs[n]["cfg"] for n in [name] + [m for m in runs if A.seed_group(m) == name] if runs.get(n, {}).get("cfg")), {})
    m, loss = cfg.get("model", {}), cfg.get("loss", {})
    return {"id": A.exp_id(name), "arch": ARCH_LABEL.get(m.get("arch"), m.get("arch", "?")),
            "encoder": ENC_LABEL.get(m.get("encoder"), m.get("encoder", "?")),
            "loss": LOSS_LABEL.get(loss.get("name"), loss.get("name", "?")),
            "pretrained": "ImageNet" if m.get("pretrained", True) else "none"}


def short_label(name: str, runs: dict) -> str:
    d = describe(name, runs)
    enc = d["encoder"].replace("ResNet-", "R")
    s = f"{d['id']} {d['arch']}/{enc} {d['loss']}"
    return s + (" (scratch)" if d["pretrained"] == "none" else "")


def best_val(info: dict) -> float:
    return info["best"]["val_miou"] if info.get("best") else float("nan")


def summ(info: dict, split: str, key: str) -> float:
    v = info["summary"].get(split, {}).get(key)
    return float(v) if v is not None else float("nan")


def ce_winner(runs: dict, decisions: dict | None) -> str | None:
    if decisions and decisions.get("best_arch", {}).get("run") in runs:
        return decisions["best_arch"]["run"]
    cands = [n for n in ("e2_unet_r50_ce", "e3_dlv3p_r50_ce") if n in runs]
    return max(cands, key=lambda n: np.nan_to_num(best_val(runs[n]), nan=-1)) if cands else None


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------

METRICS = [  # (column, header, latex header, getter)
    ("best_val_miou", "Best val mIoU (hist.)", r"\shortstack{Best val\\mIoU}", lambda i: best_val(i)),
    ("val_miou", "Val mIoU", r"\shortstack{Val\\mIoU}", lambda i: summ(i, "val", "miou")),
    ("val_pixel_acc", "Val pixel acc", r"\shortstack{Val\\PA}", lambda i: summ(i, "val", "pixel_acc")),
    ("val_group_miou", "Val group mIoU", r"\shortstack{Val\\group}", lambda i: summ(i, "val", "group_miou")),
    ("test_miou", "Test mIoU", r"\shortstack{Test\\mIoU}", lambda i: summ(i, "test", "miou")),
    ("test_pixel_acc", "Test pixel acc", r"\shortstack{Test\\PA}", lambda i: summ(i, "test", "pixel_acc")),
    ("test_group_miou", "Test group mIoU", r"\shortstack{Test\\group}", lambda i: summ(i, "test", "group_miou")),
]


def tex_escape(s: str) -> str:
    return re.sub(r"([_&%#$])", r"\\\1", str(s))


def results_table(runs: dict, groups: dict, tables: Path) -> list[dict]:
    rows = []
    for g, members in groups.items():
        d = describe(g, runs)
        row = {"id": d["id"], "run": g, "arch": d["arch"], "encoder": d["encoder"], "loss": d["loss"],
               "pretrained": d["pretrained"], "seeds": len(members)}
        for col, *_rest, get in METRICS:
            row[col] = A.mean_std([get(runs[m]) for m in members])
        rows.append(row)
    if not rows:
        return rows
    cols = [c for c, *_ in METRICS if any(r[c][2] for r in rows)]  # drop all-empty columns

    with (tables / "main_results.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "run", "arch", "encoder", "loss", "pretrained", "seeds",
                    *[x for c in cols for x in (f"{c}_mean", f"{c}_std", f"{c}_n")]])
        for r in rows:
            w.writerow([r[k] for k in ("id", "run", "arch", "encoder", "loss", "pretrained", "seeds")]
                       + [("" if v != v else f"{v:.4f}") if isinstance(v, float) else v
                          for c in cols for v in r[c]])

    hdr = {c: h for c, h, *_ in METRICS}
    md = ["| ID | Model | Loss | Pretrain | Seeds | " + " | ".join(hdr[c] for c in cols) + " |",
          "|---|---|---|---|---|" + "---|" * len(cols)]
    for r in rows:
        md.append(f"| {r['id']} | {r['arch']} / {r['encoder']} | {r['loss']} | {r['pretrained']} | {r['seeds']} | "
                  + " | ".join(A.fmt_pm(*r[c]) for c in cols) + " |")
    md.append("")
    md.append("mIoU in %, mean ± sample std over seeds where Seeds > 1. mIoU excludes void and classes in "
              "`eval.exclude_from_miou` (bridge). 'Best val mIoU (hist.)' is the best-epoch value logged in training.")
    (tables / "main_results.md").write_text("\n".join(md) + "\n")

    best = {c: max((r[c][0] for r in rows if r[c][2]), default=None) for c in cols}
    ltx = {c: lh for c, _h, lh, _g in METRICS}
    lines = [r"% Generated by scripts/analyze.py. Requires \usepackage{booktabs}.",
             r"\begin{table*}[t]", r"\centering", r"\small",
             r"\caption{Main results on RUGD (mIoU and pixel accuracy in \%). Mean $\pm$ sample std over seeds "
             r"where Seeds $>1$. mIoU excludes void and \emph{bridge} (no training pixels); group = 6-class "
             r"traversability mIoU. Best per column in bold.}",
             r"\label{tab:main_results}",
             r"\begin{tabular}{llllc" + "c" * len(cols) + "}", r"\toprule",
             r"ID & Model & Loss & Pretrain & Seeds & " + " & ".join(ltx[c] for c in cols) + r" \\", r"\midrule"]
    for r in rows:
        cells = []
        for c in cols:
            s = A.fmt_pm(*r[c], latex=True)
            if r[c][2] and best[c] is not None and abs(r[c][0] - best[c]) < 1e-12:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        lines.append(" & ".join([r["id"], tex_escape(f"{r['arch']} / {r['encoder']}"), tex_escape(r["loss"]),
                                 r["pretrained"], str(r["seeds"])] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (tables / "main_results.tex").write_text("\n".join(lines) + "\n")
    log(f"tables: main_results.{{csv,md,tex}} ({len(rows)} configs)")
    return rows


def class_order(names: list[str], freq_csv: Path, drop: tuple[str, ...]) -> list[str]:
    """Classes sorted by training-pixel frequency (descending) when available."""
    freq = {r["class"]: float(r["frequency"]) for r in A.read_csv_rows(freq_csv)}
    keep = [n for n in names if n not in drop]
    return sorted(keep, key=lambda n: -freq.get(n, -1)) if freq else keep


def per_class_values(runs: dict, members: list[str], split: str = "val") -> dict[str, tuple]:
    """class -> (mean, std, n) over seed members; evaluated CSV if present, else best-epoch history."""
    per = []
    for m in members:
        info = runs[m]
        d = info["per_class"].get(split)
        if not d and split == "val":
            d = (info["best"] or {}).get("val_iou")
        if d:
            per.append(d)
    names = {k for d in per for k in d}
    return {k: A.mean_std([d.get(k) for d in per]) for k in names}


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def linestyle(name: str) -> str:
    n = exp_num(name)
    return "-" if n <= 3 else ("--" if n <= 7 else ":")


def fig_training_curves(runs: dict, figs: Path) -> None:
    import matplotlib.pyplot as plt

    base = [n for n in runs if A.seed_group(n) == n and 1 <= exp_num(n) <= 8 and runs[n]["history"]]
    if not base:
        log("skip training_curves: no histories")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(A.PAGE_W, 2.5))
    for n in sorted(base, key=exp_num):
        h = runs[n]["history"]
        c, ls, lab = A.exp_color(n), linestyle(n), short_label(n, runs)
        ev = [(r["epoch"] + 1, r["val_miou"]) for r in h if r.get("val_miou") is not None and r["val_miou"] == r["val_miou"]]
        if ev:
            x, y = zip(*ev)
            ax1.plot(x, np.array(y) * 100, color=c, ls=ls, label=lab)
            b = runs[n]["best"]
            ax1.plot(b["epoch"] + 1, b["val_miou"] * 100, "o", color=c, ms=4, mec="white", mew=0.8)
        ax2.plot([r["epoch"] + 1 for r in h], [r["train_loss"] for r in h], color=c, ls=ls, label=lab)
    ax1.set(xlabel="epoch", ylabel="val mIoU (%)", title="Validation mIoU (dot = best epoch)")
    ax2.set(xlabel="epoch", ylabel="train loss (log)", yscale="log", title="Training loss (each run's own objective)")
    from matplotlib.ticker import LogLocator, MaxNLocator, ScalarFormatter
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    ax2.yaxis.set_major_formatter(ScalarFormatter())
    ax2.yaxis.set_minor_formatter(plt.NullFormatter())
    handles, labels = ax1.get_legend_handles_labels()
    if not handles:
        handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), bbox_to_anchor=(0.5, 1.0),
               handlelength=2.5, columnspacing=1.2)
    fig.tight_layout()
    A.savefig(fig, figs, "training_curves")
    log("figure: training_curves")


def fig_per_class_losses(runs: dict, groups: dict, figs: Path, tables: Path, freq_csv: Path,
                         decisions: dict | None, exclude_classes: tuple[str, ...]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ce = ce_winner(runs, decisions)
    loss_runs = [g for g in ([A.seed_group(ce)] if ce else []) + ["e4_weighted_ce", "e5_dice", "e6_focal", "e7_ce_dice"]
                 if g in groups]
    vals = {g: per_class_values(runs, groups[g]) for g in loss_runs}
    loss_runs = [g for g in loss_runs if vals[g]]
    if len(loss_runs) < 2:
        log("skip per_class_iou_losses: need >= 2 loss variants with per-class IoU")
        return
    names = A.class_names_of(runs[loss_runs[0]]["cfg"])
    classes = class_order(names, freq_csv, ("void", *exclude_classes))
    freq = {r["class"]: float(r["frequency"]) for r in A.read_csv_rows(freq_csv)}

    fig, ax = plt.subplots(figsize=(A.PAGE_W, 2.6))
    k = len(loss_runs)
    w = 0.8 / k
    x = np.arange(len(classes))
    hl = set(A.RARE_CLASSES) | set(A.SAFETY_CLASSES)
    for i, c in enumerate(classes):
        if c in hl:
            ax.axvspan(i - 0.5, i + 0.5, color="#f0f0f0", zorder=0, lw=0)
    for j, g in enumerate(loss_runs):
        m = np.array([vals[g].get(c, (np.nan, np.nan, 0))[0] for c in classes]) * 100
        s = np.array([vals[g].get(c, (np.nan, np.nan, 0))[1] for c in classes]) * 100
        lab = describe(g, runs)["loss"] + (" (" + describe(g, runs)["id"] + ")")
        ax.bar(x - 0.4 + w * (j + 0.5), np.nan_to_num(m), w * 0.9, color=A.exp_color(g), label=lab, zorder=2,
               yerr=np.where(np.isnan(s), 0, s) if np.any(~np.isnan(s)) else None,
               error_kw={"elinewidth": 0.6, "capsize": 0, "ecolor": A.INK_MUTED})
    ax.set_xticks(x)
    ax.set_xticklabels([A.short_class(c) for c in classes], rotation=50, ha="right", rotation_mode="anchor")
    for t in ax.get_xticklabels():
        if t.get_text() in {A.short_class(c) for c in hl}:
            t.set_fontweight("bold")
            t.set_color(A.INK)
    ax.set(ylabel="val IoU (%)", ylim=(0, 100), xlim=(-0.6, len(classes) - 0.4))
    ax.grid(axis="x", visible=False)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="#f0f0f0", edgecolor="none"))
    labels.append("rare / safety class (bold)")
    ax.legend(handles, labels, ncol=len(labels), loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2)
    if freq:
        ax.set_xlabel("classes ordered by training-pixel share (most frequent left)", color=A.INK_MUTED)
    fig.tight_layout()
    A.savefig(fig, figs, "per_class_iou_losses")

    # Tables
    hdrs = [describe(g, runs)["loss"] for g in loss_runs]
    with (tables / "per_class_iou_losses.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["class", "train_freq", *[f"{g}_mean" for g in loss_runs], *[f"{g}_std" for g in loss_runs]])
        for c in classes:
            ms = [vals[g].get(c, (np.nan, np.nan, 0)) for g in loss_runs]
            wr.writerow([c, f"{freq.get(c, float('nan')):.6f}", *[f"{m[0]:.4f}" for m in ms], *[f"{m[1]:.4f}" for m in ms]])
    lines = [r"% Generated by scripts/analyze.py. Requires \usepackage{booktabs}.", r"\begin{table}[t]", r"\centering",
             r"\small", r"\caption{Per-class val IoU (\%) by loss on the best architecture, classes ordered by "
             r"training-pixel share. Rare/safety classes in italics; best per row in bold.}",
             r"\label{tab:per_class_losses}", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lr" + "r" * len(loss_runs) + "}", r"\toprule",
             r"Class & Train \% & " + " & ".join(tex_escape(h) for h in hdrs) + r" \\", r"\midrule"]
    for c in classes:
        ms = [vals[g].get(c, (np.nan, np.nan, 0))[0] for g in loss_runs]
        top = np.nanmax(ms) if np.any(~np.isnan(ms)) else None
        cells = [("--" if v != v else (r"\textbf{%.1f}" % (v * 100) if top is not None and v == top else "%.1f" % (v * 100)))
                 for v in ms]
        cname = tex_escape(c)
        if c in hl:
            cname = r"\emph{" + cname + "}"
        fr = f"{freq[c] * 100:.2f}" if c in freq else "--"
        lines.append(" & ".join([cname, fr, *cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (tables / "per_class_iou_losses.tex").write_text("\n".join(lines) + "\n")
    log(f"figure+tables: per_class_iou_losses ({', '.join(loss_runs)})")


def group_ordered_classes(names: list[str], drop: tuple[str, ...]) -> tuple[list[int], list[int]]:
    """Class indices ordered by traversability group (then index), plus group boundary positions."""
    idx, bounds = [], []
    for g in GROUPS:
        members = [i for i, n in enumerate(names) if RUGD_TO_GROUP.get(n) == g and n not in drop]
        if members:
            idx += members
            bounds.append(len(idx))
    return idx, bounds[:-1]


def fig_confusion(runs: dict, run: str, split: str, figs: Path) -> None:
    import matplotlib.pyplot as plt

    res = runs[run]["res"] / split if run in runs else None
    if res is None or not (res / "confusion.npy").exists():
        log(f"skip confusion: results/{run}/{split}/confusion.npy not found (run src.evaluate first)")
        return
    names = A.class_names_of(runs[run]["cfg"])
    mat = np.load(res / "confusion.npy").astype(np.float64)
    order, bounds = group_ordered_classes(names, ("void",))
    m = mat[np.ix_(order, order)]
    rows = m.sum(1, keepdims=True)
    norm = np.where(rows > 0, m / np.maximum(rows, 1), np.nan)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#d9d9d9")

    fig, ax = plt.subplots(figsize=(5.2, 4.9))
    im = ax.imshow(np.ma.masked_invalid(norm), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.grid(False)
    labels = [A.short_class(names[i]) for i in order]
    ax.tick_params(length=0)
    ax.set_xticks(range(len(order)), labels, rotation=60, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(order)), labels)
    for b in bounds:
        ax.axhline(b - 0.5, color="white", lw=1.2)
        ax.axvline(b - 0.5, color="white", lw=1.2)
    for i in range(len(order)):  # annotate large off-diagonal confusions only
        for j in range(len(order)):
            v = norm[i, j]
            if v == v and i != j and v >= 0.2:
                ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.55 else A.INK)
    ax.set(xlabel="predicted", ylabel="ground truth")
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("fraction of GT pixels (row-normalized)")
    cb.outline.set_visible(False)
    ax.set_title(f"{short_label(run, runs)}, {split}. Grouped by traversability;\n"
                 "off-diagonal cells ≥ 20% labeled; gray = no GT pixels", fontsize=8, color=A.INK_MUTED)
    fig.tight_layout()
    A.savefig(fig, figs, f"confusion_{run}_{split}")

    gpath = res / "group_confusion.npy"
    if gpath.exists():
        g = np.load(gpath).astype(np.float64)
        gr = g.sum(1, keepdims=True)
        gn = np.where(gr > 0, g / np.maximum(gr, 1), np.nan)
        fig, ax = plt.subplots(figsize=(A.COL_W, 3.0))
        ax.imshow(np.ma.masked_invalid(gn), cmap=cmap, vmin=0, vmax=1)
        ax.grid(False)
        gl = [GROUP_LABEL[x] for x in GROUPS]
        ax.tick_params(length=0)
        ax.set_xticks(range(len(GROUPS)), gl, rotation=40, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(GROUPS)), gl)
        for i in range(len(GROUPS)):
            for j in range(len(GROUPS)):
                v = gn[i, j]
                if v == v:
                    ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=8,
                            color="white" if v > 0.55 else A.INK)
        ax.set(xlabel="predicted group", ylabel="ground-truth group")
        ax.spines[:].set_visible(False)
        fig.tight_layout()
        A.savefig(fig, figs, f"group_confusion_{run}_{split}")
    log(f"figure: confusion_{run}_{split}" + (" + group" if gpath.exists() else ""))


def fig_arch(runs: dict, groups: dict, figs: Path) -> None:
    import matplotlib.pyplot as plt

    sel = [g for g in ("e1_unet_r18_ce", "e2_unet_r50_ce", "e3_dlv3p_r50_ce", "e8_no_pretrain") if g in groups]
    sel = [g for g in sel if any(A.is_finite(best_val(runs[m])) for m in groups[g])]
    if len(sel) < 2:
        log("skip arch_comparison: need >= 2 of E1, E2, E3, E8")
        return
    panels = [("mIoU (%)", lambda i: best_val(i), "Val mIoU, best epoch")]
    if any(A.is_finite(summ(runs[m], "val", "group_miou")) for g in sel for m in groups[g]):
        panels.append(("group mIoU (%)", lambda i: summ(i, "val", "group_miou"), "Val traversability mIoU (6 groups)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(A.COL_W * (1 if len(panels) == 1 else 2), 2.4), squeeze=False)
    def lab(g):
        d = describe(g, runs)
        return f"{d['id']}\n{d['arch']}/{d['encoder'].replace('ResNet-', 'R')}" + ("\nno pretrain" if d["pretrained"] == "none" else "")
    labels = [lab(g) for g in sel]
    for ax, (ylab, get, title) in zip(axes[0], panels):
        here = [g for g in sel if any(A.is_finite(get(runs[m])) for m in groups[g])]
        for k, g in enumerate(here):
            vals = [get(runs[m]) for m in groups[g]]
            mu, sd, n = A.mean_std(vals)
            if n == 0:
                continue
            ax.bar(k, mu * 100, 0.65, color=A.exp_color(g), zorder=2,
                   yerr=sd * 100 if n > 1 else None, error_kw={"elinewidth": 1, "capsize": 3, "ecolor": A.INK})
            if n > 1:
                v = np.array([x for x in vals if A.is_finite(x)]) * 100
                ax.plot(np.full(len(v), k) + np.linspace(-0.12, 0.12, len(v)), v, "o", ms=3,
                        color="white", mec=A.INK, mew=0.6, zorder=3)
            ax.text(k, mu * 100 + (sd * 100 if n > 1 else 0) + 1, f"{mu * 100:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(here)), [labels[sel.index(g)] for g in here])
        ax.set(ylabel=ylab, title=title)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    fig.tight_layout()
    A.savefig(fig, figs, "arch_comparison")
    log("figure: arch_comparison (error bars = sample std over seeds where available)")


def per_sequence_tables(runs: dict, tables: Path) -> None:
    for split in ("val", "test"):
        data = {n: {r["sequence"]: r for r in A.read_csv_rows(i["res"] / split / "per_sequence.csv")}
                for n, i in runs.items()}
        data = {n: d for n, d in data.items() if d}
        if not data:
            continue
        seqs = sorted({s for d in data.values() for s in d})
        with (tables / f"per_sequence_{split}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sequence", "n_images", *data])
            for s in seqs:
                n_img = next((d[s]["n_images"] for d in data.values() if s in d), "")
                w.writerow([s, n_img, *[d.get(s, {}).get("miou", "") for d in data.values()]])
        md = ["| Sequence | Frames | " + " | ".join(A.exp_id(n) + (f" ({n[-2:]})" if A.SEED_RE.search(n) else "")
                                               for n in data) + " |", "|---|---|" + "---|" * len(data)]
        for s in seqs:
            n_img = next((d[s]["n_images"] for d in data.values() if s in d), "")
            cells = [f"{float(d[s]['miou']) * 100:.1f}" if s in d and d[s]["miou"] else "–" for d in data.values()]
            md.append(f"| {s} | {n_img} | " + " | ".join(cells) + " |")
        md.append("\nmIoU (%) per sequence, from the summed confusion matrix of all frames in the sequence.")
        (tables / f"per_sequence_{split}.md").write_text("\n".join(md) + "\n")
        log(f"tables: per_sequence_{split}.{{csv,md}} ({len(data)} runs)")


def boundary_tables(runs: dict, tables: Path) -> None:
    rows = []
    for n, info in runs.items():
        for split, s in info["summary"].items():
            bb = s.get("boundary_band") or {}
            for key, v in bb.items():
                if not key.startswith("k="):
                    continue
                rows.append({"run": n, "split": split, "k": v["k"], "band_px": v["band_pixel_fraction"],
                             "err_in_band": v["error_fraction_in_band"], "miou": s.get("miou"),
                             "interior_miou": v["interior"]["miou"], "band_miou": v["band"]["miou"],
                             "pixel_acc": s.get("pixel_acc"), "interior_pa": v["interior"]["pixel_acc"],
                             "band_pa": v["band"]["pixel_acc"]})
    if not rows:
        return
    rows.sort(key=lambda r: (exp_num(r["run"]), r["run"], r["split"], r["k"]))
    keys = list(rows[0])
    with (tables / "boundary_band.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    p = lambda v: "–" if v is None or v != v else f"{v * 100:.1f}"  # noqa: E731
    md = ["| Run | Split | K (px) | Band pixels % | Errors in band % | mIoU all | mIoU interior | mIoU band "
          "| PA all | PA interior | PA band |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['run']} | {r['split']} | {r['k']} | {p(r['band_px'])} | {p(r['err_in_band'])} | {p(r['miou'])} "
                  f"| {p(r['interior_miou'])} | {p(r['band_miou'])} | {p(r['pixel_acc'])} | {p(r['interior_pa'])} "
                  f"| {p(r['band_pa'])} |")
    md.append("\nBand = GT pixels within K px (Chebyshev) of a different label, on the resized eval masks (data.img_size). "
              "'Errors in band' = share of all misclassified pixels that fall inside the band.")
    (tables / "boundary_band.md").write_text("\n".join(md) + "\n")
    t = lambda v: "--" if v is None or v != v else f"{v * 100:.1f}"  # noqa: E731
    lines = [r"% Generated by scripts/analyze.py. Requires \usepackage{booktabs}.", r"\begin{table}[t]", r"\centering",
             r"\small", r"\caption{Boundary-band analysis. The band holds GT pixels within $K$ px of a label "
             r"change; \emph{Err.} is the share of all errors inside it. mIoU in \%.}", r"\label{tab:boundary_band}",
             r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Run & Split & $K$ & Band px & Err. & All & Interior & Band \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join([A.exp_id(r["run"]) + tex_escape(r["run"][len(A.seed_group(r['run'])):]), r["split"],
                                 str(r["k"]), t(r["band_px"]), t(r["err_in_band"]), t(r["miou"]),
                                 t(r["interior_miou"]), t(r["band_miou"])]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (tables / "boundary_band.tex").write_text("\n".join(lines) + "\n")
    log(f"tables: boundary_band.{{csv,md,tex}} ({len(rows)} rows)")


# --------------------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="runs")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default=None, help="output root (default: --results); writes <out>/figures, <out>/tables")
    p.add_argument("--cm-run", default=None, help="run for the confusion matrix (default: best loss run, else best val)")
    p.add_argument("--cm-split", default="val", choices=["val", "test"])
    p.add_argument("--class-freq", default="results/class_freq_rugd.csv")
    p.add_argument("--exclude-classes", nargs="*", default=["bridge"],
                   help="classes left out of per-class figures (default: bridge, 0 training pixels)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    A.set_style()
    runs_dir, results_dir = Path(args.runs), Path(args.results)
    out = Path(args.out) if args.out else results_dir
    figs, tables = out / "figures", out / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    runs = discover(runs_dir, results_dir)
    if not runs:
        log(f"no runs found under {runs_dir} or {results_dir}; nothing to do")
        return
    groups = groups_of(runs)
    decisions = A.read_json(results_dir / "decisions.json")
    log(f"found {len(runs)} runs in {len(groups)} configs: {', '.join(runs)}")

    results_table(runs, groups, tables)
    fig_training_curves(runs, figs)
    fig_per_class_losses(runs, groups, figs, tables, Path(args.class_freq), decisions, tuple(args.exclude_classes))
    fig_arch(runs, groups, figs)
    per_sequence_tables(runs, tables)
    boundary_tables(runs, tables)

    cm_run = args.cm_run
    if cm_run is None:
        bl = (decisions or {}).get("best_loss", {}).get("run")
        with_cm = [n for n in runs if (runs[n]["res"] / args.cm_split / "confusion.npy").exists()]
        cm_run = bl if bl in with_cm else (max(with_cm, key=lambda n: np.nan_to_num(summ(runs[n], args.cm_split, "miou"),
                                                                                   nan=-1)) if with_cm else None)
    if cm_run:
        fig_confusion(runs, cm_run, args.cm_split, figs)
    else:
        log("skip confusion: no evaluated run (run src.evaluate first)")
    log(f"done -> {figs}, {tables}")


if __name__ == "__main__":
    main()
