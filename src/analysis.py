"""Evaluation-analysis helpers: boundary bands, per-image / per-sequence metrics,
run discovery and aggregation, and a shared figure style.

Used by src/evaluate.py (opt-in flags), scripts/analyze.py and scripts/gallery.py.
Nothing here is imported by training, so it can change while runs are in progress.
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data.colormaps import IGNORE_INDEX, RUGD_CLASSES

# --------------------------------------------------------------------------------------
# Boundary band
# --------------------------------------------------------------------------------------


def boundary_band(target: torch.Tensor, k: int) -> torch.Tensor:
    """Pixels within Chebyshev distance `k` of a pixel with a different GT label.

    `target` is (B, H, W) or (H, W) integer labels (IGNORE_INDEX included). A pixel is
    in the band iff its (2k+1)x(2k+1) window contains more than one label value, so a
    straight edge between two regions yields a band 2k pixels wide (k on each side).
    Transitions to ignore (void / unknown) count as boundaries: those are exactly the
    ambiguous edges the labelers left unlabeled. The image border is not a boundary
    (replicate padding). k = 0 gives an empty band.
    """
    squeeze = target.dim() == 2
    t = target[None] if squeeze else target
    if k <= 0:
        band = torch.zeros_like(t, dtype=torch.bool)
    else:
        x = t[:, None].float()
        x = F.pad(x, (k, k, k, k), mode="replicate")
        hi = F.max_pool2d(x, 2 * k + 1, stride=1)
        lo = -F.max_pool2d(-x, 2 * k + 1, stride=1)
        band = (hi != lo)[:, 0]
    return band[0] if squeeze else band


def split_by_band(target: torch.Tensor, band: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(interior_target, band_target): the GT with band / non-band pixels set to IGNORE_INDEX."""
    ign = torch.full_like(target, IGNORE_INDEX)
    return torch.where(band, ign, target), torch.where(band, target, ign)


# --------------------------------------------------------------------------------------
# Confusion-matrix helpers (numpy; rows = GT, cols = prediction)
# --------------------------------------------------------------------------------------


def batch_confusion(pred: torch.Tensor, target: torch.Tensor, n: int) -> torch.Tensor:
    """Per-image confusion matrices, shape (B, n, n), ignoring IGNORE_INDEX targets."""
    b = target.shape[0]
    pred, target = pred.reshape(b, -1).long().cpu(), target.reshape(b, -1).long().cpu()  # CPU: bincount hangs on MPS
    img = torch.arange(b, device=target.device)[:, None].expand_as(target)
    keep = (target != IGNORE_INDEX) & (target < n) & (pred >= 0) & (pred < n)
    idx = img[keep] * n * n + target[keep] * n + pred[keep]
    return torch.bincount(idx, minlength=b * n * n).reshape(b, n, n)


def iou_from_confusion(mat: np.ndarray) -> np.ndarray:
    """Per-class IoU; NaN where the class is absent from both GT and prediction."""
    m = np.asarray(mat, dtype=np.float64)
    tp = np.diag(m)
    denom = m.sum(0) + m.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)


def miou_from_confusion(mat: np.ndarray, exclude: list[int] | None = None, gt_present_only: bool = False) -> float:
    """Mean IoU over classes (minus `exclude`). With gt_present_only, only classes with GT pixels count."""
    m = np.asarray(mat, dtype=np.float64)
    iou = iou_from_confusion(m)
    keep = np.ones(len(iou), dtype=bool)
    if exclude:
        keep[list(exclude)] = False
    if gt_present_only:
        keep &= m.sum(1) > 0
    vals = iou[keep]
    vals = vals[~np.isnan(vals)]
    return float(vals.mean()) if len(vals) else float("nan")


def pixel_acc_from_confusion(mat: np.ndarray) -> float:
    m = np.asarray(mat, dtype=np.float64)
    tot = m.sum()
    return float(np.trace(m) / tot) if tot > 0 else float("nan")


def sequence_of(path: str | Path) -> str:
    """RUGD frames live in <root>/<frames dir>/<sequence>/<file>; the parent dir is the sequence."""
    return Path(path).parent.name


def aggregate_per_sequence(sequences: list[str], mats: list[np.ndarray] | np.ndarray,
                           exclude: list[int] | None = None, image_mious: list[float] | None = None) -> list[dict]:
    """Sequence-level metrics from SUMMED per-image confusion matrices (not averaged per-image mIoU).

    Returns one dict per sequence (in first-seen order) with n_images, miou, pixel_acc,
    and mean_image_miou (mean of the per-image values, if given, for comparison).
    """
    order: list[str] = []
    sums: dict[str, np.ndarray] = {}
    per_img: dict[str, list[float]] = {}
    for i, (s, m) in enumerate(zip(sequences, mats)):
        if s not in sums:
            order.append(s)
            sums[s] = np.zeros_like(np.asarray(m), dtype=np.int64)
            per_img[s] = []
        sums[s] += np.asarray(m, dtype=np.int64)
        if image_mious is not None:
            per_img[s].append(image_mious[i])
    rows = []
    for s in order:
        v = [x for x in per_img[s] if x == x]
        rows.append({"sequence": s, "n_images": sum(1 for q in sequences if q == s),
                     "miou": miou_from_confusion(sums[s], exclude), "pixel_acc": pixel_acc_from_confusion(sums[s]),
                     "mean_image_miou": float(np.mean(v)) if v else float("nan"), "confusion": sums[s]})
    return rows


# --------------------------------------------------------------------------------------
# Runs and results on disk
# --------------------------------------------------------------------------------------

SEED_RE = re.compile(r"_s(\d+)$")


def seed_group(name: str) -> str:
    """'e3_dlv3p_r50_ce_s2' -> 'e3_dlv3p_r50_ce'."""
    return SEED_RE.sub("", name)


def exp_id(name: str) -> str:
    m = re.match(r"e(\d+)", name)
    return f"E{m.group(1)}" if m else name


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_history(run_dir: Path) -> list[dict]:
    return read_json(Path(run_dir) / "history.json") or []


def best_epoch(history: list[dict]) -> dict | None:
    """History record with the highest val mIoU (= the epoch saved as best.pt)."""
    recs = [r for r in history if r.get("val_miou") is not None and r["val_miou"] == r["val_miou"]]
    return max(recs, key=lambda r: r["val_miou"]) if recs else None


def load_run_config(run_dir: Path) -> dict:
    import yaml

    p = Path(run_dir) / "config.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def class_names_of(cfg: dict) -> list[str]:
    names = (cfg.get("_meta") or {}).get("class_names")
    return list(names) if names else [n for n, _ in RUGD_CLASSES]


def read_csv_rows(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def read_iou_csv(path: Path) -> dict[str, float]:
    return {r["class"]: (float(r["iou"]) if r["iou"] not in ("", None) else float("nan")) for r in read_csv_rows(path)}


def mean_std(values: list[float]) -> tuple[float, float, int]:
    """Mean, sample std (ddof=1; NaN for n<2), n -- ignoring NaN/None."""
    v = [float(x) for x in values if x is not None and x == x]
    if not v:
        return float("nan"), float("nan"), 0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")), len(v)


def fmt_pm(mean: float, std: float, n: int, scale: float = 100.0, digits: int = 1, latex: bool = False) -> str:
    if n == 0 or mean != mean:
        return "--" if latex else "–"
    s = f"{mean * scale:.{digits}f}"
    if n > 1 and std == std:
        s += (r" $\pm$ " if latex else " ± ") + f"{std * scale:.{digits}f}"
    return s


def load_run_model(run_dir: Path, device="cpu", ckpt: str = "best.pt"):
    """(model, cfg, class_names) for a run directory, weights from `ckpt`, in eval mode."""
    from .models import build_model

    cfg = load_run_config(run_dir)
    names = class_names_of(cfg)
    n = (cfg.get("_meta") or {}).get("num_classes") or len(names)
    model = build_model({**cfg, "model": {**cfg["model"], "pretrained": False}}, n)
    ck = torch.load(Path(run_dir) / ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg, names


# --------------------------------------------------------------------------------------
# Figure style
# --------------------------------------------------------------------------------------

# Colorblind-checked categorical order (validated: adjacent CVD dE >= 9). One fixed color
# per experiment, used in every figure so E3 is always the same hue.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
EXP_COLORS = {f"E{i + 1}": c for i, c in enumerate(SERIES)}
INK, INK_MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e4e4e4"
ERROR_RGB = (213, 94, 0)       # error pixels in error maps (vermillion)
IGNORE_RGB = (150, 150, 150)   # ignore / void pixels
CORRECT_RGB = (255, 255, 255)

RARE_CLASSES = ["water", "rock", "log", "rock-bed"]
SAFETY_CLASSES = ["person", "vehicle", "pole", "sign", "bicycle", "fence"]

COL_W, PAGE_W = 3.5, 7.16  # IEEE single / double column widths (inches)


SHORT_CLASS = {"container/generic-object": "container/object"}


def short_class(name: str) -> str:
    """Display name for figures (full names stay in CSVs)."""
    return SHORT_CLASS.get(name, name)


def exp_color(name: str) -> str:
    return EXP_COLORS.get(exp_id(name), INK_MUTED)


def set_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "font.family": "sans-serif",
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "lines.linewidth": 1.5, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 100,
    })


def savefig(fig, out_dir: Path, name: str) -> list[Path]:
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=300)
    plt.close(fig)
    return paths


def palette_array(class_rgbs: list[tuple[int, int, int]]) -> np.ndarray:
    """256x3 palette: class colors, everything else (ignore) gray."""
    pal = np.tile(np.array(IGNORE_RGB, dtype=np.uint8), (256, 1))
    for i, rgb in enumerate(class_rgbs):
        pal[i] = rgb
    return pal


def error_map(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """RGB error map: errors vermillion, correct white, ignore gray."""
    out = np.empty((*target.shape, 3), dtype=np.uint8)
    out[:] = CORRECT_RGB
    out[pred != target] = ERROR_RGB
    out[target == IGNORE_INDEX] = IGNORE_RGB
    return out


def is_finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and not math.isnan(x)
