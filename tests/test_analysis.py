"""Tests for the evaluation-analysis tooling: boundary band, per-image / per-sequence
metrics, evaluate.py's opt-in flags, and analyze.py / gallery.py end to end (CPU, fake data)."""
import csv
import json
import math
import shutil
import sys

import numpy as np
import pytest
import torch

from src import analysis as A
from src.data.colormaps import IGNORE_INDEX


# ---------------------------------------------------------------- boundary band


def _two_regions():
    t = torch.ones(6, 6, dtype=torch.long)
    t[:, 3:] = 2  # vertical edge between columns 2 and 3
    return t


def test_boundary_band_hand_checked():
    t = _two_regions()
    assert not A.boundary_band(t, 0).any()
    b1 = A.boundary_band(t, 1)
    assert b1[:, [2, 3]].all() and not b1[:, [0, 1, 4, 5]].any()  # 1 px each side
    b2 = A.boundary_band(t, 2)
    assert b2[:, 1:5].all() and not b2[:, [0, 5]].any()  # 2 px each side
    # image border is not a boundary; batched input gives the same answer
    assert torch.equal(A.boundary_band(t[None].repeat(2, 1, 1), 1)[1], b1)


def test_boundary_band_counts_void_transition():
    t = torch.ones(5, 5, dtype=torch.long)
    t[0, 0] = IGNORE_INDEX
    b = A.boundary_band(t, 1)
    expected = torch.zeros(5, 5, dtype=torch.bool)
    expected[:2, :2] = True
    assert torch.equal(b, expected)


def test_band_split_metrics():
    from src.metrics import ConfusionMatrix

    t = _two_regions()
    pred = torch.ones_like(t)  # everything predicted as class 1: right half wrong
    inner, band = A.split_by_band(t, A.boundary_band(t, 1))
    cm_in, cm_band = ConfusionMatrix(3), ConfusionMatrix(3)
    cm_in.update(pred, inner)
    cm_band.update(pred, band)
    assert cm_in.mat.sum() == 24 and cm_band.mat.sum() == 12  # cols 0,1,4,5 vs cols 2,3
    assert math.isclose(cm_in.pixel_acc(), 0.5) and math.isclose(cm_band.pixel_acc(), 0.5)
    assert math.isclose(cm_in.iou()[1].item(), 12 / 24)  # tp 12, fp 12 (class 2 interior)
    # band + interior partition the valid pixels
    full = ConfusionMatrix(3)
    full.update(pred, t)
    assert torch.equal(cm_in.mat + cm_band.mat, full.mat)


# ---------------------------------------------------------------- per-image / per-sequence


def test_batch_confusion_matches_confusion_matrix():
    from src.metrics import ConfusionMatrix

    g = torch.Generator().manual_seed(0)
    t = torch.randint(0, 4, (3, 7, 5), generator=g)
    t[0, 0, 0] = IGNORE_INDEX
    p = torch.randint(0, 4, (3, 7, 5), generator=g)
    per = A.batch_confusion(p, t, 4)
    for i in range(3):
        cm = ConfusionMatrix(4)
        cm.update(p[i], t[i])
        assert torch.equal(per[i], cm.mat)


def test_per_sequence_uses_summed_confusion():
    n, ex = 3, [0]

    def conf(gt, pr):
        return A.batch_confusion(torch.tensor([pr]), torch.tensor([gt]), n)[0].numpy()

    mats = [conf([1, 1, 1, 1], [1, 1, 1, 1]),   # seq A: perfect
            conf([2, 2, 2, 2], [1, 1, 2, 2]),   # seq A: half of class 2 -> class 1
            conf([1, 2, 1, 2], [1, 2, 1, 2])]   # seq B: perfect
    img = [A.miou_from_confusion(m, ex, gt_present_only=True) for m in mats]
    assert img == [1.0, 0.5, 1.0]  # per-image mIoU only over GT-present classes
    rows = A.aggregate_per_sequence(["A", "A", "B"], mats, ex, img)
    a, b = rows
    assert (a["sequence"], a["n_images"], b["sequence"], b["n_images"]) == ("A", 2, "B", 1)
    # summed: class1 tp4 / (gt4 + pred6 - 4) = 2/3, class2 tp2 / (4 + 2 - 2) = 1/2
    assert math.isclose(a["miou"], (2 / 3 + 1 / 2) / 2)
    assert math.isclose(a["mean_image_miou"], 0.75)  # differs: not an average of per-image values
    assert math.isclose(a["pixel_acc"], 6 / 8) and b["miou"] == 1.0


def test_seed_group_and_mean_std():
    assert A.seed_group("e3_dlv3p_r50_ce_s2") == "e3_dlv3p_r50_ce" and A.seed_group("e5_dice") == "e5_dice"
    m, s, k = A.mean_std([0.5, 0.6, None, float("nan"), 0.7])
    assert k == 3 and math.isclose(m, 0.6) and math.isclose(s, 0.1)
    assert A.fmt_pm(m, s, k) == "60.0 ± 10.0" and A.fmt_pm(0.5, float("nan"), 1) == "50.0"


# ---------------------------------------------------------------- end to end


def _rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


@pytest.mark.filterwarnings("ignore")
def test_evaluate_analyze_gallery_end_to_end(fake_rugd, tmp_path, monkeypatch):
    pytest.importorskip("segmentation_models_pytorch")
    pytest.importorskip("matplotlib")
    from scripts import analyze, gallery
    from src import evaluate, train

    runs, res = tmp_path / "runs", tmp_path / "results"
    common = [f"data.root={fake_rugd}", "data.img_size=64", "model.pretrained=false", "train.epochs=1",
              "train.batch_size=2", "train.num_workers=0", "train.warmup_steps=0", f"train.out_dir={runs}"]
    monkeypatch.setattr(sys, "argv", ["train", "--config", "configs/e1_unet_r18_ce.yaml", "--set", *common])
    train.main()
    base = runs / "e1_unet_r18_ce"

    # Fake the rest of the matrix by copying the tiny run and perturbing its history.
    names = ["e1_unet_r18_ce", "e2_unet_r50_ce", "e3_dlv3p_r50_ce", "e3_dlv3p_r50_ce_s1", "e3_dlv3p_r50_ce_s2",
             "e4_weighted_ce", "e5_dice", "e6_focal", "e7_ce_dice", "e8_no_pretrain"]
    rec = json.loads((base / "history.json").read_text())[0]
    for k, name in enumerate(names):
        d = runs / name
        if name != base.name:
            shutil.copytree(base, d)
        hist = []
        for e in range(3):
            r = json.loads(json.dumps(rec))
            r.update(epoch=e, train_loss=rec["train_loss"] / (e + 1), val_miou=0.1 + 0.02 * k + 0.05 * e)
            r["val_iou"] = {c: (v * (1 + 0.1 * k) if v == v else v) for c, v in rec["val_iou"].items()}
            hist.append(r)
        (d / "history.json").write_text(json.dumps(hist))
    (res).mkdir()
    (res / "decisions.json").write_text(json.dumps({"best_arch": {"run": "e3_dlv3p_r50_ce"},
                                                    "best_loss": {"run": "e7_ce_dice"}}))

    # evaluate: default call stays backward compatible
    evaluate.main(["--run", str(runs / "e2_unet_r50_ce"), "--split", "val", "--out", str(res)])
    out2 = res / "e2_unet_r50_ce" / "val"
    assert {"summary.json", "per_class_iou.csv", "group_iou.csv", "confusion.npy"} <= {p.name for p in out2.iterdir()}
    assert not (out2 / "per_image.csv").exists() and "boundary_band" not in json.loads((out2 / "summary.json").read_text())

    # evaluate with all analysis flags
    evaluate.main(["--run", str(runs / "e7_ce_dice"), "--split", "val", "--out", str(res),
                   "--per-image", "--boundary-band", "1", "2", "--save-preds", "2"])
    out = res / "e7_ce_dice" / "val"
    s = json.loads((out / "summary.json").read_text())
    per_img = _rows(out / "per_image.csv")
    assert len(per_img) == 4 and {r["sequence"] for r in per_img} == {"park-8"}
    assert all(0 <= float(r["miou"]) <= 1 for r in per_img) and per_img[0]["gt_classes"]
    seq = _rows(out / "per_sequence.csv")
    assert len(seq) == 1 and math.isclose(float(seq[0]["miou"]), s["miou"], abs_tol=1e-4)  # one val sequence
    for k in (1, 2):
        bb = s["boundary_band"][f"k={k}"]
        assert 0 < bb["band_pixel_fraction"] <= 1 and 0 <= bb["error_fraction_in_band"] <= 1
        # random fake labels: nearly every pixel is a boundary, so the interior may be empty (NaN)
        ia = bb["interior"]["pixel_acc"]
        assert (ia != ia or 0 <= ia <= 1) and 0 <= bb["band"]["miou"] <= 1
    assert s["boundary_band"]["k=2"]["band_pixel_fraction"] >= s["boundary_band"]["k=1"]["band_pixel_fraction"]
    assert len(_rows(out / "boundary_band_iou.csv")) == 25
    preds = sorted((out / "preds").rglob("*.png"))
    assert len(preds) == 2
    from PIL import Image
    assert np.array(Image.open(preds[0])).shape == (64, 64)

    for name in ("e3_dlv3p_r50_ce", "e3_dlv3p_r50_ce_s1"):
        evaluate.main(["--run", str(runs / name), "--split", "val", "--out", str(res)])

    analyze.main(["--runs", str(runs), "--results", str(res)])
    figs, tabs = res / "figures", res / "tables"
    for f in ("training_curves", "per_class_iou_losses", "arch_comparison", "confusion_e7_ce_dice_val",
              "group_confusion_e7_ce_dice_val"):
        assert (figs / f"{f}.pdf").exists() and (figs / f"{f}.png").exists(), f
    for f in ("main_results.csv", "main_results.md", "main_results.tex", "per_class_iou_losses.tex",
              "per_class_iou_losses.csv", "per_sequence_val.csv", "boundary_band.md", "boundary_band.tex"):
        assert (tabs / f).exists(), f
    tex = (tabs / "main_results.tex").read_text()
    assert r"\toprule" in tex and r"\bottomrule" in tex and "e0" not in tex
    main = {r["run"]: r for r in _rows(tabs / "main_results.csv")}
    assert main["e3_dlv3p_r50_ce"]["seeds"] == "3" and float(main["e3_dlv3p_r50_ce"]["best_val_miou_std"]) > 0
    assert main["e1_unet_r18_ce"]["seeds"] == "1"

    # analyze copes with an empty tree
    analyze.main(["--runs", str(tmp_path / "none"), "--results", str(tmp_path / "none_res")])

    gallery.main(["--run", str(runs / "e7_ce_dice"), "--split", "val", "--results", str(res),
                  "--compare", str(runs / "e3_dlv3p_r50_ce"), "--rare-classes", "water", "log", "rock",
                  "--n-compare", "2", "--threads", "1"])
    assert (figs / "gallery_e7_ce_dice_val.png").exists()
    assert (figs / "compare_e7_ce_dice_vs_e3_dlv3p_r50_ce_val.pdf").exists()
    with pytest.raises(SystemExit):
        gallery.main(["--run", str(runs / "e7_ce_dice"), "--split", "test", "--results", str(res)])
