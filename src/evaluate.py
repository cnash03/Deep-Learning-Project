"""Evaluate a trained checkpoint.

    # Model selection: always on val
    python -m src.evaluate --run runs/e3_dlv3p_r50_ce --split val

    # Final numbers: test split, run ONCE per final config
    python -m src.evaluate --run runs/e3_dlv3p_r50_ce --split test --confirm-test

    # Stretch goal: RUGD-trained model scored on RELLIS-3D via the shared 6-group map
    python -m src.evaluate --run runs/e3_dlv3p_r50_ce --cross-rellis /path/to/rellis --split test --confirm-test

    # Error analysis (opt-in, same single pass over the data):
    python -m src.evaluate --run runs/e3_dlv3p_r50_ce --split val --per-image --boundary-band 1 3 5 --save-preds 12

Writes to results/<run_name>/<split>[_rellis]/:
    summary.json, per_class_iou.csv, group_iou.csv, confusion.npy, group_confusion.npy
    --per-image       per_image.csv (file, sequence, mIoU over GT-present classes, pixel acc, GT classes)
                      per_sequence.csv (from summed per-sequence confusion matrices) + per_sequence_confusion.npz
    --boundary-band K summary.json["boundary_band"]["k=K"]: metrics excluding / only on GT pixels within K px
                      of a label boundary (on the resized eval masks); boundary_band_iou.csv per class
    --save-preds N    preds/<sequence>/<file>.png: N predicted index masks (uint8), evenly spaced or --pred-indices
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import build_dataset
from . import analysis as A
from .data.class_map import GROUPS, RELLIS_GROUP_LUT, RELLIS_GROUP_LUT_SEEN, RELLIS_UNSEEN, RUGD_GROUP_LUT, to_groups
from .metrics import ConfusionMatrix, predict
from .models import build_model
from .utils import get_device, load_config


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run directory containing config.yaml and best.pt")
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--confirm-test", action="store_true", help="required for --split test")
    p.add_argument("--cross-rellis", default=None, help="RELLIS-3D root; evaluate on RELLIS with group mapping")
    p.add_argument("--data-root", default=None, help="override data.root (e.g. on a different machine)")
    p.add_argument("--out", default="results")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--per-image", action="store_true", help="write per_image.csv and per_sequence.csv")
    p.add_argument("--boundary-band", type=int, nargs="+", default=[], metavar="K",
                   help="also score GT pixels farther than / within K px of a label boundary (one or more K)")
    p.add_argument("--save-preds", type=int, default=0, metavar="N", help="save N predicted masks (evenly spaced)")
    p.add_argument("--pred-indices", type=int, nargs="+", default=None,
                   help="dataset indices of the predictions to save (overrides the even spacing)")
    return p.parse_args(argv)


def write_csv(path: Path, names: list[str], ious: list[float]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "iou"])
        for n, v in zip(names, ious):
            w.writerow([n, f"{v:.4f}" if v == v else ""])


def main(argv=None):
    args = parse_args(argv)
    if args.split == "test" and not args.confirm_test:
        raise SystemExit("Refusing to touch the test split without --confirm-test. Select models on val.")

    run = Path(args.run)
    cfg = load_config(run / "config.yaml", args.set)
    meta = cfg.pop("_meta", {})
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    cfg["data"]["subset"] = None
    device = get_device()
    amp = device.type == "cuda"

    num_classes = meta.get("num_classes")
    class_names = meta.get("class_names")
    exclude = [0] if cfg["data"].get("ignore_void", True) else []

    miou_exclude = exclude + [class_names.index(n) for n in cfg.get("eval", {}).get("exclude_from_miou", [])
                              if class_names and n in class_names]

    if args.cross_rellis:
        size = cfg["data"]["img_size"]
        # Approximate RUGD's train-time squeeze (688x550 -> 384x384: 0.80); 1920x1200 -> 384x480 gives 0.78.
        cfg["data"] = {**cfg["data"], "name": "rellis", "root": args.cross_rellis,
                       "eval_size": [size, int(round(size * 1.25 / 32)) * 32]}
    ds = build_dataset(cfg, args.split, train=False)
    if num_classes is None:
        num_classes, class_names = ds.num_classes, [c if isinstance(c, str) else c[0] for c in ds.classes]

    model = build_model({**cfg, "model": {**cfg["model"], "pretrained": False}}, num_classes).to(device)
    ck = torch.load(run / args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                    num_workers=cfg["train"].get("num_workers", 2), pin_memory=device.type == "cuda")
    cm = ConfusionMatrix(num_classes, device=device) if not args.cross_rellis else None
    gcm = ConfusionMatrix(len(GROUPS), device=device)
    gcm_all = ConfusionMatrix(len(GROUPS), device=device) if args.cross_rellis else None  # unseen classes scored

    if args.cross_rellis and (args.per_image or args.boundary_band):
        raise SystemExit("--per-image / --boundary-band are only supported for in-domain evaluation")
    files = [str(p[0]) for p in getattr(ds, "pairs", [])] or [str(i) for i in range(len(ds))]
    ks = sorted({k for k in args.boundary_band if k > 0})
    band_cms = {k: (ConfusionMatrix(num_classes, device=device), ConfusionMatrix(num_classes, device=device))
                for k in ks}
    band_err = {k: 0 for k in ks}  # misclassified (non-ignore) pixels inside the band
    total_err = 0
    if args.pred_indices is not None:
        save_idx = {i for i in args.pred_indices if 0 <= i < len(ds)}
    elif args.save_preds > 0:
        save_idx = set(np.linspace(0, len(ds) - 1, min(args.save_preds, len(ds))).astype(int).tolist())
    else:
        save_idx = set()
    tag = f"{args.split}_rellis" if args.cross_rellis else args.split
    out = Path(args.out) / run.name / tag
    out.mkdir(parents=True, exist_ok=True)
    img_rows, img_mats = [], []

    offset = 0
    with torch.no_grad():
        for images, masks in dl:
            images, masks = images.to(device), masks.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                pred = predict(model(images), exclude)
            if args.cross_rellis:
                gpred = to_groups(pred, RUGD_GROUP_LUT)
                gcm.update(gpred, to_groups(masks, RELLIS_GROUP_LUT_SEEN))
                gcm_all.update(gpred, to_groups(masks, RELLIS_GROUP_LUT))
            else:
                cm.update(pred, masks)
                gcm.update(to_groups(pred, RUGD_GROUP_LUT), to_groups(masks, RUGD_GROUP_LUT))
            # Analysis extras run on CPU (bincount / indexed ops have hung the Apple MPS backend).
            if ks or args.per_image:
                pred_c, masks_c = pred.cpu(), masks.cpu()
            if ks:
                wrong = (pred_c != masks_c) & (masks_c != A.IGNORE_INDEX)
                total_err += int(wrong.sum())
                for k in ks:
                    band = A.boundary_band(masks_c, k)
                    inner, on_band = A.split_by_band(masks_c, band)
                    band_cms[k][0].update(pred_c, inner)
                    band_cms[k][1].update(pred_c, on_band)
                    band_err[k] += int((wrong & band).sum())
            if args.per_image:
                mats = A.batch_confusion(pred_c, masks_c, num_classes).numpy()
                for j, m in enumerate(mats):
                    f = files[offset + j]
                    gt = np.flatnonzero(m.sum(1) > 0)
                    img_rows.append({"index": offset + j, "file": Path(f).name, "sequence": A.sequence_of(f),
                                     "miou": A.miou_from_confusion(m, miou_exclude, gt_present_only=True),
                                     "pixel_acc": A.pixel_acc_from_confusion(m), "n_pixels": int(m.sum()),
                                     "gt_classes": ";".join(class_names[c] for c in gt)})
                    img_mats.append(m)
            for j in range(images.shape[0]):
                if offset + j in save_idx:
                    from PIL import Image

                    f = Path(files[offset + j])
                    d = out / "preds" / A.sequence_of(f)
                    d.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(pred[j].to(torch.uint8).cpu().numpy()).save(d / f"{f.stem}.png")
            offset += images.shape[0]

    summary = {"run": run.name, "split": args.split, "ckpt": args.ckpt, "ckpt_epoch": ck.get("epoch"),
               "n_images": len(ds), "git_commit_trained": meta.get("git_commit"),
               "group_miou": gcm.miou(), "group_pixel_acc": gcm.pixel_acc()}
    if gcm_all is not None:
        summary.update({"eval_size": cfg["data"]["eval_size"], "rellis_unseen_ignored": RELLIS_UNSEEN,
                        "group_miou_unseen_scored": gcm_all.miou(),
                        "group_iou_unseen_scored": dict(zip(GROUPS, gcm_all.iou().tolist()))})
    if cm is not None:
        summary.update({"miou": cm.miou(miou_exclude), "miou_incl_excluded": cm.miou(exclude),
                        "miou_gt_present": cm.miou_gt_present(miou_exclude),
                        "miou_excluded_classes": cfg.get("eval", {}).get("exclude_from_miou", []),
                        "pixel_acc": cm.pixel_acc()})
        write_csv(out / "per_class_iou.csv", class_names, cm.iou().tolist())
        np.save(out / "confusion.npy", cm.mat.cpu().numpy())
    if ks:
        total_px = int(cm.mat.sum())
        bb = {"definition": "band(K) = GT pixels whose (2K+1)x(2K+1) window contains another label "
                            "(incl. void/ignore), on the resized eval masks; 'interior' excludes the band, "
                            "'band' scores only band pixels. mIoU uses the same class exclusions as the headline.",
              "img_size": cfg["data"]["img_size"]}
        rows = {"full": cm.iou().tolist()}
        for k in ks:
            inner, on_band = band_cms[k]
            band_px = int(on_band.mat.sum())
            bb[f"k={k}"] = {
                "k": k, "band_pixel_fraction": band_px / max(total_px, 1),
                "error_fraction_in_band": band_err[k] / max(total_err, 1),
                "interior": {"miou": inner.miou(miou_exclude), "pixel_acc": inner.pixel_acc()},
                "band": {"miou": on_band.miou(miou_exclude), "pixel_acc": on_band.pixel_acc()},
            }
            rows[f"interior_k{k}"] = inner.iou().tolist()
            rows[f"band_k{k}"] = on_band.iou().tolist()
        summary["boundary_band"] = bb
        with (out / "boundary_band_iou.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class", *rows])
            for i, n in enumerate(class_names):
                w.writerow([n, *(f"{v[i]:.4f}" if v[i] == v[i] else "" for v in rows.values())])
    if args.per_image:
        with (out / "per_image.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(img_rows[0]) if img_rows else ["index"])
            w.writeheader()
            for r in img_rows:
                w.writerow({k: (f"{v:.4f}" if isinstance(v, float) and v == v else ("" if isinstance(v, float) else v))
                            for k, v in r.items()})
        seqs = A.aggregate_per_sequence([r["sequence"] for r in img_rows], img_mats, miou_exclude,
                                        [r["miou"] for r in img_rows])
        with (out / "per_sequence.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sequence", "n_images", "miou", "pixel_acc", "mean_image_miou"])
            for r in seqs:
                w.writerow([r["sequence"], r["n_images"], *(f"{r[k]:.4f}" if r[k] == r[k] else ""
                                                             for k in ("miou", "pixel_acc", "mean_image_miou"))])
        np.savez_compressed(out / "per_sequence_confusion.npz", **{r["sequence"]: r["confusion"] for r in seqs})
        summary["per_sequence_miou"] = {r["sequence"]: r["miou"] for r in seqs}
    if save_idx:
        summary["saved_preds"] = sorted(save_idx)
    write_csv(out / "group_iou.csv", GROUPS, gcm.iou().tolist())
    np.save(out / "group_confusion.npy", gcm.mat.cpu().numpy())
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
