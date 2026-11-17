"""Evaluate a trained checkpoint.

    # Model selection: always on val
    python -m src.evaluate --run runs/e3_dlv3p_r50_ce --split val

    # Final numbers: test split, run ONCE per final config
    python -m src.evaluate --run runs/e3_dlv3p_r50_ce --split test --confirm-test

    # Stretch goal: RUGD-trained model scored on RELLIS-3D via the shared 6-group map
    python -m src.evaluate --run runs/e3_dlv3p_r50_ce --cross-rellis /path/to/rellis --split test --confirm-test

Writes to results/<run_name>/<split>[_rellis]/:
    summary.json, per_class_iou.csv, group_iou.csv, confusion.npy
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
from .data.class_map import GROUPS, RELLIS_GROUP_LUT, RUGD_GROUP_LUT, to_groups
from .metrics import ConfusionMatrix, predict
from .models import build_model
from .utils import get_device, load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run directory containing config.yaml and best.pt")
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--confirm-test", action="store_true", help="required for --split test")
    p.add_argument("--cross-rellis", default=None, help="RELLIS-3D root; evaluate on RELLIS with group mapping")
    p.add_argument("--data-root", default=None, help="override data.root (e.g. on a different machine)")
    p.add_argument("--out", default="results")
    p.add_argument("--set", nargs="*", default=[])
    return p.parse_args()


def write_csv(path: Path, names: list[str], ious: list[float]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "iou"])
        for n, v in zip(names, ious):
            w.writerow([n, f"{v:.4f}" if v == v else ""])


def main():
    args = parse_args()
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
        cfg["data"] = {**cfg["data"], "name": "rellis", "root": args.cross_rellis}
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
    with torch.no_grad():
        for images, masks in dl:
            images, masks = images.to(device), masks.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                pred = predict(model(images), exclude)
            if args.cross_rellis:
                gcm.update(to_groups(pred, RUGD_GROUP_LUT), to_groups(masks, RELLIS_GROUP_LUT))
            else:
                cm.update(pred, masks)
                gcm.update(to_groups(pred, RUGD_GROUP_LUT), to_groups(masks, RUGD_GROUP_LUT))

    tag = f"{args.split}_rellis" if args.cross_rellis else args.split
    out = Path(args.out) / run.name / tag
    out.mkdir(parents=True, exist_ok=True)
    summary = {"run": run.name, "split": args.split, "ckpt": args.ckpt, "ckpt_epoch": ck.get("epoch"),
               "n_images": len(ds), "git_commit_trained": meta.get("git_commit"),
               "group_miou": gcm.miou(), "group_pixel_acc": gcm.pixel_acc()}
    if cm is not None:
        summary.update({"miou": cm.miou(miou_exclude), "miou_incl_excluded": cm.miou(exclude),
                        "miou_excluded_classes": cfg.get("eval", {}).get("exclude_from_miou", []),
                        "pixel_acc": cm.pixel_acc()})
        write_csv(out / "per_class_iou.csv", class_names, cm.iou().tolist())
        np.save(out / "confusion.npy", cm.mat.cpu().numpy())
    write_csv(out / "group_iou.csv", GROUPS, gcm.iou().tolist())
    np.save(out / "group_confusion.npy", gcm.mat.cpu().numpy())
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
