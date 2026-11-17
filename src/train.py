"""Train a segmentation model from a YAML config.

    python -m src.train --config configs/e1_unet_r18_ce.yaml
    python -m src.train --config configs/e1_unet_r18_ce.yaml --set train.epochs=5 data.root=/content/rugd

Writes to <train.out_dir>/<run_name>/: config.yaml, history.json, last.pt, best.pt.
Re-running the same command resumes from last.pt (for Colab timeouts).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .data import IGNORE_INDEX, build_dataset
from .losses import build_loss
from .metrics import ConfusionMatrix, predict
from .models import build_model
from .utils import get_device, git_commit, load_config, seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.lr=1e-4")
    return p.parse_args()


@torch.no_grad()
def validate(model, loader, num_classes, device, amp: bool, exclude: list[int]) -> ConfusionMatrix:
    model.eval()
    cm = ConfusionMatrix(num_classes, device=device)
    for images, masks in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(images)
        cm.update(predict(logits, exclude), masks)
    return cm


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    run_name = cfg.get("run_name", Path(args.config).stem)
    if "seed" in cfg and cfg.get("append_seed_to_name", False):
        run_name = f"{run_name}_s{cfg['seed']}"
    out_dir = Path(cfg["train"].get("out_dir", "runs")) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.get("seed", 0))
    device = get_device()
    amp = cfg["train"].get("amp", True) and device.type == "cuda"
    tc = cfg["train"]

    train_ds = build_dataset(cfg, "train", train=True)
    # E0 overfit check: validate on the same training subset, without augmentation.
    val_split = "train" if cfg["data"].get("val_on_train", False) else "val"
    val_ds = build_dataset(cfg, val_split, train=False)
    num_classes = train_ds.num_classes
    class_names = [c if isinstance(c, str) else c[0] for c in train_ds.classes]
    exclude = [0] if cfg["data"].get("ignore_void", True) else []

    g = torch.Generator().manual_seed(cfg.get("seed", 0))
    train_dl = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True, drop_last=True,
                          num_workers=tc.get("num_workers", 2), pin_memory=device.type == "cuda", generator=g)
    val_dl = DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=False,
                        num_workers=tc.get("num_workers", 2), pin_memory=device.type == "cuda")

    model = build_model(cfg, num_classes).to(device)
    criterion = build_loss(cfg, num_classes, device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 1e-4))
    total_steps = tc["epochs"] * len(train_dl)
    warmup = tc.get("warmup_steps", 0)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch, best_miou, history = 0, -1.0, []
    last_ckpt = out_dir / "last.pt"
    if tc.get("resume", True) and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_miou, history = ck["epoch"] + 1, ck["best_miou"], ck["history"]
        print(f"Resumed from {last_ckpt} at epoch {start_epoch}")

    cfg["_meta"] = {"git_commit": git_commit(), "num_classes": num_classes, "class_names": class_names,
                    "n_train": len(train_ds), "n_val": len(val_ds), "val_split": val_split}
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    wandb_run = None
    wc = cfg.get("wandb", {})
    if wc.get("enabled", False):
        import wandb

        wandb_run = wandb.init(project=wc.get("project", "reading-the-trail"), entity=wc.get("entity"),
                               name=run_name, id=run_name, resume="allow", config=cfg, dir=str(out_dir))

    for epoch in range(start_epoch, tc["epochs"]):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        for images, masks in train_dl:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            if (masks == IGNORE_INDEX).all():
                continue
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
                loss = criterion(logits.float(), masks)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            n += 1
        train_loss = running / max(n, 1)

        rec = {"epoch": epoch, "train_loss": train_loss, "lr": sched.get_last_lr()[0], "time_s": time.time() - t0}
        if (epoch + 1) % tc.get("eval_every", 1) == 0 or epoch + 1 == tc["epochs"]:
            cm = validate(model, val_dl, num_classes, device, amp, exclude)
            iou = cm.iou().tolist()
            rec.update({"val_miou": cm.miou(exclude), "val_pixel_acc": cm.pixel_acc(),
                        "val_iou": dict(zip(class_names, iou))})
            if rec["val_miou"] > best_miou:
                best_miou = rec["val_miou"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_miou": best_miou, "cfg": cfg},
                           out_dir / "best.pt")
        history.append(rec)
        print(f"[{run_name}] epoch {epoch + 1}/{tc['epochs']} loss {train_loss:.4f} "
              f"val_mIoU {rec.get('val_miou', float('nan')):.4f} best {best_miou:.4f} ({rec['time_s']:.0f}s)")

        if wandb_run:
            log = {k: v for k, v in rec.items() if k != "val_iou"}
            log.update({f"val_iou/{k}": v for k, v in rec.get("val_iou", {}).items()})
            wandb_run.log(log, step=epoch)

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best_miou": best_miou, "history": history},
                   last_ckpt)
        (out_dir / "history.json").write_text(json.dumps(history, indent=1))

    if wandb_run:
        wandb_run.summary["best_val_miou"] = best_miou
        wandb_run.finish()
    print(f"Done. Best val mIoU {best_miou:.4f}. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
