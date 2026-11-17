"""Run the full experiment matrix (E0-E9) in order on one GPU, making the
model-selection decisions from build.md automatically and only from VAL mIoU.

    caffeinate -is python -m scripts.run_all            # resumable: finished runs are skipped

Stages:
  1. E0 overfit sanity check (aborts everything if it fails)
  2. E1 U-Net/R18, E2 U-Net/R50, E3 DeepLabV3+/R50 (all CE)
  3. Pick best architecture of E2 vs E3 -> configs/best_arch.yaml
  4. E4-E7 losses on the best architecture
  5. Pick best loss among {CE (the E2/E3 winner), E4..E7}
  6. E8 = best arch + best loss, no ImageNet pretraining
  7. E9 = top-2 configs among E1-E7 rerun with seeds 1 and 2
  8. Val evaluation of every run (the test split is NOT touched here)

Every decision is appended to results/decisions.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from src.utils import load_config

RUNS = Path("runs")
DECISIONS = Path("results/decisions.json")
E0_MIN_MIOU = 0.7
MAX_ATTEMPTS = 30
TRANSIENT_ERRORS = ("MPS command buffer", "AcceleratorError", "kIOGPUCommandBufferCallbackError")


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    RUNS.mkdir(exist_ok=True)
    with (RUNS / "orchestrator.log").open("a") as f:
        f.write(line + "\n")


def record(key: str, value) -> None:
    DECISIONS.parent.mkdir(exist_ok=True)
    d = json.loads(DECISIONS.read_text()) if DECISIONS.exists() else {}
    d[key] = value
    DECISIONS.write_text(json.dumps(d, indent=2))


def run_dir(name: str) -> Path:
    return RUNS / name


def is_done(name: str, config: str, overrides: list[str]) -> bool:
    hist = run_dir(name) / "history.json"
    if not hist.exists() or not (run_dir(name) / "best.pt").exists():
        return False
    epochs = load_config(config, overrides)["train"]["epochs"]
    return len(json.loads(hist.read_text())) >= epochs


def best_val(name: str) -> float:
    hist = json.loads((run_dir(name) / "history.json").read_text())
    return max(r["val_miou"] for r in hist if "val_miou" in r)


def train(name: str, config: str, overrides: list[str] | None = None) -> float:
    overrides = [f"run_name={name}", *(overrides or [])]
    if is_done(name, config, overrides):
        log(f"skip {name} (done, best val mIoU {best_val(name):.4f})")
        return best_val(name)
    log(f"start {name}: {config} {' '.join(overrides)}")
    t0 = time.time()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        with (RUNS / f"{name}.log").open("a") as f:
            rc = subprocess.call([sys.executable, "-W", "ignore", "-m", "src.train", "--config", config,
                                  "--set", *overrides], stdout=f, stderr=subprocess.STDOUT)
        if rc == 0:
            break
        tail = (RUNS / f"{name}.log").read_text()[-3000:]
        # macOS GPU watchdog ("Impacting Interactivity") kills long MPS command buffers at random;
        # train.py resumes from last.pt, so retrying loses at most one epoch.
        if any(s in tail for s in TRANSIENT_ERRORS) and attempt < MAX_ATTEMPTS:
            log(f"retry {name} after transient GPU error (attempt {attempt}/{MAX_ATTEMPTS})")
            time.sleep(30)
            continue
        log(f"FAILED {name} (exit {rc}); see runs/{name}.log")
        raise SystemExit(rc)
    v = best_val(name)
    (run_dir(name) / "last.pt").unlink(missing_ok=True)  # only needed for resume; saves disk
    log(f"done {name} in {(time.time() - t0) / 3600:.2f} h, best val mIoU {v:.4f}")
    return v


def main():
    RUNS.mkdir(exist_ok=True)

    # 1. Sanity check
    train("e0_overfit", "configs/e0_overfit.yaml")
    e0 = json.loads((run_dir("e0_overfit") / "history.json").read_text())[-1]["val_miou"]
    record("e0_final_train_subset_miou", e0)
    if e0 < E0_MIN_MIOU:
        log(f"ABORT: E0 overfit mIoU {e0:.3f} < {E0_MIN_MIOU}; pipeline is likely broken")
        raise SystemExit(1)

    # 2. Architecture / encoder comparison
    res = {
        "e1_unet_r18_ce": train("e1_unet_r18_ce", "configs/e1_unet_r18_ce.yaml"),
        "e2_unet_r50_ce": train("e2_unet_r50_ce", "configs/e2_unet_r50_ce.yaml"),
        "e3_dlv3p_r50_ce": train("e3_dlv3p_r50_ce", "configs/e3_dlv3p_r50_ce.yaml"),
    }

    # 3. Best architecture (E2 vs E3, same encoder)
    arch_run = max(["e2_unet_r50_ce", "e3_dlv3p_r50_ce"], key=res.get)
    model = load_config(f"configs/{arch_run}.yaml")["model"]
    Path("configs/best_arch.yaml").write_text(
        f"# Written by scripts/run_all.py: {arch_run} won E2 vs E3 on val mIoU "
        f"({res['e2_unet_r50_ce']:.4f} vs {res['e3_dlv3p_r50_ce']:.4f}).\n"
        + yaml.safe_dump({"base": "base.yaml", "model": {"arch": model["arch"], "encoder": model["encoder"]}},
                         sort_keys=False))
    record("best_arch", {"run": arch_run, "model": model, "val": {k: res[k] for k in ("e2_unet_r50_ce", "e3_dlv3p_r50_ce")}})
    log(f"best architecture: {arch_run} ({model['arch']}/{model['encoder']})")

    # 4. Loss comparison
    for name in ("e4_weighted_ce", "e5_dice", "e6_focal", "e7_ce_dice"):
        res[name] = train(name, f"configs/{name}.yaml")

    # 5. Best loss (CE result = the architecture winner)
    loss_runs = {arch_run: "ce", "e4_weighted_ce": "weighted_ce", "e5_dice": "dice",
                 "e6_focal": "focal", "e7_ce_dice": "ce_dice"}
    best_loss_run = max(loss_runs, key=res.get)
    best_loss_cfg = load_config(f"configs/{best_loss_run}.yaml")["loss"]
    record("best_loss", {"run": best_loss_run, "loss": best_loss_cfg, "val": {k: res[k] for k in loss_runs}})
    log(f"best loss: {best_loss_cfg['name']} ({best_loss_run})")

    # 6. Pretraining ablation with the best loss
    loss_ov = [f"loss.{k}={json.dumps(v) if not isinstance(v, str) else v}" for k, v in best_loss_cfg.items()
               if v is not None]
    res["e8_no_pretrain"] = train("e8_no_pretrain", "configs/e8_no_pretrain.yaml", loss_ov)

    # 7. Seeds for the top-2 configs among E1-E7
    candidates = {k: v for k, v in res.items() if k != "e8_no_pretrain"}
    top2 = sorted(candidates, key=candidates.get, reverse=True)[:2]
    record("seed_runs", {"configs": top2, "seeds": [1, 2]})
    log(f"seed runs for: {top2}")
    for base_name in top2:
        for seed in (1, 2):
            res[f"{base_name}_s{seed}"] = train(f"{base_name}_s{seed}", f"configs/{base_name}.yaml", [f"seed={seed}"])

    record("best_val_miou_all_runs", res)

    # 8. Val evaluation of every run
    for name in res:
        out = Path("results") / name / "val" / "summary.json"
        if out.exists():
            continue
        log(f"evaluate {name} on val")
        subprocess.check_call([sys.executable, "-m", "src.evaluate", "--run", str(run_dir(name)), "--split", "val"],
                              stdout=subprocess.DEVNULL)
    log("ALL TRAINING DONE")


if __name__ == "__main__":
    main()
