"""Turn finished training runs into every reported number, table and figure.

Run once, after ALL training is done (this is the only place the test split is touched):

    python -m scripts.finalize --runs reading-the-trail/runs --colab-results reading-the-trail/results

Steps (each skipped if its output already exists, so it is safe to re-run):
  1. Copy decisions.json from the Colab results folder (model-selection record).
  2. Evaluate every run (except E0) on val and test: per-image, per-sequence, boundary band 1/3/5 px.
  3. Zero-shot RELLIS-3D evaluation of the CE architecture winner and the best-loss model.
  4. Tables and figures (scripts/analyze.py), confusion matrices on test for the final model.
  5. Qualitative galleries on test (best/median/worst; CE vs best loss on rare-class frames).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def sh(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.check_call([sys.executable, "-W", "ignore", *args])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="training runs directory (e.g. reading-the-trail/runs)")
    ap.add_argument("--colab-results", default=None, help="Colab results dir holding decisions.json")
    ap.add_argument("--results", default="results")
    ap.add_argument("--data-root", default="data/rugd")
    ap.add_argument("--rellis", default="data/rellis")
    ap.add_argument("--final-run", default=None,
                    help="run to treat as the final model (default: decisions.json best_loss run). Use a seed "
                         "replicate of the selected config if the selected run's weights are unavailable.")
    args = ap.parse_args()

    runs, results = Path(args.runs), Path(args.results)
    results.mkdir(exist_ok=True)

    # 1. Decisions made during training (val-only selection)
    if args.colab_results and (Path(args.colab_results) / "decisions.json").exists():
        shutil.copy(Path(args.colab_results) / "decisions.json", results / "decisions.json")
    decisions = json.loads((results / "decisions.json").read_text())
    ce_run = decisions["best_arch"]["run"]
    final_run = args.final_run or decisions["best_loss"]["run"]
    for name in (ce_run, final_run):
        if not (runs / name / "best.pt").exists():
            raise SystemExit(f"{runs / name / 'best.pt'} missing; pass --final-run <seed replicate>")
    print(f"CE architecture winner: {ce_run}; final (best-loss) model: {final_run}")

    # 2. Val + test evaluation of every finished run
    names = sorted(p.name for p in runs.iterdir() if (p / "best.pt").exists() and p.name != "e0_overfit")
    for name in names:
        for split in ("val", "test"):
            if (results / name / split / "summary.json").exists():
                continue
            extra = ["--confirm-test"] if split == "test" else []
            sh("-m", "src.evaluate", "--run", str(runs / name), "--split", split, *extra, "--data-root", args.data_root,
               "--out", str(results), "--per-image", "--boundary-band", "1", "3", "5")

    # 3. Cross-dataset (stretch goal)
    if Path(args.rellis, "split", "test.lst").exists():
        for name in dict.fromkeys([ce_run, final_run]):
            if not (results / name / "test_rellis" / "summary.json").exists():
                sh("-m", "src.evaluate", "--run", str(runs / name), "--split", "test", "--confirm-test",
                   "--cross-rellis", args.rellis, "--out", str(results))
    else:
        print("RELLIS-3D not found; skipping cross-dataset evaluation")

    # 4. Tables and figures
    sh("-m", "scripts.analyze", "--runs", str(runs), "--results", str(results), "--cm-run", final_run, "--cm-split", "test")

    # 5. Qualitative galleries
    sh("-m", "scripts.gallery", "--run", str(runs / final_run), "--split", "test", "--confirm-test",
       "--results", str(results), "--data-root", args.data_root)
    if ce_run != final_run:
        sh("-m", "scripts.gallery", "--run", str(runs / ce_run), "--compare", str(runs / final_run), "--split", "test",
           "--confirm-test", "--results", str(results), "--data-root", args.data_root)
    print("FINALIZE DONE")


if __name__ == "__main__":
    main()
