# Reading the Trail: Off-Road Terrain Segmentation

Semantic segmentation of off-road camera images. Every pixel gets a terrain or object class (dirt, gravel, grass, water, rock, log, …). We compare U-Net and DeepLabV3+ with ImageNet-pretrained encoders, and test loss functions built for class imbalance, on the [RUGD](http://rugd.vision) dataset. As a stretch goal, we test cross-dataset generalization to [RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D).

**Team:** Callen Nash (models & documentation), Xander Navojosky (data & evaluation)

- Project plan and experiment matrix: [build.md](build.md)
- Course requirements: [Requirements.md](Requirements.md)

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q          # unit tests + a CPU smoke test on fake data
```

On Colab, use [notebooks/colab_runner.ipynb](notebooks/colab_runner.ipynb).

## Data

```bash
bash scripts/download_rugd.sh data/rugd            # ~5 GB
python -m scripts.check_splits --root data/rugd    # expects 4,779 / 733 / 1,924 frames
python -m scripts.compute_class_weights --root data/rugd   # needed for E4 (weighted CE)
```

RELLIS-3D (stretch goal) is a manual download: see [scripts/download_rellis.md](scripts/download_rellis.md).

The data root defaults to `data/rugd`. Override it with `DATA_ROOT=/path` or `--set data.root=/path`.

## Training

Each experiment is one YAML file in [configs/](configs/), and inherits from `base.yaml`:

```bash
python -m src.train --config configs/e0_overfit.yaml          # sanity check first
python -m src.train --config configs/e1_unet_r18_ce.yaml
python -m src.train --config configs/e1_unet_r18_ce.yaml --set train.epochs=5 wandb.enabled=true
```

Each run writes the following to `runs/<config-name>/`:
- `config.yaml`: the resolved config plus the git commit hash
- `history.json`: per-epoch loss, val mIoU and per-class IoU
- `last.pt`: written every epoch, used for resume
- `best.pt`: the epoch with the best val mIoU

Re-running the same command resumes from `last.pt`.

## Evaluation

```bash
python -m src.evaluate --run runs/e1_unet_r18_ce --split val
python -m src.evaluate --run runs/<final> --split test --confirm-test          # once, at the end
python -m src.evaluate --run runs/<final> --split test --confirm-test --cross-rellis data/rellis
```

Outputs go to `results/<run>/<split>/`:
- `summary.json`: mIoU, pixel accuracy, 6-group traversability mIoU
- `per_class_iou.csv`
- `group_iou.csv`
- `confusion.npy`

## Experiments

| ID | Model | Loss | Val mIoU | Test mIoU |
|---|---|---|---|---|
| E1 | U-Net / ResNet-18 | CE | | |
| E2 | U-Net / ResNet-50 | CE | | |
| E3 | DeepLabV3+ / ResNet-50 | CE | | |
| E4 | best arch | Weighted CE | | |
| E5 | best arch | Dice | | |
| E6 | best arch | Focal | | |
| E7 | best arch | CE + Dice | | |
| E8 | best arch, no pretraining | best loss | | |
| E9 | best 2 configs × 3 seeds | | mean ± std | |

## Conventions

- **Void pixels** (RUGD class 0) are ignored in the loss and the metrics, and are never predicted.
- mIoU averages only over classes present in ground truth or predictions.
- **Splits** follow RUGD's video sequences (see `RUGD_SPLITS` in [src/data/rugd.py](src/data/rugd.py)), so consecutive near-identical frames never appear in more than one split.
- **Model selection** uses only the val split. The test split requires `--confirm-test`.

## Repository layout

```
configs/    experiment YAMLs
src/        data/ (datasets, color maps, transforms, class groups), models/, losses, metrics, train, evaluate
scripts/    data download, split check, class weights
notebooks/  Colab runner (EDA notebooks go here too)
tests/      pytest suite
results/    small CSV/JSON results and figures (committed)
paper/      final report
video/      presentation materials
```

## Analysis and figures

Error-analysis extras are opt-in flags on `src.evaluate` (computed in the same pass over the data). Plain `--run X --split val` behaves as before.

```bash
# Per-image / per-sequence metrics, boundary-band metrics at K = 1, 3, 5 px, and 12 saved prediction masks
python -m src.evaluate --run runs/e3_dlv3p_r50_ce --split val --per-image --boundary-band 1 3 5 --save-preds 12

# Tables and figures from every run / result found (missing pieces are skipped)
python -m scripts.analyze
python -m scripts.analyze --cm-run e3_dlv3p_r50_ce --cm-split val     # pick the confusion-matrix run

# Qualitative figures (CPU inference on a few frames)
DEVICE=cpu python -m scripts.gallery --run runs/e3_dlv3p_r50_ce --split val                 # best / median / worst frames
DEVICE=cpu python -m scripts.gallery --run runs/e3_dlv3p_r50_ce --compare runs/e7_ce_dice   # two runs on water/log/rock frames
```

Extra outputs of `src.evaluate` in `results/<run>/<split>/`:
- `--per-image`: `per_image.csv` (file, sequence, per-image mIoU over the classes in that frame's ground truth, pixel accuracy, GT classes) and `per_sequence.csv` (mIoU from the summed confusion matrix of each sequence, plus the mean per-image mIoU for comparison).
- `--boundary-band K ...`: `summary.json["boundary_band"]["k=K"]` gives metrics on the *interior* (GT pixels more than K px from a label change) and on the *band* only, the band's share of pixels, and the share of all errors that fall in the band. `boundary_band_iou.csv` has the same split per class. The band is computed on the resized eval masks (`data.img_size`). A transition to void counts as a boundary.
- `--save-preds N` (or `--pred-indices i j ...`): `preds/<sequence>/<frame>.png`, uint8 class-index masks.

`scripts.analyze` writes:
- `results/tables/main_results.{csv,md,tex}`: one row per config. Seed reruns (`X_s1`, `X_s2`) are pooled with `X` as mean ± sample std. Columns are best val mIoU from `history.json` plus val/test mIoU, pixel accuracy and group mIoU from `summary.json` when they exist. The `.tex` file is a `booktabs` table.
- `results/tables/per_class_iou_losses.{csv,tex}`, `per_sequence_<split>.{csv,md}`, `boundary_band.{csv,md,tex}`
- `results/figures/` (PDF + PNG): `training_curves`, `per_class_iou_losses` (CE winner vs. E4–E7, rare/safety classes shaded), `arch_comparison` (E1/E2/E3/E8 with seed error bars), `confusion_<run>_<split>` (row-normalized, classes grouped by traversability) and `group_confusion_<run>_<split>`.

`scripts.gallery` writes `results/figures/gallery_<run>_<split>` and `compare_<run>_vs_<run2>_<split>`. Comparison frames are chosen from ground truth alone (largest rare-class share, at most one frame per 30 consecutive frames), so no model gets to pick its best frames. The gallery reads `per_image.csv`, so run `src.evaluate --per-image` first. The test split needs `--confirm-test` in both scripts.
