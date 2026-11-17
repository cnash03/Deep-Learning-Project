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
