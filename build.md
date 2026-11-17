# Build Plan: Reading the Trail (Off-Road Terrain Segmentation)

This is the execution plan for the project in the [proposal](<Off-Road Terrain Segmentation — Project Proposal.pdf>). It follows the deliverables in [Requirements.md](Requirements.md). Weeks count from proposal approval.

**Roles**
- **Callen Nash** (model and documentation lead): models, training, loss experiments, tuning, experiment tracking, paper and video draft.
- **Xander Navojosky** (data and evaluation lead): data download, preprocessing, augmentation, EDA, metrics, error analysis, figures.

---

## 1. Repo setup (Week 1, both members, about day 1)

### Directory structure

```
configs/            # one YAML per experiment (model, encoder, loss, lr, seed, res)
src/
  data/             # rugd.py, rellis.py, colormaps.py, transforms.py, class_map.py
  models/build.py   # SMP factory: arch + encoder + pretrained flag
  losses.py         # CE, weighted CE, Dice, Focal, CE+Dice
  metrics.py        # confusion-matrix-based IoU, mIoU, pixel acc
  train.py          # python -m src.train --config configs/xxx.yaml
  evaluate.py       # val/test eval, saves per-class CSV + confusion matrix
scripts/            # download_rugd.sh, download_rellis.sh, compute_class_weights.py
notebooks/          # 01_eda.ipynb, colab_runner.ipynb (thin wrapper calling src/)
results/            # CSVs + figures (committed); checkpoints go to Drive/W&B, not git
paper/
video/
requirements.txt
README.md
```

### Conventions
- **Config-driven runs.** Every experiment is a YAML file in `configs/`, so no hyperparameters are hard-coded in notebooks.
- **Notebooks only drive the code.** `colab_runner.ipynb` clones the repo and calls `python -m src.train`. Experiment logic lives in `src/`.
- **Experiment tracking.** Use Weights & Biases (W&B). Log for every run:
  - the full config
  - the git commit hash
  - the seed
  - per-epoch val mIoU
  - per-class IoU
- **Pinned dependencies.** `requirements.txt` pins the versions of torch, segmentation_models_pytorch, albumentations and wandb.
- **Checkpoints** go to Google Drive or W&B artifacts, never to git.
- **Determinism.** Set seeds for Python, NumPy and torch. Record the seed in the config.

---

## 2. Phase plan

### Week 1: Data and EDA (lead: Xander)
- [ ] Download RUGD.
- [ ] Download the RELLIS-3D images and labels. Skip the LiDAR scans and ROS bags. Do this now, even though RELLIS is the stretch goal, because Google Drive download limits can block it later.
- [ ] Decode the color-coded label PNGs into integer class-index masks. Add a unit test that round-trips sample masks and checks that no pixels are left unmapped.
- [ ] Check the official split. List the video sequences in each of train, val and test, and confirm:
  - no sequence appears in more than one split
  - the counts are 4,779 / 733 / 1,924
- [ ] EDA figures (these go in the paper):
  - [ ] class pixel-frequency bar chart, on a log scale
  - [ ] sample image and mask pairs
  - [ ] per-sequence class presence
- [ ] Compute class weights (inverse or median-frequency) from the **training split only**.
- [ ] Decide how to handle void/ignore pixels (`ignore_index`) and classes that are nearly absent. Record the decision for the Methods section.

### Week 2: Baseline (lead: Callen)
- [ ] **E0 sanity check:** overfit about 50 images. The loss should approach 0, which shows the pipeline works end to end.
- [ ] **E1 baseline:** U-Net with a ResNet-18 encoder and plain cross-entropy (CE). Train at 384² first, then at 512².
  - AdamW optimizer with a cosine learning-rate schedule
  - automatic mixed precision (AMP)
  - a checkpoint every epoch
  - resume-from-checkpoint support, for Colab timeouts
- [ ] Build `evaluate.py`, which saves per-class IoU, mIoU and the confusion matrix to CSV.
- **Milestone:** baseline val mIoU recorded and the pipeline frozen.

### Weeks 3–4: Main experiments (lead: Callen; Xander tunes augmentation)
Change one variable per run, and do all model selection on the **val** split.

| ID | Architecture | Encoder | Loss | Question it answers |
|---|---|---|---|---|
| E1 | U-Net | ResNet-18 | CE | Baseline |
| E2 | U-Net | ResNet-50 | CE | Effect of the encoder alone |
| E3 | DeepLabV3+ | ResNet-50 | CE | Architecture comparison against E2 with the same encoder |
| E4 | best of E2/E3 | ResNet-50 | Weighted CE | Imbalance handling |
| E5 | best of E2/E3 | ResNet-50 | Dice | Imbalance handling |
| E6 | best of E2/E3 | ResNet-50 | Focal | Imbalance handling |
| E7 | best of E2/E3 | ResNet-50 | CE + Dice | Combined loss |
| E8 | best config | ResNet-50, random init | best loss | How much ImageNet pretraining helps |
| E9 | best 2 configs | — | — | 3 seeds each, to report mean ± std |

- [ ] That is about 13–15 runs, which fits the compute budget. If time runs short, run E8 on a shorter schedule.
- [ ] Optional: an augmentation ablation on the baseline, with and without color jitter and scale.
- **Milestone (end of week 4):** freeze the final model. No new training after this point except the stretch goal.

### Week 5: Evaluation, error analysis and stretch goal (lead: Xander; Callen runs RELLIS-3D)
- [ ] Run the **test** set once on the final configs. Report mIoU, per-class IoU and pixel accuracy.
- [ ] Report traversability-group IoU over 6 shared classes: smooth ground, rough ground, vegetation, water, obstacle, sky.
- [ ] Error analysis:
  - [ ] normalized confusion matrix, the likely headline figure
  - [ ] gallery of the worst frames, showing image / ground truth / prediction / error map
  - [ ] mIoU with and without a thin boundary band, to see how much error comes from ambiguous labels
  - [ ] per-sequence mIoU, to find the hardest conditions such as blur, shade or water
  - [ ] rare-class IoU (water, rock, log) compared across the loss functions
- [ ] **Stretch goal:** score the RUGD-trained model on the RELLIS-3D test split with no retraining, using the shared class map, and report the performance drop. Optionally fine-tune on a small RELLIS subset. This is the first thing cut if the schedule slips.

### Week 6: Paper, video and showcase (Callen writes; Xander does figures and demo)
- [ ] Run inference on our own trail photos for the demo. Strip EXIF and GPS data, and avoid people and license plates in frame.
- [ ] Final paper with all required sections, plus the ethics and risk reflection.
- [ ] Record a 6–8 minute video in which both members speak.
- [ ] Post the video to the course forum and answer at least 2 classmate questions.
- [ ] Finish the README: setup, download scripts, train and evaluate commands for each config, expected numbers, and links to W&B and the best checkpoint.

---

## 3. Decision points and cut lines
- **End of week 2:** if the baseline isn't running, use 384² and ResNet-18 for all runs.
- **End of week 4:** if behind, cut the stretch goal first, then E8. Keep E9 (multiple seeds), because it carries the rigor score.
- **Week 5:** never tune on the test split. If a bug turns up after the test evaluation, fix it, re-validate on val, and disclose it in the paper.
