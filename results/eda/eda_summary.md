# RUGD exploratory data analysis: summary

Reproduce with `DEVICE=cpu python -m scripts.eda --root data/rugd` (about 1 min on 2 CPU workers, or a few seconds when the
cache is used; `--recount` rebuilds the cache). Every number below comes from `results/eda/eda_numbers.json`, which the
script writes.

**How the numbers are counted.** Each label PNG is decoded with `RUGDDataset(root, split, ignore_void=False).load_mask(i)`
and counted with `np.bincount`. "Labeled pixels" means all pixels except void (class 0) and unmapped colors. A class's
**share** is its pixels divided by the labeled pixels of the same split (or sequence). The 24 non-void classes are the
classes we evaluate. The training-split counts match `results/class_freq_rugd.csv` exactly for all 24 classes. The only
difference is that the earlier CSV reports 0 void pixels because it counted after void had been mapped to ignore.

## Dataset structure

| | train | val | test |
|---|---|---|---|
| Sequences | 12 | 2 | 4 |
| Frames | 4,779 | 733 | 1,924 |
| Labeled pixels | 1,807,979,701 | 277,205,753 | 727,960,520 |
| Classes present (of 24) | 23 | 20 | 23 |
| Classes absent | bridge | sand, rock-bed, bicycle, bridge | bicycle |
| tree + grass share | 63.18% | 64.43% | 55.86% |
| Top-5 classes share | 91.68% (tree, grass, mulch, gravel, sky) | 91.99% (tree, grass, mulch, sky, gravel) | 82.50% (tree, grass, bush, rock-bed, sky) |
| Classes < 1% / < 0.1% of pixels (present only) | 15 / 5 | 13 / 6 | 12 / 6 |
| Safety classes combined (water, log, rock, rock-bed, dirt, sand) | 1.11% | 1.17% | 10.29% |

- **Total:** 7,436 annotated frames in 18 sequences. No sequence appears in two splits (`RUGD_SPLITS`).
- **Resolution:** every image and every mask is **688 × 550** px. No image and mask differ in size.
- **Temporal sampling:** annotated frames are every 5th video frame. The stride is exactly 5 in all sequences, except
  20 gaps of 4 in `trail`. Neighboring labeled frames are therefore highly correlated, so a frame-level random split
  would leak near-duplicate frames. This is why the split is made by sequence.
- **Frames per sequence:**
  - train: trail-4 758, trail 693, park-2 656, trail-3 580, trail-6 439, trail-11 438, trail-12 352, trail-15 324,
    trail-14 314, village 117, trail-9 59, trail-10 49
  - val: trail-5 376, park-8 357
  - test: creek 836, park-1 627, trail-7 289, trail-13 172
- **Void:** 0.0226% of all pixels over all splits (train 0.0218%, val 0.0582%, test 0.0111%). 1,775 of 7,436 frames
  have any void, and the median is only 17 void pixels per such frame. The largest case is `park-8_01116.png`, with
  41,243 px: a vertical unlabeled strip about 75 px wide along the left edge. **No mask contains a color outside the
  25-entry colormap** (0 unmapped pixels), so decoding is lossless.

## Class imbalance

- **Imbalance ratio** (pixels of the most frequent present class ÷ pixels of the least frequent present class, per split):
  - **train: tree / sign = 711,145,352 / 111,437 = 6,382 : 1**
  - val: tree / person = 119,274,410 / 4,019 = 29,678 : 1
  - test: tree / person = 259,782,138 / 4,487 = 57,897 : 1
- **Train shares of the rare safety-relevant classes:**
  - log 0.510%
  - sand 0.184%
  - dirt 0.132%
  - water 0.106%
  - rock 0.104%
  - rock-bed 0.070%

  Together they make up 1.11% of the training split. RUGD has no "mud" class; dirt and sand are the nearest
  deformable-ground labels, which is why they are in the safety group (highlighted in the figures).
- **Train shares of the rarest other classes:**
  - picnic-table 0.043%
  - person 0.0118%
  - bicycle 0.0073%
  - sign 0.0062%
- **How many training frames contain each safety class** (any pixel of it):
  - log 35.5%
  - water 7.5%
  - rock 5.6%
  - sand 4.3%
  - dirt 3.5%
  - rock-bed 1.0%

  Water appears in only about 1 in 13 training frames.

## Distribution shift between splits (the most important finding)

- **`bridge` has 0 train pixels and 4,634,144 test pixels (0.64% of the test split).** All of them come from `creek`,
  and they appear in 19.0% of test frames. No model trained on the train split can predict this class. Its test IoU is
  0 by construction, so mIoU over 24 classes is capped below 1 for every method. We exclude bridge from the reported
  mIoU and state this in the paper.
- **The val split lacks 4 classes:** sand, rock-bed, bicycle and bridge. Val mIoU therefore averages over 20 classes,
  and model selection on val carries no signal for rock-bed or sand.
- **The test split is much more hazard-heavy than train.**
  - Rock-bed: **7.96%** of test pixels vs 0.070% of train pixels (114×).
  - Water: **0.935%** vs 0.106% (8.9×).
  - Dirt: 0.299% vs 0.132% (2.3×).
  - Bush: 11.57% vs 2.37% (4.9×).
  - The safety group as a whole: 10.29% of test vs 1.11% of train.
- **One sequence drives most of this:** `creek` (836 frames, 43.5% of the test frames).
  - It contains 97.9% of all rock-bed pixels in the dataset, 76.3% of all water, 39.9% of all dirt and 100% of bridge.
  - Rock-bed is present in only 3 training sequences: trail-6, trail-4 and trail. Those hold 1.47%, 0.65% and 0.02% of
    all rock-bed pixels.
  - Expect test performance on water and rock-bed to reflect generalization to a new creek-bed scene more than
    within-domain accuracy.
- **Total-variation distance between split class distributions** (½·Σ|p − q| over the 24 class shares):
  - train–val: 0.114
  - train–test: 0.215
  - val–test: 0.246

  So **val is a closer proxy for train than for test**, and val scores will likely overestimate test scores.
- **Other shifts:**
  - Gravel: 8.97% train, 5.50% val, 2.37% test.
  - Asphalt: 1.58% train, 3.83% val, 0.08% test.
  - Sand is absent from val but makes up 0.18% of test.

## Per-sequence composition

- Tree is the largest class in **17 of 18 sequences**, at 29.7–46.0% of each sequence's labeled pixels. The exception is
  `village` (117 frames), an urban-edge sequence where sky is largest (27.6%), followed by grass (21.7%). Village has
  no mulch or sand, and it has the most building, vehicle and container pixels per frame.
- Grass or mulch is always among the top 2 classes, except in `creek` (tree 35.3%, then bush 24.0%).
- Class diversity (number of non-void classes present):
  - from 8 in trail-10 and 11 in trail-9, the two shortest sequences
  - to 21 in trail-4 and park-2
- Water is concentrated: 76.3% in creek, then trail-15 (6.2%) and trail-11 (5.7%). Several sequences have none: trail,
  trail-10, park-8, park-1 and trail-7.
- Log is the most evenly spread safety class. It is present in all 18 sequences, and the largest sources are trail-7
  (18.7%), trail-4 (17.6%) and park-1 (13.3%).

## Qualitative observations (from inspecting images and masks)

To flag frames, we computed the variance of the Laplacian (blur), the fraction of pixels ≥ 250 (over-exposure) and the
fraction ≤ 10 (under-exposure) on every 15th frame. We then viewed the flagged frames and the sample-grid frames.
This screening was a one-off and is not part of `scripts/eda.py`.

1. **Coarse, polygonal annotation.**
   - Labels are drawn as polygons with straight edges. In `trail-5/trail-5_00601.png`, the mulch, gravel and grass
     regions are blocky shapes that do not follow the real ground-cover boundaries.
   - Canopy-sky boundaries are approximate everywhere. In `park-8/park-8_01116.png`, the sky gaps in the canopy are
     hand-drawn blobs.
   - Expect much of the error to sit in a band near boundaries. This supports the planned "mIoU without a boundary
     band" analysis.
2. **Ambiguous ground-material labels.**
   - `village/village_00408.png`: sandy-looking ground is labeled **gravel**.
   - `park-1/park-1_03021.png`: a gravel-strewn dirt lot is labeled **concrete**, and a pipe lying on the ground is not
     labeled separately.
   - `creek/creek_00901.png`: a pebble bar with grass tufts is labeled **rock-bed** as a single region.
   - The dirt / sand / gravel / mulch / concrete distinctions depend on annotator judgment, and that will show up as
     confusion between these classes.
3. **Thin structures.**
   - In `trail-11/trail-11_01176.png`, overhead **power lines are labeled tree**, since RUGD has no wire class.
   - Thin sticks are labeled **log**, such as the diagonal branch in `trail-5/trail-5_00601.png` and in
     `trail-3/trail-3_00991.png`. These segments are a few pixels wide, so log IoU will be sensitive to boundary
     precision.
4. **Lighting and exposure.**
   - Strong over-exposure: in `trail-11/trail-11_01176.png`, 32% of pixels are ≥ 250 (washed-out sky) while the
     ground is under-exposed. In `trail-14/trail-14_00341.png` and `trail-11/trail-11_01251.png`, 35% of pixels are
     saturated.
   - Hard cast shadows: the van's shadow in `village/village_00408.png`.
   - Dappled canopy light in `trail-5/trail-5_00601.png`.
   - Dark water with reflections in `creek/creek_00901.png`.
5. **Optics and blur.**
   - All frames show **strong vignetting** with a purple/blue cast in the corners, from a wide-angle lens. Examples
     are `park-1/park-1_03021.png` and `trail-11/trail-11_01176.png`.
   - The camera is mounted low, so the near-field ground often shows motion and defocus blur. The bottom of
     `trail-5/trail-5_00601.png` is an example. The lowest-sharpness frames in the screening were
     `park-1/park-1_03021.png`, `trail-11/trail-11_01176.png` and `trail-13/trail-13_00826.png`.
6. **Palette caveat for figures.** In the official colormap, `concrete` (101,101,11) and `rock-bed` (102,102,0) are
   nearly the same olive color, and `mulch` (153,76,0) / `dirt` (108,64,20) and `tree` / `bicycle` are also close.
   Readers of mask figures cannot tell concrete from rock-bed. Captions should say so, or prediction figures could
   use a modified palette.

**Risk and ethics notes.**
- People are rare (0.0118% of train pixels, in 2.6% of train frames). Person IoU will be poorly estimated and should
  not be read as evidence of pedestrian safety.
- Test hazards (water, rock-bed) come almost entirely from a single sequence, so rare-class results have high
  variance and may not generalize.
- The frames show vehicles and buildings, but no legible faces or license plates were seen in the inspected frames.

## Figures and tables

| File | Contents | Intended width |
|---|---|---|
| `results/figures/class_frequency.{pdf,png}` | Train share as bars (safety classes highlighted), val and test shares as markers, log x-axis, absent splits annotated | **IEEE single column (3.5 in)** |
| `results/figures/sample_grid.{pdf,png}` | 6 frames (trail-3_00991, village_00408, trail-15_00316 from train; trail-5_00601 from val; creek_00026, park-1_02081 from test), image above mask, legend of the 21 classes shown | **Double column (7.16 in)** |
| `results/figures/sequence_class_heatmap.{pdf,png}` | 18 sequences grouped by split, plus aggregate rows for each split, × 24 classes; cell = % of the sequence's labeled pixels (log color), dot = absent | **Double column (7.16 in)** |
| `results/eda/split_summary.{csv,tex}` | Per-split sequences, frames, shares of the top 6 classes, other, safety total, absent classes (`table*`, booktabs) | Double column |
| `results/eda/class_pixels_by_sequence.csv` | Long format: split, sequence, class, pixels, frames_present, n_frames | — |
| `results/eda/class_pixels_by_frame.csv` | Per-frame pixel counts (the cache) | — |

Sample-grid frames were chosen automatically. In each chosen sequence, the script picks the frame with the most classes
at ≥ 0.5% of its pixels, adding +2 for each safety class at that level.
