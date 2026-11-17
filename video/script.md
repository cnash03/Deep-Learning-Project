# Reading the Trail: video script

Target length **7:00** (hard limits 6:00 to 8:00). The timings assume about 140 words per minute. Slide numbers match `video/slides.pdf`.

- **Callen** (model and documentation lead) presents the motivation, the architecture, the losses and protocol, the main results, and the lessons learned.
- **Xander** (data and evaluation lead) presents the data, the imbalance, the distribution shift, the error analysis, RELLIS-3D, and the data-integrity half of the ethics slide.
- Both speak on the title and closing slides.

Anything in **[TBD: …]** must be filled in from `results/` before recording. The slides mark the same spots with a red **TBD**.

| # | Slide | Speaker | Time | Running total |
|---|---|---|---|---|
| 1 | Title | Callen, then Xander | 0:15 | 0:15 |
| 2 | Why off-road perception is hard | Callen | 0:30 | 0:45 |
| 3 | Task and data | Xander | 0:30 | 1:15 |
| 4 | The imbalance problem | Xander | 0:30 | 1:45 |
| 5 | Distribution shift | Xander | 0:30 | 2:15 |
| 6 | Approach and architecture | Callen | 0:30 | 2:45 |
| 7 | Five losses | Callen | 0:20 | 3:05 |
| 8 | Experimental protocol | Callen | 0:30 | 3:35 |
| 9 | Results | Callen | 0:30 | 4:05 |
| 10 | Losses and rare classes | Callen | 0:30 | 4:35 |
| 11 | Error analysis | Xander | 0:40 | 5:15 |
| 12 | RELLIS-3D | Xander | 0:25 | 5:40 |
| 13 | Risk and ethics | Xander, then Callen | 0:50 | 6:30 |
| 14 | Lessons and next steps | Callen | 0:25 | 6:55 |
| 15 | Thank you | Xander, then Callen | 0:10 | 7:05 |

The script has about 910 words, which is about 6:30 of speech at 140 words per minute. Pauses and slide changes bring it to about 7:00. Callen speaks for about 3:40 and Xander for about 3:00, plus the shared intro and close. If a take runs long, shorten the sentence after each number; don't speed up.

---

## 1. Title (0:15)

**Callen:** Hi, I'm Callen Nash, and I led the models and the write-up.

**Xander:** I'm Xander Navojosky, and I led the data and evaluation. Our project is *Reading the Trail*: labeling every pixel of an off-road camera image.

## 2. Why off-road perception is hard (0:30) — Callen

Most self-driving research happens on city streets, where lane lines, curbs and signs give a model a lot of structure. Trails have none of that. Grass, mulch and gravel blend together, the light swings between shade and glare, and the camera shakes. Rescue robots, farm vehicles and rovers still have to tell dirt from water or a log, because one mistake can strand the vehicle.

## 3. Task and data (0:30) — Xander

We treat this as semantic segmentation on RUGD: 7,400 labeled frames from eighteen videos shot by a small ground robot, with twenty-four classes. The top row shows images, the bottom row their labels. Consecutive frames are nearly identical, so we kept the official split *by sequence*: 4,779 training, 733 validation and 1,924 test frames. No near-duplicate frame ends up in both training and test.

## 4. The imbalance problem (0:30) — Xander

The first thing we found is how skewed the data is. Tree and grass cover sixty-three percent of training pixels, and the most common class outnumbers the rarest by more than six thousand to one. The safety classes, shown as orange bars, are water, log, rock, rock-bed, dirt and sand. Together they make up about one percent. A model can score well on average and still miss every hazard.

## 5. Distribution shift (0:30) — Xander

The second finding is that the test split looks different from training. In training, safety classes make up one percent of pixels. In test they make up ten percent, mostly because of one sequence, *creek*, which holds almost all of the dataset's rock-bed. *Bridge* has zero training pixels but 4.6 million test pixels, and validation is missing four classes entirely. So we expected validation scores to overstate test scores.

## 6. Approach and architecture (0:30) — Callen

Here is the model. A 384-pixel image goes into a ResNet encoder pretrained on ImageNet, either ResNet-18 or ResNet-50. We compare two decoders. U-Net, on the left, reuses encoder features at every scale for fine detail. DeepLabV3+, on the right, uses dilated convolutions for wide context, plus one high-resolution skip connection. Both predict a class at every pixel.

## 7. Five losses (0:20) — Callen

To handle the imbalance, we compare plain cross-entropy with four alternatives. Weighted cross-entropy upweights rare classes. Dice optimizes overlap for each class regardless of its size. Focal loss focuses on hard pixels. The fifth combines cross-entropy and Dice.

## 8. Experimental protocol (0:30) — Callen

Each experiment changes one thing: the encoder, the decoder, the loss, or ImageNet pretraining. The two best configurations are then rerun with three seeds to measure noise. A script picks every winner from validation scores alone and logs the decision. The test set is used once, at the end. Every run uses 384 pixels and 25 epochs on a Colab GPU.

## 9. Results (0:30) — Callen

Here is every configuration. The selection script picked **[TBD: best config]**, which reaches **[TBD]** percent mIoU on validation and **[TBD]** on the held-out test set. That gap matches the shift Xander described. On the six traversability groups (ground, vegetation, water, obstacles and so on), it reaches **[TBD]** percent. Seeds vary by about **[TBD]** points. Removing ImageNet pretraining costs **[TBD]** points.

## 10. Losses and rare classes (0:30) — Callen

Now the question we cared about most: do these losses help the rare classes? The shaded region holds the safety classes. Water goes from **[TBD]** to **[TBD]** IoU, log from **[TBD]** to **[TBD]**, and rock from **[TBD]** to **[TBD]**. On the common classes, the cost is **[TBD]**. In short, **[TBD: one-sentence finding, only what the data supports]**.

## 11. Error analysis (0:40) — Xander

So where does the model fail? On the left is the test confusion matrix, with classes grouped by traversability. Most of the confusion is **[TBD: e.g. grass versus bush]**, which stays inside a group and is fairly harmless. The dangerous errors cross groups, like **[TBD: e.g. water predicted as ground]**. On the right are the best, median and worst frames, with errors in red. The worst ones show **[TBD: failure mode]**. Labels are coarse polygons, and **[TBD]** percent of all errors fall within three pixels of a boundary.

## 12. Zero-shot RELLIS-3D (0:25) — Xander

We then ran the RUGD model, with no retraining, on RELLIS-3D. It was recorded in Texas by a different, taller vehicle. With both label sets mapped to six shared groups, group mIoU falls from **[TBD]** to **[TBD]**, and **[TBD: group]** drops the most. RELLIS's *mud* class has no RUGD counterpart, so we ignore it.

## 13. Updated risk and ethics (0:50) — Xander, then Callen

**Xander:** A few data-integrity issues came up. *Bridge* can't be learned, so we leave it out of the headline score but still report it. We report validation and test separately, check the sequence split with a script, and measure the effect of coarse labels. Our compute budget meant 384 pixels and 25 epochs instead of 512 and 50, applied to every run and disclosed.

**Callen:** On bias and safety: this data covers one region, seen by one robot with a camera under twenty-five centimeters off the ground. A missed hazard is the worst error, so this is a research model, not a safety system. We also flag dual use, since RUGD came from an Army lab, and RELLIS is licensed for non-commercial use only.

## 14. Lessons and next steps (0:25) — Callen

What we learned: look at the data first, because our exploration caught the bridge and validation problems before any training. Average scores hide the classes that matter. Automated selection keeps you honest about the test set. **[TBD: one lesson from the results.]** Next steps are full resolution, a validation split that covers every class, video consistency, and calibrated uncertainty.

## 15. Thank you (0:10)

**Xander:** Thanks for watching. All the code is on GitHub.

**Callen:** We'd love your questions in the forum.

---

## Recording tips

**Setup**
- Record as a **Zoom meeting with both of you joined**. One person shares the slides (`slides.pdf` in full-screen presentation mode), and both cameras stay on. Use the *Active speaker* or *Gallery* layout, so both webcam bubbles stay visible next to the slides. This shows on screen that both members contribute. Use *Record to this computer* and enable "Record a separate audio file for each participant", which makes cleanup easier.
- If you record separately: in **OBS**, use a *Display Capture* of the slides plus a *Video Capture Device* webcam source cropped into a corner bubble, with one scene per speaker. In **QuickTime**, use *New Screen Recording* with the camera shown in a floating *New Movie Recording* window placed in a corner.
- Use a quiet room and headphones, so there's no echo from the other person's audio. Put the slides on the screen you face, so your eyes point toward the camera.

**Takes**
- Record **one take per section**, for example slides 1–2, 3–5, 6–10, 11–13 and 14–15. Leave a 2-second pause at the start and end of each take for clean cuts.
- Read the script once aloud with a timer before recording. If a section runs more than 10% over its budget, trim words rather than speeding up.
- Fill every **[TBD]** first, and check each spoken number against the slide.

**Stitching**
- **iMovie** or **QuickTime** (*Edit → Add Clip to End*), **Clipchamp**, or **DaVinci Resolve** all work. Put the clips in slide order and trim the pauses.
- Or use ffmpeg: write `file 'take1.mp4'` lines into `list.txt`, then run `ffmpeg -f concat -safe 0 -i list.txt -c copy presentation.mp4`. This only works if every take uses the same resolution and codec.

**Final check**
- Total length must be between **6:00 and 8:00**. Aim for 6:45 to 7:15 so there's margin.
- Watch the whole thing once. Both voices and both faces should appear, every figure should be readable at 1080p, and no TBD should be visible.
- Export at 1080p MP4 (H.264) and upload. Then add the link to `showcase_post.md`.
