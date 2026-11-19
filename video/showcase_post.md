# Peer Showcase post

Fill in the video link, then paste everything below the line into the course forum.

---

**Title:** Reading the Trail: segmenting off-road terrain when the hazards are the rarest pixels

**Team:** Callen Nash and Xander Navojosky

We trained networks to label every pixel of off-road camera images from the RUGD dataset (24 classes: grass, gravel, water, logs, rock, and more), comparing U-Net and DeepLabV3+ on ImageNet-pretrained ResNet encoders. The catch is imbalance. The safety-relevant classes (water, log, rock, rock-bed, dirt, sand) make up only about 1% of training pixels, so we compared four imbalance-aware losses against plain cross-entropy. Every model choice was made automatically on validation, and the test set was used once. We also took the final model, with no retraining, to a second dataset, RELLIS-3D, recorded with a different vehicle in a different state.

**Key finding:** DeepLabV3+/ResNet-50 with cross-entropy + Dice reaches 37.7% test mIoU (60.0% over six traversability groups). ImageNet pretraining was the biggest single factor (about 6 points), Dice terms helped rare classes on validation, and one unseen creek scene caused 80% of test errors by labeling shallow water as drivable ground. Zero-shot on RELLIS-3D, group mIoU drops from 60.0% to 43.4%.

We also found that the official test split contains a class (*bridge*) with zero training pixels, and that validation lacks four classes. Ask us how we handled that.

- Video (7 min): [add video link]
- Code, configs, and paper: https://github.com/cnash03/Deep-Learning-Project

Questions and critiques are welcome. We'll answer here.
