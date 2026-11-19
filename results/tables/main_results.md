| ID | Model | Loss | Pretrain | Seeds | Best val mIoU (hist.) | Val mIoU | Val pixel acc | Val group mIoU | Test mIoU | Test pixel acc | Test group mIoU |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | U-Net / ResNet-18 | CE | ImageNet | 1 | 37.1 | 37.1 | 90.7 | 75.6 | 34.2 | 76.0 | 58.6 |
| E2 | U-Net / ResNet-50 | CE | ImageNet | 1 | 37.7 | 37.7 | 90.8 | 76.8 | 34.0 | 76.1 | 60.9 |
| E3 | DeepLabV3+ / ResNet-50 | CE | ImageNet | 1 | 43.3 | 43.3 | 90.9 | 76.6 | 37.0 | 77.4 | 62.7 |
| E4 | DeepLabV3+ / ResNet-50 | Weighted CE | ImageNet | 1 | 33.7 | – | – | – | – | – | – |
| E5 | DeepLabV3+ / ResNet-50 | Dice | ImageNet | 1 | 42.3 | – | – | – | – | – | – |
| E6 | DeepLabV3+ / ResNet-50 | Focal | ImageNet | 1 | 41.1 | – | – | – | – | – | – |
| E7 | DeepLabV3+ / ResNet-50 | CE + Dice | ImageNet | 2 | 44.7 ± 0.7 | 45.2 | 90.9 | 77.4 | 37.7 | 76.5 | 60.0 |
| E8 | DeepLabV3+ / ResNet-50 | CE + Dice | none | 1 | 38.5 | 38.5 | 89.9 | 75.0 | 32.2 | 73.9 | 58.2 |

mIoU in %, mean ± sample std over seeds where Seeds > 1. mIoU excludes void and classes in `eval.exclude_from_miou` (bridge). 'Best val mIoU (hist.)' is the best-epoch value logged in training.
