import math

import torch

from src.data.colormaps import IGNORE_INDEX
from src.metrics import ConfusionMatrix, predict


def test_perfect_prediction():
    t = torch.randint(0, 4, (2, 8, 8))
    cm = ConfusionMatrix(4)
    cm.update(t, t)
    assert cm.miou() == 1.0 and cm.pixel_acc() == 1.0


def test_known_iou_and_ignore():
    target = torch.tensor([0, 0, 1, 1, IGNORE_INDEX])
    pred = torch.tensor([0, 1, 1, 1, 0])
    cm = ConfusionMatrix(3)
    cm.update(pred, target)
    iou = cm.iou()
    assert math.isclose(iou[0].item(), 1 / 2)  # tp=1, fp=0, fn=1
    assert math.isclose(iou[1].item(), 2 / 3)  # tp=2, fp=1, fn=0
    assert math.isnan(iou[2].item())           # absent everywhere -> excluded from mean
    assert math.isclose(cm.miou(), (1 / 2 + 2 / 3) / 2)
    assert cm.mat.sum() == 4                   # ignore pixel not counted


def test_predict_excludes_class():
    logits = torch.zeros(1, 3, 2, 2)
    logits[:, 0] = 10.0
    logits[:, 2] = 1.0
    assert (predict(logits, exclude=[0]) == 2).all()


def test_miou_gt_present_ignores_absent_false_positives():
    target = torch.tensor([0, 0, 1, 1])
    pred = torch.tensor([0, 0, 1, 2])  # class 2 absent from GT, predicted once
    cm = ConfusionMatrix(3)
    cm.update(pred, target)
    assert math.isclose(cm.miou(), (1 + 0.5 + 0) / 3)
    assert math.isclose(cm.miou_gt_present(), (1 + 0.5) / 2)
