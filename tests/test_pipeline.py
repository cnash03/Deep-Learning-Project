"""End-to-end smoke test: dataset -> train 1 epoch -> evaluate, on a fake RUGD tree (CPU, tiny images)."""
import json
import sys

import pytest

pytest.importorskip("segmentation_models_pytorch")


def test_dataset(fake_rugd):
    from src.data import build_dataset

    cfg = {"data": {"name": "rugd", "root": str(fake_rugd), "img_size": 64}, "augment": {}}
    ds = build_dataset(cfg, "train", train=True)
    img, mask = ds[0]
    assert img.shape == (3, 64, 64) and mask.shape == (64, 64)
    assert mask.dtype.is_floating_point is False
    assert set(mask.unique().tolist()) <= set(range(1, 25)) | {255}  # void (0) -> ignore


def test_train_and_evaluate(fake_rugd, tmp_path, monkeypatch):
    from src import evaluate, train

    out = tmp_path / "runs"
    common = [f"data.root={fake_rugd}", "data.img_size=64", "model.pretrained=false",
              "train.epochs=1", "train.batch_size=2", "train.num_workers=0", "train.warmup_steps=0",
              f"train.out_dir={out}"]
    monkeypatch.setattr(sys, "argv", ["train", "--config", "configs/e1_unet_r18_ce.yaml", "--set", *common])
    train.main()
    run = out / "e1_unet_r18_ce"
    assert (run / "best.pt").exists() and (run / "last.pt").exists()

    monkeypatch.setattr(sys, "argv", ["evaluate", "--run", str(run), "--split", "val", "--out", str(tmp_path / "res")])
    evaluate.main()
    summary = json.loads((tmp_path / "res" / "e1_unet_r18_ce" / "val" / "summary.json").read_text())
    assert 0.0 <= summary["miou"] <= 1.0 and summary["n_images"] == 4

    monkeypatch.setattr(sys, "argv", ["evaluate", "--run", str(run), "--split", "test"])
    with pytest.raises(SystemExit):
        evaluate.main()  # test split guarded
