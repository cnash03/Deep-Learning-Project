from src.utils import load_config


def test_inheritance_and_overrides():
    cfg = load_config("configs/e4_weighted_ce.yaml", ["train.epochs=3", "data.subset=10"])
    assert cfg["model"]["arch"] == "deeplabv3plus"          # from best_arch.yaml
    assert cfg["loss"]["name"] == "weighted_ce"             # from e4
    assert cfg["train"]["batch_size"] == 8                  # from base.yaml
    assert cfg["train"]["epochs"] == 3 and cfg["data"]["subset"] == 10


def test_all_configs_load():
    from pathlib import Path

    for p in Path("configs").glob("e*.yaml"):
        cfg = load_config(p)
        assert {"data", "model", "loss", "train"} <= cfg.keys(), p
