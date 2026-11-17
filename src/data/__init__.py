from __future__ import annotations

from .colormaps import IGNORE_INDEX


def build_dataset(cfg: dict, split: str, train: bool):
    """Build the dataset named in cfg['data']['name'] for the given split."""
    from .transforms import build_transforms

    d = cfg["data"]
    tf = build_transforms(cfg, train=train)
    subset = d.get("subset")
    if d["name"] == "rugd":
        from .rugd import RUGDDataset

        return RUGDDataset(d["root"], split, transform=tf, subset=subset, ignore_void=d.get("ignore_void", True))
    if d["name"] == "rellis":
        from .rellis import RELLISDataset

        return RELLISDataset(d["root"], split, transform=tf, subset=subset)
    raise ValueError(f"Unknown dataset {d['name']}")


__all__ = ["IGNORE_INDEX", "build_dataset"]
