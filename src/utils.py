"""Config loading, seeding, and run metadata."""
from __future__ import annotations

import copy
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config. A top-level `base:` key names a parent config (relative path) to inherit from.

    `overrides` are "dotted.key=value" strings, e.g. ["train.epochs=5", "data.root=/content/rugd"].
    """
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    if "base" in cfg:
        parent = load_config(path.parent / cfg.pop("base"))
        cfg = _merge(parent, cfg)
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    # Let an env var point at the data without editing configs (handy on Colab).
    if os.environ.get("DATA_ROOT") and "data" in cfg:
        cfg["data"]["root"] = os.environ["DATA_ROOT"]
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
