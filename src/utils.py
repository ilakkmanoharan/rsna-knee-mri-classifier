"""Shared helpers: config, metrics, seeding."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:  # noqa: BLE001
        pass


def per_target_roc_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    targets: list[str],
) -> dict[str, float | None]:
    """Return per-target AUC; None when a fold/set has a single class."""
    from sklearn.metrics import roc_auc_score

    out: dict[str, float | None] = {}
    for i, t in enumerate(targets):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        mask = np.isfinite(yt)
        yt = yt[mask]
        yp = yp[mask]
        if len(yt) < 2 or len(np.unique(yt)) < 2:
            out[t] = None
        else:
            try:
                out[t] = float(roc_auc_score(yt, yp))
            except ValueError:
                out[t] = None
    return out


def macro_auc_from_dict(d: dict[str, float | None]) -> float | None:
    vals = [v for v in d.values() if v is not None]
    return float(np.mean(vals)) if vals else None
