"""Aggregation mask / plane pool tests."""

from __future__ import annotations

import torch

from src.models.pooling import PlanePool, SlicePool
from src.symbolic.constraints import consistency_loss, masked_bce_with_logits


def test_slice_pool_mean_max_mask():
    pool = SlicePool(dim=4, method="mean_max")
    feats = torch.randn(2, 5, 4)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.float32)
    out = pool(feats, mask)
    assert out.shape == (2, 8)


def test_plane_pool_availability():
    pp = PlanePool(dim=3, n_planes=4)
    feats = torch.randn(1, 3, 3)
    plane_ids = torch.tensor([[0, 0, 2]])
    mask = torch.tensor([[1, 1, 1]], dtype=torch.float32)
    flat, avail = pp(feats, plane_ids, mask)
    assert flat.shape == (1, 12)
    assert avail[0, 0] == 1
    assert avail[0, 1] == 0
    assert avail[0, 2] == 1


def test_masked_bce():
    logits = torch.zeros(2, 3)
    labels = torch.zeros(2, 3)
    mask = torch.tensor([[1, 0, 1], [0, 0, 0]], dtype=torch.float32)
    loss = masked_bce_with_logits(logits, labels, mask)
    assert torch.isfinite(loss)


def test_consistency_nonneg():
    logits = torch.randn(4, 12)
    targets = [
        "ACL",
        "MCL",
        "Medial Meniscus",
        "Lateral Meniscus",
        "Medial OA",
        "Lateral OA",
        "PF OA",
        "Effusion",
        "Synovitis",
        "Baker's",
        "Contusion",
        "Fracture",
    ]
    loss = consistency_loss(logits, targets, preferred_coverage=torch.ones(4, 12) * 0.5)
    assert loss.ndim == 0
    assert float(loss) >= 0
