"""Soft relational consistency penalties (never hard diagnosis rules)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.symbolic.ontology import SOFT_COOCCURRENCE


def consistency_loss(
    logits: torch.Tensor,
    targets: list[str],
    availability: torch.Tensor | None = None,
    preferred_coverage: torch.Tensor | None = None,
    weight: float = 0.02,
) -> torch.Tensor:
    """
    Soft penalties:
    - encourage mild co-occurrence agreement for validated pairs
    - gently down-weight confidence when preferred planes missing (via coverage feature path;
      here we only regularize prediction pairs)

    logits: [B, T]
    availability unused for hard zeros — missing view must not force negative labels.
    """
    if weight <= 0:
        return logits.new_zeros(())
    probs = torch.sigmoid(logits)
    loss = logits.new_zeros(())
    name_to_idx = {t: i for i, t in enumerate(targets)}
    n_pairs = 0
    for a, b in SOFT_COOCCURRENCE:
        if a not in name_to_idx or b not in name_to_idx:
            continue
        ia, ib = name_to_idx[a], name_to_idx[b]
        # Soft agreement: penalize large |p_a - p_b| only mildly when both mid-high
        pa, pb = probs[:, ia], probs[:, ib]
        pair = (pa - pb).pow(2) * (pa + pb) * 0.5
        loss = loss + pair.mean()
        n_pairs += 1

    if preferred_coverage is not None:
        # If preferred plane missing (low coverage), discourage extreme certainty
        # without pushing toward 0 or 1 specifically.
        entropy = -(probs * (probs + 1e-6).log() + (1 - probs) * (1 - probs + 1e-6).log())
        # Low coverage -> prefer higher entropy (less overconfident)
        miss = (1.0 - preferred_coverage).clamp(0, 1)
        loss = loss + (miss * (1.2 - entropy).clamp(min=0)).mean()

    if n_pairs == 0 and preferred_coverage is None:
        return logits.new_zeros(())
    return weight * loss


def masked_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    logits/labels/mask: [B, T]
    mask: 1 where label is supervised
    """
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if sample_weight is not None:
        loss = loss * sample_weight
    loss = loss * mask
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom
