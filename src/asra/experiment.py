"""ASRA experiment runner helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from src.asra.hypothesis import Hypothesis
from src.asra.ledger import HypothesisLedger


def macro_auc(per_target: dict[str, float | None]) -> float | None:
    vals = [v for v in per_target.values() if v is not None and np.isfinite(v)]
    if not vals:
        return None
    return float(np.mean(vals))


def run_experiment(
    ledger: HypothesisLedger,
    hyp: Hypothesis,
    train_fn: Callable[[], dict[str, Any]],
    baseline_macro: float | None = None,
    min_delta: float = 0.001,
    collapse_auc: float = 0.45,
) -> Hypothesis:
    """Execute train_fn which returns {per_target_auc, macro_oof_auc?, extras...}."""
    hyp.status = "running"
    ledger.update(hyp)
    t0 = time.time()
    result = train_fn()
    runtime = time.time() - t0
    per_target = result.get("per_target_auc", {})
    macro = result.get("macro_oof_auc")
    if macro is None:
        macro = macro_auc(per_target)
    if macro is None:
        hyp.status = "inconclusive"
        hyp.decision_reason = "Macro AUC undefined"
        hyp.runtime_sec = runtime
        ledger.update(hyp)
        return hyp
    return ledger.decide(
        hyp,
        macro_oof=float(macro),
        baseline_macro=baseline_macro,
        per_target_auc=per_target,
        runtime_sec=runtime,
        min_delta=min_delta,
        collapse_auc=collapse_auc,
    )


def save_oof_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str))
