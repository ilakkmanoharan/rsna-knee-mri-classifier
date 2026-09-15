"""ASRA machine-readable hypothesis ledger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

from src.asra.hypothesis import INITIAL_HYPOTHESES, Hypothesis


class HypothesisLedger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")
            for h in INITIAL_HYPOTHESES:
                self.append(h)

    def append(self, hyp: Hypothesis) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(hyp.to_dict()) + "\n")

    def update(self, hyp: Hypothesis) -> None:
        rows = self.read_all()
        found = False
        for i, row in enumerate(rows):
            if row.get("hypothesis_id") == hyp.hypothesis_id and row.get("status") in {
                "proposed",
                "running",
            }:
                rows[i] = hyp.to_dict()
                found = True
                break
        if not found:
            rows.append(hyp.to_dict())
        with self.path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def decide(
        self,
        hyp: Hypothesis,
        macro_oof: float,
        baseline_macro: float | None,
        per_target_auc: dict[str, float | None],
        runtime_sec: float,
        min_delta: float = 0.001,
        collapse_auc: float = 0.45,
        require_no_collapse: bool = True,
    ) -> Hypothesis:
        hyp.macro_oof_auc = macro_oof
        hyp.runtime_sec = runtime_sec
        hyp.fold_results = {"per_target_auc": per_target_auc, "baseline_macro": baseline_macro}

        collapsed = [
            t
            for t, v in per_target_auc.items()
            if v is not None and v < collapse_auc
        ]
        improved = baseline_macro is None or macro_oof >= baseline_macro + min_delta

        if require_no_collapse and collapsed:
            hyp.status = "reject"
            hyp.decision_reason = f"Target collapse: {collapsed}"
        elif improved:
            hyp.status = "accept"
            hyp.decision_reason = (
                f"Macro OOF {macro_oof:.4f} vs baseline {baseline_macro}"
            )
        elif baseline_macro is not None and abs(macro_oof - baseline_macro) < min_delta:
            hyp.status = "inconclusive"
            hyp.decision_reason = "Delta within noise threshold"
        else:
            hyp.status = "reject"
            hyp.decision_reason = (
                f"Macro OOF {macro_oof:.4f} did not beat baseline {baseline_macro}"
            )
        self.update(hyp)
        return hyp
