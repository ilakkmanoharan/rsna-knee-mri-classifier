"""ASRA hypothesis records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class Hypothesis:
    hypothesis_id: str
    hypothesis: str
    mechanism: str
    config_delta: dict[str, Any]
    expected_targets: list[str]
    falsification: str
    status: str = "proposed"  # proposed | running | accept | reject | inconclusive
    fold_results: dict[str, Any] = field(default_factory=dict)
    macro_oof_auc: Optional[float] = None
    runtime_sec: Optional[float] = None
    decision_reason: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Seed hypotheses from the spec (H0–H5)
INITIAL_HYPOTHESES: list[Hypothesis] = [
    Hypothesis(
        hypothesis_id="H0",
        hypothesis="Constant prevalence priors provide a schema-valid sanity baseline.",
        mechanism="Predict train-fold prevalence (or 0.5) for every study/target.",
        config_delta={"model": "prevalence_baseline"},
        expected_targets=["*"],
        falsification="Submission fails structural checks or macro AUC undefined.",
    ),
    Hypothesis(
        hypothesis_id="H1",
        hypothesis="Report-derived weak labels train a better image model than gold labels alone.",
        mechanism="Add confidence-weighted soft BCE from report parser.",
        config_delta={"train.lambda_weak": 0.25},
        expected_targets=["*"],
        falsification="Macro OOF AUC does not improve vs visual-only; or target collapse.",
    ),
    Hypothesis(
        hypothesis_id="H2",
        hypothesis="Plane-aware aggregation improves ranking over pooling all series together.",
        mechanism="Pool series into sagittal/coronal/axial/unknown slots.",
        config_delta={"model.use_plane_slots": True},
        expected_targets=["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "PF OA"],
        falsification="Macro OOF AUC <= pooled baseline within noise.",
    ),
    Hypothesis(
        hypothesis_id="H3",
        hypothesis="Symbolic evidence confidence beats hard report pseudo-labels.",
        mechanism="Use soft mapping + confidence weights instead of 0/1 pseudo labels.",
        config_delta={"weak_labels": "soft_confidence"},
        expected_targets=["*"],
        falsification="Hard pseudo-labels match or beat soft mapping on OOF.",
    ),
    Hypothesis(
        hypothesis_id="H4",
        hypothesis="Availability masks prevent missing planes from being read as negative evidence.",
        mechanism="Concatenate plane availability mask to study embedding.",
        config_delta={"model.use_availability_mask": True},
        expected_targets=["*"],
        falsification="No improvement; or missing-plane studies systematically lower.",
    ),
    Hypothesis(
        hypothesis_id="H5",
        hypothesis="Conservative soft constraints improve stability without lowering per-target AUC.",
        mechanism="Small co-occurrence + anti-overconfidence regularizer.",
        config_delta={"train.lambda_consistency": 0.02},
        expected_targets=["Effusion", "Synovitis"],
        falsification="Any target AUC collapses or macro AUC drops.",
    ),
]
