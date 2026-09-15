"""Symbolic knee ontology: entities, evidence states, and soft relations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.constants import DEFAULT_TARGETS, EVIDENCE_STATES, PLANES


class EvidenceState(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNCERTAIN = "uncertain"
    HISTORICAL = "historical"
    UNMENTIONED = "unmentioned"


class Plane(str, Enum):
    SAGITTAL = "sagittal"
    CORONAL = "coronal"
    AXIAL = "axial"
    UNKNOWN = "unknown"


class SequenceFamily(str, Enum):
    T1 = "t1_like"
    T2 = "t2_like"
    PD = "pd_like"
    FAT_SUPPRESSED = "fat_suppressed"
    UNKNOWN = "unknown"


# Finding -> preferred anatomical structure(s)
FINDING_STRUCTURES: dict[str, tuple[str, ...]] = {
    "ACL": ("ACL",),
    "MCL": ("MCL",),
    "Medial Meniscus": ("medial_meniscus",),
    "Lateral Meniscus": ("lateral_meniscus",),
    "Medial OA": ("medial_compartment",),
    "Lateral OA": ("lateral_compartment",),
    "PF OA": ("patellofemoral_compartment",),
    "Effusion": ("joint_space",),
    "Synovitis": ("synovium",),
    "Baker's": ("popliteal_region",),
    "Contusion": ("bone",),
    "Fracture": ("bone",),
}

# Soft preferred planes (never hard requirements)
FINDING_PREFERRED_PLANES: dict[str, tuple[str, ...]] = {
    "ACL": ("sagittal",),
    "MCL": ("coronal",),
    "Medial Meniscus": ("sagittal", "coronal"),
    "Lateral Meniscus": ("sagittal", "coronal"),
    "Medial OA": ("coronal",),
    "Lateral OA": ("coronal",),
    "PF OA": ("axial", "sagittal"),
    "Effusion": ("axial", "sagittal"),
    "Synovitis": ("axial", "sagittal"),
    "Baker's": ("axial", "sagittal"),
    "Contusion": ("sagittal", "coronal", "axial"),
    "Fracture": ("sagittal", "coronal", "axial"),
}

# Soft co-occurrence pairs (regularizers only)
SOFT_COOCCURRENCE: tuple[tuple[str, str], ...] = (
    ("Effusion", "Synovitis"),
    ("Medial OA", "Medial Meniscus"),
    ("Lateral OA", "Lateral Meniscus"),
    ("Contusion", "Fracture"),
)


@dataclass
class EvidenceRecord:
    finding: str
    state: EvidenceState
    confidence: float
    match_category: str
    rule_id: str
    contradiction: bool = False
    parser_version: str = "report_parser_v1"


@dataclass
class SeriesNode:
    series_uid: str
    plane: Plane = Plane.UNKNOWN
    sequence: SequenceFamily = SequenceFamily.UNKNOWN
    n_slices: int = 0
    fluid_sensitive: Optional[int] = None
    fat_suppression: Optional[int] = None


@dataclass
class KneeState:
    study_uid: str
    series: list[SeriesNode] = field(default_factory=list)
    evidence: list[EvidenceRecord] = field(default_factory=list)
    findings: dict[str, float] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)

    def plane_availability(self) -> dict[str, float]:
        present = {p.value: 0.0 for p in Plane}
        for s in self.series:
            present[s.plane.value] = 1.0
        return present

    def evidence_for(self, finding: str) -> EvidenceRecord:
        for e in self.evidence:
            if e.finding == finding:
                return e
        return EvidenceRecord(
            finding=finding,
            state=EvidenceState.UNMENTIONED,
            confidence=0.0,
            match_category="none",
            rule_id="default_unmentioned",
        )


def assert_targets_known(targets: list[str] | None = None) -> list[str]:
    targets = targets or list(DEFAULT_TARGETS)
    for t in targets:
        if t not in FINDING_STRUCTURES:
            # Allow competition schema evolution; soft-map unknown findings
            FINDING_STRUCTURES[t] = ("unknown_structure",)
            FINDING_PREFERRED_PLANES[t] = tuple(PLANES[:3])
    return targets


def evidence_state_from_str(s: str) -> EvidenceState:
    s = s.lower().strip()
    if s not in EVIDENCE_STATES:
        raise ValueError(f"Unknown evidence state: {s}")
    return EvidenceState(s)
