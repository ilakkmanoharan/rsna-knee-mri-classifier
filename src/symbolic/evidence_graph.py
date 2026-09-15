"""Evidence graph construction and symbolic feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.constants import PLANES, STUDY_ID_COL
from src.symbolic.ontology import (
    FINDING_PREFERRED_PLANES,
    EvidenceRecord,
    EvidenceState,
    KneeState,
    Plane,
    SeriesNode,
    SequenceFamily,
)
from src.symbolic.report_parser import ReportParser, evidence_to_soft_label


@dataclass
class SymbolicFeatures:
    study_uid: str
    availability: np.ndarray  # [4] sag/cor/ax/unk
    series_counts: np.ndarray  # [4]
    evidence_state_oh: np.ndarray  # [n_targets, 5]
    evidence_confidence: np.ndarray  # [n_targets]
    contradiction: np.ndarray  # [n_targets]
    preferred_plane_coverage: np.ndarray  # [n_targets]
    soft_labels: np.ndarray  # [n_targets]
    soft_weights: np.ndarray  # [n_targets]

    def as_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                self.availability.astype(np.float32),
                self.series_counts.astype(np.float32),
                self.evidence_state_oh.reshape(-1).astype(np.float32),
                self.evidence_confidence.astype(np.float32),
                self.contradiction.astype(np.float32),
                self.preferred_plane_coverage.astype(np.float32),
            ]
        )


STATE_INDEX = {s: i for i, s in enumerate(EvidenceState)}


def build_knee_state(
    study_uid: str,
    series_df: Optional[pd.DataFrame],
    report_text: str | None,
    parser: ReportParser,
    plane_col: str = "Anatomical_Plane",
    series_id_col: str = "SeriesInstanceUID",
) -> KneeState:
    state = KneeState(study_uid=study_uid)
    if series_df is not None and len(series_df):
        sub = series_df[series_df[STUDY_ID_COL].astype(str) == str(study_uid)]
        for _, row in sub.iterrows():
            plane_raw = str(row.get(plane_col, "") or "").lower()
            plane = Plane.UNKNOWN
            for p in Plane:
                if p.value in plane_raw:
                    plane = p
                    break
            seq = SequenceFamily.UNKNOWN
            fs = row.get("Fluid_Sensitive", None)
            fat = row.get("Fat_Suppression", None)
            if fat == 1 or fat == "1":
                seq = SequenceFamily.FAT_SUPPRESSED
            elif fs == 1 or fs == "1":
                seq = SequenceFamily.T2
            state.series.append(
                SeriesNode(
                    series_uid=str(row.get(series_id_col, "")),
                    plane=plane,
                    sequence=seq,
                    fluid_sensitive=int(fs) if pd.notna(fs) else None,
                    fat_suppression=int(fat) if pd.notna(fat) else None,
                )
            )
    if report_text:
        state.evidence = parser.parse_report(report_text)
    else:
        state.evidence = [
            EvidenceRecord(
                finding=t,
                state=EvidenceState.UNMENTIONED,
                confidence=0.0,
                match_category="no_report",
                rule_id="R_no_report",
                parser_version=parser.parser_version,
            )
            for t in parser.targets
        ]
    return state


def knee_state_to_features(
    state: KneeState,
    targets: list[str],
    soft_mapping: dict[str, float],
    prevalence: dict[str, float] | None = None,
    unmentioned_shrinkage: float = 0.5,
) -> SymbolicFeatures:
    plane_list = list(PLANES)
    availability = np.zeros(len(plane_list), dtype=np.float32)
    counts = np.zeros(len(plane_list), dtype=np.float32)
    for s in state.series:
        idx = plane_list.index(s.plane.value)
        availability[idx] = 1.0
        counts[idx] += 1.0

    n = len(targets)
    state_oh = np.zeros((n, len(EvidenceState)), dtype=np.float32)
    conf = np.zeros(n, dtype=np.float32)
    contra = np.zeros(n, dtype=np.float32)
    pref_cov = np.zeros(n, dtype=np.float32)
    soft = np.zeros(n, dtype=np.float32)
    weights = np.zeros(n, dtype=np.float32)

    evid_map = {e.finding: e for e in state.evidence}
    for i, t in enumerate(targets):
        e = evid_map.get(t) or state.evidence_for(t)
        state_oh[i, STATE_INDEX[e.state]] = 1.0
        conf[i] = float(e.confidence)
        contra[i] = 1.0 if e.contradiction else 0.0
        prefs = FINDING_PREFERRED_PLANES.get(t, ())
        if prefs:
            pref_cov[i] = float(
                np.mean([availability[plane_list.index(p)] for p in prefs if p in plane_list])
            )
        else:
            pref_cov[i] = float(availability[:3].mean())
        prev = None if prevalence is None else prevalence.get(t)
        soft[i] = evidence_to_soft_label(
            e.state, soft_mapping, prevalence=prev, shrinkage=unmentioned_shrinkage
        )
        # Weight: zero for unmentioned low-confidence; else parser confidence
        if e.state == EvidenceState.UNMENTIONED and e.confidence < 0.2:
            weights[i] = 0.0
        else:
            weights[i] = max(float(e.confidence), 0.05)

    return SymbolicFeatures(
        study_uid=state.study_uid,
        availability=availability,
        series_counts=counts,
        evidence_state_oh=state_oh,
        evidence_confidence=conf,
        contradiction=contra,
        preferred_plane_coverage=pref_cov,
        soft_labels=soft,
        soft_weights=weights,
    )


def symbolic_feature_dim(n_targets: int) -> int:
    # availability(4) + counts(4) + state_oh(n*5) + conf(n) + contra(n) + pref(n)
    return 4 + 4 + n_targets * 5 + n_targets + n_targets + n_targets
