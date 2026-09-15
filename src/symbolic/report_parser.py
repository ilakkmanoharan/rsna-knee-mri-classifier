"""Multilingual, negation-aware radiology report parser for weak supervision."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from src.constants import DEFAULT_TARGETS
from src.symbolic.ontology import EvidenceRecord, EvidenceState

PARSER_VERSION = "report_parser_v1"

# Target synonym dictionaries (EN / common Romance / Germanic stubs).
# Extend via ASRA when audit reveals language coverage gaps.
TARGET_SYNONYMS: dict[str, tuple[str, ...]] = {
    "ACL": (
        r"\bacl\b",
        r"anterior\s+cruciate(?:\s+ligament)?",
        r"ligament\s+crois[eé]\s+ant[eé]rieur",
        r"\blca\b",
        r"vorderes?\s+kreuzband",
    ),
    "MCL": (
        r"\bmcl\b",
        r"medial\s+collateral(?:\s+ligament)?",
        r"ligament\s+collat[eé]ral\s+m[eé]dial",
        r"innenband",
    ),
    "Medial Meniscus": (
        r"medial\s+meniscus",
        r"meniscus\s+medialis",
        r"m[eé]nisque\s+m[eé]dial",
        r"innenmeniskus",
        r"\bmm\b(?=.{0,40}(tear|rupt|lesion|degener))",
    ),
    "Lateral Meniscus": (
        r"lateral\s+meniscus",
        r"meniscus\s+lateralis",
        r"m[eé]nisque\s+lat[eé]ral",
        r"aussenmeniskus|außenmeniskus",
        r"\blm\b(?=.{0,40}(tear|rupt|lesion|degener))",
    ),
    "Medial OA": (
        r"medial(?:\s+compartment)?\s+(?:oa|osteoarthrit\w+|arthros\w+)",
        r"(?:oa|osteoarthrit\w+|arthros\w+).{0,30}medial(?:\s+compartment)?",
        r"gonarthrose\s+m[eé]diale",
    ),
    "Lateral OA": (
        r"lateral(?:\s+compartment)?\s+(?:oa|osteoarthrit\w+|arthros\w+)",
        r"(?:oa|osteoarthrit\w+|arthros\w+).{0,30}lateral(?:\s+compartment)?",
        r"gonarthrose\s+lat[eé]rale",
    ),
    "PF OA": (
        r"patellofemoral(?:\s+compartment)?\s+(?:oa|osteoarthrit\w+|arthros\w+)",
        r"(?:pf|patello-?femoral).{0,20}(?:oa|osteoarthrit\w+|arthros\w+)",
        r"(?:oa|osteoarthrit\w+|arthros\w+).{0,20}(?:pf|patello-?femoral)",
    ),
    "Effusion": (
        r"\beffusion\b",
        r"joint\s+effusion",
        r"[eé]panchement",
        r"gelenkerguss",
        r"liquid(?:e)?\s+intra-?articulaire",
    ),
    "Synovitis": (
        r"\bsynovitis\b",
        r"synovite",
        r"synovial\s+thickening",
        r"synovialitis",
    ),
    "Baker's": (
        r"baker'?s?\s+(?:cyst|zyste)",
        r"popliteal\s+cyst",
        r"kyste\s+poplit[eé]",
        r"bakerzyste",
    ),
    "Contusion": (
        r"\bcontusion\b",
        r"bone\s+bruise",
        r"bone\s+marrow\s+edema",
        r"contusion\s+osseuse",
        r"knochenmark[oö]dem",
    ),
    "Fracture": (
        r"\bfracture\b",
        r"\bfractura\b",
        r"\bbruch\b",
        r"fractur",
    ),
}

# Generic OA — must NOT auto-assign all compartments
GENERIC_OA = (
    r"\bosteoarthrit\w+\b",
    r"\barthros\w+\b",
    r"\bgonarthrose\b",
    r"\boa\b",
)

POSITIVE_CUES = (
    r"\btear\b",
    r"\btorn\b",
    r"\brupture\b",
    r"\bruptured\b",
    r"\blesion\b",
    r"\bsprain\b",
    r"\binjury\b",
    r"\bpresent\b",
    r"\bevident\b",
    r"\bdemonstrat\w+\b",
    r"\bcompatib\w+\b",
    r"\bconsistent\s+with\b",
    r"\bpartial\s+tear\b",
    r"\bcomplete\s+tear\b",
    r"\bruptur\w+\b",
    r"\bl[aä]sion\b",
    r"\bruptur\b",
)

NEGATION_CUES = (
    r"\bno\b",
    r"\bwithout\b",
    r"\bdenies\b",
    r"\babsent\b",
    r"\bnegative\s+for\b",
    r"\brules?\s+out\b",
    r"\bunremarkable\b",
    r"\bnormal\b",
    r"\bintact\b",
    r"\bno\s+evidence\b",
    r"\bsans\b",
    r"\bpas\s+de\b",
    r"\babsence\s+de\b",
    r"\bkein[e]?\b",
    r"\bohne\b",
)

UNCERTAINTY_CUES = (
    r"\bpossible\b",
    r"\bpossibly\b",
    r"\bprobable\b",
    r"\blikely\b",
    r"\bsuggestive\b",
    r"\bcannot\s+exclude\b",
    r"\bmay\s+represent\b",
    r"\bquestionable\b",
    r"\bequivocal\b",
    r"\bdifferential\b",
    r"\bsuspicious\b",
    r"\bappearances?\s+of\b",
    r"\bpossible\b",
    r"\bpeut[- ][eê]tre\b",
    r"\bvermutlich\b",
)

HISTORICAL_CUES = (
    r"\bstatus\s+post\b",
    r"\bs/p\b",
    r"\bpost[- ]?op(?:erative)?\b",
    r"\bprior\b",
    r"\bprevious\b",
    r"\bremote\b",
    r"\bchronic\s+post[- ]?surgical\b",
    r"\bhistory\s+of\b",
    r"\breconstruct(?:ion|ed)\b",
    r"\bgraft\b",
)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u00a0", " ")
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_clauses(text: str) -> list[str]:
    # Clause boundaries: sentence punctuation and common report separators
    parts = re.split(r"[.!?;\n]|(?:\s+-\s+)|(?:\s+/\s+)", text)
    return [p.strip() for p in parts if p and p.strip()]


def _any_match(patterns: tuple[str, ...], text: str) -> Optional[re.Match]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m
    return None


def _window_has(cues: tuple[str, ...], text: str, center: int, radius: int = 60) -> bool:
    lo = max(0, center - radius)
    hi = min(len(text), center + radius)
    window = text[lo:hi]
    return _any_match(cues, window) is not None


@dataclass
class ParseHit:
    finding: str
    state: EvidenceState
    confidence: float
    match_category: str
    rule_id: str
    span: tuple[int, int]


class ReportParser:
    def __init__(
        self,
        targets: list[str] | None = None,
        parser_version: str = PARSER_VERSION,
    ) -> None:
        self.targets = targets or list(DEFAULT_TARGETS)
        self.parser_version = parser_version
        self._compiled: dict[str, list[re.Pattern]] = {}
        for t in self.targets:
            pats = TARGET_SYNONYMS.get(t, (re.escape(t.lower()),))
            self._compiled[t] = [re.compile(p, re.IGNORECASE) for p in pats]

    def detect_language(self, text: str) -> str:
        try:
            from langdetect import detect

            return detect(text) if text.strip() else "unknown"
        except Exception:  # noqa: BLE001
            return "unknown"

    def parse_report(self, text: str) -> list[EvidenceRecord]:
        norm = normalize_text(text)
        if not norm:
            return [
                EvidenceRecord(
                    finding=t,
                    state=EvidenceState.UNMENTIONED,
                    confidence=0.0,
                    match_category="empty_report",
                    rule_id="R0_empty",
                    parser_version=self.parser_version,
                )
                for t in self.targets
            ]

        hits_by_target: dict[str, list[ParseHit]] = {t: [] for t in self.targets}
        clauses = split_clauses(norm)

        for clause in clauses:
            for finding, patterns in self._compiled.items():
                for pi, pat in enumerate(patterns):
                    for m in pat.finditer(clause):
                        hit = self._classify_hit(finding, clause, m, rule_suffix=f"{pi}")
                        hits_by_target[finding].append(hit)

        # Generic OA: family evidence only — do not force compartment targets positive
        # (exposed as match_category on OA targets if unmentioned)
        generic_oa = _any_match(GENERIC_OA, norm) is not None

        records: list[EvidenceRecord] = []
        for finding in self.targets:
            hits = hits_by_target[finding]
            if not hits:
                cat = "unmentioned"
                conf = 0.0
                if generic_oa and finding in {"Medial OA", "Lateral OA", "PF OA"}:
                    cat = "generic_oa_family_only"
                    conf = 0.15
                records.append(
                    EvidenceRecord(
                        finding=finding,
                        state=EvidenceState.UNMENTIONED,
                        confidence=conf,
                        match_category=cat,
                        rule_id="R_unmentioned",
                        parser_version=self.parser_version,
                    )
                )
                continue

            # Precedence: positive > negative > uncertain > historical
            chosen = self._resolve_precedence(hits)
            contradiction = self._has_contradiction(hits)
            records.append(
                EvidenceRecord(
                    finding=finding,
                    state=chosen.state,
                    confidence=chosen.confidence * (0.7 if contradiction else 1.0),
                    match_category=chosen.match_category,
                    rule_id=chosen.rule_id,
                    contradiction=contradiction,
                    parser_version=self.parser_version,
                )
            )
        return records

    def _classify_hit(
        self, finding: str, clause: str, match: re.Match, rule_suffix: str
    ) -> ParseHit:
        center = match.start()
        historical = _window_has(HISTORICAL_CUES, clause, center)
        negated = _window_has(NEGATION_CUES, clause, center)
        uncertain = _window_has(UNCERTAINTY_CUES, clause, center)
        positive = _window_has(POSITIVE_CUES, clause, center) or True  # mention defaults soft-pos

        if historical and not negated:
            state = EvidenceState.HISTORICAL
            conf = 0.55
            cat = "historical_mention"
            rid = f"R_hist_{finding}_{rule_suffix}"
        elif negated:
            state = EvidenceState.NEGATIVE
            conf = 0.85
            cat = "negated_mention"
            rid = f"R_neg_{finding}_{rule_suffix}"
        elif uncertain:
            state = EvidenceState.UNCERTAIN
            conf = 0.55
            cat = "uncertain_mention"
            rid = f"R_unc_{finding}_{rule_suffix}"
        elif positive:
            state = EvidenceState.POSITIVE
            conf = 0.8 if _window_has(POSITIVE_CUES, clause, center) else 0.55
            cat = "positive_mention" if conf >= 0.8 else "bare_mention"
            rid = f"R_pos_{finding}_{rule_suffix}"
        else:
            state = EvidenceState.UNCERTAIN
            conf = 0.5
            cat = "bare_mention"
            rid = f"R_bare_{finding}_{rule_suffix}"

        return ParseHit(
            finding=finding,
            state=state,
            confidence=conf,
            match_category=cat,
            rule_id=rid,
            span=(match.start(), match.end()),
        )

    @staticmethod
    def _resolve_precedence(hits: list[ParseHit]) -> ParseHit:
        order = {
            EvidenceState.POSITIVE: 0,
            EvidenceState.NEGATIVE: 1,
            EvidenceState.UNCERTAIN: 2,
            EvidenceState.HISTORICAL: 3,
            EvidenceState.UNMENTIONED: 4,
        }
        return sorted(hits, key=lambda h: (order[h.state], -h.confidence))[0]

    @staticmethod
    def _has_contradiction(hits: list[ParseHit]) -> bool:
        states = {h.state for h in hits}
        return EvidenceState.POSITIVE in states and EvidenceState.NEGATIVE in states


def evidence_to_soft_label(
    state: EvidenceState,
    mapping: dict[str, float],
    prevalence: float | None = None,
    shrinkage: float = 0.5,
) -> float:
    """Map evidence state to soft probability. Unmentioned uses prevalence shrinkage."""
    if state == EvidenceState.UNMENTIONED:
        prior = 0.5 if prevalence is None else float(prevalence)
        return float(shrinkage * prior + (1.0 - shrinkage) * 0.5)
    return float(mapping[state.value])
