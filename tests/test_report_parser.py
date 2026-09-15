"""Unit tests for report parser evidence precedence and compartment rules."""

from __future__ import annotations

from src.symbolic.ontology import EvidenceState
from src.symbolic.report_parser import ReportParser, evidence_to_soft_label, normalize_text


def test_normalize_unicode():
    assert "acl" in normalize_text("ACL\u00a0tear")


def test_positive_acl():
    p = ReportParser()
    recs = {r.finding: r for r in p.parse_report("There is a complete tear of the ACL.")}
    assert recs["ACL"].state == EvidenceState.POSITIVE
    assert recs["ACL"].confidence > 0.5


def test_negated_acl():
    p = ReportParser()
    recs = {r.finding: r for r in p.parse_report("No ACL tear. Intact ACL.")}
    assert recs["ACL"].state == EvidenceState.NEGATIVE


def test_uncertain_meniscus():
    p = ReportParser()
    recs = {r.finding: r for r in p.parse_report("Possible medial meniscus tear.")}
    assert recs["Medial Meniscus"].state == EvidenceState.UNCERTAIN


def test_historical_acl():
    p = ReportParser()
    recs = {r.finding: r for r in p.parse_report("Status post ACL reconstruction with graft.")}
    assert recs["ACL"].state == EvidenceState.HISTORICAL


def test_generic_oa_does_not_force_compartments():
    p = ReportParser()
    recs = {r.finding: r for r in p.parse_report("Mild osteoarthritis of the knee.")}
    for t in ("Medial OA", "Lateral OA", "PF OA"):
        assert recs[t].state == EvidenceState.UNMENTIONED
        assert recs[t].match_category == "generic_oa_family_only"


def test_compartment_specific_oa():
    p = ReportParser()
    recs = {r.finding: r for r in p.parse_report("Severe medial compartment osteoarthritis.")}
    assert recs["Medial OA"].state == EvidenceState.POSITIVE
    assert recs["Lateral OA"].state == EvidenceState.UNMENTIONED


def test_contradiction_flag():
    p = ReportParser()
    text = "ACL tear is present. No ACL tear identified."
    recs = {r.finding: r for r in p.parse_report(text)}
    assert recs["ACL"].contradiction is True


def test_unmentioned_default():
    p = ReportParser()
    recs = {r.finding: r for r in p.parse_report("Normal exam of the soft tissues.")}
    assert recs["Fracture"].state == EvidenceState.UNMENTIONED


def test_soft_label_mapping():
    mapping = {
        "positive": 0.9,
        "negative": 0.1,
        "uncertain": 0.6,
        "historical": 0.5,
    }
    assert evidence_to_soft_label(EvidenceState.POSITIVE, mapping) == 0.9
    assert evidence_to_soft_label(EvidenceState.NEGATIVE, mapping) == 0.1
    u = evidence_to_soft_label(EvidenceState.UNMENTIONED, mapping, prevalence=0.2, shrinkage=0.5)
    assert 0.3 < u < 0.45
