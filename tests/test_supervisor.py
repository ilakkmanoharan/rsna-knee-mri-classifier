"""Unit tests for the Grok supervisor's deterministic checks."""

from datetime import timedelta

from agent.clock import competition_day_start
from agent.supervisor import _cycle_num, _quality_issues, slots_due

CFG = {
    "cycle_interval_minutes": 90,
    "max_submissions_per_day": 5,
    "supervisor": {"grace_minutes": 25},
}


def test_cycle_num_parses_agent_filenames():
    day = "2026-09-15"
    assert _cycle_num(f"{day}_cycle03_20260915T114110-0500_research.md", day) == 3
    assert _cycle_num(f"{day}_cycle00_20260915T114110-0500_hypotheses.md", day) == 0
    assert _cycle_num("2026-09-14_cycle01_x_research.md", day) is None
    assert _cycle_num("notes.md", day) is None


def test_slots_due_counts_elapsed_slots_with_grace():
    start = competition_day_start()
    # 10 minutes into the day: first slot still inside its grace window
    assert slots_due(CFG, start + timedelta(minutes=10), start)["due_count"] == 0
    # 30 minutes in: slot 0 is overdue
    assert slots_due(CFG, start + timedelta(minutes=30), start)["due_count"] == 1
    # third slot starts at 180 min and is overdue 25 min later
    assert slots_due(CFG, start + timedelta(minutes=200), start)["due_count"] == 2
    assert slots_due(CFG, start + timedelta(minutes=210), start)["due_count"] == 3
    # end of day is capped at the daily quota
    assert slots_due(CFG, start + timedelta(hours=20), start)["due_count"] == 5


def test_quality_issues_flag_empty_stage_writeups():
    assert _quality_issues("research", "", 1200)
    assert _quality_issues("analysis", "", 700)
    assert _quality_issues("hypothesis", "", 500)


def test_quality_issues_pass_substantive_writeups():
    research = (
        "# Methods to consider\n\nBecause macro AUC rewards ranking, plane-aware pooling should raise the score.\n"
        "https://arxiv.org/abs/1 https://arxiv.org/abs/2 https://arxiv.org/abs/3\n" + "detail. " * 300
    )
    assert _quality_issues("research", research, 1200) == []

    analysis = (
        "## Why the score is low\nBest publicScore 0.498 is at the prevalence floor.\n"
        "## How to improve\nAdd metadata ranking features next.\n" + "detail. " * 150
    )
    assert _quality_issues("analysis", analysis, 700) == []

    hypothesis = (
        "H1: If we add plane-availability features from Research and the Analysis of submission logs,\n"
        "then OOF macro AUC improves by >= 0.01; we expect the metric to move above 0.52.\n" + "detail. " * 100
    )
    assert _quality_issues("hypothesis", hypothesis, 500) == []
