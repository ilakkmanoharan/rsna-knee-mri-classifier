"""Unit tests for the Grok supervisor's deterministic checks."""

from datetime import timedelta

from agent.clock import competition_day_start
from agent.pacing import expected_submissions, next_gap_minutes, pace_status
from agent.supervisor import _cycle_num, _quality_issues

CFG = {
    "max_submissions_per_day": 5,
    "pacing": {
        "min_interval_minutes": 30,
        "max_interval_minutes": 60,
        "target_finish_hours": 4,
        "grace_minutes": 20,
        "hard_stop_buffer_minutes": 60,
    },
}


def test_cycle_num_parses_agent_filenames():
    day = "2026-09-15"
    assert _cycle_num(f"{day}_cycle03_20260915T114110-0500_research.md", day) == 3
    assert _cycle_num(f"{day}_cycle00_20260915T114110-0500_hypotheses.md", day) == 0
    assert _cycle_num("2026-09-14_cycle01_x_research.md", day) is None
    assert _cycle_num("notes.md", day) is None


def test_expected_submissions_follows_slowest_acceptable_pace():
    start = competition_day_start()
    # inside the grace window nothing is late yet
    assert expected_submissions(CFG, start + timedelta(minutes=10), start) == 0
    assert expected_submissions(CFG, start + timedelta(minutes=25), start) == 1
    # one more every 60 min at the slowest acceptable pace
    assert expected_submissions(CFG, start + timedelta(minutes=85), start) == 2
    assert expected_submissions(CFG, start + timedelta(minutes=205), start) == 4
    # never more than the daily quota
    assert expected_submissions(CFG, start + timedelta(hours=20), start) == 5


def test_next_gap_relaxes_when_on_track_and_tightens_when_behind():
    start = competition_day_start()
    # one submission in at 01:20 with the whole target window left → near the 60-min ceiling
    assert next_gap_minutes(CFG, 1, start + timedelta(minutes=20), start) == 55
    # falling behind early still keeps the ceiling honored
    assert next_gap_minutes(CFG, 0, start + timedelta(minutes=5), start) == 47.0
    # four still to go with 90 min left → clamps down to the 30-min floor
    assert next_gap_minutes(CFG, 1, start + timedelta(minutes=210), start) == 30
    # mid-day, two left and ~100 min before the hard edge → in-between spacing
    gap = next_gap_minutes(CFG, 3, start + timedelta(hours=21, minutes=20), start)
    assert 30 <= gap <= 60
    # quota spent: no further cycles today
    assert next_gap_minutes(CFG, 5, start + timedelta(minutes=200), start) is None


def test_pace_status_flags_unused_quota_at_risk():
    start = competition_day_start()
    healthy = pace_status(CFG, 2, start + timedelta(minutes=100), start)
    assert healthy["quota_left"] == 3
    assert not healthy["quota_at_risk"]

    # 23h into the day, 3 submissions still unspent: cannot fit them before the rollover
    tight = pace_status(CFG, 2, start + timedelta(hours=23), start)
    assert tight["quota_at_risk"]
    assert tight["behind"]


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
