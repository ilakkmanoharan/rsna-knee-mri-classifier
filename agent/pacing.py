"""Adaptive submission pacing.

No fixed slots: the goal is to spend all five daily Kaggle submissions, spacing them
30-60 minutes apart depending on how much of the competition day is left and how many
submissions still need to go out. The gap shrinks toward `min_interval_minutes` when the
agent is behind and relaxes toward `max_interval_minutes` when it is on track.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

DEFAULTS = {
    "min_interval_minutes": 30,
    "max_interval_minutes": 60,
    "target_finish_hours": 4.0,      # aim to have all submissions in this early
    "grace_minutes": 20,             # slack before a submission counts as late
    "hard_stop_buffer_minutes": 60,  # stop submitting this long before the day rolls over
}


def pace_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    p = dict(DEFAULTS)
    p.update({k: v for k, v in (cfg.get("pacing") or {}).items() if v is not None})
    p["quota"] = int(cfg.get("max_submissions_per_day", 5))
    p["min_interval_minutes"] = float(p["min_interval_minutes"])
    p["max_interval_minutes"] = max(float(p["max_interval_minutes"]), p["min_interval_minutes"])
    p["target_finish_hours"] = float(p["target_finish_hours"])
    p["grace_minutes"] = float(p["grace_minutes"])
    p["hard_stop_buffer_minutes"] = float(p["hard_stop_buffer_minutes"])
    return p


def expected_submissions(cfg: dict[str, Any], now: datetime, day_start: datetime) -> int:
    """How many submissions should already be in by `now` at the slowest acceptable pace."""
    p = pace_cfg(cfg)
    elapsed = (now - day_start).total_seconds() / 60.0
    if elapsed < p["grace_minutes"]:
        return 0
    n = 1 + int((elapsed - p["grace_minutes"]) // p["max_interval_minutes"])
    return int(min(p["quota"], max(0, n)))


def deadline(cfg: dict[str, Any], now: datetime, day_start: datetime) -> datetime:
    """When the day's submissions must be finished: target finish, or the day's hard edge."""
    p = pace_cfg(cfg)
    target = day_start + timedelta(hours=p["target_finish_hours"])
    hard = day_start + timedelta(days=1) - timedelta(minutes=p["hard_stop_buffer_minutes"])
    if now >= target:
        return hard
    return min(target, hard)


def next_gap_minutes(
    cfg: dict[str, Any], used: int, now: datetime, day_start: datetime
) -> Optional[float]:
    """Minutes to wait before the next cycle, or None when the daily quota is spent."""
    p = pace_cfg(cfg)
    remaining = p["quota"] - int(used)
    if remaining <= 0:
        return None
    minutes_left = (deadline(cfg, now, day_start) - now).total_seconds() / 60.0
    if minutes_left <= 0:
        return p["min_interval_minutes"]
    even_spread = minutes_left / remaining
    return float(min(p["max_interval_minutes"], max(p["min_interval_minutes"], even_spread)))


def pace_status(cfg: dict[str, Any], used: Optional[int], now: datetime, day_start: datetime) -> dict[str, Any]:
    """Cadence snapshot shared by the daemon, the cycle runner and the supervisor."""
    p = pace_cfg(cfg)
    used_n = int(used or 0)
    expected = expected_submissions(cfg, now, day_start)
    gap = next_gap_minutes(cfg, used_n, now, day_start)
    dl = deadline(cfg, now, day_start)
    return {
        "quota": p["quota"],
        "used": used if used is not None else None,
        "expected_by_now": expected,
        "deficit": max(0, expected - used_n),
        "quota_left": max(0, p["quota"] - used_n),
        "behind": used_n < expected,
        "min_interval_minutes": p["min_interval_minutes"],
        "max_interval_minutes": p["max_interval_minutes"],
        "grace_minutes": p["grace_minutes"],
        "next_gap_minutes": gap,
        "deadline": dl.isoformat(),
        "minutes_to_deadline": round((dl - now).total_seconds() / 60.0, 1),
        "quota_at_risk": bool(
            p["quota"] - used_n > 0 and (dl - now).total_seconds() / 60.0 < (p["quota"] - used_n) * p["min_interval_minutes"]
        ),
    }
