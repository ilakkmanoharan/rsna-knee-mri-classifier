"""Competition clock helpers (America/Chicago / CST-CDT)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo


def competition_tz(name: str = "America/Chicago") -> ZoneInfo:
    return ZoneInfo(name)


def now_cst(tz_name: str = "America/Chicago") -> datetime:
    return datetime.now(tz=competition_tz(tz_name))


def competition_day_start(when: Optional[datetime] = None, hour: int = 1, tz_name: str = "America/Chicago") -> datetime:
    """Return the 01:00 America/Chicago boundary for the competition day containing `when`."""
    tz = competition_tz(tz_name)
    when = when or datetime.now(tz=tz)
    if when.tzinfo is None:
        when = when.replace(tzinfo=tz)
    else:
        when = when.astimezone(tz)
    start = when.replace(hour=hour, minute=0, second=0, microsecond=0)
    if when < start:
        start = start - timedelta(days=1)
    return start


def competition_day_id(when: Optional[datetime] = None, hour: int = 1, tz_name: str = "America/Chicago") -> str:
    return competition_day_start(when, hour=hour, tz_name=tz_name).strftime("%Y-%m-%d")


def seconds_until_next_day_start(hour: int = 1, tz_name: str = "America/Chicago") -> float:
    tz = competition_tz(tz_name)
    now = datetime.now(tz=tz)
    start = competition_day_start(now, hour=hour, tz_name=tz_name)
    nxt = start + timedelta(days=1)
    return max(0.0, (nxt - now).total_seconds())


def cycle_index(when: Optional[datetime] = None, hour: int = 1, interval_min: int = 90, tz_name: str = "America/Chicago") -> int:
    """0-based cycle index within the competition day."""
    tz = competition_tz(tz_name)
    when = (when or datetime.now(tz=tz)).astimezone(tz)
    start = competition_day_start(when, hour=hour, tz_name=tz_name)
    elapsed = (when - start).total_seconds()
    return int(elapsed // (interval_min * 60))


def stamp(tz_name: str = "America/Chicago") -> str:
    return now_cst(tz_name).strftime("%Y%m%dT%H%M%S%z")
