"""Persistent agent state (quota, cycles)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class DayState:
    day_id: str
    submissions_used: int = 0
    cycles_completed: int = 0
    last_cycle_at: Optional[str] = None
    last_submission_ref: Optional[str] = None
    last_public_score: Optional[float] = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DayState":
        return cls(
            day_id=d["day_id"],
            submissions_used=int(d.get("submissions_used", 0)),
            cycles_completed=int(d.get("cycles_completed", 0)),
            last_cycle_at=d.get("last_cycle_at"),
            last_submission_ref=d.get("last_submission_ref"),
            last_public_score=d.get("last_public_score"),
            history=list(d.get("history") or []),
        )


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self, day_id: str) -> DayState:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            if data.get("day_id") == day_id:
                return DayState.from_dict(data)
        return DayState(day_id=day_id)

    def save(self, state: DayState) -> None:
        self.path.write_text(json.dumps(state.to_dict(), indent=2))
