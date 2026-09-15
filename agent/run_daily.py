"""Daily Agent-1 daemon: wait for 01:00 America/Chicago, then cycle every 90 minutes until quota."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.clock import (
    competition_day_id,
    competition_day_start,
    now_cst,
    seconds_until_next_day_start,
)
from agent.run_cycle import run_cycle
from agent.state import StateStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("agent1.daily")


def sleep_until_day_start(hour: int, tz_name: str) -> None:
    """If we're before today's 01:00 boundary window logic: always align to next cycle boundary.

    Spec: start as soon as the competition day starts (01:00 CST).
    If launched after day start, begin immediately.
    If launched before day start, sleep until it.
    """
    now = now_cst(tz_name)
    start = competition_day_start(now, hour=hour, tz_name=tz_name)
    # If current time is before today's wall-clock hour on a fresh calendar sense:
    # competition_day_start already returned the active day's 01:00.
    # If now is before start, we're in previous day — sleep until start.
    if now < start:
        wait = (start - now).total_seconds()
        logger.info("Waiting %.0fs until competition day start %s", wait, start.isoformat())
        time.sleep(wait)
    else:
        logger.info("Competition day already started at %s (now %s)", start.isoformat(), now.isoformat())


def main() -> None:
    ap = argparse.ArgumentParser(description="Agent-1 daily loop")
    ap.add_argument("--config", default=str(ROOT / "agent" / "config.yaml"))
    ap.add_argument("--skip-submit", action="store_true")
    ap.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    ap.add_argument("--no-wait-for-day-start", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    tz = cfg.get("day_start_tz", "America/Chicago")
    hour = int(cfg.get("day_start_hour_cst", 1))
    interval = int(cfg.get("cycle_interval_minutes", 90)) * 60
    max_sub = int(cfg.get("max_submissions_per_day", 5))

    if not args.no_wait_for_day_start and not args.once:
        sleep_until_day_start(hour, tz)

    while True:
        day_id = competition_day_id(hour=hour, tz_name=tz)
        store = StateStore(ROOT / cfg["paths"]["state"] / "day_state.json")
        state = store.load(day_id)
        if state.submissions_used >= max_sub:
            wait = seconds_until_next_day_start(hour=hour, tz_name=tz)
            logger.info(
                "Quota full for %s (%d/%d). Sleeping %.0fs until next competition day.",
                day_id,
                state.submissions_used,
                max_sub,
                wait,
            )
            if args.once:
                break
            time.sleep(min(wait + 5, wait + 60))
            continue

        result = run_cycle(cfg, skip_submit=args.skip_submit)
        logger.info("Cycle result: %s", json.dumps(result, default=str)[:500])
        if args.once:
            break

        # Reload state after cycle
        state = store.load(competition_day_id(hour=hour, tz_name=tz))
        if state.submissions_used >= max_sub:
            logger.info("Reached daily submission cap.")
            wait = seconds_until_next_day_start(hour=hour, tz_name=tz)
            time.sleep(min(wait + 5, wait + 60))
            continue

        logger.info("Sleeping %d seconds until next cycle", interval)
        time.sleep(interval)


if __name__ == "__main__":
    main()
