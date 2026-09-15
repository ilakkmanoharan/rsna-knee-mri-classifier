"""Daily Agent-1 daemon: wait for 01:00 America/Chicago, then cycle every 90 minutes until quota.

On errors: keep retrying (never exit the daily loop) and email the operator.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
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
from agent.notify import send_alert
from agent.run_cycle import run_cycle
from agent.state import StateStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("agent1.daily")


def sleep_until_day_start(hour: int, tz_name: str) -> None:
    """Start as soon as the competition day starts (01:00 CST). If already started, begin now."""
    now = now_cst(tz_name)
    start = competition_day_start(now, hour=hour, tz_name=tz_name)
    if now < start:
        wait = (start - now).total_seconds()
        logger.info("Waiting %.0fs until competition day start %s", wait, start.isoformat())
        time.sleep(wait)
    else:
        logger.info("Competition day already started at %s (now %s)", start.isoformat(), now.isoformat())


def _retry_sleep_seconds(cfg: dict, attempt: int) -> float:
    base = int((cfg.get("resilience") or {}).get("retry_base_seconds", 300))
    cap = int((cfg.get("resilience") or {}).get("retry_max_seconds", 1800))
    return float(min(cap, base * (2 ** max(0, attempt - 1))))


def _alert_error(cfg: dict, where: str, exc: BaseException, extra: str = "") -> None:
    day_id = competition_day_id(
        hour=int(cfg.get("day_start_hour_cst", 1)),
        tz_name=cfg.get("day_start_tz", "America/Chicago"),
    )
    tb = traceback.format_exc()
    subject = f"[RSNA Agent1] ERROR on {day_id}: {where}"
    body = (
        f"Agent-1 hit an error and will keep retrying.\n\n"
        f"When: {now_cst(cfg.get('day_start_tz', 'America/Chicago')).isoformat()}\n"
        f"Where: {where}\n"
        f"Error: {type(exc).__name__}: {exc}\n"
        f"{extra}\n\n"
        f"Traceback:\n{tb}\n"
        f"Repo: {cfg.get('github_repo')}\n"
        f"Logs: artifacts/agent_state/agent1.stdout.log / agent1.stderr.log\n"
    )
    result = send_alert(subject, body, cfg=cfg, state_dir=ROOT / cfg["paths"]["state"])
    logger.info("Alert dispatched: %s", result)


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
    consecutive_failures = 0

    if not args.no_wait_for_day_start and not args.once:
        try:
            sleep_until_day_start(hour, tz)
        except Exception as exc:  # noqa: BLE001
            _alert_error(cfg, "sleep_until_day_start", exc)
            # Fall through and keep running

    while True:
        try:
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
                # Sleep in chunks so a clock skew / wake still progresses
                remaining = wait + 5
                while remaining > 0:
                    chunk = min(300.0, remaining)
                    time.sleep(chunk)
                    remaining -= chunk
                consecutive_failures = 0
                continue

            result = run_cycle(cfg, skip_submit=args.skip_submit)
            logger.info("Cycle result: %s", json.dumps(result, default=str)[:500])
            consecutive_failures = 0

            if args.once:
                break

            state = store.load(competition_day_id(hour=hour, tz_name=tz))
            if state.submissions_used >= max_sub:
                logger.info("Reached daily submission cap.")
                wait = seconds_until_next_day_start(hour=hour, tz_name=tz)
                remaining = wait + 5
                while remaining > 0:
                    chunk = min(300.0, remaining)
                    time.sleep(chunk)
                    remaining -= chunk
                continue

            logger.info("Sleeping %d seconds until next cycle", interval)
            time.sleep(interval)

        except Exception as exc:  # noqa: BLE001
            consecutive_failures += 1
            retry_in = _retry_sleep_seconds(cfg, consecutive_failures)
            logger.exception(
                "Cycle failed (attempt %d). Will retry in %.0fs.",
                consecutive_failures,
                retry_in,
            )
            _alert_error(
                cfg,
                "run_daily_loop",
                exc,
                extra=f"consecutive_failures={consecutive_failures}\nretry_in_seconds={retry_in}",
            )
            if args.once:
                # Still alert, then exit non-zero for --once debugging
                raise
            time.sleep(retry_in)
            # loop continues — keep trying


if __name__ == "__main__":
    main()
