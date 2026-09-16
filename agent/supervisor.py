"""Grok supervisor bot for agent1.

Audits whether the Cursor/GitHub agent actually performed the three supervised stages of
private/agent/agent1.md for every due 90-minute slot of the competition day:

  Research/    - literature write-up: which methods to consider, why, how they raise the score
  Analysis/    - previous Kaggle submission logs: why the score was low, how to improve
  Hypothesis/  - falsifiable hypotheses derived from Research + Analysis

then checks the Kaggle submission cadence (5 per competition day, one per 90 minutes), asks Grok
for a verdict, remediates by re-running the agent through GitHub Actions, and escalates to the
operator by email + GitHub issue when a human decision is required.

Usage:
  python agent/supervisor.py                 # audit + remediate + report + ask
  python agent/supervisor.py --mode audit    # read-only audit (no git, no dispatch, no email)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import github_api, grok
from agent.clock import competition_day_id, competition_day_start, now_cst, stamp
from agent.git_sync import commit_and_push
from agent.notify import send_alert

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("grok.supervisor")

STAGES = ("research", "analysis", "hypothesis")
DEFAULT_MIN_CHARS = {"research": 1200, "analysis": 700, "hypothesis": 500}
DEFAULT_GRACE_MIN = 25


# --------------------------------------------------------------------------------------
# config helpers
# --------------------------------------------------------------------------------------
def load_cfg(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def sup_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("supervisor") or {}


def stage_dirs(cfg: dict[str, Any], stage: str) -> list[Path]:
    """Primary folder from paths.<stage> plus any legacy aliases still holding artifacts."""
    paths = cfg.get("paths") or {}
    names = [str(paths.get(stage) or stage.capitalize())]
    aliases = sup_cfg(cfg).get(f"{stage}_aliases") or []
    names += [str(a) for a in aliases]
    seen: list[Path] = []
    for n in names:
        p = ROOT / n
        if p not in seen:
            seen.append(p)
    return seen


# --------------------------------------------------------------------------------------
# stage artifact audit
# --------------------------------------------------------------------------------------
def _cycle_num(name: str, day_id: str) -> Optional[int]:
    m = re.match(rf"^{re.escape(day_id)}_cycle(\d+)_", name)
    return int(m.group(1)) if m else None


def _research_sources(md_path: Path) -> Optional[int]:
    """Number of papers the research stage actually retrieved, from its sibling JSON."""
    side = md_path.with_name(md_path.name.replace("_research.md", "_research.json"))
    if not side.exists():
        return None
    try:
        return len(json.loads(side.read_text()).get("papers") or [])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unreadable research JSON %s: %s", side, exc)
        return None


def _quality_issues(stage: str, text: str, min_chars: int) -> list[str]:
    issues: list[str] = []
    low = text.lower()
    if len(text) < min_chars:
        issues.append(f"{stage} write-up is thin ({len(text)} chars < {min_chars} expected)")
    if stage == "research":
        links = len(re.findall(r"https?://", text))
        if links < 3:
            issues.append(f"research cites only {links} sources (expected >= 3 paper/web links)")
        if not any(k in low for k in ("why", "because", "rationale")):
            issues.append("research does not explain *why* each method should raise the score")
        if not any(k in low for k in ("score", "auc", "metric")):
            issues.append("research is not tied to the competition metric")
    elif stage == "analysis":
        if not any(k in low for k in ("publicscore", "public score", "score")):
            issues.append("analysis does not reference previous submission scores/logs")
        if not any(k in low for k in ("why the score", "low", "regress")):
            issues.append("analysis does not diagnose why the score was low")
        if not any(k in low for k in ("improve", "next", "fix")):
            issues.append("analysis proposes no improvement path")
    elif stage == "hypothesis":
        if not any(k in low for k in ("if ", "then", "predict", "expect")):
            issues.append("hypotheses are not stated as testable if/then predictions")
        if not any(k in low for k in ("auc", "metric", "oof", "score")):
            issues.append("hypotheses have no measurable acceptance criterion")
        if not any(k in low for k in ("research", "analysis")):
            issues.append("hypotheses are not linked back to Research + Analysis")
    return issues


def audit_stage(cfg: dict[str, Any], stage: str, day_id: str) -> dict[str, Any]:
    min_chars = int((sup_cfg(cfg).get("min_chars") or {}).get(stage, DEFAULT_MIN_CHARS[stage]))
    kind = {"research": "research", "analysis": "analysis", "hypothesis": "hypotheses"}[stage]
    dirs = stage_dirs(cfg, stage)
    files: list[dict[str, Any]] = []
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob(f"{day_id}_cycle*_{kind}.md")):
            cyc = _cycle_num(p.name, day_id)
            if cyc is None:
                continue
            text = p.read_text(errors="replace")
            issues = _quality_issues(stage, text, min_chars)
            papers = _research_sources(p) if stage == "research" else None
            if papers is not None and papers < 3:
                issues.append(
                    f"research stage retrieved {papers} paper(s) — the internet/arXiv search returned "
                    "nothing usable, so the write-up is not grounded in literature"
                )
            files.append(
                {
                    "cycle": cyc,
                    "path": str(p.relative_to(ROOT)),
                    "chars": len(text),
                    "papers_retrieved": papers,
                    "issues": issues,
                    "excerpt": text[:900],
                }
            )
    files.sort(key=lambda f: (f["cycle"], f["path"]))
    cycles = sorted({f["cycle"] for f in files})
    latest = files[-1] if files else None
    return {
        "folders": [str(d.relative_to(ROOT)) for d in dirs],
        "folder_exists": any(d.exists() for d in dirs),
        "cycles_with_artifact": cycles,
        "count_today": len(cycles),
        "latest": latest,
        "quality_issues": (latest or {}).get("issues", []),
        "min_chars": min_chars,
    }


# --------------------------------------------------------------------------------------
# external state: Kaggle + GitHub Actions
# --------------------------------------------------------------------------------------
def kaggle_submissions_today(cfg: dict[str, Any], day_start: datetime) -> dict[str, Any]:
    try:
        from agent.stages.analyze import fetch_submissions

        rows = fetch_submissions(cfg["competition"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Kaggle submission history unavailable: %s", exc)
        return {"available": False, "error": f"{type(exc).__name__}: {exc}", "count_today": None, "rows": []}

    today: list[dict[str, Any]] = []
    for r in rows:
        status = str(r.get("status") or "").upper()
        if not any(x in status for x in ("COMPLETE", "PENDING", "SCORING", "SUBMITTED", "RUNNING")):
            continue
        try:
            dt = datetime.fromisoformat(str(r.get("date") or "").replace("Z", "+00:00"))
            dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
            if dt.astimezone(day_start.tzinfo) < day_start:
                continue
        except Exception:  # noqa: BLE001
            pass  # undated rows: count conservatively
        today.append(r)
    scores = [r.get("publicScore") for r in today if r.get("publicScore") is not None]
    all_scores = [r.get("publicScore") for r in rows if r.get("publicScore") is not None]
    return {
        "available": True,
        "count_today": len(today),
        "best_today": max(scores) if scores else None,
        "best_all_time": max(all_scores) if all_scores else None,
        "errored_today": [r for r in today if "ERROR" in str(r.get("status") or "").upper()],
        "rows": today[:10],
    }


def workflow_state(cfg: dict[str, Any], day_start: datetime) -> dict[str, Any]:
    wf = sup_cfg(cfg).get("workflow_file") or "agent1-cloud.yml"
    runs = github_api.list_workflow_runs(wf, cfg=cfg)
    today = []
    for r in runs:
        try:
            created = datetime.fromisoformat(str(r["created_at"]).replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            continue
        if created.astimezone(day_start.tzinfo) >= day_start:
            today.append(r)
    return {
        "workflow_file": wf,
        "token_present": bool(github_api.token()),
        "can_dispatch": github_api.has_pat(),
        "runs_today": today,
        "failed_today": [r for r in today if r.get("conclusion") == "failure"],
        "in_progress": [r for r in today if r.get("status") in {"in_progress", "queued", "requested", "waiting"}],
    }


# --------------------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------------------
def slots_due(cfg: dict[str, Any], now: datetime, day_start: datetime) -> dict[str, Any]:
    interval = int(cfg.get("cycle_interval_minutes", 90))
    max_sub = int(cfg.get("max_submissions_per_day", 5))
    grace = int(sup_cfg(cfg).get("grace_minutes", DEFAULT_GRACE_MIN))
    schedule = []
    due = 0
    for i in range(max_sub):
        slot_at = day_start + timedelta(minutes=interval * i)
        is_due = now >= slot_at + timedelta(minutes=grace)
        schedule.append({"slot": i, "at": slot_at.isoformat(), "due": is_due})
        due += int(is_due)
    next_slot = next((s for s in schedule if not s["due"]), None)
    return {
        "interval_minutes": interval,
        "max_per_day": max_sub,
        "grace_minutes": grace,
        "due_count": due,
        "schedule": schedule,
        "next_slot_at": (next_slot or {}).get("at"),
    }


def build_audit(cfg: dict[str, Any], now: Optional[datetime] = None) -> dict[str, Any]:
    tz = cfg.get("day_start_tz", "America/Chicago")
    hour = int(cfg.get("day_start_hour_cst", 1))
    now = now or now_cst(tz)
    day_start = competition_day_start(now, hour=hour, tz_name=tz)
    day_id = competition_day_id(now, hour=hour, tz_name=tz)

    stages = {s: audit_stage(cfg, s, day_id) for s in STAGES}
    timing = slots_due(cfg, now, day_start)
    kaggle = kaggle_submissions_today(cfg, day_start)
    actions = workflow_state(cfg, day_start)

    complete_cycles = sorted(
        set(stages["research"]["cycles_with_artifact"])
        & set(stages["analysis"]["cycles_with_artifact"])
        & set(stages["hypothesis"]["cycles_with_artifact"])
    )
    due = int(timing["due_count"])
    max_sub = int(timing["max_per_day"])
    submissions = kaggle["count_today"]

    artifact_deficit = max(0, due - len(complete_cycles))
    submission_deficit = max(0, due - submissions) if submissions is not None else 0
    quota_left = max(0, max_sub - submissions) if submissions is not None else max_sub

    violations: list[str] = []
    for s in STAGES:
        st = stages[s]
        folder = st["folders"][0]
        if not st["folder_exists"]:
            violations.append(f"{folder}/ folder is missing (agent1.md requires it)")
        elif st["count_today"] < due:
            violations.append(
                f"{folder}/ has {st['count_today']} write-up(s) for day {day_id} but {due} slot(s) are due"
            )
        violations += [f"{folder}/: {i}" for i in st["quality_issues"]]
    if submissions is not None and submission_deficit:
        violations.append(f"Kaggle submissions today: {submissions} of {due} due (cap {max_sub})")
    if kaggle.get("errored_today"):
        violations.append(f"{len(kaggle['errored_today'])} Kaggle submission(s) today ended in ERROR")
    if actions["failed_today"]:
        violations.append(f"{len(actions['failed_today'])} agent workflow run(s) failed today")

    remediate = bool(quota_left > 0 and (artifact_deficit or submission_deficit) and not actions["in_progress"])

    return {
        "generated_at": now.isoformat(),
        "day_id": day_id,
        "day_start": day_start.isoformat(),
        "competition": cfg.get("competition"),
        "timing": timing,
        "stages": stages,
        "cycles_with_full_pipeline": complete_cycles,
        "kaggle": kaggle,
        "github_actions": actions,
        "deficits": {
            "artifact_deficit": artifact_deficit,
            "submission_deficit": submission_deficit,
            "quota_left": quota_left,
        },
        "violations": violations,
        "remediation_needed": remediate,
    }


# --------------------------------------------------------------------------------------
# supervisor state (cooldowns, dedupe)
# --------------------------------------------------------------------------------------
def state_path(cfg: dict[str, Any]) -> Path:
    return ROOT / (cfg.get("paths") or {}).get("state", "artifacts/agent_state") / "supervisor_state.json"


def load_state(cfg: dict[str, Any], day_id: str) -> dict[str, Any]:
    p = state_path(cfg)
    if p.exists():
        try:
            data = json.loads(p.read_text())
            if data.get("day_id") == day_id:
                return data
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unreadable supervisor state: %s", exc)
    return {"day_id": day_id, "dispatches": [], "asked": {}, "verdicts": []}


def save_state(cfg: dict[str, Any], state: dict[str, Any]) -> None:
    p = state_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


def dispatch_allowed(cfg: dict[str, Any], state: dict[str, Any], now: datetime) -> tuple[bool, str]:
    s = sup_cfg(cfg)
    max_disp = int(s.get("max_dispatches_per_day", 6))
    cooldown = int(s.get("dispatch_cooldown_minutes", 40))
    dispatches = state.get("dispatches") or []
    if len(dispatches) >= max_disp:
        return False, f"dispatch cap reached ({len(dispatches)}/{max_disp} today)"
    if dispatches:
        try:
            last = datetime.fromisoformat(dispatches[-1])
            wait = cooldown - (now - last.astimezone(now.tzinfo)).total_seconds() / 60
            if wait > 0:
                return False, f"cooldown active ({wait:.0f} min left)"
        except Exception:  # noqa: BLE001
            pass
    return True, "ok"


# --------------------------------------------------------------------------------------
# report + escalation
# --------------------------------------------------------------------------------------
def render_report(audit: dict[str, Any], verdict: dict[str, Any], actions_taken: list[str]) -> str:
    stages = audit["stages"]
    lines = [
        f"# Grok supervision report — day {audit['day_id']}",
        "",
        f"_Generated {audit['generated_at']} by the Grok supervisor bot._",
        "",
        f"- Verdict: **{verdict.get('verdict', 'unknown')}** "
        f"({'Grok' if verdict.get('source') == 'grok' else 'rule-based'} review)",
        f"- Slots due so far: {audit['timing']['due_count']} / {audit['timing']['max_per_day']}"
        f" (every {audit['timing']['interval_minutes']} min from {audit['day_start']})",
        f"- Cycles with Research + Analysis + Hypothesis: {audit['cycles_with_full_pipeline']}",
        f"- Kaggle submissions today: {audit['kaggle'].get('count_today')} "
        f"(best {audit['kaggle'].get('best_today')}, best all-time {audit['kaggle'].get('best_all_time')})",
        f"- Next slot: {audit['timing'].get('next_slot_at')}",
        "",
        "## Supervised stages (agent1.md steps 1-3)",
        "",
        "| stage | folder | write-ups today | latest file | quality issues |",
        "|---|---|---|---|---|",
    ]
    for s in STAGES:
        st = stages[s]
        latest = (st.get("latest") or {}).get("path", "—")
        lines.append(
            f"| {s} | `{st['folders'][0]}` | {st['count_today']} | `{latest}` | "
            f"{len(st['quality_issues'])} |"
        )
    lines += ["", "## Violations", ""]
    lines += [f"- {v}" for v in audit["violations"]] or ["- none"]
    if verdict.get("stage_findings"):
        lines += ["", "## Grok stage findings", ""]
        for k, v in (verdict["stage_findings"] or {}).items():
            lines.append(f"- **{k}**: {v}")
    if verdict.get("recommended_actions"):
        lines += ["", "## Recommended actions", ""]
        lines += [f"- {a}" for a in verdict["recommended_actions"]]
    lines += ["", "## Actions taken by the bot", ""]
    lines += [f"- {a}" for a in actions_taken] or ["- none"]
    if verdict.get("questions_for_human"):
        lines += ["", "## Questions for the operator", ""]
        lines += [f"- {q}" for q in verdict["questions_for_human"]]
    lines.append("")
    return "\n".join(lines)


def ask_human(
    questions: list[str],
    audit: dict[str, Any],
    verdict: dict[str, Any],
    cfg: dict[str, Any],
    state: dict[str, Any],
) -> list[str]:
    """Open/comment a GitHub issue and email the operator, deduped per day."""
    acted: list[str] = []
    ask = sup_cfg(cfg).get("ask") or {}
    fresh = []
    for q in questions:
        key = hashlib.sha1(q.strip().lower().encode()).hexdigest()[:12]
        if key in (state.get("asked") or {}):
            continue
        state.setdefault("asked", {})[key] = audit["generated_at"]
        fresh.append(q)
    if not fresh:
        return acted

    marker = f"[grok-bot] needs input — day {audit['day_id']}"
    body_lines = [
        f"The Grok supervisor bot needs a decision from you (verdict: **{verdict.get('verdict')}**).",
        "",
        "### Questions",
        *[f"- [ ] {q}" for q in fresh],
        "",
        "### Why",
        *[f"- {v}" for v in audit["violations"][:10]],
        "",
        f"Slots due: {audit['timing']['due_count']}/{audit['timing']['max_per_day']} · "
        f"submissions today: {audit['kaggle'].get('count_today')} · "
        f"full pipelines: {audit['cycles_with_full_pipeline']}",
        "",
        "Reply in a comment on this issue; the bot reads open issues on its next run.",
    ]
    body = "\n".join(body_lines)

    if ask.get("github_issue", True):
        existing = github_api.find_open_issue(marker, cfg=cfg)
        if existing:
            if github_api.comment_issue(int(existing["number"]), body, cfg=cfg):
                acted.append(f"commented on issue #{existing['number']} with {len(fresh)} question(s)")
        else:
            issue = github_api.create_issue(
                marker,
                body,
                labels=list(ask.get("issue_labels") or ["grok-bot"]),
                assignees=list(ask.get("assignees") or []),
                cfg=cfg,
            )
            if issue:
                acted.append(f"opened issue #{issue.get('number')} with {len(fresh)} question(s)")

    if ask.get("email", True):
        res = send_alert(
            f"[grok-bot] needs your input — RSNA knee day {audit['day_id']}",
            body,
            cfg=cfg,
            state_dir=ROOT / (cfg.get("paths") or {}).get("state", "artifacts/agent_state"),
        )
        acted.append(f"emailed {res.get('to')} (sent={res.get('emailed')})")
    return acted


def rule_verdict(audit: dict[str, Any]) -> dict[str, Any]:
    d = audit["deficits"]
    if not audit["violations"]:
        v = "on_track"
    elif d["artifact_deficit"] >= 2 or audit["github_actions"]["failed_today"]:
        v = "failing"
    else:
        v = "degraded"
    questions: list[str] = []
    ga = audit["github_actions"]
    if not ga["token_present"]:
        questions.append(
            "No GitHub token is visible to the bot. Add a repo secret AGENT1_GH_PAT "
            "(scopes: repo, workflow) so I can re-trigger agent runs and open issues."
        )
    elif not ga["can_dispatch"]:
        questions.append(
            "Only the built-in GITHUB_TOKEN is available; GitHub ignores workflow_dispatch from it. "
            "Add secret AGENT1_GH_PAT (repo + workflow scopes) so I can re-run agent1-cloud.yml directly."
        )
    if not audit["kaggle"].get("available"):
        questions.append(
            f"I cannot read Kaggle submission history ({audit['kaggle'].get('error')}). "
            "Confirm KAGGLE_USERNAME / KAGGLE_KEY secrets are valid."
        )
    if not grok.available():
        questions.append(
            "XAI_API_KEY is not set, so I am running rule-based checks without Grok's review. "
            "Add the XAI_API_KEY repo secret to enable full Grok supervision."
        )
    if audit["kaggle"].get("errored_today"):
        questions.append(
            "A Kaggle submission errored today. Should I keep spending quota on the current strategy "
            "or fall back to the last known-good notebook?"
        )
    return {
        "verdict": v,
        "source": "rules",
        "violations": audit["violations"],
        "recommended_actions": ["Re-run the agent cycle to fill the missing slot"] if audit["remediation_needed"] else [],
        "rerun_cycle": audit["remediation_needed"],
        "questions_for_human": questions,
        "confidence": 0.5,
    }


def grok_payload(audit: dict[str, Any]) -> dict[str, Any]:
    """Compact audit for the model: keep excerpts, drop bulky rows."""
    trimmed = json.loads(json.dumps(audit, default=str))
    for s in STAGES:
        st = trimmed["stages"][s]
        if st.get("latest"):
            st["latest"]["excerpt"] = (st["latest"].get("excerpt") or "")[:900]
    trimmed["kaggle"]["rows"] = trimmed["kaggle"].get("rows", [])[:5]
    trimmed["github_actions"]["runs_today"] = trimmed["github_actions"].get("runs_today", [])[:5]
    return trimmed


# --------------------------------------------------------------------------------------
# main supervise flow
# --------------------------------------------------------------------------------------
def supervise(
    cfg: dict[str, Any],
    *,
    mode: str = "supervise",
    allow_dispatch: bool = True,
    allow_ask: bool = True,
    allow_git: bool = True,
) -> dict[str, Any]:
    tz = cfg.get("day_start_tz", "America/Chicago")
    now = now_cst(tz)
    audit = build_audit(cfg, now=now)
    state = load_state(cfg, audit["day_id"])

    grok_enabled = bool((sup_cfg(cfg).get("grok") or {}).get("enabled", True))
    verdict = grok.review(grok_payload(audit), cfg=cfg) if grok_enabled else None
    if verdict:
        verdict["source"] = "grok"
        rules = rule_verdict(audit)
        # Rule-based questions (missing secrets etc.) are facts Grok cannot see; always keep them.
        verdict["questions_for_human"] = list(
            dict.fromkeys(list(verdict.get("questions_for_human") or []) + rules["questions_for_human"])
        )
        verdict["violations"] = list(dict.fromkeys(list(verdict.get("violations") or []) + audit["violations"]))
    else:
        verdict = rule_verdict(audit)

    actions_taken: list[str] = []
    needs_rerun = bool(audit["remediation_needed"] or verdict.get("rerun_cycle"))
    if needs_rerun and audit["deficits"]["quota_left"] <= 0:
        needs_rerun = False
        actions_taken.append("no rerun: daily submission quota already used")
    if needs_rerun and audit["github_actions"]["in_progress"]:
        needs_rerun = False
        actions_taken.append("no rerun: an agent workflow run is already in progress")

    dispatched = False
    if mode == "supervise" and needs_rerun:
        ok, why = dispatch_allowed(cfg, state, now)
        if not ok:
            actions_taken.append(f"no dispatch: {why}")
        elif not allow_dispatch:
            actions_taken.append("dispatch disabled by flag; leaving remediation to the caller")
        elif not github_api.has_pat():
            actions_taken.append(
                "cannot dispatch agent1-cloud.yml without an AGENT1_GH_PAT secret; "
                "falling back to inline remediation"
            )
        else:
            wf = audit["github_actions"]["workflow_file"]
            dispatched = github_api.dispatch_workflow(
                wf, ref=str(cfg.get("branch", "main")), inputs={"force": "false"}, cfg=cfg
            )
            if dispatched:
                state.setdefault("dispatches", []).append(now.isoformat())
                actions_taken.append(f"dispatched {wf} to run the missing cycle")
            else:
                actions_taken.append(f"failed to dispatch {wf} (see log); inline remediation required")

    # Tell the calling workflow whether it should run the cycle itself (no PAT needed).
    self_heal = bool(mode == "supervise" and needs_rerun and not dispatched)
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        reason = "; ".join(audit["violations"][:3]) or "deficit detected"
        with open(gh_out, "a") as fh:
            fh.write(f"remediate={'true' if self_heal else 'false'}\n")
            fh.write(f"verdict={verdict.get('verdict', 'unknown')}\n")
            fh.write(f"reason={reason[:400]}\n")
    if self_heal:
        actions_taken.append("requested inline remediation (supervisor workflow runs the cycle itself)")

    if mode == "supervise" and allow_ask:
        actions_taken += ask_human(list(verdict.get("questions_for_human") or []), audit, verdict, cfg, state)

    report_dir = ROOT / (sup_cfg(cfg).get("report_dir") or "Supervision")
    report_path = report_dir / f"{audit['day_id']}_{stamp(tz)}_supervision.md"
    if mode == "supervise":
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_report(audit, verdict, actions_taken))
        report_path.with_suffix(".json").write_text(
            json.dumps({"audit": audit, "verdict": verdict, "actions": actions_taken}, indent=2, default=str)
        )
        state.setdefault("verdicts", []).append(
            {"at": audit["generated_at"], "verdict": verdict.get("verdict"), "report": str(report_path.relative_to(ROOT))}
        )
        save_state(cfg, state)
        if allow_git and cfg.get("git", {}).get("auto_commit", True):
            commit_and_push(
                [str(report_dir.relative_to(ROOT)), (cfg.get("paths") or {}).get("state", "artifacts/agent_state")],
                message=f"grok-bot: supervision {audit['day_id']} verdict={verdict.get('verdict')} "
                f"due={audit['timing']['due_count']} pipelines={len(audit['cycles_with_full_pipeline'])}",
                branch=str(cfg.get("branch", "main")),
                auto_push=bool(cfg.get("git", {}).get("auto_push", True)),
            )

    return {
        "verdict": verdict,
        "audit": audit,
        "actions": actions_taken,
        "needs_rerun": needs_rerun,
        "dispatched": dispatched,
        "self_heal_requested": self_heal,
        "report": str(report_path) if mode == "supervise" else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Grok supervisor for agent1")
    ap.add_argument("--config", default=str(ROOT / "agent" / "config.yaml"))
    ap.add_argument("--mode", choices=["audit", "supervise"], default="supervise")
    ap.add_argument("--no-dispatch", action="store_true", help="Never trigger GitHub workflow runs")
    ap.add_argument("--no-ask", action="store_true", help="Never email / open issues")
    ap.add_argument("--no-git", action="store_true", help="Write the report but do not commit")
    ap.add_argument("--json", action="store_true", help="Print the full audit JSON")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    result = supervise(
        cfg,
        mode=args.mode,
        allow_dispatch=not args.no_dispatch,
        allow_ask=not args.no_ask,
        allow_git=not args.no_git,
    )
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        v = result["verdict"]
        a = result["audit"]
        print(f"verdict: {v.get('verdict')} ({v.get('source', 'grok')})")
        print(f"day {a['day_id']}: {a['timing']['due_count']} slot(s) due, "
              f"pipelines {a['cycles_with_full_pipeline']}, submissions {a['kaggle'].get('count_today')}")
        for viol in a["violations"]:
            print(f"  ! {viol}")
        for act in result["actions"]:
            print(f"  -> {act}")
        for q in v.get("questions_for_human") or []:
            print(f"  ? {q}")


if __name__ == "__main__":
    main()
