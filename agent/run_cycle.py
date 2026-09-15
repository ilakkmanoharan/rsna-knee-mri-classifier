"""Run a single agent1 cycle: research → analyze → hypothesize → plan → implement → submit."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.clock import competition_day_id, now_cst, stamp
from agent.git_sync import commit_and_push, ensure_dirs
from agent.state import StateStore
from agent.stages.analyze import run_analysis
from agent.stages.hypothesize import run_hypothesize
from agent.stages.implement import implement_notebook
from agent.stages.plan import run_plan
from agent.stages.research import run_research
from agent.stages.submit import _kaggle_username, push_and_submit

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("agent1.cycle")


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def read_strategy(plan_json: Path) -> str:
    if plan_json.exists():
        return json.loads(plan_json.read_text()).get("strategy", "metadata_prior_blend")
    return "metadata_prior_blend"


def run_cycle(cfg: dict, *, skip_submit: bool = False, force: bool = False) -> dict:
    tz = cfg.get("day_start_tz", "America/Chicago")
    hour = int(cfg.get("day_start_hour_cst", 1))
    day_id = competition_day_id(hour=hour, tz_name=tz)
    state_path = ROOT / cfg["paths"]["state"] / "day_state.json"
    store = StateStore(state_path)
    state = store.load(day_id)

    max_sub = int(cfg.get("max_submissions_per_day", 5))
    if state.submissions_used >= max_sub and not force:
        logger.warning("Daily quota exhausted (%d/%d). Skipping.", state.submissions_used, max_sub)
        return {"skipped": True, "reason": "quota", "state": state.to_dict()}

    cycle_num = state.cycles_completed
    cycle_id = f"{cycle_num:02d}_{stamp(tz)}"

    research_dir = ROOT / cfg["paths"]["research"]
    analysis_dir = ROOT / cfg["paths"]["analysis"]
    hypo_dir = ROOT / cfg["paths"]["hypothesis"]
    plans_dir = ROOT / cfg["paths"]["plans"]
    ensure_dirs([research_dir, analysis_dir, hypo_dir, plans_dir, ROOT / cfg["paths"]["state"]])

    logger.info("=== Agent1 cycle %s (day %s, used %d/%d) ===", cycle_id, day_id, state.submissions_used, max_sub)

    research_md = run_research(
        research_dir,
        queries=list(cfg.get("research", {}).get("queries") or []),
        max_arxiv=int(cfg.get("research", {}).get("max_arxiv_results", 12)),
        cycle_id=cycle_id,
        day_id=day_id,
    )
    analysis_md = run_analysis(
        analysis_dir,
        competition=cfg["competition"],
        cycle_id=cycle_id,
        day_id=day_id,
    )
    hypo_md = run_hypothesize(
        hypo_dir,
        research_md=research_md,
        analysis_md=analysis_md,
        cycle_id=cycle_id,
        day_id=day_id,
        cycle_num=cycle_num,
    )
    plan_md = run_plan(
        plans_dir,
        research_md=research_md,
        analysis_md=analysis_md,
        hypothesis_md=hypo_md,
        cycle_id=cycle_id,
        day_id=day_id,
        cycle_num=cycle_num,
    )
    strategy = read_strategy(plans_dir / f"{day_id}_cycle{cycle_id}_plan.json")

    username = _kaggle_username()
    slug = f"{cfg['kaggle']['kernel_slug_prefix']}-{day_id.replace('-', '')}-c{cycle_num:02d}"
    # Kaggle slug constraints: lowercase etc.
    slug = slug.lower().replace("_", "-")
    kernel_dir = ROOT / cfg["paths"]["kaggle_kernel_dir"] / slug
    implement_notebook(kernel_dir, strategy=strategy, cycle_id=cycle_id, kernel_slug=slug, username=username)

    # Commit research artifacts before submit so git has the paper trail even if submit fails
    if cfg.get("git", {}).get("auto_commit", True):
        commit_and_push(
            [
                cfg["paths"]["research"],
                cfg["paths"]["analysis"],
                cfg["paths"]["hypothesis"],
                cfg["paths"]["plans"],
                "agent",
                cfg["paths"]["kaggle_kernel_dir"],
            ],
            message=f"{cfg.get('git', {}).get('commit_prefix', 'agent1')}: cycle {cycle_id} research/analysis/plan ({strategy})",
            branch=cfg.get("branch", "main"),
            auto_push=bool(cfg.get("git", {}).get("auto_push", True)),
        )

    submit_info = {"skipped": True}
    if not skip_submit:
        msg = f"agent1 {day_id} cycle {cycle_num} strategy={strategy} | {hypo_md.name}"
        submit_info = push_and_submit(
            kernel_dir,
            competition=cfg["competition"],
            message=msg[:900],
        )
        state.submissions_used += 1
        state.last_submission_ref = str(submit_info.get("submission_ref"))
        state.last_public_score = submit_info.get("public_score")

    state.cycles_completed += 1
    state.last_cycle_at = now_cst(tz).isoformat()
    state.history.append(
        {
            "cycle_id": cycle_id,
            "strategy": strategy,
            "research": str(research_md),
            "analysis": str(analysis_md),
            "hypothesis": str(hypo_md),
            "plan": str(plan_md),
            "submit": submit_info,
        }
    )
    store.save(state)

    if cfg.get("git", {}).get("auto_commit", True):
        commit_and_push(
            [cfg["paths"]["state"], cfg["paths"]["analysis"]],
            message=f"{cfg.get('git', {}).get('commit_prefix', 'agent1')}: cycle {cycle_id} submit "
            f"score={submit_info.get('public_score')}",
            branch=cfg.get("branch", "main"),
            auto_push=bool(cfg.get("git", {}).get("auto_push", True)),
        )

    return {"skipped": False, "cycle_id": cycle_id, "strategy": strategy, "submit": submit_info, "state": state.to_dict()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one Agent-1 cycle")
    ap.add_argument("--config", default=str(ROOT / "agent" / "config.yaml"))
    ap.add_argument("--skip-submit", action="store_true")
    ap.add_argument("--force", action="store_true", help="Ignore daily quota guard")
    args = ap.parse_args()
    cfg = load_cfg(Path(args.config))
    result = run_cycle(cfg, skip_submit=args.skip_submit, force=args.force)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
