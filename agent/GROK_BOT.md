# Grok supervisor bot

A second, independent bot that watches agent1 and repairs it. It does not trust the agent's own
logs — it verifies what actually landed in git and on Kaggle.

- Code: [`agent/supervisor.py`](supervisor.py), [`agent/grok.py`](grok.py), [`agent/github_api.py`](github_api.py)
- Schedule: [`.github/workflows/grok-supervisor.yml`](../.github/workflows/grok-supervisor.yml) —
  hourly, all day. `agent1-cloud.yml` only fires across the early competition day, so the
  supervisor is what recovers quota if the agent falls behind later on
- Reports: `Supervision/<day>_<stamp>_supervision.md` (+ `.json`), committed to `main`
- Bot state: `artifacts/agent_state/supervisor_state.json` (dispatch cooldowns, asked questions)

## What it supervises (agent1.md steps 1–3)

| Stage | Folder | Checks |
|---|---|---|
| Research | `Research/` | one write-up per submission due by now; ≥1200 chars; ≥3 paper/web links; papers actually retrieved (`_research.json → papers`); explains *why* each method helps; tied to the metric |
| Analysis | `Analysis/` | one write-up per submission due by now; references previous submission scores/logs; diagnoses why the score was low; proposes an improvement path |
| Hypothesis | `Hypothesis/` | one write-up per submission due by now; stated as testable if/then predictions; has a measurable acceptance criterion; links back to Research + Analysis |

Alongside that it checks the adaptive cadence from [`pacing.py`](pacing.py): all five submissions
must be spent each day, 30-60 minutes apart. "Due by now" is the slowest acceptable pace — one
submission after a 20-minute grace, then one per hour — so a day that drifts slower than that is
flagged. It also raises `quota_at_risk` when the submissions still owed cannot fit before the day
rolls over, and reports Kaggle errors and failed `agent1-cloud.yml` runs.

## What it does about problems

1. **Verdict** — the audit goes to Grok (`XAI_API_KEY`, model `grok-4-latest`) which returns
   `on_track` / `degraded` / `failing` plus per-stage findings and questions. Without an API key
   the same checks still run rule-based, so supervision never silently stops.
2. **Repair** — when a due submission has no complete Research+Analysis+Hypothesis pipeline, or the
   submission count is behind pace, and quota remains:
   - with an `AGENT1_GH_PAT` secret it dispatches `agent1-cloud.yml`;
   - otherwise the supervisor workflow runs `python agent/run_cycle.py` inline in its own job.
   Guards: no repair while an agent run is in progress, a 30-minute dispatch cooldown,
   `max_dispatches_per_day: 8`, and `run_cycle.py`'s own quota sync plus 30-minute submission
   floor — so it cannot double-submit or bunch submissions.
3. **Ask you** — questions (missing credentials, repeated identical failures, strategy calls it
   shouldn't make alone) are emailed to you and posted as a GitHub issue titled
   `[grok-bot] needs input — day <date>`, deduplicated per day. Answer by commenting on the issue.

## Secrets (repo → Settings → Secrets and variables → Actions)

| Secret | Needed for |
|---|---|
| `XAI_API_KEY` | Grok's review (falls back to rule checks if absent) |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | reading submission history, inline repair submits |
| `AGENT1_GH_PAT` | optional; `repo` + `workflow` scopes so the bot can re-trigger `agent1-cloud.yml`. GitHub ignores `workflow_dispatch` made with the built-in `GITHUB_TOKEN`, which is why inline repair exists as the fallback |
| `AGENT1_SMTP_USER`, `AGENT1_SMTP_PASSWORD` | the "needs your input" email (Gmail app password) |
| `AGENT1_NOTIFY_TO` | optional; defaults to `ilakkmanoharan@gmail.com` |

## Run it by hand

```bash
python agent/supervisor.py --mode audit        # read-only: no git, no email, no dispatch
python agent/supervisor.py                     # full pass: report + repair + ask
python agent/supervisor.py --no-dispatch --no-ask --no-git --json
```

GitHub → Actions → **grok-supervisor** → *Run workflow* does the same in the cloud
(`mode: audit` for a dry run).

## Tuning the cadence

Everything lives under `pacing` in [`config.yaml`](config.yaml):

```yaml
pacing:
  min_interval_minutes: 30        # never submit faster than this
  max_interval_minutes: 60        # never drift slower than this while quota remains
  target_finish_hours: 4          # aim to have all 5 in this early in the day
  grace_minutes: 20               # slack before a submission counts as late
  hard_stop_buffer_minutes: 60    # stop submitting this long before the day rolls over
```
