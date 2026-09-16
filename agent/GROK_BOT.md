# Grok supervisor bot

A second, independent bot that watches agent1 and repairs it. It does not trust the agent's own
logs — it verifies what actually landed in git and on Kaggle.

- Code: [`agent/supervisor.py`](supervisor.py), [`agent/grok.py`](grok.py), [`agent/github_api.py`](github_api.py)
- Schedule: [`.github/workflows/grok-supervisor.yml`](../.github/workflows/grok-supervisor.yml) —
  25 minutes after each of the five competition slots, plus an end-of-day sweep
- Reports: `Supervision/<day>_<stamp>_supervision.md` (+ `.json`), committed to `main`
- Bot state: `artifacts/agent_state/supervisor_state.json` (dispatch cooldowns, asked questions)

## What it supervises (agent1.md steps 1–3)

| Stage | Folder | Checks |
|---|---|---|
| Research | `Research/` | one write-up per due slot; ≥1200 chars; ≥3 paper/web links; papers actually retrieved (`_research.json → papers`); explains *why* each method helps; tied to the metric |
| Analysis | `Analysis/` | one write-up per due slot; references previous submission scores/logs; diagnoses why the score was low; proposes an improvement path |
| Hypothesis | `Hypothesis/` | one write-up per due slot; stated as testable if/then predictions; has a measurable acceptance criterion; links back to Research + Analysis |

Alongside that it checks cadence: five 90-minute slots from 01:00 America/Chicago, how many
Kaggle submissions actually landed today, whether any submission errored, and whether
`agent1-cloud.yml` runs failed. Slots get a 25-minute grace period before counting as missed.

## What it does about problems

1. **Verdict** — the audit goes to Grok (`XAI_API_KEY`, model `grok-4-latest`) which returns
   `on_track` / `degraded` / `failing` plus per-stage findings and questions. Without an API key
   the same checks still run rule-based, so supervision never silently stops.
2. **Repair** — when a due slot has no complete Research+Analysis+Hypothesis pipeline, or the
   submission count is behind, and quota remains:
   - with an `AGENT1_GH_PAT` secret it dispatches `agent1-cloud.yml`;
   - otherwise the supervisor workflow runs `python agent/run_cycle.py` inline in its own job.
   Guards: no repair while an agent run is in progress, `dispatch_cooldown_minutes: 40`,
   `max_dispatches_per_day: 6`, and `run_cycle.py`'s own Kaggle quota sync — so it cannot
   double-submit.
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
