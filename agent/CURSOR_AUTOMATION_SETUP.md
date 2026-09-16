# RSNA Knee Agent-1 — Cursor Automation (paste into cursor.com/automations)

## Automation name
`rsna-knee-agent1-daily`

## Repository
- Repo: `ilakkmanoharan/rsna-knee-mri-classifier`
- Branch: `main`
- **Must select a repository** (scheduled triggers default to no-repo — override that)

## Triggers (schedule)
Use **custom cron** (UTC), one trigger per line-item in Cursor. These fire every 30 minutes from **01:00 America/Chicago (CDT = UTC−5)** so all 5 submissions get used; the agent itself enforces the daily cap and the 30-minute floor between submissions:

```text
0,30 6 * * *
0,30 7 * * *
0,30 8 * * *
0,30 9 * * *
0,30 10 * * *
0,30 11 * * *
0,30 12 * * *
0,30 13 * * *
```

(The window already covers the CST / UTC−6 shift; extra triggers are harmless no-ops once quota is spent.)

## Tools to enable
- Pull request creation: **off** (agent should push commits to `main`, not open PRs unless blocked)
- Memories: **on** (remember last score / what failed)
- Computer use: optional
- Send email / Slack: optional if configured

## Secrets the cloud environment needs
Configure in Cursor Cloud Agent / environment secrets (do **not** commit to git):
- `KAGGLE_USERNAME`
- `KAGGLE_KEY`
- Optional: `AGENT1_SMTP_USER`, `AGENT1_SMTP_PASSWORD`, `AGENT1_NOTIFY_TO`

## Instructions (prompt — paste entirely)

```text
You are Agent-1 for the Kaggle competition RSNA Knee Abnormality Detection.

Repository: https://github.com/ilakkmanoharan/rsna-knee-mri-classifier (branch main).
Work only in this repo. Pull latest main before changing anything.

Run EXACTLY ONE improvement cycle per automation invocation. Do not loop 5 times in one run.

Pipeline (in order):
1) Research: search/read relevant methods to raise macro ROC-AUC beyond our current public score. Write a dated markdown under Research/.
2) Analysis: fetch Kaggle competition submissions for rsna-knee-abnormality-detection; explain why the score is low and how to improve. Write under Analysis/.
3) Hypothesis: propose the next testable change under Hypothesis/.
4) Plan: write an implementation spec under Plans/.
5) Implement + submit: prefer `python agent/run_cycle.py` after `pip install kaggle pyyaml pandas numpy`. Ensure ~/.kaggle/kaggle.json exists from environment secrets KAGGLE_USERNAME + KAGGLE_KEY.
6) Git: commit and push Research/, Analysis/, Hypothesis/, Plans/, artifacts/agent_state/ to main using author:
   name "ilakk manoharan"
   email "28582192+ilakkmanoharan@users.noreply.github.com"
   (override per-commit; do not rewrite git config permanently).

Quota rules:
- Use all 5 Kaggle submissions every competition day (day starts 01:00 America/Chicago); unused quota is a failure.
- Keep 30-60 minutes between submissions: skip submitting if the previous submission was under 30 minutes ago.
- If quota is full, write analysis noting skip and exit successfully without submitting.
- Never use test radiology reports. Kaggle notebooks must run offline.

On errors:
- Write artifacts/agent_state/last_alert.txt with the failure.
- Retry once after a short fix if the error is obvious (missing dep, path).
- Summarize the error clearly in your final message.
- Email ilakkmanoharan@gmail.com if SMTP secrets are available; otherwise local alert file is enough.

Constraints:
- Do not invent competition paths; discover /kaggle/input/competitions/... only inside Kaggle notebooks.
- Do not commit secrets.
- Prefer metadata/prior blend strategies until real MRI training is practical.
```

## After saving
1. Activate the automation.
2. Click **Run now** once to verify.
3. Confirm GitHub Actions `agent1-cloud` is either disabled **or** you accept possible double runs — prefer **one** scheduler only.
