# Cursor Cloud Automation — Agent-1

Use this as the instruction text for a scheduled automation at
https://cursor.com/automations (or `/automate` in Cursor).

## Trigger

- Schedule / cron covering competition day starts at **01:00 America/Chicago**, then every **90 minutes**, up to **5** runs/day.
- Repository: `https://github.com/ilakkmanoharan/rsna-knee-mri-classifier.git` (branch `main`).

## Goal

Run **one** Agent-1 improvement cycle for RSNA Knee Abnormality Detection and push artifacts to GitHub. Do not burn more than one Kaggle submission per automation run.

## Steps (must follow in order)

1. Pull latest `main`.
2. Ensure folders exist: `Research/`, `Analysis/`, `Hypothesis analysis/`, `Plans/`.
3. Run:
   ```bash
   python agent/run_cycle.py
   ```
   (Install `kaggle pyyaml pandas numpy` if needed. Kaggle creds must be available in the cloud environment / secrets.)
4. If `run_cycle.py` is unavailable, manually:
   - Research methods to beat current public score; write markdown under `Research/`.
   - Fetch Kaggle submissions; write analysis under `Analysis/`.
   - Write hypotheses under `Hypothesis analysis/` and a plan under `Plans/`.
   - Implement the planned offline Kaggle notebook, push kernel, submit via `competition_submit_code`.
5. Commit and push all new Research/Analysis/Hypothesis/Plans/state files to `main`.
6. On failure: keep notes in `artifacts/agent_state/last_alert.txt` and do not exit silently — summarize the error in the agent result.

## Hard constraints

- Internet disabled inside the **Kaggle** notebook; no test-report shortcut.
- Max 5 Kaggle submissions per competition day; skip if quota full.
- Prefer metadata/prior strategies until real MRI training assets are mounted.
- Never put SMTP/Kaggle passwords into the repo.
