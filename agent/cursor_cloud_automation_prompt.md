# Cursor Cloud Automation — Agent-1

Use this as the instruction text for a scheduled automation at
https://cursor.com/automations (or `/automate` in Cursor).

## Trigger

- Schedule / cron every **30 minutes** from **01:00 America/Chicago** onward; the agent submits at most 5 times per day and enforces a 30-minute floor between submissions, so extra triggers are no-ops.
- Repository: `https://github.com/ilakkmanoharan/rsna-knee-mri-classifier.git` (branch `main`).

## Goal

Run **one** Agent-1 improvement cycle for RSNA Knee Abnormality Detection and push artifacts to GitHub. Do not burn more than one Kaggle submission per automation run.

## Steps (must follow in order)

1. Pull latest `main`.
2. Ensure folders exist: `Research/`, `Analysis/`, `Hypothesis/`, `Plans/`.
3. Run:
   ```bash
   python agent/run_cycle.py
   ```
   (Install `kaggle pyyaml pandas numpy` if needed. Kaggle creds must be available in the cloud environment / secrets.)
4. If `run_cycle.py` is unavailable, manually:
   - Research methods to beat current public score; write markdown under `Research/`.
   - Fetch Kaggle submissions; write analysis under `Analysis/`.
   - Write hypotheses under `Hypothesis/` and a plan under `Plans/`.
   - Implement the planned offline Kaggle notebook, push kernel, submit via `competition_submit_code`.
5. Commit and push all new Research/Analysis/Hypothesis/Plans/state files to `main`.
6. On failure: keep notes in `artifacts/agent_state/last_alert.txt` and do not exit silently — summarize the error in the agent result.

## Hard constraints

- Internet disabled inside the **Kaggle** notebook; no test-report shortcut.
- Use all 5 Kaggle submissions per competition day, 30-60 minutes apart; skip if quota full or the last submission was under 30 minutes ago.
- Prefer metadata/prior strategies until real MRI training assets are mounted.
- Never put SMTP/Kaggle passwords into the repo.
