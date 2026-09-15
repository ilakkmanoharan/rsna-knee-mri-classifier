# Running Agent-1 with the laptop closed

Local LaunchAgent **cannot** run while the Mac is asleep. Use **GitHub Actions** (primary) and optionally **Cursor Cloud Automations**.

## 1. GitHub Actions (recommended)

Workflow: [`.github/workflows/agent1-cloud.yml`](../.github/workflows/agent1-cloud.yml)

It runs ~5 times per day on GitHub’s runners (no laptop needed), executes `python agent/run_cycle.py`, commits Research/Analysis/Hypothesis/Plans, and submits to Kaggle.

### One-time secrets

In the GitHub repo → **Settings → Secrets and variables → Actions**, add:

| Secret | Required | Purpose |
|--------|----------|---------|
| `KAGGLE_USERNAME` | yes | Kaggle API username |
| `KAGGLE_KEY` | yes | Kaggle API key |
| `AGENT1_SMTP_USER` | no | Gmail address for alerts |
| `AGENT1_SMTP_PASSWORD` | no | Gmail App Password |
| `AGENT1_NOTIFY_TO` | no | defaults to `ilakkmanoharan@gmail.com` |

Create a Gmail App Password at https://myaccount.google.com/apppasswords if you want email on failures.

### Enable Actions

1. Push this workflow to `main` (already in repo once merged).
2. Confirm **Actions** are enabled for the repository.
3. Run **agent1-cloud** once via **Actions → agent1-cloud → Run workflow** to verify.
4. Disable the local LaunchAgent so you don’t double-submit (see below).

### Manual run

```bash
gh workflow run agent1-cloud.yml
# or with inputs:
gh workflow run agent1-cloud.yml -f skip_submit=true
```

## 2. Cursor Cloud Automation (optional)

1. Open https://cursor.com/automations (or run `/automate` in Cursor).
2. Select repo `ilakkmanoharan/rsna-knee-mri-classifier`.
3. Add a **schedule** matching 01:00 America/Chicago + 90‑minute slots (max 5/day).
4. Paste instructions from [`cursor_cloud_automation_prompt.md`](cursor_cloud_automation_prompt.md).
5. Provide Kaggle credentials to the cloud environment/secrets as required by Cursor.

Use this when you want a full cloud coding agent to interpret Research/Analysis and change strategy code—not only run the fixed `run_cycle.py` path.

## 3. Turn off local scheduling

```bash
launchctl unload ~/Library/LaunchAgents/com.ilakkmanoharan.rsna-knee-agent1.plist 2>/dev/null || true
pkill -f 'agent/run_daily.py' 2>/dev/null || true
```

Keep only **one** runner (GitHub Actions **or** local), never both.

## Quota / day boundary

Agent state lives in `artifacts/agent_state/day_state.json` (committed). The runner also syncs COMPLETE/PENDING counts from Kaggle so a laptop crash mid-day won’t over-submit when cloud takes over.
