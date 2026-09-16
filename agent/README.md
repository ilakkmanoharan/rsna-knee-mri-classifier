# Agent-1 — daily competition improvement loop

Starts at **01:00 America/Chicago** (competition quota reset) and keeps cycling until all **5** daily Kaggle submissions are used. Spacing is adaptive — **30-60 minutes** between cycles, tightening toward 30 when behind pace or the day is running out, relaxing toward 60 when on track (`pacing` in [`config.yaml`](config.yaml), logic in [`pacing.py`](pacing.py)). Leaving quota unused counts as a failure.

## Pipeline each cycle

1. **Research** → `Research/` (arXiv + curated methods write-up)
2. **Analysis** → `Analysis/` (Kaggle submission logs + score diagnosis)
3. **Hypothesis** → `Hypothesis/`
4. **Plan / spec** → `Plans/`
5. **Implement + submit** offline Kaggle notebook for `rsna-knee-abnormality-detection`
6. **Git commit + push** to `https://github.com/ilakkmanoharan/rsna-knee-mri-classifier.git`

## Cloud (recommended when laptop is closed)

See [`CLOUD.md`](CLOUD.md). Use GitHub Actions — not LaunchAgent — for unattended daily runs.

## Supervision

A separate Grok bot audits steps 1–3 and the submission cadence 25 minutes after every slot,
re-runs missed cycles, and emails/opens an issue when it needs you. See [`GROK_BOT.md`](GROK_BOT.md).

## Failure behavior

- Cycle errors are caught: Agent-1 **keeps retrying** (5 min → up to 30 min backoff).
- Each failure writes `artifacts/agent_state/last_alert.txt` and emails **ilakkmanoharan@gmail.com**.
- LaunchAgent uses `KeepAlive` so a crashed process is restarted.

### Email setup (required for SMTP)

```bash
cp private/agent/email.env.example private/agent/email.env
# edit email.env — set AGENT1_SMTP_PASSWORD to a Gmail App Password
```

Without SMTP credentials, alerts are still logged locally and shown as a macOS notification.

