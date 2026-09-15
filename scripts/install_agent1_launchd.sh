#!/usr/bin/env bash
set -euo pipefail
ROOT="/Users/ilakkmanoharan2026/Projects/rsna-knee-mri-classifier"
PLIST_SRC="$ROOT/launchd/com.ilakkmanoharan.rsna-knee-agent1.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.ilakkmanoharan.rsna-knee-agent1.plist"
mkdir -p "$HOME/Library/LaunchAgents" "$ROOT/artifacts/agent_state"
cp "$PLIST_SRC" "$PLIST_DST"
launchctl unload "$PLIST_DST" 2>/dev/null || true
launchctl load "$PLIST_DST"
echo "Loaded $PLIST_DST"
launchctl list | grep rsna-knee-agent1 || true
echo "Agent1 will start at 01:00 local time (set Mac timezone to America/Chicago for CST/CDT alignment)."
echo "Manual run: python3 $ROOT/agent/run_daily.py --once --no-wait-for-day-start"
