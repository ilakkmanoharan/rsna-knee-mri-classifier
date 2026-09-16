"""Grok (xAI) client used by the supervisor bot.

Stdlib-only so it runs in a bare GitHub Actions job. Without an API key the
caller falls back to the deterministic rule checks in agent/supervisor.py.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-4-latest"

SYSTEM_PROMPT = """You are Grok, the supervisor of an autonomous Kaggle agent ("agent1") that must,
every 90 minutes (5 times per competition day, day starts 01:00 America/Chicago):

1. Research literature/internet and write up which methods to consider, why, and how they improve
   the score -> committed to the Research/ folder.
2. Pull logs from previous Kaggle submissions, analyze why the score was low and how to improve it
   -> committed to the Analysis/ folder.
3. Combine Research + Analysis into falsifiable hypotheses for improving the score
   -> committed to the Hypothesis/ folder.
4. Turn those into a plan/spec (Plans/), implement it, and submit to Kaggle.

You receive a machine audit of today's artifacts plus excerpts. Judge whether stages 1-3 were really
done with substance (not empty/duplicated boilerplate), whether the submission cadence is on track,
and what must be re-run. Be strict but concrete: cite file names and observed numbers.

Reply with ONLY a JSON object:
{
  "verdict": "on_track" | "degraded" | "failing",
  "stage_findings": {"research": "...", "analysis": "...", "hypothesis": "..."},
  "violations": ["..."],
  "recommended_actions": ["..."],
  "rerun_cycle": true | false,
  "questions_for_human": ["..."],
  "confidence": 0.0
}
Put a question in questions_for_human ONLY when the operator must act or decide (missing
credentials, repeated identical failures, a strategy choice the agent cannot make alone).
"""


def grok_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    g = ((cfg or {}).get("supervisor") or {}).get("grok") or {}
    key = (
        os.environ.get("XAI_API_KEY")
        or os.environ.get("GROK_API_KEY")
        or os.environ.get("AGENT1_XAI_API_KEY")
        or ""
    ).strip()
    return {
        "api_key": key,
        "base_url": (os.environ.get("XAI_BASE_URL") or g.get("base_url") or DEFAULT_BASE_URL).rstrip("/"),
        "model": os.environ.get("GROK_MODEL") or g.get("model") or DEFAULT_MODEL,
        "max_tokens": int(g.get("max_output_tokens", 1500)),
        "timeout": int(g.get("timeout_seconds", 120)),
        "enabled": bool(g.get("enabled", True)) and bool(key),
    }


def available(cfg: dict[str, Any] | None = None) -> bool:
    return grok_config(cfg)["enabled"]


def chat(prompt: str, cfg: dict[str, Any] | None = None, system: str = SYSTEM_PROMPT) -> Optional[str]:
    """Single-turn Grok completion. Returns None when unavailable or on error."""
    gc = grok_config(cfg)
    if not gc["enabled"]:
        logger.info("Grok disabled or XAI_API_KEY missing — using rule-based supervision only")
        return None

    payload = json.dumps(
        {
            "model": gc["model"],
            "temperature": 0.1,
            "max_tokens": gc["max_tokens"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
    ).encode()
    req = urllib.request.Request(
        f"{gc['base_url']}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {gc['api_key']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=gc["timeout"]) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as exc:  # noqa: PERF203
        body = exc.read().decode(errors="replace")[:500]
        logger.error("Grok HTTP %s: %s", exc.code, body)
    except Exception as exc:  # noqa: BLE001
        logger.error("Grok call failed: %s", exc)
    return None


def review(audit: dict[str, Any], cfg: dict[str, Any] | None = None) -> Optional[dict[str, Any]]:
    """Ask Grok to review the audit; returns parsed JSON verdict or None."""
    prompt = "Machine audit of the current competition day:\n\n" + json.dumps(audit, indent=2, default=str)
    raw = chat(prompt, cfg=cfg)
    if not raw:
        return None
    parsed = parse_json(raw)
    if parsed is None:
        logger.warning("Grok returned non-JSON output; keeping raw text")
        return {"verdict": "degraded", "raw": raw[:4000], "questions_for_human": [], "violations": []}
    parsed["raw"] = raw[:4000]
    return parsed


def parse_json(text: str) -> Optional[dict[str, Any]]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        first, last = text.find("{"), text.rfind("}")
        if first == -1 or last <= first:
            return None
        text = text[first : last + 1]
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        return None
