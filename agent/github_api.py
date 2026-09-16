"""Minimal GitHub REST client for the supervisor bot (stdlib only).

Used to read workflow-run history, re-trigger the agent workflow, and open/comment
issues when the bot needs the operator.

Tokens, in priority order:
  AGENT1_GH_PAT / GH_PAT  -> personal access token (repo + workflow scopes).
                             Required to *trigger* workflow runs: runs dispatched with the
                             built-in GITHUB_TOKEN are intentionally ignored by GitHub.
  GH_TOKEN / GITHUB_TOKEN -> enough for reading runs and writing issues.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import urllib.error
import urllib.request
from typing import Any, Optional

logger = logging.getLogger(__name__)

API = "https://api.github.com"
PAT_VARS = ("AGENT1_GH_PAT", "GH_PAT", "GROK_GH_PAT")
TOKEN_VARS = PAT_VARS + ("GH_TOKEN", "GITHUB_TOKEN")


def token() -> str:
    for var in TOKEN_VARS:
        val = (os.environ.get(var) or "").strip()
        if val:
            return val
    return ""


def has_pat() -> bool:
    """True when a token that can trigger workflow runs is configured."""
    return any((os.environ.get(v) or "").strip() for v in PAT_VARS)


def repo_slug(cfg: dict[str, Any] | None = None) -> str:
    env = (os.environ.get("GITHUB_REPOSITORY") or "").strip()
    if env:
        return env
    url = str((cfg or {}).get("github_repo") or "")
    m = re.search(r"github\.com[:/]+([^/]+/[^/\s]+?)(?:\.git)?/?$", url)
    if m:
        return m.group(1)
    try:
        out = subprocess.run(
            ["git", "remote", "get-url", "origin"], capture_output=True, text=True, check=True
        ).stdout.strip()
        m = re.search(r"github\.com[:/]+([^/]+/[^/\s]+?)(?:\.git)?/?$", out)
        if m:
            return m.group(1)
    except Exception as exc:  # noqa: BLE001
        logger.debug("git remote lookup failed: %s", exc)
    return ""


def _request(method: str, path: str, body: Optional[dict] = None) -> Optional[Any]:
    tok = token()
    if not tok:
        logger.warning("No GitHub token available; skipping %s %s", method, path)
        return None
    req = urllib.request.Request(
        f"{API}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bearer {tok}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "rsna-knee-grok-bot",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as exc:
        logger.error("GitHub %s %s -> %s: %s", method, path, exc.code, exc.read().decode(errors="replace")[:400])
    except Exception as exc:  # noqa: BLE001
        logger.error("GitHub %s %s failed: %s", method, path, exc)
    return None


def list_workflow_runs(
    workflow_file: str, cfg: dict[str, Any] | None = None, per_page: int = 20
) -> list[dict[str, Any]]:
    slug = repo_slug(cfg)
    if not slug:
        return []
    data = _request("GET", f"/repos/{slug}/actions/workflows/{workflow_file}/runs?per_page={per_page}")
    runs = (data or {}).get("workflow_runs") or []
    return [
        {
            "id": r.get("id"),
            "status": r.get("status"),
            "conclusion": r.get("conclusion"),
            "event": r.get("event"),
            "created_at": r.get("created_at"),
            "updated_at": r.get("updated_at"),
            "url": r.get("html_url"),
        }
        for r in runs
    ]


def dispatch_workflow(
    workflow_file: str, ref: str = "main", inputs: Optional[dict[str, str]] = None, cfg: dict[str, Any] | None = None
) -> bool:
    slug = repo_slug(cfg)
    if not slug:
        logger.error("Cannot resolve repo slug for workflow dispatch")
        return False
    if not has_pat():
        logger.warning(
            "Only GITHUB_TOKEN available — GitHub ignores workflow_dispatch from it. "
            "Add an AGENT1_GH_PAT secret (repo+workflow scopes) to let the bot re-trigger runs."
        )
    body: dict[str, Any] = {"ref": ref}
    if inputs:
        body["inputs"] = inputs
    res = _request("POST", f"/repos/{slug}/actions/workflows/{workflow_file}/dispatches", body)
    ok = res is not None
    logger.info("Dispatch %s on %s: %s", workflow_file, ref, "ok" if ok else "failed")
    return ok


def find_open_issue(marker: str, cfg: dict[str, Any] | None = None) -> Optional[dict[str, Any]]:
    slug = repo_slug(cfg)
    if not slug:
        return None
    data = _request("GET", f"/repos/{slug}/issues?state=open&per_page=50")
    for issue in data or []:
        if marker in (issue.get("title") or ""):
            return issue
    return None


def create_issue(
    title: str,
    body: str,
    labels: Optional[list[str]] = None,
    assignees: Optional[list[str]] = None,
    cfg: dict[str, Any] | None = None,
) -> Optional[dict[str, Any]]:
    slug = repo_slug(cfg)
    if not slug:
        return None
    payload: dict[str, Any] = {"title": title, "body": body}
    if labels:
        payload["labels"] = labels
    if assignees:
        payload["assignees"] = assignees
    issue = _request("POST", f"/repos/{slug}/issues", payload)
    if issue:
        logger.info("Opened issue #%s: %s", issue.get("number"), title)
    return issue


def comment_issue(number: int, body: str, cfg: dict[str, Any] | None = None) -> bool:
    slug = repo_slug(cfg)
    if not slug:
        return False
    return _request("POST", f"/repos/{slug}/issues/{number}/comments", {"body": body}) is not None
