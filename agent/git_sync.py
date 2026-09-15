"""Git commit/push helpers for agent artifacts."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Workspace rule: never write git config; override author/committer per commit.
GIT_AUTHOR = [
    "-c",
    "user.name=ilakk manoharan",
    "-c",
    "user.email=28582192+ilakkmanoharan@users.noreply.github.com",
]


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    logger.info("$ %s", " ".join(cmd))
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=check)


def ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
        keep = p / ".gitkeep"
        if not any(p.iterdir()) and not keep.exists():
            keep.write_text("")


def commit_and_push(
    paths: list[str],
    message: str,
    branch: str = "main",
    auto_push: bool = True,
) -> Optional[str]:
    """Stage paths, commit with required identity, optionally push. Returns commit sha or None."""
    existing = [p for p in paths if (REPO_ROOT / p).exists()]
    if not existing:
        logger.warning("No paths to commit: %s", paths)
        return None

    _run(["git", "add", "--"] + existing, check=False)
    staged = _run(["git", "diff", "--cached", "--name-only"], check=False)
    if not staged.stdout.strip():
        logger.info("Nothing staged; skip commit")
        return None

    body = message.strip() + "\n"
    proc = subprocess.run(
        ["git", *GIT_AUTHOR, "commit", "-m", body],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        logger.error("commit failed: %s %s", proc.stdout, proc.stderr)
        return None

    sha = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    logger.info("Committed %s", sha)
    if auto_push:
        push = subprocess.run(
            ["git", "push", "-u", "origin", branch],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if push.returncode != 0:
            logger.error("push failed: %s %s", push.stdout, push.stderr)
        else:
            logger.info("Pushed to origin/%s", branch)
    return sha
