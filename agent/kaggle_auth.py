"""Ensure Kaggle credentials work for both legacy keys and KGAT access tokens."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_kaggle_credentials() -> Path:
    """Write ~/.kaggle files from env. KGAT_* secrets are access tokens, not legacy keys."""
    home = Path(os.environ.get("HOME") or Path.home())
    kdir = home / ".kaggle"
    kdir.mkdir(parents=True, exist_ok=True)
    user = (os.environ.get("KAGGLE_USERNAME") or "").strip()
    key = (os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_TOKEN") or "").strip()
    if user and key:
        cred = kdir / "kaggle.json"
        cred.write_text(json.dumps({"username": user, "key": key}))
        cred.chmod(0o600)
    if key.startswith("KGAT_"):
        token_path = kdir / "access_token"
        token_path.write_text(key + "\n")
        token_path.chmod(0o600)
        os.environ["KAGGLE_API_TOKEN"] = key
        logger.info("Configured Kaggle KGAT access token at ~/.kaggle/access_token")
    return kdir
