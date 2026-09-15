"""Stage 5b: push Kaggle notebook and submit to the competition."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _kaggle_username() -> str:
    cred = json.loads(Path.home().joinpath(".kaggle/kaggle.json").read_text())
    return cred["username"]


def push_and_submit(
    kernel_dir: Path,
    competition: str,
    message: str,
    poll_seconds: int = 600,
) -> dict[str, Any]:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    meta = json.loads((kernel_dir / "kernel-metadata.json").read_text())
    kernel = meta["id"]  # owner/slug

    logger.info("Pushing kernel %s from %s", kernel, kernel_dir)
    push_result = api.kernels_push(str(kernel_dir))
    # Extract version if present
    version = None
    try:
        version = int(getattr(push_result, "versionNumber", None) or getattr(push_result, "version_number", None) or 0) or None
    except Exception:  # noqa: BLE001
        version = None
    logger.info("Push result: %s version=%s", push_result, version)

    # Poll until complete/error
    deadline = time.time() + poll_seconds
    status = None
    while time.time() < deadline:
        try:
            st = api.kernels_status(kernel)
            status = str(getattr(st, "status", st))
        except Exception:  # noqa: BLE001
            # CLI fallback
            import subprocess

            proc = subprocess.run(
                ["kaggle", "kernels", "status", "-k", kernel],
                capture_output=True,
                text=True,
            )
            status = proc.stdout.strip() or proc.stderr.strip()
        logger.info("Kernel status: %s", status)
        low = (status or "").lower()
        if "complete" in low:
            break
        if "error" in low or "cancel" in low:
            raise RuntimeError(f"Kernel failed: {status}")
        time.sleep(8)
    else:
        raise TimeoutError(f"Kernel did not complete within {poll_seconds}s: {status}")

    # Resolve latest version via status object if needed
    if version is None:
        try:
            st = api.kernels_status(kernel)
            version = int(getattr(st, "version_number", None) or getattr(st, "current_version_number", None) or 1)
        except Exception:  # noqa: BLE001
            version = 1

    logger.info("Submitting code competition kernel=%s version=%s", kernel, version)
    resp = api.competition_submit_code(
        file_name="submission.csv",
        message=message,
        competition=competition,
        kernel=kernel,
        kernel_version=int(version),
    )
    ref = getattr(resp, "ref", None)
    logger.info("Submit response ref=%s", ref)

    # Poll score
    public_score = None
    sub_status = None
    deadline = time.time() + poll_seconds
    while time.time() < deadline:
        rows = []
        try:
            subs = api.competition_submissions(competition)
            for s in subs:
                rows.append(s)
        except Exception as exc:  # noqa: BLE001
            logger.warning("list submissions: %s", exc)
            time.sleep(10)
            continue
        if rows:
            top = rows[0]
            sub_status = str(getattr(top, "status", ""))
            public_score = getattr(top, "publicScore", None) or getattr(top, "public_score", None)
            logger.info("Submission status=%s score=%s", sub_status, public_score)
            if "COMPLETE" in sub_status.upper() or "ERROR" in sub_status.upper() or "INVALID" in sub_status.upper():
                break
        time.sleep(15)

    return {
        "kernel": kernel,
        "kernel_version": version,
        "submission_ref": ref,
        "status": sub_status,
        "public_score": float(public_score) if public_score not in (None, "") else None,
        "message": message,
    }
