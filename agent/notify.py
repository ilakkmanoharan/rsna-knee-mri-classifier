"""Email / alert notifications for Agent-1 failures."""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
import subprocess
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def _apply_env_files() -> None:
    _load_dotenv(ROOT / ".env")
    _load_dotenv(ROOT / "private" / "agent" / "email.env")
    _load_dotenv(Path.home() / ".config" / "rsna-knee-agent1" / "email.env")


def notify_config(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    _apply_env_files()
    n = (cfg or {}).get("notify") or {}
    return {
        "to": os.environ.get("AGENT1_NOTIFY_TO") or n.get("to") or "ilakkmanoharan@gmail.com",
        "from_addr": os.environ.get("AGENT1_NOTIFY_FROM")
        or n.get("from_addr")
        or os.environ.get("AGENT1_SMTP_USER")
        or n.get("smtp_user")
        or "ilakkmanoharan@gmail.com",
        "smtp_host": os.environ.get("AGENT1_SMTP_HOST") or n.get("smtp_host") or "smtp.gmail.com",
        "smtp_port": int(os.environ.get("AGENT1_SMTP_PORT") or n.get("smtp_port") or 587),
        "smtp_user": os.environ.get("AGENT1_SMTP_USER") or n.get("smtp_user") or "",
        "smtp_password": os.environ.get("AGENT1_SMTP_PASSWORD") or n.get("smtp_password") or "",
        "enabled": bool(
            str(os.environ.get("AGENT1_NOTIFY_ENABLED", n.get("enabled", True))).lower()
            not in {"0", "false", "no"}
        ),
    }


def _write_local_alert(subject: str, body: str, state_dir: Path) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "last_alert.txt"
    path.write_text(f"{subject}\n\n{body}\n")
    log_path = state_dir / "alerts.log"
    with log_path.open("a") as f:
        f.write(f"\n===== {subject} =====\n{body}\n")
    return path


def _smtp_send(cfg: dict[str, Any], subject: str, body: str) -> bool:
    user = cfg["smtp_user"]
    password = cfg["smtp_password"]
    if not user or not password:
        logger.warning(
            "Email SMTP not configured (set AGENT1_SMTP_USER / AGENT1_SMTP_PASSWORD "
            "or private/agent/email.env). Skipping SMTP send."
        )
        return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg["from_addr"]
    msg["To"] = cfg["to"]
    msg.set_content(body)
    context = ssl.create_default_context()
    with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"], timeout=60) as server:
        server.starttls(context=context)
        server.login(user, password)
        server.send_message(msg)
    logger.info("Sent email to %s: %s", cfg["to"], subject)
    return True


def _macos_notification(subject: str, body: str) -> None:
    """Best-effort desktop banner; does not replace email."""
    try:
        safe_sub = subject.replace('"', "'")[:100]
        safe_body = body.replace('"', "'")[:200]
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification "{safe_body}" with title "RSNA Agent1" subtitle "{safe_sub}"',
            ],
            check=False,
            capture_output=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("osascript notify failed: %s", exc)


def send_alert(
    subject: str,
    body: str,
    cfg: dict[str, Any] | None = None,
    state_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Persist alert locally, try SMTP email, show macOS notification."""
    ncfg = notify_config(cfg)
    state_dir = state_dir or (ROOT / "artifacts" / "agent_state")
    alert_path = _write_local_alert(subject, body, state_dir)
    _macos_notification(subject, body)
    emailed = False
    error = None
    if ncfg["enabled"]:
        try:
            emailed = _smtp_send(ncfg, subject, body)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("Failed to send alert email")
    return {"emailed": emailed, "to": ncfg["to"], "alert_path": str(alert_path), "error": error}
