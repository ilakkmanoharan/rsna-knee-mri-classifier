from __future__ import annotations

import json
from pathlib import Path

from agent.kaggle_auth import ensure_kaggle_credentials


def test_kgat_secret_writes_access_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("KAGGLE_USERNAME", "tester")
    monkeypatch.setenv("KAGGLE_KEY", "KGAT_dummytokenvalue")
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    kdir = ensure_kaggle_credentials()
    assert kdir == tmp_path / ".kaggle"
    cred = json.loads((kdir / "kaggle.json").read_text())
    assert cred["username"] == "tester"
    assert (kdir / "access_token").read_text().strip() == "KGAT_dummytokenvalue"


def test_legacy_key_does_not_write_access_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("KAGGLE_USERNAME", "tester")
    monkeypatch.setenv("KAGGLE_KEY", "abc123hexlegacykey")
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    kdir = ensure_kaggle_credentials()
    assert (kdir / "kaggle.json").exists()
    assert not (kdir / "access_token").exists()
