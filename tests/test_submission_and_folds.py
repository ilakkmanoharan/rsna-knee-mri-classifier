"""Submission validation and fold leakage checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.constants import DEFAULT_TARGETS, STUDY_ID_COL
from src.data.metadata import make_folds
from src.validate_submission import (
    SubmissionValidationError,
    constant_baseline,
    prevalence_baseline,
    validate_submission,
    write_submission,
)


def _sample(n=5):
    rows = []
    for i in range(n):
        row = {STUDY_ID_COL: f"study_{i}"}
        for t in DEFAULT_TARGETS:
            row[t] = 0.5
        rows.append(row)
    return pd.DataFrame(rows)


def test_validate_ok(tmp_path):
    sample = _sample()
    rng = np.random.default_rng(0)
    sub = sample.copy()
    for t in DEFAULT_TARGETS:
        sub[t] = np.clip(0.5 + rng.normal(0, 0.05, len(sub)), 1e-6, 1 - 1e-6)
    report = validate_submission(sub, sample)
    assert report["ok"]


def test_rejects_wrong_order():
    sample = _sample()
    sub = sample.iloc[::-1].reset_index(drop=True)
    with pytest.raises(SubmissionValidationError):
        validate_submission(sub, sample)


def test_rejects_out_of_range():
    sample = _sample()
    sub = sample.copy()
    sub["ACL"] = 1.5
    with pytest.raises(SubmissionValidationError):
        validate_submission(sub, sample)


def test_write_submission(tmp_path):
    sample = _sample()
    rng = np.random.default_rng(1)
    sub = sample.copy()
    for t in DEFAULT_TARGETS:
        sub[t] = np.clip(0.4 + rng.random(len(sub)) * 0.2, 1e-6, 1 - 1e-6)
    path = tmp_path / "submission.csv"
    report = write_submission(sub, sample, path)
    assert path.exists()
    assert report["ok"]


def test_prevalence_baseline():
    sample = _sample()
    sub = prevalence_baseline(sample, {"ACL": 0.2})
    assert abs(sub["ACL"].iloc[0] - 0.2) < 1e-9


def test_folds_no_study_leakage():
    ids = [f"s{i}" for i in range(20)]
    groups = pd.Series([f"g{i // 2}" for i in range(20)])  # pairs share patient
    folds = make_folds(ids, groups=groups, n_folds=5, seed=0)
    # Each study once
    assert folds[STUDY_ID_COL].nunique() == 20
    # Patients not split across folds
    merged = folds.copy()
    merged["group"] = groups.values
    for g, gdf in merged.groupby("group"):
        assert gdf["fold"].nunique() == 1
