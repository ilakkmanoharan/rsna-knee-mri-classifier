"""Submission structural validation contract (Stage B / Stage F)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.constants import STUDY_ID_COL


class SubmissionValidationError(ValueError):
    pass


def validate_submission(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
    clip_eps: float = 0.0,
    require_nontrivial_variance: bool = True,
    min_std: float = 1e-6,
) -> dict:
    """Assert schema/order/range contract. Raises SubmissionValidationError on failure."""
    errors: list[str] = []

    if list(submission.columns) != list(sample.columns):
        errors.append(
            f"Column mismatch.\n  got: {list(submission.columns)}\n  expected: {list(sample.columns)}"
        )

    if len(submission) != len(sample):
        errors.append(f"Row count {len(submission)} != sample {len(sample)}")

    if STUDY_ID_COL not in submission.columns:
        errors.append(f"Missing {STUDY_ID_COL}")
    else:
        if submission[STUDY_ID_COL].duplicated().any():
            errors.append("Duplicate StudyInstanceUID values")
        if not submission[STUDY_ID_COL].astype(str).equals(sample[STUDY_ID_COL].astype(str)):
            # Allow same multiset but wrong order — still fail per contract
            if set(submission[STUDY_ID_COL].astype(str)) != set(sample[STUDY_ID_COL].astype(str)):
                errors.append("StudyInstanceUID set differs from sample_submission")
            else:
                errors.append("Row order does not match sample_submission.csv")

    targets = [c for c in sample.columns if c != STUDY_ID_COL]
    stats: dict[str, dict] = {}
    for t in targets:
        if t not in submission.columns:
            errors.append(f"Missing target column: {t}")
            continue
        col = pd.to_numeric(submission[t], errors="coerce")
        if col.isna().any():
            errors.append(f"{t}: missing/NaN values")
        if np.isinf(col.to_numpy(dtype=float, na_value=np.nan)).any():
            errors.append(f"{t}: non-finite values")
        vals = col.to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if len(finite):
            if finite.min() < 0.0 - clip_eps or finite.max() > 1.0 + clip_eps:
                errors.append(f"{t}: values outside [0,1] (min={finite.min()}, max={finite.max()})")
            st = {
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "p10": float(np.percentile(finite, 10)),
                "p50": float(np.percentile(finite, 50)),
                "p90": float(np.percentile(finite, 90)),
            }
            stats[t] = st
            if require_nontrivial_variance and st["std"] < min_std and len(finite) > 1:
                errors.append(f"{t}: near-constant predictions (std={st['std']})")

    report = {"ok": not errors, "errors": errors, "per_target_stats": stats, "n_rows": len(submission)}
    if errors:
        raise SubmissionValidationError("; ".join(errors))
    return report


def write_submission(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
    out_path: Path,
    clip_eps: float = 1e-6,
    require_nontrivial_variance: bool = True,
) -> dict:
    """Clip to (eps,1-eps), validate, write submission.csv."""
    out = submission.copy()
    # Enforce sample ID order
    out[STUDY_ID_COL] = out[STUDY_ID_COL].astype(str)
    sample = sample.copy()
    sample[STUDY_ID_COL] = sample[STUDY_ID_COL].astype(str)
    out = sample[[STUDY_ID_COL]].merge(out, on=STUDY_ID_COL, how="left")
    targets = [c for c in sample.columns if c != STUDY_ID_COL]
    for t in targets:
        if t not in out.columns:
            out[t] = 0.5
        out[t] = pd.to_numeric(out[t], errors="coerce").fillna(0.5)
        out[t] = out[t].clip(clip_eps, 1.0 - clip_eps)
    out = out[sample.columns]
    report = validate_submission(
        out,
        sample,
        require_nontrivial_variance=require_nontrivial_variance,
    )
    out_path = Path(out_path)
    if out_path.name != "submission.csv":
        # Contract prefers exact name; still allow path directory flexibility
        pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    report["path"] = str(out_path)
    return report


def prevalence_baseline(
    sample: pd.DataFrame,
    prevalence: dict[str, float] | None = None,
    default: float = 0.5,
) -> pd.DataFrame:
    targets = [c for c in sample.columns if c != STUDY_ID_COL]
    out = sample[[STUDY_ID_COL]].copy()
    for t in targets:
        p = default if prevalence is None else float(prevalence.get(t, default))
        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        out[t] = p
    return out[sample.columns]


def constant_baseline(sample: pd.DataFrame, value: float = 0.5) -> pd.DataFrame:
    out = sample.copy()
    targets = [c for c in sample.columns if c != STUDY_ID_COL]
    for t in targets:
        out[t] = value
    return out
