"""Stage 5a: build Kaggle submission notebook implementing the plan strategy."""

from __future__ import annotations

import json
import uuid
from pathlib import Path


def _notebook(source: str) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "id": uuid.uuid4().hex[:12],
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in source.splitlines()],
            }
        ],
    }


def _source_for_strategy(strategy: str, cycle_id: str) -> str:
    # Shared preamble discovers competition root without walking DICOMs.
    return f'''
from pathlib import Path
import numpy as np
import pandas as pd

STRATEGY = {strategy!r}
CYCLE_ID = {cycle_id!r}
EPS = 1e-6
SEED = 42

inp = Path("/kaggle/input")
print("top-level", [p.name for p in inp.iterdir()] if inp.exists() else None)

def discover_root() -> Path:
    comps = inp / "competitions"
    roots = []
    if comps.exists():
        roots.extend([p for p in comps.iterdir() if p.is_dir()])
    if inp.exists():
        roots.extend([p for p in inp.iterdir() if p.is_dir()])
    for r in roots:
        if (r / "sample_submission.csv").exists():
            return r
    raise FileNotFoundError("sample_submission.csv not found under /kaggle/input")

ROOT = discover_root()
print("ROOT", ROOT, "STRATEGY", STRATEGY, "CYCLE", CYCLE_ID)
sample = pd.read_csv(ROOT / "sample_submission.csv")
train = pd.read_csv(ROOT / "train.csv")
series_path = ROOT / "test_series.csv"
test_series = pd.read_csv(series_path) if series_path.exists() else None
study_col = "StudyInstanceUID"
targets = [c for c in sample.columns if c != study_col]

# Gold prevalence
prev = {{}}
for t in targets:
    if t in train.columns and train[t].notna().any():
        prev[t] = float(np.clip(train[t].mean(), EPS, 1 - EPS))
    else:
        prev[t] = 0.5

# Preferred plane soft weights (ontology-inspired, not hard rules)
PREF = {{
    "ACL": {{"Sagittal": 0.35, "Coronal": 0.05, "Axial": 0.05}},
    "MCL": {{"Coronal": 0.35, "Sagittal": 0.05, "Axial": 0.0}},
    "Medial Meniscus": {{"Sagittal": 0.2, "Coronal": 0.2, "Axial": 0.0}},
    "Lateral Meniscus": {{"Sagittal": 0.2, "Coronal": 0.2, "Axial": 0.0}},
    "Medial OA": {{"Coronal": 0.25, "Sagittal": 0.05, "Axial": 0.05}},
    "Lateral OA": {{"Coronal": 0.25, "Sagittal": 0.05, "Axial": 0.05}},
    "PF OA": {{"Axial": 0.3, "Sagittal": 0.15, "Coronal": 0.0}},
    "Effusion": {{"Axial": 0.2, "Sagittal": 0.15, "Coronal": 0.05}},
    "Synovitis": {{"Axial": 0.2, "Sagittal": 0.1, "Coronal": 0.05}},
    "Baker's": {{"Axial": 0.25, "Sagittal": 0.15, "Coronal": 0.0}},
    "Contusion": {{"Sagittal": 0.1, "Coronal": 0.1, "Axial": 0.1}},
    "Fracture": {{"Sagittal": 0.1, "Coronal": 0.1, "Axial": 0.1}},
}}
FLUID_BOOST = {{"Effusion": 0.25, "Synovitis": 0.2, "Contusion": 0.15, "Baker's": 0.1}}

def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

def study_features(uid: str) -> dict:
    feats = {{"Sagittal": 0.0, "Coronal": 0.0, "Axial": 0.0, "fluid": 0.0, "fat": 0.0, "n": 0.0}}
    if test_series is None:
        return feats
    sub = test_series[test_series[study_col].astype(str) == str(uid)]
    feats["n"] = float(len(sub))
    for _, row in sub.iterrows():
        plane = str(row.get("Anatomical_Plane", "") or "")
        if plane in feats:
            feats[plane] = 1.0
        if int(row.get("Fluid_Sensitive", 0) or 0) == 1:
            feats["fluid"] += 1.0
        if int(row.get("Fat_Suppression", 0) or 0) == 1:
            feats["fat"] += 1.0
    if feats["n"] > 0:
        feats["fluid"] /= feats["n"]
        feats["fat"] /= feats["n"]
    return feats

def offset_for(t: str, feats: dict, strategy: str) -> float:
    pref = PREF.get(t, {{}})
    off = 0.0
    for plane, w in pref.items():
        # missing preferred plane → mild negative (NOT forced zero label)
        off += w * (feats.get(plane, 0.0) - 0.5)
    if strategy in {{"metadata_prior_blend", "report_shrinkage_priors", "fluid_gate_metadata", "rank_ensemble_safe"}}:
        fb = FLUID_BOOST.get(t, 0.0)
        if strategy == "fluid_gate_metadata":
            fb *= 1.5
        off += fb * (feats.get("fluid", 0.0) - 0.5)
    return float(off)

# Optional report-shrinkage constants (train-only derived; embedded, no test reports)
# Slightly nudge meniscus/synovitis priors toward empirical soft positives rate.
SHRINK = {{
    "Medial Meniscus": 0.03,
    "Lateral Meniscus": 0.02,
    "Synovitis": 0.02,
    "Effusion": 0.02,
}}

rng = np.random.default_rng(SEED)
rows = []
meta_mat = []
prev_mat = []
for uid in sample[study_col].astype(str).tolist():
    feats = study_features(uid)
    probs = {{}}
    for t in targets:
        p0 = prev[t]
        if STRATEGY == "report_shrinkage_priors":
            p0 = float(np.clip(p0 + SHRINK.get(t, 0.0), EPS, 1 - EPS))
        off = offset_for(t, feats, STRATEGY)
        if STRATEGY == "metadata_prior_blend" or STRATEGY == "report_shrinkage_priors" or STRATEGY == "fluid_gate_metadata":
            p = float(sigmoid(logit(p0) + off))
        else:
            p = p0
        # tiny deterministic noise for variance / rank jitter
        p = float(np.clip(p + rng.normal(0, 0.005), EPS, 1 - EPS))
        probs[t] = p
    prev_mat.append([prev[t] for t in targets])
    meta_mat.append([probs[t] for t in targets])
    rows.append({{study_col: uid, **probs}})

out = pd.DataFrame(rows)[sample.columns]

if STRATEGY == "rank_ensemble_safe":
    prev_arr = np.array(prev_mat)
    meta_arr = np.array(meta_mat)
    # average ranks
    def rank_cols(a):
        r = np.empty_like(a)
        for j in range(a.shape[1]):
            order = np.argsort(a[:, j])
            ranks = np.empty(len(a))
            ranks[order] = np.linspace(EPS, 1 - EPS, len(a))
            r[:, j] = ranks
        return r
    blended = 0.5 * rank_cols(prev_arr) + 0.5 * rank_cols(meta_arr)
    out = sample[[study_col]].copy()
    for j, t in enumerate(targets):
        out[t] = blended[:, j]
    out = out[sample.columns]

assert list(out[study_col].astype(str)) == list(sample[study_col].astype(str))
assert not out[targets].isna().any().any()
path = Path("/kaggle/working/submission.csv")
out.to_csv(path, index=False)
print("Wrote", path, out.shape)
print(out.describe().T[["mean", "std", "min", "max"]].head(12))
'''.lstrip()


def implement_notebook(kernel_dir: Path, strategy: str, cycle_id: str, kernel_slug: str, username: str) -> Path:
    kernel_dir.mkdir(parents=True, exist_ok=True)
    nb_name = "notebook.ipynb"
    src = _source_for_strategy(strategy, cycle_id)
    (kernel_dir / nb_name).write_text(json.dumps(_notebook(src), indent=1))
    meta = {
        "id": f"{username}/{kernel_slug}",
        "title": kernel_slug,
        "code_file": nb_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": [],
        "model_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return kernel_dir / nb_name
