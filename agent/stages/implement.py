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
train_series_path = ROOT / "train_series.csv"
train_series = pd.read_csv(train_series_path) if train_series_path.exists() else None
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
FAT_BOOST = {{"Medial OA": 0.12, "Lateral OA": 0.12, "PF OA": 0.1, "Fracture": 0.12, "Contusion": 0.08}}

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
    if strategy in {{"metadata_prior_blend", "report_shrinkage_priors", "fluid_gate_metadata", "rank_ensemble_safe", "gold_meta_logit", "gold_rank_interact", "gold_rank_w50", "gold_rank_w70", "gold_rank_w40", "gold_rank_lam2"}}:
        fb = FLUID_BOOST.get(t, 0.0)
        if strategy == "fluid_gate_metadata":
            fb *= 1.5
        off += fb * (feats.get("fluid", 0.0) - 0.5)
        off += FAT_BOOST.get(t, 0.0) * (feats.get("fat", 0.0) - 0.5)
    return float(off)

def feat_map(series_df, interact=False):
    """Per-study vector: intercept, log1p(n), sag/cor/ax fractions, fluid frac, fat frac.
    If interact, also sag/cor/ax × fluid and sag/cor/ax × fat fractions (13-d).
    """
    dim = 13 if interact else 7
    empty = np.zeros(dim, dtype=float)
    empty[0] = 1.0
    if series_df is None or series_df.empty:
        return {{}}, empty
    df = series_df.copy()
    uid = df[study_col].astype(str)
    plane = df["Anatomical_Plane"].astype(str) if "Anatomical_Plane" in df.columns else ""
    df["_sag"] = (plane == "Sagittal").astype(float)
    df["_cor"] = (plane == "Coronal").astype(float)
    df["_ax"] = (plane == "Axial").astype(float)
    df["_fluid"] = pd.to_numeric(df["Fluid_Sensitive"], errors="coerce").fillna(0.0) if "Fluid_Sensitive" in df.columns else 0.0
    df["_fat"] = pd.to_numeric(df["Fat_Suppression"], errors="coerce").fillna(0.0) if "Fat_Suppression" in df.columns else 0.0
    df["_sag_fl"] = df["_sag"] * df["_fluid"]
    df["_cor_fl"] = df["_cor"] * df["_fluid"]
    df["_ax_fl"] = df["_ax"] * df["_fluid"]
    df["_sag_ft"] = df["_sag"] * df["_fat"]
    df["_cor_ft"] = df["_cor"] * df["_fat"]
    df["_ax_ft"] = df["_ax"] * df["_fat"]
    agg = df.groupby(uid).agg(
        n=(study_col, "size"),
        sag=("_sag", "sum"),
        cor=("_cor", "sum"),
        ax=("_ax", "sum"),
        fluid=("_fluid", "sum"),
        fat=("_fat", "sum"),
        sag_fl=("_sag_fl", "sum"),
        cor_fl=("_cor_fl", "sum"),
        ax_fl=("_ax_fl", "sum"),
        sag_ft=("_sag_ft", "sum"),
        cor_ft=("_cor_ft", "sum"),
        ax_ft=("_ax_ft", "sum"),
    )
    out = {{}}
    for sid, row in agg.iterrows():
        n = max(float(row["n"]), 1.0)
        vec = [
            1.0,
            float(np.log1p(row["n"])),
            float(row["sag"] / n),
            float(row["cor"] / n),
            float(row["ax"] / n),
            float(row["fluid"] / n),
            float(row["fat"] / n),
        ]
        if interact:
            vec += [
                float(row["sag_fl"] / n),
                float(row["cor_fl"] / n),
                float(row["ax_fl"] / n),
                float(row["sag_ft"] / n),
                float(row["cor_ft"] / n),
                float(row["ax_ft"] / n),
            ]
        out[str(sid)] = np.array(vec, dtype=float)
    return out, empty

def fit_ridge_logit(X, y, lam=1.0, steps=40):
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    # intercept is not regularized as strongly
    reg = lam * np.ones(d)
    reg[0] = 0.05 * lam
    for _ in range(steps):
        p = sigmoid(X @ w)
        g = X.T @ (p - y) / max(n, 1) + reg * w
        s = np.clip(p * (1.0 - p), 1e-4, 0.25)
        H = (X.T * s) @ X / max(n, 1) + np.diag(reg)
        try:
            w = w - np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            w = w - 0.2 * g
    return w

def rank_cols(a):
    r = np.empty_like(a, dtype=float)
    for j in range(a.shape[1]):
        order = np.argsort(a[:, j], kind="mergesort")
        ranks = np.empty(len(a), dtype=float)
        ranks[order] = np.linspace(EPS, 1 - EPS, len(a))
        r[:, j] = ranks
    return r

# Optional report-shrinkage constants (train-only derived; embedded, no test reports)
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
uids = sample[study_col].astype(str).tolist()
for uid in uids:
    feats = study_features(uid)
    probs = {{}}
    for t in targets:
        p0 = prev[t]
        if STRATEGY == "report_shrinkage_priors":
            p0 = float(np.clip(p0 + SHRINK.get(t, 0.0), EPS, 1 - EPS))
        off = offset_for(t, feats, STRATEGY)
        if STRATEGY in {{"metadata_prior_blend", "report_shrinkage_priors", "fluid_gate_metadata", "gold_meta_logit", "gold_rank_interact", "gold_rank_w50", "gold_rank_w70", "gold_rank_w40", "gold_rank_lam2"}}:
            p = float(sigmoid(logit(p0) + off))
        else:
            p = p0
        # Hand-tuned strategies keep tiny jitter; learned ranking must not be scrambled.
        if STRATEGY not in {{"gold_meta_logit", "gold_rank_interact", "gold_rank_w50", "gold_rank_w70", "gold_rank_w40", "gold_rank_lam2"}}:
            p = float(np.clip(p + rng.normal(0, 0.005), EPS, 1 - EPS))
        else:
            p = float(np.clip(p, EPS, 1 - EPS))
        probs[t] = p
    prev_mat.append([prev[t] for t in targets])
    meta_mat.append([probs[t] for t in targets])
    rows.append({{study_col: uid, **probs}})

out = pd.DataFrame(rows)[sample.columns]

def learned_scores(interact=False, lam=2.0):
    """Ridge logits on gold studies. Returns (scores, n_gold) or (None, n)."""
    if train_series is None:
        return None, 0
    tr_map, default_x = feat_map(train_series, interact=interact)
    te_map, _ = feat_map(test_series, interact=interact)
    gold = train.copy()
    gold[study_col] = gold[study_col].astype(str)
    labeled = gold.dropna(subset=[t for t in targets if t in gold.columns], how="all")
    X_rows, y_cols = [], {{t: [] for t in targets}}
    for i, sid in enumerate(labeled[study_col].astype(str).tolist()):
        x = tr_map.get(sid)
        if x is None:
            continue
        X_rows.append(x)
        row = labeled.iloc[i]
        for t in targets:
            y_cols[t].append(row[t] if t in labeled.columns else np.nan)
    n_gold = len(X_rows)
    if n_gold < 20:
        return None, n_gold
    X = np.vstack(X_rows)
    mu = X.mean(axis=0)
    sd = np.clip(X.std(axis=0), 1e-6, None)
    mu[0], sd[0] = 0.0, 1.0
    Xs = (X - mu) / sd
    Xte = np.vstack([(te_map.get(uid, default_x) - mu) / sd for uid in uids])
    scores = np.zeros((len(uids), len(targets)), dtype=float)
    for j, t in enumerate(targets):
        y = np.array(y_cols[t], dtype=float)
        mask = np.isfinite(y)
        if mask.sum() < 12 or y[mask].min() == y[mask].max():
            scores[:, j] = logit(prev[t])
            continue
        w = fit_ridge_logit(Xs[mask], y[mask], lam=lam, steps=40)
        scores[:, j] = Xte @ w
    return scores, n_gold

used_learned = False
n7 = nI = 0
BLEND_W7 = {{"gold_rank_interact": 0.60, "gold_rank_w50": 0.50, "gold_rank_w70": 0.70, "gold_rank_w40": 0.40, "gold_rank_lam2": 0.50}}
INTERACT_LAM = {{"gold_rank_interact": 3.5, "gold_rank_w50": 3.5, "gold_rank_w70": 3.5, "gold_rank_w40": 3.5, "gold_rank_lam2": 2.0}}
LEARNED = {{"gold_meta_logit", "gold_rank_interact", "gold_rank_w50", "gold_rank_w70", "gold_rank_w40", "gold_rank_lam2"}}
if STRATEGY in LEARNED and train_series is not None:
    scores7, n7 = learned_scores(False, 2.0)
    ranked = None
    if STRATEGY in BLEND_W7:
        w7 = float(BLEND_W7[STRATEGY])
        lamI = float(INTERACT_LAM.get(STRATEGY, 3.5))
        scoresI, nI = learned_scores(True, lamI)
        if scores7 is not None and scoresI is not None:
            # Frozen gold_rank_w50 is 0.50/0.50 at λI=3.5, public 0.518. lam2 keeps that blend, λI=2.0.
            ranked = w7 * rank_cols(scores7) + (1.0 - w7) * rank_cols(scoresI)
            print(STRATEGY, "blend", w7, "*7d +", 1.0 - w7, "*plane-protocol", "lamI", lamI, "n7", n7, "nI", nI)
        elif scores7 is not None:
            ranked = rank_cols(scores7)
            print(STRATEGY, "fallback to 7d", "n7", n7, "nI", nI)
    elif scores7 is not None:
        ranked = rank_cols(scores7)
        print("gold_meta_logit used_learned", True, "n_gold_with_series", n7)
    if ranked is not None:
        out = sample[[study_col]].copy()
        for j, t in enumerate(targets):
            out[t] = np.clip(prev[t] + 0.25 * (ranked[:, j] - 0.5), EPS, 1 - EPS)
        out = out[sample.columns]
        used_learned = True
    else:
        print("learned metadata path skipped; n7", n7, "nI", nI)

if STRATEGY == "rank_ensemble_safe":
    prev_arr = np.array(prev_mat)
    meta_arr = np.array(meta_mat)
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
        "id": f"kaggle-user/{kernel_slug}",
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
