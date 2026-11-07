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
import re
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
    if strategy in {{"metadata_prior_blend", "report_shrinkage_priors", "fluid_gate_metadata", "rank_ensemble_safe", "gold_meta_logit", "gold_rank_interact", "gold_rank_w50", "gold_rank_w70", "gold_rank_w40", "gold_rank_lam2", "weak_rank_calibrate", "weak_rank_goldfill", "weak_rank_confident", "weak_rank_named", "weak_rank_named_mix", "weak_rank_bakers_mix", "weak_rank_bakers_w60", "weak_rank_bakers_w40", "weak_rank_bakers_acl_mix", "weak_rank_bakers_mm_mix", "weak_rank_bakers_goldstd", "weak_rank_bakers_silence", "weak_rank_bakers_dropfat", "weak_rank_bakers_llm", "weak_rank_bakers_acl_llm", "weak_rank_bakers_mcl_llm", "weak_rank_bakers_eff_llm", "weak_rank_bakers_syn_llm", "weak_rank_bakers_syn_from_eff", "weak_rank_bakers_cont_llm", "weak_rank_bakers_frac_llm", "weak_rank_bakers_moa_llm", "weak_rank_bakers_loa_llm", "weak_rank_bakers_pfoa_llm", "weak_rank_bakers_frac_hyb_llm", "weak_rank_bakers_frac_hyb_w60", "weak_rank_bakers_frac_hyb_w40", "weak_rank_bakers_frac_gem_llm", "weak_rank_bakers_frac_hyb_gem_llm", "weak_rank_bakers_frac_hyb_mm_llm", "weak_rank_bakers_frac_hyb_lm_llm", "weak_rank_bakers_frac_hyb_lm_w60", "weak_rank_bakers_frac_hyb_lm_w40", "weak_rank_bakers_frac_hyb_lm_eff_w60", "weak_rank_bakers_frac_hyb_lm_eff_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_acl_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_eff_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_cheng"}}:
        fb = FLUID_BOOST.get(t, 0.0)
        if strategy == "fluid_gate_metadata":
            fb *= 1.5
        off += fb * (feats.get("fluid", 0.0) - 0.5)
        off += FAT_BOOST.get(t, 0.0) * (feats.get("fat", 0.0) - 0.5)
    return float(off)

def feat_map(series_df, interact=False, drop_fat=False):
    """Per-study vector: intercept, log1p(n), sag/cor/ax fractions, fluid frac, fat frac.
    If interact, also sag/cor/ax × fluid and sag/cor/ax × fat fractions (13-d).
    drop_fat=True drops fat and plane×fat (Fluid_Sensitive ≡ Fat_Suppression): 6-d / 9-d.
    """
    if drop_fat:
        dim = 9 if interact else 6
    else:
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
        ]
        if not drop_fat:
            vec.append(float(row["fat"] / n))
        if interact:
            vec += [
                float(row["sag_fl"] / n),
                float(row["cor_fl"] / n),
                float(row["ax_fl"] / n),
            ]
            if not drop_fat:
                vec += [
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
        if STRATEGY in {{"metadata_prior_blend", "report_shrinkage_priors", "fluid_gate_metadata", "gold_meta_logit", "gold_rank_interact", "gold_rank_w50", "gold_rank_w70", "gold_rank_w40", "gold_rank_lam2", "weak_rank_calibrate", "weak_rank_goldfill", "weak_rank_confident", "weak_rank_named", "weak_rank_named_mix", "weak_rank_bakers_mix", "weak_rank_bakers_w60", "weak_rank_bakers_w40", "weak_rank_bakers_acl_mix", "weak_rank_bakers_mm_mix", "weak_rank_bakers_goldstd", "weak_rank_bakers_silence", "weak_rank_bakers_dropfat", "weak_rank_bakers_llm", "weak_rank_bakers_acl_llm", "weak_rank_bakers_mcl_llm", "weak_rank_bakers_eff_llm", "weak_rank_bakers_syn_llm", "weak_rank_bakers_syn_from_eff", "weak_rank_bakers_cont_llm", "weak_rank_bakers_frac_llm", "weak_rank_bakers_moa_llm", "weak_rank_bakers_loa_llm", "weak_rank_bakers_pfoa_llm", "weak_rank_bakers_frac_hyb_llm", "weak_rank_bakers_frac_hyb_w60", "weak_rank_bakers_frac_hyb_w40", "weak_rank_bakers_frac_gem_llm", "weak_rank_bakers_frac_hyb_gem_llm", "weak_rank_bakers_frac_hyb_mm_llm", "weak_rank_bakers_frac_hyb_lm_llm", "weak_rank_bakers_frac_hyb_lm_w60", "weak_rank_bakers_frac_hyb_lm_w40", "weak_rank_bakers_frac_hyb_lm_eff_w60", "weak_rank_bakers_frac_hyb_lm_eff_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_acl_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_eff_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_cheng"}}:
            p = float(sigmoid(logit(p0) + off))
        else:
            p = p0
        # Hand-tuned strategies keep tiny jitter; learned ranking must not be scrambled.
        if STRATEGY not in {{"gold_meta_logit", "gold_rank_interact", "gold_rank_w50", "gold_rank_w70", "gold_rank_w40", "gold_rank_lam2", "weak_rank_calibrate", "weak_rank_goldfill", "weak_rank_confident", "weak_rank_named", "weak_rank_named_mix", "weak_rank_bakers_mix", "weak_rank_bakers_w60", "weak_rank_bakers_w40", "weak_rank_bakers_acl_mix", "weak_rank_bakers_mm_mix", "weak_rank_bakers_goldstd", "weak_rank_bakers_silence", "weak_rank_bakers_dropfat", "weak_rank_bakers_llm", "weak_rank_bakers_acl_llm", "weak_rank_bakers_mcl_llm", "weak_rank_bakers_eff_llm", "weak_rank_bakers_syn_llm", "weak_rank_bakers_syn_from_eff", "weak_rank_bakers_cont_llm", "weak_rank_bakers_frac_llm", "weak_rank_bakers_moa_llm", "weak_rank_bakers_loa_llm", "weak_rank_bakers_pfoa_llm", "weak_rank_bakers_frac_hyb_llm", "weak_rank_bakers_frac_hyb_w60", "weak_rank_bakers_frac_hyb_w40", "weak_rank_bakers_frac_gem_llm", "weak_rank_bakers_frac_hyb_gem_llm", "weak_rank_bakers_frac_hyb_mm_llm", "weak_rank_bakers_frac_hyb_lm_llm", "weak_rank_bakers_frac_hyb_lm_w60", "weak_rank_bakers_frac_hyb_lm_w40", "weak_rank_bakers_frac_hyb_lm_eff_w60", "weak_rank_bakers_frac_hyb_lm_eff_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_acl_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_eff_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_cheng"}}:
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
    drop_fat = STRATEGY == "weak_rank_bakers_dropfat"
    tr_map, default_x = feat_map(train_series, interact=interact, drop_fat=drop_fat)
    te_map, _ = feat_map(test_series, interact=interact, drop_fat=drop_fat)
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

# Train-report soft labels only. Never read test.csv / test reports.
SYN = {{
    "ACL": (r"\\bacl\\b", r"anterior\\s+cruciate", r"\\blca\\b", r"vorderes?\\s+kreuzband", r"ligament\\s+crois"),
    "MCL": (r"\\bmcl\\b", r"medial\\s+collateral", r"innenband", r"ligament\\s+collat"),
    "Medial Meniscus": (r"medial\\s+meniscus", r"m[eé]nisque\\s+m[eé]dial", r"innenmeniskus"),
    "Lateral Meniscus": (r"lateral\\s+meniscus", r"m[eé]nisque\\s+lat", r"aussenmeniskus|außenmeniskus"),
    "Medial OA": (r"medial.{{0,24}}(?:oa|osteoarthrit|arthros)", r"(?:oa|osteoarthrit|arthros).{{0,24}}medial"),
    "Lateral OA": (r"lateral.{{0,24}}(?:oa|osteoarthrit|arthros)", r"(?:oa|osteoarthrit|arthros).{{0,24}}lateral"),
    "PF OA": (r"patello-?femoral.{{0,20}}(?:oa|osteoarthrit|arthros)", r"(?:pf\\s+oa|\\bpf\\b.{{0,12}}oa)"),
    "Effusion": (r"\\beffusion\\b", r"[eé]panchement", r"gelenkerguss", r"ef[uü]zyon"),
    "Synovitis": (r"\\bsynovitis\\b", r"synovite", r"sinovit"),
    "Baker's": (r"baker'?s?\\s+(?:cyst|zyste)", r"popliteal\\s+cyst", r"kyste\\s+poplit"),
    "Contusion": (r"\\bcontusion\\b", r"bone\\s+bruise", r"bone\\s+marrow\\s+edema", r"knochenmark"),
    "Fracture": (r"\\bfracture\\b", r"\\bfractura\\b", r"\\bbruch\\b", r"fractur"),
}}
NEG = (
    r"\\bno\\b", r"\\bwithout\\b", r"\\bintact\\b", r"\\babsent\\b", r"\\bnormal\\b", r"\\bnegative\\b",
    r"\\bunremarkable\\b", r"\\bsans\\b", r"\\bpas\\s+de\\b", r"\\bkein", r"\\bohne\\b",
    r"izlenmedi", r"görülmedi", r"gorulmedi", r"saptanmad", r"tespit edilmedi", r"\\byok\\b",
    r"mevcut de[gğ]il", r"negatif",
)
UNC = (r"possible", r"probable", r"suggestive", r"cannot\\s+exclude", r"equivocal", r"questionable")
HIST = (r"status\\s+post", r"\\bs/p\\b", r"\\bprior\\b", r"\\bprevious\\b", r"history\\s+of")
SOFT_MAP = {{"pos": 0.85, "neg": 0.12, "unc": 0.50, "hist": 0.40, "unmentioned": 0.38}}

def parse_report_states(text, tcols):
    """Evidence states from a train report. Right-side window catches Turkish post-negation."""
    norm = re.sub(r"\\s+", " ", str(text or "").lower())
    out = {{}}
    for t in tcols:
        state = "unmentioned"
        for pat in SYN.get(t, (re.escape(t.lower()),)):
            for m in re.finditer(pat, norm, flags=re.IGNORECASE):
                lo = max(0, m.start() - 50)
                hi = min(len(norm), m.end() + 50)
                win = norm[lo:hi]
                right = norm[m.end(): min(len(norm), m.end() + 40)]
                if any(re.search(p, win) for p in NEG) or any(re.search(p, right) for p in NEG):
                    state = "neg"
                elif state != "neg" and any(re.search(p, win) for p in HIST):
                    state = "hist" if state == "unmentioned" else state
                elif state not in ("neg", "pos") and any(re.search(p, win) for p in UNC):
                    state = "unc"
                elif state != "neg":
                    state = "pos"
        out[t] = state
    return out

def parse_report_soft(text, tcols):
    """Soft labels from a train report. Right-side window catches Turkish post-negation."""
    return {{t: float(SOFT_MAP[s]) for t, s in parse_report_states(text, tcols).items()}}

NAMED_PARSER = {{"ACL", "Baker's", "MCL"}}
NAMED_PLUS_MM = {{"ACL", "Baker's", "MCL", "Medial Meniscus"}}
BAKERS_ONLY = {{"Baker's"}}
BAKERS_ACL = {{"Baker's", "ACL"}}
BAKERS_MM = {{"Baker's", "Medial Meniscus"}}
MIX_TARGETS = {{
    "weak_rank_named_mix": NAMED_PARSER,
    "weak_rank_bakers_mix": BAKERS_ONLY,
    "weak_rank_bakers_w60": BAKERS_ONLY,
    "weak_rank_bakers_w40": BAKERS_ONLY,
    "weak_rank_bakers_acl_mix": BAKERS_ACL,
    "weak_rank_bakers_mm_mix": BAKERS_MM,
    "weak_rank_bakers_goldstd": BAKERS_ONLY,
    "weak_rank_bakers_silence": BAKERS_ONLY,
    "weak_rank_bakers_dropfat": BAKERS_ONLY,
    "weak_rank_bakers_llm": BAKERS_ONLY,
    "weak_rank_bakers_acl_llm": BAKERS_ONLY,
    "weak_rank_bakers_mcl_llm": BAKERS_ONLY,
    "weak_rank_bakers_eff_llm": BAKERS_ONLY,
    "weak_rank_bakers_syn_llm": BAKERS_ONLY,
    "weak_rank_bakers_syn_from_eff": BAKERS_ONLY,
    "weak_rank_bakers_cont_llm": BAKERS_ONLY,
    "weak_rank_bakers_frac_llm": BAKERS_ONLY,
    "weak_rank_bakers_moa_llm": BAKERS_ONLY,
    "weak_rank_bakers_loa_llm": BAKERS_ONLY,
    "weak_rank_bakers_pfoa_llm": BAKERS_ONLY,
    "weak_rank_bakers_frac_hyb_llm": BAKERS_ONLY,
    "weak_rank_bakers_frac_hyb_w60": BAKERS_ONLY,
    "weak_rank_bakers_frac_hyb_w40": BAKERS_ONLY,
    "weak_rank_bakers_frac_gem_llm": BAKERS_ONLY,
    "weak_rank_bakers_frac_hyb_gem_llm": BAKERS_ONLY,
    "weak_rank_bakers_frac_hyb_mm_llm": BAKERS_ONLY,
    "weak_rank_bakers_frac_hyb_lm_llm": BAKERS_ONLY,
    "weak_rank_bakers_frac_hyb_lm_w60": BAKERS_ONLY,
    "weak_rank_bakers_frac_hyb_lm_w40": BAKERS_ONLY,
    "weak_rank_bakers_frac_hyb_lm_eff_w60": BAKERS_ONLY,
    "weak_rank_bakers_frac_hyb_lm_eff_w40": BAKERS_ONLY,
    "weak_rank_bakers_frac_hyb_lm_eff_acl_w60": BAKERS_ONLY,
    "weak_rank_bakers_frac_hyb_lm_eff_acl_w40": BAKERS_ONLY,
    "weak_rank_bakers_frac_hyb_lm_eff_acl_gem": BAKERS_ONLY,
    "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60": BAKERS_ONLY,
    "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40": BAKERS_ONLY,
    "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem": BAKERS_ONLY,
    "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60": BAKERS_ONLY,
    "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40": BAKERS_ONLY,
    "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_acl_qwen": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_eff_qwen": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w60": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w40": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_qwen": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_qwen": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_front": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_front": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_front": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_front": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_front": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_pilk": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_pilk": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_pilk": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_pilk": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_2src": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_2src": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_2src": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_2src": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_sol": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_sol": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_sol": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_sol": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_3src": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_3src": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_3src": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_3src": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_song": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_song": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_song": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_song": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_cheng": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_cheng": BAKERS_ONLY, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_cheng": BAKERS_ONLY,
}}
MIX_GOLD_W = {{
    "weak_rank_named_mix": 0.50,
    "weak_rank_bakers_mix": 0.50,
    "weak_rank_bakers_w60": 0.40,
    "weak_rank_bakers_w40": 0.60,
    "weak_rank_bakers_acl_mix": 0.50,
    "weak_rank_bakers_mm_mix": 0.50,
    "weak_rank_bakers_goldstd": 0.50,
    "weak_rank_bakers_silence": 0.50,
    "weak_rank_bakers_dropfat": 0.50,
    "weak_rank_bakers_llm": 0.50,
    "weak_rank_bakers_acl_llm": 0.50,
    "weak_rank_bakers_mcl_llm": 0.50,
    "weak_rank_bakers_eff_llm": 0.50,
    "weak_rank_bakers_syn_llm": 0.50,
    "weak_rank_bakers_syn_from_eff": 0.50,
    "weak_rank_bakers_cont_llm": 0.50,
    "weak_rank_bakers_frac_llm": 0.50,
    "weak_rank_bakers_moa_llm": 0.50,
    "weak_rank_bakers_loa_llm": 0.50,
    "weak_rank_bakers_pfoa_llm": 0.50,
    "weak_rank_bakers_frac_hyb_llm": 0.50,
    "weak_rank_bakers_frac_hyb_w60": 0.50,
    "weak_rank_bakers_frac_hyb_w40": 0.50,
    "weak_rank_bakers_frac_gem_llm": 0.50,
    "weak_rank_bakers_frac_hyb_gem_llm": 0.50,
    "weak_rank_bakers_frac_hyb_mm_llm": 0.50,
    "weak_rank_bakers_frac_hyb_lm_llm": 0.50,
    "weak_rank_bakers_frac_hyb_lm_w60": 0.50,
    "weak_rank_bakers_frac_hyb_lm_w40": 0.50,
    "weak_rank_bakers_frac_hyb_lm_eff_w60": 0.50,
    "weak_rank_bakers_frac_hyb_lm_eff_w40": 0.50,
    "weak_rank_bakers_frac_hyb_lm_eff_acl_w60": 0.50,
    "weak_rank_bakers_frac_hyb_lm_eff_acl_w40": 0.50,
    "weak_rank_bakers_frac_hyb_lm_eff_acl_gem": 0.50,
    "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60": 0.50,
    "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40": 0.50,
    "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem": 0.50,
    "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_acl_qwen": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_eff_qwen": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w60": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w40": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_qwen": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_qwen": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_front": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_front": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_front": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_front": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_front": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_pilk": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_pilk": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_pilk": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_pilk": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_2src": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_2src": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_2src": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_2src": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_sol": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_sol": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_sol": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_sol": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_3src": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_3src": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_3src": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_3src": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_song": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_song": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_song": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_song": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_cheng": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_cheng": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_cheng": 0.50,
}}
MIX_NAMED_SET = {{
    "weak_rank_bakers_mm_mix": NAMED_PLUS_MM,
}}

def learned_scores_weak(interact=False, lam=2.0, prefer_gold=False, confident_only=False, named_only=False, named_targets=None, std_on_gold=False, silence_neg_targets=None):
    """Ridge logits on train-report soft labels. Returns (scores, n_weak) or (None, n).
    prefer_gold=True keeps expert 0/1 on the 58 and uses the parser only for unlabeled rows.
    confident_only=True keeps only parser pos/neg; masks unmentioned/unc/hist (discussion 734117).
    named_only=True further restricts parser labels to ACL / Baker's / MCL.
    std_on_gold=True uses feature μ/σ from gold-labeled studies (same space as gold_rank_w50).
    silence_neg_targets maps unmentioned cells to neg for those findings (Baker's silence ≈ negative, discussion 733932).
    """
    if train_series is None or "Report" not in train.columns:
        return None, 0
    drop_fat = STRATEGY == "weak_rank_bakers_dropfat"
    tr_map, default_x = feat_map(train_series, interact=interact, drop_fat=drop_fat)
    te_map, _ = feat_map(test_series, interact=interact, drop_fat=drop_fat)
    X_rows, y_cols = [], {{t: [] for t in targets}}
    for _, row in train.iterrows():
        sid = str(row[study_col])
        x = tr_map.get(sid)
        if x is None:
            continue
        raw = row.get("Report")
        if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
            continue
        # Guard: never treat a test UID report as supervision (test reports are forbidden).
        if sid in set(uids):
            continue
        states = parse_report_states(raw, targets)
        X_rows.append(x)
        for t in targets:
            y_use = float("nan")
            if prefer_gold and t in train.columns:
                try:
                    gv = float(row[t])
                except (TypeError, ValueError):
                    gv = float("nan")
                if np.isfinite(gv):
                    y_cols[t].append(float(gv))
                    continue
            named_set = set(named_targets) if named_targets else NAMED_PARSER
            if named_only and t not in named_set:
                y_cols[t].append(float("nan"))
                continue
            state = states[t]
            silence_neg = set(silence_neg_targets or ())
            if t in silence_neg and state == "unmentioned":
                y_cols[t].append(float(SOFT_MAP["neg"]))
            elif confident_only and state not in ("pos", "neg"):
                y_cols[t].append(float("nan"))
            else:
                y_cols[t].append(float(SOFT_MAP[state]))
    n_weak = len(X_rows)
    if n_weak < 20:
        return None, n_weak
    X = np.vstack(X_rows)
    if std_on_gold:
        gold_xs = []
        for _, grow in train.iterrows():
            gx = tr_map.get(str(grow[study_col]))
            if gx is None:
                continue
            has = False
            for t in targets:
                if t not in train.columns:
                    continue
                try:
                    gv = float(grow[t])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(gv):
                    has = True
                    break
            if has:
                gold_xs.append(gx)
        if len(gold_xs) >= 12:
            Xg = np.vstack(gold_xs)
            mu = Xg.mean(axis=0)
            sd = np.clip(Xg.std(axis=0), 1e-6, None)
            print(STRATEGY, "std_on_gold n_gold_std", len(gold_xs))
        else:
            mu = X.mean(axis=0)
            sd = np.clip(X.std(axis=0), 1e-6, None)
            print(STRATEGY, "std_on_gold fallback to weak-row moments n_gold_std", len(gold_xs))
    else:
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
    return scores, n_weak

def discover_llm_labels():
    """Find llm_labels_v4_blend.csv without walking DICOM trees.

    Kaggle now mounts extra datasets under /kaggle/input/datasets/<owner>/<slug>/,
    which is deeper than one/two levels from /kaggle/input. Check known mounts
    first, then BFS a few levels and skip image/DICOM folders (no rglob).
    """
    names = {{"llm_labels_v4_blend.csv"}}
    skip_dirs = {{
        "train", "test", "train_images", "test_images", "images",
        "dicom", "dcms", "series",
    }}
    explicit = [
        inp / "datasets" / "stevenleehans" / "rsna-knee-llm-report-labels" / "llm_labels_v4_blend.csv",
        inp / "rsna-knee-llm-report-labels" / "llm_labels_v4_blend.csv",
        inp / "stevenleehans" / "rsna-knee-llm-report-labels" / "llm_labels_v4_blend.csv",
        ROOT / "llm_labels_v4_blend.csv",
    ]
    for p in explicit:
        if p.is_file():
            print(STRATEGY, "llm csv explicit", p)
            return p
    ds = inp / "datasets"
    if ds.exists():
        try:
            print(STRATEGY, "datasets children", [c.name for c in ds.iterdir()][:24])
        except OSError:
            print(STRATEGY, "datasets listing failed")
    roots = []
    if inp.exists():
        roots.append(inp)
    roots.append(ROOT)
    parent = ROOT.parent
    if parent != ROOT:
        roots.append(parent)
    seen = set()
    queue = [(r, 0) for r in roots if r.exists() and r.is_dir()]
    while queue:
        base, depth = queue.pop(0)
        try:
            key = str(base.resolve())
        except OSError:
            key = str(base)
        if key in seen:
            continue
        seen.add(key)
        try:
            children = list(base.iterdir())
        except OSError:
            continue
        has_sample = any(c.is_file() and c.name == "sample_submission.csv" for c in children)
        for child in children:
            if child.is_file():
                cname = child.name.lower()
                if cname in names or (cname.startswith("llm_labels_v4") and cname.endswith(".csv")):
                    print(STRATEGY, "llm csv bfs", child, "depth", depth)
                    return child
            if not child.is_dir() or depth >= 6:
                continue
            cname = child.name.lower()
            if cname in skip_dirs:
                continue
            if has_sample and cname in {{"train", "test", "train_images", "test_images"}}:
                continue
            queue.append((child, depth + 1))
    print(STRATEGY, "llm search visited", len(seen), "dirs; csv missing")
    return None

def discover_hybrid_labels():
    """Find report_labels_v4hybrid.csv without walking DICOM trees."""
    names = {{"report_labels_v4hybrid.csv"}}
    skip_dirs = {{
        "train", "test", "train_images", "test_images", "images",
        "dicom", "dcms", "series",
    }}
    explicit = [
        inp / "datasets" / "flight0234" / "rsna-knee-hybrid-report-labels" / "report_labels_v4hybrid.csv",
        inp / "rsna-knee-hybrid-report-labels" / "report_labels_v4hybrid.csv",
        inp / "flight0234" / "rsna-knee-hybrid-report-labels" / "report_labels_v4hybrid.csv",
        ROOT / "report_labels_v4hybrid.csv",
    ]
    for p in explicit:
        if p.is_file():
            print(STRATEGY, "hybrid csv explicit", p)
            return p
    roots = []
    if inp.exists():
        roots.append(inp)
    roots.append(ROOT)
    parent = ROOT.parent
    if parent != ROOT:
        roots.append(parent)
    seen = set()
    queue = [(r, 0) for r in roots if r.exists() and r.is_dir()]
    while queue:
        base, depth = queue.pop(0)
        try:
            key = str(base.resolve())
        except OSError:
            key = str(base)
        if key in seen:
            continue
        seen.add(key)
        try:
            children = list(base.iterdir())
        except OSError:
            continue
        has_sample = any(c.is_file() and c.name == "sample_submission.csv" for c in children)
        for child in children:
            if child.is_file():
                cname = child.name.lower()
                if cname in names or ("hybrid" in cname and cname.endswith(".csv") and "report_labels" in cname):
                    print(STRATEGY, "hybrid csv bfs", child, "depth", depth)
                    return child
            if not child.is_dir() or depth >= 6:
                continue
            cname = child.name.lower()
            if cname in skip_dirs:
                continue
            if has_sample and cname in {{"train", "test", "train_images", "test_images"}}:
                continue
            queue.append((child, depth + 1))
    print(STRATEGY, "hybrid search visited", len(seen), "dirs; csv missing")
    return None

def discover_gemini_labels():
    """Find report_labels_v5.csv without walking DICOM trees."""
    names = {{"report_labels_v5.csv"}}
    skip_dirs = {{
        "train", "test", "train_images", "test_images", "images",
        "dicom", "dcms", "series",
    }}
    explicit = [
        inp / "datasets" / "yunusgmsoy" / "rsna-knee-llm-labels-4-source-merged" / "report_labels_v5.csv",
        inp / "rsna-knee-llm-labels-4-source-merged" / "report_labels_v5.csv",
        inp / "yunusgmsoy" / "rsna-knee-llm-labels-4-source-merged" / "report_labels_v5.csv",
        ROOT / "report_labels_v5.csv",
    ]
    for p in explicit:
        if p.is_file():
            print(STRATEGY, "gemini csv explicit", p)
            return p
    roots = []
    if inp.exists():
        roots.append(inp)
    roots.append(ROOT)
    parent = ROOT.parent
    if parent != ROOT:
        roots.append(parent)
    seen = set()
    queue = [(r, 0) for r in roots if r.exists() and r.is_dir()]
    while queue:
        base, depth = queue.pop(0)
        try:
            key = str(base.resolve())
        except OSError:
            key = str(base)
        if key in seen:
            continue
        seen.add(key)
        try:
            children = list(base.iterdir())
        except OSError:
            continue
        has_sample = any(c.is_file() and c.name == "sample_submission.csv" for c in children)
        for child in children:
            if child.is_file():
                cname = child.name.lower()
                if cname in names or (cname.startswith("report_labels_v5") and cname.endswith(".csv")):
                    print(STRATEGY, "gemini csv bfs", child, "depth", depth)
                    return child
            if not child.is_dir() or depth >= 6:
                continue
            cname = child.name.lower()
            if cname in skip_dirs:
                continue
            if has_sample and cname in {{"train", "test", "train_images", "test_images"}}:
                continue
            queue.append((child, depth + 1))
    print(STRATEGY, "gemini search visited", len(seen), "dirs; csv missing")
    return None

def discover_qwen_labels():
    """Find Qwen2.5-32B v2 weak_labels_v2_filled.csv without walking DICOM trees."""
    names = {{"weak_labels_v2_filled.csv", "weak_labels_v2_raw.csv"}}
    skip_dirs = {{
        "train", "test", "train_images", "test_images", "images",
        "dicom", "dcms", "series",
    }}
    explicit = [
        inp / "datasets" / "laymond" / "rsna-knee-abnormality-qwen3-8b-weak-labels" / "weak_labels_v2_filled.csv",
        inp / "rsna-knee-abnormality-qwen3-8b-weak-labels" / "weak_labels_v2_filled.csv",
        inp / "laymond" / "rsna-knee-abnormality-qwen3-8b-weak-labels" / "weak_labels_v2_filled.csv",
        ROOT / "weak_labels_v2_filled.csv",
        inp / "datasets" / "laymond" / "rsna-knee-abnormality-qwen3-8b-weak-labels" / "weak_labels_v2_raw.csv",
        ROOT / "weak_labels_v2_raw.csv",
    ]
    for p in explicit:
        if p.is_file():
            print(STRATEGY, "qwen csv explicit", p)
            return p
    roots = []
    if inp.exists():
        roots.append(inp)
    roots.append(ROOT)
    parent = ROOT.parent
    if parent != ROOT:
        roots.append(parent)
    seen = set()
    queue = [(r, 0) for r in roots if r.exists() and r.is_dir()]
    while queue:
        base, depth = queue.pop(0)
        try:
            key = str(base.resolve())
        except OSError:
            key = str(base)
        if key in seen:
            continue
        seen.add(key)
        try:
            children = list(base.iterdir())
        except OSError:
            continue
        has_sample = any(c.is_file() and c.name == "sample_submission.csv" for c in children)
        for child in children:
            if child.is_file():
                cname = child.name.lower()
                if cname in names:
                    print(STRATEGY, "qwen csv bfs", child, "depth", depth)
                    return child
            if not child.is_dir() or depth >= 6:
                continue
            cname = child.name.lower()
            if cname in skip_dirs:
                continue
            if has_sample and cname in {{"train", "test", "train_images", "test_images"}}:
                continue
            queue.append((child, depth + 1))
    print(STRATEGY, "qwen search visited", len(seen), "dirs; csv missing")
    return None

def discover_front_labels():
    """Find Deotte frontier Argon+Astra+Opus blend CSV without walking DICOM trees."""
    names = {{"blend_top3_argon_astra_opus_all4314_labels.csv"}}
    skip_dirs = {{
        "train", "test", "train_images", "test_images", "images",
        "dicom", "dcms", "series",
    }}
    explicit = [
        inp / "datasets" / "cdeotte" / "rsna-knee-frontier-llm-hybrid-labels" / "blend_top3_argon_astra_opus_all4314_labels.csv",
        inp / "rsna-knee-frontier-llm-hybrid-labels" / "blend_top3_argon_astra_opus_all4314_labels.csv",
        inp / "cdeotte" / "rsna-knee-frontier-llm-hybrid-labels" / "blend_top3_argon_astra_opus_all4314_labels.csv",
        ROOT / "blend_top3_argon_astra_opus_all4314_labels.csv",
    ]
    for p in explicit:
        if p.is_file():
            print(STRATEGY, "front csv explicit", p)
            return p
    roots = []
    if inp.exists():
        roots.append(inp)
    roots.append(ROOT)
    parent = ROOT.parent
    if parent != ROOT:
        roots.append(parent)
    seen = set()
    queue = [(r, 0) for r in roots if r.exists() and r.is_dir()]
    while queue:
        base, depth = queue.pop(0)
        try:
            key = str(base.resolve())
        except OSError:
            key = str(base)
        if key in seen:
            continue
        seen.add(key)
        try:
            children = list(base.iterdir())
        except OSError:
            continue
        has_sample = any(c.is_file() and c.name == "sample_submission.csv" for c in children)
        for child in children:
            if child.is_file():
                cname = child.name.lower()
                if cname in names:
                    print(STRATEGY, "front csv bfs", child, "depth", depth)
                    return child
            if not child.is_dir() or depth >= 6:
                continue
            cname = child.name.lower()
            if cname in skip_dirs:
                continue
            if has_sample and cname in {{"train", "test", "train_images", "test_images"}}:
                continue
            queue.append((child, depth + 1))
    print(STRATEGY, "front search visited", len(seen), "dirs; csv missing")
    return None

def discover_pilk_labels():
    """Find pilkwang report_labels_v2.csv without walking DICOM trees."""
    names = {{"report_labels_v2.csv"}}
    skip_dirs = {{
        "train", "test", "train_images", "test_images", "images",
        "dicom", "dcms", "series",
    }}
    explicit = [
        inp / "datasets" / "pilkwang" / "rsna-knee-llm-labels" / "report_labels_v2.csv",
        inp / "rsna-knee-llm-labels" / "report_labels_v2.csv",
        inp / "pilkwang" / "rsna-knee-llm-labels" / "report_labels_v2.csv",
        ROOT / "report_labels_v2.csv",
    ]
    for p in explicit:
        if p.is_file():
            print(STRATEGY, "pilk csv explicit", p)
            return p
    roots = []
    if inp.exists():
        roots.append(inp)
    roots.append(ROOT)
    parent = ROOT.parent
    if parent != ROOT:
        roots.append(parent)
    seen = set()
    queue = [(r, 0) for r in roots if r.exists() and r.is_dir()]
    while queue:
        base, depth = queue.pop(0)
        try:
            key = str(base.resolve())
        except OSError:
            key = str(base)
        if key in seen:
            continue
        seen.add(key)
        try:
            children = list(base.iterdir())
        except OSError:
            continue
        has_sample = any(c.is_file() and c.name == "sample_submission.csv" for c in children)
        for child in children:
            if child.is_file():
                cname = child.name.lower()
                if cname in names:
                    print(STRATEGY, "pilk csv bfs", child, "depth", depth)
                    return child
            if not child.is_dir() or depth >= 6:
                continue
            cname = child.name.lower()
            if cname in skip_dirs:
                continue
            if has_sample and cname in {{"train", "test", "train_images", "test_images"}}:
                continue
            queue.append((child, depth + 1))
    print(STRATEGY, "pilk search visited", len(seen), "dirs; csv missing")
    return None

def discover_2src_labels():
    """Find yunusgmsoy 2-source report_labels_v2.csv without walking DICOM trees."""
    names = {{"report_labels_v2.csv"}}
    skip_dirs = {{
        "train", "test", "train_images", "test_images", "images",
        "dicom", "dcms", "series",
    }}
    explicit = [
        inp / "datasets" / "yunusgmsoy" / "rsna-knee-llm-labels-2-source-merged" / "report_labels_v2.csv",
        inp / "rsna-knee-llm-labels-2-source-merged" / "report_labels_v2.csv",
        inp / "yunusgmsoy" / "rsna-knee-llm-labels-2-source-merged" / "report_labels_v2.csv",
        ROOT / "report_labels_v2.csv",
    ]
    for p in explicit:
        if p.is_file():
            print(STRATEGY, "2src csv explicit", p)
            return p
    roots = []
    if inp.exists():
        roots.append(inp)
    roots.append(ROOT)
    parent = ROOT.parent
    if parent != ROOT:
        roots.append(parent)
    seen = set()
    queue = [(r, 0) for r in roots if r.exists() and r.is_dir()]
    while queue:
        base, depth = queue.pop(0)
        try:
            key = str(base.resolve())
        except OSError:
            key = str(base)
        if key in seen:
            continue
        seen.add(key)
        try:
            children = list(base.iterdir())
        except OSError:
            continue
        has_sample = any(c.is_file() and c.name == "sample_submission.csv" for c in children)
        for child in children:
            if child.is_file():
                cname = child.name.lower()
                path_l = str(child).lower()
                if cname in names and ("2-source" in path_l or "2_source" in path_l or child.parent == ROOT):
                    print(STRATEGY, "2src csv bfs", child, "depth", depth)
                    return child
            if not child.is_dir() or depth >= 6:
                continue
            cname = child.name.lower()
            if cname in skip_dirs:
                continue
            if has_sample and cname in {{"train", "test", "train_images", "test_images"}}:
                continue
            queue.append((child, depth + 1))
    print(STRATEGY, "2src search visited", len(seen), "dirs; csv missing")
    return None


def discover_sol_labels():
    """Find lixin73 GPT-5.6-Sol report labels without walking DICOM trees."""
    names = {{"report_labels_gpt56sol.csv", "labels_llm_gpt56sol.csv"}}
    skip_dirs = {{
        "train", "test", "train_images", "test_images", "images",
        "dicom", "dcms", "series",
    }}
    explicit = [
        inp / "datasets" / "lixin73" / "rsna-knee-llm-report-labels-sol56" / "report_labels_gpt56sol.csv",
        inp / "rsna-knee-llm-report-labels-sol56" / "report_labels_gpt56sol.csv",
        inp / "lixin73" / "rsna-knee-llm-report-labels-sol56" / "report_labels_gpt56sol.csv",
        ROOT / "report_labels_gpt56sol.csv",
        inp / "datasets" / "lixin73" / "rsna-knee-llm-report-labels-sol56" / "labels_llm_gpt56sol.csv",
        ROOT / "labels_llm_gpt56sol.csv",
    ]
    for pth in explicit:
        if pth.is_file():
            print(STRATEGY, "sol csv explicit", pth)
            return pth
    roots = []
    if inp.exists():
        roots.append(inp)
    roots.append(ROOT)
    parent = ROOT.parent
    if parent != ROOT:
        roots.append(parent)
    seen = set()
    queue = [(r, 0) for r in roots if r.exists() and r.is_dir()]
    while queue:
        base, depth = queue.pop(0)
        try:
            key = str(base.resolve())
        except OSError:
            key = str(base)
        if key in seen:
            continue
        seen.add(key)
        try:
            children = list(base.iterdir())
        except OSError:
            continue
        has_sample = any(c.is_file() and c.name == "sample_submission.csv" for c in children)
        for child in children:
            if child.is_file():
                cname = child.name.lower()
                path_l = str(child).lower()
                if cname in names and ("sol56" in path_l or "gpt56" in path_l or child.parent == ROOT):
                    print(STRATEGY, "sol csv bfs", child, "depth", depth)
                    return child
            if not child.is_dir() or depth >= 6:
                continue
            cname = child.name.lower()
            if cname in skip_dirs:
                continue
            if has_sample and cname in {{"train", "test", "train_images", "test_images"}}:
                continue
            queue.append((child, depth + 1))
    print(STRATEGY, "sol search visited", len(seen), "dirs; csv missing")
    return None


def discover_3src_labels():
    """Find yunusgmsoy 3-source report_labels_v3.csv without walking DICOM trees."""
    names = {{"report_labels_v3.csv"}}
    skip_dirs = {{
        "train", "test", "train_images", "test_images", "images",
        "dicom", "dcms", "series",
    }}
    explicit = [
        inp / "datasets" / "yunusgmsoy" / "rsna-knee-abnormality-3-source-merged-labels" / "report_labels_v3.csv",
        inp / "rsna-knee-abnormality-3-source-merged-labels" / "report_labels_v3.csv",
        inp / "yunusgmsoy" / "rsna-knee-abnormality-3-source-merged-labels" / "report_labels_v3.csv",
        ROOT / "report_labels_v3.csv",
    ]
    for pth in explicit:
        if pth.is_file():
            print(STRATEGY, "3src csv explicit", pth)
            return pth
    roots = []
    if inp.exists():
        roots.append(inp)
    roots.append(ROOT)
    parent = ROOT.parent
    if parent != ROOT:
        roots.append(parent)
    seen = set()
    queue = [(r, 0) for r in roots if r.exists() and r.is_dir()]
    while queue:
        base, depth = queue.pop(0)
        try:
            key = str(base.resolve())
        except OSError:
            key = str(base)
        if key in seen:
            continue
        seen.add(key)
        try:
            children = list(base.iterdir())
        except OSError:
            continue
        has_sample = any(c.is_file() and c.name == "sample_submission.csv" for c in children)
        for child in children:
            if child.is_file():
                cname = child.name.lower()
                path_l = str(child).lower()
                if cname in names and ("3-source" in path_l or "3_source" in path_l or child.parent == ROOT):
                    print(STRATEGY, "3src csv bfs", child, "depth", depth)
                    return child
            if not child.is_dir() or depth >= 6:
                continue
            cname = child.name.lower()
            if cname in skip_dirs:
                continue
            if has_sample and cname in {{"train", "test", "train_images", "test_images"}}:
                continue
            queue.append((child, depth + 1))
    print(STRATEGY, "3src search visited", len(seen), "dirs; csv missing")
    return None



def discover_song_labels():
    """Find songqizhou 7th-place 9_final_label_with_gold58_studies.csv without walking DICOM trees."""
    names = {{"9_final_label_with_gold58_studies.csv", "8_final_label.csv"}}
    skip_dirs = {{
        "train", "test", "train_images", "test_images", "images",
        "dicom", "dcms", "series",
    }}
    explicit = [
        inp / "datasets" / "songqizhou" / "rsna-knee-llm-labels" / "9_final_label_with_gold58_studies.csv",
        inp / "rsna-knee-llm-labels" / "9_final_label_with_gold58_studies.csv",
        inp / "songqizhou" / "rsna-knee-llm-labels" / "9_final_label_with_gold58_studies.csv",
        ROOT / "9_final_label_with_gold58_studies.csv",
        inp / "datasets" / "songqizhou" / "rsna-knee-llm-labels" / "8_final_label.csv",
        inp / "rsna-knee-llm-labels" / "8_final_label.csv",
        inp / "songqizhou" / "rsna-knee-llm-labels" / "8_final_label.csv",
        ROOT / "8_final_label.csv",
    ]
    for pth in explicit:
        if pth.is_file():
            print(STRATEGY, "song csv explicit", pth)
            return pth
    roots = []
    if inp.exists():
        roots.append(inp)
    roots.append(ROOT)
    parent = ROOT.parent
    if parent != ROOT:
        roots.append(parent)
    seen = set()
    queue = [(r, 0) for r in roots if r.exists() and r.is_dir()]
    while queue:
        base, depth = queue.pop(0)
        try:
            key = str(base.resolve())
        except OSError:
            key = str(base)
        if key in seen:
            continue
        seen.add(key)
        try:
            children = list(base.iterdir())
        except OSError:
            continue
        has_sample = any(c.is_file() and c.name == "sample_submission.csv" for c in children)
        for child in children:
            if child.is_file():
                cname = child.name.lower()
                path_l = str(child).lower()
                if cname in names and ("songqizhou" in path_l or "rsna-knee-llm-labels" in path_l or child.parent == ROOT):
                    print(STRATEGY, "song csv bfs", child, "depth", depth)
                    return child
            if not child.is_dir() or depth >= 6:
                continue
            cname = child.name.lower()
            if cname in skip_dirs:
                continue
            if has_sample and cname in {{"train", "test", "train_images", "test_images"}}:
                continue
            queue.append((child, depth + 1))
    print(STRATEGY, "song search visited", len(seen), "dirs; csv missing")
    return None



def discover_cheng_labels():
    """Find chengtingyi 1_text/labels4407_text.csv without walking DICOM trees."""
    names = {{"labels4407_text.csv"}}
    skip_dirs = {{
        "train", "test", "train_images", "test_images", "images",
        "dicom", "dcms", "series", "2_vision_full", "3_vision_gold_reads",
    }}
    explicit = [
        inp / "datasets" / "chengtingyi" / "rsna-knee-llm-weak-labels" / "1_text" / "labels4407_text.csv",
        inp / "rsna-knee-llm-weak-labels" / "1_text" / "labels4407_text.csv",
        inp / "chengtingyi" / "rsna-knee-llm-weak-labels" / "1_text" / "labels4407_text.csv",
        inp / "datasets" / "chengtingyi" / "rsna-knee-llm-weak-labels" / "labels4407_text.csv",
        inp / "rsna-knee-llm-weak-labels" / "labels4407_text.csv",
        ROOT / "labels4407_text.csv",
    ]
    for pth in explicit:
        if pth.is_file():
            print(STRATEGY, "cheng csv explicit", pth)
            return pth
    roots = []
    if inp.exists():
        roots.append(inp)
    roots.append(ROOT)
    parent = ROOT.parent
    if parent != ROOT:
        roots.append(parent)
    seen = set()
    queue = [(r, 0) for r in roots if r.exists() and r.is_dir()]
    while queue:
        base, depth = queue.pop(0)
        try:
            key = str(base.resolve())
        except OSError:
            key = str(base)
        if key in seen:
            continue
        seen.add(key)
        try:
            children = list(base.iterdir())
        except OSError:
            continue
        has_sample = any(c.is_file() and c.name == "sample_submission.csv" for c in children)
        for child in children:
            if child.is_file():
                cname = child.name.lower()
                path_l = str(child).lower()
                if cname in names and ("chengtingyi" in path_l or "rsna-knee-llm-weak-labels" in path_l or "1_text" in path_l or child.parent == ROOT):
                    print(STRATEGY, "cheng csv bfs", child, "depth", depth)
                    return child
            if not child.is_dir() or depth >= 6:
                continue
            cname = child.name.lower()
            if cname in skip_dirs:
                continue
            if has_sample and cname in {{"train", "test", "train_images", "test_images"}}:
                continue
            queue.append((child, depth + 1))
    print(STRATEGY, "cheng search visited", len(seen), "dirs; csv missing")
    return None


def learned_scores_llm(interact=False, lam=2.0, target_name="Baker's", finder=None):
    """Ridge logits on public train-only LLM Baker's labels. Never use test UIDs."""
    if train_series is None:
        return None, 0
    llm_path = (finder or discover_llm_labels)()
    if llm_path is None:
        print(STRATEGY, "llm csv not found")
        return None, 0
    llm = pd.read_csv(llm_path)
    if target_name not in llm.columns:
        for alt in (
            f"{{target_name}}__confidence",
            target_name.replace(" ", "_"),
            target_name.replace(" ", "_") + "__confidence",
        ):
            if alt in llm.columns:
                llm[target_name] = pd.to_numeric(llm[alt], errors="coerce")
                break
    if study_col not in llm.columns or target_name not in llm.columns:
        print(STRATEGY, "llm csv missing columns", list(llm.columns)[:8])
        return None, 0
    test_uids = set(uids)
    llm[study_col] = llm[study_col].astype(str)
    llm = llm[~llm[study_col].isin(test_uids)]
    tr_map, default_x = feat_map(train_series, interact=interact, drop_fat=False)
    te_map, _ = feat_map(test_series, interact=interact, drop_fat=False)
    X_rows, y_rows = [], []
    for _, row in llm.iterrows():
        sid = str(row[study_col])
        x = tr_map.get(sid)
        if x is None:
            continue
        try:
            yv = float(row[target_name])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(yv):
            continue
        X_rows.append(x)
        y_rows.append(float(np.clip(yv, EPS, 1 - EPS)))
    n_llm = len(X_rows)
    if n_llm < 20:
        return None, n_llm
    X = np.vstack(X_rows)
    y = np.array(y_rows, dtype=float)
    mu = X.mean(axis=0)
    sd = np.clip(X.std(axis=0), 1e-6, None)
    mu[0], sd[0] = 0.0, 1.0
    Xs = (X - mu) / sd
    Xte = np.vstack([(te_map.get(uid, default_x) - mu) / sd for uid in uids])
    scores = np.zeros((len(uids), len(targets)), dtype=float)
    for j, t in enumerate(targets):
        if t != target_name:
            scores[:, j] = logit(prev[t])
            continue
        if y.min() == y.max():
            scores[:, j] = logit(prev[t])
            continue
        w = fit_ridge_logit(Xs, y, lam=lam, steps=40)
        scores[:, j] = Xte @ w
    print(STRATEGY, "llm", target_name, "n", n_llm, "path", llm_path.name)
    return scores, n_llm

used_learned = False
n7 = nI = 0
WEAK_STRATS = {{"weak_rank_calibrate", "weak_rank_goldfill", "weak_rank_confident", "weak_rank_named"}}
BLEND_W7 = {{"gold_rank_interact": 0.60, "gold_rank_w50": 0.50, "gold_rank_w70": 0.70, "gold_rank_w40": 0.40, "gold_rank_lam2": 0.50, "weak_rank_calibrate": 0.50, "weak_rank_goldfill": 0.50, "weak_rank_confident": 0.50, "weak_rank_named": 0.50, "weak_rank_named_mix": 0.50, "weak_rank_bakers_mix": 0.50, "weak_rank_bakers_w60": 0.50, "weak_rank_bakers_w40": 0.50, "weak_rank_bakers_acl_mix": 0.50, "weak_rank_bakers_mm_mix": 0.50, "weak_rank_bakers_goldstd": 0.50, "weak_rank_bakers_silence": 0.50, "weak_rank_bakers_dropfat": 0.50, "weak_rank_bakers_llm": 0.50, "weak_rank_bakers_acl_llm": 0.50, "weak_rank_bakers_mcl_llm": 0.50, "weak_rank_bakers_eff_llm": 0.50, "weak_rank_bakers_syn_llm": 0.50, "weak_rank_bakers_syn_from_eff": 0.50, "weak_rank_bakers_cont_llm": 0.50, "weak_rank_bakers_frac_llm": 0.50, "weak_rank_bakers_moa_llm": 0.50, "weak_rank_bakers_loa_llm": 0.50, "weak_rank_bakers_pfoa_llm": 0.50, "weak_rank_bakers_frac_hyb_llm": 0.50, "weak_rank_bakers_frac_hyb_w60": 0.50, "weak_rank_bakers_frac_hyb_w40": 0.50, "weak_rank_bakers_frac_gem_llm": 0.50, "weak_rank_bakers_frac_hyb_gem_llm": 0.50, "weak_rank_bakers_frac_hyb_mm_llm": 0.50, "weak_rank_bakers_frac_hyb_lm_llm": 0.50, "weak_rank_bakers_frac_hyb_lm_w60": 0.50, "weak_rank_bakers_frac_hyb_lm_w40": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_w60": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_w40": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_w60": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_w40": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_acl_qwen": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_eff_qwen": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w60": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w40": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_qwen": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_qwen": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_front": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_front": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_front": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_front": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_front": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_pilk": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_pilk": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_pilk": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_pilk": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_2src": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_2src": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_2src": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_2src": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_sol": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_sol": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_sol": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_sol": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_3src": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_3src": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_3src": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_3src": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_song": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_song": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_song": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_song": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_cheng": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_cheng": 0.50, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_cheng": 0.50}}
INTERACT_LAM = {{"gold_rank_interact": 3.5, "gold_rank_w50": 3.5, "gold_rank_w70": 3.5, "gold_rank_w40": 3.5, "gold_rank_lam2": 2.0, "weak_rank_calibrate": 3.5, "weak_rank_goldfill": 3.5, "weak_rank_confident": 3.5, "weak_rank_named": 3.5, "weak_rank_named_mix": 3.5, "weak_rank_bakers_mix": 3.5, "weak_rank_bakers_w60": 3.5, "weak_rank_bakers_w40": 3.5, "weak_rank_bakers_acl_mix": 3.5, "weak_rank_bakers_mm_mix": 3.5, "weak_rank_bakers_goldstd": 3.5, "weak_rank_bakers_silence": 3.5, "weak_rank_bakers_dropfat": 3.5, "weak_rank_bakers_llm": 3.5, "weak_rank_bakers_acl_llm": 3.5, "weak_rank_bakers_mcl_llm": 3.5, "weak_rank_bakers_eff_llm": 3.5, "weak_rank_bakers_syn_llm": 3.5, "weak_rank_bakers_syn_from_eff": 3.5, "weak_rank_bakers_cont_llm": 3.5, "weak_rank_bakers_frac_llm": 3.5, "weak_rank_bakers_moa_llm": 3.5, "weak_rank_bakers_loa_llm": 3.5, "weak_rank_bakers_pfoa_llm": 3.5, "weak_rank_bakers_frac_hyb_llm": 3.5, "weak_rank_bakers_frac_hyb_w60": 3.5, "weak_rank_bakers_frac_hyb_w40": 3.5, "weak_rank_bakers_frac_gem_llm": 3.5, "weak_rank_bakers_frac_hyb_gem_llm": 3.5, "weak_rank_bakers_frac_hyb_mm_llm": 3.5, "weak_rank_bakers_frac_hyb_lm_llm": 3.5, "weak_rank_bakers_frac_hyb_lm_w60": 3.5, "weak_rank_bakers_frac_hyb_lm_w40": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_w60": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_w40": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_w60": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_w40": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_acl_qwen": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_eff_qwen": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w60": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w40": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_qwen": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_qwen": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_front": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_front": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_front": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_front": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_front": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_pilk": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_pilk": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_pilk": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_pilk": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_2src": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_2src": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_2src": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_2src": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_sol": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_sol": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_sol": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_sol": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_3src": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_3src": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_3src": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_3src": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_song": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_song": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_song": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_song": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_cheng": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_cheng": 3.5, "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_cheng": 3.5}}
LEARNED = {{"gold_meta_logit", "gold_rank_interact", "gold_rank_w50", "gold_rank_w70", "gold_rank_w40", "gold_rank_lam2", "weak_rank_named_mix", "weak_rank_bakers_mix", "weak_rank_bakers_w60", "weak_rank_bakers_w40", "weak_rank_bakers_acl_mix", "weak_rank_bakers_mm_mix", "weak_rank_bakers_goldstd", "weak_rank_bakers_silence", "weak_rank_bakers_dropfat", "weak_rank_bakers_llm", "weak_rank_bakers_acl_llm", "weak_rank_bakers_mcl_llm", "weak_rank_bakers_eff_llm", "weak_rank_bakers_syn_llm", "weak_rank_bakers_syn_from_eff", "weak_rank_bakers_cont_llm", "weak_rank_bakers_frac_llm", "weak_rank_bakers_moa_llm", "weak_rank_bakers_loa_llm", "weak_rank_bakers_pfoa_llm", "weak_rank_bakers_frac_hyb_llm", "weak_rank_bakers_frac_hyb_w60", "weak_rank_bakers_frac_hyb_w40", "weak_rank_bakers_frac_gem_llm", "weak_rank_bakers_frac_hyb_gem_llm", "weak_rank_bakers_frac_hyb_mm_llm", "weak_rank_bakers_frac_hyb_lm_llm", "weak_rank_bakers_frac_hyb_lm_w60", "weak_rank_bakers_frac_hyb_lm_w40", "weak_rank_bakers_frac_hyb_lm_eff_w60", "weak_rank_bakers_frac_hyb_lm_eff_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_acl_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_eff_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_cheng"}}
if STRATEGY in WEAK_STRATS and train_series is not None:
    w7 = float(BLEND_W7[STRATEGY])
    lamI = float(INTERACT_LAM[STRATEGY])
    prefer_gold = STRATEGY in {{"weak_rank_goldfill", "weak_rank_confident", "weak_rank_named"}}
    confident_only = STRATEGY in {{"weak_rank_confident", "weak_rank_named"}}
    named_only = STRATEGY == "weak_rank_named"
    scores7, n7 = learned_scores_weak(False, 2.0, prefer_gold=prefer_gold, confident_only=confident_only, named_only=named_only)
    scoresI, nI = learned_scores_weak(True, lamI, prefer_gold=prefer_gold, confident_only=confident_only, named_only=named_only)
    ranked = None
    if scores7 is not None and scoresI is not None:
        ranked = w7 * rank_cols(scores7) + (1.0 - w7) * rank_cols(scoresI)
        print(STRATEGY, "blend", w7, "*7d +", 1.0 - w7, "*plane-protocol", "lamI", lamI, "prefer_gold", prefer_gold, "confident_only", confident_only, "named_only", named_only, "n7", n7, "nI", nI)
    elif scores7 is not None:
        ranked = rank_cols(scores7)
        print(STRATEGY, "fallback to 7d weak", "prefer_gold", prefer_gold, "confident_only", confident_only, "named_only", named_only, "n7", n7, "nI", nI)
    if ranked is not None:
        out = sample[[study_col]].copy()
        for j, t in enumerate(targets):
            out[t] = np.clip(prev[t] + 0.25 * (ranked[:, j] - 0.5), EPS, 1 - EPS)
        out = out[sample.columns]
        used_learned = True
    else:
        print(STRATEGY, "skipped; falling back to gold_rank_w50. n7", n7, "nI", nI)

if (STRATEGY in LEARNED or (STRATEGY in WEAK_STRATS and not used_learned)) and train_series is not None and not used_learned:
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

if STRATEGY in MIX_TARGETS and used_learned and train_series is not None:
    # Keep frozen gold ranks for all 12; mix named-weak ranks onto MIX_TARGETS[STRATEGY] only.
    mix_set = MIX_TARGETS[STRATEGY]
    named_for_mix = MIX_NAMED_SET.get(STRATEGY, NAMED_PARSER)
    std_gold = STRATEGY == "weak_rank_bakers_goldstd"
    silence_neg = {{"Baker's"}} if STRATEGY == "weak_rank_bakers_silence" else None
    gold_arr = out[targets].to_numpy(dtype=float)
    if STRATEGY == "weak_rank_bakers_llm":
        scores7w, n7w = learned_scores_llm(False, 2.0)
        scoresIw, nIw = learned_scores_llm(True, 3.5)
        if scores7w is None:
            # 2026-10-05 kernel missed /kaggle/input/datasets/... and fell back to gold_rank_w50 (0.518).
            # Plan says missing CSV → parser Baker's mix (0.519), not the prior gold-only ranks.
            print(STRATEGY, "LLM missing; falling back to parser Baker's mix")
            scores7w, n7w = learned_scores_weak(False, 2.0, prefer_gold=True, confident_only=True, named_only=True, named_targets=NAMED_PARSER)
            scoresIw, nIw = learned_scores_weak(True, 3.5, prefer_gold=True, confident_only=True, named_only=True, named_targets=NAMED_PARSER)
    else:
        scores7w, n7w = learned_scores_weak(False, 2.0, prefer_gold=True, confident_only=True, named_only=True, named_targets=named_for_mix, std_on_gold=std_gold, silence_neg_targets=silence_neg)
        scoresIw, nIw = learned_scores_weak(True, 3.5, prefer_gold=True, confident_only=True, named_only=True, named_targets=named_for_mix, std_on_gold=std_gold, silence_neg_targets=silence_neg)
    ranked_w = None
    if scores7w is not None and scoresIw is not None:
        ranked_w = 0.5 * rank_cols(scores7w) + 0.5 * rank_cols(scoresIw)
        print(STRATEGY, "named-weak blend 0.50/0.50 n7w", n7w, "nIw", nIw)
    elif scores7w is not None:
        ranked_w = rank_cols(scores7w)
        print(STRATEGY, "named-weak 7d only n7w", n7w, "nIw", nIw)
    if ranked_w is not None:
        mixed = gold_arr.copy()
        w_gold = float(MIX_GOLD_W.get(STRATEGY, 0.50))
        for j, t in enumerate(targets):
            if t in mix_set:
                # gold_arr is already prevalence-mapped; convert named ranks the same way
                weak_col = np.clip(prev[t] + 0.25 * (ranked_w[:, j] - 0.5), EPS, 1 - EPS)
                mixed[:, j] = w_gold * gold_arr[:, j] + (1.0 - w_gold) * weak_col
        out = sample[[study_col]].copy()
        for j, t in enumerate(targets):
            out[t] = mixed[:, j]
        out = out[sample.columns]
        print(STRATEGY, "mixed named targets", sorted(mix_set), "w_gold", w_gold)
    else:
        print(STRATEGY, "skipped named mix; keeping gold_rank_w50. n7w", n7w, "nIw", nIw)

if STRATEGY in {"weak_rank_bakers_acl_llm", "weak_rank_bakers_mcl_llm", "weak_rank_bakers_eff_llm", "weak_rank_bakers_syn_llm", "weak_rank_bakers_syn_from_eff", "weak_rank_bakers_cont_llm", "weak_rank_bakers_frac_llm", "weak_rank_bakers_moa_llm", "weak_rank_bakers_loa_llm", "weak_rank_bakers_pfoa_llm", "weak_rank_bakers_frac_hyb_llm", "weak_rank_bakers_frac_hyb_w60", "weak_rank_bakers_frac_hyb_w40", "weak_rank_bakers_frac_gem_llm", "weak_rank_bakers_frac_hyb_gem_llm", "weak_rank_bakers_frac_hyb_mm_llm", "weak_rank_bakers_frac_hyb_lm_llm", "weak_rank_bakers_frac_hyb_lm_w60", "weak_rank_bakers_frac_hyb_lm_w40", "weak_rank_bakers_frac_hyb_lm_eff_w60", "weak_rank_bakers_frac_hyb_lm_eff_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_acl_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_eff_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_cheng"} and used_learned and train_series is not None:
    # Keep parser Baker's mix; add LLM ACL only if that head has public-test variance.
    # 2026-10-06 Baker's LLM head was constant (std~0) so mixing it was rank-preserving.
    scores7l, n7l = learned_scores_llm(False, 2.0, target_name="ACL")
    scoresIl, nIl = learned_scores_llm(True, 3.5, target_name="ACL")
    ranked_l = None
    if scores7l is not None and scoresIl is not None:
        ranked_l = 0.5 * rank_cols(scores7l) + 0.5 * rank_cols(scoresIl)
        print(STRATEGY, "llm ACL blend 0.50/0.50 n7l", n7l, "nIl", nIl)
    elif scores7l is not None:
        ranked_l = rank_cols(scores7l)
        print(STRATEGY, "llm ACL 7d only n7l", n7l, "nIl", nIl)
    if ranked_l is not None and "ACL" in targets:
        acl_j = targets.index("ACL")
        acl_std = float(np.std(ranked_l[:, acl_j]))
        if acl_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["ACL"] + 0.25 * (ranked_l[:, acl_j] - 0.5), EPS, 1 - EPS)
            w_gold = 0.40 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_w60" else 0.60 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_w40" else 0.50
            mixed[:, acl_j] = w_gold * mixed[:, acl_j] + (1.0 - w_gold) * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed LLM ACL std", acl_std, "w_gold", w_gold)
        else:
            print(STRATEGY, "skip constant LLM ACL std", acl_std, "; keeping parser Baker's mix")
    else:
        print(STRATEGY, "LLM ACL missing; keeping parser Baker's mix. n7l", n7l, "nIl", nIl)

if STRATEGY in {"weak_rank_bakers_frac_hyb_lm_eff_acl_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_acl_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_eff_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_cheng"} and used_learned and train_series is not None:
    # Keep the 0.531 Gemini ACL stack. Majority Gemini (0.40/0.60) tied 0.531 (57768062)
    # after compressing public ACL 0.336–0.476 → 0.339–0.464. gem_w40 gives minority Gemini.
    scores7ga, n7ga = learned_scores_llm(False, 2.0, target_name="ACL", finder=discover_gemini_labels)
    scoresIga, nIga = learned_scores_llm(True, 3.5, target_name="ACL", finder=discover_gemini_labels)
    ranked_ga = None
    if scores7ga is not None and scoresIga is not None:
        ranked_ga = 0.5 * rank_cols(scores7ga) + 0.5 * rank_cols(scoresIga)
        print(STRATEGY, "gemini ACL blend 0.50/0.50 n7ga", n7ga, "nIga", nIga)
    elif scores7ga is not None:
        ranked_ga = rank_cols(scores7ga)
        print(STRATEGY, "gemini ACL 7d only n7ga", n7ga, "nIga", nIga)
    if ranked_ga is not None and "ACL" in targets:
        acl_j = targets.index("ACL")
        acl_std = float(np.std(ranked_ga[:, acl_j]))
        if acl_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["ACL"] + 0.25 * (ranked_ga[:, acl_j] - 0.5), EPS, 1 - EPS)
            w_gold = 0.40 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60" else 0.60 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40" else 0.50
            mixed[:, acl_j] = w_gold * mixed[:, acl_j] + (1.0 - w_gold) * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed Gemini ACL std", acl_std, "w_gold", w_gold)
        else:
            print(STRATEGY, "skip constant Gemini ACL std", acl_std, "; keeping 0.531 stack")
    else:
        print(STRATEGY, "Gemini ACL missing; keeping 0.531 stack. n7ga", n7ga, "nIga", nIga)

if STRATEGY == "weak_rank_bakers_mcl_llm" and used_learned and train_series is not None:
    # Keep parser Baker's + LLM ACL (0.520); add LLM MCL only if that head has variance.
    # Parser MCL poisoned named_mix (0.514); this uses the public LLM key instead.
    scores7m, n7m = learned_scores_llm(False, 2.0, target_name="MCL")
    scoresIm, nIm = learned_scores_llm(True, 3.5, target_name="MCL")
    ranked_m = None
    if scores7m is not None and scoresIm is not None:
        ranked_m = 0.5 * rank_cols(scores7m) + 0.5 * rank_cols(scoresIm)
        print(STRATEGY, "llm MCL blend 0.50/0.50 n7m", n7m, "nIm", nIm)
    elif scores7m is not None:
        ranked_m = rank_cols(scores7m)
        print(STRATEGY, "llm MCL 7d only n7m", n7m, "nIm", nIm)
    if ranked_m is not None and "MCL" in targets:
        mcl_j = targets.index("MCL")
        mcl_std = float(np.std(ranked_m[:, mcl_j]))
        if mcl_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["MCL"] + 0.25 * (ranked_m[:, mcl_j] - 0.5), EPS, 1 - EPS)
            mixed[:, mcl_j] = 0.50 * mixed[:, mcl_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed LLM MCL std", mcl_std)
        else:
            print(STRATEGY, "skip constant LLM MCL std", mcl_std, "; keeping bakers_acl_llm stack")
    else:
        print(STRATEGY, "LLM MCL missing; keeping bakers_acl_llm stack. n7m", n7m, "nIm", nIm)

if STRATEGY in {"weak_rank_bakers_eff_llm", "weak_rank_bakers_syn_llm", "weak_rank_bakers_syn_from_eff", "weak_rank_bakers_cont_llm", "weak_rank_bakers_frac_llm", "weak_rank_bakers_moa_llm", "weak_rank_bakers_loa_llm", "weak_rank_bakers_pfoa_llm", "weak_rank_bakers_frac_hyb_llm", "weak_rank_bakers_frac_hyb_w60", "weak_rank_bakers_frac_hyb_w40", "weak_rank_bakers_frac_gem_llm", "weak_rank_bakers_frac_hyb_gem_llm", "weak_rank_bakers_frac_hyb_mm_llm", "weak_rank_bakers_frac_hyb_lm_llm", "weak_rank_bakers_frac_hyb_lm_w60", "weak_rank_bakers_frac_hyb_lm_w40", "weak_rank_bakers_frac_hyb_lm_eff_w60", "weak_rank_bakers_frac_hyb_lm_eff_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_acl_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_eff_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_cheng"} and used_learned and train_series is not None:
    # Keep parser Baker's + LLM ACL (0.520); add LLM Effusion only if that head has variance.
    # Accepted at 0.524 (56995081). syn_llm reuses this stack before adding Synovitis.
    scores7e, n7e = learned_scores_llm(False, 2.0, target_name="Effusion")
    scoresIe, nIe = learned_scores_llm(True, 3.5, target_name="Effusion")
    ranked_e = None
    if scores7e is not None and scoresIe is not None:
        ranked_e = 0.5 * rank_cols(scores7e) + 0.5 * rank_cols(scoresIe)
        print(STRATEGY, "llm Effusion blend 0.50/0.50 n7e", n7e, "nIe", nIe)
    elif scores7e is not None:
        ranked_e = rank_cols(scores7e)
        print(STRATEGY, "llm Effusion 7d only n7e", n7e, "nIe", nIe)
    if ranked_e is not None and "Effusion" in targets:
        eff_j = targets.index("Effusion")
        eff_std = float(np.std(ranked_e[:, eff_j]))
        if eff_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Effusion"] + 0.25 * (ranked_e[:, eff_j] - 0.5), EPS, 1 - EPS)
            w_gold = 0.40 if STRATEGY in {{"weak_rank_bakers_frac_hyb_lm_eff_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_acl_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_eff_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_cheng"}} else 0.60 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_w40" else 0.50
            mixed[:, eff_j] = w_gold * mixed[:, eff_j] + (1.0 - w_gold) * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed LLM Effusion std", eff_std, "w_gold", w_gold)
        else:
            print(STRATEGY, "skip constant LLM Effusion std", eff_std, "; keeping bakers_acl_llm stack")
    else:
        print(STRATEGY, "LLM Effusion missing; keeping bakers_acl_llm stack. n7e", n7e, "nIe", nIe)

if STRATEGY == "weak_rank_bakers_syn_llm" and used_learned and train_series is not None:
    # Keep parser Baker's + LLM ACL + LLM Effusion (0.524); add LLM Synovitis if variance.
    # v4_blend MCL poisoned (0.510); effusion transferred (+0.004). Synovitis is the other fluid head.
    scores7s, n7s = learned_scores_llm(False, 2.0, target_name="Synovitis")
    scoresIs, nIs = learned_scores_llm(True, 3.5, target_name="Synovitis")
    ranked_s = None
    if scores7s is not None and scoresIs is not None:
        ranked_s = 0.5 * rank_cols(scores7s) + 0.5 * rank_cols(scoresIs)
        print(STRATEGY, "llm Synovitis blend 0.50/0.50 n7s", n7s, "nIs", nIs)
    elif scores7s is not None:
        ranked_s = rank_cols(scores7s)
        print(STRATEGY, "llm Synovitis 7d only n7s", n7s, "nIs", nIs)
    if ranked_s is not None and "Synovitis" in targets:
        syn_j = targets.index("Synovitis")
        syn_std = float(np.std(ranked_s[:, syn_j]))
        if syn_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Synovitis"] + 0.25 * (ranked_s[:, syn_j] - 0.5), EPS, 1 - EPS)
            mixed[:, syn_j] = 0.50 * mixed[:, syn_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed LLM Synovitis std", syn_std)
        else:
            print(STRATEGY, "skip constant LLM Synovitis std", syn_std, "; keeping bakers_eff_llm stack")
    else:
        print(STRATEGY, "LLM Synovitis missing; keeping bakers_eff_llm stack. n7s", n7s, "nIs", nIs)

if STRATEGY == "weak_rank_bakers_syn_from_eff" and used_learned and train_series is not None:
    # Keep parser Baker's + LLM ACL + LLM Effusion (0.524).
    # Do NOT use the Synovitis LLM field (57032978 flattened public Synovitis to 0.465517).
    # Mix LLM Effusion ranks onto Synovitis (effusion vs gold synovitis 0.7115 > synovitis field 0.678).
    scores7p, n7p = learned_scores_llm(False, 2.0, target_name="Effusion")
    scoresIp, nIp = learned_scores_llm(True, 3.5, target_name="Effusion")
    ranked_p = None
    if scores7p is not None and scoresIp is not None:
        ranked_p = 0.5 * rank_cols(scores7p) + 0.5 * rank_cols(scoresIp)
        print(STRATEGY, "llm Effusion→Synovitis blend 0.50/0.50 n7p", n7p, "nIp", nIp)
    elif scores7p is not None:
        ranked_p = rank_cols(scores7p)
        print(STRATEGY, "llm Effusion→Synovitis 7d only n7p", n7p, "nIp", nIp)
    if ranked_p is not None and "Synovitis" in targets and "Effusion" in targets:
        syn_j = targets.index("Synovitis")
        eff_j = targets.index("Effusion")
        weak_col = np.clip(prev["Synovitis"] + 0.25 * (ranked_p[:, eff_j] - 0.5), EPS, 1 - EPS)
        mixed = out[targets].to_numpy(dtype=float)
        candidate = 0.50 * mixed[:, syn_j] + 0.50 * weak_col
        mix_std = float(np.std(candidate))
        if mix_std > 1e-8:
            mixed[:, syn_j] = candidate
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed effusion ranks onto Synovitis std", mix_std)
        else:
            print(STRATEGY, "skip flatten effusion→Synovitis std", mix_std, "; keeping bakers_eff_llm stack")
    else:
        print(STRATEGY, "LLM Effusion missing for Synovitis proxy; keeping bakers_eff_llm stack. n7p", n7p, "nIp", nIp)

if STRATEGY == "weak_rank_bakers_cont_llm" and used_learned and train_series is not None:
    # Keep parser Baker's + LLM ACL + LLM Effusion (0.524); add LLM Contusion if variance.
    # Synovitis remaps are exhausted (field 0.522, effusion-proxy 0.521).
    scores7c, n7c = learned_scores_llm(False, 2.0, target_name="Contusion")
    scoresIc, nIc = learned_scores_llm(True, 3.5, target_name="Contusion")
    ranked_c = None
    if scores7c is not None and scoresIc is not None:
        ranked_c = 0.5 * rank_cols(scores7c) + 0.5 * rank_cols(scoresIc)
        print(STRATEGY, "llm Contusion blend 0.50/0.50 n7c", n7c, "nIc", nIc)
    elif scores7c is not None:
        ranked_c = rank_cols(scores7c)
        print(STRATEGY, "llm Contusion 7d only n7c", n7c, "nIc", nIc)
    if ranked_c is not None and "Contusion" in targets:
        con_j = targets.index("Contusion")
        con_std = float(np.std(ranked_c[:, con_j]))
        if con_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Contusion"] + 0.25 * (ranked_c[:, con_j] - 0.5), EPS, 1 - EPS)
            mixed[:, con_j] = 0.50 * mixed[:, con_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed LLM Contusion std", con_std)
        else:
            print(STRATEGY, "skip constant LLM Contusion std", con_std, "; keeping bakers_eff_llm stack")
    else:
        print(STRATEGY, "LLM Contusion missing; keeping bakers_eff_llm stack. n7c", n7c, "nIc", nIc)

if STRATEGY == "weak_rank_bakers_frac_llm" and used_learned and train_series is not None:
    # Keep parser Baker's + LLM ACL + LLM Effusion (0.524); add LLM Fracture if variance.
    # Contusion remaps are exhausted (0.519 compress).
    scores7f, n7f = learned_scores_llm(False, 2.0, target_name="Fracture")
    scoresIf, nIf = learned_scores_llm(True, 3.5, target_name="Fracture")
    ranked_f = None
    if scores7f is not None and scoresIf is not None:
        ranked_f = 0.5 * rank_cols(scores7f) + 0.5 * rank_cols(scoresIf)
        print(STRATEGY, "llm Fracture blend 0.50/0.50 n7f", n7f, "nIf", nIf)
    elif scores7f is not None:
        ranked_f = rank_cols(scores7f)
        print(STRATEGY, "llm Fracture 7d only n7f", n7f, "nIf", nIf)
    if ranked_f is not None and "Fracture" in targets:
        frac_j = targets.index("Fracture")
        frac_std = float(np.std(ranked_f[:, frac_j]))
        if frac_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Fracture"] + 0.25 * (ranked_f[:, frac_j] - 0.5), EPS, 1 - EPS)
            mixed[:, frac_j] = 0.50 * mixed[:, frac_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed LLM Fracture std", frac_std)
        else:
            print(STRATEGY, "skip constant LLM Fracture std", frac_std, "; keeping bakers_eff_llm stack")
    else:
        print(STRATEGY, "LLM Fracture missing; keeping bakers_eff_llm stack. n7f", n7f, "nIf", nIf)

if STRATEGY == "weak_rank_bakers_moa_llm" and used_learned and train_series is not None:
    # Keep parser Baker's + LLM ACL + LLM Effusion (0.524); add LLM Medial OA if variance.
    # v4 Fracture tied 0.524 (57132860). Medial OA is the strongest unused v4 head (0.931 vs gold).
    scores7o, n7o = learned_scores_llm(False, 2.0, target_name="Medial OA")
    scoresIo, nIo = learned_scores_llm(True, 3.5, target_name="Medial OA")
    ranked_o = None
    if scores7o is not None and scoresIo is not None:
        ranked_o = 0.5 * rank_cols(scores7o) + 0.5 * rank_cols(scoresIo)
        print(STRATEGY, "llm Medial OA blend 0.50/0.50 n7o", n7o, "nIo", nIo)
    elif scores7o is not None:
        ranked_o = rank_cols(scores7o)
        print(STRATEGY, "llm Medial OA 7d only n7o", n7o, "nIo", nIo)
    if ranked_o is not None and "Medial OA" in targets:
        moa_j = targets.index("Medial OA")
        moa_std = float(np.std(ranked_o[:, moa_j]))
        if moa_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Medial OA"] + 0.25 * (ranked_o[:, moa_j] - 0.5), EPS, 1 - EPS)
            mixed[:, moa_j] = 0.50 * mixed[:, moa_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed LLM Medial OA std", moa_std)
        else:
            print(STRATEGY, "skip constant LLM Medial OA std", moa_std, "; keeping bakers_eff_llm stack")
    else:
        print(STRATEGY, "LLM Medial OA missing; keeping bakers_eff_llm stack. n7o", n7o, "nIo", nIo)

if STRATEGY == "weak_rank_bakers_loa_llm" and used_learned and train_series is not None:
    # Keep parser Baker's + LLM ACL + LLM Effusion (0.524); add LLM Lateral OA if variance.
    # v4 Medial OA poisoned 0.523 (57169674). Lateral OA is the remaining unused compartment OA head.
    scores7a, n7a = learned_scores_llm(False, 2.0, target_name="Lateral OA")
    scoresIa, nIa = learned_scores_llm(True, 3.5, target_name="Lateral OA")
    ranked_a = None
    if scores7a is not None and scoresIa is not None:
        ranked_a = 0.5 * rank_cols(scores7a) + 0.5 * rank_cols(scoresIa)
        print(STRATEGY, "llm Lateral OA blend 0.50/0.50 n7a", n7a, "nIa", nIa)
    elif scores7a is not None:
        ranked_a = rank_cols(scores7a)
        print(STRATEGY, "llm Lateral OA 7d only n7a", n7a, "nIa", nIa)
    if ranked_a is not None and "Lateral OA" in targets:
        loa_j = targets.index("Lateral OA")
        loa_std = float(np.std(ranked_a[:, loa_j]))
        if loa_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Lateral OA"] + 0.25 * (ranked_a[:, loa_j] - 0.5), EPS, 1 - EPS)
            mixed[:, loa_j] = 0.50 * mixed[:, loa_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed LLM Lateral OA std", loa_std)
        else:
            print(STRATEGY, "skip constant LLM Lateral OA std", loa_std, "; keeping bakers_eff_llm stack")
    else:
        print(STRATEGY, "LLM Lateral OA missing; keeping bakers_eff_llm stack. n7a", n7a, "nIa", nIa)

if STRATEGY == "weak_rank_bakers_pfoa_llm" and used_learned and train_series is not None:
    # Keep parser Baker's + LLM ACL + LLM Effusion (0.524); add LLM PF OA if variance.
    # v4 Lateral OA poisoned 0.523 (57218864). PF OA is the remaining unused OA head.
    scores7p, n7pf = learned_scores_llm(False, 2.0, target_name="PF OA")
    scoresIp, nIpf = learned_scores_llm(True, 3.5, target_name="PF OA")
    ranked_pf = None
    if scores7p is not None and scoresIp is not None:
        ranked_pf = 0.5 * rank_cols(scores7p) + 0.5 * rank_cols(scoresIp)
        print(STRATEGY, "llm PF OA blend 0.50/0.50 n7p", n7pf, "nIp", nIpf)
    elif scores7p is not None:
        ranked_pf = rank_cols(scores7p)
        print(STRATEGY, "llm PF OA 7d only n7p", n7pf, "nIp", nIpf)
    if ranked_pf is not None and "PF OA" in targets:
        pfoa_j = targets.index("PF OA")
        pfoa_std = float(np.std(ranked_pf[:, pfoa_j]))
        if pfoa_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["PF OA"] + 0.25 * (ranked_pf[:, pfoa_j] - 0.5), EPS, 1 - EPS)
            mixed[:, pfoa_j] = 0.50 * mixed[:, pfoa_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed LLM PF OA std", pfoa_std)
        else:
            print(STRATEGY, "skip constant LLM PF OA std", pfoa_std, "; keeping bakers_eff_llm stack")
    else:
        print(STRATEGY, "LLM PF OA missing; keeping bakers_eff_llm stack. n7p", n7pf, "nIp", nIpf)

if STRATEGY in {"weak_rank_bakers_frac_hyb_llm", "weak_rank_bakers_frac_hyb_w60", "weak_rank_bakers_frac_hyb_w40", "weak_rank_bakers_frac_hyb_gem_llm", "weak_rank_bakers_frac_hyb_mm_llm", "weak_rank_bakers_frac_hyb_lm_llm", "weak_rank_bakers_frac_hyb_lm_w60", "weak_rank_bakers_frac_hyb_lm_w40", "weak_rank_bakers_frac_hyb_lm_eff_w60", "weak_rank_bakers_frac_hyb_lm_eff_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_acl_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_eff_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_cheng"} and used_learned and train_series is not None:
    # Keep parser Baker's + v4 LLM ACL + v4 LLM Effusion; add hybrid Fracture if variance.
    # frac_hyb_llm accepted 0.527 (57298558). w60 gives hybrid Fracture majority weight.
    scores7h, n7h = learned_scores_llm(False, 2.0, target_name="Fracture", finder=discover_hybrid_labels)
    scoresIh, nIh = learned_scores_llm(True, 3.5, target_name="Fracture", finder=discover_hybrid_labels)
    ranked_h = None
    if scores7h is not None and scoresIh is not None:
        ranked_h = 0.5 * rank_cols(scores7h) + 0.5 * rank_cols(scoresIh)
        print(STRATEGY, "hybrid Fracture blend 0.50/0.50 n7h", n7h, "nIh", nIh)
    elif scores7h is not None:
        ranked_h = rank_cols(scores7h)
        print(STRATEGY, "hybrid Fracture 7d only n7h", n7h, "nIh", nIh)
    if ranked_h is not None and "Fracture" in targets:
        frac_j = targets.index("Fracture")
        frac_std = float(np.std(ranked_h[:, frac_j]))
        w_gold = 0.40 if STRATEGY == "weak_rank_bakers_frac_hyb_w60" else 0.60 if STRATEGY == "weak_rank_bakers_frac_hyb_w40" else 0.50
        if frac_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Fracture"] + 0.25 * (ranked_h[:, frac_j] - 0.5), EPS, 1 - EPS)
            mixed[:, frac_j] = w_gold * mixed[:, frac_j] + (1.0 - w_gold) * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed hybrid Fracture std", frac_std, "w_gold", w_gold)
        else:
            print(STRATEGY, "skip constant hybrid Fracture std", frac_std, "; keeping bakers_frac_hyb_llm stack")
    else:
        print(STRATEGY, "hybrid Fracture missing; keeping bakers_frac_hyb_llm stack. n7h", n7h, "nIh", nIh)

if STRATEGY == "weak_rank_bakers_frac_hyb_mm_llm" and used_learned and train_series is not None:
    # Keep the 0.527 hybrid stack; add v4 LLM Medial Meniscus if variance.
    # Parser MM poisoned 0.517 (56745095). Fracture Gemini stacking tied 0.527 (57468780).
    scores7mm, n7mm = learned_scores_llm(False, 2.0, target_name="Medial Meniscus")
    scoresImm, nImm = learned_scores_llm(True, 3.5, target_name="Medial Meniscus")
    ranked_mm = None
    if scores7mm is not None and scoresImm is not None:
        ranked_mm = 0.5 * rank_cols(scores7mm) + 0.5 * rank_cols(scoresImm)
        print(STRATEGY, "llm Medial Meniscus blend 0.50/0.50 n7mm", n7mm, "nImm", nImm)
    elif scores7mm is not None:
        ranked_mm = rank_cols(scores7mm)
        print(STRATEGY, "llm Medial Meniscus 7d only n7mm", n7mm, "nImm", nImm)
    if ranked_mm is not None and "Medial Meniscus" in targets:
        mm_j = targets.index("Medial Meniscus")
        mm_std = float(np.std(ranked_mm[:, mm_j]))
        if mm_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Medial Meniscus"] + 0.25 * (ranked_mm[:, mm_j] - 0.5), EPS, 1 - EPS)
            mixed[:, mm_j] = 0.50 * mixed[:, mm_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed LLM Medial Meniscus std", mm_std)
        else:
            print(STRATEGY, "skip constant LLM Medial Meniscus std", mm_std, "; keeping bakers_frac_hyb_llm stack")
    else:
        print(STRATEGY, "LLM Medial Meniscus missing; keeping bakers_frac_hyb_llm stack. n7mm", n7mm, "nImm", nImm)

if STRATEGY in {"weak_rank_bakers_frac_hyb_lm_llm", "weak_rank_bakers_frac_hyb_lm_w60", "weak_rank_bakers_frac_hyb_lm_w40", "weak_rank_bakers_frac_hyb_lm_eff_w60", "weak_rank_bakers_frac_hyb_lm_eff_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_acl_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_eff_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_cheng"} and used_learned and train_series is not None:
    # Keep the 0.529 hybrid+LM stack; lm_llm accepted 0.529 (57551203). w60 majority tied 0.529 (57584615).
    # w40 is the remaining mix-weight. Parser MM poisoned 0.517; LLM MM tied 0.527 (57512380).
    scores7lm, n7lm = learned_scores_llm(False, 2.0, target_name="Lateral Meniscus")
    scoresIlm, nIlm = learned_scores_llm(True, 3.5, target_name="Lateral Meniscus")
    ranked_lm = None
    if scores7lm is not None and scoresIlm is not None:
        ranked_lm = 0.5 * rank_cols(scores7lm) + 0.5 * rank_cols(scoresIlm)
        print(STRATEGY, "llm Lateral Meniscus blend 0.50/0.50 n7lm", n7lm, "nIlm", nIlm)
    elif scores7lm is not None:
        ranked_lm = rank_cols(scores7lm)
        print(STRATEGY, "llm Lateral Meniscus 7d only n7lm", n7lm, "nIlm", nIlm)
    if ranked_lm is not None and "Lateral Meniscus" in targets:
        lm_j = targets.index("Lateral Meniscus")
        lm_std = float(np.std(ranked_lm[:, lm_j]))
        w_gold = 0.40 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_w60" else 0.60 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_w40" else 0.50
        if lm_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Lateral Meniscus"] + 0.25 * (ranked_lm[:, lm_j] - 0.5), EPS, 1 - EPS)
            mixed[:, lm_j] = w_gold * mixed[:, lm_j] + (1.0 - w_gold) * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed LLM Lateral Meniscus std", lm_std, "w_gold", w_gold)
        else:
            print(STRATEGY, "skip constant LLM Lateral Meniscus std", lm_std, "; keeping bakers_frac_hyb_lm_llm stack")
    else:
        print(STRATEGY, "LLM Lateral Meniscus missing; keeping bakers_frac_hyb_lm_llm stack. n7lm", n7lm, "nIlm", nIlm)

if STRATEGY in {{"weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_acl_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_eff_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_cheng"}} and used_learned and train_series is not None:
    # Keep the 0.533 majority Gemini Effusion stack (57800238). w40 gives Gemini Effusion minority weight.
    # Gemini Effusion 50/50 transferred (+0.001) after widening public Effusion 0.478–0.703 → 0.478–0.716.
    scores7ge, n7ge = learned_scores_llm(False, 2.0, target_name="Effusion", finder=discover_gemini_labels)
    scoresIge, nIge = learned_scores_llm(True, 3.5, target_name="Effusion", finder=discover_gemini_labels)
    ranked_ge = None
    if scores7ge is not None and scoresIge is not None:
        ranked_ge = 0.5 * rank_cols(scores7ge) + 0.5 * rank_cols(scoresIge)
        print(STRATEGY, "gemini Effusion blend 0.50/0.50 n7ge", n7ge, "nIge", nIge)
    elif scores7ge is not None:
        ranked_ge = rank_cols(scores7ge)
        print(STRATEGY, "gemini Effusion 7d only n7ge", n7ge, "nIge", nIge)
    if ranked_ge is not None and "Effusion" in targets:
        eff_j = targets.index("Effusion")
        eff_std = float(np.std(ranked_ge[:, eff_j]))
        w_gold = 0.40 if STRATEGY in {{"weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_acl_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_eff_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_cheng"}} else 0.60 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40" else 0.50
        if eff_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Effusion"] + 0.25 * (ranked_ge[:, eff_j] - 0.5), EPS, 1 - EPS)
            mixed[:, eff_j] = w_gold * mixed[:, eff_j] + (1.0 - w_gold) * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed Gemini Effusion std", eff_std, "w_gold", w_gold)
        else:
            print(STRATEGY, "skip constant Gemini Effusion std", eff_std, "; keeping 0.532 stack")
    else:
        print(STRATEGY, "Gemini Effusion missing; keeping 0.532 stack. n7ge", n7ge, "nIge", nIge)


if STRATEGY in {"weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_acl_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_eff_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_cheng"} and used_learned and train_series is not None:
    # Keep the 0.534 Gemini LM 50/50 stack. Mix-weights of Gemini Effusion
    # plateaued (50/50=0.532, w60=0.533, w40=0.532 poison). Gemini LM 50/50
    # transferred (+0.001) after compressing public LM 0.303–0.522 → 0.287–0.490.
    scores7glm, n7glm = learned_scores_llm(False, 2.0, target_name="Lateral Meniscus", finder=discover_gemini_labels)
    scoresIglm, nIglm = learned_scores_llm(True, 3.5, target_name="Lateral Meniscus", finder=discover_gemini_labels)
    ranked_glm = None
    if scores7glm is not None and scoresIglm is not None:
        ranked_glm = 0.5 * rank_cols(scores7glm) + 0.5 * rank_cols(scoresIglm)
        print(STRATEGY, "gemini Lateral Meniscus blend 0.50/0.50 n7glm", n7glm, "nIglm", nIglm)
    elif scores7glm is not None:
        ranked_glm = rank_cols(scores7glm)
        print(STRATEGY, "gemini Lateral Meniscus 7d only n7glm", n7glm, "nIglm", nIglm)
    if ranked_glm is not None and "Lateral Meniscus" in targets:
        lm_j = targets.index("Lateral Meniscus")
        lm_std = float(np.std(ranked_glm[:, lm_j]))
        w_gold = 0.40 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60" else 0.60 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40" else 0.50
        if lm_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Lateral Meniscus"] + 0.25 * (ranked_glm[:, lm_j] - 0.5), EPS, 1 - EPS)
            mixed[:, lm_j] = w_gold * mixed[:, lm_j] + (1.0 - w_gold) * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed Gemini Lateral Meniscus std", lm_std, "w_gold", w_gold)
        else:
            print(STRATEGY, "skip constant Gemini Lateral Meniscus std", lm_std, "; keeping 0.534 stack")
    else:
        print(STRATEGY, "Gemini Lateral Meniscus missing; keeping 0.534 stack. n7glm", n7glm, "nIglm", nIglm)

if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem" and used_learned and train_series is not None:
    # Keep the 0.534 Gemini LM 50/50 stack. Gemini LM mix-weights plateaued
    # (50/50=0.534, w60=0.534 tie, w40=0.534 tie). v4 MM tied 0.527, so Gemini MM
    # is the remaining unused Gemini named-object column.
    scores7gmm, n7gmm = learned_scores_llm(False, 2.0, target_name="Medial Meniscus", finder=discover_gemini_labels)
    scoresIgmm, nIgmm = learned_scores_llm(True, 3.5, target_name="Medial Meniscus", finder=discover_gemini_labels)
    ranked_gmm = None
    if scores7gmm is not None and scoresIgmm is not None:
        ranked_gmm = 0.5 * rank_cols(scores7gmm) + 0.5 * rank_cols(scoresIgmm)
        print(STRATEGY, "gemini Medial Meniscus blend 0.50/0.50 n7gmm", n7gmm, "nIgmm", nIgmm)
    elif scores7gmm is not None:
        ranked_gmm = rank_cols(scores7gmm)
        print(STRATEGY, "gemini Medial Meniscus 7d only n7gmm", n7gmm, "nIgmm", nIgmm)
    if ranked_gmm is not None and "Medial Meniscus" in targets:
        mm_j = targets.index("Medial Meniscus")
        mm_std = float(np.std(ranked_gmm[:, mm_j]))
        if mm_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Medial Meniscus"] + 0.25 * (ranked_gmm[:, mm_j] - 0.5), EPS, 1 - EPS)
            mixed[:, mm_j] = 0.50 * mixed[:, mm_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed Gemini Medial Meniscus std", mm_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant Gemini Medial Meniscus std", mm_std, "; keeping 0.534 stack")
    else:
        print(STRATEGY, "Gemini Medial Meniscus missing; keeping 0.534 stack. n7gmm", n7gmm, "nIgmm", nIgmm)

if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem" and used_learned and train_series is not None:
    # Keep the 0.534 Gemini LM 50/50 stack. Gemini MM 50/50 tied 0.534 (57833621)
    # after mixing std=0.354; public MM stayed 0.323–0.511. v4 Baker's LLM was
    # constant on the public 3-study sample; Gemini Baker's is the remaining
    # unused Gemini named-object that is not a documented poison.
    scores7gb, n7gb = learned_scores_llm(False, 2.0, target_name="Baker's", finder=discover_gemini_labels)
    scoresIgb, nIgb = learned_scores_llm(True, 3.5, target_name="Baker's", finder=discover_gemini_labels)
    ranked_gb = None
    if scores7gb is not None and scoresIgb is not None:
        ranked_gb = 0.5 * rank_cols(scores7gb) + 0.5 * rank_cols(scoresIgb)
        print(STRATEGY, "gemini Baker's blend 0.50/0.50 n7gb", n7gb, "nIgb", nIgb)
    elif scores7gb is not None:
        ranked_gb = rank_cols(scores7gb)
        print(STRATEGY, "gemini Baker's 7d only n7gb", n7gb, "nIgb", nIgb)
    if ranked_gb is not None and "Baker's" in targets:
        bak_j = targets.index("Baker's")
        bak_std = float(np.std(ranked_gb[:, bak_j]))
        if bak_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Baker's"] + 0.25 * (ranked_gb[:, bak_j] - 0.5), EPS, 1 - EPS)
            mixed[:, bak_j] = 0.50 * mixed[:, bak_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed Gemini Baker's std", bak_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant Gemini Baker's std", bak_std, "; keeping 0.534 stack")
    else:
        print(STRATEGY, "Gemini Baker's missing; keeping 0.534 stack. n7gb", n7gb, "nIgb", nIgb)

if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_acl_qwen" and used_learned and train_series is not None:
    # Keep the 0.534 Gemini LM 50/50 stack. Gemini Baker's 50/50 poisoned
    # to 0.531 (57835922) after mixing std=0.408 and widening public Baker's
    # 0.206897 → 0.144–0.269. Do not remix Baker's. Qwen2.5-32B v2 ACL is
    # 0.981 vs gold and is a new source on a transferring named object.
    scores7q, n7q = learned_scores_llm(False, 2.0, target_name="ACL", finder=discover_qwen_labels)
    scoresIq, nIq = learned_scores_llm(True, 3.5, target_name="ACL", finder=discover_qwen_labels)
    ranked_q = None
    if scores7q is not None and scoresIq is not None:
        ranked_q = 0.5 * rank_cols(scores7q) + 0.5 * rank_cols(scoresIq)
        print(STRATEGY, "qwen ACL blend 0.50/0.50 n7q", n7q, "nIq", nIq)
    elif scores7q is not None:
        ranked_q = rank_cols(scores7q)
        print(STRATEGY, "qwen ACL 7d only n7q", n7q, "nIq", nIq)
    if ranked_q is not None and "ACL" in targets:
        acl_j = targets.index("ACL")
        acl_std = float(np.std(ranked_q[:, acl_j]))
        if acl_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["ACL"] + 0.25 * (ranked_q[:, acl_j] - 0.5), EPS, 1 - EPS)
            mixed[:, acl_j] = 0.50 * mixed[:, acl_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed Qwen ACL std", acl_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant Qwen ACL std", acl_std, "; keeping 0.534 stack")
    else:
        print(STRATEGY, "Qwen ACL missing; keeping 0.534 stack. n7q", n7q, "nIq", nIq)

if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_eff_qwen" and used_learned and train_series is not None:
    # Keep the 0.534 Gemini LM 50/50 stack. Qwen ACL 50/50 tied 0.534
    # (57837907) after mixing std=0.408 and widening public ACL
    # 0.336–0.476 → 0.312–0.508. Do not remix Qwen ACL weights.
    # Effusion is the remaining unused Qwen column on a transferring
    # named object (v4 0.524/0.530, Gemini 0.532/0.533).
    scores7qe, n7qe = learned_scores_llm(False, 2.0, target_name="Effusion", finder=discover_qwen_labels)
    scoresIqe, nIqe = learned_scores_llm(True, 3.5, target_name="Effusion", finder=discover_qwen_labels)
    ranked_qe = None
    if scores7qe is not None and scoresIqe is not None:
        ranked_qe = 0.5 * rank_cols(scores7qe) + 0.5 * rank_cols(scoresIqe)
        print(STRATEGY, "qwen Effusion blend 0.50/0.50 n7qe", n7qe, "nIqe", nIqe)
    elif scores7qe is not None:
        ranked_qe = rank_cols(scores7qe)
        print(STRATEGY, "qwen Effusion 7d only n7qe", n7qe, "nIqe", nIqe)
    if ranked_qe is not None and "Effusion" in targets:
        eff_j = targets.index("Effusion")
        eff_std = float(np.std(ranked_qe[:, eff_j]))
        if eff_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Effusion"] + 0.25 * (ranked_qe[:, eff_j] - 0.5), EPS, 1 - EPS)
            mixed[:, eff_j] = 0.50 * mixed[:, eff_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed Qwen Effusion std", eff_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant Qwen Effusion std", eff_std, "; keeping 0.534 stack")
    else:
        print(STRATEGY, "Qwen Effusion missing; keeping 0.534 stack. n7qe", n7qe, "nIqe", nIqe)

if STRATEGY in {"weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_cheng"} and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). Qwen LM std=0.408
    # and public LM widened 0.287–0.490 → 0.279–0.506. Majority Qwen LM
    # (0.40 stack / 0.60 Qwen) is the next one-lever mix-weight.
    w_gold = 0.40 if STRATEGY.endswith("_w60") else 0.60 if STRATEGY.endswith("_w40") else 0.50
    scores7ql, n7ql = learned_scores_llm(False, 2.0, target_name="Lateral Meniscus", finder=discover_qwen_labels)
    scoresIql, nIql = learned_scores_llm(True, 3.5, target_name="Lateral Meniscus", finder=discover_qwen_labels)
    ranked_ql = None
    if scores7ql is not None and scoresIql is not None:
        ranked_ql = 0.5 * rank_cols(scores7ql) + 0.5 * rank_cols(scoresIql)
        print(STRATEGY, "qwen Lateral Meniscus blend 0.50/0.50 n7ql", n7ql, "nIql", nIql)
    elif scores7ql is not None:
        ranked_ql = rank_cols(scores7ql)
        print(STRATEGY, "qwen Lateral Meniscus 7d only n7ql", n7ql, "nIql", nIql)
    if ranked_ql is not None and "Lateral Meniscus" in targets:
        lm_j = targets.index("Lateral Meniscus")
        lm_std = float(np.std(ranked_ql[:, lm_j]))
        if lm_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Lateral Meniscus"] + 0.25 * (ranked_ql[:, lm_j] - 0.5), EPS, 1 - EPS)
            mixed[:, lm_j] = w_gold * mixed[:, lm_j] + (1.0 - w_gold) * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed Qwen Lateral Meniscus std", lm_std, "w_gold", w_gold)
        else:
            print(STRATEGY, "skip constant Qwen Lateral Meniscus std", lm_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "Qwen Lateral Meniscus missing; keeping 0.535 stack. n7ql", n7ql, "nIql", nIql)


if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_qwen" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). Qwen LM mix-weights
    # are exhausted (50/50=0.535, w60=0.534 poison, w40=0.535 tie). Next unused
    # Qwen named-object column is Medial Meniscus (v4 tied 0.527, Gemini tied 0.534).
    scores7qm, n7qm = learned_scores_llm(False, 2.0, target_name="Medial Meniscus", finder=discover_qwen_labels)
    scoresIqm, nIqm = learned_scores_llm(True, 3.5, target_name="Medial Meniscus", finder=discover_qwen_labels)
    ranked_qm = None
    if scores7qm is not None and scoresIqm is not None:
        ranked_qm = 0.5 * rank_cols(scores7qm) + 0.5 * rank_cols(scoresIqm)
        print(STRATEGY, "qwen Medial Meniscus blend 0.50/0.50 n7qm", n7qm, "nIqm", nIqm)
    elif scores7qm is not None:
        ranked_qm = rank_cols(scores7qm)
        print(STRATEGY, "qwen Medial Meniscus 7d only n7qm", n7qm, "nIqm", nIqm)
    if ranked_qm is not None and "Medial Meniscus" in targets:
        mm_j = targets.index("Medial Meniscus")
        mm_std = float(np.std(ranked_qm[:, mm_j]))
        if mm_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Medial Meniscus"] + 0.25 * (ranked_qm[:, mm_j] - 0.5), EPS, 1 - EPS)
            mixed[:, mm_j] = 0.50 * mixed[:, mm_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed Qwen Medial Meniscus std", mm_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant Qwen Medial Meniscus std", mm_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "Qwen Medial Meniscus missing; keeping 0.535 stack. n7qm", n7qm, "nIqm", nIqm)


if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_qwen" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). Qwen MM 50/50
    # tied 0.535 (57871069) after mixing std=0.354; public MM stayed 0.323–0.511.
    # Fracture is the remaining unused Qwen named-object that is not a documented poison.
    scores7qf, n7qf = learned_scores_llm(False, 2.0, target_name="Fracture", finder=discover_qwen_labels)
    scoresIqf, nIqf = learned_scores_llm(True, 3.5, target_name="Fracture", finder=discover_qwen_labels)
    ranked_qf = None
    if scores7qf is not None and scoresIqf is not None:
        ranked_qf = 0.5 * rank_cols(scores7qf) + 0.5 * rank_cols(scoresIqf)
        print(STRATEGY, "qwen Fracture blend 0.50/0.50 n7qf", n7qf, "nIqf", nIqf)
    elif scores7qf is not None:
        ranked_qf = rank_cols(scores7qf)
        print(STRATEGY, "qwen Fracture 7d only n7qf", n7qf, "nIqf", nIqf)
    if ranked_qf is not None and "Fracture" in targets:
        frac_j = targets.index("Fracture")
        frac_std = float(np.std(ranked_qf[:, frac_j]))
        if frac_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Fracture"] + 0.25 * (ranked_qf[:, frac_j] - 0.5), EPS, 1 - EPS)
            mixed[:, frac_j] = 0.50 * mixed[:, frac_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed Qwen Fracture std", frac_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant Qwen Fracture std", frac_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "Qwen Fracture missing; keeping 0.535 stack. n7qf", n7qf, "nIqf", nIqf)


if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_front" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). Qwen Fracture 50/50
    # poisoned to 0.529 (57900067) after mixing std=0.408; public Fracture
    # 0.247845-0.404095 vs hybrid 0.248-0.373. Do not retry Qwen Fracture
    # mix-weights. Qwen named-objects are exhausted. Next unused source is
    # cdeotte frontier Argon+Astra+Opus; LM is the last transferring object.
    scores7fl, n7fl = learned_scores_llm(False, 2.0, target_name="Lateral Meniscus", finder=discover_front_labels)
    scoresIfl, nIfl = learned_scores_llm(True, 3.5, target_name="Lateral Meniscus", finder=discover_front_labels)
    ranked_fl = None
    if scores7fl is not None and scoresIfl is not None:
        ranked_fl = 0.5 * rank_cols(scores7fl) + 0.5 * rank_cols(scoresIfl)
        print(STRATEGY, "front Lateral Meniscus blend 0.50/0.50 n7fl", n7fl, "nIfl", nIfl)
    elif scores7fl is not None:
        ranked_fl = rank_cols(scores7fl)
        print(STRATEGY, "front Lateral Meniscus 7d only n7fl", n7fl, "nIfl", nIfl)
    if ranked_fl is not None and "Lateral Meniscus" in targets:
        lm_j = targets.index("Lateral Meniscus")
        lm_std = float(np.std(ranked_fl[:, lm_j]))
        if lm_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Lateral Meniscus"] + 0.25 * (ranked_fl[:, lm_j] - 0.5), EPS, 1 - EPS)
            mixed[:, lm_j] = 0.50 * mixed[:, lm_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed frontier Lateral Meniscus std", lm_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant frontier Lateral Meniscus std", lm_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "frontier Lateral Meniscus missing; keeping 0.535 stack. n7fl", n7fl, "nIfl", nIfl)



if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_front" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). Frontier LM 50/50
    # tied 0.535 (57902815) after mixing n=4314 std=0.408; public LM 0.275458-0.513739
    # vs Qwen 0.279-0.506. Do not retry frontier LM mix-weights. Next unused
    # transferring named-object on the Deotte frontier key is ACL (v4 0.520, Gemini 0.531,
    # Qwen ACL tied 0.534).
    scores7fa, n7fa = learned_scores_llm(False, 2.0, target_name="ACL", finder=discover_front_labels)
    scoresIfa, nIfa = learned_scores_llm(True, 3.5, target_name="ACL", finder=discover_front_labels)
    ranked_fa = None
    if scores7fa is not None and scoresIfa is not None:
        ranked_fa = 0.5 * rank_cols(scores7fa) + 0.5 * rank_cols(scoresIfa)
        print(STRATEGY, "front ACL blend 0.50/0.50 n7fa", n7fa, "nIfa", nIfa)
    elif scores7fa is not None:
        ranked_fa = rank_cols(scores7fa)
        print(STRATEGY, "front ACL 7d only n7fa", n7fa, "nIfa", nIfa)
    if ranked_fa is not None and "ACL" in targets:
        acl_j = targets.index("ACL")
        acl_std = float(np.std(ranked_fa[:, acl_j]))
        if acl_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["ACL"] + 0.25 * (ranked_fa[:, acl_j] - 0.5), EPS, 1 - EPS)
            mixed[:, acl_j] = 0.50 * mixed[:, acl_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed frontier ACL std", acl_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant frontier ACL std", acl_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "frontier ACL missing; keeping 0.535 stack. n7fa", n7fa, "nIfa", nIfa)



if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_front" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). Frontier ACL 50/50
    # tied 0.535 (57905322) after mixing n=4314 std=0.354; public ACL 0.312231-0.476293
    # vs Gemini 0.336-0.476. Do not retry frontier ACL mix-weights. Next unused
    # transferring named-object on the Deotte frontier key is Effusion (v4 0.524/0.530,
    # Gemini 0.532/0.533, Qwen Effusion tied 0.534).
    scores7fe, n7fe = learned_scores_llm(False, 2.0, target_name="Effusion", finder=discover_front_labels)
    scoresIfe, nIfe = learned_scores_llm(True, 3.5, target_name="Effusion", finder=discover_front_labels)
    ranked_fe = None
    if scores7fe is not None and scoresIfe is not None:
        ranked_fe = 0.5 * rank_cols(scores7fe) + 0.5 * rank_cols(scoresIfe)
        print(STRATEGY, "front Effusion blend 0.50/0.50 n7fe", n7fe, "nIfe", nIfe)
    elif scores7fe is not None:
        ranked_fe = rank_cols(scores7fe)
        print(STRATEGY, "front Effusion 7d only n7fe", n7fe, "nIfe", nIfe)
    if ranked_fe is not None and "Effusion" in targets:
        eff_j = targets.index("Effusion")
        eff_std = float(np.std(ranked_fe[:, eff_j]))
        if eff_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Effusion"] + 0.25 * (ranked_fe[:, eff_j] - 0.5), EPS, 1 - EPS)
            mixed[:, eff_j] = 0.50 * mixed[:, eff_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed frontier Effusion std", eff_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant frontier Effusion std", eff_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "frontier Effusion missing; keeping 0.535 stack. n7fe", n7fe, "nIfe", nIfe)


if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_front" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). Frontier Effusion 50/50
    # tied 0.535 (57907807) after mixing n=4314 std=0.408; public Effusion 0.478449-0.723448
    # vs Gemini majority 0.478-0.718. Do not retry frontier Effusion mix-weights. Next unused
    # named-object on the Deotte frontier key is Medial Meniscus (v4 0.527 tie, Gemini 0.534
    # tie, Qwen 0.535 tie) — unused on this key and not a documented poison.
    scores7fm, n7fm = learned_scores_llm(False, 2.0, target_name="Medial Meniscus", finder=discover_front_labels)
    scoresIfm, nIfm = learned_scores_llm(True, 3.5, target_name="Medial Meniscus", finder=discover_front_labels)
    ranked_fm = None
    if scores7fm is not None and scoresIfm is not None:
        ranked_fm = 0.5 * rank_cols(scores7fm) + 0.5 * rank_cols(scoresIfm)
        print(STRATEGY, "front Medial Meniscus blend 0.50/0.50 n7fm", n7fm, "nIfm", nIfm)
    elif scores7fm is not None:
        ranked_fm = rank_cols(scores7fm)
        print(STRATEGY, "front Medial Meniscus 7d only n7fm", n7fm, "nIfm", nIfm)
    if ranked_fm is not None and "Medial Meniscus" in targets:
        mm_j = targets.index("Medial Meniscus")
        mm_std = float(np.std(ranked_fm[:, mm_j]))
        if mm_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Medial Meniscus"] + 0.25 * (ranked_fm[:, mm_j] - 0.5), EPS, 1 - EPS)
            mixed[:, mm_j] = 0.50 * mixed[:, mm_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed frontier Medial Meniscus std", mm_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant frontier Medial Meniscus std", mm_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "frontier Medial Meniscus missing; keeping 0.535 stack. n7fm", n7fm, "nIfm", nIfm)


if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_front" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). Frontier MM 50/50
    # tied 0.535 (57935572) after mixing n=4314 std=0.408; public MM stayed 0.323-0.542.
    # Frontier LM/ACL/Effusion also tied 0.535. Last unused frontier named-object that
    # is not a documented poison on every key is Fracture (hybrid transferred +0.003;
    # Qwen Fracture poisoned 0.529). Mix frontier Fracture 50/50 only if std>1e-8.
    scores7ff, n7ff = learned_scores_llm(False, 2.0, target_name="Fracture", finder=discover_front_labels)
    scoresIff, nIff = learned_scores_llm(True, 3.5, target_name="Fracture", finder=discover_front_labels)
    ranked_ff = None
    if scores7ff is not None and scoresIff is not None:
        ranked_ff = 0.5 * rank_cols(scores7ff) + 0.5 * rank_cols(scoresIff)
        print(STRATEGY, "front Fracture blend 0.50/0.50 n7ff", n7ff, "nIff", nIff)
    elif scores7ff is not None:
        ranked_ff = rank_cols(scores7ff)
        print(STRATEGY, "front Fracture 7d only n7ff", n7ff, "nIff", nIff)
    if ranked_ff is not None and "Fracture" in targets:
        ff_j = targets.index("Fracture")
        ff_std = float(np.std(ranked_ff[:, ff_j]))
        if ff_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Fracture"] + 0.25 * (ranked_ff[:, ff_j] - 0.5), EPS, 1 - EPS)
            mixed[:, ff_j] = 0.50 * mixed[:, ff_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed frontier Fracture std", ff_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant frontier Fracture std", ff_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "frontier Fracture missing; keeping 0.535 stack. n7ff", n7ff, "nIff", nIff)


if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_pilk" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). Frontier named-objects
    # are exhausted: LM/ACL/Effusion/MM tied 0.535; Fracture poisoned 0.530
    # (57938474) after mixing n=4314 std=0.408 and expanding public Fracture
    # 0.248-0.373 -> 0.247845-0.404095 (same band as Qwen Fracture 0.529).
    # Next unused report-only key is pilkwang/rsna-knee-llm-labels
    # report_labels_v2.csv (n=4406, ACL std=0.328). Mix ACL 50/50 only if std>1e-8.
    scores7pa, n7pa = learned_scores_llm(False, 2.0, target_name="ACL", finder=discover_pilk_labels)
    scoresIpa, nIpa = learned_scores_llm(True, 3.5, target_name="ACL", finder=discover_pilk_labels)
    ranked_pa = None
    if scores7pa is not None and scoresIpa is not None:
        ranked_pa = 0.5 * rank_cols(scores7pa) + 0.5 * rank_cols(scoresIpa)
        print(STRATEGY, "pilk ACL blend 0.50/0.50 n7pa", n7pa, "nIpa", nIpa)
    elif scores7pa is not None:
        ranked_pa = rank_cols(scores7pa)
        print(STRATEGY, "pilk ACL 7d only n7pa", n7pa, "nIpa", nIpa)
    if ranked_pa is not None and "ACL" in targets:
        pa_j = targets.index("ACL")
        pa_std = float(np.std(ranked_pa[:, pa_j]))
        if pa_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["ACL"] + 0.25 * (ranked_pa[:, pa_j] - 0.5), EPS, 1 - EPS)
            mixed[:, pa_j] = 0.50 * mixed[:, pa_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed pilk ACL std", pa_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant pilk ACL std", pa_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "pilk ACL missing; keeping 0.535 stack. n7pa", n7pa, "nIpa", nIpa)


if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_pilk" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). pilkwang ACL 50/50
    # just poisoned (0.534, 57941669) after mixing n=4406 std=0.408 and widening
    # public ACL 0.336-0.476 -> 0.312231-0.484105. Do not retry pilk ACL mix-weights.
    # Mix pilkwang Lateral Meniscus 50/50 only if that head has std>1e-8. LM is the
    # last transferring named object (v4 0.529, Gemini 0.534, Qwen 0.535).
    scores7pl, n7pl = learned_scores_llm(False, 2.0, target_name="Lateral Meniscus", finder=discover_pilk_labels)
    scoresIpl, nIpl = learned_scores_llm(True, 3.5, target_name="Lateral Meniscus", finder=discover_pilk_labels)
    ranked_pl = None
    if scores7pl is not None and scoresIpl is not None:
        ranked_pl = 0.5 * rank_cols(scores7pl) + 0.5 * rank_cols(scoresIpl)
        print(STRATEGY, "pilk LM blend 0.50/0.50 n7pl", n7pl, "nIpl", nIpl)
    elif scores7pl is not None:
        ranked_pl = rank_cols(scores7pl)
        print(STRATEGY, "pilk LM 7d only n7pl", n7pl, "nIpl", nIpl)
    if ranked_pl is not None and "Lateral Meniscus" in targets:
        pl_j = targets.index("Lateral Meniscus")
        pl_std = float(np.std(ranked_pl[:, pl_j]))
        if pl_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Lateral Meniscus"] + 0.25 * (ranked_pl[:, pl_j] - 0.5), EPS, 1 - EPS)
            mixed[:, pl_j] = 0.50 * mixed[:, pl_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed pilk Lateral Meniscus std", pl_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant pilk LM std", pl_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "pilk LM missing; keeping 0.535 stack. n7pl", n7pl, "nIpl", nIpl)



if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_pilk" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). pilkwang LM 50/50
    # just poisoned (0.534, 57944096) on the same report_labels_v2 key that already
    # poisoned ACL (0.534, 57941669). Do not retry pilk LM or ACL mix-weights.
    # Mix pilkwang Effusion 50/50 only if that head has std>1e-8. Effusion is the
    # remaining unused transferring named object (v4 0.524/0.530, Gemini 0.532/0.533,
    # Qwen Effusion tied 0.534).
    scores7pe, n7pe = learned_scores_llm(False, 2.0, target_name="Effusion", finder=discover_pilk_labels)
    scoresIpe, nIpe = learned_scores_llm(True, 3.5, target_name="Effusion", finder=discover_pilk_labels)
    ranked_pe = None
    if scores7pe is not None and scoresIpe is not None:
        ranked_pe = 0.5 * rank_cols(scores7pe) + 0.5 * rank_cols(scoresIpe)
        print(STRATEGY, "pilk Effusion blend 0.50/0.50 n7pe", n7pe, "nIpe", nIpe)
    elif scores7pe is not None:
        ranked_pe = rank_cols(scores7pe)
        print(STRATEGY, "pilk Effusion 7d only n7pe", n7pe, "nIpe", nIpe)
    if ranked_pe is not None and "Effusion" in targets:
        pe_j = targets.index("Effusion")
        pe_std = float(np.std(ranked_pe[:, pe_j]))
        if pe_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Effusion"] + 0.25 * (ranked_pe[:, pe_j] - 0.5), EPS, 1 - EPS)
            mixed[:, pe_j] = 0.50 * mixed[:, pe_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed pilk Effusion std", pe_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant pilk Effusion std", pe_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "pilk Effusion missing; keeping 0.535 stack. n7pe", n7pe, "nIpe", nIpe)


if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_pilk" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). pilkwang Effusion 50/50
    # just tied (0.535, 57945154) after mixing n=4406 std=0.408 and widening public
    # Effusion 0.478-0.718 -> 0.478449-0.723448. Do not retry pilk Effusion/LM/ACL
    # mix-weights. Mix pilkwang Medial Meniscus 50/50 only if that head has std>1e-8.
    # MM is the remaining unused pilkwang named-object (parser 0.517 poison; v4 0.527
    # tie; Gemini 0.534 tie; Qwen 0.535 tie; frontier 0.535 tie).
    scores7pm, n7pm = learned_scores_llm(False, 2.0, target_name="Medial Meniscus", finder=discover_pilk_labels)
    scoresIpm, nIpm = learned_scores_llm(True, 3.5, target_name="Medial Meniscus", finder=discover_pilk_labels)
    ranked_pm = None
    if scores7pm is not None and scoresIpm is not None:
        ranked_pm = 0.5 * rank_cols(scores7pm) + 0.5 * rank_cols(scoresIpm)
        print(STRATEGY, "pilk MM blend 0.50/0.50 n7pm", n7pm, "nIpm", nIpm)
    elif scores7pm is not None:
        ranked_pm = rank_cols(scores7pm)
        print(STRATEGY, "pilk MM 7d only n7pm", n7pm, "nIpm", nIpm)
    if ranked_pm is not None and "Medial Meniscus" in targets:
        pm_j = targets.index("Medial Meniscus")
        pm_std = float(np.std(ranked_pm[:, pm_j]))
        if pm_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Medial Meniscus"] + 0.25 * (ranked_pm[:, pm_j] - 0.5), EPS, 1 - EPS)
            mixed[:, pm_j] = 0.50 * mixed[:, pm_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed pilk Medial Meniscus std", pm_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant pilk MM std", pm_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "pilk MM missing; keeping 0.535 stack. n7pm", n7pm, "nIpm", nIpm)


if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_2src" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). pilkwang named-objects
    # are exhausted: ACL/LM poisoned 0.534; Effusion/MM tied 0.535 (57947039).
    # Next unused report-only key is yunusgmsoy/rsna-knee-llm-labels-2-source-merged
    # report_labels_v2.csv. Mix 2-source ACL 50/50 only if that head has std>1e-8.
    scores7s2, n7s2 = learned_scores_llm(False, 2.0, target_name="ACL", finder=discover_2src_labels)
    scoresIs2, nIs2 = learned_scores_llm(True, 3.5, target_name="ACL", finder=discover_2src_labels)
    ranked_s2 = None
    if scores7s2 is not None and scoresIs2 is not None:
        ranked_s2 = 0.5 * rank_cols(scores7s2) + 0.5 * rank_cols(scoresIs2)
        print(STRATEGY, "2src ACL blend 0.50/0.50 n7s2", n7s2, "nIs2", nIs2)
    elif scores7s2 is not None:
        ranked_s2 = rank_cols(scores7s2)
        print(STRATEGY, "2src ACL 7d only n7s2", n7s2, "nIs2", nIs2)
    if ranked_s2 is not None and "ACL" in targets:
        s2_j = targets.index("ACL")
        s2_std = float(np.std(ranked_s2[:, s2_j]))
        if s2_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["ACL"] + 0.25 * (ranked_s2[:, s2_j] - 0.5), EPS, 1 - EPS)
            mixed[:, s2_j] = 0.50 * mixed[:, s2_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed 2src ACL std", s2_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant 2src ACL std", s2_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "2src ACL missing; keeping 0.535 stack. n7s2", n7s2, "nIs2", nIs2)


if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_2src" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). 2-source ACL 50/50
    # just tied (0.535, 57976260) after mixing n=4407 std=0.354 and leaving public
    # ACL 0.312231-0.476293 (same band as frontier ACL). Do not retry 2src ACL
    # mix-weights. Mix 2-source Lateral Meniscus 50/50 only if that head has std>1e-8.
    # LM is the remaining unused 2-source named-object that actually transferred
    # on this stack (v4 0.529, Gemini 0.534, Qwen 0.535).
    scores7s2l, n7s2l = learned_scores_llm(False, 2.0, target_name="Lateral Meniscus", finder=discover_2src_labels)
    scoresIs2l, nIs2l = learned_scores_llm(True, 3.5, target_name="Lateral Meniscus", finder=discover_2src_labels)
    ranked_s2l = None
    if scores7s2l is not None and scoresIs2l is not None:
        ranked_s2l = 0.5 * rank_cols(scores7s2l) + 0.5 * rank_cols(scoresIs2l)
        print(STRATEGY, "2src LM blend 0.50/0.50 n7s2l", n7s2l, "nIs2l", nIs2l)
    elif scores7s2l is not None:
        ranked_s2l = rank_cols(scores7s2l)
        print(STRATEGY, "2src LM 7d only n7s2l", n7s2l, "nIs2l", nIs2l)
    if ranked_s2l is not None and "Lateral Meniscus" in targets:
        s2l_j = targets.index("Lateral Meniscus")
        s2l_std = float(np.std(ranked_s2l[:, s2l_j]))
        if s2l_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Lateral Meniscus"] + 0.25 * (ranked_s2l[:, s2l_j] - 0.5), EPS, 1 - EPS)
            mixed[:, s2l_j] = 0.50 * mixed[:, s2l_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed 2src Lateral Meniscus std", s2l_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant 2src LM std", s2l_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "2src LM missing; keeping 0.535 stack. n7s2l", n7s2l, "nIs2l", nIs2l)



if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_2src" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). 2-source LM 50/50
    # just poisoned (0.534, 57979214) after mixing n=4407 std=0.408 and leaving
    # public LM 0.275458-0.513739 (same band as frontier LM). Do not retry 2src
    # LM mix-weights. Mix 2-source Effusion 50/50 only if that head has std>1e-8.
    # Effusion is the remaining unused 2-source named-object that actually
    # transferred on this stack (v4 0.524/0.530, Gemini 0.532/0.533,
    # Qwen/frontier/pilk ties).
    scores7s2e, n7s2e = learned_scores_llm(False, 2.0, target_name="Effusion", finder=discover_2src_labels)
    scoresIs2e, nIs2e = learned_scores_llm(True, 3.5, target_name="Effusion", finder=discover_2src_labels)
    ranked_s2e = None
    if scores7s2e is not None and scoresIs2e is not None:
        ranked_s2e = 0.5 * rank_cols(scores7s2e) + 0.5 * rank_cols(scoresIs2e)
        print(STRATEGY, "2src Effusion blend 0.50/0.50 n7s2e", n7s2e, "nIs2e", nIs2e)
    elif scores7s2e is not None:
        ranked_s2e = rank_cols(scores7s2e)
        print(STRATEGY, "2src Effusion 7d only n7s2e", n7s2e, "nIs2e", nIs2e)
    if ranked_s2e is not None and "Effusion" in targets:
        s2e_j = targets.index("Effusion")
        s2e_std = float(np.std(ranked_s2e[:, s2e_j]))
        if s2e_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Effusion"] + 0.25 * (ranked_s2e[:, s2e_j] - 0.5), EPS, 1 - EPS)
            mixed[:, s2e_j] = 0.50 * mixed[:, s2e_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed 2src Effusion std", s2e_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant 2src Effusion std", s2e_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "2src Effusion missing; keeping 0.535 stack. n7s2e", n7s2e, "nIs2e", nIs2e)


if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_2src" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). 2-source Effusion
    # 50/50 just tied (0.535, 57981528) after mixing n=4407 std=0.408 and leaving
    # public Effusion 0.478449-0.723448 (same band as frontier/pilk Effusion).
    # Do not retry 2src Effusion mix-weights. Mix 2-source Medial Meniscus 50/50
    # only if that head has std>1e-8. MM is the remaining unused 2-source
    # named-object (parser 0.517 poison; v4 0.527 tie; Gemini 0.534 tie;
    # Qwen/frontier/pilk ties).
    scores7s2m, n7s2m = learned_scores_llm(False, 2.0, target_name="Medial Meniscus", finder=discover_2src_labels)
    scoresIs2m, nIs2m = learned_scores_llm(True, 3.5, target_name="Medial Meniscus", finder=discover_2src_labels)
    ranked_s2m = None
    if scores7s2m is not None and scoresIs2m is not None:
        ranked_s2m = 0.5 * rank_cols(scores7s2m) + 0.5 * rank_cols(scoresIs2m)
        print(STRATEGY, "2src Medial Meniscus blend 0.50/0.50 n7s2m", n7s2m, "nIs2m", nIs2m)
    elif scores7s2m is not None:
        ranked_s2m = rank_cols(scores7s2m)
        print(STRATEGY, "2src Medial Meniscus 7d only n7s2m", n7s2m, "nIs2m", nIs2m)
    if ranked_s2m is not None and "Medial Meniscus" in targets:
        s2m_j = targets.index("Medial Meniscus")
        s2m_std = float(np.std(ranked_s2m[:, s2m_j]))
        if s2m_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Medial Meniscus"] + 0.25 * (ranked_s2m[:, s2m_j] - 0.5), EPS, 1 - EPS)
            mixed[:, s2m_j] = 0.50 * mixed[:, s2m_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed 2src Medial Meniscus std", s2m_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant 2src Medial Meniscus std", s2m_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "2src Medial Meniscus missing; keeping 0.535 stack. n7s2m", n7s2m, "nIs2m", nIs2m)



if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_sol" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). 2-source named-objects
    # are exhausted: ACL/Effusion/MM tied 0.535; LM poisoned 0.534 (57979214).
    # mm_2src just tied (0.535, 57983832) after mixing n=4407 std=0.408 and widening
    # public MM 0.323-0.511 -> 0.323276-0.542026 (same band as pilkwang MM).
    # Next unused report-only key is lixin73/rsna-knee-llm-report-labels-sol56
    # report_labels_gpt56sol.csv (n=4407, ACL std=0.387). Mix ACL 50/50 only if std>1e-8.
    scores7sol, n7sol = learned_scores_llm(False, 2.0, target_name="ACL", finder=discover_sol_labels)
    scoresIsol, nIsol = learned_scores_llm(True, 3.5, target_name="ACL", finder=discover_sol_labels)
    ranked_sol = None
    if scores7sol is not None and scoresIsol is not None:
        ranked_sol = 0.5 * rank_cols(scores7sol) + 0.5 * rank_cols(scoresIsol)
        print(STRATEGY, "sol ACL blend 0.50/0.50 n7sol", n7sol, "nIsol", nIsol)
    elif scores7sol is not None:
        ranked_sol = rank_cols(scores7sol)
        print(STRATEGY, "sol ACL 7d only n7sol", n7sol, "nIsol", nIsol)
    if ranked_sol is not None and "ACL" in targets:
        sol_j = targets.index("ACL")
        sol_std = float(np.std(ranked_sol[:, sol_j]))
        if sol_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["ACL"] + 0.25 * (ranked_sol[:, sol_j] - 0.5), EPS, 1 - EPS)
            mixed[:, sol_j] = 0.50 * mixed[:, sol_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed sol ACL std", sol_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant sol ACL std", sol_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "sol ACL missing; keeping 0.535 stack. n7sol", n7sol, "nIsol", nIsol)


if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_sol" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). sol ACL just
    # poisoned (0.534, 58012035) after mixing n=4407 std=0.408 and widening
    # public ACL 0.336-0.476 -> 0.312231-0.507543 (same band as pilkwang ACL).
    # Do not retry sol ACL mix-weights. Next unused sol named-object is
    # Lateral Meniscus (v4 0.529, Gemini 0.534, Qwen 0.535). Mix LM 50/50
    # only if std>1e-8.
    scores7sol, n7sol = learned_scores_llm(False, 2.0, target_name="Lateral Meniscus", finder=discover_sol_labels)
    scoresIsol, nIsol = learned_scores_llm(True, 3.5, target_name="Lateral Meniscus", finder=discover_sol_labels)
    ranked_sol = None
    if scores7sol is not None and scoresIsol is not None:
        ranked_sol = 0.5 * rank_cols(scores7sol) + 0.5 * rank_cols(scoresIsol)
        print(STRATEGY, "sol Lateral Meniscus blend 0.50/0.50 n7sol", n7sol, "nIsol", nIsol)
    elif scores7sol is not None:
        ranked_sol = rank_cols(scores7sol)
        print(STRATEGY, "sol Lateral Meniscus 7d only n7sol", n7sol, "nIsol", nIsol)
    if ranked_sol is not None and "Lateral Meniscus" in targets:
        sol_j = targets.index("Lateral Meniscus")
        sol_std = float(np.std(ranked_sol[:, sol_j]))
        if sol_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Lateral Meniscus"] + 0.25 * (ranked_sol[:, sol_j] - 0.5), EPS, 1 - EPS)
            mixed[:, sol_j] = 0.50 * mixed[:, sol_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed sol Lateral Meniscus std", sol_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant sol Lateral Meniscus std", sol_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "sol Lateral Meniscus missing; keeping 0.535 stack. n7sol", n7sol, "nIsol", nIsol)



if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_sol" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). sol LM just
    # tied (0.535, 58014616) after mixing n=4407 std=0.408 and widening
    # public LM 0.279-0.506 -> 0.275458-0.513739 (same band as frontier LM).
    # Do not retry sol LM mix-weights. Next unused sol named-object is
    # Effusion (v4 0.524/0.530, Gemini 0.532/0.533, later keys tied).
    # Mix Effusion 50/50 only if std>1e-8.
    scores7sol, n7sol = learned_scores_llm(False, 2.0, target_name="Effusion", finder=discover_sol_labels)
    scoresIsol, nIsol = learned_scores_llm(True, 3.5, target_name="Effusion", finder=discover_sol_labels)
    ranked_sol = None
    if scores7sol is not None and scoresIsol is not None:
        ranked_sol = 0.5 * rank_cols(scores7sol) + 0.5 * rank_cols(scoresIsol)
        print(STRATEGY, "sol Effusion blend 0.50/0.50 n7sol", n7sol, "nIsol", nIsol)
    elif scores7sol is not None:
        ranked_sol = rank_cols(scores7sol)
        print(STRATEGY, "sol Effusion 7d only n7sol", n7sol, "nIsol", nIsol)
    if ranked_sol is not None and "Effusion" in targets:
        sol_j = targets.index("Effusion")
        sol_std = float(np.std(ranked_sol[:, sol_j]))
        if sol_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Effusion"] + 0.25 * (ranked_sol[:, sol_j] - 0.5), EPS, 1 - EPS)
            mixed[:, sol_j] = 0.50 * mixed[:, sol_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed sol Effusion std", sol_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant sol Effusion std", sol_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "sol Effusion missing; keeping 0.535 stack. n7sol", n7sol, "nIsol", nIsol)



if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_sol" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). sol Effusion just
    # tied (0.535, 58017601) after mixing n=4407 std=0.408 and leaving public
    # Effusion 0.478449-0.723448 (same band as Qwen/frontier/pilk/2src Effusion).
    # Do not retry sol Effusion mix-weights. Next unused sol named-object is
    # Medial Meniscus (parser 0.517 poison; v4 0.527 tie; Gemini 0.534 tie;
    # Qwen/frontier/pilk/2src ties). Mix MM 50/50 only if std>1e-8.
    scores7sol, n7sol = learned_scores_llm(False, 2.0, target_name="Medial Meniscus", finder=discover_sol_labels)
    scoresIsol, nIsol = learned_scores_llm(True, 3.5, target_name="Medial Meniscus", finder=discover_sol_labels)
    ranked_sol = None
    if scores7sol is not None and scoresIsol is not None:
        ranked_sol = 0.5 * rank_cols(scores7sol) + 0.5 * rank_cols(scoresIsol)
        print(STRATEGY, "sol Medial Meniscus blend 0.50/0.50 n7sol", n7sol, "nIsol", nIsol)
    elif scores7sol is not None:
        ranked_sol = rank_cols(scores7sol)
        print(STRATEGY, "sol Medial Meniscus 7d only n7sol", n7sol, "nIsol", nIsol)
    if ranked_sol is not None and "Medial Meniscus" in targets:
        sol_j = targets.index("Medial Meniscus")
        sol_std = float(np.std(ranked_sol[:, sol_j]))
        if sol_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Medial Meniscus"] + 0.25 * (ranked_sol[:, sol_j] - 0.5), EPS, 1 - EPS)
            mixed[:, sol_j] = 0.50 * mixed[:, sol_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed sol Medial Meniscus std", sol_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant sol Medial Meniscus std", sol_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "sol Medial Meniscus missing; keeping 0.535 stack. n7sol", n7sol, "nIsol", nIsol)


if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_3src" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). sol named-objects
    # are exhausted: ACL/MM poisoned 0.534; LM/Effusion tied 0.535. mm_sol just
    # poisoned (0.534, 58019833) after mixing n=4407 std=0.408 and widening
    # public MM 0.323-0.511 -> 0.323276-0.542026 (same band as pilkwang/2src MM).
    # Next unused report-only key is yunusgmsoy/rsna-knee-abnormality-3-source-merged-labels
    # report_labels_v3.csv (n=4407, ACL std=0.286). Mix ACL 50/50 only if std>1e-8.
    scores7s3, n7s3 = learned_scores_llm(False, 2.0, target_name="ACL", finder=discover_3src_labels)
    scoresIs3, nIs3 = learned_scores_llm(True, 3.5, target_name="ACL", finder=discover_3src_labels)
    ranked_s3 = None
    if scores7s3 is not None and scoresIs3 is not None:
        ranked_s3 = 0.5 * rank_cols(scores7s3) + 0.5 * rank_cols(scoresIs3)
        print(STRATEGY, "3src ACL blend 0.50/0.50 n7s3", n7s3, "nIs3", nIs3)
    elif scores7s3 is not None:
        ranked_s3 = rank_cols(scores7s3)
        print(STRATEGY, "3src ACL 7d only n7s3", n7s3, "nIs3", nIs3)
    if ranked_s3 is not None and "ACL" in targets:
        s3_j = targets.index("ACL")
        s3_std = float(np.std(ranked_s3[:, s3_j]))
        if s3_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["ACL"] + 0.25 * (ranked_s3[:, s3_j] - 0.5), EPS, 1 - EPS)
            mixed[:, s3_j] = 0.50 * mixed[:, s3_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed 3src ACL std", s3_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant 3src ACL std", s3_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "3src ACL missing; keeping 0.535 stack. n7s3", n7s3, "nIs3", nIs3)


if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_3src" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). 3-source ACL just
    # poisoned (0.534, 58022193) after mixing n=4407 std=0.354 and widening
    # public ACL 0.336-0.476 -> 0.312231-0.476293 (same band as frontier/2src ACL).
    # Do not retry 3-source ACL mix-weights. Next unused 3-source named-object is
    # Lateral Meniscus (v4 0.529, Gemini 0.534, Qwen 0.535). Mix LM 50/50 only if std>1e-8.
    scores7s3, n7s3 = learned_scores_llm(False, 2.0, target_name="Lateral Meniscus", finder=discover_3src_labels)
    scoresIs3, nIs3 = learned_scores_llm(True, 3.5, target_name="Lateral Meniscus", finder=discover_3src_labels)
    ranked_s3 = None
    if scores7s3 is not None and scoresIs3 is not None:
        ranked_s3 = 0.5 * rank_cols(scores7s3) + 0.5 * rank_cols(scoresIs3)
        print(STRATEGY, "3src Lateral Meniscus blend 0.50/0.50 n7s3", n7s3, "nIs3", nIs3)
    elif scores7s3 is not None:
        ranked_s3 = rank_cols(scores7s3)
        print(STRATEGY, "3src Lateral Meniscus 7d only n7s3", n7s3, "nIs3", nIs3)
    if ranked_s3 is not None and "Lateral Meniscus" in targets:
        s3_j = targets.index("Lateral Meniscus")
        s3_std = float(np.std(ranked_s3[:, s3_j]))
        if s3_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Lateral Meniscus"] + 0.25 * (ranked_s3[:, s3_j] - 0.5), EPS, 1 - EPS)
            mixed[:, s3_j] = 0.50 * mixed[:, s3_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed 3src Lateral Meniscus std", s3_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant 3src Lateral Meniscus std", s3_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "3src Lateral Meniscus missing; keeping 0.535 stack. n7s3", n7s3, "nIs3", nIs3)


if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_3src" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). 3-source LM just
    # tied (0.535, 58048080) after mixing n=4407 std=0.408 and leaving public
    # LM 0.275458-0.513739 (same band as frontier/2src LM vs Qwen 0.279-0.506).
    # Do not retry 3-source LM mix-weights. Next unused 3-source named-object is
    # Effusion (v4 0.524/0.530, Gemini 0.532/0.533, Qwen/frontier/pilk/2src/sol ties).
    # Mix Effusion 50/50 only if std>1e-8.
    scores7s3e, n7s3e = learned_scores_llm(False, 2.0, target_name="Effusion", finder=discover_3src_labels)
    scoresIs3e, nIs3e = learned_scores_llm(True, 3.5, target_name="Effusion", finder=discover_3src_labels)
    ranked_s3e = None
    if scores7s3e is not None and scoresIs3e is not None:
        ranked_s3e = 0.5 * rank_cols(scores7s3e) + 0.5 * rank_cols(scoresIs3e)
        print(STRATEGY, "3src Effusion blend 0.50/0.50 n7s3e", n7s3e, "nIs3e", nIs3e)
    elif scores7s3e is not None:
        ranked_s3e = rank_cols(scores7s3e)
        print(STRATEGY, "3src Effusion 7d only n7s3e", n7s3e, "nIs3e", nIs3e)
    if ranked_s3e is not None and "Effusion" in targets:
        s3e_j = targets.index("Effusion")
        s3e_std = float(np.std(ranked_s3e[:, s3e_j]))
        if s3e_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Effusion"] + 0.25 * (ranked_s3e[:, s3e_j] - 0.5), EPS, 1 - EPS)
            mixed[:, s3e_j] = 0.50 * mixed[:, s3e_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed 3src Effusion std", s3e_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant 3src Effusion std", s3e_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "3src Effusion missing; keeping 0.535 stack. n7s3e", n7s3e, "nIs3e", nIs3e)


if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_3src" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). 3-source Effusion just
    # tied (0.535, 58050095) after mixing n=4407 std=0.408 and leaving public
    # Effusion 0.478449-0.723448 (same band as Qwen/frontier/pilk/2src/sol Effusion
    # vs Gemini 0.478-0.718). Do not retry 3-source Effusion mix-weights. Next unused
    # 3-source named-object is Medial Meniscus (parser 0.517 poison; v4 0.527 tie;
    # Gemini 0.534 tie; Qwen/frontier/pilk/2src ties; sol 0.534 poison). Mix MM 50/50
    # only if std>1e-8.
    scores7s3m, n7s3m = learned_scores_llm(False, 2.0, target_name="Medial Meniscus", finder=discover_3src_labels)
    scoresIs3m, nIs3m = learned_scores_llm(True, 3.5, target_name="Medial Meniscus", finder=discover_3src_labels)
    ranked_s3m = None
    if scores7s3m is not None and scoresIs3m is not None:
        ranked_s3m = 0.5 * rank_cols(scores7s3m) + 0.5 * rank_cols(scoresIs3m)
        print(STRATEGY, "3src Medial Meniscus blend 0.50/0.50 n7s3m", n7s3m, "nIs3m", nIs3m)
    elif scores7s3m is not None:
        ranked_s3m = rank_cols(scores7s3m)
        print(STRATEGY, "3src Medial Meniscus 7d only n7s3m", n7s3m, "nIs3m", nIs3m)
    if ranked_s3m is not None and "Medial Meniscus" in targets:
        s3m_j = targets.index("Medial Meniscus")
        s3m_std = float(np.std(ranked_s3m[:, s3m_j]))
        if s3m_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Medial Meniscus"] + 0.25 * (ranked_s3m[:, s3m_j] - 0.5), EPS, 1 - EPS)
            mixed[:, s3m_j] = 0.50 * mixed[:, s3m_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed 3src Medial Meniscus std", s3m_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant 3src Medial Meniscus std", s3m_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "3src Medial Meniscus missing; keeping 0.535 stack. n7s3m", n7s3m, "nIs3m", nIs3m)



if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_song" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). 3-source named-objects
    # that transfer on this stack are exhausted: ACL poison 0.534 (58022193), LM/Effusion/MM
    # ties 0.535 (58048080 / 58050095 / 58052796). mm_3src mixed n=4407 std=0.408 and left
    # public MM 0.323276-0.542026 (same band as pilk/2src/sol MM). Do not retry 3-source MM
    # mix-weights. Next unused report-only key is songqizhou/rsna-knee-llm-labels
    # 9_final_label_with_gold58_studies.csv (n=4407, gold58 macro 0.952, no report text).
    # Mix ACL 50/50 only if std>1e-8.
    scores7sg, n7sg = learned_scores_llm(False, 2.0, target_name="ACL", finder=discover_song_labels)
    scoresIsg, nIsg = learned_scores_llm(True, 3.5, target_name="ACL", finder=discover_song_labels)
    ranked_sg = None
    if scores7sg is not None and scoresIsg is not None:
        ranked_sg = 0.5 * rank_cols(scores7sg) + 0.5 * rank_cols(scoresIsg)
        print(STRATEGY, "song ACL blend 0.50/0.50 n7sg", n7sg, "nIsg", nIsg)
    elif scores7sg is not None:
        ranked_sg = rank_cols(scores7sg)
        print(STRATEGY, "song ACL 7d only n7sg", n7sg, "nIsg", nIsg)
    if ranked_sg is not None and "ACL" in targets:
        sg_j = targets.index("ACL")
        sg_std = float(np.std(ranked_sg[:, sg_j]))
        if sg_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["ACL"] + 0.25 * (ranked_sg[:, sg_j] - 0.5), EPS, 1 - EPS)
            mixed[:, sg_j] = 0.50 * mixed[:, sg_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed song ACL std", sg_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant song ACL std", sg_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "song ACL missing; keeping 0.535 stack. n7sg", n7sg, "nIsg", nIsg)


if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_song" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). song ACL just poisoned
    # (0.533, 58054860) after mixing n=4407 std=0.408 and compressing public ACL
    # 0.336-0.476 -> 0.382543-0.437231. Do not retry song ACL mix-weights. Next unused
    # transferring named-object on this key is Lateral Meniscus (v4 0.529, Gemini 0.534,
    # Qwen 0.535). Mix LM 50/50 only if std>1e-8.
    scores7sgl, n7sgl = learned_scores_llm(False, 2.0, target_name="Lateral Meniscus", finder=discover_song_labels)
    scoresIsgl, nIsgl = learned_scores_llm(True, 3.5, target_name="Lateral Meniscus", finder=discover_song_labels)
    ranked_sgl = None
    if scores7sgl is not None and scoresIsgl is not None:
        ranked_sgl = 0.5 * rank_cols(scores7sgl) + 0.5 * rank_cols(scoresIsgl)
        print(STRATEGY, "song Lateral Meniscus blend 0.50/0.50 n7sgl", n7sgl, "nIsgl", nIsgl)
    elif scores7sgl is not None:
        ranked_sgl = rank_cols(scores7sgl)
        print(STRATEGY, "song Lateral Meniscus 7d only n7sgl", n7sgl, "nIsgl", nIsgl)
    if ranked_sgl is not None and "Lateral Meniscus" in targets:
        sgl_j = targets.index("Lateral Meniscus")
        sgl_std = float(np.std(ranked_sgl[:, sgl_j]))
        if sgl_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Lateral Meniscus"] + 0.25 * (ranked_sgl[:, sgl_j] - 0.5), EPS, 1 - EPS)
            mixed[:, sgl_j] = 0.50 * mixed[:, sgl_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed song Lateral Meniscus std", sgl_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant song Lateral Meniscus std", sgl_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "song Lateral Meniscus missing; keeping 0.535 stack. n7sgl", n7sgl, "nIsgl", nIsgl)



if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_song" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). song LM just tied
    # (0.535, 58056924) after mixing n=4407 std=0.408 and leaving public LM
    # 0.275458-0.513739 (same band as frontier/2src/sol/3src LM vs Qwen 0.279-0.506).
    # Do not retry song LM mix-weights. Next unused transferring named-object on
    # this key is Effusion (v4 0.524/0.530, Gemini 0.532/0.533, Qwen/frontier/pilk/
    # 2src/sol/3src ties). Mix Effusion 50/50 only if std>1e-8.
    scores7sge, n7sge = learned_scores_llm(False, 2.0, target_name="Effusion", finder=discover_song_labels)
    scoresIsge, nIsge = learned_scores_llm(True, 3.5, target_name="Effusion", finder=discover_song_labels)
    ranked_sge = None
    if scores7sge is not None and scoresIsge is not None:
        ranked_sge = 0.5 * rank_cols(scores7sge) + 0.5 * rank_cols(scoresIsge)
        print(STRATEGY, "song Effusion blend 0.50/0.50 n7sge", n7sge, "nIsge", nIsge)
    elif scores7sge is not None:
        ranked_sge = rank_cols(scores7sge)
        print(STRATEGY, "song Effusion 7d only n7sge", n7sge, "nIsge", nIsge)
    if ranked_sge is not None and "Effusion" in targets:
        sge_j = targets.index("Effusion")
        sge_std = float(np.std(ranked_sge[:, sge_j]))
        if sge_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Effusion"] + 0.25 * (ranked_sge[:, sge_j] - 0.5), EPS, 1 - EPS)
            mixed[:, sge_j] = 0.50 * mixed[:, sge_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed song Effusion std", sge_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant song Effusion std", sge_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "song Effusion missing; keeping 0.535 stack. n7sge", n7sge, "nIsge", nIsge)



if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_song" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). song Effusion just
    # poisoned (0.530, 58081630) after mixing n=4407 std=0.408 and compressing
    # public Effusion 0.478-0.718 -> 0.598448-0.608448 (std=0.005). Do not retry
    # song Effusion mix-weights. Next unused named-object on this song key is
    # Medial Meniscus (parser 0.517 poison; v4 0.527 tie; Gemini 0.534 tie;
    # Qwen/frontier/pilk/2src ties; sol 0.534 poison; 3src 0.535 tie). Mix MM
    # 50/50 only if std>1e-8.
    scores7sgm, n7sgm = learned_scores_llm(False, 2.0, target_name="Medial Meniscus", finder=discover_song_labels)
    scoresIsgm, nIsgm = learned_scores_llm(True, 3.5, target_name="Medial Meniscus", finder=discover_song_labels)
    ranked_sgm = None
    if scores7sgm is not None and scoresIsgm is not None:
        ranked_sgm = 0.5 * rank_cols(scores7sgm) + 0.5 * rank_cols(scoresIsgm)
        print(STRATEGY, "song Medial Meniscus blend 0.50/0.50 n7sgm", n7sgm, "nIsgm", nIsgm)
    elif scores7sgm is not None:
        ranked_sgm = rank_cols(scores7sgm)
        print(STRATEGY, "song Medial Meniscus 7d only n7sgm", n7sgm, "nIsgm", nIsgm)
    if ranked_sgm is not None and "Medial Meniscus" in targets:
        sgm_j = targets.index("Medial Meniscus")
        sgm_std = float(np.std(ranked_sgm[:, sgm_j]))
        if sgm_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Medial Meniscus"] + 0.25 * (ranked_sgm[:, sgm_j] - 0.5), EPS, 1 - EPS)
            mixed[:, sgm_j] = 0.50 * mixed[:, sgm_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed song Medial Meniscus std", sgm_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant song Medial Meniscus std", sgm_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "song Medial Meniscus missing; keeping 0.535 stack. n7sgm", n7sgm, "nIsgm", nIsgm)



if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_cheng" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). song named-objects
    # are exhausted: ACL poison 0.533 (58054860), LM tie 0.535 (58056924),
    # Effusion poison 0.530 (58081630), MM tie 0.535 (58084331). Mix-weights on
    # a tying head must not be retried. chengtingyi 1_text/labels4407_text.csv
    # is an unused report-only key (n=4407, ACL std=0.315, native 12 targets).
    # Mix ACL 50/50 only if std>1e-8. Do not read vision folders.
    scores7ca, n7ca = learned_scores_llm(False, 2.0, target_name="ACL", finder=discover_cheng_labels)
    scoresIca, nIca = learned_scores_llm(True, 3.5, target_name="ACL", finder=discover_cheng_labels)
    ranked_ca = None
    if scores7ca is not None and scoresIca is not None:
        ranked_ca = 0.5 * rank_cols(scores7ca) + 0.5 * rank_cols(scoresIca)
        print(STRATEGY, "cheng ACL blend 0.50/0.50 n7ca", n7ca, "nIca", nIca)
    elif scores7ca is not None:
        ranked_ca = rank_cols(scores7ca)
        print(STRATEGY, "cheng ACL 7d only n7ca", n7ca, "nIca", nIca)
    if ranked_ca is not None and "ACL" in targets:
        ca_j = targets.index("ACL")
        ca_std = float(np.std(ranked_ca[:, ca_j]))
        if ca_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["ACL"] + 0.25 * (ranked_ca[:, ca_j] - 0.5), EPS, 1 - EPS)
            mixed[:, ca_j] = 0.50 * mixed[:, ca_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed cheng ACL std", ca_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant cheng ACL std", ca_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "cheng ACL missing; keeping 0.535 stack. n7ca", n7ca, "nIca", nIca)




if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_cheng" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). cheng ACL just
    # poisoned (0.534, 58086759) after mixing n=4407 std=0.408 and shifting
    # public ACL 0.336-0.476 -> 0.374731-0.484105. Do not retry cheng ACL
    # mix-weights. Next unused transferring named-object on this cheng key
    # is Lateral Meniscus (v4 0.529, Gemini 0.534, Qwen 0.535). Mix LM 50/50
    # only if std>1e-8. Do not read vision folders.
    scores7cl, n7cl = learned_scores_llm(False, 2.0, target_name="Lateral Meniscus", finder=discover_cheng_labels)
    scoresIcl, nIcl = learned_scores_llm(True, 3.5, target_name="Lateral Meniscus", finder=discover_cheng_labels)
    ranked_cl = None
    if scores7cl is not None and scoresIcl is not None:
        ranked_cl = 0.5 * rank_cols(scores7cl) + 0.5 * rank_cols(scoresIcl)
        print(STRATEGY, "cheng Lateral Meniscus blend 0.50/0.50 n7cl", n7cl, "nIcl", nIcl)
    elif scores7cl is not None:
        ranked_cl = rank_cols(scores7cl)
        print(STRATEGY, "cheng Lateral Meniscus 7d only n7cl", n7cl, "nIcl", nIcl)
    if ranked_cl is not None and "Lateral Meniscus" in targets:
        cl_j = targets.index("Lateral Meniscus")
        cl_std = float(np.std(ranked_cl[:, cl_j]))
        if cl_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Lateral Meniscus"] + 0.25 * (ranked_cl[:, cl_j] - 0.5), EPS, 1 - EPS)
            mixed[:, cl_j] = 0.50 * mixed[:, cl_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed cheng Lateral Meniscus std", cl_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant cheng Lateral Meniscus std", cl_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "cheng Lateral Meniscus missing; keeping 0.535 stack. n7cl", n7cl, "nIcl", nIcl)



if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_cheng" and used_learned and train_series is not None:
    # Keep the accepted Qwen LM 50/50 stack (0.535, 57864395). cheng LM just
    # tied (0.535, 58088938) after mixing n=4407 std=0.408 and leaving public
    # LM 0.275458-0.513739 (same band as frontier/2src/sol/3src/song LM vs
    # Qwen 0.279-0.506). Do not retry cheng LM mix-weights. Next unused
    # transferring named-object on this cheng key is Effusion (v4 0.524/0.530,
    # Gemini 0.532/0.533). Mix Effusion 50/50 only if std>1e-8. Do not read
    # vision folders.
    scores7ce, n7ce = learned_scores_llm(False, 2.0, target_name="Effusion", finder=discover_cheng_labels)
    scoresIce, nIce = learned_scores_llm(True, 3.5, target_name="Effusion", finder=discover_cheng_labels)
    ranked_ce = None
    if scores7ce is not None and scoresIce is not None:
        ranked_ce = 0.5 * rank_cols(scores7ce) + 0.5 * rank_cols(scoresIce)
        print(STRATEGY, "cheng Effusion blend 0.50/0.50 n7ce", n7ce, "nIce", nIce)
    elif scores7ce is not None:
        ranked_ce = rank_cols(scores7ce)
        print(STRATEGY, "cheng Effusion 7d only n7ce", n7ce, "nIce", nIce)
    if ranked_ce is not None and "Effusion" in targets:
        ce_j = targets.index("Effusion")
        ce_std = float(np.std(ranked_ce[:, ce_j]))
        if ce_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Effusion"] + 0.25 * (ranked_ce[:, ce_j] - 0.5), EPS, 1 - EPS)
            mixed[:, ce_j] = 0.50 * mixed[:, ce_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed cheng Effusion std", ce_std, "w_gold", 0.50)
        else:
            print(STRATEGY, "skip constant cheng Effusion std", ce_std, "; keeping 0.535 stack")
    else:
        print(STRATEGY, "cheng Effusion missing; keeping 0.535 stack. n7ce", n7ce, "nIce", nIce)




if STRATEGY in {"weak_rank_bakers_frac_gem_llm", "weak_rank_bakers_frac_hyb_gem_llm"} and used_learned and train_series is not None:
    # gem_llm: 0.524 stack + Gemini Fracture (falsified 0.525). hyb_gem: 0.527 hybrid stack then Gemini.
    # Gemini-only transferred vs 0.524 but lost to hybrid; stacking is the remaining one-lever test.
    scores7g, n7g = learned_scores_llm(False, 2.0, target_name="Fracture", finder=discover_gemini_labels)
    scoresIg, nIg = learned_scores_llm(True, 3.5, target_name="Fracture", finder=discover_gemini_labels)
    ranked_g = None
    if scores7g is not None and scoresIg is not None:
        ranked_g = 0.5 * rank_cols(scores7g) + 0.5 * rank_cols(scoresIg)
        print(STRATEGY, "gemini Fracture blend 0.50/0.50 n7g", n7g, "nIg", nIg)
    elif scores7g is not None:
        ranked_g = rank_cols(scores7g)
        print(STRATEGY, "gemini Fracture 7d only n7g", n7g, "nIg", nIg)
    if ranked_g is not None and "Fracture" in targets:
        frac_j = targets.index("Fracture")
        frac_std = float(np.std(ranked_g[:, frac_j]))
        if frac_std > 1e-8:
            mixed = out[targets].to_numpy(dtype=float)
            weak_col = np.clip(prev["Fracture"] + 0.25 * (ranked_g[:, frac_j] - 0.5), EPS, 1 - EPS)
            mixed[:, frac_j] = 0.50 * mixed[:, frac_j] + 0.50 * weak_col
            out = sample[[study_col]].copy()
            for j, t in enumerate(targets):
                out[t] = mixed[:, j]
            out = out[sample.columns]
            print(STRATEGY, "mixed Gemini Fracture std", frac_std)
        else:
            print(STRATEGY, "skip constant Gemini Fracture std", frac_std, "; keeping bakers_eff_llm stack")
    else:
        print(STRATEGY, "Gemini Fracture missing; keeping bakers_eff_llm stack. n7g", n7g, "nIg", nIg)

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
        "dataset_sources": (
            ["stevenleehans/rsna-knee-llm-report-labels", "flight0234/rsna-knee-hybrid-report-labels", "yunusgmsoy/rsna-knee-llm-labels-4-source-merged", "laymond/rsna-knee-abnormality-qwen3-8b-weak-labels", "chengtingyi/rsna-knee-llm-weak-labels"]
            if strategy in {"weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_cheng", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_cheng"}
            else ["stevenleehans/rsna-knee-llm-report-labels", "flight0234/rsna-knee-hybrid-report-labels", "yunusgmsoy/rsna-knee-llm-labels-4-source-merged", "laymond/rsna-knee-abnormality-qwen3-8b-weak-labels", "songqizhou/rsna-knee-llm-labels"]
            if strategy in {"weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_song", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_song"}
            else ["stevenleehans/rsna-knee-llm-report-labels", "flight0234/rsna-knee-hybrid-report-labels", "yunusgmsoy/rsna-knee-llm-labels-4-source-merged", "laymond/rsna-knee-abnormality-qwen3-8b-weak-labels", "yunusgmsoy/rsna-knee-abnormality-3-source-merged-labels"]
            if strategy in {"weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_3src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_3src"}
            else ["stevenleehans/rsna-knee-llm-report-labels", "flight0234/rsna-knee-hybrid-report-labels", "yunusgmsoy/rsna-knee-llm-labels-4-source-merged", "laymond/rsna-knee-abnormality-qwen3-8b-weak-labels", "lixin73/rsna-knee-llm-report-labels-sol56"]
            if strategy in {"weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_sol", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_sol"}
            else ["stevenleehans/rsna-knee-llm-report-labels", "flight0234/rsna-knee-hybrid-report-labels", "yunusgmsoy/rsna-knee-llm-labels-4-source-merged", "laymond/rsna-knee-abnormality-qwen3-8b-weak-labels", "yunusgmsoy/rsna-knee-llm-labels-2-source-merged"]
            if strategy in {"weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_2src", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_2src"}
            else ["stevenleehans/rsna-knee-llm-report-labels", "flight0234/rsna-knee-hybrid-report-labels", "yunusgmsoy/rsna-knee-llm-labels-4-source-merged", "laymond/rsna-knee-abnormality-qwen3-8b-weak-labels", "pilkwang/rsna-knee-llm-labels"]
            if strategy in {"weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_pilk"}
            else ["stevenleehans/rsna-knee-llm-report-labels", "flight0234/rsna-knee-hybrid-report-labels", "yunusgmsoy/rsna-knee-llm-labels-4-source-merged", "laymond/rsna-knee-abnormality-qwen3-8b-weak-labels", "cdeotte/rsna-knee-frontier-llm-hybrid-labels"]
            if strategy in {"weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_front", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_pilk", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_pilk"}
            else ["stevenleehans/rsna-knee-llm-report-labels", "flight0234/rsna-knee-hybrid-report-labels", "yunusgmsoy/rsna-knee-llm-labels-4-source-merged", "laymond/rsna-knee-abnormality-qwen3-8b-weak-labels"]
            if strategy in {"weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_acl_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_eff_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_qwen", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_qwen"}
            else ["stevenleehans/rsna-knee-llm-report-labels", "flight0234/rsna-knee-hybrid-report-labels", "yunusgmsoy/rsna-knee-llm-labels-4-source-merged"]
            if strategy in {"weak_rank_bakers_frac_hyb_gem_llm", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem"}
            else ["stevenleehans/rsna-knee-llm-report-labels", "yunusgmsoy/rsna-knee-llm-labels-4-source-merged"]
            if strategy == "weak_rank_bakers_frac_gem_llm"
            else ["stevenleehans/rsna-knee-llm-report-labels", "flight0234/rsna-knee-hybrid-report-labels"]
            if strategy in {"weak_rank_bakers_frac_hyb_llm", "weak_rank_bakers_frac_hyb_w60", "weak_rank_bakers_frac_hyb_w40", "weak_rank_bakers_frac_hyb_mm_llm", "weak_rank_bakers_frac_hyb_lm_llm", "weak_rank_bakers_frac_hyb_lm_w60", "weak_rank_bakers_frac_hyb_lm_w40", "weak_rank_bakers_frac_hyb_lm_eff_w60", "weak_rank_bakers_frac_hyb_lm_eff_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_w40"}
            else ["stevenleehans/rsna-knee-llm-report-labels"]
            if strategy in {"weak_rank_bakers_llm", "weak_rank_bakers_acl_llm", "weak_rank_bakers_mcl_llm", "weak_rank_bakers_eff_llm", "weak_rank_bakers_syn_llm", "weak_rank_bakers_syn_from_eff", "weak_rank_bakers_cont_llm", "weak_rank_bakers_frac_llm", "weak_rank_bakers_moa_llm", "weak_rank_bakers_loa_llm", "weak_rank_bakers_pfoa_llm"}
            else []
        ),
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": [],
        "model_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return kernel_dir / nb_name
