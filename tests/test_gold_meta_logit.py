"""Local checks for the gold_meta_logit Kaggle notebook strategy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent.stages.hypothesize import run_hypothesize
from agent.stages.implement import _source_for_strategy, implement_notebook
from agent.stages.plan import run_plan
from src.constants import DEFAULT_TARGETS, STUDY_ID_COL


def test_cycle0_plan_selects_gold_rank_lam2(tmp_path):
    dummy = tmp_path / "dummy.md"
    dummy.write_text("x")
    md = run_plan(tmp_path, dummy, dummy, dummy, cycle_id="00_test", day_id="2026-09-20", cycle_num=0)
    assert "gold_rank_lam2" in md.read_text()
    sidecar = json.loads((tmp_path / "2026-09-20_cycle00_test_plan.json").read_text())
    assert sidecar["strategy"] == "gold_rank_lam2"


def test_cycle1_plan_keeps_gold_rank_w50_fallback(tmp_path):
    dummy = tmp_path / "dummy.md"
    dummy.write_text("x")
    md = run_plan(tmp_path, dummy, dummy, dummy, cycle_id="01_test", day_id="2026-09-18", cycle_num=1)
    sidecar = json.loads((tmp_path / "2026-09-18_cycle01_test_plan.json").read_text())
    assert sidecar["strategy"] == "gold_rank_w50"
    assert "0.50" in md.read_text()


def test_cycle2_plan_keeps_gold_rank_interact_fallback(tmp_path):
    dummy = tmp_path / "dummy.md"
    dummy.write_text("x")
    sidecar_md = run_plan(tmp_path, dummy, dummy, dummy, cycle_id="02_test", day_id="2026-09-18", cycle_num=2)
    sidecar = json.loads((tmp_path / "2026-09-18_cycle02_test_plan.json").read_text())
    assert sidecar["strategy"] == "gold_rank_interact"
    assert "0.60" in sidecar_md.read_text()


def test_notebook_contains_learned_metadata_path(tmp_path):
    nb = implement_notebook(tmp_path, "gold_meta_logit", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'gold_meta_logit'" in src
    assert "train_series.csv" in src
    assert "fit_ridge_logit" in src
    assert "rank_cols" in src
    assert "enable_internet" not in src  # notebook body; metadata is separate
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert meta["id"] == "kaggle-user/slug"
    assert meta["competition_sources"] == ["rsna-knee-abnormality-detection"]


def _write_synth(root: Path) -> None:
    rng = np.random.default_rng(0)
    targets = DEFAULT_TARGETS
    n_gold, n_unlab, n_test = 40, 20, 24
    gold_ids = [f"g{i:03d}" for i in range(n_gold)]
    unlab_ids = [f"u{i:03d}" for i in range(n_unlab)]
    test_ids = [f"t{i:03d}" for i in range(n_test)]

    def series_for(uids, tag):
        rows = []
        sag_frac = {}
        for i, uid in enumerate(uids):
            # ACL-positive studies get more sagittal + fluid series
            high = (i % 2 == 0) if tag != "test" else (i < n_test // 2)
            sag_frac[uid] = 0.7 if high else 0.2
            n = 6
            n_sag = int(round(n * sag_frac[uid]))
            planes = ["Sagittal"] * n_sag + ["Coronal"] * (n - n_sag)
            for j, plane in enumerate(planes):
                rows.append(
                    {
                        STUDY_ID_COL: uid,
                        "SeriesInstanceUID": f"{uid}_s{j}",
                        "Fluid_Sensitive": int(high),
                        "Fat_Suppression": int(not high),
                        "Anatomical_Plane": plane,
                    }
                )
        return pd.DataFrame(rows), sag_frac

    train_series, gold_frac = series_for(gold_ids + unlab_ids, "train")
    test_series, _ = series_for(test_ids, "test")

    train_rows = []
    for i, uid in enumerate(gold_ids):
        high = gold_frac[uid] > 0.5
        row = {STUDY_ID_COL: uid, "Report": "n/a"}
        for t in targets:
            if t == "ACL":
                row[t] = int(high)
            else:
                row[t] = int(rng.random() < 0.3)
        train_rows.append(row)
    for uid in unlab_ids:
        row = {STUDY_ID_COL: uid, "Report": "unlabeled"}
        for t in targets:
            row[t] = np.nan
        train_rows.append(row)
    train = pd.DataFrame(train_rows)

    sample_rows = []
    for uid in test_ids:
        row = {STUDY_ID_COL: uid}
        for t in targets:
            row[t] = 0.5
        sample_rows.append(row)
    sample = pd.DataFrame(sample_rows)

    root.mkdir(parents=True, exist_ok=True)
    train.to_csv(root / "train.csv", index=False)
    train_series.to_csv(root / "train_series.csv", index=False)
    test_series.to_csv(root / "test_series.csv", index=False)
    sample.to_csv(root / "sample_submission.csv", index=False)


def test_gold_meta_logit_ranks_synthetic_acl(tmp_path):
    data = tmp_path / "input"
    work = tmp_path / "work"
    work.mkdir()
    _write_synth(data)
    src = _source_for_strategy("gold_meta_logit", "00_test")
    src = src.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
    src = src.replace(
        'path = Path("/kaggle/working/submission.csv")',
        f"path = Path({str(work / 'submission.csv')!r})",
    )
    ns: dict = {}
    exec(compile(src, "gold_meta_logit_nb.py", "exec"), ns, ns)
    out = pd.read_csv(work / "submission.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    # First half of test ids were constructed as high-sagittal / high-fluid (ACL+ protocol)
    high = out.iloc[:12]["ACL"].mean()
    low = out.iloc[12:]["ACL"].mean()
    assert high > low, (high, low)
    assert ns["used_learned"] is True


def _run_strategy(tmp_path, strategy: str):
    data = tmp_path / "input"
    work = tmp_path / "work"
    work.mkdir()
    _write_synth(data)
    src = _source_for_strategy(strategy, "00_test")
    src = src.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
    src = src.replace(
        'path = Path("/kaggle/working/submission.csv")',
        f"path = Path({str(work / 'submission.csv')!r})",
    )
    ns: dict = {}
    exec(compile(src, f"{strategy}_nb.py", "exec"), ns, ns)
    out = pd.read_csv(work / "submission.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    return ns, out, sample


def test_gold_rank_interact_notebook_and_synthetic_acl(tmp_path):
    nb = implement_notebook(tmp_path / "nb", "gold_rank_interact", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'gold_rank_interact'" in src
    assert "BLEND_W7" in src
    assert "INTERACT_LAM" in src
    assert "0.60" in src
    meta = json.loads((tmp_path / "nb" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert meta["id"] == "kaggle-user/slug"

    ns, out, sample = _run_strategy(tmp_path, "gold_rank_interact")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    high = out.iloc[:12]["ACL"].mean()
    low = out.iloc[12:]["ACL"].mean()
    assert high > low, (high, low)
    assert ns["used_learned"] is True
    assert ns["nI"] >= 20


def test_gold_rank_w50_notebook_and_synthetic_acl(tmp_path):
    nb = implement_notebook(tmp_path / "nb", "gold_rank_w50", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'gold_rank_w50'" in src
    assert '"gold_rank_w50": 0.50' in src
    assert "enable_internet" not in src
    meta = json.loads((tmp_path / "nb" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False

    ns, out, sample = _run_strategy(tmp_path, "gold_rank_w50")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    high = out.iloc[:12]["ACL"].mean()
    low = out.iloc[12:]["ACL"].mean()
    assert high > low, (high, low)
    assert ns["used_learned"] is True
    assert ns["nI"] >= 20


def test_gold_rank_w40_notebook_and_synthetic_acl(tmp_path):
    nb = implement_notebook(tmp_path / "nb", "gold_rank_w40", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'gold_rank_w40'" in src
    assert '"gold_rank_w40": 0.40' in src
    assert "enable_internet" not in src
    meta = json.loads((tmp_path / "nb" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False

    ns, out, sample = _run_strategy(tmp_path, "gold_rank_w40")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    high = out.iloc[:12]["ACL"].mean()
    low = out.iloc[12:]["ACL"].mean()
    assert high > low, (high, low)
    assert ns["used_learned"] is True
    assert ns["nI"] >= 20


def test_gold_rank_w70_notebook_and_synthetic_acl(tmp_path):
    nb = implement_notebook(tmp_path / "nb", "gold_rank_w70", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'gold_rank_w70'" in src
    assert '"gold_rank_w70": 0.70' in src
    assert "enable_internet" not in src
    meta = json.loads((tmp_path / "nb" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False

    ns, out, sample = _run_strategy(tmp_path, "gold_rank_w70")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    high = out.iloc[:12]["ACL"].mean()
    low = out.iloc[12:]["ACL"].mean()
    assert high > low, (high, low)
    assert ns["used_learned"] is True
    assert ns["nI"] >= 20


def test_gold_rank_lam2_notebook_and_synthetic_acl(tmp_path):
    nb = implement_notebook(tmp_path / "nb", "gold_rank_lam2", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'gold_rank_lam2'" in src
    assert '"gold_rank_lam2": 0.50' in src
    assert '"gold_rank_lam2": 2.0' in src
    assert "enable_internet" not in src
    meta = json.loads((tmp_path / "nb" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False

    ns, out, sample = _run_strategy(tmp_path, "gold_rank_lam2")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    high = out.iloc[:12]["ACL"].mean()
    low = out.iloc[12:]["ACL"].mean()
    assert high > low, (high, low)
    assert ns["used_learned"] is True
    assert ns["nI"] >= 20


def test_hypothesize_cycle0_is_lam2(tmp_path):
    dummy = tmp_path / "dummy.md"
    dummy.write_text("x")
    md = run_hypothesize(
        tmp_path / "Hypothesis",
        dummy,
        dummy,
        cycle_id="00_test",
        day_id="2026-09-20",
        cycle_num=0,
    )
    text = md.read_text()
    assert "H_gold_rank_lam2" in text
    assert "0.518" in text
    assert "λI=2.0" in text or "λ=2.0" in text or "2.0" in text
    assert "gold_rank_w50" in text
    assert "56382621" in text or "tied" in text.lower() or "0.40/0.60" in text
