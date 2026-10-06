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


def test_cycle0_plan_selects_weak_rank_bakers_llm(tmp_path):
    dummy = tmp_path / "dummy.md"
    dummy.write_text("x")
    md = run_plan(tmp_path, dummy, dummy, dummy, cycle_id="00_test", day_id="2026-09-24", cycle_num=0)
    assert "weak_rank_bakers_llm" in md.read_text()
    sidecar = json.loads((tmp_path / "2026-09-24_cycle00_test_plan.json").read_text())
    assert sidecar["strategy"] == "weak_rank_bakers_llm"


def test_cycle1_plan_selects_weak_rank_bakers_llm(tmp_path):
    dummy = tmp_path / "dummy.md"
    dummy.write_text("x")
    md = run_plan(tmp_path, dummy, dummy, dummy, cycle_id="01_test", day_id="2026-09-24", cycle_num=1)
    sidecar = json.loads((tmp_path / "2026-09-24_cycle01_test_plan.json").read_text())
    assert sidecar["strategy"] == "weak_rank_bakers_llm"
    assert "llm" in md.read_text().lower()


def test_cycle2_plan_keeps_weak_rank_bakers_mix_fallback(tmp_path):
    dummy = tmp_path / "dummy.md"
    dummy.write_text("x")
    sidecar_md = run_plan(tmp_path, dummy, dummy, dummy, cycle_id="02_test", day_id="2026-09-24", cycle_num=2)
    sidecar = json.loads((tmp_path / "2026-09-24_cycle02_test_plan.json").read_text())
    assert sidecar["strategy"] == "weak_rank_bakers_mix"
    assert "0.50" in sidecar_md.read_text()


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
        row = {
            STUDY_ID_COL: uid,
            "Report": (
                "Complete ACL tear with joint effusion."
                if high
                else "ACL is intact. No tear. Ligaments unremarkable."
            ),
        }
        for t in targets:
            if t == "ACL":
                row[t] = int(high)
            else:
                row[t] = int(rng.random() < 0.3)
        train_rows.append(row)
    for uid in unlab_ids:
        high = gold_frac[uid] > 0.5
        row = {
            STUDY_ID_COL: uid,
            "Report": (
                "Anterior cruciate ligament rupture."
                if high
                else "No ACL tear. Cruciate ligaments intact."
            ),
        }
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


def test_hypothesize_cycle0_is_weak_rank_bakers_llm(tmp_path):
    dummy = tmp_path / "dummy.md"
    dummy.write_text("x")
    md = run_hypothesize(
        tmp_path / "Hypothesis",
        dummy,
        dummy,
        cycle_id="00_test",
        day_id="2026-09-24",
        cycle_num=0,
    )
    text = md.read_text()
    assert "H_weak_rank_bakers_llm" in text
    assert "0.519" in text
    assert "weak_rank_bakers_mix" in text
    assert "llm" in text.lower()
    assert "56815942" in text


def test_weak_rank_calibrate_notebook_and_synthetic_acl(tmp_path):
    nb = implement_notebook(tmp_path / "nb", "weak_rank_calibrate", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_calibrate'" in src
    assert "parse_report_soft" in src
    assert "learned_scores_weak" in src
    assert "izlenmedi" in src
    assert "test.csv" not in src or "Never" in src or "never" in src
    assert "enable_internet" not in src
    meta = json.loads((tmp_path / "nb" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False

    ns, out, sample = _run_strategy(tmp_path, "weak_rank_calibrate")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    high = out.iloc[:12]["ACL"].mean()
    low = out.iloc[12:]["ACL"].mean()
    assert high > low, (high, low)
    assert ns["used_learned"] is True
    assert ns["n7"] >= 20


def test_weak_rank_goldfill_notebook_and_synthetic_acl(tmp_path):
    nb = implement_notebook(tmp_path / "nb_gf", "weak_rank_goldfill", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_goldfill'" in src
    assert "prefer_gold" in src
    meta = json.loads((tmp_path / "nb_gf" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False

    ns, out, sample = _run_strategy(tmp_path, "weak_rank_goldfill")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    high = out.iloc[:12]["ACL"].mean()
    low = out.iloc[12:]["ACL"].mean()
    assert high > low, (high, low)
    assert ns["used_learned"] is True
    assert ns["n7"] >= 20


def test_weak_rank_confident_notebook_and_synthetic_acl(tmp_path):
    nb = implement_notebook(tmp_path / "nb_cf", "weak_rank_confident", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_confident'" in src
    assert "confident_only" in src
    assert "parse_report_states" in src
    assert "prefer_gold" in src
    assert "enable_internet" not in src
    meta = json.loads((tmp_path / "nb_cf" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert meta["id"] == "kaggle-user/slug"

    ns, out, sample = _run_strategy(tmp_path, "weak_rank_confident")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    high = out.iloc[:12]["ACL"].mean()
    low = out.iloc[12:]["ACL"].mean()
    assert high > low, (high, low)
    assert ns["used_learned"] is True
    assert ns["n7"] >= 20
    states = ns["parse_report_states"]("Routine knee MRI without additional comment.", ["ACL"])
    assert states["ACL"] == "unmentioned"
    soft = ns["parse_report_soft"]("Complete ACL tear with joint effusion.", ["ACL"])
    assert soft["ACL"] == 0.85


def test_weak_rank_named_notebook_and_synthetic_acl(tmp_path):
    nb = implement_notebook(tmp_path / "nb_nm", "weak_rank_named", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_named'" in src
    assert "named_only" in src
    assert "NAMED_PARSER" in src
    assert "enable_internet" not in src
    meta = json.loads((tmp_path / "nb_nm" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False

    ns, out, sample = _run_strategy(tmp_path, "weak_rank_named")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    high = out.iloc[:12]["ACL"].mean()
    low = out.iloc[12:]["ACL"].mean()
    assert high > low, (high, low)
    assert ns["used_learned"] is True
    assert ns["n7"] >= 20
    assert "ACL" in ns["NAMED_PARSER"]
    assert "Effusion" not in ns["NAMED_PARSER"]


def test_weak_rank_named_mix_notebook_and_synthetic_acl(tmp_path):
    nb = implement_notebook(tmp_path / "nb_mix", "weak_rank_named_mix", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_named_mix'" in src
    assert "named_only" in src
    assert "NAMED_PARSER" in src
    assert "enable_internet" not in src
    meta = json.loads((tmp_path / "nb_mix" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False

    ns, out, sample = _run_strategy(tmp_path, "weak_rank_named_mix")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    high = out.iloc[:12]["ACL"].mean()
    low = out.iloc[12:]["ACL"].mean()
    assert high > low, (high, low)
    assert ns["used_learned"] is True
    assert "ACL" in ns["NAMED_PARSER"]


def test_weak_rank_bakers_mix_notebook_and_synthetic_acl(tmp_path):
    nb = implement_notebook(tmp_path / "nb_bk", "weak_rank_bakers_mix", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_mix'" in src
    assert "BAKERS_ONLY" in src
    assert "MIX_TARGETS" in src
    assert "enable_internet" not in src
    meta = json.loads((tmp_path / "nb_bk" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False

    ns, out, sample = _run_strategy(tmp_path, "weak_rank_bakers_mix")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    high = out.iloc[:12]["ACL"].mean()
    low = out.iloc[12:]["ACL"].mean()
    assert high > low, (high, low)
    assert ns["used_learned"] is True
    assert ns["MIX_TARGETS"]["weak_rank_bakers_mix"] == {"Baker's"}
    assert "ACL" not in ns["MIX_TARGETS"]["weak_rank_bakers_mix"]


def test_weak_rank_bakers_w40_notebook_and_synthetic_acl(tmp_path):
    nb = implement_notebook(tmp_path / "nb_w40", "weak_rank_bakers_w40", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_w40'" in src
    assert "MIX_GOLD_W" in src
    assert "enable_internet" not in src
    meta = json.loads((tmp_path / "nb_w40" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False

    ns, out, sample = _run_strategy(tmp_path, "weak_rank_bakers_w40")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    high = out.iloc[:12]["ACL"].mean()
    low = out.iloc[12:]["ACL"].mean()
    assert high > low, (high, low)
    assert ns["used_learned"] is True
    assert ns["MIX_GOLD_W"]["weak_rank_bakers_w40"] == 0.60
    assert ns["MIX_TARGETS"]["weak_rank_bakers_w40"] == {"Baker's"}


def test_weak_rank_bakers_w60_notebook_and_synthetic_acl(tmp_path):
    nb = implement_notebook(tmp_path / "nb_w60", "weak_rank_bakers_w60", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_w60'" in src
    assert "MIX_GOLD_W" in src
    assert "enable_internet" not in src
    meta = json.loads((tmp_path / "nb_w60" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False

    ns, out, sample = _run_strategy(tmp_path, "weak_rank_bakers_w60")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    high = out.iloc[:12]["ACL"].mean()
    low = out.iloc[12:]["ACL"].mean()
    assert high > low, (high, low)
    assert ns["used_learned"] is True
    assert ns["MIX_GOLD_W"]["weak_rank_bakers_w60"] == 0.40
    assert ns["MIX_TARGETS"]["weak_rank_bakers_w60"] == {"Baker's"}


def test_weak_rank_bakers_acl_mix_notebook_and_synthetic_acl(tmp_path):
    nb = implement_notebook(tmp_path / "nb_ba", "weak_rank_bakers_acl_mix", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_acl_mix'" in src
    assert "BAKERS_ACL" in src
    assert "MIX_TARGETS" in src
    assert "enable_internet" not in src
    meta = json.loads((tmp_path / "nb_ba" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False

    ns, out, sample = _run_strategy(tmp_path, "weak_rank_bakers_acl_mix")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    high = out.iloc[:12]["ACL"].mean()
    low = out.iloc[12:]["ACL"].mean()
    assert high > low, (high, low)
    assert ns["used_learned"] is True
    assert ns["MIX_GOLD_W"]["weak_rank_bakers_acl_mix"] == 0.50
    assert ns["MIX_TARGETS"]["weak_rank_bakers_acl_mix"] == {"Baker's", "ACL"}
    assert "MCL" not in ns["MIX_TARGETS"]["weak_rank_bakers_acl_mix"]


def test_weak_rank_bakers_mm_mix_notebook_and_synthetic_acl(tmp_path):
    nb = implement_notebook(tmp_path / "nb_mm", "weak_rank_bakers_mm_mix", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_mm_mix'" in src
    assert "BAKERS_MM" in src
    assert "NAMED_PLUS_MM" in src
    assert "MIX_TARGETS" in src
    assert "enable_internet" not in src
    meta = json.loads((tmp_path / "nb_mm" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False

    ns, out, sample = _run_strategy(tmp_path, "weak_rank_bakers_mm_mix")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    high = out.iloc[:12]["ACL"].mean()
    low = out.iloc[12:]["ACL"].mean()
    assert high > low, (high, low)
    assert ns["used_learned"] is True
    assert ns["MIX_GOLD_W"]["weak_rank_bakers_mm_mix"] == 0.50
    assert ns["MIX_TARGETS"]["weak_rank_bakers_mm_mix"] == {"Baker's", "Medial Meniscus"}
    assert "ACL" not in ns["MIX_TARGETS"]["weak_rank_bakers_mm_mix"]
    assert "MCL" not in ns["MIX_TARGETS"]["weak_rank_bakers_mm_mix"]
    assert "Medial Meniscus" in ns["NAMED_PLUS_MM"]


def test_weak_rank_bakers_goldstd_notebook_and_synthetic_acl(tmp_path):
    nb = implement_notebook(tmp_path / "nb_gs", "weak_rank_bakers_goldstd", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_goldstd'" in src
    assert "std_on_gold" in src
    assert "MIX_TARGETS" in src
    assert "enable_internet" not in src
    meta = json.loads((tmp_path / "nb_gs" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False

    ns, out, sample = _run_strategy(tmp_path, "weak_rank_bakers_goldstd")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    high = out.iloc[:12]["ACL"].mean()
    low = out.iloc[12:]["ACL"].mean()
    assert high > low, (high, low)
    assert ns["used_learned"] is True
    assert ns["MIX_GOLD_W"]["weak_rank_bakers_goldstd"] == 0.50
    assert ns["MIX_TARGETS"]["weak_rank_bakers_goldstd"] == {"Baker's"}
    assert "Medial Meniscus" not in ns["MIX_TARGETS"]["weak_rank_bakers_goldstd"]
    assert "ACL" not in ns["MIX_TARGETS"]["weak_rank_bakers_goldstd"]


def test_weak_rank_bakers_silence_notebook_and_synthetic_acl(tmp_path):
    nb = implement_notebook(tmp_path / "nb_sil", "weak_rank_bakers_silence", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_silence'" in src
    assert "silence_neg_targets" in src
    assert "MIX_TARGETS" in src
    assert "enable_internet" not in src
    meta = json.loads((tmp_path / "nb_sil" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False

    ns, out, sample = _run_strategy(tmp_path, "weak_rank_bakers_silence")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    high = out.iloc[:12]["ACL"].mean()
    low = out.iloc[12:]["ACL"].mean()
    assert high > low, (high, low)
    assert ns["used_learned"] is True
    assert ns["MIX_GOLD_W"]["weak_rank_bakers_silence"] == 0.50
    assert ns["MIX_TARGETS"]["weak_rank_bakers_silence"] == {"Baker's"}
    assert "Medial Meniscus" not in ns["MIX_TARGETS"]["weak_rank_bakers_silence"]
    assert "ACL" not in ns["MIX_TARGETS"]["weak_rank_bakers_silence"]


def test_weak_rank_bakers_dropfat_notebook_and_synthetic_acl(tmp_path):
    nb = implement_notebook(tmp_path / "nb_df", "weak_rank_bakers_dropfat", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_dropfat'" in src
    assert "drop_fat" in src
    assert "MIX_TARGETS" in src
    assert "enable_internet" not in src
    meta = json.loads((tmp_path / "nb_df" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False

    ns, out, sample = _run_strategy(tmp_path, "weak_rank_bakers_dropfat")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    high = out.iloc[:12]["ACL"].mean()
    low = out.iloc[12:]["ACL"].mean()
    assert high > low, (high, low)
    assert ns["used_learned"] is True
    assert ns["MIX_GOLD_W"]["weak_rank_bakers_dropfat"] == 0.50
    assert ns["MIX_TARGETS"]["weak_rank_bakers_dropfat"] == {"Baker's"}
    assert "Medial Meniscus" not in ns["MIX_TARGETS"]["weak_rank_bakers_dropfat"]
    assert "ACL" not in ns["MIX_TARGETS"]["weak_rank_bakers_dropfat"]
    default7, _ = ns["feat_map"](ns["train_series"], interact=False, drop_fat=True)
    default13, _ = ns["feat_map"](ns["train_series"], interact=True, drop_fat=True)
    any_uid = next(iter(default7))
    assert default7[any_uid].shape[0] == 6
    assert default13[any_uid].shape[0] == 9


def test_weak_rank_bakers_llm_notebook_and_synthetic_acl(tmp_path):
    nb = implement_notebook(tmp_path / "nb_llm", "weak_rank_bakers_llm", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_llm'" in src
    assert "discover_llm_labels" in src
    assert "learned_scores_llm" in src
    assert "llm_labels_v4_blend.csv" in src
    assert "datasets" in src and "stevenleehans" in src
    assert "llm csv explicit" in src
    assert "enable_internet" not in src
    meta = json.loads((tmp_path / "nb_llm" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "stevenleehans/rsna-knee-llm-report-labels" in meta["dataset_sources"]

    data = tmp_path / "input"
    work = tmp_path / "work"
    work.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    sag = series.groupby(STUDY_ID_COL)["Anatomical_Plane"].apply(lambda s: (s == "Sagittal").mean())
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["Baker's"] = sag.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)

    src = _source_for_strategy("weak_rank_bakers_llm", "00_test")
    src = src.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
    src = src.replace(
        'path = Path("/kaggle/working/submission.csv")',
        f"path = Path({str(work / 'submission.csv')!r})",
    )
    ns: dict = {}
    exec(compile(src, "weak_rank_bakers_llm_nb.py", "exec"), ns, ns)
    out = pd.read_csv(work / "submission.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    high = out.iloc[:12]["ACL"].mean()
    low = out.iloc[12:]["ACL"].mean()
    assert high > low, (high, low)
    assert ns["used_learned"] is True
    assert ns["MIX_GOLD_W"]["weak_rank_bakers_llm"] == 0.50
    assert ns["MIX_TARGETS"]["weak_rank_bakers_llm"] == {"Baker's"}
    assert "Medial Meniscus" not in ns["MIX_TARGETS"]["weak_rank_bakers_llm"]
    assert "ACL" not in ns["MIX_TARGETS"]["weak_rank_bakers_llm"]


def test_weak_rank_bakers_llm_finds_kaggle_datasets_layout(tmp_path):
    """2026-10-05 kernel missed /kaggle/input/datasets/<owner>/<slug>/csv (n7w=0)."""
    data = tmp_path / "input"
    work = tmp_path / "work"
    work.mkdir()
    _write_synth(data)
    nested = data / "datasets" / "stevenleehans" / "rsna-knee-llm-report-labels"
    nested.mkdir(parents=True)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    sag = series.groupby(STUDY_ID_COL)["Anatomical_Plane"].apply(lambda s: (s == "Sagittal").mean())
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["Baker's"] = sag.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm.to_csv(nested / "llm_labels_v4_blend.csv", index=False)

    src = _source_for_strategy("weak_rank_bakers_llm", "00_test")
    src = src.replace('inp = Path("/kaggle/input")', f"inp = Path({str(data)!r})")
    src = src.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
    src = src.replace(
        'path = Path("/kaggle/working/submission.csv")',
        f"path = Path({str(work / 'submission.csv')!r})",
    )
    ns: dict = {}
    exec(compile(src, "weak_rank_bakers_llm_nested.py", "exec"), ns, ns)
    found = ns["discover_llm_labels"]()
    assert found is not None
    assert found.name == "llm_labels_v4_blend.csv"
    out = pd.read_csv(work / "submission.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert ns["used_learned"] is True


def test_weak_rank_bakers_llm_missing_csv_falls_back_to_parser_mix(tmp_path):
    data = tmp_path / "input"
    work = tmp_path / "work"
    work.mkdir()
    _write_synth(data)
    src = _source_for_strategy("weak_rank_bakers_llm", "00_test")
    src = src.replace('inp = Path("/kaggle/input")', f"inp = Path({str(data)!r})")
    src = src.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
    src = src.replace(
        'path = Path("/kaggle/working/submission.csv")',
        f"path = Path({str(work / 'submission.csv')!r})",
    )
    ns: dict = {}
    exec(compile(src, "weak_rank_bakers_llm_fallback.py", "exec"), ns, ns)
    assert ns["discover_llm_labels"]() is None
    out = pd.read_csv(work / "submission.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    assert ns["used_learned"] is True
