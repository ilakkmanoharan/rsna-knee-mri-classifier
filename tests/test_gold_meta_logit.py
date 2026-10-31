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


def test_cycle0_plan_selects_weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem(tmp_path):
    dummy = tmp_path / "dummy.md"
    dummy.write_text("x")
    md = run_plan(tmp_path, dummy, dummy, dummy, cycle_id="00_test", day_id="2026-09-24", cycle_num=0)
    assert "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem" in md.read_text()
    sidecar = json.loads((tmp_path / "2026-09-24_cycle00_test_plan.json").read_text())
    assert sidecar["strategy"] == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem"


def test_cycle1_plan_selects_weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem(tmp_path):
    dummy = tmp_path / "dummy.md"
    dummy.write_text("x")
    md = run_plan(tmp_path, dummy, dummy, dummy, cycle_id="01_test", day_id="2026-09-24", cycle_num=1)
    sidecar = json.loads((tmp_path / "2026-09-24_cycle01_test_plan.json").read_text())
    assert sidecar["strategy"] == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem"
    assert "llm" in md.read_text().lower()


def test_cycle2_plan_keeps_acl_gem_eff_gem_lm_gem_bak_gem_not_gold_meta_logit(tmp_path):
    dummy = tmp_path / "dummy.md"
    dummy.write_text("x")
    sidecar_md = run_plan(tmp_path, dummy, dummy, dummy, cycle_id="02_test", day_id="2026-09-24", cycle_num=2)
    sidecar = json.loads((tmp_path / "2026-09-24_cycle02_test_plan.json").read_text())
    assert sidecar["strategy"] == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem"
    assert "0.534" in sidecar_md.read_text()


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


def test_hypothesize_cycle0_is_weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem(tmp_path):
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
    assert "H_weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem" in text
    assert "`H_weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem` ← primary" in text
    assert "H_weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem" in text
    assert "H_weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40" in text
    assert "H_weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60" in text
    assert "0.534" in text
    assert "0.533" in text
    assert "0.532" in text
    assert "0.531" in text
    assert "0.530" in text
    assert "0.529" in text
    assert "ACL" in text
    assert "Effusion" in text
    assert "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem" in text
    assert "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40" in text
    assert "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60" in text
    assert "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem" in text
    assert "Lateral Meniscus" in text
    assert "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40" in text
    assert "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60" in text
    assert "weak_rank_bakers_frac_hyb_lm_eff_acl_gem" in text
    assert "weak_rank_bakers_frac_hyb_lm_eff_acl_w40" in text
    assert "weak_rank_bakers_frac_hyb_lm_eff_acl_w60" in text
    assert "weak_rank_bakers_frac_hyb_lm_eff_w40" in text
    assert "weak_rank_bakers_frac_hyb_lm_eff_w60" in text
    assert "weak_rank_bakers_frac_hyb_lm_llm" in text
    assert "llm" in text.lower()
    assert "57804837" in text
    assert "57802682" in text
    assert "57800238" in text
    assert "57797758" in text
    assert "57770241" in text
    assert "57768062" in text
    assert "57765842" in text
    assert "57760822" in text
    assert "57726811" in text
    assert "57691260" in text


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


def test_weak_rank_bakers_acl_llm_mixes_parser_bakers_and_llm_acl(tmp_path):
    nb = implement_notebook(tmp_path / "nb_acl_llm", "weak_rank_bakers_acl_llm", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_acl_llm'" in src
    assert "mixed LLM ACL" in src
    assert "skip constant LLM ACL" in src
    meta = json.loads((tmp_path / "nb_acl_llm" / "kernel-metadata.json").read_text())
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
    llm["ACL"] = sag.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)

    src = _source_for_strategy("weak_rank_bakers_acl_llm", "00_test")
    src = src.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
    src = src.replace(
        'path = Path("/kaggle/working/submission.csv")',
        f"path = Path({str(work / 'submission.csv')!r})",
    )
    ns: dict = {}
    exec(compile(src, "weak_rank_bakers_acl_llm_nb.py", "exec"), ns, ns)
    out = pd.read_csv(work / "submission.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    high = out.iloc[:12]["ACL"].mean()
    low = out.iloc[12:]["ACL"].mean()
    assert high > low, (high, low)
    assert ns["used_learned"] is True
    assert ns["MIX_TARGETS"]["weak_rank_bakers_acl_llm"] == {"Baker's"}


def test_weak_rank_bakers_mcl_llm_mixes_parser_bakers_llm_acl_and_llm_mcl(tmp_path):
    nb = implement_notebook(tmp_path / "nb_mcl_llm", "weak_rank_bakers_mcl_llm", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_mcl_llm'" in src
    assert "mixed LLM MCL" in src
    assert "skip constant LLM MCL" in src
    meta = json.loads((tmp_path / "nb_mcl_llm" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "stevenleehans/rsna-knee-llm-report-labels" in meta["dataset_sources"]

    data = tmp_path / "input"
    work = tmp_path / "work"
    work.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    sag = series.groupby(STUDY_ID_COL)["Anatomical_Plane"].apply(lambda s: (s == "Sagittal").mean())
    cor = series.groupby(STUDY_ID_COL)["Anatomical_Plane"].apply(lambda s: (s == "Coronal").mean())
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = sag.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["MCL"] = cor.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)

    src = _source_for_strategy("weak_rank_bakers_mcl_llm", "00_test")
    src = src.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
    src = src.replace(
        'path = Path("/kaggle/working/submission.csv")',
        f"path = Path({str(work / 'submission.csv')!r})",
    )
    ns: dict = {}
    exec(compile(src, "weak_rank_bakers_mcl_llm_nb.py", "exec"), ns, ns)
    out = pd.read_csv(work / "submission.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    acl_high = out.iloc[:12]["ACL"].mean()
    acl_low = out.iloc[12:]["ACL"].mean()
    assert acl_high > acl_low, (acl_high, acl_low)
    mcl_high = out.iloc[12:]["MCL"].mean()
    mcl_low = out.iloc[:12]["MCL"].mean()
    assert mcl_high > mcl_low, (mcl_high, mcl_low)
    assert ns["used_learned"] is True
    assert ns["MIX_TARGETS"]["weak_rank_bakers_mcl_llm"] == {"Baker's"}


def test_weak_rank_bakers_eff_llm_mixes_parser_bakers_llm_acl_and_llm_effusion(tmp_path):
    nb = implement_notebook(tmp_path / "nb_eff_llm", "weak_rank_bakers_eff_llm", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_eff_llm'" in src
    assert "mixed LLM Effusion" in src
    assert "skip constant LLM Effusion" in src
    meta = json.loads((tmp_path / "nb_eff_llm" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "stevenleehans/rsna-knee-llm-report-labels" in meta["dataset_sources"]

    data = tmp_path / "input"
    work = tmp_path / "work"
    work.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    sag = series.groupby(STUDY_ID_COL)["Anatomical_Plane"].apply(lambda s: (s == "Sagittal").mean())
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = sag.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)

    src = _source_for_strategy("weak_rank_bakers_eff_llm", "00_test")
    src = src.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
    src = src.replace(
        'path = Path("/kaggle/working/submission.csv")',
        f"path = Path({str(work / 'submission.csv')!r})",
    )
    ns: dict = {}
    exec(compile(src, "weak_rank_bakers_eff_llm_nb.py", "exec"), ns, ns)
    out = pd.read_csv(work / "submission.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    acl_high = out.iloc[:12]["ACL"].mean()
    acl_low = out.iloc[12:]["ACL"].mean()
    assert acl_high > acl_low, (acl_high, acl_low)
    assert float(out["Effusion"].std()) > 1e-6
    assert float(out["Effusion"].max()) > float(out["Effusion"].min())
    assert ns["used_learned"] is True
    assert ns["MIX_TARGETS"]["weak_rank_bakers_eff_llm"] == {"Baker's"}


def test_weak_rank_bakers_syn_llm_mixes_parser_bakers_llm_acl_effusion_and_synovitis(tmp_path):
    nb = implement_notebook(tmp_path / "nb_syn_llm", "weak_rank_bakers_syn_llm", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_syn_llm'" in src
    assert "mixed LLM Synovitis" in src
    assert "skip constant LLM Synovitis" in src
    meta = json.loads((tmp_path / "nb_syn_llm" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "stevenleehans/rsna-knee-llm-report-labels" in meta["dataset_sources"]

    data = tmp_path / "input"
    work = tmp_path / "work"
    work.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    sag = series.groupby(STUDY_ID_COL)["Anatomical_Plane"].apply(lambda s: (s == "Sagittal").mean())
    ax = series.groupby(STUDY_ID_COL)["Anatomical_Plane"].apply(lambda s: (s == "Axial").mean())
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = sag.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Synovitis"] = ax.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)

    src = _source_for_strategy("weak_rank_bakers_syn_llm", "00_test")
    src = src.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
    src = src.replace(
        'path = Path("/kaggle/working/submission.csv")',
        f"path = Path({str(work / 'submission.csv')!r})",
    )
    ns: dict = {}
    exec(compile(src, "weak_rank_bakers_syn_llm_nb.py", "exec"), ns, ns)
    out = pd.read_csv(work / "submission.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    acl_high = out.iloc[:12]["ACL"].mean()
    acl_low = out.iloc[12:]["ACL"].mean()
    assert acl_high > acl_low, (acl_high, acl_low)
    assert float(out["Effusion"].std()) > 1e-6
    assert float(out["Synovitis"].std()) > 1e-6
    assert float(out["Synovitis"].max()) > float(out["Synovitis"].min())
    assert ns["used_learned"] is True
    assert ns["MIX_TARGETS"]["weak_rank_bakers_syn_llm"] == {"Baker's"}


def test_weak_rank_bakers_syn_from_eff_mixes_effusion_ranks_onto_synovitis(tmp_path):
    nb = implement_notebook(tmp_path / "nb_syn_from_eff", "weak_rank_bakers_syn_from_eff", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_syn_from_eff'" in src
    assert "mixed effusion ranks onto Synovitis" in src
    assert "skip flatten effusion→Synovitis" in src
    assert 'target_name="Synovitis"' not in src or "Do NOT use the Synovitis LLM field" in src
    meta = json.loads((tmp_path / "nb_syn_from_eff" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "stevenleehans/rsna-knee-llm-report-labels" in meta["dataset_sources"]

    data = tmp_path / "input"
    work = tmp_path / "work"
    work.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    sag = series.groupby(STUDY_ID_COL)["Anatomical_Plane"].apply(lambda s: (s == "Sagittal").mean())
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = sag.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    # Poisoned Synovitis field must be ignored; constant on purpose.
    llm["Synovitis"] = 0.5
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)

    src = _source_for_strategy("weak_rank_bakers_syn_from_eff", "00_test")
    src = src.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
    src = src.replace(
        'path = Path("/kaggle/working/submission.csv")',
        f"path = Path({str(work / 'submission.csv')!r})",
    )
    ns: dict = {}
    exec(compile(src, "weak_rank_bakers_syn_from_eff_nb.py", "exec"), ns, ns)
    out = pd.read_csv(work / "submission.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    acl_high = out.iloc[:12]["ACL"].mean()
    acl_low = out.iloc[12:]["ACL"].mean()
    assert acl_high > acl_low, (acl_high, acl_low)
    assert float(out["Effusion"].std()) > 1e-6
    assert float(out["Synovitis"].std()) > 1e-6
    assert float(out["Synovitis"].max()) > float(out["Synovitis"].min())
    assert ns["used_learned"] is True
    assert ns["MIX_TARGETS"]["weak_rank_bakers_syn_from_eff"] == {"Baker's"}


def test_weak_rank_bakers_cont_llm_mixes_parser_bakers_llm_acl_effusion_and_contusion(tmp_path):
    nb = implement_notebook(tmp_path / "nb_cont_llm", "weak_rank_bakers_cont_llm", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_cont_llm'" in src
    assert "mixed LLM Contusion" in src
    assert "skip constant LLM Contusion" in src
    assert 'target_name="Synovitis"' not in src or "Do NOT use the Synovitis LLM field" in src
    meta = json.loads((tmp_path / "nb_cont_llm" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "stevenleehans/rsna-knee-llm-report-labels" in meta["dataset_sources"]

    data = tmp_path / "input"
    work = tmp_path / "work"
    work.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    sag = series.groupby(STUDY_ID_COL)["Anatomical_Plane"].apply(lambda s: (s == "Sagittal").mean())
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = sag.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Contusion"] = (0.4 * sag + 0.6 * fluid).reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Synovitis"] = 0.5
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)

    src = _source_for_strategy("weak_rank_bakers_cont_llm", "00_test")
    src = src.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
    src = src.replace(
        'path = Path("/kaggle/working/submission.csv")',
        f"path = Path({str(work / 'submission.csv')!r})",
    )
    ns: dict = {}
    exec(compile(src, "weak_rank_bakers_cont_llm_nb.py", "exec"), ns, ns)
    out = pd.read_csv(work / "submission.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    acl_high = out.iloc[:12]["ACL"].mean()
    acl_low = out.iloc[12:]["ACL"].mean()
    assert acl_high > acl_low, (acl_high, acl_low)
    assert float(out["Effusion"].std()) > 1e-6
    assert float(out["Contusion"].std()) > 1e-6
    assert float(out["Contusion"].max()) > float(out["Contusion"].min())
    assert ns["used_learned"] is True
    assert ns["MIX_TARGETS"]["weak_rank_bakers_cont_llm"] == {"Baker's"}


def test_weak_rank_bakers_frac_llm_mixes_parser_bakers_llm_acl_effusion_and_fracture(tmp_path):
    nb = implement_notebook(tmp_path / "nb_frac_llm", "weak_rank_bakers_frac_llm", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_frac_llm'" in src
    assert "mixed LLM Fracture" in src
    assert "skip constant LLM Fracture" in src
    assert 'target_name="Fracture"' in src
    meta = json.loads((tmp_path / "nb_frac_llm" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "stevenleehans/rsna-knee-llm-report-labels" in meta["dataset_sources"]

    data = tmp_path / "input"
    work = tmp_path / "work"
    work.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    sag = series.groupby(STUDY_ID_COL)["Anatomical_Plane"].apply(lambda s: (s == "Sagittal").mean())
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = sag.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Fracture"] = (0.5 * sag + 0.5 * fluid).reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Contusion"] = 0.5
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)

    src = _source_for_strategy("weak_rank_bakers_frac_llm", "00_test")
    src = src.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
    src = src.replace(
        'path = Path("/kaggle/working/submission.csv")',
        f"path = Path({str(work / 'submission.csv')!r})",
    )
    ns: dict = {}
    exec(compile(src, "weak_rank_bakers_frac_llm_nb.py", "exec"), ns, ns)
    out = pd.read_csv(work / "submission.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    acl_high = out.iloc[:12]["ACL"].mean()
    acl_low = out.iloc[12:]["ACL"].mean()
    assert acl_high > acl_low, (acl_high, acl_low)
    assert float(out["Effusion"].std()) > 1e-6
    assert float(out["Fracture"].std()) > 1e-6
    assert float(out["Fracture"].max()) > float(out["Fracture"].min())
    assert ns["used_learned"] is True
    assert ns["MIX_TARGETS"]["weak_rank_bakers_frac_llm"] == {"Baker's"}


def test_weak_rank_bakers_moa_llm_mixes_parser_bakers_llm_acl_effusion_and_medial_oa(tmp_path):
    nb = implement_notebook(tmp_path / "nb_moa_llm", "weak_rank_bakers_moa_llm", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_moa_llm'" in src
    assert "mixed LLM Medial OA" in src
    assert "skip constant LLM Medial OA" in src
    assert 'target_name="Medial OA"' in src
    meta = json.loads((tmp_path / "nb_moa_llm" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "stevenleehans/rsna-knee-llm-report-labels" in meta["dataset_sources"]

    data = tmp_path / "input"
    work = tmp_path / "work"
    work.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    sag = series.groupby(STUDY_ID_COL)["Anatomical_Plane"].apply(lambda s: (s == "Sagittal").mean())
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = sag.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Medial OA"] = (0.5 * sag + 0.5 * fluid).reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Fracture"] = 0.5
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)

    src = _source_for_strategy("weak_rank_bakers_moa_llm", "00_test")
    src = src.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
    src = src.replace(
        'path = Path("/kaggle/working/submission.csv")',
        f"path = Path({str(work / 'submission.csv')!r})",
    )
    ns: dict = {}
    exec(compile(src, "weak_rank_bakers_moa_llm_nb.py", "exec"), ns, ns)
    out = pd.read_csv(work / "submission.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    acl_high = out.iloc[:12]["ACL"].mean()
    acl_low = out.iloc[12:]["ACL"].mean()
    assert acl_high > acl_low, (acl_high, acl_low)
    assert float(out["Effusion"].std()) > 1e-6
    assert float(out["Medial OA"].std()) > 1e-6
    assert float(out["Medial OA"].max()) > float(out["Medial OA"].min())
    assert ns["used_learned"] is True
    assert ns["MIX_TARGETS"]["weak_rank_bakers_moa_llm"] == {"Baker's"}


def test_weak_rank_bakers_loa_llm_mixes_parser_bakers_llm_acl_effusion_and_lateral_oa(tmp_path):
    nb = implement_notebook(tmp_path / "nb_loa_llm", "weak_rank_bakers_loa_llm", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_loa_llm'" in src
    assert "mixed LLM Lateral OA" in src
    assert "skip constant LLM Lateral OA" in src
    assert 'target_name="Lateral OA"' in src
    meta = json.loads((tmp_path / "nb_loa_llm" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "stevenleehans/rsna-knee-llm-report-labels" in meta["dataset_sources"]

    data = tmp_path / "input"
    work = tmp_path / "work"
    work.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    sag = series.groupby(STUDY_ID_COL)["Anatomical_Plane"].apply(lambda s: (s == "Sagittal").mean())
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = sag.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Lateral OA"] = (0.5 * sag + 0.5 * fluid).reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Medial OA"] = 0.5
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)

    src = _source_for_strategy("weak_rank_bakers_loa_llm", "00_test")
    src = src.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
    src = src.replace(
        'path = Path("/kaggle/working/submission.csv")',
        f"path = Path({str(work / 'submission.csv')!r})",
    )
    ns: dict = {}
    exec(compile(src, "weak_rank_bakers_loa_llm_nb.py", "exec"), ns, ns)
    out = pd.read_csv(work / "submission.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    acl_high = out.iloc[:12]["ACL"].mean()
    acl_low = out.iloc[12:]["ACL"].mean()
    assert acl_high > acl_low, (acl_high, acl_low)
    assert float(out["Effusion"].std()) > 1e-6
    assert float(out["Lateral OA"].std()) > 1e-6
    assert float(out["Lateral OA"].max()) > float(out["Lateral OA"].min())
    assert ns["used_learned"] is True
    assert ns["MIX_TARGETS"]["weak_rank_bakers_loa_llm"] == {"Baker's"}


def test_weak_rank_bakers_pfoa_llm_mixes_parser_bakers_llm_acl_effusion_and_pf_oa(tmp_path):
    nb = implement_notebook(tmp_path / "nb_pfoa_llm", "weak_rank_bakers_pfoa_llm", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_pfoa_llm'" in src
    assert "mixed LLM PF OA" in src
    assert "skip constant LLM PF OA" in src
    assert 'target_name="PF OA"' in src
    meta = json.loads((tmp_path / "nb_pfoa_llm" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "stevenleehans/rsna-knee-llm-report-labels" in meta["dataset_sources"]

    data = tmp_path / "input"
    work = tmp_path / "work"
    work.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    sag = series.groupby(STUDY_ID_COL)["Anatomical_Plane"].apply(lambda s: (s == "Sagittal").mean())
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = sag.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["PF OA"] = (0.5 * sag + 0.5 * fluid).reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Lateral OA"] = 0.5
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)

    src = _source_for_strategy("weak_rank_bakers_pfoa_llm", "00_test")
    src = src.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
    src = src.replace(
        'path = Path("/kaggle/working/submission.csv")',
        f"path = Path({str(work / 'submission.csv')!r})",
    )
    ns: dict = {}
    exec(compile(src, "weak_rank_bakers_pfoa_llm_nb.py", "exec"), ns, ns)
    out = pd.read_csv(work / "submission.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    acl_high = out.iloc[:12]["ACL"].mean()
    acl_low = out.iloc[12:]["ACL"].mean()
    assert acl_high > acl_low, (acl_high, acl_low)
    assert float(out["Effusion"].std()) > 1e-6
    assert float(out["PF OA"].std()) > 1e-6
    assert float(out["PF OA"].max()) > float(out["PF OA"].min())
    assert ns["used_learned"] is True
    assert ns["MIX_TARGETS"]["weak_rank_bakers_pfoa_llm"] == {"Baker's"}


def test_weak_rank_bakers_frac_hyb_llm_mixes_parser_bakers_llm_acl_effusion_and_hybrid_fracture(tmp_path):
    nb = implement_notebook(tmp_path / "nb_frac_hyb", "weak_rank_bakers_frac_hyb_llm", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_frac_hyb_llm'" in src
    assert "mixed hybrid Fracture" in src
    assert "skip constant hybrid Fracture" in src
    assert "discover_hybrid_labels" in src
    assert 'target_name="Fracture"' in src
    meta = json.loads((tmp_path / "nb_frac_hyb" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "stevenleehans/rsna-knee-llm-report-labels" in meta["dataset_sources"]
    assert "flight0234/rsna-knee-hybrid-report-labels" in meta["dataset_sources"]

    data = tmp_path / "input"
    work = tmp_path / "work"
    work.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    sag = series.groupby(STUDY_ID_COL)["Anatomical_Plane"].apply(lambda s: (s == "Sagittal").mean())
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = sag.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Fracture"] = 0.5
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)
    hyb = llm.copy()
    hyb["Fracture"] = (0.5 * sag + 0.5 * fluid).reindex(hyb[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    hyb.to_csv(data / "report_labels_v4hybrid.csv", index=False)

    src = _source_for_strategy("weak_rank_bakers_frac_hyb_llm", "00_test")
    src = src.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
    src = src.replace(
        'path = Path("/kaggle/working/submission.csv")',
        f"path = Path({str(work / 'submission.csv')!r})",
    )
    ns: dict = {}
    exec(compile(src, "weak_rank_bakers_frac_hyb_llm_nb.py", "exec"), ns, ns)
    out = pd.read_csv(work / "submission.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out[DEFAULT_TARGETS].isna().any().any() == False
    acl_high = out.iloc[:12]["ACL"].mean()
    acl_low = out.iloc[12:]["ACL"].mean()
    assert acl_high > acl_low, (acl_high, acl_low)
    assert float(out["Effusion"].std()) > 1e-6
    assert float(out["Fracture"].std()) > 1e-6
    assert float(out["Fracture"].max()) > float(out["Fracture"].min())
    assert ns["used_learned"] is True
    assert ns["MIX_TARGETS"]["weak_rank_bakers_frac_hyb_llm"] == {"Baker's"}


def test_weak_rank_bakers_frac_hyb_w60_uses_majority_hybrid_fracture_weight(tmp_path):
    nb = implement_notebook(tmp_path / "nb_frac_hyb_w60", "weak_rank_bakers_frac_hyb_w60", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_frac_hyb_w60'" in src
    assert 'w_gold = 0.40 if STRATEGY == "weak_rank_bakers_frac_hyb_w60" else 0.60 if STRATEGY == "weak_rank_bakers_frac_hyb_w40" else 0.50' in src
    assert "mixed hybrid Fracture" in src
    meta = json.loads((tmp_path / "nb_frac_hyb_w60" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "stevenleehans/rsna-knee-llm-report-labels" in meta["dataset_sources"]
    assert "flight0234/rsna-knee-hybrid-report-labels" in meta["dataset_sources"]

    data = tmp_path / "input"
    work50 = tmp_path / "work50"
    work60 = tmp_path / "work60"
    work50.mkdir()
    work60.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    sag = series.groupby(STUDY_ID_COL)["Anatomical_Plane"].apply(lambda s: (s == "Sagittal").mean())
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = sag.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Fracture"] = 0.5
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)
    hyb = llm.copy()
    # Invert sagittal so hybrid Fracture ranks disagree with gold and mix-weight matters.
    hyb["Fracture"] = (1.0 - sag.reindex(hyb[STUDY_ID_COL]).fillna(0.5)).clip(0.05, 0.95).to_numpy()
    hyb.to_csv(data / "report_labels_v4hybrid.csv", index=False)

    def _run(strategy: str, work: Path) -> pd.DataFrame:
        src = _source_for_strategy(strategy, "00_test")
        src = src.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
        src = src.replace(
            'path = Path("/kaggle/working/submission.csv")',
            f"path = Path({str(work / 'submission.csv')!r})",
        )
        ns: dict = {}
        exec(compile(src, f"{strategy}_nb.py", "exec"), ns, ns)
        return pd.read_csv(work / "submission.csv"), ns

    out50, _ = _run("weak_rank_bakers_frac_hyb_llm", work50)
    out60, ns60 = _run("weak_rank_bakers_frac_hyb_w60", work60)
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out60[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out60[DEFAULT_TARGETS].isna().any().any() == False
    assert float(out60["Fracture"].std()) > 1e-6
    assert not np.allclose(out50["Fracture"].to_numpy(), out60["Fracture"].to_numpy())
    assert ns60["MIX_TARGETS"]["weak_rank_bakers_frac_hyb_w60"] == {"Baker's"}


def test_weak_rank_bakers_frac_hyb_w40_uses_minority_hybrid_fracture_weight(tmp_path):
    nb = implement_notebook(tmp_path / "nb_frac_hyb_w40", "weak_rank_bakers_frac_hyb_w40", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_frac_hyb_w40'" in src
    assert 'w_gold = 0.40 if STRATEGY == "weak_rank_bakers_frac_hyb_w60" else 0.60 if STRATEGY == "weak_rank_bakers_frac_hyb_w40" else 0.50' in src
    assert "mixed hybrid Fracture" in src
    meta = json.loads((tmp_path / "nb_frac_hyb_w40" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "stevenleehans/rsna-knee-llm-report-labels" in meta["dataset_sources"]
    assert "flight0234/rsna-knee-hybrid-report-labels" in meta["dataset_sources"]

    data = tmp_path / "input"
    work50 = tmp_path / "work50"
    work40 = tmp_path / "work40"
    work50.mkdir()
    work40.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    sag = series.groupby(STUDY_ID_COL)["Anatomical_Plane"].apply(lambda s: (s == "Sagittal").mean())
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = sag.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Fracture"] = 0.5
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)
    hyb = llm.copy()
    hyb["Fracture"] = (1.0 - sag.reindex(hyb[STUDY_ID_COL]).fillna(0.5)).clip(0.05, 0.95).to_numpy()
    hyb.to_csv(data / "report_labels_v4hybrid.csv", index=False)

    def _run(strategy: str, work: Path) -> pd.DataFrame:
        src = _source_for_strategy(strategy, "00_test")
        src = src.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
        src = src.replace(
            'path = Path("/kaggle/working/submission.csv")',
            f"path = Path({str(work / 'submission.csv')!r})",
        )
        ns: dict = {}
        exec(compile(src, f"{strategy}_nb.py", "exec"), ns, ns)
        return pd.read_csv(work / "submission.csv"), ns

    out50, _ = _run("weak_rank_bakers_frac_hyb_llm", work50)
    out40, ns40 = _run("weak_rank_bakers_frac_hyb_w40", work40)
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out40[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out40[DEFAULT_TARGETS].isna().any().any() == False
    assert float(out40["Fracture"].std()) > 1e-6
    assert not np.allclose(out50["Fracture"].to_numpy(), out40["Fracture"].to_numpy())
    assert ns40["MIX_TARGETS"]["weak_rank_bakers_frac_hyb_w40"] == {"Baker's"}


def test_weak_rank_bakers_frac_gem_llm_uses_gemini_fracture(tmp_path):
    nb = implement_notebook(tmp_path / "nb_frac_gem", "weak_rank_bakers_frac_gem_llm", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_frac_gem_llm'" in src
    assert "mixed Gemini Fracture" in src
    assert "discover_gemini_labels" in src
    assert "report_labels_v5.csv" in src
    meta = json.loads((tmp_path / "nb_frac_gem" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "stevenleehans/rsna-knee-llm-report-labels" in meta["dataset_sources"]
    assert "yunusgmsoy/rsna-knee-llm-labels-4-source-merged" in meta["dataset_sources"]
    assert "flight0234/rsna-knee-hybrid-report-labels" not in meta["dataset_sources"]

    data = tmp_path / "input"
    work_hyb = tmp_path / "work_hyb"
    work_gem = tmp_path / "work_gem"
    work_hyb.mkdir()
    work_gem.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    sag = series.groupby(STUDY_ID_COL)["Anatomical_Plane"].apply(lambda s: (s == "Sagittal").mean())
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = sag.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Fracture"] = 0.5
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)
    hyb = llm.copy()
    hyb["Fracture"] = sag.reindex(hyb[STUDY_ID_COL]).fillna(0.5).clip(0.05, 0.95).to_numpy()
    hyb.to_csv(data / "report_labels_v4hybrid.csv", index=False)
    gem = llm.copy()
    gem["Fracture"] = (1.0 - sag.reindex(gem[STUDY_ID_COL]).fillna(0.5)).clip(0.05, 0.95).to_numpy()
    gem.to_csv(data / "report_labels_v5.csv", index=False)

    def _run(strategy: str, work: Path) -> pd.DataFrame:
        src = _source_for_strategy(strategy, "00_test")
        src = src.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
        src = src.replace(
            'path = Path("/kaggle/working/submission.csv")',
            f"path = Path({str(work / 'submission.csv')!r})",
        )
        ns: dict = {}
        exec(compile(src, f"{strategy}_nb.py", "exec"), ns, ns)
        return pd.read_csv(work / "submission.csv"), ns

    out_hyb, _ = _run("weak_rank_bakers_frac_hyb_llm", work_hyb)
    out_gem, ns_gem = _run("weak_rank_bakers_frac_gem_llm", work_gem)
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out_gem[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out_gem[DEFAULT_TARGETS].isna().any().any() == False
    assert float(out_gem["Fracture"].std()) > 1e-6
    assert not np.allclose(out_hyb["Fracture"].to_numpy(), out_gem["Fracture"].to_numpy())
    assert ns_gem["MIX_TARGETS"]["weak_rank_bakers_frac_gem_llm"] == {"Baker's"}


def test_weak_rank_bakers_frac_hyb_gem_llm_stacks_gemini_on_hybrid(tmp_path):
    nb = implement_notebook(tmp_path / "nb_frac_hyb_gem", "weak_rank_bakers_frac_hyb_gem_llm", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_frac_hyb_gem_llm'" in src
    assert "mixed Gemini Fracture" in src
    assert "discover_gemini_labels" in src
    assert "discover_hybrid_labels" in src
    assert "report_labels_v5.csv" in src
    assert "report_labels_v4hybrid.csv" in src
    meta = json.loads((tmp_path / "nb_frac_hyb_gem" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "stevenleehans/rsna-knee-llm-report-labels" in meta["dataset_sources"]
    assert "yunusgmsoy/rsna-knee-llm-labels-4-source-merged" in meta["dataset_sources"]
    assert "flight0234/rsna-knee-hybrid-report-labels" in meta["dataset_sources"]

    data = tmp_path / "input"
    work_hyb = tmp_path / "work_hyb"
    work_gem = tmp_path / "work_gem"
    work_stack = tmp_path / "work_stack"
    work_hyb.mkdir()
    work_gem.mkdir()
    work_stack.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    sag = series.groupby(STUDY_ID_COL)["Anatomical_Plane"].apply(lambda s: (s == "Sagittal").mean())
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = sag.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Fracture"] = 0.5
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)
    hyb = llm.copy()
    # Independent of sag/fluid so stacking differs from gold, hybrid-only, and Gemini-only.
    hyb["Fracture"] = np.linspace(0.08, 0.92, len(hyb))
    hyb.to_csv(data / "report_labels_v4hybrid.csv", index=False)
    gem = llm.copy()
    gem["Fracture"] = np.linspace(0.92, 0.08, len(gem))
    gem.to_csv(data / "report_labels_v5.csv", index=False)

    def _run(strategy: str, work: Path) -> pd.DataFrame:
        src_s = _source_for_strategy(strategy, "00_test")
        src_s = src_s.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
        src_s = src_s.replace(
            'path = Path("/kaggle/working/submission.csv")',
            f"path = Path({str(work / 'submission.csv')!r})",
        )
        ns: dict = {}
        exec(compile(src_s, f"{strategy}_nb.py", "exec"), ns, ns)
        return pd.read_csv(work / "submission.csv"), ns

    out_hyb, _ = _run("weak_rank_bakers_frac_hyb_llm", work_hyb)
    out_gem, _ = _run("weak_rank_bakers_frac_gem_llm", work_gem)
    out_stack, ns_stack = _run("weak_rank_bakers_frac_hyb_gem_llm", work_stack)
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out_stack[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out_stack[DEFAULT_TARGETS].isna().any().any() == False
    assert float(out_stack["Fracture"].std()) > 1e-6
    assert not np.allclose(out_hyb["Fracture"].to_numpy(), out_stack["Fracture"].to_numpy())
    assert not np.allclose(out_gem["Fracture"].to_numpy(), out_stack["Fracture"].to_numpy())
    assert ns_stack["MIX_TARGETS"]["weak_rank_bakers_frac_hyb_gem_llm"] == {"Baker's"}


def test_weak_rank_bakers_frac_hyb_mm_llm_mixes_medial_meniscus(tmp_path):
    nb = implement_notebook(tmp_path / "nb_frac_hyb_mm", "weak_rank_bakers_frac_hyb_mm_llm", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_frac_hyb_mm_llm'" in src
    assert "mixed LLM Medial Meniscus" in src
    assert "discover_hybrid_labels" in src
    assert "target_name=\"Medial Meniscus\"" in src
    meta = json.loads((tmp_path / "nb_frac_hyb_mm" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "stevenleehans/rsna-knee-llm-report-labels" in meta["dataset_sources"]
    assert "flight0234/rsna-knee-hybrid-report-labels" in meta["dataset_sources"]
    assert "yunusgmsoy/rsna-knee-llm-labels-4-source-merged" not in meta["dataset_sources"]

    data = tmp_path / "input"
    work_hyb = tmp_path / "work_hyb"
    work_mm = tmp_path / "work_mm"
    work_hyb.mkdir()
    work_mm.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    sag = series.groupby(STUDY_ID_COL)["Anatomical_Plane"].apply(lambda s: (s == "Sagittal").mean())
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = sag.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Medial Meniscus"] = np.linspace(0.07, 0.93, len(llm))
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)
    hyb = llm.copy()
    hyb["Fracture"] = np.linspace(0.08, 0.92, len(hyb))
    hyb.to_csv(data / "report_labels_v4hybrid.csv", index=False)

    def _run(strategy: str, work: Path) -> pd.DataFrame:
        src_s = _source_for_strategy(strategy, "00_test")
        src_s = src_s.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
        src_s = src_s.replace(
            'path = Path("/kaggle/working/submission.csv")',
            f"path = Path({str(work / 'submission.csv')!r})",
        )
        ns: dict = {}
        exec(compile(src_s, f"{strategy}_nb.py", "exec"), ns, ns)
        return pd.read_csv(work / "submission.csv"), ns

    out_hyb, _ = _run("weak_rank_bakers_frac_hyb_llm", work_hyb)
    out_mm, ns_mm = _run("weak_rank_bakers_frac_hyb_mm_llm", work_mm)
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out_mm[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out_mm[DEFAULT_TARGETS].isna().any().any() == False
    assert float(out_mm["Medial Meniscus"].std()) > 1e-6
    assert not np.allclose(out_hyb["Medial Meniscus"].to_numpy(), out_mm["Medial Meniscus"].to_numpy())
    assert np.allclose(out_hyb["Fracture"].to_numpy(), out_mm["Fracture"].to_numpy())
    assert ns_mm["MIX_TARGETS"]["weak_rank_bakers_frac_hyb_mm_llm"] == {"Baker's"}


def test_weak_rank_bakers_frac_hyb_lm_llm_mixes_lateral_meniscus(tmp_path):
    nb = implement_notebook(tmp_path / "nb_frac_hyb_lm", "weak_rank_bakers_frac_hyb_lm_llm", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_frac_hyb_lm_llm'" in src
    assert "mixed LLM Lateral Meniscus" in src
    assert "discover_hybrid_labels" in src
    assert 'target_name="Lateral Meniscus"' in src
    meta = json.loads((tmp_path / "nb_frac_hyb_lm" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "stevenleehans/rsna-knee-llm-report-labels" in meta["dataset_sources"]
    assert "flight0234/rsna-knee-hybrid-report-labels" in meta["dataset_sources"]
    assert "yunusgmsoy/rsna-knee-llm-labels-4-source-merged" not in meta["dataset_sources"]

    data = tmp_path / "input"
    work_hyb = tmp_path / "work_hyb"
    work_lm = tmp_path / "work_lm"
    work_hyb.mkdir()
    work_lm.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    sag = series.groupby(STUDY_ID_COL)["Anatomical_Plane"].apply(lambda s: (s == "Sagittal").mean())
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = sag.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Lateral Meniscus"] = np.linspace(0.06, 0.94, len(llm))
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)
    hyb = llm.copy()
    hyb["Fracture"] = np.linspace(0.08, 0.92, len(hyb))
    hyb.to_csv(data / "report_labels_v4hybrid.csv", index=False)

    def _run(strategy: str, work: Path) -> pd.DataFrame:
        src_s = _source_for_strategy(strategy, "00_test")
        src_s = src_s.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
        src_s = src_s.replace(
            'path = Path("/kaggle/working/submission.csv")',
            f"path = Path({str(work / 'submission.csv')!r})",
        )
        ns: dict = {}
        exec(compile(src_s, f"{strategy}_nb.py", "exec"), ns, ns)
        return pd.read_csv(work / "submission.csv"), ns

    out_hyb, _ = _run("weak_rank_bakers_frac_hyb_llm", work_hyb)
    out_lm, ns_lm = _run("weak_rank_bakers_frac_hyb_lm_llm", work_lm)
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out_lm[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out_lm[DEFAULT_TARGETS].isna().any().any() == False
    assert float(out_lm["Lateral Meniscus"].std()) > 1e-6
    assert not np.allclose(out_hyb["Lateral Meniscus"].to_numpy(), out_lm["Lateral Meniscus"].to_numpy())
    assert np.allclose(out_hyb["Fracture"].to_numpy(), out_lm["Fracture"].to_numpy())
    assert ns_lm["MIX_TARGETS"]["weak_rank_bakers_frac_hyb_lm_llm"] == {"Baker's"}


def test_weak_rank_bakers_frac_hyb_lm_w60_gives_majority_lm_weight(tmp_path):
    nb = implement_notebook(tmp_path / "nb_frac_hyb_lm_w60", "weak_rank_bakers_frac_hyb_lm_w60", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_frac_hyb_lm_w60'" in src
    assert "mixed LLM Lateral Meniscus" in src
    assert 'w_gold = 0.40 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_w60" else 0.60 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_w40" else 0.50' in src
    meta = json.loads((tmp_path / "nb_frac_hyb_lm_w60" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "stevenleehans/rsna-knee-llm-report-labels" in meta["dataset_sources"]
    assert "flight0234/rsna-knee-hybrid-report-labels" in meta["dataset_sources"]

    data = tmp_path / "input"
    work_lm = tmp_path / "work_lm"
    work_w60 = tmp_path / "work_w60"
    work_lm.mkdir()
    work_w60.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    sag = series.groupby(STUDY_ID_COL)["Anatomical_Plane"].apply(lambda s: (s == "Sagittal").mean())
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = sag.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Lateral Meniscus"] = np.linspace(0.06, 0.94, len(llm))
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)
    hyb = llm.copy()
    hyb["Fracture"] = np.linspace(0.08, 0.92, len(hyb))
    hyb.to_csv(data / "report_labels_v4hybrid.csv", index=False)

    def _run(strategy: str, work: Path) -> pd.DataFrame:
        src_s = _source_for_strategy(strategy, "00_test")
        src_s = src_s.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
        src_s = src_s.replace(
            'path = Path("/kaggle/working/submission.csv")',
            f"path = Path({str(work / 'submission.csv')!r})",
        )
        ns: dict = {}
        exec(compile(src_s, f"{strategy}_nb.py", "exec"), ns, ns)
        return pd.read_csv(work / "submission.csv"), ns

    out_lm, _ = _run("weak_rank_bakers_frac_hyb_lm_llm", work_lm)
    out_w60, ns_w60 = _run("weak_rank_bakers_frac_hyb_lm_w60", work_w60)
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out_w60[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out_w60[DEFAULT_TARGETS].isna().any().any() == False
    assert float(out_w60["Lateral Meniscus"].std()) > 1e-6
    assert not np.allclose(out_lm["Lateral Meniscus"].to_numpy(), out_w60["Lateral Meniscus"].to_numpy())
    assert np.allclose(out_lm["Fracture"].to_numpy(), out_w60["Fracture"].to_numpy())
    assert ns_w60["MIX_TARGETS"]["weak_rank_bakers_frac_hyb_lm_w60"] == {"Baker's"}


def test_weak_rank_bakers_frac_hyb_lm_eff_acl_w60_gives_majority_acl_weight(tmp_path):
    nb = implement_notebook(tmp_path / "nb_acl_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_w60", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_frac_hyb_lm_eff_acl_w60'" in src
    assert "mixed LLM ACL std" in src
    assert 'w_gold = 0.40 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_w60" else 0.60 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_w40" else 0.50' in src
    assert 'STRATEGY in {"weak_rank_bakers_frac_hyb_lm_eff_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem"}' in src
    meta = json.loads((tmp_path / "nb_acl_w60" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "stevenleehans/rsna-knee-llm-report-labels" in meta["dataset_sources"]
    assert "flight0234/rsna-knee-hybrid-report-labels" in meta["dataset_sources"]

    data = tmp_path / "input"
    work_base = tmp_path / "work_base"
    work_acl = tmp_path / "work_acl"
    work_base.mkdir()
    work_acl.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = np.linspace(0.94, 0.06, len(llm))
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Lateral Meniscus"] = np.linspace(0.06, 0.94, len(llm))
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)
    hyb = llm.copy()
    hyb["Fracture"] = np.linspace(0.08, 0.92, len(hyb))
    hyb.to_csv(data / "report_labels_v4hybrid.csv", index=False)

    def _run(strategy: str, work: Path):
        src_s = _source_for_strategy(strategy, "00_test")
        src_s = src_s.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
        src_s = src_s.replace(
            'path = Path("/kaggle/working/submission.csv")',
            f"path = Path({str(work / 'submission.csv')!r})",
        )
        ns: dict = {}
        exec(compile(src_s, f"{strategy}_nb.py", "exec"), ns, ns)
        return pd.read_csv(work / "submission.csv"), ns

    out_base, _ = _run("weak_rank_bakers_frac_hyb_lm_eff_w60", work_base)
    out_acl, ns_acl = _run("weak_rank_bakers_frac_hyb_lm_eff_acl_w60", work_acl)
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out_acl[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out_acl[DEFAULT_TARGETS].isna().any().any() == False
    assert float(out_acl["ACL"].std()) > 1e-6
    assert np.allclose(out_base["Effusion"].to_numpy(), out_acl["Effusion"].to_numpy())
    assert np.allclose(out_base["Lateral Meniscus"].to_numpy(), out_acl["Lateral Meniscus"].to_numpy())
    assert ns_acl["MIX_TARGETS"]["weak_rank_bakers_frac_hyb_lm_eff_acl_w60"] == {"Baker's"}
    assert 'w_gold = 0.40 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_w60" else 0.60 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_w40" else 0.50' in src


def test_weak_rank_bakers_frac_hyb_lm_eff_acl_w40_gives_minority_acl_weight(tmp_path):
    nb = implement_notebook(tmp_path / "nb_acl_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_w40", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_frac_hyb_lm_eff_acl_w40'" in src
    assert 'w_gold = 0.40 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_w60" else 0.60 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_w40" else 0.50' in src
    meta = json.loads((tmp_path / "nb_acl_w40" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "stevenleehans/rsna-knee-llm-report-labels" in meta["dataset_sources"]
    assert "flight0234/rsna-knee-hybrid-report-labels" in meta["dataset_sources"]

    data = tmp_path / "input"
    work_w60 = tmp_path / "work_w60"
    work_w40 = tmp_path / "work_w40"
    work_w60.mkdir()
    work_w40.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = np.linspace(0.94, 0.06, len(llm))
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Lateral Meniscus"] = np.linspace(0.06, 0.94, len(llm))
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)
    hyb = llm.copy()
    hyb["Fracture"] = np.linspace(0.08, 0.92, len(hyb))
    hyb.to_csv(data / "report_labels_v4hybrid.csv", index=False)

    def _run(strategy: str, work: Path):
        src_s = _source_for_strategy(strategy, "00_test")
        src_s = src_s.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
        src_s = src_s.replace(
            'path = Path("/kaggle/working/submission.csv")',
            f"path = Path({str(work / 'submission.csv')!r})",
        )
        ns: dict = {}
        exec(compile(src_s, f"{strategy}_nb.py", "exec"), ns, ns)
        return pd.read_csv(work / "submission.csv"), ns

    out_w60, _ = _run("weak_rank_bakers_frac_hyb_lm_eff_acl_w60", work_w60)
    out_w40, ns_w40 = _run("weak_rank_bakers_frac_hyb_lm_eff_acl_w40", work_w40)
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out_w40[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out_w40[DEFAULT_TARGETS].isna().any().any() == False
    assert float(out_w40["ACL"].std()) > 1e-6
    assert np.allclose(out_w60["Effusion"].to_numpy(), out_w40["Effusion"].to_numpy())
    assert np.allclose(out_w60["Lateral Meniscus"].to_numpy(), out_w40["Lateral Meniscus"].to_numpy())
    assert ns_w40["MIX_TARGETS"]["weak_rank_bakers_frac_hyb_lm_eff_acl_w40"] == {"Baker's"}
    assert "mixed LLM ACL std" in src


def test_weak_rank_bakers_frac_hyb_lm_eff_acl_gem_mixes_gemini_acl(tmp_path):
    nb = implement_notebook(tmp_path / "nb_acl_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_frac_hyb_lm_eff_acl_gem'" in src
    assert "mixed Gemini ACL std" in src
    assert "discover_gemini_labels" in src
    meta = json.loads((tmp_path / "nb_acl_gem" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "stevenleehans/rsna-knee-llm-report-labels" in meta["dataset_sources"]
    assert "flight0234/rsna-knee-hybrid-report-labels" in meta["dataset_sources"]
    assert "yunusgmsoy/rsna-knee-llm-labels-4-source-merged" in meta["dataset_sources"]

    data = tmp_path / "input"
    work_base = tmp_path / "work_base"
    work_gem = tmp_path / "work_gem"
    work_base.mkdir()
    work_gem.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = np.linspace(0.94, 0.06, len(llm))
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Lateral Meniscus"] = np.linspace(0.06, 0.94, len(llm))
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)
    hyb = llm.copy()
    hyb["Fracture"] = np.linspace(0.08, 0.92, len(hyb))
    hyb.to_csv(data / "report_labels_v4hybrid.csv", index=False)
    gem = llm.copy()
    gem["ACL"] = np.linspace(0.05, 0.95, len(gem))
    gem.to_csv(data / "report_labels_v5.csv", index=False)

    def _run(strategy: str, work: Path):
        src_s = _source_for_strategy(strategy, "00_test")
        src_s = src_s.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
        src_s = src_s.replace(
            'path = Path("/kaggle/working/submission.csv")',
            f"path = Path({str(work / 'submission.csv')!r})",
        )
        ns: dict = {}
        exec(compile(src_s, f"{strategy}_nb.py", "exec"), ns, ns)
        return pd.read_csv(work / "submission.csv"), ns

    out_base, _ = _run("weak_rank_bakers_frac_hyb_lm_eff_w60", work_base)
    out_gem, ns_gem = _run("weak_rank_bakers_frac_hyb_lm_eff_acl_gem", work_gem)
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out_gem[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out_gem[DEFAULT_TARGETS].isna().any().any() == False
    assert float(out_gem["ACL"].std()) > 1e-6
    assert not np.allclose(out_base["ACL"].to_numpy(), out_gem["ACL"].to_numpy())
    assert np.allclose(out_base["Effusion"].to_numpy(), out_gem["Effusion"].to_numpy())
    assert np.allclose(out_base["Lateral Meniscus"].to_numpy(), out_gem["Lateral Meniscus"].to_numpy())
    assert ns_gem["MIX_TARGETS"]["weak_rank_bakers_frac_hyb_lm_eff_acl_gem"] == {"Baker's"}


def test_weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60_gives_majority_gemini_acl(tmp_path):
    nb = implement_notebook(tmp_path / "nb_acl_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60'" in src
    assert "mixed Gemini ACL std" in src
    assert 'w_gold = 0.40 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60" else 0.60 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40" else 0.50' in src
    assert "discover_gemini_labels" in src
    meta = json.loads((tmp_path / "nb_acl_gem_w60" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "stevenleehans/rsna-knee-llm-report-labels" in meta["dataset_sources"]
    assert "flight0234/rsna-knee-hybrid-report-labels" in meta["dataset_sources"]
    assert "yunusgmsoy/rsna-knee-llm-labels-4-source-merged" in meta["dataset_sources"]

    data = tmp_path / "input"
    work_gem = tmp_path / "work_gem"
    work_w60 = tmp_path / "work_w60"
    work_gem.mkdir()
    work_w60.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = np.linspace(0.94, 0.06, len(llm))
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Lateral Meniscus"] = np.linspace(0.06, 0.94, len(llm))
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)
    hyb = llm.copy()
    hyb["Fracture"] = np.linspace(0.08, 0.92, len(hyb))
    hyb.to_csv(data / "report_labels_v4hybrid.csv", index=False)
    gem = llm.copy()
    gem["ACL"] = np.linspace(0.05, 0.95, len(gem))
    gem.to_csv(data / "report_labels_v5.csv", index=False)

    def _run(strategy: str, work: Path):
        src_s = _source_for_strategy(strategy, "00_test")
        src_s = src_s.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
        src_s = src_s.replace(
            'path = Path("/kaggle/working/submission.csv")',
            f"path = Path({str(work / 'submission.csv')!r})",
        )
        ns: dict = {}
        exec(compile(src_s, f"{strategy}_nb.py", "exec"), ns, ns)
        return pd.read_csv(work / "submission.csv"), ns

    out_gem, _ = _run("weak_rank_bakers_frac_hyb_lm_eff_acl_gem", work_gem)
    out_w60, ns_w60 = _run("weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60", work_w60)
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out_w60[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out_w60[DEFAULT_TARGETS].isna().any().any() == False
    assert float(out_w60["ACL"].std()) > 1e-6
    assert not np.allclose(out_gem["ACL"].to_numpy(), out_w60["ACL"].to_numpy())
    assert np.allclose(out_gem["Effusion"].to_numpy(), out_w60["Effusion"].to_numpy())
    assert np.allclose(out_gem["Lateral Meniscus"].to_numpy(), out_w60["Lateral Meniscus"].to_numpy())
    assert ns_w60["MIX_TARGETS"]["weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60"] == {"Baker's"}


def test_weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40_gives_minority_gemini_acl(tmp_path):
    nb = implement_notebook(tmp_path / "nb_acl_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40", "00_test", "slug", "user")
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40'" in src
    assert "mixed Gemini ACL std" in src
    assert 'w_gold = 0.40 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60" else 0.60 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40" else 0.50' in src
    assert "discover_gemini_labels" in src
    meta = json.loads((tmp_path / "nb_acl_gem_w40" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "stevenleehans/rsna-knee-llm-report-labels" in meta["dataset_sources"]
    assert "flight0234/rsna-knee-hybrid-report-labels" in meta["dataset_sources"]
    assert "yunusgmsoy/rsna-knee-llm-labels-4-source-merged" in meta["dataset_sources"]

    data = tmp_path / "input"
    work_gem = tmp_path / "work_gem"
    work_w40 = tmp_path / "work_w40"
    work_gem.mkdir()
    work_w40.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = np.linspace(0.94, 0.06, len(llm))
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Lateral Meniscus"] = np.linspace(0.06, 0.94, len(llm))
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)
    hyb = llm.copy()
    hyb["Fracture"] = np.linspace(0.08, 0.92, len(hyb))
    hyb.to_csv(data / "report_labels_v4hybrid.csv", index=False)
    gem = llm.copy()
    gem["ACL"] = np.linspace(0.05, 0.95, len(gem))
    gem.to_csv(data / "report_labels_v5.csv", index=False)

    def _run(strategy: str, work: Path):
        src_s = _source_for_strategy(strategy, "00_test")
        src_s = src_s.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
        src_s = src_s.replace(
            'path = Path("/kaggle/working/submission.csv")',
            f"path = Path({str(work / 'submission.csv')!r})",
        )
        ns: dict = {}
        exec(compile(src_s, f"{strategy}_nb.py", "exec"), ns, ns)
        return pd.read_csv(work / "submission.csv"), ns

    out_gem, _ = _run("weak_rank_bakers_frac_hyb_lm_eff_acl_gem", work_gem)
    out_w40, ns_w40 = _run("weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40", work_w40)
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out_w40[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out_w40[DEFAULT_TARGETS].isna().any().any() == False
    assert float(out_w40["ACL"].std()) > 1e-6
    assert not np.allclose(out_gem["ACL"].to_numpy(), out_w40["ACL"].to_numpy())
    assert np.allclose(out_gem["Effusion"].to_numpy(), out_w40["Effusion"].to_numpy())
    assert np.allclose(out_gem["Lateral Meniscus"].to_numpy(), out_w40["Lateral Meniscus"].to_numpy())
    assert ns_w40["MIX_TARGETS"]["weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40"] == {"Baker's"}


def test_weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_mixes_gemini_effusion(tmp_path):
    nb = implement_notebook(
        tmp_path / "nb_acl_gem_eff",
        "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem",
        "00_test",
        "slug",
        "user",
    )
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem'" in src
    assert "mixed Gemini Effusion std" in src
    assert "discover_gemini_labels" in src
    meta = json.loads((tmp_path / "nb_acl_gem_eff" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "stevenleehans/rsna-knee-llm-report-labels" in meta["dataset_sources"]
    assert "flight0234/rsna-knee-hybrid-report-labels" in meta["dataset_sources"]
    assert "yunusgmsoy/rsna-knee-llm-labels-4-source-merged" in meta["dataset_sources"]

    data = tmp_path / "input"
    work_gem = tmp_path / "work_gem"
    work_eff = tmp_path / "work_eff"
    work_gem.mkdir()
    work_eff.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = np.linspace(0.94, 0.06, len(llm))
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Lateral Meniscus"] = np.linspace(0.06, 0.94, len(llm))
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)
    hyb = llm.copy()
    hyb["Fracture"] = np.linspace(0.08, 0.92, len(hyb))
    hyb.to_csv(data / "report_labels_v4hybrid.csv", index=False)
    gem = llm.copy()
    gem["ACL"] = np.linspace(0.05, 0.95, len(gem))
    gem["Effusion"] = np.linspace(0.97, 0.03, len(gem))
    gem.to_csv(data / "report_labels_v5.csv", index=False)

    def _run(strategy: str, work: Path):
        src_s = _source_for_strategy(strategy, "00_test")
        src_s = src_s.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
        src_s = src_s.replace(
            'path = Path("/kaggle/working/submission.csv")',
            f"path = Path({str(work / 'submission.csv')!r})",
        )
        ns: dict = {}
        exec(compile(src_s, f"{strategy}_nb.py", "exec"), ns, ns)
        return pd.read_csv(work / "submission.csv"), ns

    out_gem, _ = _run("weak_rank_bakers_frac_hyb_lm_eff_acl_gem", work_gem)
    out_eff, ns_eff = _run("weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem", work_eff)
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out_eff[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out_eff[DEFAULT_TARGETS].isna().any().any() == False
    assert float(out_eff["Effusion"].std()) > 1e-6
    assert np.allclose(out_gem["ACL"].to_numpy(), out_eff["ACL"].to_numpy())
    assert not np.allclose(out_gem["Effusion"].to_numpy(), out_eff["Effusion"].to_numpy())
    assert np.allclose(out_gem["Lateral Meniscus"].to_numpy(), out_eff["Lateral Meniscus"].to_numpy())
    assert ns_eff["MIX_TARGETS"]["weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem"] == {"Baker's"}


def test_weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60_gives_majority_gemini_effusion(tmp_path):
    nb = implement_notebook(
        tmp_path / "nb_acl_gem_eff_w60",
        "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60",
        "00_test",
        "slug",
        "user",
    )
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60'" in src
    assert "mixed Gemini Effusion std" in src
    assert 'w_gold = 0.40 if STRATEGY in {"weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem"} else 0.60 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40" else 0.50' in src
    meta = json.loads((tmp_path / "nb_acl_gem_eff_w60" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "yunusgmsoy/rsna-knee-llm-labels-4-source-merged" in meta["dataset_sources"]

    data = tmp_path / "input"
    work_eff = tmp_path / "work_eff"
    work_w60 = tmp_path / "work_w60"
    work_eff.mkdir()
    work_w60.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = np.linspace(0.94, 0.06, len(llm))
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Lateral Meniscus"] = np.linspace(0.06, 0.94, len(llm))
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)
    hyb = llm.copy()
    hyb["Fracture"] = np.linspace(0.08, 0.92, len(hyb))
    hyb.to_csv(data / "report_labels_v4hybrid.csv", index=False)
    gem = llm.copy()
    gem["ACL"] = np.linspace(0.05, 0.95, len(gem))
    gem["Effusion"] = np.linspace(0.97, 0.03, len(gem))
    gem.to_csv(data / "report_labels_v5.csv", index=False)

    def _run(strategy: str, work: Path):
        src_s = _source_for_strategy(strategy, "00_test")
        src_s = src_s.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
        src_s = src_s.replace(
            'path = Path("/kaggle/working/submission.csv")',
            f"path = Path({str(work / 'submission.csv')!r})",
        )
        ns: dict = {}
        exec(compile(src_s, f"{strategy}_nb.py", "exec"), ns, ns)
        return pd.read_csv(work / "submission.csv"), ns

    out_eff, _ = _run("weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem", work_eff)
    out_w60, ns_w60 = _run("weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60", work_w60)
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out_w60[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out_w60[DEFAULT_TARGETS].isna().any().any() == False
    assert float(out_w60["Effusion"].std()) > 1e-6
    assert np.allclose(out_eff["ACL"].to_numpy(), out_w60["ACL"].to_numpy())
    assert not np.allclose(out_eff["Effusion"].to_numpy(), out_w60["Effusion"].to_numpy())
    assert np.allclose(out_eff["Lateral Meniscus"].to_numpy(), out_w60["Lateral Meniscus"].to_numpy())
    assert ns_w60["MIX_TARGETS"]["weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60"] == {"Baker's"}


def test_weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40_gives_minority_gemini_effusion(tmp_path):
    nb = implement_notebook(
        tmp_path / "nb_acl_gem_eff_w40",
        "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40",
        "00_test",
        "slug",
        "user",
    )
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40'" in src
    assert "mixed Gemini Effusion std" in src
    assert 'w_gold = 0.40 if STRATEGY in {"weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem", "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem"} else 0.60 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40" else 0.50' in src
    meta = json.loads((tmp_path / "nb_acl_gem_eff_w40" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "yunusgmsoy/rsna-knee-llm-labels-4-source-merged" in meta["dataset_sources"]

    data = tmp_path / "input"
    work_w60 = tmp_path / "work_w60"
    work_w40 = tmp_path / "work_w40"
    work_w60.mkdir()
    work_w40.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = np.linspace(0.94, 0.06, len(llm))
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Lateral Meniscus"] = np.linspace(0.06, 0.94, len(llm))
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)
    hyb = llm.copy()
    hyb["Fracture"] = np.linspace(0.08, 0.92, len(hyb))
    hyb.to_csv(data / "report_labels_v4hybrid.csv", index=False)
    gem = llm.copy()
    gem["ACL"] = np.linspace(0.05, 0.95, len(gem))
    gem["Effusion"] = np.linspace(0.97, 0.03, len(gem))
    gem.to_csv(data / "report_labels_v5.csv", index=False)

    def _run(strategy: str, work: Path):
        src_s = _source_for_strategy(strategy, "00_test")
        src_s = src_s.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
        src_s = src_s.replace(
            'path = Path("/kaggle/working/submission.csv")',
            f"path = Path({str(work / 'submission.csv')!r})",
        )
        ns: dict = {}
        exec(compile(src_s, f"{strategy}_nb.py", "exec"), ns, ns)
        return pd.read_csv(work / "submission.csv"), ns

    out_w60, _ = _run("weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60", work_w60)
    out_w40, ns_w40 = _run("weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40", work_w40)
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out_w40[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out_w40[DEFAULT_TARGETS].isna().any().any() == False
    assert float(out_w40["Effusion"].std()) > 1e-6
    assert np.allclose(out_w60["ACL"].to_numpy(), out_w40["ACL"].to_numpy())
    assert not np.allclose(out_w60["Effusion"].to_numpy(), out_w40["Effusion"].to_numpy())
    assert np.allclose(out_w60["Lateral Meniscus"].to_numpy(), out_w40["Lateral Meniscus"].to_numpy())
    assert ns_w40["MIX_TARGETS"]["weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40"] == {"Baker's"}


def test_weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mixes_gemini_lm(tmp_path):
    nb = implement_notebook(
        tmp_path / "nb_acl_gem_lm",
        "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem",
        "00_test",
        "slug",
        "user",
    )
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem'" in src
    assert "mixed Gemini Lateral Meniscus std" in src
    assert "discover_gemini_labels" in src
    meta = json.loads((tmp_path / "nb_acl_gem_lm" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "yunusgmsoy/rsna-knee-llm-labels-4-source-merged" in meta["dataset_sources"]

    data = tmp_path / "input"
    work_w60 = tmp_path / "work_w60"
    work_lm = tmp_path / "work_lm"
    work_w60.mkdir()
    work_lm.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = np.linspace(0.94, 0.06, len(llm))
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Lateral Meniscus"] = np.linspace(0.06, 0.94, len(llm))
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)
    hyb = llm.copy()
    hyb["Fracture"] = np.linspace(0.08, 0.92, len(hyb))
    hyb.to_csv(data / "report_labels_v4hybrid.csv", index=False)
    gem = llm.copy()
    gem["ACL"] = np.linspace(0.05, 0.95, len(gem))
    gem["Effusion"] = np.linspace(0.97, 0.03, len(gem))
    gem["Lateral Meniscus"] = np.linspace(0.99, 0.01, len(gem))
    gem.to_csv(data / "report_labels_v5.csv", index=False)

    def _run(strategy: str, work: Path):
        src_s = _source_for_strategy(strategy, "00_test")
        src_s = src_s.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
        src_s = src_s.replace(
            'path = Path("/kaggle/working/submission.csv")',
            f"path = Path({str(work / 'submission.csv')!r})",
        )
        ns: dict = {}
        exec(compile(src_s, f"{strategy}_nb.py", "exec"), ns, ns)
        return pd.read_csv(work / "submission.csv"), ns

    out_w60, _ = _run("weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60", work_w60)
    out_lm, ns_lm = _run("weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem", work_lm)
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out_lm[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out_lm[DEFAULT_TARGETS].isna().any().any() == False
    assert float(out_lm["Lateral Meniscus"].std()) > 1e-6
    assert np.allclose(out_w60["ACL"].to_numpy(), out_lm["ACL"].to_numpy())
    assert np.allclose(out_w60["Effusion"].to_numpy(), out_lm["Effusion"].to_numpy())
    assert not np.allclose(out_w60["Lateral Meniscus"].to_numpy(), out_lm["Lateral Meniscus"].to_numpy())
    assert ns_lm["MIX_TARGETS"]["weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem"] == {"Baker's"}


def test_weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60_gives_majority_gemini_lm(tmp_path):
    nb = implement_notebook(
        tmp_path / "nb_acl_gem_lm_w60",
        "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60",
        "00_test",
        "slug",
        "user",
    )
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60'" in src
    assert "mixed Gemini Lateral Meniscus std" in src
    assert 'w_gold = 0.40 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60" else 0.60 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40" else 0.50' in src
    meta = json.loads((tmp_path / "nb_acl_gem_lm_w60" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "yunusgmsoy/rsna-knee-llm-labels-4-source-merged" in meta["dataset_sources"]

    data = tmp_path / "input"
    work_lm = tmp_path / "work_lm"
    work_w60 = tmp_path / "work_lm_w60"
    work_lm.mkdir()
    work_w60.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = np.linspace(0.94, 0.06, len(llm))
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Lateral Meniscus"] = np.linspace(0.06, 0.94, len(llm))
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)
    hyb = llm.copy()
    hyb["Fracture"] = np.linspace(0.08, 0.92, len(hyb))
    hyb.to_csv(data / "report_labels_v4hybrid.csv", index=False)
    gem = llm.copy()
    gem["ACL"] = np.linspace(0.05, 0.95, len(gem))
    gem["Effusion"] = np.linspace(0.97, 0.03, len(gem))
    gem["Lateral Meniscus"] = np.linspace(0.99, 0.01, len(gem))
    gem.to_csv(data / "report_labels_v5.csv", index=False)

    def _run(strategy: str, work: Path):
        src_s = _source_for_strategy(strategy, "00_test")
        src_s = src_s.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
        src_s = src_s.replace(
            'path = Path("/kaggle/working/submission.csv")',
            f"path = Path({str(work / 'submission.csv')!r})",
        )
        ns: dict = {}
        exec(compile(src_s, f"{strategy}_nb.py", "exec"), ns, ns)
        return pd.read_csv(work / "submission.csv"), ns

    out_lm, _ = _run("weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem", work_lm)
    out_w60, ns_w60 = _run("weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60", work_w60)
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out_w60[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out_w60[DEFAULT_TARGETS].isna().any().any() == False
    assert float(out_w60["Lateral Meniscus"].std()) > 1e-6
    assert np.allclose(out_lm["ACL"].to_numpy(), out_w60["ACL"].to_numpy())
    assert np.allclose(out_lm["Effusion"].to_numpy(), out_w60["Effusion"].to_numpy())
    assert not np.allclose(out_lm["Lateral Meniscus"].to_numpy(), out_w60["Lateral Meniscus"].to_numpy())
    assert ns_w60["MIX_TARGETS"]["weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60"] == {"Baker's"}


def test_weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40_gives_minority_gemini_lm(tmp_path):
    nb = implement_notebook(
        tmp_path / "nb_acl_gem_lm_w40",
        "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40",
        "00_test",
        "slug",
        "user",
    )
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40'" in src
    assert "mixed Gemini Lateral Meniscus std" in src
    assert 'w_gold = 0.40 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60" else 0.60 if STRATEGY == "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40" else 0.50' in src
    meta = json.loads((tmp_path / "nb_acl_gem_lm_w40" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "yunusgmsoy/rsna-knee-llm-labels-4-source-merged" in meta["dataset_sources"]

    data = tmp_path / "input"
    work_lm = tmp_path / "work_lm"
    work_w40 = tmp_path / "work_lm_w40"
    work_lm.mkdir()
    work_w40.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = np.linspace(0.94, 0.06, len(llm))
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Lateral Meniscus"] = np.linspace(0.06, 0.94, len(llm))
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)
    hyb = llm.copy()
    hyb["Fracture"] = np.linspace(0.08, 0.92, len(hyb))
    hyb.to_csv(data / "report_labels_v4hybrid.csv", index=False)
    gem = llm.copy()
    gem["ACL"] = np.linspace(0.05, 0.95, len(gem))
    gem["Effusion"] = np.linspace(0.97, 0.03, len(gem))
    gem["Lateral Meniscus"] = np.linspace(0.99, 0.01, len(gem))
    gem.to_csv(data / "report_labels_v5.csv", index=False)

    def _run(strategy: str, work: Path):
        src_s = _source_for_strategy(strategy, "00_test")
        src_s = src_s.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
        src_s = src_s.replace(
            'path = Path("/kaggle/working/submission.csv")',
            f"path = Path({str(work / 'submission.csv')!r})",
        )
        ns: dict = {}
        exec(compile(src_s, f"{strategy}_nb.py", "exec"), ns, ns)
        return pd.read_csv(work / "submission.csv"), ns

    out_lm, _ = _run("weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem", work_lm)
    out_w40, ns_w40 = _run("weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40", work_w40)
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out_w40[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out_w40[DEFAULT_TARGETS].isna().any().any() == False
    assert float(out_w40["Lateral Meniscus"].std()) > 1e-6
    assert np.allclose(out_lm["ACL"].to_numpy(), out_w40["ACL"].to_numpy())
    assert np.allclose(out_lm["Effusion"].to_numpy(), out_w40["Effusion"].to_numpy())
    assert not np.allclose(out_lm["Lateral Meniscus"].to_numpy(), out_w40["Lateral Meniscus"].to_numpy())
    assert ns_w40["MIX_TARGETS"]["weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40"] == {"Baker's"}


def test_weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem_mixes_gemini_mm(tmp_path):
    nb = implement_notebook(
        tmp_path / "nb_acl_gem_mm",
        "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem",
        "00_test",
        "slug",
        "user",
    )
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem'" in src
    assert "mixed Gemini Medial Meniscus std" in src
    meta = json.loads((tmp_path / "nb_acl_gem_mm" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "yunusgmsoy/rsna-knee-llm-labels-4-source-merged" in meta["dataset_sources"]

    data = tmp_path / "input"
    work_lm = tmp_path / "work_lm"
    work_mm = tmp_path / "work_mm"
    work_lm.mkdir()
    work_mm.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = np.linspace(0.94, 0.06, len(llm))
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Lateral Meniscus"] = np.linspace(0.06, 0.94, len(llm))
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)
    hyb = llm.copy()
    hyb["Fracture"] = np.linspace(0.08, 0.92, len(hyb))
    hyb.to_csv(data / "report_labels_v4hybrid.csv", index=False)
    gem = llm.copy()
    gem["ACL"] = np.linspace(0.05, 0.95, len(gem))
    gem["Effusion"] = np.linspace(0.97, 0.03, len(gem))
    gem["Lateral Meniscus"] = np.linspace(0.99, 0.01, len(gem))
    gem["Medial Meniscus"] = np.linspace(0.02, 0.98, len(gem))
    gem.to_csv(data / "report_labels_v5.csv", index=False)

    def _run(strategy: str, work: Path):
        src_s = _source_for_strategy(strategy, "00_test")
        src_s = src_s.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
        src_s = src_s.replace(
            'path = Path("/kaggle/working/submission.csv")',
            f"path = Path({str(work / 'submission.csv')!r})",
        )
        ns: dict = {}
        exec(compile(src_s, f"{strategy}_nb.py", "exec"), ns, ns)
        return pd.read_csv(work / "submission.csv"), ns

    out_lm, _ = _run("weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem", work_lm)
    out_mm, ns_mm = _run("weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem", work_mm)
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out_mm[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out_mm[DEFAULT_TARGETS].isna().any().any() == False
    assert float(out_mm["Medial Meniscus"].std()) > 1e-6
    assert np.allclose(out_lm["ACL"].to_numpy(), out_mm["ACL"].to_numpy())
    assert np.allclose(out_lm["Effusion"].to_numpy(), out_mm["Effusion"].to_numpy())
    assert np.allclose(out_lm["Lateral Meniscus"].to_numpy(), out_mm["Lateral Meniscus"].to_numpy())
    assert not np.allclose(out_lm["Medial Meniscus"].to_numpy(), out_mm["Medial Meniscus"].to_numpy())
    assert ns_mm["MIX_TARGETS"]["weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem"] == {"Baker's"}


def test_weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem_mixes_gemini_bakers(tmp_path):
    nb = implement_notebook(
        tmp_path / "nb_acl_gem_bak",
        "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem",
        "00_test",
        "slug",
        "user",
    )
    src = "".join(json.loads(nb.read_text())["cells"][0]["source"])
    assert "STRATEGY = 'weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem'" in src
    assert "mixed Gemini Baker's std" in src
    meta = json.loads((tmp_path / "nb_acl_gem_bak" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False
    assert "yunusgmsoy/rsna-knee-llm-labels-4-source-merged" in meta["dataset_sources"]

    data = tmp_path / "input"
    work_lm = tmp_path / "work_lm"
    work_bak = tmp_path / "work_bak"
    work_lm.mkdir()
    work_bak.mkdir()
    _write_synth(data)
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    fluid = series.groupby(STUDY_ID_COL)["Fluid_Sensitive"].mean()
    llm = pd.DataFrame({STUDY_ID_COL: train[STUDY_ID_COL].astype(str)})
    for t in DEFAULT_TARGETS:
        llm[t] = 0.25
    llm["ACL"] = np.linspace(0.94, 0.06, len(llm))
    llm["Effusion"] = fluid.reindex(llm[STUDY_ID_COL]).fillna(0.25).clip(0.05, 0.95).to_numpy()
    llm["Lateral Meniscus"] = np.linspace(0.06, 0.94, len(llm))
    llm.to_csv(data / "llm_labels_v4_blend.csv", index=False)
    hyb = llm.copy()
    hyb["Fracture"] = np.linspace(0.08, 0.92, len(hyb))
    hyb.to_csv(data / "report_labels_v4hybrid.csv", index=False)
    gem = llm.copy()
    gem["ACL"] = np.linspace(0.05, 0.95, len(gem))
    gem["Effusion"] = np.linspace(0.97, 0.03, len(gem))
    gem["Lateral Meniscus"] = np.linspace(0.99, 0.01, len(gem))
    # Invert vs typical parser Baker's / series-count ranking so the 50/50 overlay moves.
    gem["Baker's"] = np.linspace(0.97, 0.03, len(gem))
    gem.to_csv(data / "report_labels_v5.csv", index=False)

    def _run(strategy: str, work: Path):
        src_s = _source_for_strategy(strategy, "00_test")
        src_s = src_s.replace("ROOT = discover_root()", f"ROOT = Path({str(data)!r})")
        src_s = src_s.replace(
            'path = Path("/kaggle/working/submission.csv")',
            f"path = Path({str(work / 'submission.csv')!r})",
        )
        ns: dict = {}
        exec(compile(src_s, f"{strategy}_nb.py", "exec"), ns, ns)
        return pd.read_csv(work / "submission.csv"), ns

    out_lm, _ = _run("weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem", work_lm)
    out_bak, ns_bak = _run("weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem", work_bak)
    sample = pd.read_csv(data / "sample_submission.csv")
    assert list(out_bak[STUDY_ID_COL].astype(str)) == list(sample[STUDY_ID_COL].astype(str))
    assert out_bak[DEFAULT_TARGETS].isna().any().any() == False
    assert float(out_bak["Baker's"].std()) > 1e-6
    assert np.allclose(out_lm["ACL"].to_numpy(), out_bak["ACL"].to_numpy())
    assert np.allclose(out_lm["Effusion"].to_numpy(), out_bak["Effusion"].to_numpy())
    assert np.allclose(out_lm["Lateral Meniscus"].to_numpy(), out_bak["Lateral Meniscus"].to_numpy())
    assert not np.allclose(out_lm["Baker's"].to_numpy(), out_bak["Baker's"].to_numpy())
    assert ns_bak["MIX_TARGETS"]["weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem"] == {"Baker's"}


