"""Stage 4: write implementation plan/spec → Plans/."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def run_plan(
    out_dir: Path,
    research_md: Path,
    analysis_md: Path,
    hypothesis_md: Path,
    cycle_id: str,
    day_id: str,
    cycle_num: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    strategy = [
        "weak_rank_bakers_syn_llm",
        "weak_rank_bakers_syn_llm",
        "weak_rank_bakers_eff_llm",
        "weak_rank_bakers_acl_llm",
        "gold_meta_logit",
    ][cycle_num % 5]

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md_path = out_dir / f"{day_id}_cycle{cycle_id}_plan.md"
    lines = [
        f"# Next-submission plan & specification — day {day_id}, cycle {cycle_id}",
        "",
        f"_Generated {ts} by agent1._",
        "",
        "## Inputs",
        "",
        f"- Research: `{research_md.as_posix()}`",
        f"- Analysis: `{analysis_md.as_posix()}`",
        f"- Hypothesis: `{hypothesis_md.as_posix()}`",
        "",
        f"## Selected implementation strategy: `{strategy}`",
        "",
        "### Specification",
        "",
        "1. Offline Kaggle notebook (internet disabled) with competition data mounted under "
        "`/kaggle/input/competitions/rsna-knee-abnormality-detection`.",
        "2. Discover root by locating `sample_submission.csv` (do **not** rglob DICOM trees).",
        "3. Build predictions for every test `StudyInstanceUID` in sample order.",
        "4. Write `/kaggle/working/submission.csv` with exact columns/order; probs in (1e-6, 1-1e-6).",
        "5. Ensure nontrivial per-target variance on the hidden test set.",
        "",
        "### Strategy details",
        "",
    ]
    if strategy == "gold_rank_w50":
        lines += [
            "- Same two learned heads as frozen `gold_rank_interact` (public 0.517).",
            "- Load `train.csv` gold labels (only rows with non-null targets, ~58 studies).",
            "- Load `train_series.csv` and `test_series.csv`; do **not** walk DICOM trees.",
            "- Fit the frozen 7-d gold_meta_logit (λ=2): intercept, log1p(n), sag/cor/ax fractions,",
            "  fluid fraction, fat fraction.",
            "- Fit a 13-d interaction model (λ=3.5) that also includes sag/cor/ax × fluid and",
            "  sag/cor/ax × fat fractions.",
            "- Rank-transform each model's 12 heads; blend `0.50 * rank(7-d) + 0.50 * rank(interact)`.",
            "- Map blended ranks through gold prevalence. No Gaussian noise. No test reports.",
            "- If interact fit fails, fall back to 7-d ranks alone (the 0.514 path).",
            "- If gold+series < 20, fall back to hand-tuned metadata offsets.",
            "- Do **not** resubmit 0.60/0.40 (`gold_rank_interact`) as this cycle.",
            "",
        ]
    elif strategy == "weak_rank_bakers_syn_llm":
        lines += [
            "- Start from frozen `weak_rank_bakers_eff_llm` (public 0.524): gold_rank_w50 + parser Baker's 50/50 + LLM ACL 50/50 + LLM Effusion 50/50 if std>1e-8.",
            "- **Add stevenleehans `llm_labels_v4_blend` Synovitis 50/50 only if that LLM head has std>1e-8.**",
            "- Attach dataset_source `stevenleehans/rsna-knee-llm-report-labels`; keep `enable_internet` false.",
            "- Discover `llm_labels_v4_blend.csv` via explicit `/kaggle/input/datasets/<owner>/<slug>` mounts plus depth-6 BFS. Skip image/DICOM dirs; do **not** rglob DICOM trees.",
            "- Fit 7-d/13-d ridge heads on train-only LLM Synovitis soft labels. Drop any test UID.",
            "- Do **not** remix LLM Baker's or LLM MCL: Baker's LLM was constant (56872526); v4_blend MCL scored 0.510 poison (56938248).",
            "- Leave the other eight targets as the 0.524 stack (gold_rank_w50 + parser Baker's + LLM ACL + LLM Effusion).",
            "- **Never open test reports.** No live LLM/API calls.",
            "- If n_llm < 20, the CSV is missing, or Synovitis LLM std is 0, fall back to weak_rank_bakers_eff_llm (0.524).",
            "- Do **not** resubmit `weak_rank_bakers_eff_llm` (0.524 accepted, 56995081), `weak_rank_bakers_mcl_llm` (0.510 poison, 56938248), `weak_rank_bakers_acl_llm` (0.520 accepted, 56902005), `weak_rank_bakers_llm` (0.519 tie, 56872526), `weak_rank_bakers_dropfat` (0.519 tie, 56815942), `weak_rank_bakers_silence` (0.516, 56790917), `weak_rank_bakers_goldstd` (0.519 tie, 56767585), `weak_rank_bakers_mm_mix` (0.517, 56745095), `weak_rank_bakers_acl_mix` (0.519 tie, 56698950), `weak_rank_bakers_w40` (0.519 tie, 56664719), `weak_rank_bakers_w60` (0.518, 56631726), `weak_rank_bakers_mix` (0.519, 56600302), or earlier weak-label failures.",
            "",
        ]
    elif strategy == "weak_rank_bakers_eff_llm":
        lines += [
            "- Already accepted at public 0.524 (submission 56995081); keep as the frozen fallback.",
            "- Start from frozen `weak_rank_bakers_acl_llm` (public 0.520): gold_rank_w50 + parser Baker's 50/50 + LLM ACL 50/50 if std>1e-8.",
            "- **Add stevenleehans `llm_labels_v4_blend` Effusion 50/50 only if that LLM head has std>1e-8.**",
            "- Attach dataset_source `stevenleehans/rsna-knee-llm-report-labels`; keep `enable_internet` false.",
            "- Discover `llm_labels_v4_blend.csv` via explicit `/kaggle/input/datasets/<owner>/<slug>` mounts plus depth-6 BFS. Skip image/DICOM dirs; do **not** rglob DICOM trees.",
            "- Fit 7-d/13-d ridge heads on train-only LLM Effusion soft labels. Drop any test UID.",
            "- Do **not** remix LLM Baker's or LLM MCL: Baker's LLM was constant (56872526); v4_blend MCL scored 0.510 poison (56938248).",
            "- Leave the other nine targets as the 0.520 stack (gold_rank_w50 + parser Baker's + LLM ACL).",
            "- **Never open test reports.** No live LLM/API calls.",
            "- If n_llm < 20, the CSV is missing, or Effusion LLM std is 0, fall back to weak_rank_bakers_acl_llm (0.520).",
            "- Do **not** resubmit `weak_rank_bakers_mcl_llm` (0.510 poison, 56938248), `weak_rank_bakers_acl_llm` (0.520 accepted, 56902005), `weak_rank_bakers_llm` (0.519 tie, 56872526), `weak_rank_bakers_dropfat` (0.519 tie, 56815942), `weak_rank_bakers_silence` (0.516, 56790917), `weak_rank_bakers_goldstd` (0.519 tie, 56767585), `weak_rank_bakers_mm_mix` (0.517, 56745095), `weak_rank_bakers_acl_mix` (0.519 tie, 56698950), `weak_rank_bakers_w40` (0.519 tie, 56664719), `weak_rank_bakers_w60` (0.518, 56631726), `weak_rank_bakers_mix` (0.519, 56600302), or earlier weak-label failures.",
            "",
        ]
    elif strategy == "weak_rank_bakers_mcl_llm":
        lines += [
            "- Start from frozen `weak_rank_bakers_acl_llm` (public 0.520): gold_rank_w50 + parser Baker's 50/50 + LLM ACL 50/50 if std>1e-8.",
            "- **Add stevenleehans `llm_labels_v4_blend` MCL 50/50 only if that LLM head has std>1e-8.**",
            "- Attach dataset_source `stevenleehans/rsna-knee-llm-report-labels`; keep `enable_internet` false.",
            "- Discover `llm_labels_v4_blend.csv` via explicit `/kaggle/input/datasets/<owner>/<slug>` mounts plus depth-6 BFS. Skip image/DICOM dirs; do **not** rglob DICOM trees.",
            "- Fit 7-d/13-d ridge heads on train-only LLM MCL soft labels. Drop any test UID.",
            "- Do **not** remix LLM Baker's: 2026-10-06 (56872526) Baker's LLM scores were constant on the public 3-study sample.",
            "- Already falsified at public 0.510 (submission 56938248); v4_blend MCL is poison. Keep weak_rank_bakers_acl_llm as the frozen fallback.",
            "- **Never open test reports.** No live LLM/API calls.",
            "",
        ]
    elif strategy == "weak_rank_bakers_acl_llm":
        lines += [
            "- Start from frozen `gold_rank_w50` ranks, then apply the accepted parser Baker's-only 50/50 mix.",
            "- **Add stevenleehans `llm_labels_v4_blend` ACL 50/50 only if that LLM head has std>1e-8.**",
            "- Attach dataset_source `stevenleehans/rsna-knee-llm-report-labels`; keep `enable_internet` false.",
            "- Discover `llm_labels_v4_blend.csv` via explicit `/kaggle/input/datasets/<owner>/<slug>` mounts plus depth-6 BFS. Skip image/DICOM dirs; do **not** rglob DICOM trees.",
            "- Fit 7-d/13-d ridge heads on train-only LLM ACL soft labels. Drop any test UID.",
            "- Do **not** remix LLM Baker's: 2026-10-06 (56872526) found the CSV (n=4407) but Baker's LLM scores were constant on the public 3-study sample (tie 0.519).",
            "- Leave the other ten targets as frozen gold_rank_w50 (plus parser Baker's).",
            "- **Never open test reports.** No live LLM/API calls.",
            "- Already accepted at public 0.520 (submission 56902005); keep as the frozen fallback.",
            "- Do **not** resubmit `weak_rank_bakers_llm` (0.519 tie, 56872526), `weak_rank_bakers_dropfat` (0.519 tie, 56815942), `weak_rank_bakers_silence` (0.516, 56790917), `weak_rank_bakers_goldstd` (0.519 tie, 56767585), `weak_rank_bakers_mm_mix` (0.517, 56745095), `weak_rank_bakers_acl_mix` (0.519 tie, 56698950), `weak_rank_bakers_w40` (0.519 tie, 56664719), `weak_rank_bakers_w60` (0.518, 56631726), `weak_rank_bakers_mix` (0.519, 56600302), or earlier weak-label failures.",
            "",
        ]
    elif strategy == "weak_rank_bakers_llm":
        lines += [
            "- Start from frozen `gold_rank_w50` ranks (7-d λ=2 + 13-d λI=3.5, 0.50/0.50, fat features kept).",
            "- **Replace the keyword-parser Baker's weak head with stevenleehans `llm_labels_v4_blend`.**",
            "- Attach dataset_source `stevenleehans/rsna-knee-llm-report-labels`; keep `enable_internet` false.",
            "- Discover `llm_labels_v4_blend.csv` with a depth-6 BFS under `/kaggle/input` and `/kaggle/input/datasets/<owner>/<slug>` — skip image/DICOM dirs; do **not** rglob DICOM trees.",
            "- Fit 7-d/13-d ridge heads on train-only LLM Baker's soft labels. Drop any test UID.",
            "- **Mix only Baker's:** `0.50 * gold_rank + 0.50 * llm_weak_rank`.",
            "- Leave the other eleven targets as frozen gold_rank_w50.",
            "- **Never open test reports.** No live LLM/API calls.",
            "- Already falsified at public 0.519 tie (submission 56872526); Baker's LLM head was constant. Keep weak_rank_bakers_mix as the frozen fallback.",
            "",
        ]
    elif strategy == "weak_rank_bakers_dropfat":
        lines += [
            "- Start from frozen `gold_rank_w50` ranks, then apply the accepted Baker's-only 50/50 mix.",
            "- **Drop fat-suppression and plane×fat features.** Afshar grouped-CV notes Fluid_Sensitive ≡ Fat_Suppression on all 24,371 series, so fat is a duplicate on n=58.",
            "- Fit a 6-d additive head (intercept, log1p(n), sag/cor/ax, fluid) at λ=2 and a 9-d interact head (plus sag/cor/ax × fluid) at λI=3.5; blend 0.50/0.50.",
            "- Separately fit named-only weak heads on the same no-fat features (gold 0/1 + parser pos/neg on ACL / Baker's / MCL; unmentioned masked).",
            "- **Mix only Baker's:** `0.50 * gold_rank + 0.50 * named_weak_rank`.",
            "- Leave the other eleven targets as the no-fat gold ranks.",
            "- Discover `train.csv` Report column only. **Never open test reports.**",
            "- Already falsified at public 0.519 tie (submission 56815942); keep weak_rank_bakers_mix as the frozen fallback.",
            "",
        ]
    elif strategy == "weak_rank_bakers_silence":
        lines += [
            "- Start from frozen `gold_rank_w50` ranks, then apply the accepted Baker's-only 50/50 mix.",
            "- Separately fit named-only weak heads (gold 0/1 + parser pos/neg on ACL / Baker's / MCL).",
            "- **Baker's unmentioned → neg=0.12** (discussion 733932: silence ≈ negative). Keep ACL/MCL unmentioned masked.",
            "- **Mix only Baker's:** `0.50 * gold_rank + 0.50 * named_weak_rank`.",
            "- Leave the other eleven targets as pure gold_rank_w50 (including ACL, MCL, and both menisci).",
            "- Discover `train.csv` Report column only. **Never open test reports.**",
            "- Already falsified at public 0.516 (submission 56790917); keep weak_rank_bakers_mix as the frozen fallback.",
            "",
        ]
    elif strategy == "weak_rank_bakers_goldstd":
        lines += [
            "- Start from frozen `gold_rank_w50` ranks, then apply the accepted Baker's-only 50/50 mix.",
            "- Separately fit named-only weak heads (gold 0/1 + parser pos/neg on ACL / Baker's / MCL).",
            "- **Standardize those weak heads on the 58 gold studies' series-feature μ/σ**, not the 4,407-row weak matrix.",
            "- **Mix only Baker's:** `0.50 * gold_rank + 0.50 * named_weak_rank`.",
            "- Leave the other eleven targets as pure gold_rank_w50 (including ACL, MCL, and both menisci).",
            "- Discover `train.csv` Report column only. **Never open test reports.**",
            "- Already falsified at public 0.519 tie (submission 56767585); keep weak_rank_bakers_mix as the frozen fallback.",
            "",
        ]
    elif strategy == "weak_rank_bakers_mm_mix":
        lines += [
            "- Start from frozen `gold_rank_w50` ranks, then apply the accepted Baker's 50/50 mix plus Medial Meniscus.",
            "- Separately fit named-only weak heads (gold 0/1 + parser pos/neg on ACL / Baker's / MCL / Medial Meniscus).",
            "- **Mix Baker's and Medial Meniscus:** `0.50 * gold_rank + 0.50 * named_weak_rank` on those two columns.",
            "- Leave ACL, MCL, and the other nine targets as pure gold_rank_w50.",
            "- Discover `train.csv` Report column only. **Never open test reports.**",
            "- Already falsified at public 0.517 (submission 56745095); keep weak_rank_bakers_mix as the frozen fallback.",
            "",
        ]
    elif strategy == "weak_rank_bakers_acl_mix":
        lines += [
            "- Start from frozen `gold_rank_w50` ranks, then apply the accepted Baker's 50/50 mix plus ACL.",
            "- Separately fit named-only weak heads (gold 0/1 + parser pos/neg on ACL / Baker's / MCL).",
            "- **Mix Baker's and ACL:** `0.50 * gold_rank + 0.50 * named_weak_rank` on those two columns.",
            "- Leave MCL and the other ten targets as pure gold_rank_w50.",
            "- Discover `train.csv` Report column only. **Never open test reports.**",
            "- Already falsified at public 0.519 tie (submission 56698950); keep weak_rank_bakers_mix as the frozen fallback.",
            "",
        ]
    elif strategy == "weak_rank_bakers_w40":
        lines += [
            "- Start from frozen `gold_rank_w50` ranks, then apply the accepted Baker's mix with less weak weight.",
            "- Separately fit named-only weak heads (gold 0/1 + parser pos/neg on ACL / Baker's / MCL).",
            "- **Mix only Baker's:** `0.60 * gold_rank + 0.40 * named_weak_rank`.",
            "- Leave the other eleven targets as pure gold_rank_w50 (including ACL and MCL).",
            "- Discover `train.csv` Report column only. **Never open test reports.**",
            "- Already falsified at public 0.519 tie (submission 56664719); keep weak_rank_bakers_mix as the frozen fallback.",
            "",
        ]
    elif strategy == "weak_rank_bakers_w60":
        lines += [
            "- Start from frozen `gold_rank_w50` ranks, then apply the accepted Baker's mix with more weak weight.",
            "- Separately fit named-only weak heads (gold 0/1 + parser pos/neg on ACL / Baker's / MCL).",
            "- **Mix only Baker's:** `0.40 * gold_rank + 0.60 * named_weak_rank`.",
            "- Leave the other eleven targets as pure gold_rank_w50 (including ACL and MCL).",
            "- Discover `train.csv` Report column only. **Never open test reports.**",
            "- Already falsified at public 0.518 (submission 56631726); keep weak_rank_bakers_mix as the frozen fallback.",
            "",
        ]
    elif strategy == "weak_rank_bakers_mix":
        lines += [
            "- Start from frozen `gold_rank_w50` ranks (public 0.518): 7-d λ=2 + 13-d λI=3.5, 0.50/0.50,",
            "  standardized on the 58 gold studies only.",
            "- Separately fit named-only weak heads (gold 0/1 + parser pos/neg on ACL / Baker's / MCL).",
            "- **Mix only Baker's:** `0.50 * gold_rank + 0.50 * named_weak_rank`.",
            "- Leave the other eleven targets as pure gold_rank_w50 (including ACL and MCL).",
            "- Discover `train.csv` Report column only. **Never open test reports.**",
            "- If n_weak < 20 or Report is missing, fall back to gold_rank_w50.",
            "- Already accepted at public 0.519 (submission 56600302); keep as the frozen fallback.",
            "",
        ]
    elif strategy == "weak_rank_named_mix":
        lines += [
            "- Start from frozen `gold_rank_w50` ranks (public 0.518): 7-d λ=2 + 13-d λI=3.5, 0.50/0.50,",
            "  standardized on the 58 gold studies only.",
            "- Separately fit named-only weak heads (gold 0/1 + parser pos/neg on ACL / Baker's / MCL).",
            "- **Mix only those three columns:** `0.50 * gold_rank + 0.50 * named_weak_rank`.",
            "- Leave the other nine targets as pure gold_rank_w50.",
            "- Discover `train.csv` Report column only. **Never open test reports.**",
            "- If n_weak < 20 or Report is missing, fall back to gold_rank_w50.",
            "- Already falsified at public 0.514 (submission 56571590); keep only as a documented fallback.",
            "",
        ]
    elif strategy == "weak_rank_named":
        lines += [
            "- Same two series-metadata heads as frozen `gold_rank_w50` (public 0.518).",
            "- **Keep gold 0/1 on the 58 labeled studies.**",
            "- Parser pos/neg only for **named objects** ACL / Baker's / MCL (discussion 734117).",
            "- Mask unmentioned/unc/hist and all unlabeled graded/unstated targets as NaN.",
            "- Discover `train.csv` Report column only. **Never open test reports.**",
            "- Fit 7-d (λ=2.0) and 13-d interact (λ=3.5); blend `0.50 * rank(7-d) + 0.50 * rank(interact)`.",
            "- Map blended ranks through gold prevalence. No Gaussian noise.",
            "- If n_weak < 20 or Report is missing, fall back to gold_rank_w50.",
            "- Do **not** resubmit `weak_rank_confident` (0.511, 56513588), `weak_rank_goldfill` (0.504), or `weak_rank_calibrate` (0.499).",
            "",
        ]
    elif strategy == "weak_rank_confident":
        lines += [
            "- Same two series-metadata heads as frozen `gold_rank_w50` (public 0.518).",
            "- **Keep gold 0/1 on the 58 labeled studies.**",
            "- On unlabeled reports, keep only parser **pos/neg**. Mask unmentioned / uncertain / historical as NaN",
            "  (discussion 734117: empty extractions stay unlabeled, not all-negative).",
            "- Discover `train.csv` Report column only. **Never open test reports.**",
            "- Fit 7-d (λ=2.0) and 13-d interact (λ=3.5); blend `0.50 * rank(7-d) + 0.50 * rank(interact)`.",
            "- Map blended ranks through gold prevalence. No Gaussian noise.",
            "- If n_weak < 20 or Report is missing, fall back to gold_rank_w50.",
            "- Do **not** resubmit `weak_rank_goldfill` (0.504, 56484736) or parser-only `weak_rank_calibrate` (0.499).",
            "",
        ]
    elif strategy == "weak_rank_goldfill":
        lines += [
            "- Same two series-metadata heads as frozen `gold_rank_w50` (public 0.518).",
            "- **Keep gold 0/1 on the 58 labeled studies.** Use parser soft labels only when the gold cell is missing.",
            "- Discover `train.csv` Report column only. **Never open test reports.**",
            "- Fit 7-d (λ=2.0) and 13-d interact (λ=3.5); blend `0.50 * rank(7-d) + 0.50 * rank(interact)`.",
            "- Map blended ranks through gold prevalence. No Gaussian noise.",
            "- If n_weak < 20 or Report is missing, fall back to gold_rank_w50.",
            "- Already falsified at public 0.504 (submission 56484736); keep only as a documented fallback.",
            "",
        ]
    elif strategy == "weak_rank_calibrate":
        lines += [
            "- Same two series-metadata heads as frozen `gold_rank_w50` (public 0.518), but **fit on train-report soft labels** (n≈4407) instead of the 58 gold rows.",
            "- Discover `train.csv` Report column only. **Never open test reports** (test.csv Report is absent at scoring).",
            "- Multilingual keyword matcher + left/right negation window (Turkish `izlenmedi` / `görülmedi`).",
            "- Soft labels: pos=0.85, neg=0.12, uncertain=0.50, historical=0.40, unmentioned=0.38.",
            "- Fit 7-d (λ=2.0) and 13-d interact (λ=3.5); blend `0.50 * rank(7-d) + 0.50 * rank(interact)`.",
            "- Map blended ranks through gold prevalence (monotone; AUC-invariant). No Gaussian noise.",
            "- If n_weak < 100 or Report is missing, fall back to gold_rank_w50 (gold-only 7-d/13-d).",
            "- Do **not** resubmit gold_rank_lam2 (0.515), w40/w50/w70 as this cycle.",
            "",
        ]
    elif strategy == "gold_rank_lam2":
        lines += [
            "- Same 0.50/0.50 rank-blend as frozen `gold_rank_w50` (public 0.518).",
            "- Keep 7-d ridge at λ=2.0. Fit the 13-d interact head at **λ=2.0** (was 3.5).",
            "- Map blended ranks through gold prevalence. No Gaussian noise. No test reports.",
            "- If interact fit fails, fall back to 7-d ranks alone.",
            "- Do **not** resubmit blend-weight ablations (w40/w50/w70) as this cycle.",
            "- Already falsified at public 0.515 (submission 56418615); keep only as a documented fallback.",
            "",
        ]
    elif strategy == "gold_rank_w40":
        lines += [
            "- Same two learned heads as frozen `gold_rank_w50` (public 0.518).",
            "- Rank-blend `0.40 * rank(7-d) + 0.60 * rank(interact)` (more weight on the interact head).",
            "- Map blended ranks through gold prevalence. No Gaussian noise. No test reports.",
            "- If interact fit fails, fall back to 7-d ranks alone.",
            "- Do **not** resubmit 0.50/0.50 (`gold_rank_w50`) or 0.70/0.30 (`gold_rank_w70`) as this cycle.",
            "",
        ]
    elif strategy == "gold_rank_w70":
        lines += [
            "- Same two learned heads as frozen `gold_rank_w50` (public 0.518).",
            "- Rank-blend `0.70 * rank(7-d) + 0.30 * rank(interact)` (more weight on the additive head).",
            "- Map blended ranks through gold prevalence. No Gaussian noise. No test reports.",
            "- If interact fit fails, fall back to 7-d ranks alone.",
            "- Do **not** resubmit 0.50/0.50 (`gold_rank_w50`) as this cycle.",
            "",
        ]
    elif strategy == "gold_rank_interact":
        lines += [
            "- Load `train.csv` gold labels (only rows with non-null targets, ~58 studies).",
            "- Load `train_series.csv` and `test_series.csv`; do **not** walk DICOM trees.",
            "- Fit the frozen 7-d gold_meta_logit (λ=2): intercept, log1p(n), sag/cor/ax fractions,",
            "  fluid fraction, fat fraction.",
            "- Fit a 13-d interaction model (λ=3.5) that also includes sag/cor/ax × fluid and",
            "  sag/cor/ax × fat fractions. Stronger L2 offsets the extra dimensions on n≈58.",
            "- Rank-transform each model's 12 heads; blend `0.60 * rank(7-d) + 0.40 * rank(interact)`.",
            "- Map blended ranks through gold prevalence. No Gaussian noise. No test reports.",
            "- If interact fit fails, fall back to 7-d ranks alone (the 0.514 path).",
            "- If gold+series < 20, fall back to hand-tuned metadata offsets.",
            "",
        ]
    elif strategy == "gold_meta_logit":
        lines += [
            "- Load `train.csv` gold labels (only rows with non-null targets, ~58 studies).",
            "- Load `train_series.csv` and `test_series.csv`; do **not** walk DICOM trees.",
            "- Per study, build a 7-d vector: intercept, log1p(n_series), sagittal/coronal/axial",
            "  fractions, fluid-sensitive fraction, fat-suppression fraction.",
            "- Standardize non-intercept features on gold; fit ridge logistic (λ≈2, 40 IRLS steps)",
            "  independently per target. Skip a head if it has <12 labels or no class contrast.",
            "- Score every test study; rank-transform each column; map ranks through gold prevalence.",
            "- If train_series is missing or gold+series < 20, fall back to hand-tuned metadata offsets.",
            "- No Gaussian rank-scrambling noise. No test reports. No pixel model.",
            "",
        ]
    elif strategy == "metadata_prior_blend":
        lines += [
            "- Load `train.csv` gold labels → per-target prevalence.",
            "- Load `test_series.csv` → for each study compute:",
            "  - plane availability (sag/cor/ax)",
            "  - counts of fluid-sensitive and fat-suppressed series",
            "- Map features to per-target logit offsets (hand-specified soft preferences from ontology).",
            "- `prob = sigmoid(logit(prev) + offset)`; clip.",
            "- Deterministic seed 42.",
            "",
        ]
    elif strategy == "report_shrinkage_priors":
        lines += [
            "- Same metadata offsets as metadata_prior_blend.",
            "- Additionally shrink prevalence using train-report evidence-state frequencies "
            "(precomputed constants embedded in notebook — no test report I/O).",
            "",
        ]
    elif strategy == "fluid_gate_metadata":
        lines += [
            "- Emphasize fluid-sensitive series count for Effusion/Synovitis/Contusion offsets.",
            "- Keep ligament offsets tied to sagittal/coronal availability.",
            "",
        ]
    else:
        lines += [
            "- Average ranks of (prevalence vector, metadata-blend vector) per target, rescale to probabilities.",
            "",
        ]

    lines += [
        "### Out of scope this cycle",
        "",
        "- Full visual encoder training on all train DICOMs (schedule when GPU + data mount time allows).",
        "- LLM/API calls during inference.",
        "- Test radiology reports (forbidden).",
        "",
        "### Acceptance",
        "",
        "- Notebook completes; submission status COMPLETE.",
            "- Public score > 0.524 (frozen weak_rank_bakers_eff_llm), or document falsification in Analysis next cycle.",
            "- Keep weak_rank_bakers_eff_llm as the fallback notebook if this ablation does not beat 0.524.",
        "",
    ]
    md_path.write_text("\n".join(lines))
    # machine-readable sidecar
    (out_dir / f"{day_id}_cycle{cycle_id}_plan.json").write_text(
        '{"strategy": "%s", "cycle": %d}\n' % (strategy, cycle_num)
    )
    return md_path
