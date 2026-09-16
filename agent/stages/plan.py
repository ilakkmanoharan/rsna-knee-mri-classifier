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
        "gold_meta_logit",
        "report_shrinkage_priors",
        "metadata_prior_blend",  # visual needs GPU dataset; keep safe default until assets ready
        "fluid_gate_metadata",
        "rank_ensemble_safe",
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
    if strategy == "gold_meta_logit":
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
        "",
        "### Acceptance",
        "",
        "- Notebook completes; submission status COMPLETE.",
        "- Public score > previous best, or document falsification in Analysis next cycle.",
        "",
    ]
    md_path.write_text("\n".join(lines))
    # machine-readable sidecar
    (out_dir / f"{day_id}_cycle{cycle_id}_plan.json").write_text(
        '{"strategy": "%s", "cycle": %d}\n' % (strategy, cycle_num)
    )
    return md_path
