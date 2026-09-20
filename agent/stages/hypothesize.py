"""Stage 3: form hypotheses from Research + Analysis → Hypothesis/."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def run_hypothesize(
    out_dir: Path,
    research_md: Path,
    analysis_md: Path,
    cycle_id: str,
    day_id: str,
    cycle_num: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Cycle-indexed primary hypothesis so each slot tests something different
    catalog = [
        {
            "id": "H_gold_rank_lam2",
            "hypothesis": "If the 0.40/0.60 and 0.50/0.50 blends both score 0.518, the 13-d interact head is over-regularized (λ=3.5) and nearly collinear with the 7-d ranks; lowering λI to 2.0 at the frozen 0.50 blend will make plane×protocol ranks distinctive and beat 0.518.",
            "mechanism": "Same 7-d (λ=2) + 13-d interact heads as gold_rank_w50; keep 0.50·rank(7-d)+0.50·rank(interact); fit the interact head with λ=2.0 instead of 3.5. Fallback to 7-d if interact fit fails. No test reports, no pixels, no Gaussian noise.",
            "falsify": "Public score ≤ 0.518 (frozen gold_rank_w50 / tied gold_rank_w40).",
            "expected_targets": ["Effusion", "Synovitis", "Contusion", "ACL", "PF OA"],
        },
        {
            "id": "H_gold_rank_w40",
            "hypothesis": "Giving the 13-d interact head majority weight (0.40/0.60) will beat 0.518 because public LB has risen monotonically as interact weight went 0.00→0.30→0.40→0.50.",
            "mechanism": "Same two learned heads as gold_rank_w50; blend 0.40·rank(7-d)+0.60·rank(interact). Fallback to 7-d if interact fit fails. No test reports, no pixels, no Gaussian noise.",
            "falsify": "Public score ≤ 0.518 (frozen gold_rank_w50).",
            "expected_targets": ["Effusion", "Synovitis", "Contusion", "ACL", "PF OA"],
        },
        {
            "id": "H_gold_rank_w70",
            "hypothesis": "Putting more weight on the proven 7-d ranks (0.70/0.30) will beat 0.518 if the 13-d interact head is noisy on rare targets at equal weight.",
            "mechanism": "Same two learned heads as gold_rank_w50; blend 0.70·rank(7-d)+0.30·rank(interact). Fallback to 7-d if interact fit fails. No test reports, no pixels, no Gaussian noise.",
            "falsify": "Public score ≤ 0.518 (frozen gold_rank_w50).",
            "expected_targets": ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"],
        },
        {
            "id": "H_gold_rank_w50",
            "hypothesis": "Equal rank-blend (0.50·7-d + 0.50·13-d interact) of the frozen gold_rank_interact heads will lift public macro ROC-AUC above 0.517 because the interaction head already added +0.003 at 0.40 weight.",
            "mechanism": "Fit the frozen 7-d ridge logistic (λ=2) and the 13-d plane×fluid / plane×fat model (λ=3.5) on the 58 gold labels using only train_series.csv flags. Rank-transform each head, then 0.50·rank(7-d)+0.50·rank(interact). Map ranks through gold prevalence. No test reports, no DICOM pixels, no Gaussian noise.",
            "falsify": "Public score ≤ 0.517 (frozen gold_rank_interact 0.60/0.40) or a notebook error falls back to 7-d ranks only.",
            "expected_targets": ["Effusion", "Synovitis", "Contusion", "ACL", "PF OA"],
        },
        {
            "id": "H_gold_rank_interact",
            "hypothesis": "Rank-blending the accepted 7-d gold_meta_logit ranks (public 0.514) with a stronger-regularized plane×fluid / plane×fat interaction logit will lift macro ROC-AUC above 0.514 without replacing the frozen ranking.",
            "mechanism": "Fit the frozen 7-d ridge logistic (λ=2) and a 13-d interaction model (λ=3.5) on the 58 gold labels using only train_series.csv flags. Rank-transform each head, then 0.60·rank(7-d)+0.40·rank(interact). Map ranks through gold prevalence. No test reports, no DICOM pixels, no Gaussian noise.",
            "falsify": "Public score ≤ 0.514 (frozen gold_meta_logit) or a notebook error falls back to 7-d ranks only.",
            "expected_targets": ["Effusion", "Synovitis", "Contusion", "ACL", "Medial OA"],
        },
        {
            "id": "H_gold_meta_logit",
            "hypothesis": "Ridge logistic models fit on the 58 gold-labeled train studies using series-metadata features will rank test studies better than hand-tuned plane/fluid offsets (public macro AUC > 0.499).",
            "mechanism": "From train_series.csv/test_series.csv build per-study vectors (log series count, sagittal/coronal/axial fractions, fluid-sensitive fraction, fat-suppression fraction). Fit L2-regularized logistic regression independently per target on gold labels only. Apply weights to test metadata, rank-transform each column, map through prevalence. No test reports, no DICOM pixels.",
            "falsify": "Public score ≤ 0.499 (best hand-tuned metadata) or a notebook error falls back without lift.",
            "expected_targets": ["ACL", "MCL", "Effusion", "Synovitis", "PF OA"],
        },
        {
            "id": "H_ensemble_rank",
            "hypothesis": "Rank-average of prevalence and metadata models beats any single model.",
            "mechanism": "Per-target rank across models → average → rescale to (eps,1-eps).",
            "falsify": "Public score ≤ best single member.",
            "expected_targets": ["*"],
        },
        {
            "id": "H_report_shrinkage",
            "hypothesis": "Train-time report soft-label shrinkage priors (global target means conditioned on evidence state frequencies) improve calibration of the metadata model without using test reports.",
            "mechanism": "Parse train reports offline; estimate P(y=1|evidence_state) on gold; at inference use only metadata + global priors (no report text).",
            "falsify": "No lift vs H_meta_prior or target collapse on rare labels.",
            "expected_targets": ["Medial Meniscus", "Lateral Meniscus", "Synovitis"],
        },
        {
            "id": "H_plane_slots_visual",
            "hypothesis": "Frozen EfficientNet-B0 with plane slots trained on real (or JPEG) MRI lifts macro AUC over metadata-only.",
            "mechanism": "Kaggle GPU notebook trains heads on mounted train_series; ensemble folds; infer offline.",
            "falsify": "OOF macro ≤ prevalence or public score drops.",
            "expected_targets": ["*"],
        },
    ]
    primary = catalog[cycle_num % len(catalog)]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md_path = out_dir / f"{day_id}_cycle{cycle_id}_hypotheses.md"
    lines = [
        f"# Hypothesis analysis — competition day {day_id}, cycle {cycle_id}",
        "",
        f"_Generated {ts} by agent1._",
        "",
        f"Inputs: `{research_md.name}`, `{analysis_md.name}`",
        "",
        "## Why this hypothesis now",
        "",
        "gold_rank_w50 (0.50·7-d + 0.50·13-d plane×protocol, λI=3.5) is the frozen public baseline at",
        "**0.518** (submission 56322623). gold_rank_w40 (0.40/0.60) **tied 0.518** (56382621) and is",
        "falsified as an improvement; gold_rank_w70 scored 0.516. The weight curve has plateaued, so",
        "the two heads are likely too similar under λI=3.5. The next testable change is **λI=2.0**",
        "at the frozen 0.50 blend (`gold_rank_lam2`), not another blend weight. Visual MRI encoders",
        "stay out of scope until this metadata ablation beats 0.518 or is clearly falsified.",
        "",
        "## Primary hypothesis this cycle",
        "",
        f"- **ID:** `{primary['id']}`",
        f"- **Hypothesis:** {primary['hypothesis']}",
        f"- **Mechanism:** {primary['mechanism']}",
        f"- **Falsification:** {primary['falsify']}",
        f"- **Expected targets:** {', '.join(primary['expected_targets'])}",
        "",
        "## Test protocol",
        "",
        "- One offline Kaggle notebook; internet disabled; discover `sample_submission.csv`.",
        "- Fit only on gold rows of `train.csv` joined to `train_series.csv`.",
        "- Infer from `test_series.csv` metadata only — never open test radiology reports.",
        "- Accept into the frozen baseline iff public score > 0.518 and status COMPLETE.",
        "- Otherwise keep `gold_rank_w50` (0.518) as the fallback path.",
        "",
        "## Backlog (ASRA queue)",
        "",
    ]
    for h in catalog:
        mark = "← primary" if h["id"] == primary["id"] else ""
        lines.append(f"- `{h['id']}` {mark}: {h['hypothesis']}")
    lines += [
        "",
        "## Decision rule",
        "",
        "Accept into the next frozen baseline only if public score improves and no unexplained single-target collapse. "
        "Otherwise keep previous best notebook as the fallback submission path.",
        "",
    ]
    md_path.write_text("\n".join(lines))
    repo_root = Path(__file__).resolve().parents[2]
    if out_dir.resolve() == (repo_root / "Hypothesis").resolve():
        alias_dir = repo_root / "Hypothesis analysis"
        alias_dir.mkdir(parents=True, exist_ok=True)
        (alias_dir / md_path.name).write_text(md_path.read_text())
    return md_path
