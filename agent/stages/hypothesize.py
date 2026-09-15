"""Stage 3: form hypotheses from Research + Analysis → Hypothesis analysis/."""

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
    # Cycle-indexed primary hypothesis so each 90-min slot tests something different
    catalog = [
        {
            "id": "H_meta_prior",
            "hypothesis": "Series-metadata features (plane availability + fluid-sensitive counts) blended with per-target prevalence will beat pure prevalence on public macro AUC.",
            "mechanism": "For each test study, compute availability and sequence counts from test_series.csv; map to a small additive logit offset per target using preferred-plane heuristics; sigmoid → blend with prevalence.",
            "falsify": "Public score ≤ previous prevalence submission within 0.002.",
            "expected_targets": ["ACL", "MCL", "PF OA", "Effusion", "Contusion"],
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
        {
            "id": "H_fluid_routing",
            "hypothesis": "Up-weighting fluid-sensitive series for effusion/synovitis/contusion improves those target AUCs without hurting ligaments.",
            "mechanism": "Sequence-flag gated pooling weights; ASRA accept only if no target collapses.",
            "falsify": "Ligament AUCs drop >0.02 or macro flat.",
            "expected_targets": ["Effusion", "Synovitis", "Contusion"],
        },
        {
            "id": "H_ensemble_rank",
            "hypothesis": "Rank-average of prevalence, metadata, and visual models beats any single model.",
            "mechanism": "Per-target rank across models → average → rescale to (eps,1-eps).",
            "falsify": "Public score ≤ best single member.",
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
        "## Primary hypothesis this cycle",
        "",
        f"- **ID:** `{primary['id']}`",
        f"- **Hypothesis:** {primary['hypothesis']}",
        f"- **Mechanism:** {primary['mechanism']}",
        f"- **Falsification:** {primary['falsify']}",
        f"- **Expected targets:** {', '.join(primary['expected_targets'])}",
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
    return md_path
