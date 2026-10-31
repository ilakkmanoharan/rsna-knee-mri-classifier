"""Stage 2: pull Kaggle submission logs → Analysis/."""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def fetch_submissions(competition: str) -> list[dict[str, Any]]:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    # Prefer structured API; fall back to CLI-like CSV parsing via submissions list
    try:
        subs = api.competition_submissions(competition)
        rows = []
        for s in subs:
            rows.append(
                {
                    "fileName": getattr(s, "fileName", None) or getattr(s, "file_name", None),
                    "date": str(getattr(s, "date", "") or getattr(s, "submitted_by", "")),
                    "description": getattr(s, "description", None) or getattr(s, "error_description", None),
                    "status": str(getattr(s, "status", "")),
                    "publicScore": _to_float(getattr(s, "publicScore", None) or getattr(s, "public_score", None)),
                    "privateScore": _to_float(getattr(s, "privateScore", None) or getattr(s, "private_score", None)),
                    "ref": getattr(s, "ref", None),
                }
            )
        if rows:
            return rows
    except Exception as exc:  # noqa: BLE001
        logger.warning("competition_submissions failed: %s", exc)

    # Fallback: subprocess CSV
    import subprocess

    proc = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", competition, "--csv"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    reader = csv.DictReader(io.StringIO(proc.stdout))
    rows = []
    for r in reader:
        rows.append(
            {
                "fileName": r.get("fileName") or r.get("file_name"),
                "date": r.get("date"),
                "description": r.get("description"),
                "status": r.get("status"),
                "publicScore": _to_float(r.get("publicScore") or r.get("public_score")),
                "privateScore": _to_float(r.get("privateScore") or r.get("private_score")),
            }
        )
    return rows


def _to_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return None


def run_analysis(out_dir: Path, competition: str, cycle_id: str, day_id: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = fetch_submissions(competition)
    scores = [r["publicScore"] for r in rows if r.get("publicScore") is not None]
    best = max(scores) if scores else None
    latest = rows[0] if rows else None

    why_low: list[str] = []
    improvements: list[str] = []
    if not rows:
        why_low.append("No prior submissions found — establish a schema-valid baseline first.")
        improvements.append("Submit prevalence baseline, then metadata-aware ranking model.")
    else:
        if best is not None and best < 0.55:
            why_low.append(
                f"Best public score {best:.3f} is near random/prevalence floor for macro AUC — "
                "predictions likely lack study-level discriminative signal."
            )
            why_low.append(
                "Score ladder: Stage-B prevalence+noise 0.494; hand-tuned metadata_prior_blend / "
                "report_shrinkage_priors 0.498; fluid_gate_metadata 0.499–0.505; gold_meta_logit 0.514; "
                "gold_rank_interact 0.517; gold_rank_w50 0.518 (accepted, submission 56322623); "
                "gold_rank_w40 tied 0.518; gold_rank_lam2 scored 0.515 (falsified, 56418615); "
                "weak_rank_calibrate scored 0.499 (56455239); weak_rank_goldfill scored 0.504 "
                "(56484736); weak_rank_confident scored 0.511 (56513588); weak_rank_named scored "
                "0.502 (56541791); weak_rank_named_mix scored 0.514 (56571590); "
                "weak_rank_bakers_mix scored 0.519 (56600302, accepted); "
                "weak_rank_bakers_w60 scored 0.518 (56631726, falsified); "
                "weak_rank_bakers_w40 scored 0.519 (56664719, tie); "
                "weak_rank_bakers_acl_mix scored 0.519 (56698950, tie); "
                "weak_rank_bakers_mm_mix scored 0.517 (56745095, falsified); "
                "weak_rank_bakers_goldstd scored 0.519 (56767585, tie); "
                "weak_rank_bakers_silence scored 0.516 (56790917, poison); "
                "weak_rank_bakers_dropfat scored 0.519 (56815942, tie); "
                "weak_rank_bakers_llm scored 0.518 (56844594) on a discovery miss, then "
                "0.519 (56872526) after finding v4_blend (n=4407) — Baker's LLM head was "
                "constant on the public 3-study sample (tie); "
                "weak_rank_bakers_acl_llm scored 0.520 (56902005, accepted); "
                "weak_rank_bakers_mcl_llm scored 0.510 (56938248, poison); "
                "weak_rank_bakers_eff_llm scored 0.524 (56995081, accepted); "
                "weak_rank_bakers_syn_llm scored 0.522 (57032978, poison; public "
                "Synovitis flattened to 0.465517); "
                "weak_rank_bakers_syn_from_eff scored 0.521 (57064376, poison); "
                "weak_rank_bakers_cont_llm scored 0.519 (57097591, poison; public "
                "Contusion compressed 0.265–0.453 → 0.296–0.359); "
                "weak_rank_bakers_frac_llm scored 0.524 (57132860, tie; public "
                "Fracture 0.217–0.404); "
                "weak_rank_bakers_moa_llm scored 0.523 (57169674, poison; public "
                "Medial OA compressed 0.134–0.384 → 0.134–0.321); "
                "weak_rank_bakers_loa_llm scored 0.523 (57218864, poison; public "
                "3-study Lateral OA stayed 0.065–0.315); "
                "weak_rank_bakers_pfoa_llm scored 0.523 (57257540, poison; public "
                "3-study PF OA stayed 0.237–0.487); "
                "weak_rank_bakers_frac_hyb_llm scored 0.527 (57298558, accepted; public "
                "Fracture 0.248–0.373); "
                "weak_rank_bakers_frac_hyb_w60 scored 0.526 (57341916, poison; public "
                "Fracture compressed 0.248–0.373 → 0.248–0.360); "
                "weak_rank_bakers_frac_hyb_w40 scored 0.526 (57385111, poison; public "
                "Fracture 0.248–0.385); "
                "weak_rank_bakers_frac_gem_llm scored 0.525 (57425148, poison; public "
                "Fracture 0.217–0.404); "
                "weak_rank_bakers_frac_hyb_gem_llm scored 0.527 (57468780, tie; public "
                "Fracture 0.217–0.373); "
                "weak_rank_bakers_frac_hyb_mm_llm scored 0.527 (57512380, tie; public "
                "Medial Meniscus 0.323–0.542); "
                "weak_rank_bakers_frac_hyb_lm_llm scored 0.529 (57551203, accepted; public "
                "Lateral Meniscus 0.303–0.522); "
                "weak_rank_bakers_frac_hyb_lm_w60 scored 0.529 (57584615, tie; public "
                "Lateral Meniscus 0.297–0.522); "
                "weak_rank_bakers_frac_hyb_lm_w40 scored 0.528 (57619038, poison; public "
                "Lateral Meniscus 0.309–0.522); "
                "weak_rank_bakers_frac_hyb_lm_eff_w60 scored 0.530 (57656351, accepted; public "
                "Effusion 0.478–0.703); "
                "weak_rank_bakers_frac_hyb_lm_eff_w40 scored 0.528 (57691260, poison; public "
                "Effusion compressed 0.478–0.703 → 0.478–0.691); "
                "weak_rank_bakers_frac_hyb_lm_eff_acl_w60 scored 0.530 (57726811, tie; public "
                "ACL 0.314–0.539); "
                "weak_rank_bakers_frac_hyb_lm_eff_acl_w40 scored 0.529 (57760822, poison; public "
                "ACL compressed 0.314–0.539 → 0.326–0.539); "
                "weak_rank_bakers_frac_hyb_lm_eff_acl_gem scored 0.531 (57765842, accepted; public "
                "ACL 0.336–0.476); "
                "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60 scored 0.531 (57768062, tie; public "
                "ACL compressed 0.336–0.476 → 0.339–0.464); "
                "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40 scored 0.530 (57770241, poison; public "
                "ACL widened 0.336–0.476 → 0.333–0.489); "
                "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem scored 0.532 (57797758, accepted; public "
                "Effusion widened 0.478–0.703 → 0.478–0.716); "
                "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60 scored 0.533 (57800238, accepted; public "
                "Effusion widened 0.478–0.716 → 0.478–0.718); "
                "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40 scored 0.532 (57802682, poison; public "
                "Effusion compressed 0.478–0.718 → 0.478–0.713); "
                "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem scored 0.534 (57804837, accepted; public "
                "LM compressed 0.303–0.522 → 0.287–0.490); "
                "weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60 scored 0.534 (57828760, tie; public "
                "LM compressed 0.287–0.490 → 0.284–0.484); weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40 scored 0.534 (57831368, tie; public LM widened 0.287–0.490 → 0.290–0.497). "
                "Replacing learned ranks "
                "with shrinkage priors scored 0.504 — a regression. Per-target constant shrinkage is "
                "AUC-invariant; Gaussian 0.005 noise can scramble weak ranks."
            )
            why_low.append(
                "The 7-d/13-d heads were fit on only 58 gold studies. That set is prevalence-enriched "
                "and cannot resolve ~0.01 macro-AUC gaps (σ≈0.0125, discussion 733876). gold_rank_w50 "
                "(0.50·7-d + 0.50·interact, λI=3.5) is the prior floor at 0.518. gold_rank_w40 tied; "
                "gold_rank_lam2 dropped to 0.515. Parser-only overwrite scored 0.499; gold-fill plus "
                "unmentioned=0.38 scored 0.504; all-target confident pos/neg scored 0.511; named-only "
                "scored 0.502; named-mix on ACL/Baker's/MCL scored 0.514; Baker's-only 50/50 mix "
                "scored 0.519 and is the frozen floor; Baker's 0.40/0.60 mix scored 0.518; Baker's "
                "0.60/0.40 mix tied 0.519; Baker's+ACL 50/50 mix tied 0.519; Baker's+MM 50/50 mix "
                "scored 0.517 (poison); Baker's goldstd mix tied 0.519; Baker's silence→neg scored "
                "0.516 (poison); Baker's dropfat scored 0.519 (tie — fat ≡ fluid). Remaining lift "
                "is minority Gemini Lateral Meniscus (0.60/0.40) on the 0.534 floor "
                "(`weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40`), not a resubmit of frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60 "
                "(0.534 tie), frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem "
                "(0.534 accepted), frac_hyb_lm_eff_acl_gem_eff_gem_w40 "
                "(0.532 poison), frac_hyb_lm_eff_acl_gem_eff_gem_w60 "
                "(0.533 accepted), frac_hyb_lm_eff_acl_gem_eff_gem "
                "(0.532 accepted), frac_hyb_lm_eff_acl_gem_w40 "
                "(0.530 poison), frac_hyb_lm_eff_acl_gem_w60 "
                "(0.531 tie), frac_hyb_lm_eff_acl_gem "
                "(0.531 accepted), frac_hyb_lm_eff_acl_w40 "
                "(0.529 poison), frac_hyb_lm_eff_acl_w60 "
                "(0.530 tie), frac_hyb_lm_eff_w40 "
                "(0.528 poison), frac_hyb_lm_eff_w60 "
                "(0.530 accepted), frac_hyb_lm_w40 "
                "(0.528 poison), frac_hyb_lm_w60 "
                "(0.529 tie), frac_hyb_lm_llm "
                "(0.529 accepted), frac_hyb_mm_llm "
                "(0.527 tie), frac_hyb_gem_llm "
                "(0.527 tie), frac_gem_llm "
                "(0.525 poison), frac_hyb_w40 "
                "(0.526 poison), frac_hyb_w60 "
                "(0.526 poison), frac_hyb_llm "
                "(0.527 accepted), PF OA mix "
                "(0.523 poison), Lateral OA mix "
                "(0.523 poison), Medial OA mix "
                "(0.523 poison), v4 Fracture mix "
                "(0.524 tie), Contusion mix "
                "(0.519 compress), Synovitis remap "
                "(field 0.522, effusion-proxy 0.521), MCL mix (parser 0.514, v4_blend 0.510), "
                "or a resubmit of pfoa_llm / loa_llm / moa_llm / frac_llm / cont_llm / syn_from_eff / syn_llm / eff_llm / acl_llm."
            )
            why_low.append(
                "Local visual training used synthetic DICOMs for gold studies — those weights do not "
                "transfer to real test MRI; do not spend quota on that checkpoint until trained on real data."
            )
        improvements += [
            "Next: `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem` — keep parser Baker's + v4 LLM ACL 50/50 + majority LLM Effusion + hybrid Fracture + LLM Lateral Meniscus + Gemini ACL + majority Gemini Effusion + Gemini LM (0.534) and mix Gemini Medial Meniscus 50/50 only if that head has variance. `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40` scored 0.534 (57831368, tie; public LM widened 0.287–0.490 → 0.290–0.497). `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60` scored 0.534 (57828760, tie; public LM compressed 0.287–0.490 → 0.284–0.484). `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem` scored 0.534 (57804837, accepted; public LM compressed 0.303–0.522 → 0.287–0.490) and is the frozen fallback. `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40` scored 0.532 (57802682, poison; public Effusion compressed 0.478–0.718 → 0.478–0.713). `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60` scored 0.533 (57800238, accepted; public Effusion widened 0.478–0.716 → 0.478–0.718) and is the prior fallback. `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem` scored 0.532 (57797758, accepted; public Effusion widened 0.478–0.703 → 0.478–0.716) and is the prior fallback. `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40` scored 0.530 (57770241, poison; public ACL widened 0.336–0.476 → 0.333–0.489). `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60` scored 0.531 (57768062, tie; public ACL compressed 0.336–0.476 → 0.339–0.464). `weak_rank_bakers_frac_hyb_lm_eff_acl_gem` scored 0.531 (57765842, accepted; public ACL 0.336–0.476) and is the prior fallback. `weak_rank_bakers_frac_hyb_lm_eff_acl_w40` scored 0.529 (57760822, poison; public ACL compressed 0.314–0.539 → 0.326–0.539). `weak_rank_bakers_frac_hyb_lm_eff_acl_w60` scored 0.530 (57726811, tie; public ACL 0.314–0.539). `weak_rank_bakers_frac_hyb_lm_eff_w40` scored 0.528 (57691260, poison; public Effusion compressed 0.478–0.703 → 0.478–0.691). `weak_rank_bakers_frac_hyb_lm_eff_w60` scored 0.530 (57656351, accepted; public Effusion 0.478–0.703) and is the frozen fallback. `weak_rank_bakers_frac_hyb_lm_w40` scored 0.528 (57619038, poison; public LM 0.309–0.522). `weak_rank_bakers_frac_hyb_lm_w60` scored 0.529 (57584615, tie; public LM 0.297–0.522). `weak_rank_bakers_frac_hyb_lm_llm` scored 0.529 (57551203, accepted; public LM 0.303–0.522) and is the frozen fallback. `weak_rank_bakers_frac_hyb_mm_llm` scored 0.527 (57512380, tie; public MM 0.323–0.542). `weak_rank_bakers_frac_hyb_gem_llm` scored 0.527 (57468780, tie; public Fracture 0.217–0.373). `weak_rank_bakers_frac_gem_llm` scored 0.525 (57425148, poison; public Fracture 0.217–0.404). `weak_rank_bakers_frac_hyb_w40` scored 0.526 (57385111, poison; public Fracture 0.248–0.385). `weak_rank_bakers_frac_hyb_w60` scored 0.526 (57341916, poison; public Fracture compressed 0.248–0.373 → 0.248–0.360). `weak_rank_bakers_frac_hyb_llm` scored 0.527 (57298558, accepted; public Fracture 0.248–0.373). `weak_rank_bakers_pfoa_llm` scored 0.523 (57257540, poison; public 3-study PF OA stayed 0.237–0.487). `weak_rank_bakers_loa_llm` scored 0.523 (57218864, poison; public 3-study Lateral OA stayed 0.065–0.315). `weak_rank_bakers_moa_llm` scored 0.523 (57169674, poison; public Medial OA compressed 0.134–0.384 → 0.134–0.321). `weak_rank_bakers_frac_llm` scored 0.524 (57132860, tie; public Fracture 0.217–0.404). `weak_rank_bakers_cont_llm` scored 0.519 (57097591, poison; public Contusion compressed 0.265–0.453 → 0.296–0.359). `weak_rank_bakers_syn_from_eff` scored 0.521 (57064376, poison). `weak_rank_bakers_syn_llm` scored 0.522 (57032978, poison; public Synovitis flattened). `weak_rank_bakers_eff_llm` scored 0.524 (56995081, accepted). `weak_rank_bakers_mcl_llm` scored 0.510 (56938248, poison). `weak_rank_bakers_acl_llm` scored 0.520 (56902005, accepted). `weak_rank_bakers_llm` scored 0.519 (56872526, tie; Baker's LLM constant). `weak_rank_bakers_dropfat` scored 0.519 (56815942, tie); `weak_rank_bakers_silence` scored 0.516 (56790917, poison); `weak_rank_bakers_goldstd` scored 0.519 (56767585, tie); `weak_rank_bakers_mm_mix` scored 0.517 (56745095, falsified); `weak_rank_bakers_acl_mix` scored 0.519 (56698950, tie); `weak_rank_bakers_w40` scored 0.519 (56664719); `weak_rank_bakers_w60` scored 0.518 (56631726); `weak_rank_bakers_mix` scored 0.519 (56600302).",
            "Do not resubmit weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w40, weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60, weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem, weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40, weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60, weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem, weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w40, weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60, weak_rank_bakers_frac_hyb_lm_eff_acl_gem, weak_rank_bakers_frac_hyb_lm_eff_acl_w40, weak_rank_bakers_frac_hyb_lm_eff_acl_w60, weak_rank_bakers_frac_hyb_lm_eff_w40, weak_rank_bakers_frac_hyb_lm_eff_w60, weak_rank_bakers_frac_hyb_lm_w40, weak_rank_bakers_frac_hyb_lm_w60, weak_rank_bakers_frac_hyb_lm_llm, weak_rank_bakers_frac_hyb_mm_llm, weak_rank_bakers_frac_hyb_gem_llm, weak_rank_bakers_frac_gem_llm, weak_rank_bakers_frac_hyb_w40, weak_rank_bakers_frac_hyb_w60, weak_rank_bakers_frac_hyb_llm, weak_rank_bakers_pfoa_llm, weak_rank_bakers_loa_llm, weak_rank_bakers_moa_llm, weak_rank_bakers_frac_llm, weak_rank_bakers_cont_llm, weak_rank_bakers_syn_from_eff, weak_rank_bakers_syn_llm, weak_rank_bakers_eff_llm, weak_rank_bakers_mcl_llm, weak_rank_bakers_acl_llm, weak_rank_bakers_llm, weak_rank_bakers_dropfat, weak_rank_bakers_silence, weak_rank_bakers_goldstd, weak_rank_bakers_mm_mix, weak_rank_bakers_acl_mix, weak_rank_bakers_w40, weak_rank_bakers_w60, weak_rank_bakers_mix, weak_rank_named_mix, weak_rank_named, weak_rank_confident, weak_rank_goldfill, weak_rank_calibrate, gold_rank_lam2, gold_rank_w40, gold_rank_w70, gold_rank_w50, or gold_meta_logit as cycle 0. Late 2026-10-04/05/08 slots wasted quota on gold_meta_logit 0.514, report_shrinkage_priors 0.504, metadata_prior_blend 0.504, and fluid_gate_metadata 0.505.",
            "Forum ceiling: series composition ~0.595 and scanner-grouped DICOM-header ~0.598 on report-derived labels (discussion 733517). Our 0.534 public score is still below that series-flag ceiling because we fit on 58 gold rows instead of 4,407 reports (discussion 733876).",
            "Parse train.csv Report only, with a right-side Turkish negation window (discussion 734106). Never open test reports. Infer from test_series.csv metadata only.",
            "Use real train DICOMs (or official JPEG caches) on Kaggle/GPU for the visual model only after metadata ablations stall. Public visual notebooks already report ~0.926 LB; the 2026-10-24 2nd-place write-up is 0.969 public / 0.960 private and keeps 0.7 report-label weight on ACL.",
            "ASRA-ablate one ranking change per submission; reserve ≥1 daily submission for a validated OOF-improving change only.",
        ]

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md_path = out_dir / f"{day_id}_cycle{cycle_id}_analysis.md"
    lines = [
        f"# Submission analysis — competition day {day_id}, cycle {cycle_id}",
        "",
        f"_Generated {ts} by agent1._",
        "",
        f"Competition: `{competition}`",
        "",
        "## Submission log (newest first)",
        "",
        "| date | status | publicScore | description |",
        "|---|---|---|---|",
    ]
    for r in rows[:20]:
        lines.append(
            f"| {r.get('date','')} | {r.get('status','')} | {r.get('publicScore','')} | "
            f"{(r.get('description') or '')[:80]} |"
        )
    lines += [
        "",
        f"**Best public score seen:** {best}",
        f"**Latest:** {json.dumps(latest, default=str) if latest else None}",
        "",
        "## Why the score is low",
        "",
    ]
    for w in why_low:
        lines.append(f"- {w}")
    lines += ["", "## How to improve", ""]
    for w in improvements:
        lines.append(f"- {w}")
    lines.append("")
    md_path.write_text("\n".join(lines))
    (out_dir / f"{day_id}_cycle{cycle_id}_submissions.json").write_text(json.dumps(rows, indent=2, default=str))
    return md_path
