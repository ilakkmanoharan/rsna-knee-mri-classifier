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
                "gold_rank_interact 0.517; gold_rank_w50 0.518 (accepted, submission 56322623). "
                "Replacing learned ranks with shrinkage priors scored 0.504 — a regression. "
                "Per-target constant shrinkage is AUC-invariant; Gaussian 0.005 noise can scramble weak ranks."
            )
            why_low.append(
                "The 7-d additive metadata model cannot represent plane×fluid protocols (sagittal "
                "fluid-sensitive vs axial fluid-sensitive). gold_rank_w50 (0.50·7-d + 0.50·interact) "
                "is the frozen floor. Remaining metadata lift must change ranking further (blend weight "
                "0.70/0.30, λ, or train-report weak labels), not calibration."
            )
            why_low.append(
                "Local visual training used synthetic DICOMs for gold studies — those weights do not "
                "transfer to real test MRI; do not spend quota on that checkpoint until trained on real data."
            )
        improvements += [
            "Ablate the gold_rank_w50 blend: 0.70/0.30 next (more 7-d). Drop scrambling noise.",
            "Do not resubmit gold_rank_w50 or gold_meta_logit as cycle 0; keep gold_rank_w50 (0.518) as fallback.",
            "Use real train DICOMs (or official JPEG caches) on Kaggle/GPU for the visual model only after metadata ablations stall.",
            "Train with report weak labels on the large unlabeled train set; infer without reports.",
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
