"""Stage 1: web/arxiv research write-up → Research/."""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"


def _arxiv_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    import time

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = ARXIV_API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "rsna-knee-agent1/0.1"})
    data = None
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            # arXiv asks for polite spacing between requests
            time.sleep(3 if attempt == 0 else 8 * attempt)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("arxiv attempt %d failed: %s", attempt + 1, exc)
    if data is None:
        raise RuntimeError(str(last_exc))
    root = ET.fromstring(data)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("a:entry", ns):
        title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip().replace("\n", " ")
        summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip().replace("\n", " ")
        link = ""
        for l in entry.findall("a:link", ns):
            if l.attrib.get("type") == "text/html" or l.attrib.get("rel") == "alternate":
                link = l.attrib.get("href", "")
                break
        published = entry.findtext("a:published", default="", namespaces=ns) or ""
        papers.append({"title": title, "summary": summary[:1200], "url": link, "published": published[:10]})
    return papers


def _curated_techniques() -> list[dict[str, str]]:
    """Competition-specific methods grounded in prior RSNA / multi-label MRI practice."""
    return [
        {
            "name": "Plane-aware multi-series aggregation",
            "why": "Sagittal/coronal/axial emphasize different structures (ACL sagittal, MCL coronal, PF OA axial). Pooling all series together dilutes plane-specific signal and hurts macro AUC equally across 12 targets.",
            "how": "Encode slices → series emb → fixed plane slots + availability mask; never treat missing plane as clinical negative.",
        },
        {
            "name": "Weak supervision from multilingual reports",
            "why": "Only ~58 gold labels vs thousands of reports. Ignoring reports leaves ranking models undertrained; hard pseudo-labels from negation mistakes invert targets.",
            "how": "Negation/uncertainty/temporality-aware parser → soft labels + confidence weights; calibrate on gold folds only; never use test reports.",
        },
        {
            "name": "Small-n gold-set metadata logistic",
            "why": "Hand-tuned plane/fluid offsets only moved public macro AUC 0.494→0.499. The 58 gold labels fitted a 7-d ridge logistic that reached 0.514. That is now the frozen floor, not a research idea.",
            "how": "Keep gold_meta_logit as fallback. Do not resubmit it as the day's first cycle; ablate ranking changes on top of it.",
        },
        {
            "name": "Plane × protocol interaction ranks",
            "why": "Additive sag/cor/ax fractions plus a global fluid fraction cannot represent 'sagittal fluid-sensitive' (ACL/meniscus/effusion) vs 'axial fluid-sensitive' (PF OA, Baker's). Public models (MRNet plane-wise logits; RSNA knee label-attention with plane×fluid×fat counts) treat those combinations as first-class features. A 13-d interaction logit on the same 58 gold rows can re-rank fluid/OA heads.",
            "how": "Extend the 7-d vector with sag/cor/ax × fluid and sag/cor/ax × fat fractions; fit ridge λ=3.5; rank-blend 0.60·7-d + 0.40·interact so a noisy interact head cannot fully overwrite the 0.514 ranking.",
        },
        {
            "name": "Per-target calibrated prevalence prior blend",
            "why": "Macro AUC cares about ranking per target. Pure constant prevalence (~0.49 public) is a floor; blending study-level model logits with target prevalence stabilizes rare labels (MCL, Baker's).",
            "how": "p = clip((1-α)·σ(logit) + α·prev_t); tune α on OOF; larger α for rare/unstable heads.",
        },
        {
            "name": "Fluid-sensitive / fat-suppression series routing",
            "why": "train_series.csv exposes Fluid_Sensitive and Fat_Suppression. Contusion/effusion/synovitis benefit from fluid-sensitive series; bone findings may prefer others.",
            "how": "Weight series embeddings by sequence flags before plane pooling; ablate via ASRA H-routing.",
        },
        {
            "name": "Frozen encoder then last-block unfreeze",
            "why": "Tiny gold set overfits if the full backbone trains early. Frozen ImageNet/RadImageNet encoder + heads first is the reliable baseline.",
            "how": "Train heads only until OOF beats prevalence; then unfreeze last block at low LR.",
        },
        {
            "name": "Target-specific view preference soft regularizer",
            "why": "Hard rules (no sagittal ⇒ ACL=0) destroy recall. Soft preferred-plane coverage features improve ranking without forcing zeros.",
            "how": "Concatenate preferred-plane coverage; small consistency loss only after ablation accepts it.",
        },
        {
            "name": "Ensemble fold checkpoints + rank average",
            "why": "Single fold on 58 labels is noisy. Probability or rank ensembling across folds lifts macro AUC more than architecture churn.",
            "how": "Save fold*.pt; at inference mean probs or average ranks then rescale to [eps,1-eps].",
        },
    ]


def run_research(out_dir: Path, queries: list[str], max_arxiv: int, cycle_id: str, day_id: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    papers: list[dict[str, Any]] = []
    errors: list[str] = []
    per_q = max(1, max_arxiv // max(len(queries), 1))
    for q in queries:
        try:
            hits = _arxiv_search(q, max_results=per_q)
            for h in hits:
                h["query"] = q
            papers.extend(hits)
            logger.info("arxiv query %r → %d hits", q, len(hits))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{q}: {type(exc).__name__}: {exc}")
            logger.warning("arxiv failed for %s: %s", q, exc)

    # Dedup by title
    seen = set()
    uniq = []
    for p in papers:
        key = p["title"].lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)

    techniques = _curated_techniques()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md_path = out_dir / f"{day_id}_cycle{cycle_id}_research.md"
    lines = [
        f"# Research write-up — competition day {day_id}, cycle {cycle_id}",
        "",
        f"_Generated {ts} by agent1._",
        "",
        "## Goal",
        "",
        "Improve macro ROC-AUC on RSNA Knee Abnormality Detection (12 independent study-level targets) beyond the Stage-B prevalence baseline (~0.494 public).",
        "",
        "## Methods to consider",
        "",
    ]
    for i, t in enumerate(techniques, 1):
        lines += [
            f"### {i}. {t['name']}",
            "",
            f"**Why it should help:** {t['why']}",
            "",
            f"**How to apply:** {t['how']}",
            "",
        ]

    lines += [
        "## Priority for next submission (research-driven)",
        "",
        "1. **Keep gold_rank_interact (0.517) frozen** and ablate blend weight / interaction λ — do not replace learned ranks with constants (report-shrinkage cannot change AUC).",
        "2. Keep **gold_rank_interact** as the fallback notebook if the next ablation does not beat 0.517.",
        "3. Add **report weak-supervision** only on train; inference must stay MRI/metadata-only.",
        "4. Only then spend quota on heavier visual encoder changes (plane-aware EfficientNet / label-attention) once metadata ablations stall.",
        "",
        "## Literature / arXiv notes",
        "",
    ]
    if not uniq:
        lines.append("_No arXiv hits retrieved this cycle (network or API issue). Curated methods above still apply._")
        lines.append("")
    for p in uniq:
        lines += [
            f"- **{p['title']}** ({p.get('published', '')})",
            f"  - Query: `{p.get('query', '')}`",
            f"  - {p['summary'][:400]}...",
            f"  - {p.get('url', '')}",
            "",
        ]
    if errors:
        lines += ["## Fetch errors", ""] + [f"- {e}" for e in errors] + [""]

    md_path.write_text("\n".join(lines))
    meta = {"papers": uniq, "techniques": techniques, "errors": errors, "path": str(md_path)}
    (out_dir / f"{day_id}_cycle{cycle_id}_research.json").write_text(json.dumps(meta, indent=2))
    return md_path
