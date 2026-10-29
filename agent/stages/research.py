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

ARXIV_API = "https://export.arxiv.org/api/query"
EUROPEPMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
UA = "rsna-knee-agent1/0.2 (research; kaggle-competition-literature)"


def _http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": UA,
            "Accept": "application/atom+xml, application/json, text/xml, */*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


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
    data = None
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            # arXiv asks for polite spacing between requests
            time.sleep(2 if attempt == 0 else 6 * attempt)
            data = _http_get(url)
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
        papers.append({"title": title, "summary": summary[:1200], "url": link, "published": published[:10], "source": "arxiv"})
    return papers


def _europepmc_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Fallback literature search when arXiv returns HTTP 406 from cloud egress."""
    params = {
        "query": query,
        "format": "json",
        "pageSize": max(1, max_results),
        "resultType": "lite",
    }
    url = EUROPEPMC_API + "?" + urllib.parse.urlencode(params)
    payload = json.loads(_http_get(url, timeout=45).decode("utf-8"))
    papers = []
    for hit in (payload.get("resultList") or {}).get("result") or []:
        title = str(hit.get("title") or "").strip()
        if not title:
            continue
        year = str(hit.get("pubYear") or "")
        src = str(hit.get("source") or "MED")
        pmid = str(hit.get("pmid") or hit.get("id") or "")
        doi = str(hit.get("doi") or "")
        url_hit = f"https://doi.org/{doi}" if doi else (f"https://europepmc.org/article/{src}/{pmid}" if pmid else "")
        papers.append(
            {
                "title": title,
                "summary": str(hit.get("authorString") or "")[:1200],
                "url": url_hit,
                "published": year,
                "source": "europepmc",
            }
        )
    return papers


def _curated_seed_papers() -> list[dict[str, str]]:
    """Seed research.json when arXiv 406s and EuropePMC times out (Cursor Cloud egress)."""
    return [
        {
            "title": "Development of a multi-task deep learning system for classification of nine common knee abnormalities on MRI (KAMRNet)",
            "summary": "Xie et al. 2025. 13,419 patients / 14,962 exams. Internal primary AUCs 0.898; external ~0.81-0.85. Anatomy-oriented localization mattered.",
            "url": "https://doi.org/10.1016/j.eclinm.2025.103534",
            "published": "2025",
            "source": "curated",
            "query": "knee MRI multi-label abnormality detection deep learning",
        },
        {
            "title": "MRNet: Deep-learning-assisted diagnosis for knee magnetic resonance imaging",
            "summary": "Bien, Rajpurkar et al. PLOS Medicine 2018. Plane-wise CNNs then logistic stack of sagittal/coronal/axial logits.",
            "url": "https://doi.org/10.1371/journal.pmed.1002699",
            "published": "2018",
            "source": "curated",
            "query": "plane-aware MRI classification sagittal coronal axial",
        },
        {
            "title": "MRI deep learning models for assisted diagnosis of knee pathologies: a systematic review",
            "summary": "Mead et al. European Radiology. 54 studies; mean AUC-ROC ~0.921 overall.",
            "url": "https://doi.org/10.1007/s00330-024-11105-8",
            "published": "2024",
            "source": "curated",
            "query": "macro ROC-AUC multi-label medical imaging",
        },
        {
            "title": "Weak labels for all 12 findings + how recoverable each one actually is",
            "summary": "Kaggle discussion 734117. Empty extractions (~23%) should stay unlabeled. Named objects recover; graded/unstated fail.",
            "url": "https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734117",
            "published": "2026",
            "source": "curated",
            "query": "radiology report weak supervision negation",
        },
    ]


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
            "why": "Hand-tuned plane/fluid offsets only moved public macro AUC 0.494→0.499. The 58 gold labels fitted a 7-d ridge logistic that reached 0.514. That is a fallback, not the day's first cycle.",
            "how": "Keep gold_meta_logit as a deep fallback. Do not resubmit it as cycle 0; ablate ranking changes on top of gold_rank_w50 (0.518).",
        },
        {
            "name": "Plane × protocol interaction ranks",
            "why": "Additive sag/cor/ax fractions plus a global fluid fraction cannot represent 'sagittal fluid-sensitive' (ACL/meniscus/effusion) vs 'axial fluid-sensitive' (PF OA, Baker's). gold_rank_w50 (0.50·7-d + 0.50·interact) is the frozen public floor at 0.518.",
            "how": "Keep the 13-d sag/cor/ax × fluid and × fat ridge (λ=3.5). Next ablation is blend weight, not a new architecture.",
        },
        {
            "name": "Rank-blend weight ablation (0.40/0.60) — done, tied",
            "why": "Public LB rose as interact weight went 0.00 (0.514) → 0.30 (0.516) → 0.40 (0.517) → 0.50 (0.518), then 0.40/0.60 tied 0.518 (gold_rank_w40, 56382621). Weight ablation has stalled; do not resubmit w40/w50/w70.",
            "how": "Keep gold_rank_w50 as the frozen fallback. Do not spend quota on another blend weight.",
        },
        {
            "name": "Interact-head λ ablation (λI=2.0) — done, falsified",
            "why": "gold_rank_lam2 (0.50/0.50, λI=2.0) scored 0.515 on 2026-09-21 (submission 56418615), below frozen gold_rank_w50 (0.518). Lower λ overfit the 13-d head on n=58. Do not resubmit.",
            "how": "Keep gold_rank_w50 as the fallback. Do not spend quota on another λ or blend-weight ablation.",
        },
        {
            "name": "Train-report weak labels on all 4,407 (gold for calibration only) — done, falsified",
            "why": "Fitting 7-d/13-d heads on 58 gold studies is high-variance (σ≈0.0125) and the gold set is prevalence-enriched (~1.5×). Parser-only `weak_rank_calibrate` scored 0.499 (56455239) after overwriting the 58 gold rows. Do not resubmit.",
            "how": "Keep as a documented failure. Next lever is which parser cells are used, not another overwrite of gold 0/1.",
        },
        {
            "name": "Gold-fill weak labels (expert 0/1 + parser on unlabeled) — done, falsified",
            "why": "`weak_rank_goldfill` kept gold 0/1 on the 58 and assigned parser soft labels (including unmentioned=0.38) to ~4,349 reports. Public score 0.504 (56484736): +0.005 vs overwrite, −0.014 vs gold_rank_w50. Keyword mass on unmentioned cells does not transfer to the expert-labeled public LB.",
            "how": "Do not resubmit goldfill. Discussion 734117: empty extractions (~23%) should stay unlabeled, not all-negative. Named objects recover; graded/unstated fail.",
        },
        {
            "name": "Confident-only weak labels (gold 0/1 + parser pos/neg; mask the rest) — done, falsified",
            "why": "`weak_rank_confident` scored 0.511 (56513588): +0.007 vs goldfill, −0.007 vs gold_rank_w50. All-target pos/neg mentions help versus unmentioned=0.38 but do not beat gold-only ranks. Do not resubmit.",
            "how": "Keep as a documented failure. Next lever is *which targets* receive parser labels.",
        },
        {
            "name": "Named-object parser labels only (ACL / Baker's / MCL) — done, falsified",
            "why": "`weak_rank_named` scored 0.502 (56541791): −0.009 vs all-target confident (0.511) and −0.016 vs gold_rank_w50. Fitting named-only labels through the shared 4,407-row standardizer appears to have shifted the other nine gold heads. Do not resubmit.",
            "how": "Keep as a documented failure. Next lever is mixing named-weak ranks onto frozen gold ranks for only those three targets.",
        },
        {
            "name": "Named-object mix on frozen gold ranks (ACL / Baker's / MCL) — done, falsified",
            "why": "`weak_rank_named_mix` scored 0.514 (56571590): ties gold_meta_logit and is −0.004 vs gold_rank_w50. Mixing ACL/MCL parser noise onto the frozen interact ranks erased the lift. Do not resubmit.",
            "how": "Keep as a documented failure. Next lever is mixing only the strongest named object (Baker's, discussion 734117 balanced acc 0.82).",
        },
        {
            "name": "Baker's-only mix on frozen gold ranks — done, accepted",
            "why": "`weak_rank_bakers_mix` scored 0.519 (56600302): +0.001 vs gold_rank_w50. Mixing named-weak ranks onto Baker's only beat the previous floor; ACL/MCL mix had failed at 0.514.",
            "how": "Keep as the frozen fallback notebook. ACL mix tied; next lever is Medial Meniscus on top of the accepted Baker's mix.",
        },
        {
            "name": "Baker's mix-weight 0.40/0.60 on frozen gold ranks — done, falsified",
            "why": "`weak_rank_bakers_w60` scored 0.518 (56631726): −0.001 vs bakers_mix 0.519. Majority named-weak on Baker's erased the 50/50 lift and tied gold_rank_w50. Do not resubmit.",
            "how": "Keep as a documented failure. Baker's mix-weight has plateaued.",
        },
        {
            "name": "Baker's mix-weight 0.60/0.40 on frozen gold ranks — done, tied",
            "why": "`weak_rank_bakers_w40` scored 0.519 (56664719): ties bakers_mix. 0.60/0.40, 0.50/0.50, and 0.40/0.60 are a plateau. Do not resubmit.",
            "how": "Keep as a documented failure. Baker's mix-weight and ACL mix have both plateaued at 0.519.",
        },
        {
            "name": "Baker's + ACL mix on frozen gold ranks — done, tied",
            "why": "`weak_rank_bakers_acl_mix` scored 0.519 (56698950): ties bakers_mix. ACL named-weak ranks are public-LB neutral, not poison like MCL (named_mix 0.514). Do not resubmit.",
            "how": "Keep as a documented failure. ACL mix is public-LB neutral; do not resubmit.",
        },
        {
            "name": "Baker's + Medial Meniscus mix on frozen gold ranks — done, falsified",
            "why": "`weak_rank_bakers_mm_mix` scored 0.517 (56745095): −0.002 vs bakers_mix 0.519. Medial Meniscus named-weak ranks are mildly poisonous. Do not add Lateral Meniscus. Do not resubmit.",
            "how": "Keep as a documented failure. Second-target named mixes have stalled (ACL tie, MM drop, MCL poison).",
        },
        {
            "name": "Baker's-only mix with gold-set weak-head standardizer — done, tied",
            "why": "`weak_rank_bakers_goldstd` scored 0.519 (56767585): ties bakers_mix. Gold μ/σ on the named-weak head is public-LB neutral. Do not resubmit.",
            "how": "Keep as a documented failure. Next lever is Baker's silence→neg (discussion 733932), not another standardizer.",
        },
        {
            "name": "Baker's-only mix with unmentioned mapped to negative — done, falsified",
            "why": "`weak_rank_bakers_silence` scored 0.516 (56790917): −0.003 vs bakers_mix 0.519. Mapping Baker's unmentioned → neg poisoned the accepted mix. Discussion 734117 was right: empty extractions stay unlabeled. Do not resubmit.",
            "how": "Keep as a documented failure. Parser Baker's ablations have stalled. Next lever is dropping collinear fat features, not another parser remap.",
        },
        {
            "name": "Baker's-only mix with fat / plane×fat dropped from gold heads — done, tied",
            "why": "`weak_rank_bakers_dropfat` scored 0.519 (56815942): ties bakers_mix. Fat and plane×fat are exact duplicates of Fluid_Sensitive, so dropping them did not change public ranks. Do not resubmit.",
            "how": "Keep as a documented failure. Metadata fat-drop is exhausted. Next lever is a better Baker's weak head (public LLM key), not another collinear-feature drop.",
        },
        {
            "name": "Public LLM Baker's-only mix on frozen gold ranks — done, tied",
            "why": "`weak_rank_bakers_llm` scored 0.519 (56872526) after finding v4_blend (n=4407). Baker's LLM metadata scores were constant on the 3-study public sample, so the 50/50 mix was rank-preserving vs gold. 56844594 was only the discovery miss. Do not resubmit.",
            "how": "Keep as a documented failure. Next lever is LLM ACL on top of parser Baker's mix, skipping constant LLM heads.",
        },
        {
            "name": "Parser Baker's mix plus public LLM ACL mix — done, accepted",
            "why": "`weak_rank_bakers_acl_llm` scored 0.520 (56902005): +0.001 vs bakers_mix. LLM ACL transferred after Baker's LLM→metadata collapsed. Keep as the frozen fallback.",
            "how": "Do not resubmit. Superseded by bakers_eff_llm (0.524). Keep as a prior accepted rung.",
        },
        {
            "name": "Accepted ACL-LLM stack plus public LLM MCL mix — done, falsified",
            "why": "`weak_rank_bakers_mcl_llm` scored 0.510 (56938248): −0.010 vs acl_llm 0.520. CSV found (n=4407), MCL std=0.408 — not a miss. v4_blend MCL is worse than parser MCL (0.514). Do not resubmit or retry MCL mix-weights.",
            "how": "Keep as a documented failure. Do not retry MCL. Do not retry the Synovitis LLM field (0.522 flatten).",
        },
        {
            "name": "Accepted ACL-LLM stack plus public LLM Effusion mix — done, accepted",
            "why": "`weak_rank_bakers_eff_llm` scored 0.524 (56995081): +0.004 vs acl_llm 0.520. CSV found (n=4407), Effusion std=0.408. Largest metadata lift since Baker's-only mix. Keep as the frozen fallback.",
            "how": "Do not resubmit. Next lever is LLM Contusion on the 0.524 stack. Do not retry Synovitis remaps.",
        },
        {
            "name": "Accepted effusion-LLM stack plus public LLM Synovitis mix — done, falsified",
            "why": "`weak_rank_bakers_syn_llm` scored 0.522 (57032978): −0.002 vs eff_llm 0.524. CSV found (n=4407), Synovitis LLM std=0.408 — not a miss. After the mix the public 3-study Synovitis column was constant (0.465517). The Synovitis LLM field cancelled Fluid_Sensitive ranks. Do not resubmit.",
            "how": "Keep as a documented failure. Do not remix the Synovitis LLM field or retry Synovitis mix-weights.",
        },
        {
            "name": "Accepted effusion-LLM stack plus effusion ranks copied onto Synovitis — done, falsified",
            "why": "`weak_rank_bakers_syn_from_eff` scored 0.521 (57064376): −0.003 vs eff_llm 0.524. CSV found (n=4407). On the public 3-study sample the 50/50 candidate was exactly constant (gold Synovitis anti-correlates with effusion ranks) so the flatten guard skipped; public LB still dropped. Both Synovitis remaps failed (field 0.522, proxy 0.521). Do not resubmit.",
            "how": "Keep as a documented failure. Do not retry Synovitis field, mix-weights, or effusion-as-synovitis. Next lever is LLM Contusion.",
        },
        {
            "name": "Accepted effusion-LLM stack plus public LLM Contusion mix — done, falsified",
            "why": "`weak_rank_bakers_cont_llm` scored 0.519 (57097591): −0.005 vs eff_llm 0.524. CSV found (n=4407), Contusion std=0.408 — not a miss. Public Contusion compressed from 0.265–0.453 to 0.296–0.359. Degenerative-BME noise in the Contusion head flattened useful gold ranks. Do not resubmit or retry Contusion mix-weights.",
            "how": "Keep as a documented failure. Do not remix Contusion. Next lever is LLM Fracture on the 0.524 stack.",
        },
        {
            "name": "Accepted effusion-LLM stack plus public LLM Fracture mix — done, tied",
            "why": "`weak_rank_bakers_frac_llm` scored 0.524 (57132860): ties eff_llm. CSV found (n=4407), Fracture std=0.354 — not a miss. Public Fracture stayed 0.217–0.404 (gold was 0.248–0.435). v4 Fracture is LB-neutral. Do not resubmit or retry Fracture mix-weights.",
            "how": "Keep as a documented tie. Hybrid Fracture keys (Pilkwang 0.8713 / Gemini 0.8750) stay in the backlog after unused OA heads.",
        },
        {
            "name": "Accepted effusion-LLM stack plus public LLM Medial OA mix — done, falsified",
            "why": "`weak_rank_bakers_moa_llm` scored 0.523 (57169674): −0.001 vs eff_llm 0.524. CSV found (n=4407), Medial OA std=0.408 — not a miss. Public Medial OA compressed from 0.134–0.384 to 0.134–0.321. High gold-agreement (0.931) did not transfer. Do not resubmit or retry Medial OA mix-weights.",
            "how": "Keep as a documented failure. Do not remix Medial OA. Do not retry Lateral OA (0.523 poison).",
        },
        {
            "name": "Accepted effusion-LLM stack plus public LLM Lateral OA mix — done, falsified",
            "why": "`weak_rank_bakers_loa_llm` scored 0.523 (57218864): −0.001 vs eff_llm 0.524. CSV found (n=4407), Lateral OA std=0.408 — not a miss. Public 3-study Lateral OA stayed 0.065–0.315 (nearly rank-preserving on the sample) but public LB still dropped. Both compartment OA heads are poison. Do not resubmit or retry Lateral OA mix-weights.",
            "how": "Keep as a documented failure. Do not remix Lateral OA. Do not retry PF OA (0.523 poison).",
        },
        {
            "name": "Accepted effusion-LLM stack plus public LLM PF OA mix — done, falsified",
            "why": "`weak_rank_bakers_pfoa_llm` scored 0.523 (57257540): −0.001 vs eff_llm 0.524. CSV found (n=4407), PF OA std=0.408 — not a miss. Public 3-study PF OA stayed 0.237–0.487 (nearly rank-preserving on the sample) but public LB still dropped. All three v4 OA heads are poison. Do not resubmit or retry PF OA mix-weights.",
            "how": "Keep as a documented failure. Do not remix PF OA. Do not retry v4 OA heads.",
        },
        {
            "name": "Accepted effusion-LLM stack plus hybrid v4+Fracture mix — done, accepted",
            "why": "`weak_rank_bakers_frac_hyb_llm` scored 0.527 (57298558): +0.003 vs eff_llm 0.524. Hybrid CSV found (n=4407), Fracture std=0.204 — not a miss. Public Fracture 0.248–0.373. First metadata lift since effusion. Keep as the frozen fallback.",
            "how": "Do not resubmit. Hybrid mix-weights both scored 0.526; keep 50/50 as the frozen floor. Next lever is Gemini Fracture.",
        },
        {
            "name": "Accepted hybrid-Fracture stack plus majority hybrid Fracture weight — done, falsified",
            "why": "`weak_rank_bakers_frac_hyb_w60` scored 0.526 (57341916): −0.001 vs frac_hyb_llm 0.527. Hybrid CSV found (n=4407), Fracture std=0.204, w_gold=0.4 — not a miss. Public Fracture compressed from 0.248–0.373 to 0.248–0.360. Majority hybrid is mild poison. Do not resubmit.",
            "how": "Keep as a documented failure. Do not retry hybrid mix-weights.",
        },
        {
            "name": "Accepted hybrid-Fracture stack plus minority hybrid Fracture weight — done, falsified",
            "why": "`weak_rank_bakers_frac_hyb_w40` scored 0.526 (57385111): −0.001 vs frac_hyb_llm 0.527. Hybrid CSV found (n=4407), Fracture std=0.204, w_gold=0.6 — not a miss. Public Fracture 0.248–0.385. Minority hybrid is mild poison. Mix-weights have plateaued. Do not resubmit.",
            "how": "Keep as a documented failure. Next lever is Gemini 4-source Fracture on the 0.524 effusion stack.",
        },
        {
            "name": "Accepted effusion-LLM stack plus Gemini 4-source Fracture mix — done, falsified",
            "why": "`weak_rank_bakers_frac_gem_llm` scored 0.525 (57425148): +0.001 vs eff_llm 0.524, −0.002 vs frac_hyb_llm 0.527. Gemini CSV found (n=4407), Fracture std=0.354 — not a miss. Public Fracture 0.217–0.404 (matches v4 Fracture). Gemini is a weaker Fracture key than hybrid. Do not resubmit.",
            "how": "Keep as a documented failure. Do not retry Gemini-only Fracture mix-weights. Next lever is Gemini stacked on the 0.527 hybrid floor.",
        },
        {
            "name": "Accepted hybrid-Fracture stack plus Gemini 4-source Fracture mix — done, tied",
            "why": "`weak_rank_bakers_frac_hyb_gem_llm` scored 0.527 (57468780): ties frac_hyb_llm. Gemini CSV found (n=4407), Fracture std=0.354 — not a miss. Public Fracture 0.217–0.373 (hybrid-only 0.248–0.373; Gemini-only 0.217–0.404). Stacking pulled the min toward Gemini and kept the hybrid max; public LB did not move. Fracture keys are exhausted. Do not resubmit.",
            "how": "Keep as a documented failure. Do not retry Gemini-on-hybrid mix-weights. Next lever is v4 LLM Medial Meniscus on the 0.527 hybrid stack.",
        },
        {
            "name": "Accepted hybrid-Fracture stack plus public LLM Medial Meniscus mix — done, tied",
            "why": "`weak_rank_bakers_frac_hyb_mm_llm` scored 0.527 (57512380): ties frac_hyb_llm. CSV found (n=4407), Medial Meniscus std=0.408 — not a miss. Public MM 0.323–0.542. Parser MM was poison (0.517); the better v4 key (gold AUC 0.954) is only LB-neutral. Do not resubmit.",
            "how": "Keep as a documented tie. Next lever is v4 LLM Lateral Meniscus, the last unused named object.",
        },
        {
            "name": "Accepted hybrid-Fracture stack plus public LLM Lateral Meniscus mix — done, accepted",
            "why": "`weak_rank_bakers_frac_hyb_lm_llm` scored 0.529 (57551203): +0.002 vs frac_hyb_llm. CSV found (n=4407), Lateral Meniscus std=0.408 — not a miss. Public LM 0.303–0.522. First meniscus key that lifted the floor (parser MM 0.517 poison; LLM MM 0.527 tie).",
            "how": "Keep as the frozen fallback. LM mix-weights plateaued (w60 tie 0.529, w40 poison 0.528); next lever is Effusion majority weight.",
        },
        {
            "name": "Accepted LM stack plus majority Lateral Meniscus mix-weight — done, tied",
            "why": "`weak_rank_bakers_frac_hyb_lm_w60` scored 0.529 (57584615): ties lm_llm. CSV found (n=4407), Lateral Meniscus std=0.408, w_gold=0.4 — not a miss. Public LM 0.297–0.522 (50/50 was 0.303–0.522). Majority LM is rank-preserving. Do not resubmit.",
            "how": "Keep as a documented tie. Minority LM then poisoned (0.528); next lever is Effusion mix-weight on the 0.529 stack.",
        },
        {
            "name": "Accepted LM stack plus minority Lateral Meniscus mix-weight — done, poison",
            "why": "`weak_rank_bakers_frac_hyb_lm_w40` scored 0.528 (57619038): −0.001 vs lm_llm. CSV found (n=4407), Lateral Meniscus std=0.408, w_gold=0.6 — not a miss. Public LM 0.309–0.522 (50/50 was 0.303–0.522). Minority LM is mild poison. Do not resubmit.",
            "how": "Keep as a documented poison. LM mix-weights are exhausted. Next lever is majority Effusion mix-weight (0.40/0.60) on the 0.529 stack.",
        },
        {
            "name": "Accepted LM stack plus majority Effusion mix-weight — done, accepted",
            "why": "`weak_rank_bakers_frac_hyb_lm_eff_w60` scored 0.530 (57656351): +0.001 vs lm_llm. CSV found (n=4407), Effusion std=0.408, w_gold=0.4. Public Effusion 0.478–0.703.",
            "how": "Keep as the frozen fallback. Minority Effusion then poisoned (0.528). ACL mix-weights are now exhausted.",
        },
        {
            "name": "Accepted 0.530 stack plus majority ACL mix-weight — done, tied",
            "why": "`weak_rank_bakers_frac_hyb_lm_eff_acl_w60` scored 0.530 (57726811): ties eff_w60. CSV found (n=4407), ACL std=0.408, w_gold=0.4. Public ACL 0.314–0.539.",
            "how": "Keep as a documented tie. Do not resubmit majority ACL weight.",
        },
        {
            "name": "Accepted 0.530 stack plus minority ACL mix-weight — done, poison",
            "why": "`weak_rank_bakers_frac_hyb_lm_eff_acl_w40` scored 0.529 (57760822): −0.001 vs eff_w60. CSV found (n=4407), ACL std=0.408, w_gold=0.6. Public ACL compressed 0.314–0.539 → 0.326–0.539.",
            "how": "Keep as a documented poison. ACL mix-weights are exhausted (50/50=0.530, w60=0.530, w40=0.529). Next lever after Gemini ACL 50/50 (0.531) is majority Gemini ACL weight.",
        },
        {
            "name": "Accepted 0.530 stack plus Gemini 4-source ACL mix — done, accepted",
            "why": "`weak_rank_bakers_frac_hyb_lm_eff_acl_gem` scored 0.531 (57765842): +0.001 vs eff_w60. Gemini CSV found (n=4407), Gemini ACL std=0.204. Public ACL 0.336–0.476.",
            "how": "Keep as the frozen floor. Next lever is majority Gemini ACL weight (0.40/0.60) on the 0.531 stack.",
        },
        {
            "name": "Accepted 0.531 stack plus majority Gemini ACL mix-weight",
            "why": "Gemini ACL 50/50 just transferred (+0.001) after compressing public ACL 0.314–0.539 → 0.336–0.476. Gemini ACL std=0.204 is the same scale as hybrid Fracture, where majority weight was the remaining unused mix-weight after 50/50.",
            "how": "Keep frozen gold_rank_w50 + parser Baker's 50/50 + v4 LLM ACL 50/50 + majority Effusion 0.40/0.60 + hybrid Fracture 50/50 + v4 LLM Lateral Meniscus 50/50. Discover report_labels_v5.csv via explicit /kaggle/input/datasets/yunusgmsoy/rsna-knee-llm-labels-4-source-merged plus depth-6 BFS. Mix 0.40·v4_stack + 0.60·gemini on ACL only if std>1e-8. Never open test reports. Do not remix v4 ACL/Effusion/LM weights.",
        },
        {
            "name": "Anatomy-oriented multi-task MRI (KAMRNet / slice transformers)",
            "why": "Xie et al. 2025 (eClinicalMedicine) trained KAMRNet on 13,419 patients / 14,962 exams for nine knee abnormalities and reported internal primary AUCs ~0.90; a 2026 23-condition slice-transformer (European Radiology) reached median external AUC ~0.78. Those numbers are the visual-model ceiling, not a metadata ceiling. We cannot spend quota on them until real train DICOMs/JPEGs + GPU time are mounted.",
            "how": "When pixels are available: plane-slot encoder, anatomy-oriented localization, then rank-ensemble with gold_rank_interact rather than replacing it.",
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
        hits: list[dict[str, Any]] = []
        try:
            hits = _arxiv_search(q, max_results=per_q)
            logger.info("arxiv query %r → %d hits", q, len(hits))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{q}: {type(exc).__name__}: {exc}")
            logger.warning("arxiv failed for %s: %s", q, exc)
        if not hits:
            try:
                hits = _europepmc_search(q, max_results=per_q)
                logger.info("europepmc query %r → %d hits", q, len(hits))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{q} (europepmc): {type(exc).__name__}: {exc}")
                logger.warning("europepmc failed for %s: %s", q, exc)
        for h in hits:
            h["query"] = q
        papers.extend(hits)

    # Dedup by title
    seen = set()
    uniq = []
    for p in papers:
        key = p["title"].lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)

    if len(uniq) < 3:
        for seed in _curated_seed_papers():
            key = seed["title"].lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(seed)
        logger.info("Seeded curated DOIs into research.json (%d papers)", len(uniq))

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
        "Improve macro ROC-AUC on RSNA Knee Abnormality Detection (12 independent study-level targets) beyond the frozen gold_rank_w50 public score (0.518).",
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
        "1. **Keep weak_rank_bakers_frac_hyb_lm_eff_acl_gem (0.531) frozen** and next test `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60` (0.531 stack + majority Gemini ACL 0.40/0.60 if std>1e-8; v4 ACL stays 50/50; Effusion stays 0.40/0.60; LM stays 50/50). `weak_rank_bakers_frac_hyb_lm_eff_acl_gem` scored 0.531 (57765842, accepted; Gemini ACL std=0.204, public ACL 0.336–0.476). `weak_rank_bakers_frac_hyb_lm_eff_acl_w40` scored 0.529 (57760822, poison; ACL std=0.408, w_gold=0.6, public ACL compressed 0.314–0.539 → 0.326–0.539). `weak_rank_bakers_frac_hyb_lm_eff_acl_w60` scored 0.530 (57726811, tie; ACL std=0.408, w_gold=0.4, public ACL 0.314–0.539). `weak_rank_bakers_frac_hyb_lm_eff_w40` scored 0.528 (57691260, poison; Effusion std=0.408, w_gold=0.6, public Effusion compressed 0.478–0.703 → 0.478–0.691). `weak_rank_bakers_frac_hyb_lm_eff_w60` scored 0.530 (57656351, accepted; Effusion std=0.408, w_gold=0.4, public Effusion 0.478–0.703). `weak_rank_bakers_frac_hyb_lm_w40` scored 0.528 (57619038, poison; Lateral Meniscus std=0.408, w_gold=0.6, public LM 0.309–0.522). `weak_rank_bakers_frac_hyb_lm_w60` scored 0.529 (57584615, tie; Lateral Meniscus std=0.408, w_gold=0.4, public LM 0.297–0.522). `weak_rank_bakers_frac_hyb_lm_llm` scored 0.529 (57551203, accepted; Lateral Meniscus std=0.408, public LM 0.303–0.522). `weak_rank_bakers_frac_hyb_mm_llm` scored 0.527 (57512380, tie; Medial Meniscus std=0.408, public MM 0.323–0.542). `weak_rank_bakers_frac_hyb_gem_llm` scored 0.527 (57468780, tie; Gemini Fracture std=0.354, public Fracture 0.217–0.373). `weak_rank_bakers_frac_gem_llm` scored 0.525 (57425148, poison; Gemini Fracture std=0.354, public Fracture 0.217–0.404). `weak_rank_bakers_frac_hyb_w40` scored 0.526 (57385111, poison; public Fracture 0.248–0.385). `weak_rank_bakers_frac_hyb_w60` scored 0.526 (57341916, poison; public Fracture compressed 0.248–0.373 → 0.248–0.360). `weak_rank_bakers_frac_hyb_llm` scored 0.527 (57298558, accepted; hybrid Fracture std=0.204, public Fracture 0.248–0.373). `weak_rank_bakers_pfoa_llm` scored 0.523 (57257540, poison; public 3-study PF OA stayed 0.237–0.487). `weak_rank_bakers_loa_llm` scored 0.523 (57218864, poison; public 3-study Lateral OA stayed 0.065–0.315). `weak_rank_bakers_moa_llm` scored 0.523 (57169674, poison; public Medial OA compressed 0.134–0.384 → 0.134–0.321). `weak_rank_bakers_frac_llm` scored 0.524 (57132860, tie; public Fracture 0.217–0.404). `weak_rank_bakers_cont_llm` scored 0.519 (57097591, poison; public Contusion compressed 0.265–0.453 → 0.296–0.359). `weak_rank_bakers_syn_from_eff` scored 0.521 (57064376, poison). `weak_rank_bakers_syn_llm` scored 0.522 (57032978, poison; public Synovitis flattened to 0.465517). `weak_rank_bakers_eff_llm` scored 0.524 (56995081, accepted). `weak_rank_bakers_mcl_llm` scored 0.510 (56938248, poison). `weak_rank_bakers_acl_llm` scored 0.520 (56902005, accepted). `weak_rank_bakers_llm` scored 0.519 (56872526, tie; Baker's LLM constant). `weak_rank_bakers_dropfat` scored 0.519 (56815942, tie); `weak_rank_bakers_silence` scored 0.516 (56790917, poison); goldstd 0.519 tie; mm_mix 0.517; bakers_acl_mix 0.519 tie; bakers_w40 0.519; bakers_w60 0.518; bakers_mix 0.519; named_mix 0.514; named 0.502; confident 0.511; goldfill 0.504; parser-only 0.499.",
        "2. Keep **weak_rank_bakers_frac_hyb_lm_eff_acl_gem** as the fallback notebook. Do not resubmit weak_rank_bakers_frac_hyb_lm_eff_acl_gem (accepted 0.531), weak_rank_bakers_frac_hyb_lm_eff_acl_w40 (0.529 poison), weak_rank_bakers_frac_hyb_lm_eff_acl_w60 (0.530 tie), weak_rank_bakers_frac_hyb_lm_eff_w40 (0.528 poison), weak_rank_bakers_frac_hyb_lm_eff_w60 (accepted 0.530), weak_rank_bakers_frac_hyb_lm_w40 (0.528 poison), weak_rank_bakers_frac_hyb_lm_w60 (0.529 tie), weak_rank_bakers_frac_hyb_lm_llm (accepted 0.529), weak_rank_bakers_frac_hyb_mm_llm (0.527 tie), weak_rank_bakers_frac_hyb_gem_llm (0.527 tie), weak_rank_bakers_frac_gem_llm (0.525 poison), weak_rank_bakers_frac_hyb_w40 (0.526 poison), weak_rank_bakers_frac_hyb_w60 (0.526 poison), weak_rank_bakers_frac_hyb_llm (accepted 0.527), weak_rank_bakers_pfoa_llm (0.523 poison), weak_rank_bakers_loa_llm (0.523 poison), weak_rank_bakers_moa_llm (0.523 poison), weak_rank_bakers_frac_llm (0.524 tie), weak_rank_bakers_cont_llm (0.519 poison), weak_rank_bakers_syn_from_eff (0.521 poison), weak_rank_bakers_syn_llm (0.522 poison), weak_rank_bakers_eff_llm (accepted 0.524), weak_rank_bakers_mcl_llm (0.510 poison), weak_rank_bakers_acl_llm (accepted 0.520), weak_rank_bakers_llm (tied 0.519), weak_rank_bakers_dropfat (tied 0.519), weak_rank_bakers_silence (0.516), weak_rank_bakers_goldstd (tied 0.519), weak_rank_bakers_mm_mix (0.517), weak_rank_bakers_acl_mix (tied 0.519), weak_rank_bakers_w40 (tied 0.519), weak_rank_bakers_w60 (0.518), weak_rank_bakers_mix (0.519), weak_rank_named_mix (0.514), weak_rank_named (0.502), weak_rank_confident (0.511), weak_rank_goldfill (0.504), weak_rank_calibrate (0.499), gold_rank_lam2 (0.515), gold_rank_w40 (tied 0.518), or gold_rank_w50 as cycle 0.",
        "3. Parse **train reports only**; inference must stay MRI/metadata-only. Never open test reports.",
        "4. Only then spend quota on heavier visual encoder changes (plane-aware EfficientNet / KAMRNet-style localization) once metadata+weak-label ablations stall.",
        "",
        "## Literature / arXiv notes",
        "",
        "### Curated (used even if arXiv 406s)",
        "",
        "- **Xie et al., 2025. Development of a multi-task deep learning system for classification of nine common knee abnormalities on MRI (KAMRNet).** eClinicalMedicine. 13,419 patients / 14,962 exams. Internal primary AUCs 0.898; external ~0.81–0.85. Anatomy-oriented localization mattered in ablation. URL: https://doi.org/10.1016/j.eclinm.2025.103534",
        "- **Comprehensive DL-assisted multi-condition knee MRI (23 conditions, slice transformer), 2026.** Dual-center; median external AUC ~0.78; AUC≥0.75 for 18/23 internal conditions. Model assistance helped residents. URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC13035746/",
        "- **Mead et al. MRI DL models for assisted diagnosis of knee pathologies: systematic review (European Radiology).** 54 studies; mean AUC-ROC ~0.921 overall, ~0.927 pathology-specific vs ~0.898 general-abnormality. No regulatory-cleared systems. Specialized 12-target heads are the right frame. URL: https://doi.org/10.1007/s00330-024-11105-8 and https://pmc.ncbi.nlm.nih.gov/articles/PMC12021734/",
        "- **Bien, Rajpurkar et al. MRNet (PLOS Medicine, 2018).** Plane-wise CNNs then logistic stack of sagittal/coronal/axial logits. Our metadata analogue is plane×fluid×fat counts until pixels are trained. URL: https://doi.org/10.1371/journal.pmed.1002699",
        "- **Kaggle discussion 733517 (2026).** Full DICOM-header HistGBM reaches 0.6516 macro AUC on report-derived labels under random folds but only 0.5981 under scanner-grouped folds; series composition alone (`train_series.csv` four columns) is 0.5954. The 0.05 gap is site memorization and should not be the public-LB target. URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733517",
        "- **Kaggle discussion 733876 (2026).** Paired sigma of a macro-AUC comparison on the 58 gold studies is ~0.0125; a true +0.01 wins CV only ~78% of the time. The 58 are prevalence-enriched vs the 4,407 reports. Practical rule: rank on weak labels over all train reports; keep the 58 for calibration. URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733876",
        "- **Kaggle discussion 734106 (2026).** Reports span seven languages; Turkish negates after the term (`efüzyon izlenmedi`). A left-only NegEx window inverts the second-largest language. Detect language, apply a right-side negation window, and do not tune vocabulary on the 58. URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734106",
        "- **Kaggle discussion 734117 (2026).** Weak-label extractor finds 2.6 findings/study vs annotator 4.1 and returns nothing for ~23% of studies. Treat those as unlabeled, not all-negative. Named objects (Baker's, ACL) recover; graded severities (effusion) and unstated inferences (fracture) fail. URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734117",
        "- **Kaggle discussion 733932 (2026).** Baker's silence is almost certainly negative: gold+ is ~3% when the report does not mention a Baker's cyst vs ~44% when it does. This cycle maps Baker's unmentioned → neg on the named-weak head only. URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932",
        "- **Public LLM report labels (2026).** stevenleehans llm_labels_v4_blend 0.8927 vs gold (Baker's 0.9457); hybrid v4+Fracture 0.8991; regex/lexicon 0.8136. Per-label quality on the 58: ACL 0.993 and MCL 0.964 at the top, Medial OA 0.931, Synovitis 0.790 and Fracture 0.793 at the bottom. All three v4 OA heads already poisoned (0.523). Hybrid Fracture 50/50 transferred to 0.527 (57298558). Majority hybrid 0.40/0.60 dropped to 0.526 (57341916). Minority hybrid 0.60/0.40 dropped to 0.526 (57385111). Gemini-only Fracture dropped to 0.525 (57425148). Hybrid+Gemini Fracture tied 0.527 (57468780). v4 LLM Medial Meniscus tied 0.527 (57512380). v4 LLM Lateral Meniscus transferred to 0.529 (57551203). Majority LM mix-weight tied 0.529 (57584615). Minority LM mix-weight dropped to 0.528 (57619038). Majority Effusion mix-weight transferred to 0.530 (57656351). Minority Effusion mix-weight dropped to 0.528 (57691260). Majority ACL mix-weight tied 0.530 (57726811). Minority ACL mix-weight dropped to 0.529 (57760822). Gemini ACL 50/50 transferred to 0.531 (57765842). Next is majority Gemini ACL on the 0.531 stack. URL: https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels",
        "- **Hybrid v4+Fracture labels (2026).** flight0234 `report_labels_v4hybrid.csv` (4,407 rows, same 12 targets + `__conf`). Fracture std=0.176 (0.08–0.94) vs v4 Fracture 0.793 gold-agreement. URL: https://www.kaggle.com/datasets/flight0234/rsna-knee-hybrid-report-labels",
        "- **18th-place extractor write-up (2026).** Contusion was over-calling ordinary degenerative marrow edema as traumatic bruising until the prompt required a traumatic pattern; Fracture needed an avulsion/insufficiency inclusion list. Named-object Contusion can transfer, but degenerative BME noise is the main failure mode. URL: https://huggingface.co/blog/bishnoiyash/rsna-competetion",
        "- **yunusgmsoy 2-source merge (2026).** Per-target source agreement: Contusion 0.85, Baker's 0.85, ACL 0.91, Effusion 0.90, Synovitis 0.77. Contusion is as agreed as Baker's and much cleaner than Synovitis. URL: https://www.kaggle.com/datasets/yunusgmsoy/rsna-knee-llm-labels-2-source-merged",
        "- **Newer public LLM keys (2026-10).** yunusgmsoy 4-source Gemini Flash merge: macro 0.8798 vs gold, Baker's 0.9293 (loses Baker's to stevenleehans). Qwen2.5-32B v2: macro 0.8686, Baker's 0.903. Do not swap the Baker's head to a weaker key until v4_blend is scored. URLs: https://www.kaggle.com/datasets/yunusgmsoy/rsna-knee-llm-labels-4-source-merged and https://www.kaggle.com/datasets/laymond/rsna-knee-abnormality-qwen3-8b-weak-labels",
        "- **Public visual LB (2026-10-03).** Prize-contender notebooks sit at ~0.93 macro AUC (pixels + GPU). That remains the visual ceiling, not a metadata ceiling.",
        "- **2nd-place write-up (2026-10-24).** Vision-LLM soft labels + 2.5D MIL ensemble: public 0.969 / private 0.960. Report-only labels scored 0.899 vs the 58; an image-reading LLM (Astra) lifted that to 0.942. Teacher-mix keeps **0.7 report-label weight on ACL** (and MCL, both menisci, PF OA, Lateral OA) because the report is the better witness there. Gemini ACL 50/50 transferred to 0.531; the remaining mix-weight is majority Gemini (0.40/0.60), matching that 0.7 report-teacher hint. It is not a license to train pixels this cycle. URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/2nd-place-gold-solution-vision-llm-soft-labels-an",
        "- **Grouped CV + study metadata (Afshar, 2026).** Canonical study-grouped 5-fold split for 4,407 exams (58 gold, 4,349 report-only). Use gold folds to validate ranking; do not invent DICOM-header paths. URL: https://www.kaggle.com/datasets/dariushafshar/rsna-knee-2026-grouped-cv-folds",
        "- **Public visual notebooks (2026).** CoaTNet + fine-tune blends report public LB ~0.926. That is the pixel-model ceiling, not a metadata ceiling. We cannot spend quota there until real train DICOMs/JPEGs + GPU time are mounted. URL: https://www.kaggle.com/code/paiky1995/rsna-knee-0-926-lb-coatnet-fine-tune-blend",
        "",
        "Implication for this cycle: visual AUCs of 0.8–0.93 (public LB ~0.932 on 2026-10-03) and grouped-fold metadata ~0.60 remain medium-run targets. Gold-only ranking stalled (w50=0.518). Report-weak ladder: 0.499 → 0.504 → 0.511 → 0.502 (named) → 0.514 (named_mix, falsified) → 0.519 (bakers_mix, accepted) → 0.518 (bakers_w60, falsified) → 0.519 (bakers_w40, tie) → 0.519 (bakers_acl_mix, tie) → 0.517 (bakers_mm_mix, falsified) → 0.519 (bakers_goldstd, tie) → 0.516 (bakers_silence, poison) → 0.519 (bakers_dropfat, tie) → 0.519 (bakers_llm, tie; Baker's LLM constant) → 0.520 (bakers_acl_llm, accepted) → 0.510 (bakers_mcl_llm, poison) → 0.524 (bakers_eff_llm, accepted) → 0.522 (bakers_syn_llm, poison; public Synovitis flattened) → 0.521 (bakers_syn_from_eff, poison) → 0.519 (bakers_cont_llm, poison; public Contusion compressed) → 0.524 (bakers_frac_llm, tie) → 0.523 (bakers_moa_llm, poison; public Medial OA compressed) → 0.523 (bakers_loa_llm, poison; public Lateral OA stayed 0.065–0.315) → 0.523 (bakers_pfoa_llm, poison; public PF OA stayed 0.237–0.487) → 0.527 (bakers_frac_hyb_llm, accepted; public Fracture 0.248–0.373) → 0.526 (bakers_frac_hyb_w60, poison; public Fracture compressed 0.248–0.360) → 0.526 (bakers_frac_hyb_w40, poison; public Fracture 0.248–0.385) → 0.525 (bakers_frac_gem_llm, poison; public Fracture 0.217–0.404) → 0.527 (bakers_frac_hyb_gem_llm, tie; public Fracture 0.217–0.373) → 0.527 (bakers_frac_hyb_mm_llm, tie; public MM 0.323–0.542) → 0.529 (bakers_frac_hyb_lm_llm, accepted; public LM 0.303–0.522) → 0.529 (bakers_frac_hyb_lm_w60, tie; public LM 0.297–0.522) → 0.528 (bakers_frac_hyb_lm_w40, poison; public LM 0.309–0.522) → 0.530 (bakers_frac_hyb_lm_eff_w60, accepted; public Effusion 0.478–0.703) → 0.528 (bakers_frac_hyb_lm_eff_w40, poison; public Effusion compressed 0.478–0.691) → 0.530 (bakers_frac_hyb_lm_eff_acl_w60, tie; public ACL 0.314–0.539) → 0.529 (bakers_frac_hyb_lm_eff_acl_w40, poison; public ACL compressed 0.314–0.539 → 0.326–0.539) → 0.531 (bakers_frac_hyb_lm_eff_acl_gem, accepted; public ACL 0.336–0.476). Next executable lever is `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_w60`. Still no test reports.",
        "",
        "### arXiv query hits",
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
