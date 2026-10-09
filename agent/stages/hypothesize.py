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
            "id": "H_weak_rank_bakers_syn_llm",
            "hypothesis": "Keeping the accepted parser-Baker's + LLM-ACL + LLM-Effusion stack (0.524) and adding a 50/50 stevenleehans v4_blend Synovitis mix will beat 0.524, because effusion just transferred (+0.004) and synovitis is the other high-prevalence fluid finding that shares Fluid_Sensitive ranking variance.",
            "mechanism": "Fit frozen gold_rank_w50 exactly (7-d λ=2 + 13-d λI=3.5, 0.50/0.50). Apply parser weak_rank_bakers_mix on Baker's only, then LLM ACL 50/50 and LLM Effusion 50/50 if std>1e-8 (the 0.524 path). Discover llm_labels_v4_blend.csv via explicit /kaggle/input/datasets mounts plus depth-6 BFS. Fit 7-d/13-d ridge heads on train-only LLM Synovitis soft labels (never test UIDs / test reports). Mix 0.50·gold + 0.50·llm_weak on Synovitis only if that head has std>1e-8; otherwise keep the 0.524 stack. Fallback to weak_rank_bakers_eff_llm if the CSV is missing or n_llm<20.",
            "falsify": "Public score ≤ 0.524 (frozen weak_rank_bakers_eff_llm).",
            "expected_targets": ["Baker's", "ACL", "Effusion", "Synovitis"],
        },
        {
            "id": "H_weak_rank_bakers_eff_llm",
            "hypothesis": "Keeping the accepted parser-Baker's + LLM-ACL stack (0.520) and adding a 50/50 stevenleehans v4_blend Effusion mix will beat 0.520, because ligament MCL is exhausted (parser 0.514, v4_blend 0.510) while effusion is a high-prevalence fluid finding whose report labels may complement Fluid_Sensitive ranks.",
            "mechanism": "Fit frozen gold_rank_w50 exactly (7-d λ=2 + 13-d λI=3.5, 0.50/0.50). Apply parser weak_rank_bakers_mix on Baker's only, then LLM ACL 50/50 if std>1e-8 (the 0.520 path). Discover llm_labels_v4_blend.csv via explicit /kaggle/input/datasets mounts plus depth-6 BFS. Fit 7-d/13-d ridge heads on train-only LLM Effusion soft labels (never test UIDs / test reports). Mix 0.50·gold + 0.50·llm_weak on Effusion only if that head has std>1e-8; otherwise keep the 0.520 stack. Fallback to weak_rank_bakers_acl_llm if the CSV is missing or n_llm<20.",
            "falsify": "Public score ≤ 0.520 (frozen weak_rank_bakers_acl_llm). Already accepted at 0.524 (56995081).",
            "expected_targets": ["Baker's", "ACL", "Effusion"],
        },
        {
            "id": "H_weak_rank_bakers_mcl_llm",
            "hypothesis": "Keeping the accepted parser-Baker's + LLM-ACL stack (0.520) and adding a 50/50 stevenleehans v4_blend MCL mix will beat 0.520, because parser MCL poisoned the named trio (0.514) while LLM ACL just transferred (+0.001) and MCL is the remaining named object.",
            "mechanism": "Fit frozen gold_rank_w50 exactly (7-d λ=2 + 13-d λI=3.5, 0.50/0.50). Apply parser weak_rank_bakers_mix on Baker's only, then LLM ACL 50/50 if std>1e-8 (the 0.520 path). Discover llm_labels_v4_blend.csv via explicit /kaggle/input/datasets mounts plus depth-6 BFS. Fit 7-d/13-d ridge heads on train-only LLM MCL soft labels (never test UIDs / test reports). Mix 0.50·gold + 0.50·llm_weak on MCL only if that head has std>1e-8; otherwise keep the 0.520 stack. Fallback to weak_rank_bakers_acl_llm if the CSV is missing or n_llm<20.",
            "falsify": "Public score ≤ 0.520 (frozen weak_rank_bakers_acl_llm). Already falsified at 0.510 (56938248); v4_blend MCL is poison.",
            "expected_targets": ["Baker's", "ACL", "MCL"],
        },
        {
            "id": "H_weak_rank_bakers_acl_llm",
            "hypothesis": "Keeping the accepted parser Baker's-only 50/50 mix and adding a 50/50 stevenleehans v4_blend ACL mix will beat 0.519, because the Baker's LLM metadata head was constant on the public 3-study sample (56872526, 0.519 tie) while ACL already has sagittal/fluid ranking variance.",
            "mechanism": "Fit frozen gold_rank_w50 exactly (7-d λ=2 + 13-d λI=3.5, 0.50/0.50). Apply parser weak_rank_bakers_mix on Baker's only. Discover llm_labels_v4_blend.csv via explicit /kaggle/input/datasets mounts plus depth-6 BFS. Fit 7-d/13-d ridge heads on train-only LLM ACL soft labels (never test UIDs / test reports). Mix 0.50·gold + 0.50·llm_weak on ACL only if that head has std>1e-8; otherwise keep parser Baker's mix. Fallback to weak_rank_bakers_mix if the CSV is missing or n_llm<20.",
            "falsify": "Public score ≤ 0.519 (frozen weak_rank_bakers_mix). Already accepted at 0.520 (56902005).",
            "expected_targets": ["Baker's", "ACL"],
        },
        {
            "id": "H_weak_rank_bakers_llm",
            "hypothesis": "Keeping the accepted Baker's-only 50/50 mix but replacing the keyword-parser weak head with stevenleehans llm_labels_v4_blend Baker's labels will beat 0.519, because the public LLM key scores 0.8927 vs gold while our regex parser is 0.8136 and parser Baker's remaps have stalled.",
            "mechanism": "Fit frozen gold_rank_w50 exactly (7-d λ=2 + 13-d λI=3.5, 0.50/0.50, standardize on the 58 gold studies). Discover llm_labels_v4_blend.csv with a depth-6 BFS under /kaggle/input and /kaggle/input/datasets (skip DICOM/image dirs; no rglob). Fit 7-d/13-d ridge heads on train-only LLM Baker's soft labels (never test UIDs / test reports). Mix 0.50·gold + 0.50·llm_weak on Baker's only. Leave the other 11 targets as gold_rank_w50. Fallback to parser weak_rank_bakers_mix if the CSV is missing or n_llm<20.",
            "falsify": "Public score ≤ 0.519 (frozen weak_rank_bakers_mix). Already falsified at 0.519 tie (56872526); Baker's LLM head was constant on the public sample.",
            "expected_targets": ["Baker's"],
        },
        {
            "id": "H_weak_rank_bakers_dropfat",
            "hypothesis": "Keeping the accepted Baker's-only 50/50 mix but dropping fat-suppression and plane×fat features will beat 0.519, because Fluid_Sensitive ≡ Fat_Suppression on every train series so those columns are duplicates that inflate the 7-d/13-d gold heads on n=58.",
            "mechanism": "Fit gold ranks with a 6-d additive head (intercept, log1p(n), sag/cor/ax, fluid; no fat) and a 9-d interact head (those plus sag/cor/ax × fluid; no plane×fat), λ=2 / λI=3.5, 0.50/0.50, standardize on the 58 gold studies. Keep the accepted Baker's-only 50/50 named-weak mix (unmentioned masked). Leave the other 11 targets as the no-fat gold ranks. Never open test reports. Fallback to weak_rank_bakers_mix if n_gold<20.",
            "falsify": "Public score ≤ 0.519 (frozen weak_rank_bakers_mix). Already falsified at 0.519 tie (56815942).",
            "expected_targets": ["Baker's", "Effusion", "Synovitis", "ACL"],
        },
        {
            "id": "H_weak_rank_bakers_silence",
            "hypothesis": "Keeping the accepted Baker's-only 50/50 mix but mapping unmentioned Baker's cells to negative (not masked) will beat 0.519, because discussion 733932 says Baker's silence is almost certainly negative (3% gold+ vs 44% when mentioned) while goldstd's standardizer change tied 0.519.",
            "mechanism": "Fit gold_rank_w50 exactly (7-d λ=2 + 13-d λI=3.5, 0.50/0.50, standardize on the 58 gold studies). Separately fit named-only weak heads (gold 0/1 + parser pos/neg on ACL / Baker's / MCL) and map Baker's unmentioned → neg=0.12. Keep ACL/MCL unmentioned masked. Mix 0.50·gold + 0.50·named_weak on Baker's only. Leave the other 11 targets as gold_rank_w50. Never open test reports. Fallback to weak_rank_bakers_mix if n_weak<20.",
            "falsify": "Public score ≤ 0.519 (frozen weak_rank_bakers_mix). Already falsified at 0.516 (56790917).",
            "expected_targets": ["Baker's"],
        },
        {
            "id": "H_weak_rank_bakers_goldstd",
            "hypothesis": "Keeping the accepted Baker's-only 50/50 mix but standardizing the named-weak head on the 58 gold feature moments will beat 0.519, because named-only (0.502) and Baker's+MM (0.517) both used the 4,407-row standardizer that shifts ranks away from gold_rank_w50 space.",
            "mechanism": "Fit gold_rank_w50 exactly (7-d λ=2 + 13-d λI=3.5, 0.50/0.50, standardize on the 58 gold studies). Separately fit named-only weak heads (gold 0/1 + parser pos/neg on ACL / Baker's / MCL) but compute μ/σ from gold-row series features, not the 4,407 weak rows. Mix 0.50·gold + 0.50·named_weak on Baker's only. Leave the other 11 targets as gold_rank_w50. Never open test reports. Fallback to weak_rank_bakers_mix if n_weak<20.",
            "falsify": "Public score ≤ 0.519 (frozen weak_rank_bakers_mix). Already falsified at 0.519 tie (56767585).",
            "expected_targets": ["Baker's"],
        },
        {
            "id": "H_weak_rank_bakers_mm_mix",
            "hypothesis": "Keeping the accepted 50/50 Baker's mix and also 50/50 mixing Medial Meniscus named-weak ranks will beat 0.519, because ACL mix was public-LB neutral (0.519 tie) while MCL poisoned the trio (0.514) — meniscus is the next named object that can recover.",
            "mechanism": "Fit gold_rank_w50 exactly (7-d λ=2 + 13-d λI=3.5, 0.50/0.50, standardize on the 58 gold studies). Separately fit named-only weak heads with Medial Meniscus added to the parser set. Mix 0.50·gold + 0.50·named_weak on Baker's and Medial Meniscus only. Leave ACL, MCL, and the other 9 targets as gold_rank_w50. Never open test reports. Fallback to weak_rank_bakers_mix if n_weak<20.",
            "falsify": "Public score ≤ 0.519 (frozen weak_rank_bakers_mix). Already falsified at 0.517 (56745095).",
            "expected_targets": ["Baker's", "Medial Meniscus"],
        },
        {
            "id": "H_weak_rank_bakers_acl_mix",
            "hypothesis": "Keeping the accepted 50/50 Baker's mix and also 50/50 mixing ACL named-weak ranks will beat 0.519, because Baker's-only mix lifted the floor and the ACL+Baker's+MCL trio failed at 0.514 — MCL is the likely poison.",
            "mechanism": "Fit gold_rank_w50 exactly (7-d λ=2 + 13-d λI=3.5, 0.50/0.50, standardize on the 58 gold studies). Separately fit named-only weak heads. Mix 0.50·gold + 0.50·named_weak on Baker's and ACL only. Leave MCL and the other 10 targets as gold_rank_w50. Never open test reports. Fallback to weak_rank_bakers_mix if n_weak<20.",
            "falsify": "Public score ≤ 0.519 (frozen weak_rank_bakers_mix). Already falsified at 0.519 tie (56698950).",
            "expected_targets": ["Baker's", "ACL"],
        },
        {
            "id": "H_weak_rank_bakers_w40",
            "hypothesis": "Giving Baker's named-weak ranks minority weight (0.60 gold / 0.40 weak) will beat 0.519, because 50/50 scored 0.519 and majority-weak 0.40/0.60 scored 0.518.",
            "mechanism": "Fit gold_rank_w50 exactly (7-d λ=2 + 13-d λI=3.5, 0.50/0.50, standardize on the 58 gold studies). Separately fit named-only weak heads. On Baker's only, blend 0.60·gold_rank + 0.40·named_weak_rank. Leave the other 11 targets as gold_rank_w50. Never open test reports. Fallback to weak_rank_bakers_mix if n_weak<20.",
            "falsify": "Public score ≤ 0.519 (frozen weak_rank_bakers_mix). Already falsified at 0.519 tie (56664719).",
            "expected_targets": ["Baker's"],
        },
        {
            "id": "H_weak_rank_bakers_w60",
            "hypothesis": "Giving the Baker's named-weak ranks majority weight (0.40 gold / 0.60 weak) will beat 0.519, because the 50/50 Baker's-only mix already lifted public macro AUC from 0.518 to 0.519.",
            "mechanism": "Fit gold_rank_w50 exactly (7-d λ=2 + 13-d λI=3.5, 0.50/0.50, standardize on the 58 gold studies). Separately fit named-only weak heads. On Baker's only, blend 0.40·gold_rank + 0.60·named_weak_rank. Leave the other 11 targets as gold_rank_w50. Never open test reports. Fallback to weak_rank_bakers_mix if n_weak<20.",
            "falsify": "Public score ≤ 0.519 (frozen weak_rank_bakers_mix). Already falsified at 0.518 (56631726).",
            "expected_targets": ["Baker's"],
        },
        {
            "id": "H_weak_rank_bakers_mix",
            "hypothesis": "Keeping frozen gold_rank_w50 ranks for all 12 targets and 50/50 mixing named-object weak ranks onto Baker's only will beat 0.518, because mixing ACL/MCL/Baker's together scored 0.514 and discussion 734117 says Baker's is the strongest named object (balanced acc 0.82).",
            "mechanism": "Fit gold_rank_w50 exactly (7-d λ=2 + 13-d λI=3.5, 0.50/0.50, standardize on the 58 gold studies). Separately fit named-only weak heads (gold 0/1 + parser pos/neg on ACL / Baker's / MCL). Average ranks on the Baker's column only. Never open test reports. Fallback to gold_rank_w50 if n_weak<20.",
            "falsify": "Public score ≤ 0.518 (frozen gold_rank_w50). Already accepted at 0.519 (56600302).",
            "expected_targets": ["Baker's"],
        },
        {
            "id": "H_weak_rank_named_mix",
            "hypothesis": "Keeping frozen gold_rank_w50 ranks for all 12 targets and 50/50 mixing named-object weak ranks onto ACL / Baker's / MCL only will beat 0.518, because all-head named-only fit scored 0.502 after shifting the 4,407-row standardizer.",
            "mechanism": "Fit gold_rank_w50 exactly (7-d λ=2 + 13-d λI=3.5, 0.50/0.50, standardize on the 58 gold studies). Separately fit named-only weak heads (gold 0/1 + parser pos/neg on ACL / Baker's / MCL). Average ranks on those three columns only. Never open test reports. Fallback to gold_rank_w50 if n_weak<20.",
            "falsify": "Public score ≤ 0.518 (frozen gold_rank_w50). Already falsified at 0.514 (56571590).",
            "expected_targets": ["ACL", "Baker's", "MCL"],
        },
        {
            "id": "H_weak_rank_named",
            "hypothesis": "Restricting parser pos/neg labels to named objects (ACL, Baker's, MCL) will beat 0.518, because all-target confident labels scored 0.511 and discussion 734117 says only named objects recover while graded/unstated fail.",
            "mechanism": "Same 7-d/13-d heads and 0.50/0.50 blend. Gold 0/1 when finite. Parser pos/neg only for ACL / Baker's / MCL; other unlabeled targets stay NaN. Never open test reports. Fallback to gold_rank_w50 if n_weak<20.",
            "falsify": "Public score ≤ 0.518 (frozen gold_rank_w50). Already falsified at 0.502 (56541791).",
            "expected_targets": ["ACL", "Baker's", "MCL"],
        },
        {
            "id": "H_weak_rank_confident",
            "hypothesis": "Keeping expert 0/1 on the 58 and using only parser pos/neg mentions (masking unmentioned/uncertain/historical) will beat 0.518, because goldfill's unmentioned=0.38 mass on ~4,349 reports scored 0.504 and empty extractions should stay unlabeled (discussion 734117).",
            "mechanism": "Same 7-d/13-d series-metadata heads and 0.50/0.50 blend as gold_rank_w50. Gold 0/1 when finite. Else parse train.csv Report only (never test reports); keep pos=0.85 / neg=0.12; drop unmentioned/unc/hist as NaN so IRLS uses the finite mask. Map ranks through gold prevalence. Fallback to gold_rank_w50 if n_weak<20.",
            "falsify": "Public score ≤ 0.518 (frozen gold_rank_w50). Already falsified at 0.511 (56513588).",
            "expected_targets": ["ACL", "Baker's", "Effusion", "Medial Meniscus", "MCL"],
        },
        {
            "id": "H_weak_rank_goldfill",
            "hypothesis": "Keeping expert 0/1 labels on the 58 gold studies and using parser soft labels only for unlabeled train reports will beat 0.518, because parser-only weak_rank_calibrate scored 0.499 after overwriting gold.",
            "mechanism": "Same 7-d/13-d series-metadata heads and 0.50/0.50 blend as gold_rank_w50. For each target, use gold 0/1 when finite; otherwise parse train.csv Report (never test reports). Fit on the mixed labels; map ranks through gold prevalence. Fallback to gold_rank_w50 if n_weak<20.",
            "falsify": "Public score ≤ 0.518 (frozen gold_rank_w50). Already falsified at 0.504 (56484736).",
            "expected_targets": ["ACL", "Effusion", "Baker's", "Medial Meniscus", "Synovitis"],
        },
        {
            "id": "H_weak_rank_calibrate",
            "hypothesis": "Fitting the frozen 7-d/13-d series-metadata heads on multilingual train-report soft labels (n≈4407) and using the 58 gold studies only to map ranks through prevalence will beat 0.518, because gold-only ranking has stalled (w50=0.518, w40=0.518, lam2=0.515) and series composition already reaches ~0.595 on report-derived labels.",
            "mechanism": "Parse train.csv Report only (never test reports). Soft-label each of 12 targets with a multilingual keyword matcher plus left+right negation (Turkish izlenmedi/görülmedi). Fit 7-d (λ=2) and 13-d interact (λ=3.5) ridge logits on those soft labels; blend 0.50·rank(7-d)+0.50·rank(interact); map through gold prevalence. If n_weak<100, fall back to gold_rank_w50. No pixels, no Gaussian noise.",
            "falsify": "Public score ≤ 0.518 (frozen gold_rank_w50) or a notebook error falls back to gold-only ranks.",
            "expected_targets": ["ACL", "Effusion", "Baker's", "Medial Meniscus", "Synovitis"],
        },
        {
            "id": "H_gold_rank_lam2",
            "hypothesis": "If the 0.40/0.60 and 0.50/0.50 blends both score 0.518, the 13-d interact head is over-regularized (λ=3.5) and nearly collinear with the 7-d ranks; lowering λI to 2.0 at the frozen 0.50 blend will make plane×protocol ranks distinctive and beat 0.518.",
            "mechanism": "Same 7-d (λ=2) + 13-d interact heads as gold_rank_w50; keep 0.50·rank(7-d)+0.50·rank(interact); fit the interact head with λ=2.0 instead of 3.5. Fallback to 7-d if interact fit fails. No test reports, no pixels, no Gaussian noise.",
            "falsify": "Public score ≤ 0.518 (frozen gold_rank_w50 / tied gold_rank_w40). Already falsified at 0.515 (56418615).",
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
    # Always lead with the next untested ablation. Cycle-index rotation was
    # resubmitting falsified goldfill/calibrate write-ups on later daily slots.
    primary = catalog[0]
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
        "`weak_rank_bakers_mix` is the frozen public baseline at **0.519** (56600302).",
        "`weak_rank_bakers_dropfat` scored **0.519** (56815942) and is a falsified tie.",
        "`weak_rank_bakers_silence` scored **0.516** (56790917) and is falsified (poison).",
        "`weak_rank_bakers_goldstd` scored **0.519** (56767585) and is a falsified tie.",
        "`weak_rank_bakers_mm_mix` scored **0.517** (56745095) and is falsified.",
        "`weak_rank_bakers_acl_mix` scored **0.519** (56698950) and is a falsified tie.",
        "`weak_rank_bakers_w40` scored **0.519** (56664719) and is a falsified tie.",
        "`weak_rank_bakers_w60` scored **0.518** (56631726) and is falsified.",
        "Parser-only `weak_rank_calibrate` scored **0.499**. `weak_rank_goldfill` scored **0.504**.",
        "`weak_rank_confident` scored **0.511**. `weak_rank_named` scored **0.502**.",
        "`weak_rank_named_mix` scored **0.514**. gold_rank_w50 remains the prior floor at **0.518**.",
        "Parser Baker's ablations stalled; dropfat tied because fat ≡ fluid.",
        "`weak_rank_bakers_llm` scored **0.519** (56872526) after finding v4_blend (n=4407);",
        "the Baker's LLM metadata head was constant on the 3-study public sample, so the mix",
        "was rank-preserving vs gold. That is a falsified tie, not a discovery miss",
        "(56844594 was the miss). `weak_rank_bakers_acl_llm` scored **0.520** (56902005).",
        "`weak_rank_bakers_mcl_llm` scored **0.510** (56938248) and is poison",
        "(CSV found, n=4407, MCL std=0.408). `weak_rank_bakers_eff_llm` scored **0.524**",
        "(56995081) and is the frozen floor (CSV found, n=4407, Effusion std=0.408).",
        "Next is **`weak_rank_bakers_syn_llm`**: keep parser Baker's + LLM ACL + LLM Effusion",
        "and add LLM Synovitis only if that head has variance. Do not retry MCL.",
        "Visual MRI encoders stay out of scope.",
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
        "- Accept into the frozen baseline iff public score > 0.524 and status COMPLETE.",
        "- Otherwise keep `weak_rank_bakers_eff_llm` (0.524) as the fallback path.",
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
