# Cycle result — 2026-11-01 cycle 03

Strategy: `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w40`
Submission: **57868885**
Public score: **0.535** (COMPLETE)
Verdict: **tie; falsified as an improvement** (vs frozen `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen` 0.535)

## Kernel evidence

- Competition root discovered at `/kaggle/input/competitions/rsna-knee-abnormality-detection`.
- v4 blend: `/kaggle/input/datasets/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv` (n=4407).
- Hybrid Fracture: `/kaggle/input/datasets/flight0234/rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv` (n=4407).
- Gemini: `/kaggle/input/datasets/yunusgmsoy/rsna-knee-llm-labels-4-source-merged/report_labels_v5.csv` (n=4407).
- Qwen: `/kaggle/input/datasets/laymond/rsna-knee-abnormality-qwen3-8b-weak-labels/weak_labels_v2_filled.csv` (n=4407).
- Mixed heads: parser Baker's; v4 ACL std=0.408 w_gold=0.5; Gemini ACL std=0.204 w_gold=0.5; v4 Effusion std=0.408 w_gold=0.4; hybrid Fracture std=0.204 w_gold=0.5; v4 LM std=0.408 w_gold=0.5; Gemini Effusion std=0.408 w_gold=0.4; Gemini LM std=0.354 w_gold=0.5; **Qwen LM std=0.408 w_gold=0.6**.
- Public 3-study LM moved **0.279–0.506 → 0.281–0.503** (0.280927–0.502802). ACL stayed 0.336–0.476. Effusion stayed 0.478–0.718. Baker's stayed constant 0.206897. MM stayed 0.323–0.511.

## Decision

Keep `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen` as the frozen fallback (0.535).
Minority Qwen LM 0.60/0.40 mixed with variance (n=4407, std=0.408) and slightly compressed public LM ranks, but public macro AUC tied 0.535.
Qwen LM mix-weights are exhausted (50/50=0.535 accepted, w60=0.534 poison, w40=0.535 tie).
Do not resubmit minority Qwen LM 0.60/0.40.
Next one-lever test: `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_qwen` (keep Qwen LM at accepted 50/50; add Qwen2.5-32B v2 Medial Meniscus 50/50 on the 0.535 stack).
Quota 4/5 for the Chicago day. Do not submit a second cycle in this run.
