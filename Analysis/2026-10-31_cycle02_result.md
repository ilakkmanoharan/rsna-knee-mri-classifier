# Cycle result — 2026-10-31 cycle 02

Strategy: `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_mm_gem`
Submission: **57833621**
Public score: **0.534** (COMPLETE)
Verdict: **tie; falsified as an improvement** (vs frozen `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem` 0.534)

## Kernel evidence

- Competition root discovered at `/kaggle/input/competitions/rsna-knee-abnormality-detection`.
- v4 blend: `/kaggle/input/datasets/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv` (n=4407).
- Hybrid Fracture: `/kaggle/input/datasets/flight0234/rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv` (n=4407).
- Gemini: `/kaggle/input/datasets/yunusgmsoy/rsna-knee-llm-labels-4-source-merged/report_labels_v5.csv` (n=4407).
- Mixed heads: parser Baker's; v4 ACL std=0.408 w_gold=0.5; Gemini ACL std=0.204 w_gold=0.5; v4 Effusion std=0.408 w_gold=0.4; hybrid Fracture std=0.204 w_gold=0.5; v4 LM std=0.408 w_gold=0.5; Gemini Effusion std=0.408 w_gold=0.4; Gemini LM std=0.354 w_gold=0.5; **Gemini MM std=0.354 w_gold=0.5**.
- Public 3-study Medial Meniscus stayed **0.323–0.511** (same as the 0.534 stack). Public LM stayed 0.287–0.490.

## Decision

Keep `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem` as the frozen fallback.
Gemini MM 50/50 mixed with variance but did not move public macro AUC (same pattern as v4 MM 0.527 tie).
Do not resubmit Gemini MM 50/50 or Gemini MM mix-weights.
Next one-lever test: `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem` (Gemini Baker's 50/50 on the 0.534 stack). v4 Baker's LLM was constant on the public 3-study sample; Gemini Baker's is the remaining unused Gemini named-object that is not a documented poison.
