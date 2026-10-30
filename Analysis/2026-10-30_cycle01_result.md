# Cycle result — 2026-10-30 cycle 01

Strategy: `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60`
Submission: **57800238**
Public score: **0.533** (COMPLETE)
Verdict: **accepted** (+0.001 vs frozen `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem` 0.532)

## Kernel evidence

- Competition root discovered at `/kaggle/input/competitions/rsna-knee-abnormality-detection`.
- v4 blend: `/kaggle/input/datasets/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv` (n=4407).
- Hybrid Fracture: `/kaggle/input/datasets/flight0234/rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv` (n=4407).
- Gemini: `/kaggle/input/datasets/yunusgmsoy/rsna-knee-llm-labels-4-source-merged/report_labels_v5.csv` (n=4407).
- Mixed heads: parser Baker's; v4 ACL std=0.408 w_gold=0.5; Gemini ACL std=0.204 w_gold=0.5; v4 Effusion std=0.408 w_gold=0.4; hybrid Fracture std=0.204 w_gold=0.5; v4 LM std=0.408 w_gold=0.5; **Gemini Effusion std=0.408 w_gold=0.4**.
- Public 3-study Effusion widened **0.478–0.716 → 0.478–0.718**. ACL stayed 0.336–0.476.

## Decision

Lock `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60` as the frozen fallback.
Next one-lever test: `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w40` (minority Gemini Effusion 0.60/0.40).
Do not resubmit majority Gemini Effusion 0.40/0.60.
