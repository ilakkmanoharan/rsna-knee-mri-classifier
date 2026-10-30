# Cycle result — 2026-10-30 cycle 03

Strategy: `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem`
Submission: **57804837**
Public score: **0.534** (COMPLETE)
Verdict: **accepted** (+0.001 vs frozen `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_w60` 0.533)

## Kernel evidence

- Competition root discovered at `/kaggle/input/competitions/rsna-knee-abnormality-detection`.
- v4 blend: `/kaggle/input/datasets/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv` (n=4407).
- Hybrid Fracture: `/kaggle/input/datasets/flight0234/rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv` (n=4407).
- Gemini: `/kaggle/input/datasets/yunusgmsoy/rsna-knee-llm-labels-4-source-merged/report_labels_v5.csv` (n=4407).
- Mixed heads: parser Baker's; v4 ACL std=0.408 w_gold=0.5; Gemini ACL std=0.204 w_gold=0.5; v4 Effusion std=0.408 w_gold=0.4; hybrid Fracture std=0.204 w_gold=0.5; v4 LM std=0.408 w_gold=0.5; Gemini Effusion std=0.408 w_gold=0.4; **Gemini LM std=0.354 w_gold=0.5**.
- Public 3-study Lateral Meniscus compressed **0.303–0.522 → 0.287–0.490**. Effusion stayed 0.478–0.718. ACL stayed 0.336–0.476.

## Decision

Keep `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem` as the frozen fallback.
Gemini LM 50/50 transferred (+0.001) even though the public 3-study LM range compressed.
Next one-lever test: `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_w60` (majority Gemini LM 0.40/0.60 on the 0.534 stack).
Do not resubmit Gemini LM 50/50.
