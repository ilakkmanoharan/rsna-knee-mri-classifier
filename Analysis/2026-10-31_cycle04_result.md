# Cycle result — 2026-10-31 cycle 04

Strategy: `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_acl_qwen`
Submission: **57837907**
Public score: **0.534** (COMPLETE)
Verdict: **tie; falsified as an improvement** (vs frozen `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem` 0.534)

## Kernel evidence

- Competition root discovered at `/kaggle/input/competitions/rsna-knee-abnormality-detection`.
- v4 blend: `/kaggle/input/datasets/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv` (n=4407).
- Hybrid Fracture: `/kaggle/input/datasets/flight0234/rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv` (n=4407).
- Gemini: `/kaggle/input/datasets/yunusgmsoy/rsna-knee-llm-labels-4-source-merged/report_labels_v5.csv` (n=4407).
- Qwen: `/kaggle/input/datasets/laymond/rsna-knee-abnormality-qwen3-8b-weak-labels/weak_labels_v2_filled.csv` (n=4407).
- Mixed heads: parser Baker's; v4 ACL std=0.408 w_gold=0.5; Gemini ACL std=0.204 w_gold=0.5; v4 Effusion std=0.408 w_gold=0.4; hybrid Fracture std=0.204 w_gold=0.5; v4 LM std=0.408 w_gold=0.5; Gemini Effusion std=0.408 w_gold=0.4; Gemini LM std=0.354 w_gold=0.5; **Qwen ACL std=0.408 w_gold=0.5**.
- Public 3-study ACL widened **0.336–0.476 → 0.312–0.508**. Baker's stayed constant 0.206897. Effusion stayed 0.478–0.718. LM stayed 0.287–0.490.

## Decision

Keep `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem` as the frozen fallback.
Qwen ACL 50/50 mixed with variance (n=4407, std=0.408) and moved public ACL ranks, but public macro AUC stayed 0.534.
Do not resubmit Qwen ACL 50/50 or Qwen ACL mix-weights (same rule as Gemini MM after a 50/50 tie).
Next one-lever test: `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_eff_qwen` (Qwen2.5-32B v2 Effusion 50/50 on the 0.534 stack). Effusion is the remaining unused Qwen column on a transferring named object (v4 0.524/0.530, Gemini 0.532/0.533).
Quota 5/5 for the Chicago day. Do not submit a second cycle in this run.
