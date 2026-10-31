# Cycle result — 2026-10-31 cycle 03

Strategy: `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_bak_gem`
Submission: **57835922**
Public score: **0.531** (COMPLETE)
Verdict: **falsified poison** (vs frozen `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem` 0.534)

## Kernel evidence

- Competition root discovered at `/kaggle/input/competitions/rsna-knee-abnormality-detection`.
- v4 blend: `/kaggle/input/datasets/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv` (n=4407).
- Hybrid Fracture: `/kaggle/input/datasets/flight0234/rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv` (n=4407).
- Gemini: `/kaggle/input/datasets/yunusgmsoy/rsna-knee-llm-labels-4-source-merged/report_labels_v5.csv` (n=4407).
- Mixed heads: parser Baker's; v4 ACL std=0.408 w_gold=0.5; Gemini ACL std=0.204 w_gold=0.5; v4 Effusion std=0.408 w_gold=0.4; hybrid Fracture std=0.204 w_gold=0.5; v4 LM std=0.408 w_gold=0.5; Gemini Effusion std=0.408 w_gold=0.4; Gemini LM std=0.354 w_gold=0.5; **Gemini Baker's std=0.408 w_gold=0.5**.
- Public 3-study Baker's widened from a constant **0.206897** on the 0.534 stack to **0.144–0.269**. Public ACL/Effusion/LM/MM/Fracture ranges were unchanged.

## Decision

Keep `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem` as the frozen fallback.
Gemini Baker's 50/50 mixed with variance but poisoned public macro AUC (−0.003). Parser Baker's was constant on the public 3-study sample; overlaying Gemini Baker's ranks disrupted the hidden public set.
Do not resubmit Gemini Baker's 50/50 or Gemini Baker's mix-weights. Do not remix Baker's with any LLM key.
Next one-lever test: `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_acl_qwen` (Qwen2.5-32B v2 ACL 50/50 on the 0.534 stack). ACL already transferred (v4 0.520, Gemini 0.531) and Qwen ACL is 0.981 vs gold — a new source, not a Baker's swap.
