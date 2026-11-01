# Cycle result — 2026-11-01 cycle 02

Strategy: `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w60`
Submission: **57866681**
Public score: **0.534** (COMPLETE)
Verdict: **poison; falsified** (vs frozen `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen` 0.535)

## Kernel evidence

- Competition root discovered at `/kaggle/input/competitions/rsna-knee-abnormality-detection`.
- v4 blend: `/kaggle/input/datasets/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv` (n=4407).
- Hybrid Fracture: `/kaggle/input/datasets/flight0234/rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv` (n=4407).
- Gemini: `/kaggle/input/datasets/yunusgmsoy/rsna-knee-llm-labels-4-source-merged/report_labels_v5.csv` (n=4407).
- Qwen: `/kaggle/input/datasets/laymond/rsna-knee-abnormality-qwen3-8b-weak-labels/weak_labels_v2_filled.csv` (n=4407).
- Mixed heads: parser Baker's; v4 ACL std=0.408 w_gold=0.5; Gemini ACL std=0.204 w_gold=0.5; v4 Effusion std=0.408 w_gold=0.4; hybrid Fracture std=0.204 w_gold=0.5; v4 LM std=0.408 w_gold=0.5; Gemini Effusion std=0.408 w_gold=0.4; Gemini LM std=0.354 w_gold=0.5; **Qwen LM std=0.408 w_gold=0.4**.
- Public 3-study LM moved **0.279–0.506 → 0.278–0.509**. ACL stayed 0.336–0.476. Effusion stayed 0.478–0.718. Baker's stayed constant 0.206897.

## Decision

Keep `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen` as the frozen fallback (0.535).
Majority Qwen LM 0.40/0.60 mixed with variance (n=4407, std=0.408) and moved public LM ranks, but public macro AUC dropped to 0.534 (−0.001).
Do not resubmit majority Qwen LM 0.40/0.60.
Next one-lever test: `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_w40` (minority Qwen2.5-32B v2 Lateral Meniscus 0.60 stack / 0.40 Qwen on the 0.535 stack).
Quota 3/5 for the Chicago day. Do not submit a second cycle in this run.
