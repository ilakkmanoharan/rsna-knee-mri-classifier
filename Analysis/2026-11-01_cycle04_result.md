# Cycle result — 2026-11-01 cycle 04

Strategy: `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_qwen`
Submission: **57871069**
Public score: **0.535** (COMPLETE)
Verdict: **tie; falsified as an improvement** (vs frozen `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen` 0.535)

## Kernel evidence

- Competition root discovered at `/kaggle/input/competitions/rsna-knee-abnormality-detection`.
- v4 blend: `/kaggle/input/datasets/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv` (n=4407).
- Hybrid Fracture: `/kaggle/input/datasets/flight0234/rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv` (n=4407).
- Gemini: `/kaggle/input/datasets/yunusgmsoy/rsna-knee-llm-labels-4-source-merged/report_labels_v5.csv` (n=4407).
- Qwen: `/kaggle/input/datasets/laymond/rsna-knee-abnormality-qwen3-8b-weak-labels/weak_labels_v2_filled.csv` (n=4407).
- Mixed heads: parser Baker's; v4 ACL std=0.408 w_gold=0.5; Gemini ACL std=0.204 w_gold=0.5; v4 Effusion std=0.408 w_gold=0.4; hybrid Fracture std=0.204 w_gold=0.5; v4 LM std=0.408 w_gold=0.5; Gemini Effusion std=0.408 w_gold=0.4; Gemini LM std=0.354 w_gold=0.5; Qwen LM std=0.408 w_gold=0.5; **Qwen MM std=0.354 w_gold=0.5**.
- Public 3-study MM stayed **0.323–0.511** (0.323276–0.510776), matching Gemini MM and the 0.535 stack. Public LM stayed 0.279–0.506 (0.279364–0.505927). Fracture stayed 0.248–0.373. Baker's stayed constant 0.206897.

## Decision

Keep `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen` as the frozen fallback (0.535).
Qwen MM 50/50 mixed with variance (n=4407, std=0.354) but did not move public MM ranks or macro AUC.
Same pattern as v4 MM (0.527 tie) and Gemini MM (0.534 tie): Medial Meniscus labels transfer enough to keep variance and stay rank-preserving on the public sample.
Do not resubmit Qwen MM 50/50. Do not spend quota on Qwen MM mix-weights after a tie.
Next one-lever test: `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_qwen` (keep Qwen LM at accepted 50/50; add Qwen2.5-32B v2 Fracture 50/50 on the 0.535 stack). Fracture is the remaining unused Qwen named-object column that is not a documented poison (Baker's LLM remix forbidden; MCL/Synovitis/Contusion/OA poisoned).
Quota 5/5 for the Chicago day. Do not submit a second cycle in this run.
