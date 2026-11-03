# Cycle result — 2026-11-03 cycle 00

Strategy: `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_frac_front`
Submission: **57938474**
Public score: **0.530** (COMPLETE)
Private score: 0.520
Verdict: **poison; falsified** (vs frozen `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen` 0.535)

Kernel v1 hit `SyntaxError: ':' expected after dictionary key` (missing dict values after `mm_front` when `frac_front` was appended). v2 after the dict-value restore completed and produced this score.

## Kernel evidence

- Competition root discovered at `/kaggle/input/competitions/rsna-knee-abnormality-detection`.
- v4 blend: `/kaggle/input/datasets/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv` (n=4407).
- Hybrid Fracture: `/kaggle/input/datasets/flight0234/rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv` (n=4407).
- Gemini: `/kaggle/input/datasets/yunusgmsoy/rsna-knee-llm-labels-4-source-merged/report_labels_v5.csv` (n=4407).
- Qwen: `/kaggle/input/datasets/laymond/rsna-knee-abnormality-qwen3-8b-weak-labels/weak_labels_v2_filled.csv` (n=4407).
- Frontier: `/kaggle/input/datasets/cdeotte/rsna-knee-frontier-llm-hybrid-labels/blend_top3_argon_astra_opus_all4314_labels.csv` (n=4314).
- Mixed heads: parser Baker's; v4 ACL std=0.408 w_gold=0.5; Gemini ACL std=0.204 w_gold=0.5; v4 Effusion std=0.408 w_gold=0.4; hybrid Fracture std=0.204 w_gold=0.5; v4 LM std=0.408 w_gold=0.5; Gemini Effusion std=0.408 w_gold=0.4; Gemini LM std=0.354 w_gold=0.5; Qwen LM std=0.408 w_gold=0.5; **frontier Fracture std=0.408 w_gold=0.5**.
- Public 3-study Fracture expanded **0.247845–0.372845 → 0.247845–0.404095** (same band as Qwen Fracture 0.529). ACL stayed 0.336–0.476. LM stayed 0.279–0.506. Effusion stayed 0.478–0.718. Baker's stayed constant 0.206897. MM stayed 0.323–0.511.

## Decision

Keep `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen` as the frozen fallback (0.535).
Frontier Fracture 50/50 mixed with variance (n=4314, std=0.408) and moved public Fracture ranks, but public macro AUC dropped to 0.530 (−0.005).
The Deotte frontier named-object sweep is exhausted: LM/ACL/Effusion/MM tied 0.535; Fracture poisoned 0.530.
Do not resubmit frontier Fracture 50/50. Do not spend quota on frontier Fracture mix-weights after a poison.
Next one-lever test: `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_pilk` (keep Qwen LM at accepted 50/50; add pilkwang `report_labels_v2.csv` ACL 50/50 on the 0.535 stack). ACL still transfers on this stack (v4 0.520, Gemini 0.531, Qwen ACL tied 0.534) and pilkwang is an unused report-only key (n=4406, ACL std=0.328).
Quota 1/5 for the Chicago day. Do not submit a second cycle in this run.
