# Cycle result — 2026-11-05 cycle 01

Strategy: `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_eff_sol`
Submission: **58017601**
Public score: **0.535** (COMPLETE)
Verdict: **tie; falsified as an improvement** (vs frozen `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen` 0.535)

Kernel v1 COMPLETE. This was a real mix, not an implementation miss.

## Kernel evidence

- Competition root discovered at `/kaggle/input/competitions/rsna-knee-abnormality-detection`.
- v4 blend: `/kaggle/input/datasets/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv` (n=4407).
- Hybrid Fracture: `/kaggle/input/datasets/flight0234/rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv` (n=4407).
- Gemini: `/kaggle/input/datasets/yunusgmsoy/rsna-knee-llm-labels-4-source-merged/report_labels_v5.csv` (n=4407).
- Qwen: `/kaggle/input/datasets/laymond/rsna-knee-abnormality-qwen3-8b-weak-labels/weak_labels_v2_filled.csv` (n=4407).
- Sol: `/kaggle/input/datasets/lixin73/rsna-knee-llm-report-labels-sol56/report_labels_gpt56sol.csv` (n=4407).
- Mixed heads: parser Baker's; v4 ACL std=0.408 w_gold=0.5; Gemini ACL std=0.204 w_gold=0.5; v4 Effusion std=0.408 w_gold=0.4; hybrid Fracture std=0.204 w_gold=0.5; v4 LM std=0.408 w_gold=0.5; Gemini Effusion std=0.408 w_gold=0.4; Gemini LM std=0.354 w_gold=0.5; Qwen LM std=0.408 w_gold=0.5; **sol Effusion std=0.408 w_gold=0.5**.
- Public 3-study Effusion stayed **0.478449–0.723448** (same band as Qwen/frontier/pilk/2src Effusion). ACL stayed 0.335668–0.476293. LM stayed 0.279364–0.505927. Fracture stayed 0.247845–0.372845. Baker's stayed constant 0.206897. MM stayed 0.323276–0.510776.

## Decision

Keep `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen` as the frozen fallback (0.535).
Sol Effusion 50/50 mixed with variance (n=4407, std=0.408) but public Effusion ranks and macro AUC stayed 0.535.
Do not resubmit sol Effusion 50/50. Do not spend quota on sol Effusion mix-weights after a tie.
Next one-lever test: `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_mm_sol` (keep Qwen LM at accepted 50/50; add lixin73 GPT-5.6-Sol `report_labels_gpt56sol.csv` Medial Meniscus 50/50 on the 0.535 stack). MM is the remaining unused sol named-object (parser 0.517 poison; v4 0.527 tie; Gemini 0.534 tie; Qwen/frontier/pilk/2src ties).
Quota 2/5 for the Chicago day 2026-11-05. Do not submit a second cycle in this run.
