# Cycle result — 2026-11-04 cycle 04

Strategy: `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_sol`
Submission: **58012035**
Public score: **0.534** (COMPLETE)
Verdict: **poison; falsified** (vs frozen `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen` 0.535)

Kernel v1 COMPLETE. This was a real mix, not an implementation miss.

## Kernel evidence

- Competition root discovered at `/kaggle/input/competitions/rsna-knee-abnormality-detection`.
- v4 blend: `/kaggle/input/datasets/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv` (n=4407).
- Hybrid Fracture: `/kaggle/input/datasets/flight0234/rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv` (n=4407).
- Gemini: `/kaggle/input/datasets/yunusgmsoy/rsna-knee-llm-labels-4-source-merged/report_labels_v5.csv` (n=4407).
- Qwen: `/kaggle/input/datasets/laymond/rsna-knee-abnormality-qwen3-8b-weak-labels/weak_labels_v2_filled.csv` (n=4407).
- Sol: `/kaggle/input/datasets/lixin73/rsna-knee-llm-report-labels-sol56/report_labels_gpt56sol.csv` (n=4407).
- Mixed heads: parser Baker's; v4 ACL std=0.408 w_gold=0.5; Gemini ACL std=0.204 w_gold=0.5; v4 Effusion std=0.408 w_gold=0.4; hybrid Fracture std=0.204 w_gold=0.5; v4 LM std=0.408 w_gold=0.5; Gemini Effusion std=0.408 w_gold=0.4; Gemini LM std=0.354 w_gold=0.5; Qwen LM std=0.408 w_gold=0.5; **sol ACL std=0.408 w_gold=0.5**.
- Public 3-study ACL widened **0.336–0.476 → 0.312231–0.507543**. LM stayed 0.279364–0.505927. Effusion stayed 0.478449–0.718448. Fracture stayed 0.247845–0.372845. Baker's stayed constant 0.206897. MM stayed 0.323276–0.510776.

## Decision

Keep `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen` as the frozen fallback (0.535).
Sol ACL 50/50 mixed with variance (n=4407, std=0.408) and moved public ACL ranks, but public macro AUC dropped to 0.534 (−0.001). Same poison pattern as pilkwang ACL (0.534, 57941669).
Do not resubmit sol ACL 50/50. Do not spend quota on sol ACL mix-weights after a poison.
Next one-lever test: `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_sol` (keep Qwen LM at accepted 50/50; add lixin73 GPT-5.6-Sol `report_labels_gpt56sol.csv` Lateral Meniscus 50/50 on the 0.535 stack). LM is the remaining unused sol named-object that actually transferred (v4 0.529, Gemini 0.534, Qwen 0.535). After that, unused non-poison sol named-objects are Effusion and MM.
Quota 5/5 for the Chicago day 2026-11-04. Do not submit a second cycle in this run.
