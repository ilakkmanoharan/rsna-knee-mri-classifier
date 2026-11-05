# Cycle result — 2026-11-05 cycle 03

Strategy: `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_acl_3src`
Submission: **58022193**
Public score: **0.534** (COMPLETE)
Verdict: **falsified poison** (vs frozen `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen` 0.535)

Kernel v1 COMPLETE. This was a real mix, not an implementation miss.

## Kernel evidence

- Competition root discovered at `/kaggle/input/competitions/rsna-knee-abnormality-detection`.
- v4 blend: `/kaggle/input/datasets/stevenleehans/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv` (n=4407).
- Hybrid Fracture: `/kaggle/input/datasets/flight0234/rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv` (n=4407).
- Gemini: `/kaggle/input/datasets/yunusgmsoy/rsna-knee-llm-labels-4-source-merged/report_labels_v5.csv` (n=4407).
- Qwen: `/kaggle/input/datasets/laymond/rsna-knee-abnormality-qwen3-8b-weak-labels/weak_labels_v2_filled.csv` (n=4407).
- 3-source: `/kaggle/input/datasets/yunusgmsoy/rsna-knee-abnormality-3-source-merged-labels/report_labels_v3.csv` (n=4407).
- Mixed heads: parser Baker's; v4 ACL std=0.408 w_gold=0.5; Gemini ACL std=0.204 w_gold=0.5; v4 Effusion std=0.408 w_gold=0.4; hybrid Fracture std=0.204 w_gold=0.5; v4 LM std=0.408 w_gold=0.5; Gemini Effusion std=0.408 w_gold=0.4; Gemini LM std=0.354 w_gold=0.5; Qwen LM std=0.408 w_gold=0.5; **3-source ACL std=0.354 w_gold=0.5**.
- Public 3-study ACL widened **0.312231–0.476293** vs Gemini 0.336–0.476 (same band as frontier/2src ACL). LM stayed 0.279364–0.505927. Effusion stayed 0.478449–0.718448. MM stayed 0.323276–0.510776. Fracture stayed 0.247845–0.372845. Baker's stayed constant 0.206897.

## Decision

Keep `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen` as the frozen fallback (0.535).
3-source ACL 50/50 mixed with variance (n=4407, std=0.354) but public ACL ranks widened into the same 0.312–0.476 band that already tied or failed on frontier/2src ACL, and public macro AUC dropped −0.001.
Do not resubmit 3-source ACL 50/50. Do not spend quota on 3-source ACL mix-weights after a poison.
Next one-lever test: `weak_rank_bakers_frac_hyb_lm_eff_acl_gem_eff_gem_lm_gem_lm_qwen_lm_3src` (keep Qwen LM at accepted 50/50; add yunusgmsoy 3-source `report_labels_v3.csv` Lateral Meniscus 50/50 on the 0.535 stack). LM is the remaining unused 3-source named-object that transferred on this stack (v4 0.529, Gemini 0.534, Qwen 0.535).
Quota 4/5 for the Chicago day 2026-11-05. Do not submit a second cycle in this run.
