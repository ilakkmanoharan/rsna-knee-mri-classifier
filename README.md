# RSNA Knee Abnormality Detection — Submission 001 (ASRA)

Reproducible first Kaggle submission for **RSNA Knee Abnormality Detection**: 12 study-level abnormality probabilities from multi-series knee MRI, with a symbolic evidence layer and an ASRA experiment controller.

This is an engineering baseline and research scaffold — not a claim that symbolic rules alone diagnose MRI.

## Targets

Order is taken from `sample_submission.csv` at runtime (authoritative). Defaults:

`ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`

## Layout

```text
configs/submission_001.yaml
notebooks/
  00_data_audit.ipynb
  01_report_world_model.ipynb
  02_train_baseline.ipynb
  03_kaggle_inference.ipynb
src/
  data/            # DICOM, metadata, sampling
  symbolic/        # ontology, report parser, evidence graph, soft constraints
  asra/            # hypothesis ledger + experiment helpers
  models/          # encoder, pooling, classifier
  train.py
  infer.py
  validate_submission.py
tests/
artifacts/         # folds, oof, checkpoints, ledger
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Mount or symlink competition data so the root contains at least `sample_submission.csv` (and typically `train.csv`, `train_series.csv`, `train_series/`, `test_*`).

```bash
# optional local override
# edit configs/submission_001.yaml paths.competition_root
```

## Quick commands

```bash
# Unit tests (parser, folds, submission contract, pooling)
pytest -q

# Stage A audit + Stage D training (needs mounted data + DICOMs)
python -m src.train --config configs/submission_001.yaml

# Offline inference → submission.csv
python -m src.infer --config configs/submission_001.yaml
```

## Design notes

- **Reports** are used only as training-time weak supervision unless the mounted test schema explicitly provides and permits them.
- **Symbolic layer** supplies soft labels, availability masks, and weak consistency penalties — never hard diagnoses from missing views.
- **ASRA** records H0–H5 in `artifacts/hypothesis_ledger/` and accepts changes only when OOF evidence supports them.
- **Inference** discovers paths, ensembles fold checkpoints, and falls back to fewer slices if the 9h runtime budget (15% safety margin) is threatened.

## Cloud (laptop closed)

See [`agent/CLOUD.md`](agent/CLOUD.md). Primary runner is **GitHub Actions** (`.github/workflows/agent1-cloud.yml`). Optional: Cursor Cloud Automations using `agent/cursor_cloud_automation_prompt.md`.

## Supervision

A Grok bot (`.github/workflows/grok-supervisor.yml`) audits `Research/`, `Analysis/`, `Hypothesis/` and whether the day is on pace to spend all 5 Kaggle submissions (30-60 min apart), re-runs missed cycles, and emails/files an issue when it needs a decision. See [`agent/GROK_BOT.md`](agent/GROK_BOT.md).

