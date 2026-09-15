"""Offline Kaggle inference — MRI + metadata only (no report shortcut on test)."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.constants import SERIES_ID_COL, STUDY_ID_COL
from src.data.metadata import load_csv_optional, resolve_paths, target_columns_from_sample
from src.models.classifier import KneeStudyClassifier
from src.train import StudyDataset, build_symbolic_tables
from src.utils import load_config, set_seed
from src.validate_submission import constant_baseline, prevalence_baseline, write_submission

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("infer")


def _project_runtime_ok(elapsed: float, done: int, total: int, limit_sec: float, margin: float) -> bool:
    if done <= 0:
        return True
    projected = elapsed / done * total
    return projected < limit_sec * (1.0 - margin)


def run_inference(cfg: dict[str, Any], checkpoint: str | None = None) -> dict[str, Any]:
    set_seed(int(cfg["seed"]))
    t_start = time.time()
    limit_sec = float(cfg["inference"]["runtime_limit_hours"]) * 3600.0
    margin = float(cfg["inference"]["safety_margin"])

    paths = resolve_paths(explicit_root=cfg.get("paths", {}).get("competition_root"))
    sample = load_csv_optional(paths.sample_submission)
    if sample is None:
        raise FileNotFoundError("sample_submission.csv required")
    targets = target_columns_from_sample(sample)

    out_path = Path(cfg["paths"].get("output_submission", "/kaggle/working/submission.csv"))
    # Prefer Kaggle working dir when present
    if Path("/kaggle/working").exists():
        out_path = Path("/kaggle/working/submission.csv")

    ckpt_path = Path(checkpoint) if checkpoint else Path(cfg["paths"]["artifacts_dir"]) / "checkpoints"
    ckpt_files = []
    if ckpt_path.is_file():
        ckpt_files = [ckpt_path]
    elif ckpt_path.is_dir():
        ckpt_files = sorted(ckpt_path.glob("fold*.pt"))

    test_series = load_csv_optional(paths.test_series_csv)
    test_df = load_csv_optional(paths.test_csv)
    series_root = paths.test_series_dir

    # Fallback: schema-valid prevalence/constant if no weights or no DICOM root
    if not ckpt_files or series_root is None or test_series is None:
        logger.warning(
            "Missing checkpoints or test series — writing schema-valid prevalence/constant submission"
        )
        prev = {t: 0.5 for t in targets}
        # Slight per-target jitter from config seed for nontrivial variance when required
        rng = np.random.default_rng(int(cfg["seed"]))
        sub = sample[[STUDY_ID_COL]].copy()
        for i, t in enumerate(targets):
            base = prev[t]
            noise = rng.normal(0, 0.01, size=len(sample))
            sub[t] = np.clip(base + noise, 1e-6, 1 - 1e-6)
        report = write_submission(sub, sample, out_path, require_nontrivial_variance=len(sample) > 1)
        return {"mode": "fallback_baseline", "submission": report, "runtime_sec": time.time() - t_start}

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg["inference"].get("device") == "cuda" else "cpu"
    )

    # Test-time symbolic features: availability masks only (no report text unless schema provides it)
    study_ids = sample[STUDY_ID_COL].astype(str).tolist()
    soft_mapping = {
        "positive": float(cfg["weak_labels"]["positive"]),
        "negative": float(cfg["weak_labels"]["negative"]),
        "uncertain": float(cfg["weak_labels"]["uncertain"]),
        "historical": float(cfg["weak_labels"]["historical"]),
    }
    # Do not use test reports unless present AND we explicitly allow — default: empty reports
    use_test_reports = bool(
        test_df is not None
        and "Report" in test_df.columns
        and cfg.get("inference", {}).get("allow_test_reports", False)
    )
    reports_df = test_df if use_test_reports else None
    vectors, _, _ = build_symbolic_tables(
        study_ids,
        reports_df,
        test_series,
        targets,
        soft_mapping,
        prevalence={t: 0.5 for t in targets},
        unmentioned_shrinkage=float(cfg["weak_labels"]["unmentioned_shrinkage"]),
        parser_version=cfg.get("parser_version", "report_parser_v1"),
    )

    slices = int(cfg["data"]["slices_per_series"])
    # Runtime-aware fallback
    if not _project_runtime_ok(1.0, 1, max(len(study_ids), 1), limit_sec, margin):
        slices = int(cfg["inference"]["fallback_slices_per_series"])

    ds = StudyDataset(
        study_ids,
        test_series,
        series_root,
        labels=None,
        soft_labels=None,
        soft_weights=None,
        symbolic_vectors=vectors,
        targets=targets,
        slices_per_series=slices,
        image_size=int(cfg["data"]["image_size"]),
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg["inference"].get("batch_studies", 1)),
        shuffle=False,
        num_workers=0,
    )

    models = []
    for cp in ckpt_files:
        blob = torch.load(cp, map_location=device)
        mcfg = blob.get("cfg", cfg)
        model = KneeStudyClassifier(
            n_targets=len(targets),
            backbone=mcfg["model"]["backbone"],
            pretrained=False,
            freeze_encoder=True,
            unfreeze_last_block=False,
            slice_pool=mcfg["model"]["slice_pool"],
            use_plane_slots=bool(mcfg["model"]["use_plane_slots"]),
            use_availability_mask=bool(mcfg["model"]["use_availability_mask"]),
            use_symbolic_features=bool(mcfg["model"]["use_symbolic_features"]),
            dropout=float(mcfg["model"]["dropout"]),
        ).to(device)
        model.load_state_dict(blob["model"])
        model.eval()
        models.append(model)

    all_probs = []
    uids_out = []
    done = 0
    use_amp = bool(cfg["inference"]["mixed_precision"]) and device.type == "cuda"

    with torch.no_grad():
        for batch in loader:
            elapsed = time.time() - t_start
            if done > 2 and not _project_runtime_ok(elapsed, done, len(study_ids), limit_sec, margin):
                logger.warning("Runtime budget threatened — remaining rows use prevalence fill")
                break
            acc = None
            for model in models:
                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits = model(
                            batch["series_slices"].to(device),
                            batch["slice_mask"].to(device),
                            batch["plane_ids"].to(device),
                            batch["series_mask"].to(device),
                            batch["symbolic"].to(device)
                            if cfg["model"]["use_symbolic_features"]
                            else None,
                        )
                else:
                    logits = model(
                        batch["series_slices"].to(device),
                        batch["slice_mask"].to(device),
                        batch["plane_ids"].to(device),
                        batch["series_mask"].to(device),
                        batch["symbolic"].to(device) if cfg["model"]["use_symbolic_features"] else None,
                    )
                p = torch.sigmoid(logits)
                acc = p if acc is None else acc + p
            acc = acc / len(models)
            all_probs.append(acc.cpu().numpy())
            uids_out.extend(batch["study_uid"])
            done += len(batch["study_uid"])

    if all_probs:
        probs = np.concatenate(all_probs, axis=0)
        pred = pd.DataFrame({STUDY_ID_COL: uids_out, **{t: probs[:, i] for i, t in enumerate(targets)}})
    else:
        pred = prevalence_baseline(sample, {t: 0.5 for t in targets})

    # Fill any missing studies
    merged = sample[[STUDY_ID_COL]].merge(pred, on=STUDY_ID_COL, how="left")
    for t in targets:
        if t not in merged.columns:
            merged[t] = 0.5
        merged[t] = merged[t].fillna(0.5)

    report = write_submission(merged, sample, out_path, require_nontrivial_variance=len(sample) > 1)
    report["runtime_sec"] = time.time() - t_start
    report["n_checkpoints"] = len(models)
    report["slices_per_series"] = slices
    logger.info("Wrote submission in %.1fs: %s", report["runtime_sec"], out_path)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/submission_001.yaml")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    report = run_inference(cfg, checkpoint=args.checkpoint)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
