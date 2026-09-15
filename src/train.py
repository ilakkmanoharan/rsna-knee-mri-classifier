"""Training entrypoint: folds, weak labels, frozen-encoder baseline, ASRA logging."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.asra.experiment import save_oof_report
from src.asra.hypothesis import Hypothesis
from src.asra.ledger import HypothesisLedger
from src.constants import PLANES, REPORT_COL, SERIES_ID_COL, STUDY_ID_COL
from src.data.dicom import load_series_array, resolve_plane
from src.data.metadata import (
    extract_labels,
    load_csv_optional,
    make_folds,
    resolve_paths,
    save_audit,
    audit_dataset,
    target_columns_from_sample,
)
from src.data.sampling import sample_volume
from src.models.classifier import KneeStudyClassifier
from src.symbolic.constraints import consistency_loss, masked_bce_with_logits
from src.symbolic.evidence_graph import build_knee_state, knee_state_to_features, symbolic_feature_dim
from src.symbolic.report_parser import ReportParser
from src.utils import load_config, macro_auc_from_dict, per_target_roc_auc, set_seed
from src.validate_submission import prevalence_baseline, write_submission

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train")

PLANE_TO_ID = {p: i for i, p in enumerate(PLANES)}


class StudyDataset(Dataset):
    def __init__(
        self,
        study_ids: list[str],
        series_df: pd.DataFrame,
        series_root: Path,
        labels: pd.DataFrame | None,
        soft_labels: pd.DataFrame | None,
        soft_weights: pd.DataFrame | None,
        symbolic_vectors: dict[str, np.ndarray],
        targets: list[str],
        slices_per_series: int = 16,
        image_size: int = 224,
        max_series: int = 8,
    ) -> None:
        self.study_ids = list(study_ids)
        self.series_df = series_df
        self.series_root = Path(series_root)
        self.labels = labels
        self.soft_labels = soft_labels
        self.soft_weights = soft_weights
        self.symbolic_vectors = symbolic_vectors
        self.targets = targets
        self.slices_per_series = slices_per_series
        self.image_size = image_size
        self.max_series = max_series

    def __len__(self) -> int:
        return len(self.study_ids)

    def _label_row(self, df: pd.DataFrame | None, study: str) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.targets)
        vals = np.full(n, np.nan, dtype=np.float32)
        if df is None or STUDY_ID_COL not in df.columns:
            return vals, np.zeros(n, dtype=np.float32)
        row = df[df[STUDY_ID_COL].astype(str) == str(study)]
        if row.empty:
            return vals, np.zeros(n, dtype=np.float32)
        r = row.iloc[0]
        for i, t in enumerate(self.targets):
            if t in r and pd.notna(r[t]):
                vals[i] = float(r[t])
        mask = np.isfinite(vals).astype(np.float32)
        vals = np.nan_to_num(vals, nan=0.0)
        return vals, mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        study = str(self.study_ids[idx])
        sub = self.series_df[self.series_df[STUDY_ID_COL].astype(str) == study]
        series_tensors = []
        slice_masks = []
        plane_ids = []

        for _, row in sub.head(self.max_series).iterrows():
            sid = str(row[SERIES_ID_COL])
            sdir = self.series_root / study / sid
            if not sdir.exists():
                # alternate layout: series_root/series_uid
                alt = self.series_root / sid
                sdir = alt if alt.exists() else sdir
            vol, metas, info = load_series_array(sdir, image_size=self.image_size)
            if vol is None or vol.shape[0] == 0:
                continue
            vol = sample_volume(vol, self.slices_per_series)
            s = vol.shape[0]
            pad_s = self.slices_per_series
            canvas = np.zeros((pad_s, self.image_size, self.image_size), dtype=np.float32)
            canvas[:s] = vol[:pad_s]
            mask = np.zeros(pad_s, dtype=np.float32)
            mask[:s] = 1.0
            meta_plane = row.get("Anatomical_Plane")
            orient = metas[0].orientation if metas else None
            plane, _ = resolve_plane(orient, metadata_plane=str(meta_plane) if pd.notna(meta_plane) else None)
            plane = info.get("plane", plane)
            series_tensors.append(canvas)
            slice_masks.append(mask)
            plane_ids.append(PLANE_TO_ID.get(plane, 3))

        nser = len(series_tensors)
        out_slices = np.zeros(
            (self.max_series, self.slices_per_series, self.image_size, self.image_size),
            dtype=np.float32,
        )
        out_smask = np.zeros((self.max_series, self.slices_per_series), dtype=np.float32)
        out_planes = np.zeros(self.max_series, dtype=np.int64)
        out_ser_mask = np.zeros(self.max_series, dtype=np.float32)
        for i in range(nser):
            out_slices[i] = series_tensors[i]
            out_smask[i] = slice_masks[i]
            out_planes[i] = plane_ids[i]
            out_ser_mask[i] = 1.0

        gold, gold_mask = self._label_row(self.labels, study)
        soft, _ = self._label_row(self.soft_labels, study)
        sw, sw_mask = self._label_row(self.soft_weights, study)
        if sw_mask.sum() == 0 and self.soft_weights is not None:
            sw = np.zeros(len(self.targets), dtype=np.float32)

        sym = self.symbolic_vectors.get(study)
        if sym is None:
            sym = np.zeros(symbolic_feature_dim(len(self.targets)), dtype=np.float32)

        return {
            "study_uid": study,
            "series_slices": torch.from_numpy(out_slices),
            "slice_mask": torch.from_numpy(out_smask),
            "plane_ids": torch.from_numpy(out_planes),
            "series_mask": torch.from_numpy(out_ser_mask),
            "symbolic": torch.from_numpy(sym.astype(np.float32)),
            "gold": torch.from_numpy(gold),
            "gold_mask": torch.from_numpy(gold_mask),
            "soft": torch.from_numpy(soft),
            "soft_weight": torch.from_numpy(sw.astype(np.float32)),
            "pref_cov": torch.from_numpy(sym[-len(self.targets) :].astype(np.float32))
            if len(sym) >= len(self.targets)
            else torch.ones(len(self.targets)),
        }


def build_symbolic_tables(
    study_ids: list[str],
    train_df: pd.DataFrame | None,
    series_df: pd.DataFrame | None,
    targets: list[str],
    soft_mapping: dict[str, float],
    prevalence: dict[str, float],
    unmentioned_shrinkage: float,
    parser_version: str,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    parser = ReportParser(targets=targets, parser_version=parser_version)
    reports = {}
    if train_df is not None and REPORT_COL in train_df.columns:
        for _, row in train_df.iterrows():
            reports[str(row[STUDY_ID_COL])] = str(row[REPORT_COL]) if pd.notna(row[REPORT_COL]) else ""

    vectors: dict[str, np.ndarray] = {}
    soft_rows = []
    weight_rows = []
    for sid in study_ids:
        state = build_knee_state(sid, series_df, reports.get(sid, ""), parser)
        feats = knee_state_to_features(
            state,
            targets,
            soft_mapping,
            prevalence=prevalence,
            unmentioned_shrinkage=unmentioned_shrinkage,
        )
        vectors[sid] = feats.as_vector()
        soft_rows.append({STUDY_ID_COL: sid, **{t: float(feats.soft_labels[i]) for i, t in enumerate(targets)}})
        weight_rows.append({STUDY_ID_COL: sid, **{t: float(feats.soft_weights[i]) for i, t in enumerate(targets)}})
    return vectors, pd.DataFrame(soft_rows), pd.DataFrame(weight_rows)


def train_one_epoch(model, loader, optimizer, cfg, targets, device, scaler=None) -> float:
    model.train()
    total = 0.0
    n = 0
    lambda_weak = float(cfg["train"]["lambda_weak"])
    lambda_cons = float(cfg["train"]["lambda_consistency"])
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        slices = batch["series_slices"].to(device)
        logits = model(
            slices,
            batch["slice_mask"].to(device),
            batch["plane_ids"].to(device),
            batch["series_mask"].to(device),
            batch["symbolic"].to(device) if cfg["model"]["use_symbolic_features"] else None,
        )
        gold = batch["gold"].to(device)
        gold_mask = batch["gold_mask"].to(device)
        soft = batch["soft"].to(device)
        soft_w = batch["soft_weight"].to(device)
        # Weak mask: where soft weight > 0 and gold missing
        weak_mask = (soft_w > 0).float() * (1.0 - gold_mask)

        def _loss():
            lg = masked_bce_with_logits(logits, gold, gold_mask)
            lw = masked_bce_with_logits(logits, soft, weak_mask, sample_weight=soft_w)
            lc = consistency_loss(
                logits,
                targets,
                preferred_coverage=batch["pref_cov"].to(device)
                if cfg["symbolic"]["use_constraints"]
                else None,
                weight=lambda_cons if cfg["symbolic"]["use_constraints"] else 0.0,
            )
            return lg + lambda_weak * lw + lc

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = _loss()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = _loss()
            loss.backward()
            optimizer.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def predict(model, loader, device, use_symbolic: bool) -> tuple[list[str], np.ndarray]:
    model.eval()
    uids = []
    probs = []
    for batch in loader:
        logits = model(
            batch["series_slices"].to(device),
            batch["slice_mask"].to(device),
            batch["plane_ids"].to(device),
            batch["series_mask"].to(device),
            batch["symbolic"].to(device) if use_symbolic else None,
        )
        p = torch.sigmoid(logits).cpu().numpy()
        probs.append(p)
        uids.extend(batch["study_uid"])
    if not probs:
        return [], np.zeros((0, model.n_targets), dtype=np.float32)
    return uids, np.concatenate(probs, axis=0)


def run_training(cfg: dict[str, Any], quick_sanity: bool = False) -> dict[str, Any]:
    set_seed(int(cfg["seed"]))
    paths = resolve_paths(explicit_root=cfg.get("paths", {}).get("competition_root"))
    sample = load_csv_optional(paths.sample_submission)
    if sample is None:
        raise FileNotFoundError("sample_submission.csv required")
    targets = target_columns_from_sample(sample)
    audit = audit_dataset(paths, targets)
    art = Path(cfg["paths"]["artifacts_dir"])
    save_audit(audit, art / "audit.json")

    train_df = load_csv_optional(paths.train_csv)
    series_df = load_csv_optional(paths.train_series_csv)
    labels_csv = load_csv_optional(paths.train_labels_csv)
    labels = extract_labels(train_df, labels_csv, targets)

    if train_df is None:
        # Sanity path: prevalence submission only
        logger.warning("No train.csv — writing prevalence/constant sanity submission only")
        prev = {t: 0.5 for t in targets}
        sub = prevalence_baseline(sample, prev)
        report = write_submission(
            sub,
            sample,
            Path(cfg["paths"].get("output_submission", "submission.csv")),
            require_nontrivial_variance=False,
        )
        return {"mode": "sanity_only", "submission": report}

    study_ids = train_df[STUDY_ID_COL].astype(str).tolist()
    # Prefer studies that have local DICOM folders (partial mounts / local smoke data).
    series_root_early = paths.train_series_dir
    if series_root_early is not None and series_root_early.exists():
        present = {p.name for p in series_root_early.iterdir() if p.is_dir()}
        filtered = [s for s in study_ids if s in present]
        if filtered:
            logger.info(
                "Filtering to %d/%d studies with on-disk DICOMs under %s",
                len(filtered),
                len(study_ids),
                series_root_early,
            )
            study_ids = filtered
            train_df = train_df[train_df[STUDY_ID_COL].astype(str).isin(study_ids)].reset_index(drop=True)
            if len(labels):
                labels = labels[labels[STUDY_ID_COL].astype(str).isin(study_ids)].reset_index(drop=True)
            if series_df is not None:
                series_df = series_df[series_df[STUDY_ID_COL].astype(str).isin(study_ids)].reset_index(drop=True)
    if quick_sanity:
        study_ids = study_ids[: min(32, len(study_ids))]

    # Groups
    group_col = audit.get("group_column")
    groups = None
    if group_col and train_df is not None and group_col in train_df.columns:
        groups = train_df.set_index(STUDY_ID_COL).loc[study_ids][group_col]
    # Stratify by any-positive if labels exist
    any_pos = None
    if len(labels):
        lab = labels.set_index(STUDY_ID_COL).reindex(study_ids)
        any_pos = (lab[targets].fillna(0) > 0.5).any(axis=1).astype(int)

    folds = make_folds(
        study_ids,
        groups=groups.reset_index(drop=True) if groups is not None else None,
        labels=any_pos.reset_index(drop=True) if any_pos is not None else None,
        n_folds=int(cfg["data"]["n_folds"]),
        seed=int(cfg["seed"]),
    )
    folds_path = art / "folds" / "folds.csv"
    folds_path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(folds_path, index=False)

    prevalence = {}
    for t in targets:
        if t in labels.columns and labels[t].notna().any():
            prevalence[t] = float(labels[t].mean())
        else:
            prevalence[t] = 0.5

    soft_mapping = {
        "positive": float(cfg["weak_labels"]["positive"]),
        "negative": float(cfg["weak_labels"]["negative"]),
        "uncertain": float(cfg["weak_labels"]["uncertain"]),
        "historical": float(cfg["weak_labels"]["historical"]),
    }
    vectors, soft_df, weight_df = build_symbolic_tables(
        study_ids,
        train_df,
        series_df,
        targets,
        soft_mapping,
        prevalence,
        float(cfg["weak_labels"]["unmentioned_shrinkage"]),
        cfg.get("parser_version", "report_parser_v1"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    series_root = paths.train_series_dir or Path(".")
    oof = np.full((len(study_ids), len(targets)), np.nan, dtype=np.float64)
    id_to_idx = {s: i for i, s in enumerate(study_ids)}
    fold_metrics = []

    n_folds = int(folds["fold"].max()) + 1 if len(folds) else 0
    for fold_i in range(n_folds):
        val_ids = folds.loc[folds["fold"] == fold_i, STUDY_ID_COL].astype(str).tolist()
        train_ids = folds.loc[folds["fold"] != fold_i, STUDY_ID_COL].astype(str).tolist()
        if not train_ids or not val_ids:
            continue
        if series_df is None:
            logger.warning("No series metadata — skipping image training fold %d", fold_i)
            continue

        train_ds = StudyDataset(
            train_ids,
            series_df,
            series_root,
            labels,
            soft_df,
            weight_df,
            vectors,
            targets,
            slices_per_series=int(cfg["data"]["slices_per_series"]),
            image_size=int(cfg["data"]["image_size"]),
        )
        val_ds = StudyDataset(
            val_ids,
            series_df,
            series_root,
            labels,
            soft_df,
            weight_df,
            vectors,
            targets,
            slices_per_series=int(cfg["data"]["slices_per_series"]),
            image_size=int(cfg["data"]["image_size"]),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=True,
            num_workers=int(cfg["train"]["num_workers"]),
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=False,
            num_workers=int(cfg["train"]["num_workers"]),
        )

        model = KneeStudyClassifier(
            n_targets=len(targets),
            backbone=cfg["model"]["backbone"],
            pretrained=bool(cfg["model"]["pretrained"]),
            freeze_encoder=bool(cfg["model"]["freeze_encoder"]),
            unfreeze_last_block=bool(cfg["model"]["unfreeze_last_block"]),
            slice_pool=cfg["model"]["slice_pool"],
            use_plane_slots=bool(cfg["model"]["use_plane_slots"]),
            use_availability_mask=bool(cfg["model"]["use_availability_mask"]),
            use_symbolic_features=bool(cfg["model"]["use_symbolic_features"]),
            dropout=float(cfg["model"]["dropout"]),
        ).to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params, lr=float(cfg["train"]["lr_head"]), weight_decay=float(cfg["train"]["weight_decay"])
        )
        scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and cfg["train"]["mixed_precision"]) else None

        best_macro = -1.0
        patience = int(cfg["train"]["early_stopping_patience"])
        stale = 0
        ckpt_path = art / "checkpoints" / f"fold{fold_i}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(int(cfg["train"]["epochs"])):
            tr_loss = train_one_epoch(model, train_loader, optimizer, cfg, targets, device, scaler)
            uids, probs = predict(model, val_loader, device, cfg["model"]["use_symbolic_features"])
            # Evaluate on gold where available
            y_true = np.full((len(uids), len(targets)), np.nan)
            lab_idx = labels.set_index(STUDY_ID_COL) if len(labels) else None
            for j, u in enumerate(uids):
                if lab_idx is not None and u in lab_idx.index:
                    for ti, t in enumerate(targets):
                        if t in lab_idx.columns and pd.notna(lab_idx.loc[u, t]):
                            y_true[j, ti] = float(lab_idx.loc[u, t])
            aucs = per_target_roc_auc(y_true, probs, targets)
            macro = macro_auc_from_dict(aucs)
            logger.info("Fold %d epoch %d loss=%.4f macro_auc=%s", fold_i, epoch, tr_loss, macro)
            score = macro if macro is not None else -1.0
            if score > best_macro:
                best_macro = score
                stale = 0
                torch.save(
                    {"model": model.state_dict(), "targets": targets, "cfg": cfg, "fold": fold_i},
                    ckpt_path,
                )
                for u, p in zip(uids, probs):
                    oof[id_to_idx[u]] = p
            else:
                stale += 1
                if stale >= patience:
                    break

        fold_metrics.append({"fold": fold_i, "best_macro": best_macro})

    # OOF report
    y_true_all = np.full((len(study_ids), len(targets)), np.nan)
    if len(labels):
        lab_idx = labels.set_index(STUDY_ID_COL)
        for i, u in enumerate(study_ids):
            if u in lab_idx.index:
                for ti, t in enumerate(targets):
                    if t in lab_idx.columns and pd.notna(lab_idx.loc[u, t]):
                        y_true_all[i, ti] = float(lab_idx.loc[u, t])
    oof_aucs = per_target_roc_auc(y_true_all, np.nan_to_num(oof, nan=0.5), targets)
    report = {
        "per_target_auc": oof_aucs,
        "macro_oof_auc": macro_auc_from_dict(oof_aucs),
        "fold_metrics": fold_metrics,
        "fingerprint": audit.get("fingerprint"),
        "n_studies": len(study_ids),
    }
    save_oof_report(art / "oof" / "oof_report.json", report)
    np.save(art / "oof" / "oof_probs.npy", oof)
    pd.DataFrame({STUDY_ID_COL: study_ids, **{t: oof[:, i] for i, t in enumerate(targets)}}).to_csv(
        art / "oof" / "oof_predictions.csv", index=False
    )

    # ASRA ledger update for default run (H1-style)
    ledger = HypothesisLedger(Path(cfg["asra"]["ledger_path"]))
    hyp = Hypothesis(
        hypothesis_id="H1_run",
        hypothesis="Weak-supervised plane-aware frozen encoder baseline.",
        mechanism="Gold BCE + weak report BCE + optional consistency; plane slots + availability + symbolic features.",
        config_delta={"config": "configs/submission_001.yaml"},
        expected_targets=["*"],
        falsification="Constant preds, inverted labels, or target collapse.",
    )
    ledger.decide(
        hyp,
        macro_oof=float(report["macro_oof_auc"] or 0.0),
        baseline_macro=0.5,
        per_target_auc=oof_aucs,
        runtime_sec=0.0,
        min_delta=float(cfg["asra"]["accept_min_macro_delta"]),
        collapse_auc=float(cfg["asra"]["target_collapse_auc"]),
        require_no_collapse=bool(cfg["asra"]["require_no_target_collapse"]),
    )

    # If we have a sample submission, write a prevalence fallback submission for pipeline proof
    sub = prevalence_baseline(sample, prevalence)
    # Prefer mean OOF when available for overlapping IDs
    oof_df = pd.DataFrame({STUDY_ID_COL: study_ids, **{t: oof[:, i] for i, t in enumerate(targets)}})
    merged = sample[[STUDY_ID_COL]].merge(oof_df, on=STUDY_ID_COL, how="left")
    for t in targets:
        merged[t] = merged[t].fillna(prevalence.get(t, 0.5))
    try:
        write_submission(
            merged,
            sample,
            Path(cfg["paths"].get("output_submission", art / "submission.csv")),
            require_nontrivial_variance=len(sample) > 1,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("OOF submission failed (%s); writing prevalence baseline", exc)
        write_submission(
            sub,
            sample,
            Path(cfg["paths"].get("output_submission", art / "submission.csv")),
            require_nontrivial_variance=False,
        )

    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/submission_001.yaml")
    ap.add_argument("--quick-sanity", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    report = run_training(cfg, quick_sanity=args.quick_sanity)
    print(json.dumps({k: report[k] for k in report if k != "fingerprint"}, indent=2, default=str))


if __name__ == "__main__":
    main()
