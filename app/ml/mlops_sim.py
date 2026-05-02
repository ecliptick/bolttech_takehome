"""In-memory MLOps-style demos: eval drift logs as synthetic DB rows, retrain comparison, S3 wording."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import get_settings
from app.ml.dataset import load_claims_training_frame, load_claims_xy_from_dataframe
from app.ml.train import holdout_eval_metrics

EVAL_DRIFT_FEATURE_PREFIX = "feature_"


def raw_claims_from_eval_drift_csv(path: Path | str) -> pd.DataFrame:
    """
    Build a claim-level frame from eval logs: only ``feature_*`` columns (raw inputs),
    plus ``status`` derived from ``ground_truth_approved`` (1 → Completed).
    """
    resolved = Path(path).resolve()
    df = pd.read_csv(resolved)
    feat_cols = [c for c in df.columns if c.startswith(EVAL_DRIFT_FEATURE_PREFIX)]
    if not feat_cols:
        raise ValueError(f"No {EVAL_DRIFT_FEATURE_PREFIX!r} columns in {resolved}")
    if "ground_truth_approved" not in df.columns:
        raise ValueError("Expected column ground_truth_approved in eval drift log")

    rename = {c: c[len(EVAL_DRIFT_FEATURE_PREFIX) :] for c in feat_cols}
    raw = df[feat_cols].rename(columns=rename).copy()

    def _to_status(v: object) -> str:
        try:
            return "Completed" if int(float(v)) == 1 else "Declined"
        except (TypeError, ValueError):
            return "Declined"

    raw["status"] = df["ground_truth_approved"].map(_to_status)
    return raw


def load_eval_drift_xy(path: Path | str) -> tuple[pd.DataFrame, pd.Series]:
    raw = raw_claims_from_eval_drift_csv(path)
    return load_claims_xy_from_dataframe(raw)


def compare_retrain_holdout(
    *,
    base_data_path: Path | str,
    drift_csv_path: Path | str,
    random_state: int = 42,
    test_size: float = 0.2,
    tune: bool = False,
) -> dict[str, Any]:
    """
    Train/eval twice (holdout metrics only): base corpus, then base + drift rows.
    No model or dataframe is persisted.
    """
    base_path = Path(base_data_path).resolve()
    drift_path = Path(drift_csv_path).resolve()

    X_base, y_base = load_claims_training_frame(base_path)
    m_before, bl_before, n_base = holdout_eval_metrics(
        X_base,
        y_base,
        random_state=random_state,
        test_size=test_size,
        tune=tune,
    )

    X_drift, y_drift = load_eval_drift_xy(drift_path)
    X_all = pd.concat([X_base, X_drift], ignore_index=True)
    y_all = pd.concat([y_base, y_drift], ignore_index=True)
    m_after, bl_after, n_all = holdout_eval_metrics(
        X_all,
        y_all,
        random_state=random_state,
        test_size=test_size,
        tune=tune,
    )

    return {
        "n_rows_base": n_base,
        "n_rows_drift": int(len(X_drift)),
        "n_rows_combined": n_all,
        "metrics_before": m_before,
        "metrics_after": m_after,
        "baseline_before": bl_before,
        "baseline_after": bl_after,
        "data_before_tooltip": (
            f"Holdout metrics from stratified split (seed={random_state}) on rows read from "
            f"{base_path.name} only (CSV/Excel raw schema; enrichment columns stripped)."
        ),
        "data_after_tooltip": (
            f"Same procedure after concatenating all rows from {base_path.name} with "
            f"{len(X_drift)} raw claim rows extracted from {drift_path.name} "
            "(feature_* columns only; labels from ground_truth_approved)."
        ),
    }


def simulated_export_payload() -> dict[str, Any]:
    """
    Describe bytes that would be uploaded to S3 in a real deployment (demo only—no network I/O).
    """
    settings = get_settings()
    out: dict[str, Any] = {"uploads": [], "bucket_hint": "s3://claim-agent-demo-artifacts/"}

    model_path = settings.artifacts_dir / "approval_model.joblib"
    meta_path = settings.artifacts_dir / "approval_model_meta.json"

    if model_path.exists():
        out["uploads"].append(
            {
                "key": "models/approval_model.joblib",
                "bytes": model_path.stat().st_size,
                "note": "Simulated PUT — bytes read from local artifacts dir (demo does not call AWS).",
            }
        )
    if meta_path.exists():
        out["uploads"].append(
            {
                "key": "logs/approval_model_meta.json",
                "bytes": meta_path.stat().st_size,
                "note": "Simulated PUT — training metadata JSON.",
            }
        )
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            out["meta_preview"] = {k: meta[k] for k in ("trained_at", "n_rows", "metrics") if k in meta}
        except (OSError, json.JSONDecodeError):
            pass

    return out
