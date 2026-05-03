"""Load trained pipeline and score claims."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.config import get_settings
from app.ml.features import engineer_features, engineered_column_order


def _artifacts_dir() -> Path:
    return get_settings().artifacts_dir


_bundle: dict[str, Any] | None = None


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an s3 URI: {uri!r}")
    rest = uri.removeprefix("s3://")
    if "/" not in rest:
        raise ValueError(f"Missing object key in s3 URI: {uri!r}")
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid s3 URI: {uri!r}")
    return bucket, key


def _download_s3_uri_to_file(uri: str, dest: Path) -> None:
    try:
        import boto3
    except ImportError as e:
        raise RuntimeError(
            "ACTIVE_MODEL_*_S3_URI is set but boto3 is not installed. "
            "Install boto3 or unset those environment variables."
        ) from e
    bucket, key = _parse_s3_uri(uri)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    boto3.client("s3").download_file(bucket, key, str(tmp))
    tmp.replace(dest)


def _maybe_fetch_artifacts_from_s3() -> None:
    """When ACTIVE_MODEL_S3_URI / ACTIVE_MODEL_META_S3_URI are set (e.g. ECS + Terraform), sync from S3 first."""
    art = _artifacts_dir()
    joblib_uri = os.environ.get("ACTIVE_MODEL_S3_URI", "").strip()
    if joblib_uri.startswith("s3://"):
        _download_s3_uri_to_file(joblib_uri, art / "approval_model.joblib")
    meta_uri = os.environ.get("ACTIVE_MODEL_META_S3_URI", "").strip()
    if meta_uri.startswith("s3://"):
        _download_s3_uri_to_file(meta_uri, art / "approval_model_meta.json")


def load_model_bundle() -> dict[str, Any]:
    """Load joblib bundle once per process (same object reused for predict + importance)."""
    global _bundle
    if _bundle is not None:
        return _bundle
    _maybe_fetch_artifacts_from_s3()
    path = _artifacts_dir() / "approval_model.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Train first: python -m app.ml.train "
            f"(artifacts dir: {_artifacts_dir()})"
        )
    _bundle = joblib.load(path)
    return _bundle


def model_identity_for_logs() -> dict[str, Any]:
    """Subset of bundle metadata for correlating prediction logs to the loaded artifact."""
    try:
        bundle = load_model_bundle()
    except FileNotFoundError:
        return {}
    meta = bundle.get("metadata")
    if not isinstance(meta, dict):
        return {}
    out: dict[str, Any] = {}
    if meta.get("trained_at") is not None:
        out["model_trained_at"] = meta["trained_at"]
    if meta.get("git_commit") is not None:
        out["model_git_commit"] = meta["git_commit"]
    if meta.get("sklearn_version") is not None:
        out["model_sklearn_version"] = meta["sklearn_version"]
    if meta.get("n_rows") is not None:
        out["model_train_n_rows"] = meta["n_rows"]
    if meta.get("random_state") is not None:
        out["model_train_random_state"] = meta["random_state"]
    joblib_path = (_artifacts_dir() / "approval_model.joblib").resolve()
    out["model_artifact_path"] = str(joblib_path)
    return out


def load_metadata() -> dict[str, Any]:
    _maybe_fetch_artifacts_from_s3()
    meta_path = _artifacts_dir() / "approval_model_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def predict_batch(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Predict labels and probability of approval (class ``1``)."""
    bundle = load_model_bundle()
    pipeline = bundle["pipeline"]
    clf = pipeline.named_steps["clf"]

    eng = engineer_features(rows)[engineered_column_order()]
    pred = pipeline.predict(eng)

    classes = clf.classes_
    if 1 in classes:
        pos_ix = int(np.where(classes == 1)[0][0])
    else:
        pos_ix = int(np.argmax(classes))

    proba = pipeline.predict_proba(eng)[:, pos_ix]
    return pred.astype(np.int32), proba.astype(np.float64)


def top_feature_names(n: int = 8) -> list[tuple[str, float]]:
    """Return coarse feature-importance labels for explanations (trees only)."""
    bundle = load_model_bundle()
    pipeline = bundle["pipeline"]
    clf = pipeline.named_steps["clf"]
    preprocessor = pipeline.named_steps["preprocess"]
    names = preprocessor.get_feature_names_out()
    imps = getattr(clf, "feature_importances_", None)
    if imps is None:
        return []
    order = np.argsort(imps)[::-1][:n]
    return [(str(names[i]), float(imps[i])) for i in order]
