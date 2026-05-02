"""Load trained pipeline and score claims."""

from __future__ import annotations

import json
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


def load_model_bundle() -> dict[str, Any]:
    """Load joblib bundle once per process (same object reused for predict + importance)."""
    global _bundle
    if _bundle is not None:
        return _bundle
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
