"""Train claim-approval model and persist artifacts with metadata for MLOps-style versioning."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from app.config import get_settings
from app.ml.dataset import load_claims_training_frame
from app.ml.features import build_preprocessor, engineer_features, engineered_column_order


@dataclass
class TrainMetadata:
    trained_at: str
    data_path: str
    n_rows: int
    holdout_fraction: float
    random_state: int
    best_params: dict[str, float | int | str]
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    sklearn_version: str
    git_commit: str | None


def _git_commit() -> str | None:
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except OSError:
        pass
    return None


def _make_base_pipeline(*, random_state: int) -> Pipeline:
    preprocessor = build_preprocessor()
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    random_state=random_state,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _fit_pipeline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    random_state: int,
    tune: bool,
) -> tuple[Pipeline, dict[str, float | int | str]]:
    param_dist = {
        "clf__n_estimators": [200, 400, 600],
        "clf__max_depth": [None, 8, 12, 16, 22],
        "clf__min_samples_leaf": [1, 2, 4, 8],
        "clf__max_features": ["sqrt", "log2", None],
    }

    base = _make_base_pipeline(random_state=random_state)
    Xt_train = engineer_features(X_train)[engineered_column_order()]

    if tune:
        search = RandomizedSearchCV(
            base,
            param_distributions=param_dist,
            n_iter=18,
            scoring="roc_auc",
            random_state=random_state,
            cv=4,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(Xt_train, y_train)
        model = search.best_estimator_
        best_params = {
            str(k): (v.item() if isinstance(v, np.generic) else v) for k, v in search.best_params_.items()
        }
    else:
        base.fit(Xt_train, y_train)
        model = base
        best_params = {"note": "tune_off_defaults"}

    return model, best_params


def _metrics_for_model(model: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray) -> dict[str, float]:
    Xt_test = engineer_features(X_test)[engineered_column_order()]
    proba = model.predict_proba(Xt_test)[:, 1]
    pred = model.predict(Xt_test)
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, average="binary")),
    }


def _baseline_metrics(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict[str, float]:
    Xt_train = engineer_features(X_train)[engineered_column_order()]
    Xt_test = engineer_features(X_test)[engineered_column_order()]
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(Xt_train, y_train)
    dummy_proba = dummy.predict_proba(Xt_test)[:, 1]
    dummy_pred = dummy.predict(Xt_test)
    return {
        "roc_auc": float(roc_auc_score(y_test, dummy_proba)),
        "accuracy": float(accuracy_score(y_test, dummy_pred)),
        "f1": float(f1_score(y_test, dummy_pred, average="binary")),
    }


def holdout_eval_metrics(
    X_df: pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    random_state: int = 42,
    test_size: float = 0.2,
    tune: bool = False,
) -> tuple[dict[str, float], dict[str, float], int]:
    """
    Train on a stratified train split, score holdout. Does not read/write artifacts
    and does not retain the fitted estimator (for Streamlit / MLOps demos).
    """
    y_arr = np.asarray(y).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X_df,
        y_arr,
        test_size=test_size,
        random_state=random_state,
        stratify=y_arr,
    )
    model, _best = _fit_pipeline(X_train, y_train, random_state=random_state, tune=tune)
    metrics = _metrics_for_model(model, X_test, y_test)
    baseline = _baseline_metrics(X_train, y_train, X_test, y_test)
    return metrics, baseline, int(len(X_df))


def train(
    *,
    data_path: Path | None = None,
    out_dir: Path | None = None,
    random_state: int = 42,
    test_size: float = 0.2,
    tune: bool = True,
) -> Path:
    import sklearn

    settings = get_settings()
    data_path = data_path or settings.claim_data_xlsx
    out_dir = out_dir or settings.artifacts_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    X_df, y_series = load_claims_training_frame(data_path)
    y = y_series.astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X_df,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model, best_params = _fit_pipeline(X_train, y_train, random_state=random_state, tune=tune)

    metrics = _metrics_for_model(model, X_test, y_test)
    baseline_metrics = _baseline_metrics(X_train, y_train, X_test, y_test)

    print("--- Model Performance ---")
    Xt_test = engineer_features(X_test)[engineered_column_order()]
    pred_print = model.predict(Xt_test)
    print(classification_report(y_test, pred_print, digits=4))
    print("Model Holdout:", metrics)

    dummy_pred = DummyClassifier(strategy="most_frequent").fit(
        engineer_features(X_train)[engineered_column_order()],
        y_train,
    ).predict(Xt_test)
    print("\n--- Baseline (No-Skill / Blanket Approve) Performance ---")
    print(classification_report(y_test, dummy_pred, digits=4, zero_division=0))
    print("Baseline Holdout:", baseline_metrics)

    meta = TrainMetadata(
        trained_at=datetime.now(tz=UTC).isoformat(),
        data_path=str(data_path.resolve()),
        n_rows=len(X_df),
        holdout_fraction=test_size,
        random_state=random_state,
        best_params=best_params,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        sklearn_version=sklearn.__version__,
        git_commit=_git_commit(),
    )

    bundle = {"pipeline": model, "metadata": asdict(meta)}
    model_path = out_dir / "approval_model.joblib"
    meta_path = out_dir / "approval_model_meta.json"

    joblib.dump(bundle, model_path)
    meta_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
    print(f"Saved model to {model_path}")
    return model_path


def main() -> None:
    p = argparse.ArgumentParser(description="Train claim approval model")
    p.add_argument("--data", type=Path, default=None, dest="data_path", help="claim_use_case_dataset.xlsx")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--no-tune", action="store_true", help="Skip hyperparameter search (faster)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(data_path=args.data_path, out_dir=args.out, random_state=args.seed, tune=not args.no_tune)
    sys.exit(0)


if __name__ == "__main__":
    main()
