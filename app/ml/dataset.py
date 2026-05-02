"""Load training data from Excel (claim_use_case_dataset.xlsx) or CSV exports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Columns produced by notebooks / enrichment that are not part of the raw claim schema.
ENRICHED_AUX_COLUMNS: frozenset[str] = frozenset(
    {
        "_pkey",
        "approved",
        "rrp_minus_balance",
        "fee_to_rrp",
        "coverage_duration_days",
        "purchase_to_policy_end_days",
        "days_after_policy_anchor_to_end",
        "coverage_duration_tier",
        "symptom_count",
        "issueDesc_en",
        "issue_desc_en",
        "_wc_issue_en",
        "persona",
    }
)


def _strip_enriched_extras(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in df.columns if c in ENRICHED_AUX_COLUMNS]
    if drop:
        df = df.drop(columns=drop, errors="ignore")
    return df


def _validate_status_and_split_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "status" not in df.columns:
        raise ValueError("Expected column 'status' (e.g. Completed / Declined)")

    s = df["status"].astype(str).str.strip().str.lower()
    if not s.isin({"completed", "declined"}).all():
        unseen = sorted(set(s.unique()) - {"completed", "declined"})
        raise ValueError(f"Unhandled status labels: {unseen}")

    y = pd.Series((s == "completed").astype(int), index=df.index)
    X = df.drop(columns=["status"])
    return X, y


def load_claims_training_frame(path: Path | None, *, sheet: str | int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Read workbook or CSV; target ``approved`` = 1 when ``status`` is Completed (vs Declined)."""
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset not found: {resolved}")

    suffix = resolved.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(resolved)
        df = _strip_enriched_extras(df)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(resolved, sheet_name=sheet, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported dataset format: {suffix} (use .csv, .xlsx, .xls)")

    return _validate_status_and_split_y(df)


def load_claims_xy_from_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Same label logic as ``load_claims_training_frame``, for in-memory frames."""
    work = _strip_enriched_extras(df.copy())
    return _validate_status_and_split_y(work)
