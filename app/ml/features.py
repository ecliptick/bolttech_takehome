"""Feature extraction and preprocessing for claim_use_case_dataset.xlsx schema."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.persona_labels import normalize_persona_label

_DAMAGE_FLOAT = [
    "turnOnOff",
    "touchScreen",
    "smashed",
    "frontCamera",
    "backCamera",
    "frontOrBackCamera",
    "audio",
    "mic",
    "buttons",
    "connection",
    "charging",
    "other",
]

_DROP_RAW = {"make", "relationship"}

_ISSUE_EN_COLS = ("issueDesc_en", "issue_desc_en", "issueDesc")


def _coverage_duration_tier_from_days(day_count: object) -> object:
    """Same SKU-style bands as ``01_load_and_explore_data.ipynb`` (policy span length in days)."""
    if day_count is None:
        return np.nan
    try:
        d = float(day_count)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(d):
        return np.nan
    if d < 120:
        return "outlier_lt120d"
    if d <= 275:
        return "tier_appx_6mo"
    if d <= 550:
        return "tier_appx_12mo"
    return "tier_appx_24mo"


def _simple_str_cell(x):  # noqa: ANN001
    """Strip placeholders; no mojibake/LLM repair (English narrative uses translated column when present)."""
    if pd.isna(x):
        return None
    t = str(x).strip()
    if not t or t.lower() in {"", "nan", "none", "<na>"}:
        return None
    return t


def _to_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(df).copy()
    # Duplicate axis labels → df["col"] can return a DataFrame and break downstream Series ops / assignments.
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()].copy()

    ps = _to_dt(out["policyStartDate"])
    pe = _to_dt(out["policyEndDate"])
    pu = _to_dt(out["purchaseDate"])
    out["policy_length_days"] = (pe - ps).dt.days.astype(float)
    out["purchase_to_end_days"] = (pe - pu).dt.days.astype(float)
    issue_src = None
    for cand in _ISSUE_EN_COLS:
        if cand in out.columns:
            issue_src = out[cand]
            break
    if issue_src is not None:
        desc_clean = issue_src.map(_simple_str_cell)
        out["issue_desc_len"] = desc_clean.map(
            lambda t: float(len(t)) if isinstance(t, str) else 0.0,
        )
    else:
        out["issue_desc_len"] = 0.0

    if "productName" in out.columns:
        pn_clean = out["productName"].map(_simple_str_cell).map(lambda t: pd.NA if t is None else t)
        out["product_name_len"] = pn_clean.map(lambda t: float(len(t)) if isinstance(t, str) else 0.0)
        out["productName"] = pn_clean
    else:
        out["product_name_len"] = 0.0
    if "productDesc" in out.columns:
        pdesc_clean = out["productDesc"].map(_simple_str_cell)
        out["product_desc_len"] = pdesc_clean.map(
            lambda t: float(len(t)) if isinstance(t, str) else 0.0,
        )
    else:
        out["product_desc_len"] = 0.0

    for col in ["excessFee", "rrp", "balanceRRP", "oldBalanceRRP", "deviceCost"]:
        base = pd.to_numeric(out[col], errors="coerce").astype(float).clip(lower=0)
        out[col] = base
        out[f"{col}_log1p"] = np.log1p(base.clip(lower=0))

    rr = pd.to_numeric(out["rrp"], errors="coerce").astype(float)
    bal = pd.to_numeric(out["balanceRRP"], errors="coerce").astype(float)
    xf = pd.to_numeric(out["excessFee"], errors="coerce").astype(float)
    out["rrp_minus_balance"] = rr - bal
    with np.errstate(divide="ignore", invalid="ignore"):
        out["fee_to_rrp"] = np.where(rr > 0, xf / rr, np.nan).astype(float)

    # Length features only for long narrative; keep `productName` for categorical OHE (was wrongly dropped before).
    drop_text = []
    for narrative_col in ("issueDesc", "issueDesc_en", "issue_desc_en", "productDesc"):
        if narrative_col in out.columns:
            drop_text.append(narrative_col)
    for d in drop_text:
        out = out.drop(columns=[d])
    out = out.drop(columns=[c for c in ["policyStartDate", "policyEndDate", "purchaseDate"] if c in out.columns])

    for dc in _DAMAGE_FLOAT:
        if dc in out.columns:
            out[dc] = pd.to_numeric(out[dc], errors="coerce").astype(float)
        else:
            out[dc] = np.nan

    out["symptom_count"] = out[_DAMAGE_FLOAT].fillna(0).sum(axis=1)

    if "persona" not in out.columns:
        cache_path = Path("notebooks/persona_cache.json")
        p_cache = {}
        if cache_path.exists():
            try:
                p_cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass
                
        def _assign_persona(row):
            txt = str(
                row.get("issueDesc_en")
                or row.get("issue_desc_en")
                or row.get("issueDesc", ""),
            )
            if txt and txt in p_cache:
                return normalize_persona_label(p_cache[txt])
                
            text_lower = txt.lower()
            ctype = str(row.get("claimType", "")).lower()
            if "theft" in ctype or re.search(r"\b(stole|stolen|robbed|thief|pickpocket)\b", text_lower):
                return "The Victim (Theft/Loss)"
            elif "liquid" in ctype or re.search(r"\b(water|spilled|liquid|toilet|rain|pool|sink|wet)\b", text_lower):
                return "The Unlucky Spiller (Liquid Damage)"
            elif re.search(r"\b(gym|run|running|bike|cycle|cycling|workout|tennis|padel|sport)\b", text_lower):
                return "The Active/Sporty (Action Damage)"
            elif re.search(r"\b(kid|child|baby|son|daughter|toddler|dog|cat|pet)\b", text_lower):
                return "The Family/Pet Owner (Chaos Damage)"
            elif re.search(r"\b(car|driving|truck|bus|train|commute|subway|tram)\b", text_lower):
                return "The Commuter (Transit Damage)"
            elif re.search(r"\b(work|office|desk|coworker|colleague|colleagues)\b", text_lower):
                return "The Professional (Workplace Accident)"
            else:
                return "The Clumsy Dropper (Standard Accidental)"
                
        out["persona"] = out.apply(_assign_persona, axis=1)

    if "persona" in out.columns:
        def _persona_cell(v):  # noqa: ANN001
            return v if pd.isna(v) else normalize_persona_label(v)

        out["persona"] = out["persona"].map(_persona_cell)

    drop_extra = _DROP_RAW.intersection(set(out.columns))
    if drop_extra:
        out = out.drop(columns=list(drop_extra))

    # Policy SKU bands (Notebook 01) — assign once after spans are finalized / columns dropped so it cannot be lost.
    if "policy_length_days" in out.columns:
        out["coverage_duration_tier"] = out["policy_length_days"].map(_coverage_duration_tier_from_days)
    else:
        out["coverage_duration_tier"] = np.nan

    def _clean_cat(s: pd.Series) -> pd.Series:
        def row(x):
            if pd.isna(x):
                return np.nan
            t = str(x).strip()
            if t.lower() in {"", "nan", "nat", "none"}:
                return np.nan
            return t

        return s.map(row)

    for col in categorical_raw_names():
        if col not in out.columns:
            out[col] = np.nan
        elif isinstance(out[col], pd.DataFrame):
            out[col] = _clean_cat(out[col].iloc[:, 0])
        else:
            out[col] = _clean_cat(out[col])

    for name in engineered_column_order():
        if name not in out.columns:
            out[name] = np.nan

    return out


def categorical_raw_names() -> list[str]:
    return [
        "coverage",
        "coverage_duration_tier",
        "policyStatus",
        "claimType",
        "channel",
        "country",
        "retailerName",
        "deviceType",
        "productCode",
        "model",
        "productName",
        "persona",
    ]


def numeric_engineered_names() -> list[str]:
    base_nums = ["excessFee", "rrp", "balanceRRP", "oldBalanceRRP", "deviceCost"]
    logs = [f"{b}_log1p" for b in base_nums]
    derived_money = ["rrp_minus_balance", "fee_to_rrp"]
    extra = [
        "policy_length_days",
        "purchase_to_end_days",
        "issue_desc_len",
        "product_name_len",
        "product_desc_len",
        "symptom_count",
    ]
    return base_nums + logs + derived_money + extra + _DAMAGE_FLOAT


def engineered_column_order() -> list[str]:
    return numeric_engineered_names() + categorical_raw_names()


def symptom_flag_columns() -> tuple[str, ...]:
    """Binary symptom/damage fields: passed as numeric 0/1 to the scaler and summed into ``symptom_count``."""
    return tuple(_DAMAGE_FLOAT)


def build_preprocessor() -> ColumnTransformer:
    nums = numeric_engineered_names()
    cats = categorical_raw_names()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            (
                "oh",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    max_categories=50,
                    min_frequency=None,
                    sparse_output=False,
                ),
            ),
        ],
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, nums),
            ("cat", categorical_pipe, cats),
        ],
        remainder="drop",
    )


def prepare_matrix(df: pd.DataFrame, preprocessor: ColumnTransformer, *, fit: bool) -> Any:
    feats = engineer_features(df)
    cols = engineered_column_order()
    X = feats[cols]
    if fit:
        return preprocessor.fit_transform(X)
    return preprocessor.transform(X)
