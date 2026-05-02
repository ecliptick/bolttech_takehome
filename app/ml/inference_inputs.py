"""Convert API / UI claim payloads into a single-row DataFrame for the ML pipeline."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from app.schemas import ClaimInput

# Spreadsheet ``other`` (and occasionally other damage flags) may contain free text; coerce non-numeric to None.
_CLAIM_FLOAT_FIELDS = frozenset({
    "excessFee",
    "rrp",
    "balanceRRP",
    "oldBalanceRRP",
    "deviceCost",
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
})


def _clean_series_value(v: Any) -> Any:
    if v is None:
        return None
    try:
        if bool(pd.isna(v)):
            return None
    except (ValueError, TypeError):
        pass
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def _coerce_optional_float(v: Any) -> float | None:
    cv = _clean_series_value(v)
    if cv is None:
        return None
    if isinstance(cv, (int, np.integer)):
        return float(cv)
    if isinstance(cv, float):
        return float(cv)
    if isinstance(cv, str):
        t = cv.strip()
        if not t:
            return None
        try:
            return float(t)
        except ValueError:
            return None
    return None


def dataframe_row_to_claim(df: pd.DataFrame, index: int) -> ClaimInput:
    row = df.iloc[index].to_dict()
    row.pop("status", None)
    cleaned: dict[str, Any] = {}
    for k, v in row.items():
        key = str(k)
        if key in _CLAIM_FLOAT_FIELDS:
            cleaned[key] = _coerce_optional_float(v)
        else:
            cleaned[key] = _clean_series_value(v)
    return ClaimInput.model_validate(cleaned)


def claim_row_dataframe(inp: ClaimInput) -> pd.DataFrame:
    d = inp.model_dump(mode="python", exclude_none=False)
    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            out[k] = np.nan
            continue
        if k in ("policyStartDate", "policyEndDate", "purchaseDate") and isinstance(v, (date, datetime)):
            out[k] = pd.Timestamp(v)
        else:
            out[k] = v
    return pd.DataFrame([out])
