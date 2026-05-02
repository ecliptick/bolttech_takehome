"""Normalize free-text fields: mojibake repair, unicode NFKC, whitespace.

Excel / exports often contain **UTF-8 misread as Latin-1** (e.g. ``vÃ¤`` → ``vä``).

Deterministic only (no LLM) so training and inference match.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd

_WS = re.compile(r"\s+", re.UNICODE)


def fix_mojibake_utf8_latin1(s: str) -> str:
    """If bytes were UTF-8 but decoded as Latin-1/cp1252, reverse the mistake."""
    if not s:
        return s
    try:
        return s.encode("latin-1", errors="strict").decode("utf-8", errors="strict")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return s


def normalize_free_text(raw: Any) -> str | None:
    """Strip, normalize unicode, squash whitespace, repair common encoding glitches."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    t = str(raw).strip()
    if not t or t.lower() in {"nan", "none", "<na>"}:
        return None

    for _ in range(2):
        prev = t
        t = fix_mojibake_utf8_latin1(t)
        if t == prev:
            break

    try:
        import ftfy  # type: ignore[import-not-found]

        t = str(ftfy.fix_text(t))
    except ImportError:
        pass

    t = unicodedata.normalize("NFKC", t)
    t = _WS.sub(" ", t).strip()
    return t or None


def normalize_free_text_series(ser: pd.Series) -> pd.Series:
    return ser.map(lambda x: normalize_free_text(x) if pd.notna(x) else pd.NA).astype(object)
