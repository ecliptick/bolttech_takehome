from __future__ import annotations

from functools import lru_cache

from app.config import get_settings


@lru_cache(maxsize=32)
def load_prompt(relative: str) -> str:
    root = get_settings().project_root
    path = root / relative
    return path.read_text(encoding="utf-8")
