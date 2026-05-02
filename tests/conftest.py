"""Isolate API tests from live Gemini/OpenAI quotas (offline deterministic stubs)."""

from __future__ import annotations

import pytest

from app.config import get_settings


@pytest.fixture(autouse=True)
def _clear_llm_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "GEMINI_API_KEY",
        "GOOGLE_AI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.setenv(key, "")
    get_settings.cache_clear()
