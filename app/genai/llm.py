"""Gemini calls wired to `app.config` (API key, model, request spacing)."""

from __future__ import annotations

from app.config import get_settings
from app.genai.gemini import generate_content as _gemini_generate_content


def generate_content(
    user_text: str,
    *,
    system_text: str = "You are a precise assistant. Follow the user's instructions exactly.",
    temperature: float = 0.35,
    max_output_tokens: int = 2048,
    timeout_s: float = 120.0,
    max_retries: int = 5,
    retry_base_s: float = 2.0,
) -> str:
    """Invoke Gemini using env-backed settings (`GEMINI_API_KEY`, `GEMINI_MODEL`, …).

    For full control (e.g. batch notebooks), import ``generate_content`` from
    :mod:`app.genai.gemini` and pass ``api_key`` / ``model`` explicitly.
    """
    settings = get_settings()
    api_key = settings.gemini_api_key
    if not api_key:
        msg = "GEMINI_API_KEY is not set. Add it to .env or the environment."
        raise ValueError(msg)
    return _gemini_generate_content(
        api_key=api_key,
        model=settings.gemini_model,
        system_text=system_text,
        user_text=user_text,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        timeout_s=timeout_s,
        max_retries=max_retries,
        retry_base_s=retry_base_s,
    )
