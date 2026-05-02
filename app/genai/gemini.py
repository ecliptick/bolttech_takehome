"""Google AI Studio / Gemini REST (`generateContent`) — no SDK required."""

from __future__ import annotations

import random
import threading
import time
from typing import Any

import httpx

# Transient server / capacity / rate-limit responses — safe to retry with backoff.
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

_throttle_lock = threading.Lock()
# Updated when a generateContent call fully finishes (success or terminal failure).
# Spacing uses this so RPM limits apply between completed requests, not overlapping long responses.
_last_request_finish_monotonic: float = 0.0


def _request_min_interval_s() -> float | None:
    try:
        from app.config import get_settings

        return get_settings().gemini_request_min_interval_s()
    except Exception:
        return None


def generate_content(
    *,
    api_key: str,
    model: str,
    system_text: str,
    user_text: str,
    temperature: float = 0.35,
    max_output_tokens: int = 2048,
    timeout_s: float = 120.0,
    max_retries: int = 5,
    retry_base_s: float = 2.0,
) -> str:
    """POST v1beta `models/{model}:generateContent` with `x-goog-api-key` header.

    Retries on transient HTTP status (429, 5xx, 408) with exponential backoff plus jitter.
    Default model id examples: ``gemma-4-31b-it``, ``gemini-2.5-flash``, etc.
    Persistent 503s often mean regional/load issues; try another model id or wait and rerun.
    404 errors usually indicate an invalid model name.

    When ``GEMINI_RPM`` or ``GEMINI_MIN_REQUEST_INTERVAL_S`` is set in the environment
    (via ``app.config.Settings``), spacing is enforced between **completed** calls: the
    lock is held for the full HTTP/retry cycle so a slow response still counts before the
    next request starts (closer to respecting provider RPM on small quotas).
    """
    global _last_request_finish_monotonic
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body: dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system_text}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_text}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    interval_s = _request_min_interval_s()
    with _throttle_lock:
        if interval_s is not None and interval_s > 0:
            now = time.monotonic()
            wait_s = _last_request_finish_monotonic + interval_s - now
            if wait_s > 0:
                time.sleep(wait_s)

        try:
            data: dict[str, Any] | None = None
            for attempt in range(max_retries + 1):
                with httpx.Client(timeout=timeout_s) as client:
                    resp = client.post(url, json=body, headers=headers)
                if (
                    resp.status_code in _RETRYABLE_STATUS
                    and attempt < max_retries
                ):
                    delay = retry_base_s * (2**attempt) + random.uniform(0.0, 0.75)
                    time.sleep(delay)
                    continue
                if resp.status_code == 404:
                    raise httpx.HTTPStatusError(
                        f"Model '{model}' not found (404). Check your GEMINI_MODEL name.",
                        request=resp.request,
                        response=resp,
                    )
                resp.raise_for_status()
                data = resp.json()
                break

            assert data is not None
            cands = data.get("candidates") or []
            if not cands:
                raise ValueError(f"No candidates in Gemini response: {data!r}")
            parts = (cands[0].get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
            if not text.strip():
                reason = cands[0].get("finishReason")
                raise ValueError(f"Empty Gemini text (finishReason={reason}): {data!r}")
            return text.strip()
        finally:
            _last_request_finish_monotonic = time.monotonic()
