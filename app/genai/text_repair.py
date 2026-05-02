"""Optional Gemini pass: garbled UTF-8 / mixed-language text → readable English JSON.

Separate from the approval model: call when you need human-readable English prose.
Deterministic baseline lives in ``app.ml.text_clean``.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Literal

from app.config import Settings, get_settings
from app.genai.gemini import generate_content as gemini_generate_content
from app.genai.prompt_loader import load_prompt
from app.ml.text_clean import normalize_free_text


def deterministic_clean_copy(
    issue_desc: str | None = None,
    product_name: str | None = None,
    product_desc: str | None = None,
) -> dict[str, str | None]:
    """Encoding + unicode cleanup only — no translation."""
    return {
        "issueDesc": normalize_free_text(issue_desc),
        "productName": normalize_free_text(product_name),
        "productDesc": normalize_free_text(product_desc),
    }


def _extract_json(raw: str) -> dict[str, Any]:
    raw_st = raw.strip()
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_st)
    blob = fenced[0].strip() if fenced else raw_st
    data = json.loads(blob)
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object from repair model")
    return data


def _merge_str(over: Any, fallback: str | None) -> str | None:
    if over is None:
        return fallback
    if not isinstance(over, str):
        return fallback
    s = over.strip()
    return s if s else fallback


async def repair_text_to_english(
    *,
    issue_desc: str | None = None,
    product_name: str | None = None,
    product_desc: str | None = None,
    settings: Settings | None = None,
) -> tuple[dict[str, str | None], str, Literal["gemini", "deterministic_only"]]:
    """Return repaired ``{issueDesc, productName, productDesc}``, rationale, backend label."""
    settings = settings or get_settings()
    base = deterministic_clean_copy(issue_desc, product_name, product_desc)

    if not settings.gemini_api_key:
        rationale = (
            "GEMINI_API_KEY unset — deterministic `text_clean` / `ftfy` only (no English rewrite)."
        )
        return base, rationale, "deterministic_only"

    system = load_prompt("prompts/text_repair.txt")
    user_obj = {
        "issueDesc": base["issueDesc"],
        "productName": base["productName"],
        "productDesc": base["productDesc"],
    }

    def _run() -> str:
        return gemini_generate_content(
            api_key=settings.gemini_api_key or "",
            model=settings.gemini_model,
            system_text=system.strip(),
            user_text=json.dumps(user_obj, ensure_ascii=False, indent=2),
            temperature=0.2,
            max_output_tokens=4096,
        )

    raw = await asyncio.to_thread(_run)

    try:
        out = _extract_json(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        rationale = f"Gemini JSON parse failed ({exc!s}); left deterministic output. Snippet: {raw[:400]!r}"
        return base, rationale, "deterministic_only"

    merged: dict[str, str | None] = {
        "issueDesc": _merge_str(out.get("issueDesc"), base["issueDesc"]),
        "productName": _merge_str(out.get("productName"), base["productName"]),
        "productDesc": _merge_str(out.get("productDesc"), base["productDesc"]),
    }

    rationale = "Gemini JSON repair layered on deterministic cleaning."
    return merged, rationale, "gemini"


def repair_text_to_english_sync(
    *,
    issue_desc: str | None = None,
    product_name: str | None = None,
    product_desc: str | None = None,
    settings: Settings | None = None,
) -> tuple[dict[str, str | None], str, Literal["gemini", "deterministic_only"]]:
    return asyncio.run(
        repair_text_to_english(
            issue_desc=issue_desc,
            product_name=product_name,
            product_desc=product_desc,
            settings=settings,
        )
    )


