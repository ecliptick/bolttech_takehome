from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from openai import OpenAI

from app.config import Settings, get_settings
from app.genai.gemini import generate_content as gemini_generate_content
from app.genai.prompt_loader import load_prompt
from app.schemas import ClaimInput, Persona

_EXPLAIN_PERSONA_PREFIX_RELPATH: dict[Persona, str] = {
    Persona.customer: "prompts/explain_customer_prefix.txt",
    Persona.claims_adjuster: "prompts/explain_adjuster_prefix.txt",
    Persona.high_value_insured: "prompts/explain_high_value_insured_prefix.txt",
    Persona.low_value_insured: "prompts/explain_low_value_insured_prefix.txt",
    Persona.small_issue: "prompts/explain_small_issue_prefix.txt",
    Persona.end_of_duration: "prompts/explain_end_of_duration_prefix.txt",
    Persona.theft_claimant: "prompts/explain_theft_claimant_prefix.txt",
    Persona.repeat_claimant: "prompts/explain_repeat_claimant_prefix.txt",
}


def persona_explain_style_prefix(persona: Persona) -> str:
    """Persona-specific style prefix text for explanation prompts (same files as ``explain_personas``)."""
    rel = _EXPLAIN_PERSONA_PREFIX_RELPATH[persona]
    return load_prompt(rel).strip()


def _openai_client(settings: Settings) -> OpenAI | None:
    key = settings.openai_api_key
    if not key:
        return None
    return OpenAI(api_key=key, base_url=settings.openai_base_url)


async def explain_personas(
    *,
    claim: ClaimInput,
    personas: list[Persona],
    model_payload: dict[str, Any],
    top_features: list[tuple[str, float]],
    settings: Settings | None = None,
) -> dict[str, str]:
    settings = settings or get_settings()
    use_gemini = bool(settings.gemini_api_key)
    oai = None if use_gemini else _openai_client(settings)

    system_base = load_prompt("prompts/explain_system.txt")
    base_user = {
        "claim": claim.model_dump(mode="json"),
        "model": model_payload,
        "top_features": [{"name": n, "importance": imp} for n, imp in top_features],
        "guidance": "Respond with JSON object mapping persona key -> explanation string. Keys: "
        + ", ".join(p.value for p in personas),
    }

    if not use_gemini and oai is None:

        def _stub() -> dict[str, str]:
            outcome = "approved" if model_payload["approved"] else "not approved"
            pap = model_payload["probability_approved"]
            rb = model_payload.get("risk_band")
            feats = "\n".join(f"- {n}: {imp:.4f}" for n, imp in top_features[:6])

            # Persona-specific header lines for offline stubs
            _persona_headers = {
                Persona.customer: "Our automated review suggests your claim would likely be",
                Persona.high_value_insured: (
                    "For your high-value device claim, our automated review suggests "
                    "the outcome would likely be"
                ),
                Persona.low_value_insured: (
                    "For your device claim, our automated review suggests "
                    "the outcome would likely be"
                ),
                Persona.small_issue: (
                    "For this single-issue repair claim, our automated review suggests "
                    "the outcome would likely be"
                ),
                Persona.end_of_duration: (
                    "Given your policy's current coverage window, our automated review suggests "
                    "the outcome would likely be"
                ),
                Persona.theft_claimant: (
                    "For this theft claim, our automated review suggests "
                    "the outcome would likely be"
                ),
                Persona.repeat_claimant: (
                    "Given the prior coverage activity on this policy, our automated review suggests "
                    "the outcome would likely be"
                ),
            }

            texts: dict[str, str] = {}
            for p in personas:
                if p is Persona.claims_adjuster:
                    band = f"- Estimated p(approve): {pap:.3f}; band={rb}\n"
                    manual = (
                        "Manual checks suggested: reserving vs. claimant statement, escalation if investigation "
                        "flags exist, and escalation if medical vs. claimant narrative mismatch.\n\n"
                    )
                    texts[p.value] = (
                        "Adjuster workstation summary:\n\n"
                        f"- Model label: {'APPROVED' if model_payload['approved'] else 'DECLINE'}\n"
                        + band
                        + f"- Drivers (RF feature importances; review before reliance):\n{feats}\n\n"
                        + manual
                        + "Offline: configure GEMINI_API_KEY (Google AI Studio) or OPENAI_API_KEY."
                    )
                else:
                    header = _persona_headers.get(p, _persona_headers[Persona.customer])
                    line_a = (
                        f"{header} {outcome.lower()} "
                        f"(estimated likelihood of approval approximately {pap:.0%}). "
                    )
                    line_b = "This is modeled guidance—not a binding decision—and a specialist "
                    line_b += "may still weigh new evidence.\n\n"
                    texts[p.value] = (
                        line_a
                        + f"Important factors include the patterns below:\n{feats}\n"
                        + line_b
                        + "Next steps you can consider: confirm dates and receipts, attach any missing accident "
                        + "report, and coordinate with your agent if timelines look tight.\n\n"
                        + "Offline: set GEMINI_API_KEY or OPENAI_API_KEY for live narrative explanations."
                    )
            return texts

        return await asyncio.to_thread(_stub)

    async def one_persona(persona: Persona) -> tuple[str, str]:
        prefix = persona_explain_style_prefix(persona)
        user_blob = {
            **base_user,
            "persona": persona.value,
            "style_prefix": prefix.strip(),
        }

        def _call() -> str:
            user_text = json.dumps(user_blob, indent=2)
            tail = f"\n\nWrite the explanation for persona={persona.value} only."

            if use_gemini:
                return gemini_generate_content(
                    api_key=settings.gemini_api_key or "",
                    model=settings.gemini_model,
                    system_text=system_base.strip(),
                    user_text=user_text + tail,
                    temperature=0.35,
                    max_output_tokens=850,
                )

            assert oai is not None
            resp = oai.chat.completions.create(
                model=settings.openai_chat_model,
                temperature=0.35,
                messages=[
                    {"role": "system", "content": system_base.strip()},
                    {"role": "user", "content": json.dumps(user_blob, indent=2)},
                    {"role": "user", "content": tail.strip()},
                ],
                max_tokens=650,
            )
            return (resp.choices[0].message.content or "").strip()

        text = await asyncio.to_thread(_call)
        return persona.value, text

    results = await asyncio.gather(*[one_persona(p) for p in personas])
    return {k: v for k, v in results}


def summarize_template(
    claim: ClaimInput,
    probability: float,
    approved: bool,
    top_features: list[tuple[str, float]],
) -> str:
    rrp = getattr(claim, "rrp", None)
    ctype = getattr(claim, "claimType", None)
    cov = getattr(claim, "coverage", None)
    amt_bits = ""
    if rrp is not None:
        amt_bits += f" RRP≈{float(rrp):,.0f}."
    ctx = f" claimType={ctype!s}, coverage={cov!s}.{amt_bits}"

    parts = [
        (
            f"Outcome tendency: {'approve' if approved else 'decline'} (p≈{probability:.2f}); "
            + ctx.strip()
        ),
        "Notable model drivers: " + "; ".join(f"{n} ({v:.3f})" for n, v in top_features[:5]) + ".",
        "This automated summary is illustrative; final decisions require human review.",
    ]
    return " ".join(parts)


async def generate_synthetic_json(
    *,
    n_scenarios: int,
    focus_value: str,
    deny_rate_hint: float,
    narrative_words_max: int,
    settings: Settings | None = None,
) -> tuple[list[dict[str, Any]], str]:
    settings = settings or get_settings()
    use_gemini = bool(settings.gemini_api_key)
    oai = None if use_gemini else _openai_client(settings)

    base_instructions = load_prompt("prompts/synthetic_generation.txt")

    user_prompt = {
        "instructions": base_instructions.strip(),
        "n_scenarios": n_scenarios,
        "focus": focus_value,
        "deny_rate_hint": deny_rate_hint,
        "narrative_words_max": narrative_words_max,
    }

    if not use_gemini and oai is None:

        def _stub() -> tuple[list[dict[str, Any]], str]:
            rows: list[dict[str, Any]] = []
            for i in range(n_scenarios):
                rows.append(
                    {
                        "structured_claim": {
                            "excessFee": 150.0 + 25 * i,
                            "claimType": "Accidental Damage" if focus_value == "borderline" else "Theft",
                            "coverage": "ADLD",
                            "deviceType": "PHONE",
                            "country": "SE",
                            "rrp": float(8990 + 500 * i),
                            "smashed": 1.0 if focus_value != "borderline" else 0.0,
                        },
                        "narrative": (
                            f"Stub scenario {i + 1}: borderline documentation and timing that would stress-test "
                            "the approval model. Offline mode — set GEMINI_API_KEY for Google AI Studio output."
                        ),
                    }
                )
            return rows, "Stub path: set GEMINI_API_KEY for LLM-generated synthetic stress cases."

        return await asyncio.to_thread(_stub)

    def _call_llm() -> tuple[list[dict[str, Any]], str]:
        user_json = json.dumps(user_prompt, indent=2)
        system = "You emit JSON only — a single JSON array, no markdown fences unless necessary."
        if use_gemini:
            raw = gemini_generate_content(
                api_key=settings.gemini_api_key or "",
                model=settings.gemini_model,
                system_text=system,
                user_text=user_json,
                temperature=0.75,
                max_output_tokens=8192,
            )
        else:
            assert oai is not None
            resp = oai.chat.completions.create(
                model=settings.openai_chat_model,
                temperature=0.75,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_json},
                ],
                max_tokens=3800,
            )
            raw = (resp.choices[0].message.content or "").strip()

        fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
        json_text = fenced[0].strip() if fenced else raw
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned non-JSON payload: {exc}\nFirst 400 chars: {raw[:400]!r}") from exc
        if not isinstance(data, list):
            raise ValueError("LLM did not return JSON array")
        rationale = (
            "Gemini-generated synthetic stress scenarios; review for PII and policy realism before retraining."
        )
        return data, rationale

    try:
        return await asyncio.to_thread(_call_llm)
    except Exception as exc:
        msg = str(exc)

        def _fallback() -> tuple[list[dict[str, Any]], str]:
            rows: list[dict[str, Any]] = []
            for i in range(n_scenarios):
                rows.append(
                    {
                        "structured_claim": {
                            "excessFee": 150.0 + 25 * i,
                            "claimType": "Accidental Damage" if focus_value == "borderline" else "Theft",
                            "coverage": "ADLD",
                            "deviceType": "PHONE",
                            "country": "SE",
                            "rrp": float(8990 + 500 * i),
                            "smashed": 1.0 if focus_value != "borderline" else 0.0,
                        },
                        "narrative": (
                            f"Fallback scenario {i + 1}: synthetic generator returned invalid JSON. "
                            f"({msg[:200]})"
                        ),
                    }
                )
            return rows, f"LLM path failed; showing deterministic fallback vignettes. Details: {msg}"

        return await asyncio.to_thread(_fallback)
