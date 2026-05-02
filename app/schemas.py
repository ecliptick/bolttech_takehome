from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Persona(StrEnum):
    customer = "customer"
    claims_adjuster = "claims_adjuster"
    # Domain-specific claimant archetypes (grounded in EDA schema)
    high_value_insured = "high_value_insured"
    low_value_insured = "low_value_insured"
    small_issue = "small_issue"
    end_of_duration = "end_of_duration"
    theft_claimant = "theft_claimant"
    repeat_claimant = "repeat_claimant"


class SyntheticFocus(StrEnum):
    denial_patterns = "denial_patterns"
    borderline = "borderline"


class ClaimInput(BaseModel):
    """One row aligned with ``claim_use_case_dataset.xlsx`` columns (excluding ``status``, the label)."""

    claimId: str | None = Field(default=None, description="Optional client-supplied id for tracing / logs")

    excessFee: float | None = Field(default=None, description="EUR / local currency fee")
    rrp: float | None = None
    balanceRRP: float | None = None
    oldBalanceRRP: float | None = None
    productName: str | None = None
    productDesc: str | None = None
    coverage: str | None = None
    productCode: str | None = None
    policyStartDate: date | datetime | str | None = None
    policyEndDate: date | datetime | str | None = None
    policyStatus: str | None = None
    retailerName: str | None = None
    deviceType: str | None = None
    make: str | None = None
    model: str | None = None
    purchaseDate: date | datetime | str | None = None
    deviceCost: float | None = None
    relationship: str | None = None
    channel: str | None = None
    claimType: str | None = None
    country: str | None = None
    turnOnOff: float | None = None
    touchScreen: float | None = None
    smashed: float | None = None
    frontCamera: float | None = None
    backCamera: float | None = None
    frontOrBackCamera: float | None = None
    audio: float | None = None
    mic: float | None = None
    buttons: float | None = None
    connection: float | None = None
    charging: float | None = None
    other: float | None = None
    issueDesc: str | None = None


class PredictResponse(BaseModel):
    approved: bool
    probability_approved: float
    probability_declined: float
    risk_band: str
    explain_snippet: str | None = Field(
        default=None,
        description="Brief template summary when GENAI explanations are unavailable",
    )


class ExplainRequest(BaseModel):
    claim: ClaimInput
    personas: list[Persona] = Field(
        default=[Persona.customer, Persona.claims_adjuster],
        min_length=1,
    )


class ExplainResponse(BaseModel):
    prediction: PredictResponse
    explanations: dict[str, str]


class SyntheticRequest(BaseModel):
    n_scenarios: int = Field(ge=1, le=25, default=5)
    focus: SyntheticFocus = SyntheticFocus.denial_patterns
    deny_rate_hint: float = Field(default=0.55, ge=0.05, le=0.95)
    narrative_words_max: int = Field(default=180, ge=60, le=600)


class SyntheticScenario(BaseModel):
    structured_claim: dict[str, Any]
    narrative: str


class SyntheticResponse(BaseModel):
    scenarios: list[SyntheticScenario]
    rationale: str


class TextRepairRequest(BaseModel):
    """Free-text columns to clean / rewrite."""

    issueDesc: str | None = None
    productName: str | None = None
    productDesc: str | None = None


class TextRepairResponse(BaseModel):
    issueDesc: str | None = None
    productName: str | None = None
    productDesc: str | None = None
    rationale: str
    backend: Literal["gemini", "deterministic_only"]
