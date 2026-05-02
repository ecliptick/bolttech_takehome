"""FastAPI service for claim prediction and GenAI explanations."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import get_settings
from app.genai.service import explain_personas, generate_synthetic_json, summarize_template
from app.genai.text_repair import repair_text_to_english
from app.ml.inference_inputs import claim_row_dataframe
from app.ml.predict import load_metadata, model_identity_for_logs, predict_batch, top_feature_names
from app.schemas import (
    ClaimInput,
    ExplainRequest,
    ExplainResponse,
    PredictResponse,
    SyntheticRequest,
    SyntheticResponse,
    SyntheticScenario,
    TextRepairRequest,
    TextRepairResponse,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("claim_agent")


def _maybe_expose_prometheus(app: FastAPI) -> None:
    if os.environ.get("DISABLE_PROMETHEUS", "").strip().lower() in {"1", "true", "yes"}:
        return
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    except Exception as exc:
        log.warning("Prometheus metrics not enabled: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ = get_settings()
    try:
        from app.ml import predict as predmod

        predmod.load_model_bundle()
    except FileNotFoundError as e:
        log.warning("Model not found at startup: %s", e)
    yield


app = FastAPI(
    title="Claim Approval Agent",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
_maybe_expose_prometheus(app)


def _claim_to_frame(claim: ClaimInput) -> pd.DataFrame:
    return claim_row_dataframe(claim)


def _emit_prediction_log(
    *,
    request_id: str,
    claim_id: str,
    pr: PredictResponse,
    feats: list[tuple[str, float]],
    features_dict: dict,
) -> None:
    payload = {
        "event": "prediction",
        "ts": datetime.now(tz=UTC).isoformat(),
        "request_id": request_id,
        "claim_id": claim_id,
        "prediction_approved": pr.approved,
        "probability_approved": pr.probability_approved,
        "probability_declined": pr.probability_declined,
        "risk_band": pr.risk_band,
        "features": features_dict,
        "top_features": [{"name": n, "importance": imp} for n, imp in feats],
        **model_identity_for_logs(),
    }
    log.info(json.dumps(payload))


def _predict_response(claim: ClaimInput) -> tuple[PredictResponse, list[tuple[str, float]], str]:
    df = _claim_to_frame(claim)
    pred, p_app = predict_batch(df)
    approved = bool(int(pred[0]) == 1)
    p = float(p_app[0])
    feats = top_feature_names(8)
    snippet = summarize_template(claim, p, approved, feats)
    pr = PredictResponse(
        approved=approved,
        probability_approved=p,
        probability_declined=1.0 - p,
        risk_band=_risk_band(p),
        explain_snippet=snippet,
    )

    prediction_request_id = str(uuid.uuid4())
    features_dict = claim.model_dump(mode="json")
    _emit_prediction_log(
        request_id=prediction_request_id,
        claim_id=claim.claimId or "unknown",
        pr=pr,
        feats=feats,
        features_dict=features_dict,
    )

    return pr, feats, prediction_request_id


def _risk_band(p: float) -> str:
    if p >= 0.75:
        return "likely_approve"
    if p >= 0.55:
        return "lean_approve"
    if p >= 0.45:
        return "borderline"
    if p >= 0.25:
        return "lean_decline"
    return "likely_decline"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


class ModelInfoResponse(BaseModel):
    metadata: dict
    prompt_files: list[str]


@app.get("/v1/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    root = get_settings().project_root
    meta = load_metadata()
    prompt_dir = root / "prompts"
    files: list[str] = []
    if prompt_dir.exists():
        files = sorted(str(p.relative_to(root)).replace("\\", "/") for p in prompt_dir.glob("*.txt"))
    return ModelInfoResponse(metadata=meta, prompt_files=files)


@app.post("/v1/predict", response_model=PredictResponse)
def predict_one(claim: ClaimInput) -> PredictResponse:
    try:
        pr, _, _pred_rid = _predict_response(claim)
        return pr
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.post("/v1/explain", response_model=ExplainResponse)
async def explain(req: ExplainRequest) -> ExplainResponse:
    try:
        pr, feats, prediction_request_id = _predict_response(req.claim)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    model_payload = {
        "approved": pr.approved,
        "probability_approved": pr.probability_approved,
        "risk_band": pr.risk_band,
    }
    t0 = time.perf_counter()
    explanations = await explain_personas(
        claim=req.claim,
        personas=req.personas,
        model_payload=model_payload,
        top_features=feats,
    )
    dt_ms = (time.perf_counter() - t0) * 1000
    log.info(
        json.dumps(
            {
                "event": "explain",
                "ts": datetime.now(tz=UTC).isoformat(),
                "request_id": str(uuid.uuid4()),
                "prediction_request_id": prediction_request_id,
                "claim_id": req.claim.claimId or "unknown",
                "latency_ms": round(dt_ms, 2),
                "personas": [p.value for p in req.personas],
                **model_identity_for_logs(),
            }
        )
    )
    return ExplainResponse(prediction=pr, explanations=explanations)


@app.post("/v1/synthetic", response_model=SyntheticResponse)
async def synthetic(req: SyntheticRequest) -> SyntheticResponse:
    try:
        raw, rationale = await generate_synthetic_json(
            n_scenarios=req.n_scenarios,
            focus_value=req.focus.value,
            deny_rate_hint=req.deny_rate_hint,
            narrative_words_max=req.narrative_words_max,
        )
    except ValueError as e:
        raise HTTPException(status_code=502, detail=f"LLM output parse error: {e}") from e
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"Invalid JSON from LLM: {e}") from e

    out: list[SyntheticScenario] = []
    for item in raw:
        out.append(SyntheticScenario(structured_claim=item["structured_claim"], narrative=item.get("narrative", "")))
    return SyntheticResponse(scenarios=out, rationale=rationale)


@app.post("/v1/repair-text", response_model=TextRepairResponse)
async def repair_text(req: TextRepairRequest) -> TextRepairResponse:
    """Deterministic UTF-8/unicode cleanup, then optional Gemini JSON rewrite to English."""
    repaired, rationale, backend = await repair_text_to_english(
        issue_desc=req.issueDesc,
        product_name=req.productName,
        product_desc=req.productDesc,
    )
    return TextRepairResponse(
        issueDesc=repaired.get("issueDesc"),
        productName=repaired.get("productName"),
        productDesc=repaired.get("productDesc"),
        rationale=rationale,
        backend=backend,
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "claim-approval-agent", "docs": "/docs"}

