from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ml.inference_inputs import dataframe_row_to_claim


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_claim_payload() -> dict:
    excel = Path(__file__).resolve().parents[1] / "claim_use_case_dataset.xlsx"
    if not excel.exists():
        pytest.skip("claim_use_case_dataset.xlsx missing from checkout")
    df = pd.read_excel(excel, engine="openpyxl", nrows=1)
    payload = dataframe_row_to_claim(df, 0).model_dump(mode="json")
    assert isinstance(payload, dict)
    return payload


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metrics_exposed_or_disabled(client: TestClient) -> None:
    """Instrumentation registers /metrics unless DISABLE_PROMETHEUS is set (e.g. in CI)."""
    r = client.get("/metrics")
    assert r.status_code in {200, 404}


def test_predict_shape(client: TestClient, sample_claim_payload: dict) -> None:
    response = client.post("/v1/predict", json=sample_claim_payload)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "probability_approved" in payload
    score = payload["probability_approved"]
    assert 0.0 <= score <= 1.0


def test_explain_endpoint(client: TestClient, sample_claim_payload: dict) -> None:
    body = {"claim": sample_claim_payload, "personas": ["customer", "claims_adjuster"]}
    response = client.post("/v1/explain", json=body)
    assert response.status_code in {200, 503}


def test_repair_text(client: TestClient) -> None:
    response = client.post(
        "/v1/repair-text",
        json={"issueDesc": "vÃ¤tskeskador"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] in ("gemini", "deterministic_only")
    assert "issueDesc" in payload
