# Production architecture

End-to-end production flow with **structured prediction logs** (no notebooks). Inference emits JSON lines suitable for CloudWatch Logs or any log drain.

## Prediction log events

Emitted by `app/main.py` via the standard logger (`claim_agent`), one JSON object per line:

| `event`       | When                         | Correlation |
|---------------|------------------------------|-------------|
| `prediction`  | Every `/v1/predict` and each `/v1/explain` (explain runs predict first) | `request_id` |
| `explain`     | After LLM explanations on `/v1/explain` only | `prediction_request_id` → matching `prediction.request_id`; own `request_id` |

**Prediction payload** includes: `ts`, `request_id`, `claim_id`, `prediction_approved`, `probability_approved`, `probability_declined`, `risk_band`, `features`, `top_features`, plus **model correlation** from the loaded bundle (`model_trained_at`, `model_git_commit`, `model_sklearn_version`, `model_train_n_rows`, `model_train_random_state`, `model_artifact_path`).

**Explain payload** adds: `latency_ms`, `personas`, `claim_id`, and the same model correlation fields.

## Diagram

```mermaid
flowchart TB
  subgraph ingest["Data ingest"]
    SRC["Upstream systems / exports"]
    LAND["S3 landing bucket"]
    CURATED["Curated dataset snapshot"]
  end

  subgraph orch["Orchestration"]
    EB["EventBridge schedule / event"]
    SF["Step Functions or Prefect/Dagster"]
  end

  subgraph train_job["Training compute"]
    TR_TASK["ECS Fargate / Batch task"]
    TR_CLI["python -m app.ml.train"]
  end

  subgraph artifacts["Artifacts"]
    S3M["S3 ml-artifacts bucket"]
    JLB["approval_model.joblib"]
    META["approval_model_meta.json"]
  end

  subgraph promote["Release"]
    GATES["Metric thresholds + tests"]
    PARAM["SSM / tagged active model URI"]
  end

  subgraph serve["Serving"]
    ECR["ECR"]
    SVC["ECS service"]
    API["FastAPI app/main.py"]
  end

  subgraph infer_logs["Prediction logs — stdout → CloudWatch"]
    LG["Log group e.g. /claim-approval-agent-dev/api"]
    PREDL["event: prediction\n+ probabilities + features\n+ model_* correlation"]
    EXPL["event: explain\n+ prediction_request_id\n+ latency_ms + personas"]
    CW_SUB["Optional: subscription → analytics"]
  end

  subgraph genai["GenAI"]
    GEM["Gemini generateContent"]
  end

  subgraph obs_extra["Metrics"]
    PROM["GET /metrics Prometheus"]
  end

  SRC --> LAND --> CURATED
  EB --> SF --> TR_TASK --> TR_CLI
  CURATED --> TR_CLI
  TR_CLI --> S3M
  S3M --> JLB
  S3M --> META
  META --> GATES --> PARAM
  ECR --> SVC --> API
  PARAM --> API
  JLB --> API
  API --> GEM
  API --> PREDL
  API --> EXPL
  PREDL --> LG
  EXPL --> LG
  LG --> CW_SUB
  API --> PROM
```
