# Claim Approval Agent

Interactive **Streamlit** app that predicts claim **completion vs declined** outcomes from **`claim_use_case_dataset.xlsx`**, layers **multi-persona GenAI** rationales from `prompts/`, and exposes the same backends through a **FastAPI** façade for automation demos.

This README covers **how to run everything**. All design rationale and decision-making — architecture (with an honest split between what is implemented and what is a production-only extension), ML choices, prompt strategy, deployment, and evaluation — is consolidated in [`DESIGN.md`](DESIGN.md). Module-specific justifications: [`docs/FEATURE_ENGINEERING.md`](docs/FEATURE_ENGINEERING.md), [`docs/AWS_DEPLOYMENT.md`](docs/AWS_DEPLOYMENT.md), [`docs/PRODUCTION_ARCHITECTURE.md`](docs/PRODUCTION_ARCHITECTURE.md).

## What this repo contains

| Area | Approach |
|------|----------|
| ML | sklearn `Pipeline` (`ColumnTransformer` + `RandomForestClassifier`) + `DummyClassifier` baseline, optional `RandomizedSearchCV` on ROC-AUC, metadata JSON beside `artifacts/approval_model.joblib` |
| GenAI | **Gemini / Gemma** via Google AI Studio `generateContent` (optional OpenAI-compatible fallback); eight personas via `prompts/explain_*_prefix.txt`; deterministic offline stubs when keys are absent |
| UI | **`streamlit run streamlit_app.py`** — primary surface; forms + seeded Excel rows + MLOps retrain-compare tab |
| API | FastAPI `/v1/predict`, `/v1/explain`, `/v1/synthetic`, `/v1/repair-text`, `/v1/model/info`, `/health`, `/metrics` |
| Ops | CI trains + lints + tests + builds the container on every push; prompts versioned in git; structured JSON-line `prediction`/`explain` events with model-lineage fields |

## Dataset and label

Place **`claim_use_case_dataset.xlsx`** next to [`app/config.py`](app/config.py) defaults (repo root).

- **`status` column**: `Completed` → positive class (claim processed successfully), `Declined` → negative.

Optional override via environment variable **`CLAIM_DATA_XLSX`** (see [`.env.example`](.env.example)).

## Prerequisites

Python **3.11+**. Install deps:

```bash
pip install -r requirements.txt
pip install -e .   # optional editable install for imports
```

## Train

Writes `artifacts/approval_model.joblib` + `artifacts/approval_model_meta.json`:

```bash
python -m app.ml.train                     # randomized search (~ CPU seconds–minutes)
python -m app.ml.train --no-tune --data claim_use_case_dataset.xlsx
```

`--data` defaults to `SETTINGS.claim_data_xlsx`.

## Run the Streamlit app (primary)

```bash
streamlit run streamlit_app.py
```

Use the sidebar to point at your workbook path, slide a historical row seed, edit fields, and run prediction + persona prompts on the Multi-persona tab.

## REST API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

Open API docs at `http://localhost:8080/docs`. Payloads mirror the Excel schema via [`app/schemas.py`](app/schemas.py) (`ClaimInput`).

The API and the Streamlit UI share the same `app.ml.predict` and `app.genai.service` helpers — running the API is optional for interactive use but needed for any integration demo.

## Configure LLM (Google AI Studio / Gemini)

Primary integration uses the **Gemini** REST API (`generativelanguage.googleapis.com` `generateContent`), with optional **OpenAI-compatible** fallback.


1. Set **`GEMINI_API_KEY=`** from [Google AI Studio](https://aistudio.google.com/apikey); optional **`GEMINI_MODEL=`** (default **`gemma-4-31b-it`** hosted via the same API — or e.g. **`gemini-2.5-flash-lite`**; see [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)). Set **`GEMINI_RPM=`** (e.g. **`15`**) or **`GEMINI_MIN_REQUEST_INTERVAL_S=`** so `generate_content` **serializes and spaces** completed requests under RPM (avoid **`15000`** if you meant **15**). **`EVAL_GEMINI_RPD_BUDGET`** (e.g. **`1500`**) and **`EVAL_EXPLAIN_BATCH_MAX`** (default **20** in `scratch/generate_eval_logs_genai.py`) **derive explain batch size**: prefer large batches (fewer calls) up to the max, while ensuring a full run stays within the RPD ceiling; if RPD is tight, batch size increases even past the max (with a console warning). **Without** `EVAL_GEMINI_RPD_BUDGET`, batch size defaults to **`min(EVAL_EXPLAIN_BATCH_MAX, slots)`** unless **`EVAL_EXPLAIN_BATCH_SIZE`** is set. If you see **`503 Service Unavailable`**, the client **retries** with exponential backoff; if failures persist, try another model id or wait (Google-side capacity). The **`01_load_and_explore_data`** notebook batches **800** `issueDesc` lines per call with a **65k** output-token cap to reduce total requests versus **~500 RPD** tiers.
3. Optionally set **`OPENAI_API_KEY`** only if you want legacy OpenAI calls.

Without keys, Streamlit and `/explain`/`/synthetic` use deterministic **offline stubs**.

**Feature engineering & text handling** (what the model uses, what can be improved): [`docs/FEATURE_ENGINEERING.md`](docs/FEATURE_ENGINEERING.md).  
**Separate text repair:** `POST /v1/repair-text` — deterministic encoding cleanup plus optional **Gemini JSON** rewrite to English (`app/genai/text_repair.py`); does not change predictions unless you paste repaired text back into a claim.

### Pipeline notebooks (inspect each step)

| Notebook | What it runs |
|----------|----------------|
| [`notebooks/01_load_and_explore_data.ipynb`](notebooks/01_load_and_explore_data.ipynb) | Load Excel, labels distribution, missingness |
| [`notebooks/02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb) | `load_claims_training_frame`, `engineer_features`, preprocessor shape |
| [`notebooks/03_train_model.ipynb`](notebooks/03_train_model.ipynb) | **`subprocess` → `python -m app.ml.train --no-tune`**, then reads `approval_model_meta.json`; optional in-process `train()` |
| [`notebooks/04_inference_and_genai.ipynb`](notebooks/04_inference_and_genai.ipynb) | Bundle load, `predict_batch`, `top_feature_names`, `explain_personas` |
| [`notebooks/05_mock_incremental_data.ipynb`](notebooks/05_mock_incremental_data.ipynb) | Sparse decline / RF-border strata → targeted **synthetic** claim generation (LLM) |
| [`notebooks/06_evaluation_and_monitoring.ipynb`](notebooks/06_evaluation_and_monitoring.ipynb) | Eval logs (ML + dual-persona GenAI), drift views, baseline-from-meta monitoring narrative |

Regenerate notebooks 01–04 from the generator (after editing steps): `python scripts/emit_pipeline_notebooks.py`

Run notebooks with the project root on `PYTHONPATH` **or** launch Jupyter from the repo root so the first cell can add `PROJECT_ROOT` to `sys.path`.

## Prompt strategy (short)

Neutral system guardrail + persona conditioners + grounded JSON payloads (scores + claimant fields + global importances) — detailed in [`DESIGN.md`](DESIGN.md).

## Tests & CI

```bash
ruff check app tests streamlit_app.py
pytest
```

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) validates the workbook exists, trains with `--no-tune`, lints `streamlit_app.py`, and exercises FastAPI endpoints.

## Container

```bash
docker build -t claim-agent:dev .
docker run --rm -p 8501:8501 -e OPENAI_API_KEY=... claim-agent:dev
```

The image exposes Streamlit on port **8501** and bakes `--no-tune` training locally.

