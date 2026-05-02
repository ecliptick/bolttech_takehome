# Claim Approval Agent (Take-Home Prototype)

Interactive **Streamlit** app that predicts claim **completion vs declined** outcomes from **`claim_use_case_dataset.xlsx`**, layers **multi-persona GenAI** rationales from `prompts/`, and exposes the same backends through an optional **FastAPI** façade for automation demos.

## What this repo demonstrates

| Area | Approach |
|------|----------|
| ML | sklearn `Pipeline` (`ColumnTransformer` + `RandomForestClassifier`), optional `RandomizedSearchCV`, JSON metadata beside `artifacts/approval_model.joblib` |
| GenAI | OpenAI-compatible chat; personas via `prompts/*.txt`; offline stubs when keys absent |
| UI | **`streamlit run streamlit_app.py`** forms + seeded Excel rows |
| API (optional) | FastAPI `/v1/predict`, `/v1/explain`, `/v1/synthetic` |
| Ops | Prompt files versioned via git, CI trains on checked-in workbook, `ruff` + `pytest` |

Full architecture and AWS/MLOps notes: [`DESIGN.md`](DESIGN.md).

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

## Optional REST API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

Open API docs at `http://localhost:8080/docs`. Payloads mirror the Excel schema via [`app/schemas.py`](app/schemas.py) (`ClaimInput`).

## Configure LLM (Google AI Studio / Gemini)

Primary integration uses the **Gemini** REST API (`generativelanguage.googleapis.com` `generateContent`), with optional **OpenAI-compatible** fallback.

1. Copy [`.env.example`](.env.example) → `.env` (never commit `.env`).
2. Set **`GEMINI_API_KEY=`** from [Google AI Studio](https://aistudio.google.com/apikey); optional **`GEMINI_MODEL=`** (default **`gemma-4-31b-it`** hosted via the same API — or e.g. **`gemini-2.5-flash-lite`**; see [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)). Set **`GEMINI_RPM=`** (e.g. **`15`**) or **`GEMINI_MIN_REQUEST_INTERVAL_S=`** so `generate_content` **serializes and spaces** completed requests under RPM (avoid **`15000`** if you meant **15**). **`EVAL_GEMINI_RPD_BUDGET`** (e.g. **`1500`**) and **`EVAL_EXPLAIN_BATCH_MAX`** (default **20** in `scratch/generate_eval_logs_genai.py`) **derive explain batch size**: prefer large batches (fewer calls) up to the max, while ensuring a full run stays within the RPD ceiling; if RPD is tight, batch size increases even past the max (with a console warning). **Without** `EVAL_GEMINI_RPD_BUDGET`, batch size defaults to **`min(EVAL_EXPLAIN_BATCH_MAX, slots)`** unless **`EVAL_EXPLAIN_BATCH_SIZE`** is set. If you see **`503 Service Unavailable`**, the client **retries** with exponential backoff; if failures persist, try another model id or wait (Google-side capacity). The **`01_load_and_explore_data`** notebook batches **800** `issueDesc` lines per call with a **65k** output-token cap to reduce total requests versus **~500 RPD** tiers.
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

Regenerate all four from the generator (after editing steps): `python scripts/emit_pipeline_notebooks.py`

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

## Responsible use

Model scores and narratives are illustrative; human adjusters adjudicate regulated insurance outcomes.
