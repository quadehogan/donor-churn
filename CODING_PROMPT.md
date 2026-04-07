# Lighthouse ML Service — Coding Agent Prompt

You are building the Python ML microservice for Lighthouse Sanctuary, a child welfare nonprofit. The pipeline plan documents dropped into this context are your specification. Your job is to implement them as working Python files.

---

## Project Context

**What this service is:**
A standalone FastAPI microservice that trains machine learning models on donor and resident data, stores model artifacts in Azure Blob Storage, writes prediction scores and outputs to a Supabase PostgreSQL database, and exposes endpoints for the .NET backend to trigger scoring runs.

**What this service is not:**
It is not the web application. It does not handle authentication for end users. It does not serve the frontend. It does not contain any business logic beyond ML training and inference.

---

## Repo Structure

Every file you create must fit this layout. Do not deviate from it.

```
lighthouse-ml/
├── .env                          ← never create or modify this — user manages it
├── .gitignore
├── Dockerfile
├── requirements.txt
├── main.py                       ← FastAPI app — add one endpoint per pipeline
├── pipelines/
│   ├── churn/                    ← donor churn pipeline (may already exist)
│   │   ├── features.py
│   │   ├── train_churn.py
│   │   └── score_churn.py
│   ├── impact_attribution/       ← create if spec is provided
│   │   ├── features.py
│   │   ├── train_impact.py
│   │   ├── score_impact.py
│   │   └── statement_builder.py
│   └── resident_risk/            ← create if spec is provided
│       ├── features.py
│       ├── train_risk.py
│       ├── train_reintegration.py
│       └── score_residents.py
├── storage/
│   └── blob_client.py            ← shared Azure Blob Storage helpers
└── db/
    └── connection.py             ← shared SQLAlchemy engine factory
```

---

## Shared Infrastructure

### `db/connection.py`
Creates the SQLAlchemy engine from `SUPABASE_DB_URL` in `.env`. All pipeline files import from here — no pipeline should create its own engine inline.

```python
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

def get_engine():
    return create_engine(os.getenv("SUPABASE_DB_URL"))
```

### `storage/blob_client.py`
All Blob Storage reads and writes go through this module. It is already defined in the pipeline specs. Implement it exactly as written — do not add new Blob functions unless a pipeline spec requires it.

---

## Environment Variables

These are always loaded from `.env` via `python-dotenv`. Never hardcode values. Never print or log them.

| Variable | Used by |
|---|---|
| `SUPABASE_DB_URL` | All pipelines — PostgreSQL connection string, Session mode port 5432 |
| `AZURE_STORAGE_CONNECTION_STRING` | All pipelines — Blob Storage |
| `ML_SERVICE_API_KEY` | `main.py` — API key check on all POST endpoints |

---

## Rules for Every File You Write

**SQL queries**
- Write all SQL in the `features.py` file for each pipeline, as module-level string constants
- Use parameterized queries via SQLAlchemy `text()` for any user-supplied values
- Never use string formatting or f-strings to build SQL

**Imports**
- Use only packages listed in `requirements.txt`
- Do not add new packages unless the pipeline spec explicitly requires one
- If a spec requires a package not yet in `requirements.txt`, add it and note it

**Training scripts**
- Always split before any preprocessing
- Always assert evaluation thresholds before saving artifacts — if assertions fail, the script must exit with a non-zero code and print a clear error message
- Always save a versioned copy of artifacts alongside the `latest` pointer in Blob Storage
- Write metadata and metrics JSON alongside every model artifact

**Inference scripts**
- Always load the model from Blob Storage — never from local filesystem in production code
- Always use `ON CONFLICT DO UPDATE` (upsert) when writing to Supabase — never plain INSERT
- Always log a summary at the end: how many records were scored, breakdown by tier

**FastAPI (`main.py`)**
- Every POST endpoint requires the `X-API-Key` header check via `Depends(verify_key)`
- Never expose a `/train` endpoint — training is triggered only by Azure Container Apps Jobs
- The `/health` endpoint requires no auth
- Return meaningful error messages with appropriate HTTP status codes

**Privacy and security**
- This service processes records of minors who are abuse survivors
- Never log individual resident data — log counts and aggregate stats only
- Never return raw resident records from any endpoint
- The resident risk pipeline output goes to a table with Supabase row-level security — do not bypass or modify that policy

---

## How to Read a Pipeline Spec

Each pipeline spec you receive is a Markdown document structured in 8 phases:

- **Phase 1** — tells you what the model predicts and what the success criteria are
- **Phase 2** — contains the exact SQL queries to use; implement these verbatim in `features.py`
- **Phase 3** — exploration code; you do not need to implement this as a deployed file, but use it to understand the data shape
- **Phase 4** — feature engineering; implement as an `engineer_features(df)` function in `features.py`
- **Phase 5** — model training code; implement in `train_*.py`
- **Phase 6** — evaluation and tuning code; include the assertion gates in `train_*.py`
- **Phase 7** — explainability; implement the factor-generation functions in `features.py` or the relevant script
- **Phase 8** — deployment; contains the exact Supabase table DDL, the training job, the inference job, and the FastAPI endpoint to add

When implementing, follow the Phase 8 code as the primary reference for the final deployed files. Phases 2–7 inform the logic inside those files.

---

## What to Produce

For each pipeline spec dropped into context, produce the following files:

1. `pipelines/<pipeline_name>/features.py` — SQL query constants + `engineer_features()` + factor-generation helpers
2. `pipelines/<pipeline_name>/train_<name>.py` — full training script
3. `pipelines/<pipeline_name>/score_<name>.py` — full inference + Supabase upsert script
4. Any additional files called out in the spec (e.g., `statement_builder.py`)
5. An updated `main.py` with the new endpoint added
6. An updated `requirements.txt` if the spec introduces new packages
7. The Supabase DDL SQL as a comment block at the top of the scoring script — clearly labeled as a one-time setup step

Do not produce Jupyter notebooks. Do not produce test files unless asked. Do not produce documentation files — the spec is the documentation.

---

## Dockerfile (reference — do not modify unless asked)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Base `requirements.txt`

Start from this. Add packages as needed by pipeline specs.

```
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.5.2
sqlalchemy==2.0.36
psycopg2-binary==2.9.10
fastapi==0.115.0
uvicorn==0.32.0
python-dotenv==1.0.1
matplotlib==3.9.2
scipy==1.13.1
azure-storage-blob==12.23.0
statsmodels==0.14.4
```
