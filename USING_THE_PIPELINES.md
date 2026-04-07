# Using the Pipelines
## Lighthouse Sanctuary — ML Service Reference

> This document explains how to trigger each pipeline, what inputs are required, and what each one returns. Intended for the .NET backend team integrating with the ML service.

---

## Overview

The ML service exposes three POST endpoints. All require the `X-API-Key` header. The `/health` endpoint requires no auth.

```
Base URL (local):       http://localhost:8000
Base URL (deployed):    https://<your-container-app>.azurecontainerapps.io
```

| Endpoint | Pipeline | Who calls it | When |
|---|---|---|---|
| `POST /score/churn` | Donor Churn | .NET backend | After new donation data is imported |
| `POST /score/impact` | Impact Attribution | .NET backend | After new allocation data is imported |
| `POST /score/residents` | Resident Risk | .NET backend | After health/education/incident records are updated |
| `GET /health` | — | Load balancer / monitoring | Anytime |

---

## Authentication

Every POST endpoint requires an API key header:

```
X-API-Key: <ML_SERVICE_API_KEY>
```

The key is set as an environment variable on the service. The .NET backend should store it as a secret and never expose it client-side.

---

## POST /score/churn

### What it does

Scores all active donors for churn risk. Loads the trained churn model from Azure Blob Storage, runs each supporter through it, and writes scores to the `donor_churn_scores` table in Supabase.

### Request

```http
POST /score/churn
X-API-Key: <key>
```

No request body required.

### Response

```json
{
  "status": "scoring complete",
  "count_scored": 312
}
```

### What gets written to Supabase

Table: `donor_churn_scores`

| Column | Type | Description |
|---|---|---|
| `supporter_id` | integer | References `supporters` |
| `churn_score` | float | Probability of churning (0.0 – 1.0) |
| `churn_tier` | text | `high`, `medium`, or `low` |
| `top_churn_factors` | jsonb | List of human-readable reasons driving the score |
| `scored_at` | timestamptz | When this score was generated |
| `model_version` | text | Timestamp of the model used |

### Tier thresholds

| Tier | Score range |
|---|---|
| `high` | ≥ 0.70 |
| `medium` | 0.40 – 0.69 |
| `low` | < 0.40 |

### When to call

Call after any batch of new donation data is imported. The scoring run replaces previous scores via upsert — safe to call repeatedly.

---

## POST /score/impact

### What it does

Generates personalized impact statements for donors based on how their allocations correlate with measurable resident outcomes at the safehouse level. Loads the trained OLS effects from Azure Blob Storage and writes one or more statements per donor to the `donor_impact_statements` table.

Statements are only generated when the underlying statistical effect is significant (p < 0.05). Donors whose allocations fall into program areas without a significant signal will not receive a statement.

### Request

```http
POST /score/impact
X-API-Key: <key>
```

No request body required.

### Response

```json
{
  "status": "impact statements generated",
  "count_scored": 83
}
```

`count_scored` is the number of statements written, not the number of donors. One donor may receive multiple statements (one per program area / outcome combination).

### What gets written to Supabase

Table: `donor_impact_statements`

| Column | Type | Description |
|---|---|---|
| `statement_id` | uuid | Primary key |
| `supporter_id` | integer | References `supporters` |
| `safehouse_id` | text | Which safehouse the allocation went to |
| `program_area` | text | e.g. `Wellbeing`, `Education` |
| `allocation_amount` | float | Amount the donor allocated (USD) |
| `outcome_metric` | text | e.g. `health_3m`, `education_6m` |
| `time_window_months` | integer | 3 or 6 months post-allocation |
| `estimated_pct_change` | float | Estimated % change in the outcome metric |
| `statement_text` | text | Ready-to-send human-readable statement |
| `generated_at` | timestamptz | When this statement was generated |
| `model_version` | text | Timestamp of the effects model used |

### Example statement

> "Your $150 allocated to Wellbeing at Safehouse Cebu was associated with a 9.2% improvement in average health and wellbeing scores over the following 3 months (estimated range: 4.1%–14.3%)."

### When to call

Call after any new allocation data is imported, or after new `safehouse_monthly_metrics` records are added. Each run regenerates all statements fresh and removes the prior version.

---

## POST /score/residents

### What it does

Scores all active residents across all safehouses on two dimensions:

1. **Regression risk** — likelihood that a resident's situation is worsening
2. **Reintegration readiness** — likelihood that a resident is on track for reintegration

Also generates `top_concern_factors` and `top_strength_factors` for each resident so social workers can understand the score without looking at raw numbers.

**Access restricted.** The `resident_risk_scores` table is protected by Supabase row-level security — only users with `Admin` or `SocialWorker` roles can read from it. This endpoint must never be surfaced to donor-facing or public-facing services.

### Request

```http
POST /score/residents
X-API-Key: <key>
```

No request body required.

### Response

```json
{
  "status": "resident scoring complete",
  "count_scored": 41
}
```

### What gets written to Supabase

Table: `resident_risk_scores` (RLS restricted)

| Column | Type | Description |
|---|---|---|
| `resident_id` | integer | Primary key — references `residents` |
| `regression_risk_score` | float | Risk probability (0.0 – 1.0) |
| `regression_risk_tier` | text | `high`, `medium`, or `low` |
| `reintegration_score` | float | Readiness probability (0.0 – 1.0) |
| `reintegration_tier` | text | `ready`, `in_progress`, or `not_ready` |
| `top_concern_factors` | jsonb | Up to 3 human-readable risk signals |
| `top_strength_factors` | jsonb | Up to 3 human-readable positive signals |
| `scored_at` | timestamptz | When this score was generated |
| `model_version` | text | Timestamp of the model used |

### Tier thresholds

**Regression risk**

| Tier | Score range |
|---|---|
| `high` | ≥ 0.70 |
| `medium` | 0.40 – 0.69 |
| `low` | < 0.40 |

**Reintegration readiness**

| Tier | Score range |
|---|---|
| `ready` | ≥ 0.70 |
| `in_progress` | 0.40 – 0.69 |
| `not_ready` | < 0.40 |

### Example output record

```json
{
  "resident_id": 1,
  "regression_risk_score": 0.8242,
  "regression_risk_tier": "high",
  "reintegration_score": 0.3269,
  "reintegration_tier": "not_ready",
  "top_concern_factors": [
    "Declining health scores",
    "Recent incidents filed",
    "Few recent counseling sessions"
  ],
  "top_strength_factors": [
    "Progress noted in counseling sessions",
    "Intervention plans on track",
    "Good family cooperation"
  ],
  "scored_at": "2026-04-07T20:30:07Z",
  "model_version": "20260407_1429"
}
```

### When to call

Call after any of the following events:
- New health or education records are added
- A new incident report is filed (immediate rescore recommended)
- A resident's case status changes

---

## GET /health

Returns `200 OK` if the service is running. No auth required. Used by Azure load balancer and monitoring.

```json
{ "status": "ok" }
```

---

## Retraining Schedule

Scoring runs can be triggered any time via the API above. Model retraining is handled by Azure Container Apps Jobs on a separate schedule and is never triggered via the API.

| Pipeline | Retrain frequency | Script |
|---|---|---|
| Donor Churn | Monthly | `train_churn.py` |
| Impact Attribution | Quarterly (Jan / Apr / Jul / Oct) | `train_impact.py` |
| Resident Risk | Bi-weekly (1st and 15th) | `train_risk.py` |

---

## Current Model Limitations

Both the impact attribution and resident risk models are trained on a small dataset (41 active residents, 59 donors). Metrics and scores will become more reliable as data accumulates. The following warnings are expected and logged at each run until the dataset grows:

- Impact: only 1 program area (`Wellbeing`) currently has a statistically significant effect
- Resident risk: AUC and recall are below target thresholds — scores are a weak signal only
- Reintegration: AUC at threshold — readiness scores should be used as supporting context, not definitive assessments

These limitations are logged automatically on each scoring run. No action is required until the resident population exceeds 100 active cases or the donor base expands significantly.
