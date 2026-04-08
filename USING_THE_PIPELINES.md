# Using the Pipelines
## Lighthouse Sanctuary — ML Service Reference

> This document explains how to trigger each pipeline, what inputs are required, and what each one returns. Intended for the .NET backend team integrating with the ML service.

---

## Overview

The ML service exposes five POST endpoints. All require the `X-API-Key` header. The `/health` endpoint requires no auth.

```
Base URL (local):       http://localhost:8000
Base URL (deployed):    https://<your-container-app>.azurecontainerapps.io
```

| Endpoint | Pipeline | Who calls it | When |
|---|---|---|---|
| `POST /score/churn` | Donor Churn | .NET backend | After new donation data is imported |
| `POST /score/impact` | Impact Attribution | .NET backend | After new allocation data is imported |
| `POST /score/residents` | Resident Risk | .NET backend | After health/education/incident records are updated |
| `POST /score/interventions` | Intervention Effectiveness | .NET backend | After process recordings or outcome records are updated |
| `POST /score/social-media` | Social Media Optimization | .NET backend | After new social media posts are logged |
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

## POST /score/interventions

### What it does

Generates per-resident intervention recommendations for all active residents. Loads the trained intervention model from Azure Blob Storage, assigns each resident to a profile cluster, and writes one recommendation per resident to the `intervention_recommendations` table in Supabase.

Recommendations include the optimal service mix, session type, session frequency, and social worker — ranked by measured composite outcome improvement within similar resident profiles. Each recommendation includes a `confidence_tier` based on cluster size.

**Access restricted.** The `intervention_recommendations` table is protected by Supabase row-level security — only users with `Admin` or `SocialWorker` roles can read from it.

### Request

```http
POST /score/interventions
X-API-Key: <key>
```

No request body required.

### Response

```json
{
  "status": "intervention scoring complete",
  "count_scored": 58
}
```

### What gets written to Supabase

Table: `intervention_recommendations` (RLS restricted)

| Column | Type | Description |
|---|---|---|
| `resident_id` | integer | Primary key — references `residents` |
| `profile_cluster` | text | Label describing the resident's profile group |
| `recommended_services` | jsonb | Ordered list of service types by predicted composite improvement |
| `recommended_session_type` | text | `Individual`, `Group`, or `Mixed` |
| `recommended_sessions_per_month` | integer | Recommended session frequency |
| `recommended_social_worker` | text | SW code with highest average outcome improvement for this profile |
| `sw_outcome_score` | float | Average composite outcome improvement for the recommended SW |
| `predicted_health_improvement` | float | Predicted health score delta (may be null if no training data) |
| `predicted_education_improvement` | float | Predicted education progress delta (may be null if no training data) |
| `similar_resident_count` | integer | Number of residents in this profile cluster |
| `confidence_tier` | text | `high` (≥15 similar), `medium` (5–14), or `low` (<5) |
| `top_outcome_factors` | jsonb | Up to 3 human-readable reasons driving the recommendation |
| `scored_at` | timestamptz | When this recommendation was generated |
| `model_version` | text | Timestamp of the model used |

### Confidence tier thresholds

| Tier | Similar residents in cluster |
|---|---|
| `high` | ≥ 15 |
| `medium` | 5 – 14 |
| `low` | < 5 |

### When to call

Call after any of the following events:
- New process recordings are added
- Health, education, or incident records are updated
- A resident's case status or risk level changes

---

## POST /score/social-media

### What it does

Generates posting recommendations for all five platforms (Facebook, Instagram, TikTok, WhatsApp, LinkedIn) in both organic and boosted contexts. Loads three trained models from Azure Blob Storage — an engagement model, a donation referrals model, and a donation value model — and writes one recommendation row per platform × boost status combination to the `social_media_recommendations` table.

Each recommendation specifies what to post, when to post it, and in what tone, optimized for donation conversion rather than raw engagement. A `conversion_signal` flag (`converts`, `noise`, or `balanced`) explicitly labels whether a recommended post type drives donations, drives engagement without donations, or both.

### Request

```http
POST /score/social-media
X-API-Key: <key>
```

No request body required.

### Response

```json
{
  "status": "social media recommendations generated",
  "count_written": 10
}
```

`count_written` is the number of recommendation rows written (up to 10: 5 platforms × 2 boost states).

### What gets written to Supabase

Table: `social_media_recommendations`

| Column | Type | Description |
|---|---|---|
| `recommendation_id` | uuid | Primary key |
| `platform` | text | `Facebook`, `Instagram`, `TikTok`, `WhatsApp`, or `LinkedIn` |
| `is_boosted` | boolean | Whether this recommendation applies to paid/boosted posts |
| `post_type` | text | Recommended post format |
| `media_type` | text | Recommended media type |
| `content_topic` | text | Recommended content topic |
| `sentiment_tone` | text | Recommended tone |
| `has_call_to_action` | boolean | Whether to include a CTA |
| `call_to_action_type` | text | Which CTA type (null if no CTA) |
| `features_resident_story` | boolean | Whether to feature a resident story |
| `best_day_of_week` | text | Day with highest predicted donation referrals for this platform |
| `best_hour` | integer | Hour (0–23) with highest predicted donation referrals |
| `recommended_hashtag_count` | integer | Optimal number of hashtags |
| `predicted_engagement_rate` | float | Model-predicted engagement rate |
| `predicted_donation_referrals` | float | Model-predicted donation referrals |
| `predicted_donation_value_php` | float | Model-predicted donation value (PHP) |
| `conversion_signal` | text | `converts`, `noise`, or `balanced` |
| `sample_count` | integer | Historical posts matching this combination |
| `confidence_tier` | text | `high` (≥30 posts), `medium` (10–29), or `low` (<10) |
| `generated_at` | timestamptz | When this recommendation was generated |
| `model_version` | text | Timestamp of the models used |

### Conversion signal labels

| Signal | Meaning |
|---|---|
| `converts` | Predicted donation referrals ≥ 1.5× platform median — prioritize this |
| `noise` | High predicted engagement but referrals below platform median — avoid over-investing |
| `balanced` | Moderate performance on both signals |

### Confidence tier thresholds

| Tier | Historical posts matching the combination |
|---|---|
| `high` | ≥ 30 |
| `medium` | 10 – 29 |
| `low` | < 10 |

### When to call

Call after new social media posts are logged. Each run deletes the previous set of recommendations and inserts a fresh one — safe to call repeatedly.

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
| Intervention Effectiveness | Monthly (1st) | `train_interventions.py` |
| Social Media Optimization | Monthly (1st) | `train_social_media.py` |

---

## Current Model Limitations

Both the impact attribution and resident risk models are trained on a small dataset (41 active residents, 59 donors). Metrics and scores will become more reliable as data accumulates. The following warnings are expected and logged at each run until the dataset grows:

- Impact: only 1 program area (`Wellbeing`) currently has a statistically significant effect
- Resident risk: AUC and recall are below target thresholds — scores are a weak signal only
- Reintegration: AUC at threshold — readiness scores should be used as supporting context, not definitive assessments

These limitations are logged automatically on each scoring run. No action is required until the resident population exceeds 100 active cases or the donor base expands significantly.
