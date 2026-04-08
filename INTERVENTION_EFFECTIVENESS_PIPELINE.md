# Intervention Effectiveness Pipeline
## Lighthouse Sanctuary — ML Pipeline Plan

> Follows the phase structure from `MASTER_ML_PIPELINE_GUIDE.md`. This pipeline is a clustering + recommendation problem that groups residents by shared profile characteristics and identifies which intervention patterns produced the strongest measurable improvements within each cluster. Output is a per-resident intervention recommendation written to the `intervention_recommendations` table in Supabase.

---

## Repo Placement

```
lighthouse-ml/
├── Dockerfile
├── requirements.txt
├── main.py
├── pipelines/
│   ├── churn/                        ← donor churn (deployed)
│   ├── impact_attribution/           ← impact attribution (deployed)
│   ├── resident_risk/                ← resident risk (deployed)
│   └── interventions/                ← THIS PIPELINE
│       ├── features.py               ← SQL queries + feature engineering
│       ├── cluster_builder.py        ← profile clustering logic
│       ├── train_interventions.py    ← effectiveness model training
│       └── score_interventions.py    ← per-resident scoring + upsert
├── storage/
│   └── blob_client.py                ← shared
└── db/
    └── connection.py                 ← shared
```

---

## Quick-Reference Pipeline Map

```
PHASE 1 — PLAN          Define recommendation problem, cluster strategy, success criteria
PHASE 2 — ACQUIRE       Join residents + process_recordings + health/education/incident tables
PHASE 3 — EXPLORE       Profile distribution, outcome variance by service mix, SW performance spread
PHASE 4 — PREPARE       Compute composite outcome deltas, derive SW roster from recordings, encode profile features
PHASE 5 — MODEL         K-Means / rule-based clustering → within-cluster service effectiveness ranking
PHASE 6 — EVALUATE      Cluster stability (silhouette), outcome lift vs. baseline, SW recommendation coverage
PHASE 7 — INTERPRET     Top outcome factors JSONB extraction → recommendation assembly
PHASE 8 — DEPLOY        Azure Container Apps (inference) + Container Apps Job (training) → Blob Storage → Supabase
```

---

## PHASE 1 — Project Planning (CRISP-DM)

### Feasibility Checklist

| Gate | Question | Status |
|---|---|---|
| Business | What specific problem is being solved? | Match each active resident to the service mix, session type/frequency, and social worker most likely to improve their outcomes — based on what worked for similar residents |
| Data | Is live, updatable data available? | Yes — `process_recordings`, `health_wellbeing_records`, `education_records`, `incident_reports`, and `residents` are all in Supabase |
| Analytical | Can data support reliable recommendations? | Partially — small dataset (~60 active residents) means most clusters will be small; confidence tiers communicate this explicitly |
| Integration | Can outputs plug into existing systems? | Yes — written to `intervention_recommendations` (RLS restricted); .NET API serves to social worker dashboard |
| Risk | Sensitivity concerns? | High — recommendations involve vulnerable minors; outputs are explicitly decision-support only, never automated actions |

### Problem Definition

This is a **clustering + within-cluster effectiveness ranking** problem, not a pure supervised prediction. The goal is to identify which combinations of services, session types, and session frequencies have produced the strongest measurable improvements for residents with a given profile, and surface those patterns as actionable recommendations.

**Unit of analysis:** One row per active resident.

**Clustering dimensions:** Residents are grouped by:
- `case_category` — e.g. `Neglected`, `Trafficked`, `Surrendered`
- Abuse sub-type flags — `sub_cat_physical_abuse`, `sub_cat_sexual_abuse`, `sub_cat_trafficked`, `sub_cat_osaec`, etc.
- `current_risk_level` — `Critical`, `High`, `Medium`, `Low`
- Age band — derived from `present_age`: `Under 10`, `10–13`, `14–17`, `18+`

**Outcome composite (3-month window):**
The composite outcome score for a given intervention period is a weighted average of three signals:
1. Change in `general_health_score` (3 months post recording)
2. Change in `progress_percent` from `education_records` (3 months post recording)
3. Reduction in incident count (3 months post recording — negative change = improvement)

```
composite_score = (Δhealth * 0.40) + (Δeducation * 0.35) + (−Δincidents * 0.25)
```

Weights are based on domain input from Lighthouse staff — adjust in `features.py` if priorities change.

**Social worker availability:** No separate staff table exists. The pipeline derives which social workers are active at each safehouse by looking at which SW codes have conducted sessions for residents at that `safehouse_id` in the last 12 months. Workers with ≥ 3 sessions at a given safehouse are considered available there.

**Model output per resident:**
- Recommended service mix (ordered by predicted composite improvement within cluster)
- Recommended session type (`Individual`, `Group`, or `Mixed`)
- Recommended sessions per month
- Recommended social worker (the SW at their current safehouse with the highest average composite outcome improvement for similar profiles)
- Confidence tier based on cluster size

**Success criteria:**
- ≥ 70% of active residents receive a recommendation with `confidence_tier` of `medium` or `high`
- Composite outcome improvement for `recommended_services` is ≥ 10% higher than the cluster baseline within the training set
- `top_outcome_factors` JSONB field populated for every scored resident

### CRISP-DM Phases

```
Business Understanding → Data Understanding → Data Preparation
       ↑                                             ↓
Deployment ← Evaluation ← Modeling ← ─────────────────
```

### Deliverables Before Proceeding

- [x] Recommendation dimensions agreed (services, session type, frequency, social worker)
- [x] Composite outcome formula agreed (health 40%, education 35%, incidents 25%)
- [x] Confidence tier thresholds agreed (≥15 = high, 5–14 = medium, <5 = low)
- [x] Data sources confirmed (Supabase PostgreSQL, direct connection)
- [x] Integration path confirmed (FastAPI → Supabase `intervention_recommendations`, RLS restricted)
- [ ] Supabase connection string obtained and stored securely in `.env`

---

## PHASE 2 — Data Acquisition

> Reference: `chapter_four_implementation.md`

### Connection

```python
import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("SUPABASE_DB_URL"))
```

### Core Query — Residents with Profile Features

```python
RESIDENTS_QUERY = """
SELECT
    r.resident_id,
    r.safehouse_id,
    r.case_status,
    r.case_category,
    r.current_risk_level,
    r.present_age,

    -- Abuse sub-type flags
    r.sub_cat_physical_abuse,
    r.sub_cat_sexual_abuse,
    r.sub_cat_trafficked,
    r.sub_cat_osaec,
    r.sub_cat_child_labor,
    r.sub_cat_cicl,
    r.sub_cat_at_risk,
    r.sub_cat_orphaned,
    r.sub_cat_street_child,
    r.sub_cat_child_with_hiv,

    -- Reintegration context
    r.reintegration_type,
    r.reintegration_status,
    r.initial_risk_level,
    r.length_of_stay

FROM residents r
WHERE r.case_status = 'Active'
"""
```

### Core Query — Process Recordings (Session History)

```python
RECORDINGS_QUERY = """
SELECT
    pr.recording_id,
    pr.resident_id,
    pr.session_date,
    pr.social_worker,
    pr.session_type,
    pr.session_duration_minutes,
    pr.interventions_applied,
    pr.progress_noted,
    pr.concerns_flagged,
    r.safehouse_id
FROM process_recordings pr
JOIN residents r ON pr.resident_id = r.resident_id
ORDER BY pr.resident_id, pr.session_date
"""
```

### Core Query — Outcome Tables

```python
HEALTH_QUERY = """
SELECT
    resident_id,
    record_date,
    general_health_score
FROM health_wellbeing_records
ORDER BY resident_id, record_date
"""

EDUCATION_QUERY = """
SELECT
    resident_id,
    record_date,
    progress_percent
FROM education_records
ORDER BY resident_id, record_date
"""

INCIDENTS_QUERY = """
SELECT
    resident_id,
    incident_date,
    severity,
    resolved
FROM incident_reports
ORDER BY resident_id, incident_date
"""
```

### Loading All Tables

```python
def load_all_data(engine):
    residents_df     = pd.read_sql(RESIDENTS_QUERY, engine)
    recordings_df    = pd.read_sql(RECORDINGS_QUERY, engine)
    health_df        = pd.read_sql(HEALTH_QUERY, engine)
    education_df     = pd.read_sql(EDUCATION_QUERY, engine)
    incidents_df     = pd.read_sql(INCIDENTS_QUERY, engine)
    return residents_df, recordings_df, health_df, education_df, incidents_df
```

---

## PHASE 3 — Exploratory Data Analysis

### Sanity Checks

```python
from utils.stats import unistats

for name, df in [
    ("residents", residents_df),
    ("recordings", recordings_df),
    ("health", health_df),
    ("education", education_df),
    ("incidents", incidents_df),
]:
    print(f"\n{'='*50}\n{name.upper()}\n{'='*50}")
    print(unistats(df))
```

### Profile Distribution

How many residents fall into each `case_category` × `current_risk_level` cell? Cells with fewer than 5 residents will collapse into an "Other" cluster at Phase 4.

```python
profile_dist = residents_df.groupby(
    ["case_category", "current_risk_level"]
).size().reset_index(name="count")
print(profile_dist.sort_values("count", ascending=False))
```

### Social Worker Roster Derivation

Derive which social workers are available at each safehouse from recording history.

```python
import pandas as pd

def derive_sw_roster(recordings_df, months_lookback=12):
    """
    Returns a DataFrame of (safehouse_id, social_worker, session_count)
    for SWs with >= 3 sessions at a safehouse in the last N months.
    """
    cutoff = pd.Timestamp.now() - pd.DateOffset(months=months_lookback)
    recent = recordings_df[recordings_df["session_date"] >= cutoff]
    roster = (
        recent.groupby(["safehouse_id", "social_worker"])
        .size()
        .reset_index(name="session_count")
    )
    return roster[roster["session_count"] >= 3]
```

### Outcome Variance by Service Type

Before modelling, inspect whether different `interventions_applied` values produce different average health outcomes. This is a sanity check, not causal analysis.

```python
# Merge recordings with health outcomes within 3-month window
# (See Phase 4 for full delta computation)
service_outcomes = (
    merged_df.groupby("interventions_applied")["composite_score_delta"]
    .agg(["mean", "std", "count"])
    .sort_values("mean", ascending=False)
)
print(service_outcomes)
```

### Session Type vs. Outcome

```python
session_outcomes = (
    merged_df.groupby("session_type")["composite_score_delta"]
    .agg(["mean", "std", "count"])
)
print(session_outcomes)
```

---

## PHASE 4 — Data Preparation

### Step 1 — Compute 3-Month Outcome Deltas per Resident

For each resident, find outcome values at a recording date and again ~90 days later. The delta is the change in the composite score over that window.

```python
def compute_health_delta(health_df, session_date, resident_id, window_days=90):
    """
    Returns delta in general_health_score between session_date and
    session_date + window_days. Returns None if no record found.
    """
    resident_health = health_df[health_df["resident_id"] == resident_id].copy()
    resident_health["record_date"] = pd.to_datetime(resident_health["record_date"])
    session_date = pd.to_datetime(session_date)

    baseline = resident_health[resident_health["record_date"] <= session_date]
    followup = resident_health[
        (resident_health["record_date"] > session_date) &
        (resident_health["record_date"] <= session_date + pd.Timedelta(days=window_days))
    ]
    if baseline.empty or followup.empty:
        return None
    return followup["general_health_score"].mean() - baseline["general_health_score"].iloc[-1]


def compute_composite_delta(resident_id, session_date, health_df, education_df, incidents_df,
                             w_health=0.40, w_edu=0.35, w_incidents=0.25, window_days=90):
    delta_health = compute_health_delta(health_df, session_date, resident_id, window_days)
    delta_edu    = compute_edu_delta(education_df, session_date, resident_id, window_days)
    delta_inc    = compute_incident_delta(incidents_df, session_date, resident_id, window_days)

    components = [(delta_health, w_health), (delta_edu, w_edu), (-delta_inc if delta_inc is not None else None, w_incidents)]
    valid = [(v, w) for v, w in components if v is not None]
    if not valid:
        return None
    total_weight = sum(w for _, w in valid)
    return sum(v * w for v, w in valid) / total_weight
```

### Step 2 — Aggregate Session Features per Resident-Period

For each resident, summarize the intervention window:
- Dominant `interventions_applied` (most frequent service in the window)
- Total session count and session type mix
- Primary social worker (most sessions in window)

```python
def summarize_intervention_window(recordings_window):
    """
    recordings_window: DataFrame of recordings within a time window for one resident.
    Returns dict of summarized features.
    """
    services = recordings_window["interventions_applied"].value_counts()
    session_types = recordings_window["session_type"].value_counts()
    sw_counts = recordings_window["social_worker"].value_counts()

    return {
        "primary_service": services.index[0] if not services.empty else None,
        "session_count": len(recordings_window),
        "sessions_per_month": len(recordings_window) / 3,  # 3-month window
        "dominant_session_type": session_types.index[0] if not session_types.empty else None,
        "primary_sw": sw_counts.index[0] if not sw_counts.empty else None,
    }
```

### Step 3 — Profile Feature Engineering

```python
def build_age_band(age_str):
    """
    Converts age strings like '15 Years 9 months' to a band label.
    Falls back gracefully if parsing fails.
    """
    try:
        years = int(str(age_str).split()[0])
        if years < 10:
            return "Under 10"
        elif years <= 13:
            return "10–13"
        elif years <= 17:
            return "14–17"
        else:
            return "18+"
    except (ValueError, IndexError):
        return "Unknown"


def build_abuse_flag_string(row):
    """
    Creates a short human-readable label from sub-category flags.
    Used in profile_cluster label generation.
    """
    flags = []
    flag_cols = [
        ("sub_cat_physical_abuse", "Physical Abuse"),
        ("sub_cat_sexual_abuse", "Sexual Abuse"),
        ("sub_cat_trafficked", "Trafficked"),
        ("sub_cat_osaec", "OSAEC"),
        ("sub_cat_child_labor", "Child Labor"),
        ("sub_cat_cicl", "CICL"),
    ]
    for col, label in flag_cols:
        if row.get(col) is True or row.get(col) == "True":
            flags.append(label)
    return " · ".join(flags) if flags else "No Sub-type"


def build_profile_label(row):
    risk = row["current_risk_level"]
    abuse = build_abuse_flag_string(row)
    age = build_age_band(row["present_age"])
    return f"{risk} · {abuse} · Age {age}"
```

### Step 4 — Encode Profile Features for Clustering

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

CATEGORICAL_PROFILE_COLS = ["case_category", "current_risk_level", "age_band"]
BINARY_FLAG_COLS = [
    "sub_cat_physical_abuse", "sub_cat_sexual_abuse", "sub_cat_trafficked",
    "sub_cat_osaec", "sub_cat_child_labor", "sub_cat_cicl",
    "sub_cat_at_risk", "sub_cat_orphaned",
]

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_PROFILE_COLS),
    ("bin", "passthrough", BINARY_FLAG_COLS),
])
```

---

## PHASE 5 — Modelling

### Clustering Strategy

With ~60 active residents and 4 profile dimensions, a pure K-Means approach may produce unstable clusters. The preferred approach is **rule-based hierarchical grouping**: segment first by `case_category`, then by `current_risk_level`, then by age band. This produces deterministic, human-interpretable clusters that social workers can reason about.

If a cluster has fewer than 5 members, collapse it into the nearest parent group and note this in `profile_cluster`.

```python
def assign_cluster(row, min_cluster_size=5, cluster_counts=None):
    """
    Rule-based cluster assignment: category → risk level → age band.
    Falls back to broader groups when cells are too small.
    """
    full_label = build_profile_label(row)
    if cluster_counts and cluster_counts.get(full_label, 0) >= min_cluster_size:
        return full_label

    # Fall back: drop age band
    partial_label = f"{row['current_risk_level']} · {build_abuse_flag_string(row)}"
    if cluster_counts and cluster_counts.get(partial_label, 0) >= min_cluster_size:
        return partial_label

    # Final fallback: risk level only
    return row["current_risk_level"]
```

### Within-Cluster Effectiveness Ranking

For each cluster, rank service combinations by average composite outcome improvement:

```python
def rank_services_within_cluster(cluster_df):
    """
    cluster_df: rows of residents in a cluster with their intervention windows and composite deltas.
    Returns ordered list of service types by mean composite improvement.
    """
    service_perf = (
        cluster_df.groupby("primary_service")["composite_score_delta"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    return service_perf["primary_service"].tolist()


def rank_session_type_within_cluster(cluster_df):
    return (
        cluster_df.groupby("dominant_session_type")["composite_score_delta"]
        .mean()
        .idxmax()
    )


def optimal_sessions_per_month(cluster_df):
    """
    Find the session frequency band (1–3, 4–6, 7+) with the highest mean composite delta.
    """
    cluster_df = cluster_df.copy()
    cluster_df["freq_band"] = pd.cut(
        cluster_df["sessions_per_month"],
        bins=[0, 3, 6, 99],
        labels=["1–3", "4–6", "7+"]
    )
    return (
        cluster_df.groupby("freq_band")["composite_score_delta"]
        .mean()
        .idxmax()
    )
```

### Social Worker Outcome Scoring

Score each social worker at a safehouse by their average composite outcome improvement for residents matching a given cluster:

```python
def score_social_workers(cluster_df, sw_roster, safehouse_id):
    """
    Returns the SW code with the highest mean composite outcome improvement
    for this cluster at the given safehouse, drawn only from active roster.
    """
    available_sws = sw_roster[sw_roster["safehouse_id"] == safehouse_id]["social_worker"].tolist()
    sw_perf = (
        cluster_df[cluster_df["primary_sw"].isin(available_sws)]
        .groupby("primary_sw")["composite_score_delta"]
        .agg(["mean", "count"])
        .reset_index()
    )
    sw_perf.columns = ["social_worker", "mean_outcome", "session_count"]
    if sw_perf.empty:
        return None, None
    best = sw_perf.sort_values("mean_outcome", ascending=False).iloc[0]
    return best["social_worker"], round(best["mean_outcome"], 3)
```

### Training Script — `train_interventions.py`

```python
import pickle
import json
from datetime import datetime
from storage.blob_client import upload_artifact
from db.connection import engine
from pipelines.interventions.features import (
    load_all_data, build_training_frame, derive_sw_roster
)
from pipelines.interventions.cluster_builder import (
    assign_cluster, rank_services_within_cluster,
    rank_session_type_within_cluster, optimal_sessions_per_month,
    score_social_workers
)

def train():
    residents_df, recordings_df, health_df, education_df, incidents_df = load_all_data(engine)
    sw_roster = derive_sw_roster(recordings_df)
    training_df = build_training_frame(residents_df, recordings_df, health_df, education_df, incidents_df)

    # Assign clusters
    cluster_counts = training_df["profile_cluster"].value_counts().to_dict()
    training_df["profile_cluster"] = training_df.apply(
        lambda row: assign_cluster(row, cluster_counts=cluster_counts), axis=1
    )

    # Build cluster-level recommendations
    cluster_recommendations = {}
    for cluster_label, cluster_df in training_df.groupby("profile_cluster"):
        cluster_recommendations[cluster_label] = {
            "recommended_services": rank_services_within_cluster(cluster_df),
            "recommended_session_type": rank_session_type_within_cluster(cluster_df),
            "sessions_per_month": optimal_sessions_per_month(cluster_df),
            "cluster_size": len(cluster_df),
        }

    model_version = datetime.now().strftime("%Y%m%d_%H%M")
    artifact = {
        "model_version": model_version,
        "cluster_recommendations": cluster_recommendations,
        "sw_roster": sw_roster.to_dict(orient="records"),
    }

    upload_artifact("interventions_model.pkl", pickle.dumps(artifact))
    print(f"Trained intervention model — {len(cluster_recommendations)} clusters | version {model_version}")

if __name__ == "__main__":
    train()
```

---

## PHASE 6 — Evaluation

### Cluster Stability

Check that cluster assignments are stable by comparing the current run to the previous run:

```python
# Compare cluster_label distribution across runs
current_dist = training_df["profile_cluster"].value_counts()
print(current_dist)
```

Clusters with fewer than 5 members will always be `low` confidence — this is expected at current dataset size and should be logged, not treated as an error.

### Outcome Lift

The primary evaluation question: do the recommended service combinations actually produce better composite outcomes than the cluster average?

```python
def compute_recommendation_lift(training_df, cluster_recommendations):
    """
    For each cluster, compute the mean composite delta for residents who received
    the recommended service vs. all others in the cluster.
    """
    results = []
    for cluster, recs in cluster_recommendations.items():
        cluster_df = training_df[training_df["profile_cluster"] == cluster]
        top_service = recs["recommended_services"][0] if recs["recommended_services"] else None
        if top_service is None:
            continue
        recommended = cluster_df[cluster_df["primary_service"] == top_service]["composite_score_delta"].mean()
        others = cluster_df[cluster_df["primary_service"] != top_service]["composite_score_delta"].mean()
        results.append({
            "cluster": cluster,
            "recommended_service": top_service,
            "recommended_mean_delta": round(recommended, 4),
            "others_mean_delta": round(others, 4),
            "lift_pct": round((recommended - others) / max(abs(others), 0.001) * 100, 1)
        })
    return pd.DataFrame(results)
```

### Coverage Check

```python
coverage = scored_df["confidence_tier"].value_counts(normalize=True)
print(coverage)
# Target: >= 70% at medium or high confidence
```

### Expected Warnings at Current Dataset Size

The following are expected and should be logged, not raised as errors, until the resident population grows:

- Most clusters will be `low` or `medium` confidence (< 15 similar residents)
- SW outcome scores are based on small per-worker sample sizes — treat as indicative, not definitive
- Composite outcome delta computation will return `None` for residents without follow-up records in the 3-month window; these residents are still scored using cluster-level recommendations

---

## PHASE 7 — Interpretation

### Top Outcome Factors JSONB

The `top_outcome_factors` field gives social workers a plain-language explanation of what is driving the recommendation:

```python
def build_top_outcome_factors(cluster_label, recommended_services, recommended_session_type,
                               sessions_per_month, recommended_sw, sw_outcome_score, cluster_size):
    """
    Returns up to 3 human-readable factors explaining the recommendation.
    """
    factors = []
    if recommended_services:
        service_name = recommended_services[0]
        factors.append(
            f"{service_name} sessions show the strongest composite improvements "
            f"for residents in the '{cluster_label}' profile"
        )
    if recommended_session_type and sessions_per_month:
        factors.append(
            f"{recommended_session_type} sessions at ~{sessions_per_month} per month "
            f"outperform other formats for this profile"
        )
    if recommended_sw and sw_outcome_score is not None:
        factors.append(
            f"{recommended_sw} has the highest average composite outcome improvement "
            f"for similar profiles at this safehouse (score: {sw_outcome_score:.2f})"
        )
    if not factors:
        factors.append(f"Based on {cluster_size} similar residents in the '{cluster_label}' cluster")
    return factors[:3]
```

### Confidence Tier Logic

```python
def assign_confidence_tier(similar_count):
    if similar_count >= 15:
        return "high"
    elif similar_count >= 5:
        return "medium"
    else:
        return "low"
```

---

## PHASE 8 — Deployment

### Scoring Script — `score_interventions.py`

```python
import pickle
import uuid
from datetime import datetime, timezone
import pandas as pd
from supabase import create_client
from storage.blob_client import download_artifact
from db.connection import engine
from pipelines.interventions.features import load_all_data, derive_sw_roster
from pipelines.interventions.cluster_builder import (
    assign_cluster, build_profile_label, build_top_outcome_factors, assign_confidence_tier
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

def score():
    artifact = pickle.loads(download_artifact("interventions_model.pkl"))
    cluster_recommendations = artifact["cluster_recommendations"]
    sw_roster_df = pd.DataFrame(artifact["sw_roster"])
    model_version = artifact["model_version"]

    residents_df, recordings_df, _, _, _ = load_all_data(engine)
    active_residents = residents_df[residents_df["case_status"] == "Active"].copy()

    # Add derived profile features
    active_residents["age_band"] = active_residents["present_age"].apply(build_age_band)
    cluster_counts = {k: v["cluster_size"] for k, v in cluster_recommendations.items()}
    active_residents["profile_cluster"] = active_residents.apply(
        lambda row: assign_cluster(row, cluster_counts=cluster_counts), axis=1
    )

    records = []
    for _, resident in active_residents.iterrows():
        cluster_label = resident["profile_cluster"]
        safehouse_id = resident["safehouse_id"]
        recs = cluster_recommendations.get(cluster_label, {})
        similar_count = recs.get("cluster_size", 0)

        recommended_sw, sw_outcome_score = score_social_workers(
            cluster_df=None,  # SW scoring uses roster only in scoring mode
            sw_roster=sw_roster_df,
            safehouse_id=safehouse_id,
        )

        sessions_label = recs.get("sessions_per_month", "4–6")
        sessions_int = {"1–3": 2, "4–6": 5, "7+": 8}.get(sessions_label, 5)

        # Predicted improvements are stored from training; fall back to None if unavailable
        predicted_health = recs.get("predicted_health_improvement")
        predicted_edu = recs.get("predicted_education_improvement")

        factors = build_top_outcome_factors(
            cluster_label, recs.get("recommended_services", []),
            recs.get("recommended_session_type"), sessions_label,
            recommended_sw, sw_outcome_score, similar_count
        )

        records.append({
            "resident_id": int(resident["resident_id"]),
            "profile_cluster": cluster_label,
            "recommended_services": recs.get("recommended_services", []),
            "recommended_session_type": recs.get("recommended_session_type"),
            "recommended_sessions_per_month": sessions_int,
            "recommended_social_worker": recommended_sw,
            "sw_outcome_score": sw_outcome_score,
            "predicted_health_improvement": predicted_health,
            "predicted_education_improvement": predicted_edu,
            "similar_resident_count": similar_count,
            "confidence_tier": assign_confidence_tier(similar_count),
            "top_outcome_factors": factors,
            "scored_at": datetime.now(timezone.utc).isoformat(),
            "model_version": model_version,
        })

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    supabase.table("intervention_recommendations").upsert(records, on_conflict="resident_id").execute()
    print(f"Intervention scoring complete — {len(records)} residents scored")
    return len(records)
```

### FastAPI Endpoint — `main.py`

```python
@app.post("/score/interventions")
def score_interventions(x_api_key: str = Header(...)):
    verify_api_key(x_api_key)
    count = score()
    return {"status": "intervention scoring complete", "count_scored": count}
```

### Supabase Table Schema

```sql
CREATE TABLE intervention_recommendations (
    resident_id                     INTEGER PRIMARY KEY REFERENCES residents(resident_id),
    profile_cluster                 TEXT,
    recommended_services            JSONB,
    recommended_session_type        TEXT,
    recommended_sessions_per_month  INTEGER,
    recommended_social_worker       TEXT,
    sw_outcome_score                FLOAT,
    predicted_health_improvement    FLOAT,
    predicted_education_improvement FLOAT,
    similar_resident_count          INTEGER,
    confidence_tier                 TEXT,
    top_outcome_factors             JSONB,
    scored_at                       TIMESTAMPTZ,
    model_version                   TEXT
);

-- Row-level security: Admin and SocialWorker roles only
ALTER TABLE intervention_recommendations ENABLE ROW LEVEL SECURITY;
```

### Azure Container Apps Job — `train_interventions.py`

Scheduled via Azure Container Apps Job. Run on the 1st of each month.

```yaml
# containerapp-job-interventions.yaml (excerpt)
schedule: "0 2 1 * *"   # 2:00 AM UTC on the 1st of each month
replicaTimeout: 1800
```

### Blob Storage Artifact

| Artifact | Key | Contents |
|---|---|---|
| Trained model | `interventions_model.pkl` | Serialized dict: `cluster_recommendations`, `sw_roster`, `model_version` |

### Environment Variables Required

| Variable | Purpose |
|---|---|
| `SUPABASE_DB_URL` | Direct PostgreSQL connection for training queries |
| `SUPABASE_URL` | REST API base URL for upsert operations |
| `SUPABASE_SERVICE_ROLE_KEY` | Service role key for bypassing RLS during write |
| `AZURE_STORAGE_CONNECTION_STRING` | Blob Storage access for model artifact |
| `ML_SERVICE_API_KEY` | Shared secret for `/score/interventions` endpoint |

---

## Current Model Limitations

- **Small dataset:** ~60 active residents means most clusters will be `low` or `medium` confidence. Recommendations will become meaningfully more reliable once the resident population exceeds ~100 active cases.
- **SW outcome scores:** Based on small per-worker sample sizes; treat as directional guidance, not statistically robust rankings.
- **No causal identification:** Composite outcome improvements are correlational — they reflect what happened to similar residents after similar interventions, not a controlled experiment. Social workers should use recommendations as structured decision support, not prescriptions.
- **Roster inference limitation:** Social worker availability is inferred from recording history. New workers who haven't yet built a recording history at a safehouse will not appear in roster recommendations until they have ≥ 3 sessions logged.
- **Incident severity not weighted:** The incident delta currently treats all incident types equally. Future versions should weight high-severity incidents more heavily.
