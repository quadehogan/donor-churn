# Social Media Optimization Pipeline
## Lighthouse Sanctuary — ML Pipeline Plan

> Follows the phase structure from `MASTER_ML_PIPELINE_GUIDE.md`. This pipeline trains two models on historical `social_media_posts` data — an engagement model and a conversion model — and writes per-platform posting recommendations to `social_media_recommendations` in Supabase. The central goal is to help a non-technical social media operator know exactly what to post, when, and in what tone to maximize donation outcomes, not just likes.

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
│   ├── interventions/                ← intervention effectiveness (deployed)
│   └── social_media/                 ← THIS PIPELINE
│       ├── features.py               ← SQL query + feature engineering
│       ├── train_social_media.py     ← train engagement + conversion models
│       └── score_social_media.py     ← generate recommendations + upsert
├── storage/
│   └── blob_client.py                ← shared
└── db/
    └── connection.py                 ← shared
```

---

## Quick-Reference Pipeline Map

```
PHASE 1 — PLAN          Define engagement vs. conversion targets, platforms, output format
PHASE 2 — ACQUIRE       Pull social_media_posts via Supabase direct PostgreSQL
PHASE 3 — EXPLORE       Platform distribution, engagement vs. conversion divergence, feature correlations
PHASE 4 — PREPARE       Feature encoding, train/test split, handle class imbalance (low-post platforms)
PHASE 5 — MODEL         RandomForest for engagement rate; GradientBoosting for donation_referrals + value
PHASE 6 — EVALUATE      R² / MAE for both models; noise vs. converts signal analysis
PHASE 7 — INTERPRET     Feature importance extraction → recommendation assembly → noise flagging
PHASE 8 — DEPLOY        Azure Container Apps (inference) + Container Apps Job (training) → Blob Storage → Supabase
```

---

## PHASE 1 — Project Planning (CRISP-DM)

### Feasibility Checklist

| Gate | Question | Status |
|---|---|---|
| Business | What specific problem is being solved? | Tell staff exactly what to post, when, and in what tone to maximize donation outcomes — not just engagement |
| Data | Is live, updatable data available? | Yes — `social_media_posts` table in Supabase, updated as posts are logged |
| Analytical | Can data support reliable predictions? | Partially — engagement is well-supported; conversion model depends on `donation_referrals` quality and volume per platform |
| Integration | Can outputs plug into existing systems? | Yes — recommendations written to `social_media_recommendations`; .NET API reads and serves to content dashboard |
| Risk | Privacy / sensitivity concerns? | Low — all post data is public-facing; no resident PII in post features |

### Problem Definition

This pipeline trains **two separate regression models** per platform and computes a per-platform recommendation for both organic and paid/boosted posting contexts:

**Model 1 — Engagement model:**
Predicts `engagement_rate` from post feature combinations. Engagement is a useful reach signal but is explicitly not the primary optimization target.

**Model 2 — Conversion model:**
Predicts `donation_referrals` and `estimated_donation_value_php` from the same features. This is the primary optimization target.

**The gap between the two models is the key insight:**
Post combinations with high predicted engagement but low predicted donation referrals are flagged as `noise`. Post combinations that convert strongly, even with moderate engagement, are flagged as `converts`. The `balanced` label indicates strong performance on both signals.

**Organic vs. boosted split:**
The pipeline produces two rows per platform — one for organic posting and one for paid/boosted. These are optimized separately because feature importance shifts meaningfully between contexts (boosted posts are less sensitive to timing; organic posts are more sensitive to content topic and tone).

**Output per platform per boost status:**
- Recommended post type, media type, content topic, sentiment tone
- Whether to include a CTA and which type
- Whether to feature a resident story
- Best day of week and hour
- Recommended hashtag count
- Predicted engagement rate and donation metrics
- `conversion_signal` flag

**Success criteria:**
- R² ≥ 0.50 for engagement model on held-out test set
- R² ≥ 0.40 for conversion model on held-out test set (lower threshold given noisier signal)
- All 5 platforms receive at least one recommendation (even if `low` confidence)
- `conversion_signal` labels are internally consistent — `converts` rows must have `predicted_donation_referrals` at least 1.5× the platform median

### CRISP-DM Phases

```
Business Understanding → Data Understanding → Data Preparation
       ↑                                             ↓
Deployment ← Evaluation ← Modeling ← ─────────────────
```

### Deliverables Before Proceeding

- [x] Dual-model approach agreed (engagement + conversion modelled separately)
- [x] Organic vs. boosted split agreed (two rows per platform)
- [x] Noise / converts flagging logic agreed (conversion_signal column)
- [x] Data source confirmed (Supabase `social_media_posts`, direct PostgreSQL connection)
- [x] Integration path confirmed (FastAPI → Supabase `social_media_recommendations`)
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

### Core Query — Social Media Posts

Pull all columns needed for both models in a single query. No joins required — all features and outcomes are in `social_media_posts`.

```python
SOCIAL_MEDIA_QUERY = """
SELECT
    post_id,
    platform,
    day_of_week,
    post_hour,
    post_type,
    media_type,
    content_topic,
    sentiment_tone,
    has_call_to_action,
    call_to_action_type,
    features_resident_story,
    num_hashtags,
    caption_length,
    is_boosted,

    -- Outcome targets
    engagement_rate,
    donation_referrals,
    estimated_donation_value_php,

    -- Context / metadata
    platform_post_id,
    created_at
FROM social_media_posts
ORDER BY created_at
"""

def load_posts(engine):
    df = pd.read_sql(SOCIAL_MEDIA_QUERY, engine)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df
```

---

## PHASE 3 — Exploratory Data Analysis

### Sanity Checks

```python
from utils.stats import unistats

print(unistats(posts_df))
print("\nPost count by platform:")
print(posts_df["platform"].value_counts())
print("\nPost count by platform + is_boosted:")
print(posts_df.groupby(["platform", "is_boosted"]).size())
```

### Engagement vs. Conversion Divergence

This is the most important exploratory step — quantify how strongly engagement and conversion diverge across post types. A large gap signals that surface-level optimization (chasing likes) would actively hurt donation outcomes.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Normalize both metrics to 0–1 for comparison
posts_df["engagement_norm"] = (
    posts_df["engagement_rate"] - posts_df["engagement_rate"].min()
) / posts_df["engagement_rate"].ptp()

posts_df["conversion_norm"] = (
    posts_df["donation_referrals"] - posts_df["donation_referrals"].min()
) / posts_df["donation_referrals"].ptp()

# Flag potential noise posts
posts_df["is_noise"] = (
    (posts_df["engagement_norm"] > 0.6) & (posts_df["conversion_norm"] < 0.3)
)

print(f"Potential noise posts: {posts_df['is_noise'].sum()} "
      f"({posts_df['is_noise'].mean()*100:.1f}% of dataset)")

divergence = posts_df.groupby("post_type")[["engagement_norm", "conversion_norm"]].mean()
print(divergence.sort_values("engagement_norm", ascending=False))
```

### Feature Correlation with Targets

```python
FEATURE_COLS = [
    "post_type", "media_type", "content_topic", "sentiment_tone",
    "has_call_to_action", "call_to_action_type", "day_of_week", "post_hour",
    "features_resident_story", "num_hashtags", "caption_length"
]

# Spearman correlations with each target
for target in ["engagement_rate", "donation_referrals", "estimated_donation_value_php"]:
    corr = posts_df[FEATURE_COLS + [target]].corr(method="spearman")[target].drop(target)
    print(f"\nSpearman correlation with {target}:")
    print(corr.sort_values(key=abs, ascending=False).head(10))
```

### Platform-Level Baseline Stats

```python
platform_stats = posts_df.groupby("platform").agg(
    post_count=("post_id", "count"),
    mean_engagement=("engagement_rate", "mean"),
    mean_referrals=("donation_referrals", "mean"),
    mean_value_php=("estimated_donation_value_php", "mean"),
).round(4)
print(platform_stats)
```

---

## PHASE 4 — Data Preparation

### Step 1 — Handle Missing Values

```python
def clean_posts(df):
    # call_to_action_type is null when has_call_to_action is False — fill with "None"
    df["call_to_action_type"] = df["call_to_action_type"].fillna("None")

    # Platforms with zero boost_budget_php should be treated as organic
    df["is_boosted"] = df["is_boosted"].fillna(False)

    # Drop rows where both engagement_rate and donation_referrals are null
    df = df.dropna(subset=["engagement_rate", "donation_referrals"])

    return df
```

### Step 2 — Feature Encoding

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

CATEGORICAL_FEATURES = [
    "platform", "post_type", "media_type", "content_topic",
    "sentiment_tone", "call_to_action_type", "day_of_week",
]
BINARY_FEATURES   = ["has_call_to_action", "features_resident_story", "is_boosted"]
NUMERIC_FEATURES  = ["post_hour", "num_hashtags", "caption_length"]

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ("bin", "passthrough", BINARY_FEATURES),
    ("num", "passthrough", NUMERIC_FEATURES),
])
```

### Step 3 — Train/Test Split

Use a time-based split (not random) to prevent data leakage — we never want future posts to inform the recommendation for past posts.

```python
from sklearn.model_selection import train_test_split

# Sort by date and use the last 20% of posts as the holdout test set
posts_df = posts_df.sort_values("created_at")
split_idx = int(len(posts_df) * 0.80)
train_df = posts_df.iloc[:split_idx]
test_df  = posts_df.iloc[split_idx:]

FEATURE_COLS_MODEL = CATEGORICAL_FEATURES + BINARY_FEATURES + NUMERIC_FEATURES

X_train = train_df[FEATURE_COLS_MODEL]
X_test  = test_df[FEATURE_COLS_MODEL]

y_train_engagement  = train_df["engagement_rate"]
y_test_engagement   = test_df["engagement_rate"]

y_train_referrals   = train_df["donation_referrals"]
y_test_referrals    = test_df["donation_referrals"]

y_train_value       = train_df["estimated_donation_value_php"]
y_test_value        = test_df["estimated_donation_value_php"]
```

---

## PHASE 5 — Modelling

### Model 1 — Engagement Rate (RandomForestRegressor)

RandomForest is preferred here over linear models because engagement rate has a non-linear relationship with timing and content combinations.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

engagement_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    ))
])

engagement_pipeline.fit(X_train, y_train_engagement)
```

### Model 2 — Donation Referrals (GradientBoostingRegressor)

GradientBoosting is preferred for the conversion model because it handles the skewed, count-like distribution of `donation_referrals` more robustly than RandomForest at small sample sizes.

```python
from sklearn.ensemble import GradientBoostingRegressor

referrals_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", GradientBoostingRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=5,
        random_state=42,
    ))
])

referrals_pipeline.fit(X_train, y_train_referrals)
```

### Model 3 — Estimated Donation Value (GradientBoostingRegressor)

```python
value_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", GradientBoostingRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=5,
        random_state=42,
    ))
])

value_pipeline.fit(X_train, y_train_value)
```

### Training Script — `train_social_media.py`

```python
import pickle
from datetime import datetime
from storage.blob_client import upload_artifact
from db.connection import engine
from pipelines.social_media.features import load_posts, clean_posts

def train():
    posts_df = clean_posts(load_posts(engine))
    posts_df = posts_df.sort_values("created_at")
    split_idx = int(len(posts_df) * 0.80)
    train_df = posts_df.iloc[:split_idx]
    test_df  = posts_df.iloc[split_idx:]

    X_train = train_df[FEATURE_COLS_MODEL]
    X_test  = test_df[FEATURE_COLS_MODEL]

    engagement_pipeline.fit(X_train, train_df["engagement_rate"])
    referrals_pipeline.fit(X_train, train_df["donation_referrals"])
    value_pipeline.fit(X_train, train_df["estimated_donation_value_php"])

    model_version = datetime.now().strftime("%Y%m%d_%H%M")
    artifact = {
        "engagement_model": engagement_pipeline,
        "referrals_model": referrals_pipeline,
        "value_model": value_pipeline,
        "model_version": model_version,
        "platform_medians": posts_df.groupby("platform")["donation_referrals"].median().to_dict(),
    }
    upload_artifact("social_media_model.pkl", pickle.dumps(artifact))
    print(f"Social media models trained | version {model_version}")

if __name__ == "__main__":
    train()
```

---

## PHASE 6 — Evaluation

### Regression Metrics — Both Models

```python
from sklearn.metrics import r2_score, mean_absolute_error

for name, pipeline, y_test in [
    ("Engagement", engagement_pipeline, y_test_engagement),
    ("Referrals",  referrals_pipeline,  y_test_referrals),
    ("Value PHP",  value_pipeline,      y_test_value),
]:
    y_pred = pipeline.predict(X_test)
    print(f"\n{name} Model:")
    print(f"  R²:  {r2_score(y_test, y_pred):.4f}   (target ≥ 0.50 engagement / ≥ 0.40 conversion)")
    print(f"  MAE: {mean_absolute_error(y_test, y_pred):.4f}")
```

### Noise vs. Converts Validation

After scoring, verify that `conversion_signal` labels are internally consistent:

```python
signal_summary = recs_df.groupby("conversion_signal").agg(
    count=("recommendation_id", "count"),
    mean_engagement=("predicted_engagement_rate", "mean"),
    mean_referrals=("predicted_donation_referrals", "mean"),
).round(4)
print(signal_summary)
# Expected: 'noise' rows have higher engagement but lower referrals than 'converts' rows
```

### Platform Coverage Check

```python
coverage = recs_df.groupby(["platform", "is_boosted"]).size().reset_index(name="count")
print(coverage)
# Expected: all 5 platforms × 2 boost states = up to 10 rows
```

### Expected Warnings at Current Dataset Size

- LinkedIn and WhatsApp recommendations will be `low` confidence until posting history on those platforms grows (target: ≥ 30 posts per platform per feature combination for `high` confidence)
- The conversion model is sensitive to seasonal donation patterns — it should be retrained after any major campaign to avoid recency bias
- Platforms with fewer than 10 posts will produce recommendations based on very sparse data; this will be reflected in `sample_count` and `confidence_tier`

---

## PHASE 7 — Interpretation

### Feature Importance Extraction

```python
def get_feature_importances(pipeline, feature_cols):
    """
    Returns a sorted DataFrame of feature importances from the trained model.
    Works with RandomForest and GradientBoosting estimators.
    """
    encoder = pipeline.named_steps["prep"]
    model   = pipeline.named_steps["model"]

    ohe_feature_names = encoder.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    all_feature_names = list(ohe_feature_names) + BINARY_FEATURES + NUMERIC_FEATURES

    importances = pd.DataFrame({
        "feature": all_feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    return importances
```

### Conversion Signal Assignment

```python
def assign_conversion_signal(predicted_engagement, predicted_referrals, platform_median_referrals):
    """
    Classify each recommendation as 'converts', 'noise', or 'balanced'.

    - 'converts':  referrals >= 1.5x platform median (strong donation signal)
    - 'noise':     engagement is high (>= 0.12) but referrals < 0.8x platform median
    - 'balanced':  everything else
    """
    if predicted_referrals >= 1.5 * platform_median_referrals:
        return "converts"
    elif predicted_engagement >= 0.12 and predicted_referrals < 0.8 * platform_median_referrals:
        return "noise"
    else:
        return "balanced"
```

### Recommendation Assembly

For each platform × is_boosted combination, find the feature combination (from historical posts that are sufficiently represented) that maximizes predicted donation referrals, then score it with the engagement model too:

```python
from itertools import product

def find_best_combination(platform, is_boosted, posts_df, engagement_model, referrals_model, value_model):
    """
    For a given platform and boost status, find the feature combination
    among historically observed combos with >= MIN_SAMPLES_PER_COMBO posts
    that maximizes predicted donation_referrals.
    """
    MIN_SAMPLES_PER_COMBO = 3  # Minimum posts to consider a combo valid

    subset = posts_df[(posts_df["platform"] == platform) & (posts_df["is_boosted"] == is_boosted)]
    if len(subset) < 5:
        return None  # Not enough data for this platform/boost combo

    combo_cols = ["post_type", "media_type", "content_topic", "sentiment_tone",
                  "has_call_to_action", "call_to_action_type", "features_resident_story"]
    combo_counts = subset.groupby(combo_cols).size().reset_index(name="sample_count")
    valid_combos = combo_counts[combo_counts["sample_count"] >= MIN_SAMPLES_PER_COMBO]

    if valid_combos.empty:
        return None

    # Find best timing (day + hour) for this platform
    best_day  = subset.groupby("day_of_week")["donation_referrals"].mean().idxmax()
    best_hour = subset.groupby("post_hour")["donation_referrals"].mean().idxmax()
    best_hashtags = int(subset.groupby("num_hashtags")["donation_referrals"].mean().idxmax())

    # Score each valid combo
    best_score = -999
    best_rec = None

    for _, combo_row in valid_combos.iterrows():
        candidate = {
            "platform": platform,
            "is_boosted": is_boosted,
            "post_type": combo_row["post_type"],
            "media_type": combo_row["media_type"],
            "content_topic": combo_row["content_topic"],
            "sentiment_tone": combo_row["sentiment_tone"],
            "has_call_to_action": combo_row["has_call_to_action"],
            "call_to_action_type": combo_row["call_to_action_type"],
            "features_resident_story": combo_row["features_resident_story"],
            "day_of_week": best_day,
            "post_hour": best_hour,
            "num_hashtags": best_hashtags,
            "caption_length": int(subset["caption_length"].median()),
        }
        candidate_df = pd.DataFrame([candidate])[FEATURE_COLS_MODEL]
        pred_referrals = float(referrals_model.predict(candidate_df)[0])

        if pred_referrals > best_score:
            best_score = pred_referrals
            pred_engagement = float(engagement_model.predict(candidate_df)[0])
            pred_value = float(value_model.predict(candidate_df)[0])
            best_rec = {**candidate,
                        "predicted_engagement_rate": round(pred_engagement, 4),
                        "predicted_donation_referrals": round(pred_referrals, 4),
                        "predicted_donation_value_php": round(pred_value, 2),
                        "sample_count": int(combo_row["sample_count"])}

    return best_rec
```

---

## PHASE 8 — Deployment

### Scoring Script — `score_social_media.py`

```python
import pickle
import uuid
from datetime import datetime, timezone
import pandas as pd
from supabase import create_client
from storage.blob_client import download_artifact
from db.connection import engine
from pipelines.social_media.features import load_posts, clean_posts
from pipelines.social_media.score_social_media import (
    find_best_combination, assign_conversion_signal, assign_confidence_tier_social
)

PLATFORMS  = ["Facebook", "Instagram", "TikTok", "WhatsApp", "LinkedIn"]
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

def score():
    artifact = pickle.loads(download_artifact("social_media_model.pkl"))
    engagement_model  = artifact["engagement_model"]
    referrals_model   = artifact["referrals_model"]
    value_model       = artifact["value_model"]
    model_version     = artifact["model_version"]
    platform_medians  = artifact["platform_medians"]

    posts_df = clean_posts(load_posts(engine))
    records = []

    for platform in PLATFORMS:
        for is_boosted in [False, True]:
            rec = find_best_combination(
                platform, is_boosted, posts_df,
                engagement_model, referrals_model, value_model
            )
            if rec is None:
                continue

            platform_median = platform_medians.get(platform, 1.0)
            conversion_signal = assign_conversion_signal(
                rec["predicted_engagement_rate"],
                rec["predicted_donation_referrals"],
                platform_median,
            )

            records.append({
                "recommendation_id": str(uuid.uuid4()),
                "platform": platform,
                "is_boosted": is_boosted,
                "post_type": rec["post_type"],
                "media_type": rec["media_type"],
                "content_topic": rec["content_topic"],
                "sentiment_tone": rec["sentiment_tone"],
                "has_call_to_action": bool(rec["has_call_to_action"]),
                "call_to_action_type": rec["call_to_action_type"] if rec["has_call_to_action"] else None,
                "features_resident_story": bool(rec["features_resident_story"]),
                "best_day_of_week": rec["day_of_week"],
                "best_hour": int(rec["post_hour"]),
                "recommended_hashtag_count": int(rec["num_hashtags"]),
                "predicted_engagement_rate": rec["predicted_engagement_rate"],
                "predicted_donation_referrals": rec["predicted_donation_referrals"],
                "predicted_donation_value_php": rec["predicted_donation_value_php"],
                "conversion_signal": conversion_signal,
                "sample_count": rec["sample_count"],
                "confidence_tier": assign_confidence_tier_social(rec["sample_count"]),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model_version": model_version,
            })

    # Delete old recommendations and insert fresh set
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    supabase.table("social_media_recommendations").delete().neq("recommendation_id", "00000000-0000-0000-0000-000000000000").execute()
    supabase.table("social_media_recommendations").insert(records).execute()

    print(f"Social media recommendations written — {len(records)} rows")
    return len(records)
```

### Confidence Tier Logic

```python
def assign_confidence_tier_social(sample_count):
    if sample_count >= 30:
        return "high"
    elif sample_count >= 10:
        return "medium"
    else:
        return "low"
```

### FastAPI Endpoint — `main.py`

```python
@app.post("/score/social-media")
def score_social_media(x_api_key: str = Header(...)):
    verify_api_key(x_api_key)
    count = score()
    return {"status": "social media recommendations generated", "count_written": count}
```

### Supabase Table Schema

```sql
CREATE TABLE social_media_recommendations (
    recommendation_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    platform                    TEXT NOT NULL,
    is_boosted                  BOOLEAN NOT NULL,
    post_type                   TEXT,
    media_type                  TEXT,
    content_topic               TEXT,
    sentiment_tone              TEXT,
    has_call_to_action          BOOLEAN,
    call_to_action_type         TEXT,
    features_resident_story     BOOLEAN,
    best_day_of_week            TEXT,
    best_hour                   INTEGER,
    recommended_hashtag_count   INTEGER,
    predicted_engagement_rate   FLOAT,
    predicted_donation_referrals FLOAT,
    predicted_donation_value_php FLOAT,
    conversion_signal           TEXT,
    sample_count                INTEGER,
    confidence_tier             TEXT,
    generated_at                TIMESTAMPTZ,
    model_version               TEXT
);

-- No RLS required — recommendations are not sensitive
-- Unique constraint to prevent duplicates per platform/boost combo
CREATE UNIQUE INDEX social_media_recs_platform_boosted
    ON social_media_recommendations (platform, is_boosted);
```

### Azure Container Apps Job — `train_social_media.py`

Scheduled via Azure Container Apps Job. Run on the 1st of each month.

```yaml
# containerapp-job-social-media.yaml (excerpt)
schedule: "0 3 1 * *"   # 3:00 AM UTC on the 1st of each month
replicaTimeout: 1800
```

### Blob Storage Artifacts

| Artifact | Key | Contents |
|---|---|---|
| Trained models | `social_media_model.pkl` | Serialized dict: `engagement_model`, `referrals_model`, `value_model`, `platform_medians`, `model_version` |

### Environment Variables Required

| Variable | Purpose |
|---|---|
| `SUPABASE_DB_URL` | Direct PostgreSQL connection for training queries |
| `SUPABASE_URL` | REST API base URL for upsert operations |
| `SUPABASE_SERVICE_ROLE_KEY` | Service role key for write operations |
| `AZURE_STORAGE_CONNECTION_STRING` | Blob Storage access for model artifact |
| `ML_SERVICE_API_KEY` | Shared secret for `/score/social-media` endpoint |

---

## Current Model Limitations

- **LinkedIn and WhatsApp volume:** Both platforms have fewer historical posts than Facebook, Instagram, and TikTok. Recommendations for these platforms will be `low` confidence until posting history grows to ≥ 30 posts per feature combination.
- **Conversion model seasonality:** The referrals and value models are sensitive to seasonal donation patterns. Recommendations trained primarily on campaign-season data will overfit to campaign conditions. Retrain after any major fundraising campaign to re-calibrate.
- **No causal identification:** Both models are correlational. High predicted donation referrals for a given post type reflects historical co-occurrence, not a controlled experiment. External factors (news events, campaigns, appeals) may be driving referral spikes that the model attributes to post features.
- **Resident story flag quality:** `features_resident_story = True` includes all posts featuring any anonymized story content — there is no distinction between highly compelling stories and brief mentions. Model importance for this feature should be interpreted cautiously.
- **Caption length as proxy:** `caption_length` is a count of characters, not a quality measure. The model may recommend longer captions simply because historically long captions correlated with conversions during a period when other factors were also favorable.
- **Boosted post sample size:** Boosted posts represent a smaller share of total posts. The boosted-specific recommendations will be `medium` or `low` confidence on most platforms until more boosted post history accumulates.
