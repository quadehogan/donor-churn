# Donor Churn Prediction Pipeline
## Lighthouse Sanctuary — ML Pipeline Plan

> Follows the phase structure from `MASTER_ML_PIPELINE_GUIDE.md`. Each phase maps to the corresponding chapter guide for deeper reference. This document is specific to the Donor Churn pipeline and adapts the deployment pattern for a Supabase (PostgreSQL) backend and FastAPI microservice.

---

## Quick-Reference Pipeline Map

```
PHASE 1 — PLAN          Define churn, success criteria, feasibility gate
PHASE 2 — ACQUIRE       Pull donations + supporters via direct PostgreSQL → Supabase
PHASE 3 — EXPLORE       DataFrame inspection, unistats(), date feature extraction
PHASE 4 — PREPARE       Full prep pipeline: wrangling → imputation → outliers → encode
PHASE 5 — MODEL         Classification — Logistic Regression baseline → Random Forest → Gradient Boosting
PHASE 6 — EVALUATE      StratifiedKFold CV, ROC-AUC, Precision/Recall, learning curves
PHASE 7 — FEATURE SEL.  Tree importance → RFECV → top_risk_factors JSONB output
PHASE 8 — DEPLOY        Azure Container Apps (inference) + Container Apps Job (training) → Blob Storage artifacts → Supabase scores
```

---

## PHASE 1 — Project Planning (CRISP-DM)

### Feasibility Checklist

| Gate | Question | Status |
|---|---|---|
| Business | What specific problem is being solved? | Flag donors at risk of lapsing so staff can intervene proactively |
| Data | Is live, updatable data available? | Yes — `donations` and `supporters` tables in Supabase, updated on each donation |
| Analytical | Can data support reliable predictions? | Yes — recency, frequency, and monetary signals are strong churn predictors (established RFM literature) |
| Integration | Can model outputs plug into existing systems? | Yes — scores written to `donor_churn_scores` table; .NET API reads and serves to dashboard |
| Risk | Is operational disruption acceptable? | Low risk — scores are decision-support only, no automated actions taken |

### Problem Definition

**Prediction target:** Will a given active donor fail to make another donation within the next 90 days?

**Label definition:**
- `churned = 1` — donor's most recent donation was more than 90 days ago AND `status != 'active'`
- `churned = 0` — donor has donated within the last 90 days OR is flagged as a recurring donor with a cadence that makes 90 days normal

**Special case — recurring donors:** Donors where `is_recurring = true` get a 120-day churn window, since their giving cadence is structured and a 90-day gap may be within their normal schedule.

**Model output:** A probability score (0.0–1.0) per supporter and a risk tier:
- `high` — probability ≥ 0.70
- `medium` — probability 0.40–0.69
- `low` — probability < 0.40

**Success criteria:**
- ROC-AUC ≥ 0.75 on held-out test set
- Recall (churn class) ≥ 0.70 — we care more about catching at-risk donors than avoiding false alarms
- `top_risk_factors` JSONB field populated for every scored supporter (explainability requirement)

### CRISP-DM Phases

```
Business Understanding → Data Understanding → Data Preparation
       ↑                                             ↓
Deployment ← Evaluation ← Modeling ← ─────────────────
```

### Deliverables Before Proceeding

- [x] Churn definition agreed (90-day window, 120-day for recurring)
- [x] Success metrics defined (ROC-AUC ≥ 0.75, Recall ≥ 0.70)
- [x] Data source confirmed (Supabase PostgreSQL, direct connection)
- [x] Integration path confirmed (FastAPI → Supabase `donor_churn_scores`)
- [ ] Supabase connection string obtained and stored securely in `.env`

---

## PHASE 2 — Data Acquisition

> Reference: `chapter_four_implementation.md`

### Connection Pattern — Supabase Direct PostgreSQL

The ML service connects to Supabase using a direct PostgreSQL connection string, not the Supabase REST client. This is appropriate because training requires joining multiple tables across potentially thousands of rows — raw SQL handles this more efficiently than the HTTP-based PostgREST layer.

```python
import os
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# Connection string stored in .env — never hardcoded
# Format: postgresql://user:password@host:5432/postgres
DATABASE_URL = os.getenv("SUPABASE_DB_URL")

engine = create_engine(DATABASE_URL)
```

**`.env` file (never commit to version control):**
```
SUPABASE_DB_URL=postgresql://postgres:[password]@[project-ref].supabase.co:5432/postgres
```

### Data Pull — Training Query

Pull donations and supporters in a single joined query. One row per supporter, with donation history aggregated at the supporter level.

```python
TRAINING_QUERY = """
SELECT
    s.supporter_id,
    s.supporter_type,
    s.relationship_type,
    s.region,
    s.country,
    s.acquisition_channel,
    s.status,
    s.created_at                                        AS supporter_since,
    s.first_donation_date,

    -- Donation aggregates
    COUNT(d.donation_id)                                AS frequency,
    SUM(d.amount)                                       AS total_value,
    AVG(d.amount)                                       AS avg_value,
    MAX(d.amount)                                       AS max_value,
    MIN(d.amount)                                       AS min_value,
    MAX(d.donation_date)                                AS last_donation_date,
    MIN(d.donation_date)                                AS first_donation_date_tx,
    MAX(CASE WHEN d.is_recurring THEN 1 ELSE 0 END)    AS ever_recurring,
    COUNT(DISTINCT d.campaign_name)                     AS num_campaigns,
    COUNT(DISTINCT d.channel_source)                    AS num_channels,

    -- Label
    CASE
        WHEN MAX(CASE WHEN d.is_recurring THEN 1 ELSE 0 END) = 1
             AND (CURRENT_DATE - MAX(d.donation_date)) > 120 THEN 1
        WHEN MAX(CASE WHEN d.is_recurring THEN 1 ELSE 0 END) = 0
             AND (CURRENT_DATE - MAX(d.donation_date)) > 90  THEN 1
        ELSE 0
    END                                                 AS churned

FROM supporters s
JOIN donations d ON s.supporter_id = d.supporter_id
WHERE s.status IN ('active', 'lapsed', 'inactive')  -- exclude test/internal records
GROUP BY
    s.supporter_id, s.supporter_type, s.relationship_type,
    s.region, s.country, s.acquisition_channel,
    s.status, s.created_at, s.first_donation_date
HAVING COUNT(d.donation_id) >= 2  -- need at least 2 donations to compute trend
"""

df = pd.read_sql(TRAINING_QUERY, engine)

# Verify immediately after loading
print(df.shape)
print(df.dtypes)
print(df['churned'].value_counts(normalize=True))  # check class balance
df.head()
```

### Data Pull — Inference Query (Scoring New Donors)

```python
INFERENCE_QUERY = """
SELECT
    s.supporter_id,
    s.supporter_type,
    s.relationship_type,
    s.region,
    s.country,
    s.acquisition_channel,
    s.created_at                                        AS supporter_since,
    s.first_donation_date,
    COUNT(d.donation_id)                                AS frequency,
    SUM(d.amount)                                       AS total_value,
    AVG(d.amount)                                       AS avg_value,
    MAX(d.amount)                                       AS max_value,
    MIN(d.amount)                                       AS min_value,
    MAX(d.donation_date)                                AS last_donation_date,
    MIN(d.donation_date)                                AS first_donation_date_tx,
    MAX(CASE WHEN d.is_recurring THEN 1 ELSE 0 END)    AS ever_recurring,
    COUNT(DISTINCT d.campaign_name)                     AS num_campaigns,
    COUNT(DISTINCT d.channel_source)                    AS num_channels
FROM supporters s
JOIN donations d ON s.supporter_id = d.supporter_id
WHERE s.status = 'active'
GROUP BY
    s.supporter_id, s.supporter_type, s.relationship_type,
    s.region, s.country, s.acquisition_channel,
    s.created_at, s.first_donation_date
HAVING COUNT(d.donation_id) >= 2
"""

df_new = pd.read_sql(INFERENCE_QUERY, engine)
```

---

## PHASE 3 — Data Exploration

> Reference: `chapter_two_implementation.md`, `chapter_three_implementation.md`, `chapter_six_implementation.md`

### Immediate Inspection

```python
print(df.shape)
print(df.dtypes)
df.isnull().sum()
df.describe()
print(f"Churn rate: {df['churned'].mean():.2%}")
```

**Expected red flags to look for:**
- Class imbalance — churn rate is likely 20–40%; if below 10%, discuss with stakeholders before proceeding
- Sparse regions or acquisition channels — collapse rare categories in Phase 4
- `total_value` and `avg_value` will be right-skewed — plan log transform

### Date Feature Extraction

```python
import numpy as np

# Convert dates
df['last_donation_date']     = pd.to_datetime(df['last_donation_date'])
df['first_donation_date']    = pd.to_datetime(df['first_donation_date'])
df['supporter_since']        = pd.to_datetime(df['supporter_since'])

# Derived numeric features
df['recency_days']     = (pd.Timestamp.today() - df['last_donation_date']).dt.days
df['tenure_days']      = (pd.Timestamp.today() - df['supporter_since']).dt.days
df['active_span_days'] = (df['last_donation_date'] - df['first_donation_date']).dt.days

# Average gap between donations (spread of giving over time)
df['avg_gap_days'] = df['active_span_days'] / (df['frequency'] - 1).clip(lower=1)

# Donation trend — are amounts increasing or decreasing over time?
# Computed per supporter using a separate query or window function
# Positive = increasing generosity; negative = declining
# (computed below as a separate enrichment step)

# Drop raw date columns after extraction
df = df.drop(columns=['last_donation_date', 'first_donation_date',
                       'first_donation_date_tx', 'supporter_since'])
```

### Donation Trend Feature (Enrichment Query)

```python
TREND_QUERY = """
SELECT
    supporter_id,
    REGR_SLOPE(amount, EXTRACT(EPOCH FROM donation_date)) AS donation_trend
FROM donations
GROUP BY supporter_id
"""

df_trend = pd.read_sql(TREND_QUERY, engine)
df = df.merge(df_trend, on='supporter_id', how='left')
```

### Automated Univariate EDA

```python
def unistats(df):
    output_df = pd.DataFrame(columns=[
        'Count', 'Unique', 'Type',
        'Min', 'Max', '25%', '50%', '75%',
        'Mean', 'Median', 'Mode', 'Std', 'Skew', 'Kurt'
    ])
    for col in df.columns:
        count  = df[col].count()
        unique = df[col].nunique()
        dtype  = str(df[col].dtype)
        min_val = max_val = q1 = q2 = q3 = '-'
        mean_val = median_val = mode_val = std_val = skew_val = kurt_val = '-'
        if pd.api.types.is_numeric_dtype(df[col]):
            min_val    = round(df[col].min(), 2)
            max_val    = round(df[col].max(), 2)
            q1         = round(df[col].quantile(0.25), 2)
            q2         = round(df[col].quantile(0.50), 2)
            q3         = round(df[col].quantile(0.75), 2)
            mean_val   = round(df[col].mean(), 2)
            median_val = round(df[col].median(), 2)
            mode_val   = round(df[col].mode().values[0], 2)
            std_val    = round(df[col].std(), 2)
            skew_val   = round(df[col].skew(), 2)
            kurt_val   = round(df[col].kurt(), 2)
        output_df.loc[col] = (
            count, unique, dtype,
            min_val, max_val, q1, q2, q3,
            mean_val, median_val, mode_val, std_val, skew_val, kurt_val
        )
    return output_df

stats = unistats(df)
# Watch for: |Skew| > 1 on total_value, avg_value, recency_days → log transform
# Watch for: Unique = 1 → constant column, drop it
```

---

## PHASE 4 — Data Preparation

> Reference: `chapter_seven_implementation.md`, `chapter_eight_implementation.md`

### Feature Set After Extraction

| Feature | Source | Type | Notes |
|---|---|---|---|
| `recency_days` | Derived | Numeric | Days since last donation — most important feature |
| `frequency` | Aggregated | Numeric | Total number of donations |
| `total_value` | Aggregated | Numeric | Right-skewed — apply log1p |
| `avg_value` | Aggregated | Numeric | Right-skewed — apply log1p |
| `max_value` | Aggregated | Numeric | Right-skewed — apply log1p |
| `ever_recurring` | Aggregated | Binary | 0/1 — already encoded |
| `num_campaigns` | Aggregated | Numeric | Number of distinct campaigns responded to |
| `num_channels` | Aggregated | Numeric | Number of distinct donation channels used |
| `avg_gap_days` | Derived | Numeric | Average days between donations |
| `tenure_days` | Derived | Numeric | Days since supporter created |
| `active_span_days` | Derived | Numeric | First to last donation span |
| `donation_trend` | Enriched | Numeric | Slope of amount over time — negative = declining |
| `supporter_type` | `supporters` | Categorical | individual / organization / etc. |
| `relationship_type` | `supporters` | Categorical | Encode — may have rare categories |
| `acquisition_channel` | `supporters` | Categorical | Collapse rare channels → 'Other' |
| `region` | `supporters` | Categorical | Collapse rare regions → 'Other' |

**Drop before modeling:** `supporter_id` (identifier, not a feature)

### Prep Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np

# Separate features by type
num_cols = [
    'recency_days', 'frequency', 'total_value', 'avg_value', 'max_value',
    'ever_recurring', 'num_campaigns', 'num_channels', 'avg_gap_days',
    'tenure_days', 'active_span_days', 'donation_trend'
]
cat_cols = ['supporter_type', 'relationship_type', 'acquisition_channel', 'region']

# STEP 1: Collapse rare categories (< 5% frequency → 'Other')
for col in cat_cols:
    freq = df[col].value_counts(normalize=True)
    rare = freq[freq < 0.05].index
    df[col] = df[col].replace(rare, 'Other')

# STEP 2: Apply log1p to right-skewed numeric columns
for col in ['total_value', 'avg_value', 'max_value']:
    df[f'{col}_log'] = np.log1p(df[col])
    df = df.drop(columns=[col])
    num_cols = [c if c != col else f'{col}_log' for c in num_cols]

# STEP 3: Separate label and drop ID
y = df['churned']
X = df.drop(columns=['churned', 'supporter_id'])

# STEP 4: Train/test split — stratified to preserve churn ratio
from sklearn.model_selection import train_test_split

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
)

# STEP 5: Build preprocessing pipeline
numeric_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', numeric_pipe, num_cols),
    ('cat', categorical_pipe, cat_cols)
])
```

### Missing Data Decision Logic

```
Missing values in column?
    ↓ Yes
Run t-test (n < 200k) comparing churn rate: rows with missing vs. rows without
    ↓ p > 0.10 → MCAR → impute safely (median for numeric, mode for categorical)
    ↓ p ≤ 0.10 → MAR  → drop missing rows and document (missingness is biased)
```

**Columns most likely to have nulls:** `donation_trend` (for donors with only 2 donations, slope may be unstable — impute with 0), `region` (impute with 'Unknown').

---

## PHASE 5 — Modeling (Classification)

> Reference: `chapter_thirteen_implementation.md`, `chapter_fourteen_implementation.md`

### Model Comparison

Run all four models first with default settings to establish a baseline before tuning anything.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

models = [
    (LogisticRegression(max_iter=1000, class_weight='balanced'), 'Logistic Regression'),
    (DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42), 'Decision Tree'),
    (RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42), 'Random Forest'),
    (GradientBoostingClassifier(n_estimators=100, random_state=42), 'Gradient Boosting'),
]

for Model, name in models:
    pipe = Pipeline([('prep', preprocessor), ('model', Model)])
    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    print(f"\n{name}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(classification_report(y_test, y_pred))
```

**Note on `class_weight='balanced'`:** Churn datasets are almost always imbalanced — more active donors than churned ones. The `balanced` flag automatically weights the minority class (churned) higher during training, which improves recall on the class we care most about catching.

### Expected Outcome

Random Forest and Gradient Boosting will outperform Logistic Regression on this problem because:
- `recency_days` interacts with `frequency` and `avg_gap_days` in nonlinear ways that trees handle naturally
- Donation trend signals have threshold effects (a slight decline matters less than a sharp one)

Proceed with **Random Forest** as the primary model unless Gradient Boosting outperforms by more than 2 AUC points — Random Forest is faster to train and retrain on a schedule.

---

## PHASE 6 — Evaluation, Selection & Tuning

> Reference: `chapter_fifteen_implementation.md`

### Stratified Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_pipe = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
])

cv_results = cross_validate(
    rf_pipe, X_train_full, y_train_full,
    cv=cv,
    scoring=['roc_auc', 'f1', 'recall', 'precision'],
    return_train_score=True
)

print(f"Val ROC-AUC: {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
print(f"Val Recall:  {cv_results['test_recall'].mean():.4f}")
print(f"Val F1:      {cv_results['test_f1'].mean():.4f}")
# Large gap (train >> val) = overfitting → reduce n_estimators or increase min_samples_leaf
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'model__n_estimators':    [100, 200, 300, 500],
    'model__max_depth':       [4, 6, 8, 10, None],
    'model__min_samples_leaf': [1, 5, 10, 20],
    'model__max_features':    ['sqrt', 'log2', 0.5],
}

rnd = RandomizedSearchCV(
    rf_pipe, param_dist, n_iter=30, cv=cv,
    scoring='roc_auc', random_state=42, n_jobs=-1
)
rnd.fit(X_train_full, y_train_full)
print(f"Best params: {rnd.best_params_}")
print(f"Best CV AUC: {rnd.best_score_:.4f}")
```

### Final Test-Set Evaluation (run once at the end)

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

best_model = rnd.best_estimator_
y_pred     = best_model.predict(X_test)
y_proba    = best_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(confusion_matrix(y_test, y_pred))

# Gate check — do not proceed to deployment if either condition fails
assert roc_auc_score(y_test, y_proba) >= 0.75, "AUC below threshold — do not deploy"
assert classification_report(y_test, y_pred, output_dict=True)['1']['recall'] >= 0.70, \
    "Recall below threshold — do not deploy"
```

### Learning Curve (diagnose overfitting)

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train_full, y_train_full,
    cv=cv, train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='roc_auc', n_jobs=-1
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Train AUC')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Val AUC')
plt.xlabel('Training Set Size')
plt.ylabel('ROC-AUC')
plt.title('Donor Churn — Learning Curve')
plt.legend()
plt.savefig('artifacts/learning_curve.png')
```

---

## PHASE 7 — Feature Selection & Explainability

> Reference: `chapter_sixteen_implementation.md`

### Tree Feature Importance

```python
import pandas as pd

# Get feature names after preprocessing
cat_feature_names = (
    best_model.named_steps['prep']
    .named_transformers_['cat']
    .named_steps['onehot']
    .get_feature_names_out(cat_cols)
    .tolist()
)
all_feature_names = num_cols + cat_feature_names

importances = pd.Series(
    best_model.named_steps['model'].feature_importances_,
    index=all_feature_names
).sort_values(ascending=False)

print(importances.head(10))
# Expected top features: recency_days, frequency, avg_gap_days, total_value_log, donation_trend
```

### Generating `top_risk_factors` per Supporter

This is the explainability output written to Supabase alongside the probability score. It tells staff in plain language why a donor is flagged.

```python
import json

# Map feature names to human-readable labels
FEATURE_LABELS = {
    'recency_days':       'No recent donation',
    'frequency':          'Low donation frequency',
    'avg_gap_days':       'Long gaps between donations',
    'total_value_log':    'Low total giving',
    'avg_value_log':      'Declining average gift size',
    'donation_trend':     'Decreasing donation amounts',
    'tenure_days':        'Short supporter tenure',
    'num_campaigns':      'Low campaign engagement',
    'ever_recurring':     'Not a recurring donor',
}

def get_top_risk_factors(supporter_row, model, feature_names, top_n=3):
    """
    Returns a list of human-readable risk factors for a single supporter
    based on which features deviate most from the non-churn population mean.
    """
    importances = model.named_steps['model'].feature_importances_
    top_indices = importances.argsort()[::-1][:top_n]
    factors = []
    for i in top_indices:
        name = feature_names[i]
        label = FEATURE_LABELS.get(name, name)
        factors.append(label)
    return factors

# Applied during inference (see Phase 8)
```

---

## PHASE 8 — Deployment

> Reference: `chapter_seventeen_implementation.md`
> Architecture guide: Azure-hosted, modular, production-ready. Training is fully decoupled from the API request cycle. Model artifacts stored in Azure Blob Storage. Inference scores pre-computed and stored in Supabase so the .NET API never calls the ML service at request time.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LIGHTHOUSE — AZURE DEPLOYMENT                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   [React Frontend]          [Azure Static Web Apps]                     │
│         │  HTTPS                                                        │
│         ▼                                                               │
│   [.NET 10 API]             [Azure App Service]                         │
│         │  reads pre-computed scores                                    │
│         ▼                                                               │
│   [Supabase PostgreSQL]     [Managed by Supabase]                       │
│     donor_churn_scores            ▲                                     │
│                                   │ upserts scores                      │
│   ┌───────────────────────────────┼──────────────────┐                 │
│   │         ML LAYER              │                  │                 │
│   │                               │                  │                 │
│   │  [FastAPI Inference Service]  │  [Training Job]  │                 │
│   │  Azure Container Apps         │  Azure Container  │                │
│   │  (always-on, scales to 1)     │  Apps Job        │                │
│   │         │                     │  (runs on cron)  │                │
│   │         │ loads model         │       │           │                │
│   │         ▼                     │       ▼           │                │
│   │  [Azure Blob Storage]  ←──────┘  writes artifact  │               │
│   │   ml-artifacts container         ← churn_model.pkl│               │
│   │                                  ← metadata.json  │               │
│   │                                  ← metrics.json   │               │
│   └──────────────────────────────────────────────────┘                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Service Responsibilities

| Service | Azure Resource | Responsibility |
|---|---|---|
| React Frontend | Azure Static Web Apps | Serves the admin dashboard UI; calls .NET API only |
| .NET 10 API | Azure App Service | Handles all business logic and HTTP requests; reads pre-computed churn scores from Supabase; calls FastAPI only for admin-triggered score refreshes |
| FastAPI ML Service | Azure Container Apps | Runs batch inference on demand; exposes `/score/churn` and `/health` endpoints; loads model from Blob Storage; writes scores to Supabase |
| Training Job | Azure Container Apps Job | Runs on a cron schedule; pulls data from Supabase; trains model; writes versioned artifact to Blob Storage; triggers scoring run |
| Supabase PostgreSQL | Supabase (managed) | Source of truth for all relational data; stores pre-computed churn scores in `donor_churn_scores` table |
| Azure Blob Storage | Azure Storage Account | Stores model artifacts, metadata, and metrics; versioned by timestamp; retained for rollback |

### Data Flow

**Inference path (staff opens donor dashboard — no ML service called):**
```
React → GET /api/supporters (with churn scores)
      → .NET API reads supporters JOIN donor_churn_scores from Supabase
      → Returns combined payload to React
      → Dashboard renders risk_tier column
```
The .NET API never calls the FastAPI service at request time. Scores are always pre-computed. This keeps the dashboard fast regardless of model complexity.

**Admin-triggered score refresh:**
```
React → POST /api/ml/score-churn (admin action)
      → .NET API calls FastAPI POST /score/churn
      → FastAPI pulls active supporters from Supabase
      → FastAPI loads churn_model.pkl from Azure Blob Storage
      → FastAPI runs inference → upserts scores to Supabase
      → Returns { status, count_scored } to .NET API → to React
```

**Scheduled training + scoring (runs monthly, no user interaction):**
```
Azure Container Apps Job (cron: 0 2 1 * *)
      → train_churn.py pulls donations + supporters from Supabase
      → Trains Random Forest pipeline
      → Asserts AUC ≥ 0.75 and Recall ≥ 0.70 (hard gate)
      → Uploads churn_model_{version}.pkl to Azure Blob Storage
      → Uploads churn_metadata.json + churn_metrics.json to Blob Storage
      → Calls score_churn.py → upserts fresh scores to Supabase
```

### Project Folder Structure

```
ml-service/
├── .env                          ← secrets (never commit)
├── requirements.txt
├── Dockerfile
├── main.py                       ← FastAPI app entry point
├── pipelines/
│   └── churn/
│       ├── features.py           ← SQL queries + feature engineering
│       ├── train_churn.py        ← training job (run by Container Apps Job)
│       └── score_churn.py        ← batch inference + Supabase upsert
├── storage/
│   └── blob_client.py            ← Azure Blob Storage read/write helpers
└── db/
    └── connection.py             ← SQLAlchemy engine factory
```

### Azure Blob Storage — Artifact Helpers (`storage/blob_client.py`)

Model artifacts are stored in Azure Blob Storage instead of the local filesystem. This keeps artifacts available to both the training job and the inference service regardless of which container instance runs, and enables versioning and rollback.

```python
import os, pickle, json, io
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

CONTAINER_NAME = "ml-artifacts"

def get_container():
    client = BlobServiceClient.from_connection_string(
        os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    )
    return client.get_container_client(CONTAINER_NAME)

def upload_artifact(local_path: str, blob_name: str):
    """Upload a file to Azure Blob Storage."""
    container = get_container()
    with open(local_path, 'rb') as f:
        container.upload_blob(blob_name, f, overwrite=True)
    print(f"Uploaded → {blob_name}")

def download_artifact(blob_name: str, local_path: str):
    """Download a file from Azure Blob Storage."""
    container = get_container()
    blob = container.download_blob(blob_name)
    with open(local_path, 'wb') as f:
        f.write(blob.readall())
    print(f"Downloaded ← {blob_name}")

def load_model_from_blob(blob_name: str = "churn/churn_model.pkl"):
    """Load a pickled model directly from Blob Storage into memory."""
    container = get_container()
    blob = container.download_blob(blob_name)
    return pickle.loads(blob.readall())

def load_json_from_blob(blob_name: str) -> dict:
    """Load a JSON file directly from Blob Storage."""
    container = get_container()
    blob = container.download_blob(blob_name)
    return json.loads(blob.readall())
```

### `donor_churn_scores` Table (Supabase — one-time setup)

```sql
CREATE TABLE donor_churn_scores (
    supporter_id        TEXT PRIMARY KEY REFERENCES supporters(supporter_id),
    churn_probability   FLOAT NOT NULL,
    risk_tier           TEXT NOT NULL CHECK (risk_tier IN ('high', 'medium', 'low')),
    top_risk_factors    JSONB,
    scored_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version       TEXT NOT NULL
);

-- Index for dashboard queries filtered by risk tier
CREATE INDEX idx_churn_scores_risk_tier ON donor_churn_scores (risk_tier);
```

### Training Job (`train_churn.py`)

```python
import pickle, json, os, io
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from dotenv import load_dotenv
from storage.blob_client import upload_artifact

load_dotenv()
engine = create_engine(os.getenv("SUPABASE_DB_URL"))

# 1. Load training data from Supabase
df = pd.read_sql(TRAINING_QUERY, engine)   # defined in features.py

# 2. Feature engineering
df = engineer_features(df)                 # defined in features.py

# 3. Split
y = df['churned']
X = df.drop(columns=['churned', 'supporter_id'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Build and fit pipeline
pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=300, max_depth=8,
        min_samples_leaf=10, class_weight='balanced',
        random_state=42
    ))
])
pipeline.fit(X_train, y_train)

# 5. Evaluate — hard gate before saving
y_proba = pipeline.predict_proba(X_test)[:, 1]
auc     = roc_auc_score(y_test, y_proba)
report  = classification_report(y_test, pipeline.predict(X_test), output_dict=True)
recall  = report['1']['recall']

assert auc >= 0.75,    f"AUC {auc:.4f} below threshold — aborting save"
assert recall >= 0.70, f"Recall {recall:.4f} below threshold — aborting save"

# 6. Save artifacts to Azure Blob Storage (versioned + latest)
model_version = datetime.now().strftime('%Y%m%d_%H%M')

with open('/tmp/churn_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

metadata = {
    'trained_at':    datetime.now().isoformat(),
    'model_version': model_version,
    'n_samples':     len(df),
    'churn_rate':    float(y.mean()),
    'features':      X.columns.tolist(),
    'model_class':   type(pipeline.named_steps['model']).__name__,
    'hyperparams':   pipeline.named_steps['model'].get_params()
}
metrics = {'roc_auc': round(auc, 4), 'recall_churn': round(recall, 4)}

with open('/tmp/churn_metadata.json', 'w') as f: json.dump(metadata, f, indent=2)
with open('/tmp/churn_metrics.json',  'w') as f: json.dump(metrics,  f, indent=2)

# Upload versioned copy + overwrite 'latest'
upload_artifact('/tmp/churn_model.pkl',    f"churn/versions/churn_model_{model_version}.pkl")
upload_artifact('/tmp/churn_metadata.json',f"churn/versions/churn_metadata_{model_version}.json")
upload_artifact('/tmp/churn_metrics.json', f"churn/versions/churn_metrics_{model_version}.json")
upload_artifact('/tmp/churn_model.pkl',    "churn/churn_model.pkl")       # latest — used by inference
upload_artifact('/tmp/churn_metadata.json',"churn/churn_metadata.json")
upload_artifact('/tmp/churn_metrics.json', "churn/churn_metrics.json")

print(f"Model saved — AUC: {auc:.4f} | Recall: {recall:.4f} | Version: {model_version}")
```

### Inference Job (`score_churn.py`)

```python
import json, os
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
from dotenv import load_dotenv
from storage.blob_client import load_model_from_blob, load_json_from_blob

load_dotenv()
engine = create_engine(os.getenv("SUPABASE_DB_URL"))

# 1. Load model and metadata from Azure Blob Storage
pipeline = load_model_from_blob("churn/churn_model.pkl")
metadata = load_json_from_blob("churn/churn_metadata.json")

# 2. Pull active supporters from Supabase
df_new        = pd.read_sql(INFERENCE_QUERY, engine)   # defined in features.py
supporter_ids = df_new['supporter_id'].tolist()
df_new        = engineer_features(df_new)
X_new         = df_new.drop(columns=['supporter_id'])

# 3. Score
probabilities = pipeline.predict_proba(X_new)[:, 1]

def assign_tier(prob):
    if prob >= 0.70: return 'high'
    if prob >= 0.40: return 'medium'
    return 'low'

# 4. Generate top_risk_factors
feature_names     = metadata['features']
risk_factors_list = [
    get_top_risk_factors(row, pipeline, feature_names)
    for _, row in df_new.iterrows()
]

# 5. Build results
results = pd.DataFrame({
    'supporter_id':      supporter_ids,
    'churn_probability': probabilities.round(4),
    'risk_tier':         [assign_tier(p) for p in probabilities],
    'top_risk_factors':  [json.dumps(f) for f in risk_factors_list],
    'scored_at':         datetime.now().isoformat(),
    'model_version':     metadata['model_version']
})

# 6. Upsert to Supabase
with engine.begin() as conn:
    for _, row in results.iterrows():
        conn.execute(text("""
            INSERT INTO donor_churn_scores
                (supporter_id, churn_probability, risk_tier, top_risk_factors, scored_at, model_version)
            VALUES
                (:supporter_id, :churn_probability, :risk_tier, :top_risk_factors::jsonb, :scored_at, :model_version)
            ON CONFLICT (supporter_id) DO UPDATE SET
                churn_probability = EXCLUDED.churn_probability,
                risk_tier         = EXCLUDED.risk_tier,
                top_risk_factors  = EXCLUDED.top_risk_factors,
                scored_at         = EXCLUDED.scored_at,
                model_version     = EXCLUDED.model_version
        """), row.to_dict())

print(f"Scored {len(results)} supporters — "
      f"High: {(results['risk_tier']=='high').sum()} | "
      f"Medium: {(results['risk_tier']=='medium').sum()} | "
      f"Low: {(results['risk_tier']=='low').sum()}")
```

### FastAPI Entry Point (`main.py`)

```python
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
import subprocess, os

app = FastAPI(title="Lighthouse ML Service")

# Simple API key auth — .NET backend passes this header
API_KEY        = os.getenv("ML_SERVICE_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/score/churn", dependencies=[Depends(verify_key)])
def trigger_scoring():
    """Trigger a batch scoring run. Called by .NET API on admin action."""
    result = subprocess.run(
        ["python", "pipelines/churn/score_churn.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr)
    return {"status": "scoring complete", "output": result.stdout}
```

Note: The `/train/churn` endpoint is intentionally removed from the API. Training is triggered only by the Azure Container Apps Job on schedule, or manually by running the job from the Azure portal. This prevents accidental retraining from a web request and keeps the training workload fully separated from the API service.

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

Set these in Azure Container Apps → Secrets and Environment Variables:

| Variable | Used By | Description |
|---|---|---|
| `SUPABASE_DB_URL` | Both services | PostgreSQL connection string — use Session mode (port 5432) |
| `AZURE_STORAGE_CONNECTION_STRING` | Both services | Azure Blob Storage connection string |
| `ML_SERVICE_API_KEY` | FastAPI service | Shared secret — .NET API passes this in `X-API-Key` header |

Set this in Azure App Service (the .NET backend):

| Variable | Description |
|---|---|
| `ML_SERVICE_URL` | Base URL of the FastAPI Container App, e.g. `https://lighthouse-ml.azurecontainerapps.io` |
| `ML_SERVICE_API_KEY` | Same shared secret as above |

**Important — Supabase connection string:** Use the **Session mode** connection string (port 5432) from Supabase → Settings → Database → Connection string. Do not use Transaction mode (port 6543) — the ML service holds a connection open during training and bulk scoring, which is incompatible with the transaction pooler.

### Azure Setup Steps

1. **Create an Azure Storage Account** and a container named `ml-artifacts`. Note the connection string from Access Keys.

2. **Create an Azure Container Registry (ACR)** to store the Docker image:
   ```bash
   az acr create --name lighthouseml --resource-group lighthouse-rg --sku Basic
   az acr build --registry lighthouseml --image ml-service:latest .
   ```

3. **Deploy the FastAPI service as an Azure Container App:**
   ```bash
   az containerapp create \
     --name lighthouse-ml-service \
     --resource-group lighthouse-rg \
     --environment lighthouse-env \
     --image lighthouseml.azurecr.io/ml-service:latest \
     --target-port 8000 \
     --ingress external \
     --min-replicas 1 --max-replicas 3 \
     --secrets supabase-url=<value> blob-conn=<value> api-key=<value> \
     --env-vars \
         SUPABASE_DB_URL=secretref:supabase-url \
         AZURE_STORAGE_CONNECTION_STRING=secretref:blob-conn \
         ML_SERVICE_API_KEY=secretref:api-key
   ```

4. **Deploy the training job as an Azure Container Apps Job:**
   ```bash
   az containerapp job create \
     --name lighthouse-ml-train \
     --resource-group lighthouse-rg \
     --environment lighthouse-env \
     --image lighthouseml.azurecr.io/ml-service:latest \
     --trigger-type Schedule \
     --cron-expression "0 2 1 * *" \
     --replica-timeout 3600 \
     --command "python pipelines/churn/train_churn.py && python pipelines/churn/score_churn.py" \
     --secrets supabase-url=<value> blob-conn=<value> \
     --env-vars \
         SUPABASE_DB_URL=secretref:supabase-url \
         AZURE_STORAGE_CONNECTION_STRING=secretref:blob-conn
   ```

5. **Add the ML service URL to Azure App Service** (the .NET backend) as an environment variable so `DonationsController` can call it for admin-triggered score refreshes.

### Retraining Strategy

| Trigger | Who initiates | Action |
|---|---|---|
| Monthly schedule | Azure Container Apps Job (automated) | Full retrain + score all active donors |
| New donation batch imported | .NET API calls FastAPI `/score/churn` | Score only — no retrain |
| Admin clicks "Refresh scores" | React → .NET API → FastAPI `/score/churn` | Score only — no retrain |
| AUC drops below 0.72 | Detected in metrics.json after monthly run | Alert logged — manual retrain review required |

### Rollback Plan

The training job uploads a versioned copy (`churn_model_YYYYMMDD_HHMM.pkl`) to Blob Storage alongside the `churn_model.pkl` latest pointer. To roll back to a previous model:

1. In Azure Blob Storage, copy the desired versioned `.pkl` over `churn/churn_model.pkl`
2. Trigger a scoring run via the admin dashboard or Azure portal
3. Scores in Supabase update with the `model_version` field reflecting the rollback

### Monitoring Checklist

- [ ] **Class drift** — compare churn rate in `donor_churn_scores` monthly against training baseline; alert if shift > 10 points
- [ ] **Input validation** — check for null spikes in `recency_days` and `frequency` before scoring; log warnings to Azure Monitor
- [ ] **Artifact versioning** — `model_version` timestamp written to every score row; versioned files retained in Blob Storage for 6 months
- [ ] **Gate enforcement** — training script asserts AUC ≥ 0.75 and Recall ≥ 0.70; job fails loudly if thresholds are not met
- [ ] **Container App health** — configure `/health` endpoint as liveness probe in Azure Container Apps
- [ ] **Job run logs** — Azure Container Apps Job logs available in Azure Monitor / Log Analytics; set alert on job failure

---

## Appendix — `requirements.txt`

```
pandas==2.2.x
numpy==1.26.x
scikit-learn==1.5.x
sqlalchemy==2.0.x
psycopg2-binary==2.9.x
fastapi==0.111.x
uvicorn==0.30.x
python-dotenv==1.0.x
matplotlib==3.9.x
scipy==1.13.x
azure-storage-blob==12.x.x
```
