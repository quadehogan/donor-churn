# Resident Risk & Progression Scoring Pipeline
## Lighthouse Sanctuary — ML Pipeline Plan

> Follows the phase structure from `MASTER_ML_PIPELINE_GUIDE.md`. This is the most mission-critical pipeline in the system. It scores each resident's current trajectory across all safehouses simultaneously, flagging regression risk and estimating reintegration readiness. Outputs are strictly internal — never exposed to donors or external users.

> **Privacy notice:** This pipeline processes records of minors who are abuse survivors. All outputs are restricted to authenticated staff with `Admin` or `SocialWorker` roles. No individual resident data is used in any external-facing feature. Access to the `resident_risk_scores` table must be restricted at the database level.

---

## Repo Placement

```
lighthouse-ml/
├── Dockerfile
├── requirements.txt
├── main.py
├── pipelines/
│   ├── churn/                        ← donor churn (deployed)
│   ├── impact_attribution/           ← impact attribution
│   └── resident_risk/                ← THIS PIPELINE
│       ├── features.py               ← SQL queries + longitudinal feature engineering
│       ├── train_risk.py             ← model A: regression risk classifier
│       ├── train_reintegration.py    ← model B: reintegration readiness classifier
│       └── score_residents.py        ← batch scoring + Supabase upsert
├── storage/
│   └── blob_client.py                ← shared
└── db/
    └── connection.py                 ← shared
```

---

## Quick-Reference Pipeline Map

```
PHASE 1 — PLAN          Define two outputs: regression risk + reintegration readiness
PHASE 2 — ACQUIRE       Multi-table join: residents + health + education + incidents + sessions + visitations + plans
PHASE 3 — EXPLORE       Longitudinal inspection, unistats(), label distribution
PHASE 4 — PREPARE       Lag features, rolling averages, trend slopes, time-since features, encode categoricals
PHASE 5 — MODEL         Two classifiers: Random Forest (regression risk) + Random Forest (reintegration readiness)
PHASE 6 — EVALUATE      StratifiedKFold CV, Recall prioritized, confusion matrix, learning curves
PHASE 7 — FEATURE SEL.  Tree importance → top_concern_factors JSONB for social worker explainability
PHASE 8 — DEPLOY        Azure Container Apps → resident_risk_scores table (restricted access)
```

---

## PHASE 1 — Project Planning (CRISP-DM)

### Feasibility Checklist

| Gate | Question | Status |
|---|---|---|
| Business | What specific problem is being solved? | Give social workers and founders an early warning system across all safehouses — catch struggling residents before an incident is filed |
| Data | Is live, updatable data available? | Yes — health, education, incident, session, visitation, and plan data all in Supabase, updated continuously |
| Analytical | Can data support reliable predictions? | Yes — `initial_risk_level` vs `current_risk_level` provides ground truth; longitudinal health and education trends are strong predictors |
| Integration | Can outputs plug into existing systems? | Yes — scores written to `resident_risk_scores`; .NET API reads for social worker dashboard (authenticated, role-restricted) |
| Risk | Privacy and safety concerns? | High — data involves minors who are abuse survivors. Outputs are decision-support only. No automated actions. Access restricted to Admin and SocialWorker roles at both API and DB level |

### Two Model Outputs

This pipeline trains and deploys **two separate classifiers** per resident:

**Model A — Regression Risk**
> Has this resident's situation worsened (risk level escalated) over the past 30–60 days, and are current signals pointing toward further regression?
- Label: `1` if `current_risk_level` is higher than `initial_risk_level` OR a high-severity incident was filed in the last 30 days
- Label: `0` if risk level is stable or improving

**Model B — Reintegration Readiness**
> Is this resident on track for reintegration based on current progress across all domains?
- Label: `1` if `reintegration_status` is `ready` or `in_progress` AND education progress ≥ 70% AND no high-severity incidents in last 90 days
- Label: `0` otherwise

### Output per Resident

```
{
  "resident_id":            "RES-00142",
  "regression_risk_score":  0.82,
  "regression_risk_tier":   "high",
  "reintegration_score":    0.21,
  "reintegration_tier":     "not_ready",
  "top_concern_factors":    ["declining health scores", "recent incident filed", "missed sessions"],
  "top_strength_factors":   ["education progress improving", "positive family visit"],
  "scored_at":              "2026-04-07T...",
  "model_version":          "20260401_0200"
}
```

### Success Criteria

- Model A Recall (regression class) ≥ 0.75 — missing a struggling resident is the worst outcome
- Model A ROC-AUC ≥ 0.75
- Model B ROC-AUC ≥ 0.70
- `top_concern_factors` and `top_strength_factors` populated for every scored resident
- Social workers can understand why a resident is flagged without looking at raw scores

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

### Core Query — Resident Base + Latest State

```python
RESIDENT_BASE_QUERY = """
SELECT
    r.resident_id,
    r.safehouse_id,
    r.case_status,
    r.sex,
    r.case_category,
    r.initial_risk_level,
    r.current_risk_level,
    r.reintegration_status,
    r.reintegration_type,
    r.date_of_admission,
    r.age_upon_admission,
    r.present_age,
    r.length_of_stay,
    r.assigned_social_worker,
    r.is_pwd,
    r.has_special_needs,
    r.family_is_4ps,
    r.family_solo_parent,
    r.family_indigenous,
    -- Vulnerability flags as numeric
    CASE WHEN r.sub_cat_trafficked    THEN 1 ELSE 0 END AS sub_trafficked,
    CASE WHEN r.sub_cat_physical_abuse THEN 1 ELSE 0 END AS sub_physical_abuse,
    CASE WHEN r.sub_cat_sexual_abuse   THEN 1 ELSE 0 END AS sub_sexual_abuse,
    CASE WHEN r.sub_cat_osaec          THEN 1 ELSE 0 END AS sub_osaec,
    CASE WHEN r.sub_cat_child_labor    THEN 1 ELSE 0 END AS sub_child_labor,
    CASE WHEN r.sub_cat_at_risk        THEN 1 ELSE 0 END AS sub_at_risk
FROM residents r
WHERE r.case_status IN ('active', 'under_care')
"""

df_base = pd.read_sql(RESIDENT_BASE_QUERY, engine)
```

### Health Records — Latest + Trend

```python
HEALTH_QUERY = """
SELECT
    resident_id,
    COUNT(*)                                              AS health_record_count,
    AVG(general_health_score)                             AS avg_health_score,
    MAX(general_health_score)                             AS max_health_score,
    MIN(general_health_score)                             AS min_health_score,
    -- Most recent scores
    LAST_VALUE(general_health_score) OVER (
        PARTITION BY resident_id ORDER BY record_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    )                                                     AS latest_health_score,
    LAST_VALUE(nutrition_score) OVER (
        PARTITION BY resident_id ORDER BY record_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    )                                                     AS latest_nutrition_score,
    LAST_VALUE(sleep_quality_score) OVER (
        PARTITION BY resident_id ORDER BY record_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    )                                                     AS latest_sleep_score,
    LAST_VALUE(psychological_checkup_done) OVER (
        PARTITION BY resident_id ORDER BY record_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    )                                                     AS latest_psych_checkup,
    -- Health trend (slope of general_health_score over time)
    REGR_SLOPE(general_health_score, EXTRACT(EPOCH FROM record_date)) AS health_trend
FROM health_wellbeing_records
GROUP BY resident_id, record_date, general_health_score,
         nutrition_score, sleep_quality_score, psychological_checkup_done
"""

HEALTH_AGG_QUERY = """
SELECT
    resident_id,
    COUNT(*)                                                AS health_record_count,
    AVG(general_health_score)                               AS avg_health_score,
    REGR_SLOPE(general_health_score,
               EXTRACT(EPOCH FROM record_date))             AS health_trend,
    MAX(record_date)                                        AS latest_health_date,
    AVG(nutrition_score)                                    AS avg_nutrition,
    AVG(sleep_quality_score)                                AS avg_sleep,
    AVG(psychological_checkup_done::int)                    AS psych_checkup_rate
FROM health_wellbeing_records
GROUP BY resident_id
"""

df_health = pd.read_sql(HEALTH_AGG_QUERY, engine)
```

### Education Records — Progress + Trend

```python
EDUCATION_QUERY = """
SELECT
    resident_id,
    COUNT(*)                                                AS edu_record_count,
    MAX(progress_percent)                                   AS max_edu_progress,
    AVG(progress_percent)                                   AS avg_edu_progress,
    REGR_SLOPE(progress_percent,
               EXTRACT(EPOCH FROM record_date))             AS edu_trend,
    MAX(record_date)                                        AS latest_edu_date,
    AVG(CASE WHEN enrollment_status = 'enrolled' THEN 1.0 ELSE 0.0 END) AS enrollment_rate,
    AVG(attendance_rate)                                    AS avg_attendance,
    MAX(CASE WHEN completion_status = 'completed' THEN 1 ELSE 0 END) AS has_completed
FROM education_records
GROUP BY resident_id
"""

df_edu = pd.read_sql(EDUCATION_QUERY, engine)
```

### Incidents, Sessions, Visitations, Plans

```python
INCIDENTS_QUERY = """
SELECT
    resident_id,
    COUNT(*)                                                AS total_incidents,
    SUM(CASE WHEN severity = 'high' THEN 1 ELSE 0 END)     AS high_severity_incidents,
    SUM(CASE WHEN resolved = false  THEN 1 ELSE 0 END)     AS open_incidents,
    MAX(incident_date)                                      AS latest_incident_date,
    SUM(CASE WHEN incident_date >= CURRENT_DATE - 30
             THEN 1 ELSE 0 END)                             AS incidents_last_30d,
    SUM(CASE WHEN incident_date >= CURRENT_DATE - 90
             AND severity = 'high'
             THEN 1 ELSE 0 END)                             AS high_incidents_last_90d
FROM incident_reports
GROUP BY resident_id
"""

SESSIONS_QUERY = """
SELECT
    resident_id,
    COUNT(*)                                                AS total_sessions,
    AVG(session_duration_minutes)                           AS avg_session_duration,
    SUM(CASE WHEN concerns_flagged = true THEN 1 ELSE 0 END) AS sessions_with_concerns,
    SUM(CASE WHEN progress_noted = true   THEN 1 ELSE 0 END) AS sessions_with_progress,
    SUM(CASE WHEN referral_made = true    THEN 1 ELSE 0 END) AS referrals_made,
    MAX(session_date)                                       AS latest_session_date,
    COUNT(CASE WHEN session_date >= CURRENT_DATE - 30
               THEN 1 END)                                  AS sessions_last_30d
FROM process_recordings
GROUP BY resident_id
"""

VISITATIONS_QUERY = """
SELECT
    resident_id,
    COUNT(*)                                                AS total_visits,
    AVG(CASE WHEN family_cooperation_level = 'good'  THEN 3
             WHEN family_cooperation_level = 'fair'  THEN 2
             WHEN family_cooperation_level = 'poor'  THEN 1
             ELSE NULL END)                                 AS avg_family_cooperation,
    SUM(CASE WHEN safety_concerns_noted = true THEN 1 ELSE 0 END) AS visits_with_safety_concerns,
    MAX(visit_date)                                         AS latest_visit_date,
    AVG(CASE WHEN visit_outcome = 'positive' THEN 1.0 ELSE 0.0 END) AS positive_visit_rate
FROM home_visitations
GROUP BY resident_id
"""

PLANS_QUERY = """
SELECT
    resident_id,
    COUNT(*)                                                AS total_plans,
    AVG(CASE WHEN status = 'completed'   THEN 1.0 ELSE 0.0 END) AS plan_completion_rate,
    SUM(CASE WHEN status = 'overdue'     THEN 1 ELSE 0 END)     AS overdue_plans,
    SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END)     AS active_plans,
    MAX(updated_at)                                         AS latest_plan_update
FROM intervention_plans
GROUP BY resident_id
"""

df_incidents   = pd.read_sql(INCIDENTS_QUERY,   engine)
df_sessions    = pd.read_sql(SESSIONS_QUERY,    engine)
df_visitations = pd.read_sql(VISITATIONS_QUERY, engine)
df_plans       = pd.read_sql(PLANS_QUERY,       engine)
```

### Merge All Sources

```python
df = df_base.copy()
for df_src in [df_health, df_edu, df_incidents, df_sessions, df_visitations, df_plans]:
    df = df.merge(df_src, on='resident_id', how='left')

print(df.shape)
print(df.dtypes)
df.isnull().sum().sort_values(ascending=False).head(20)
```

---

## PHASE 3 — Data Exploration

> Reference: `chapter_two_implementation.md`, `chapter_six_implementation.md`

### Label Distribution Check

```python
# Model A label
risk_map = {'low': 0, 'medium': 1, 'high': 2}
df['initial_risk_num'] = df['initial_risk_level'].map(risk_map)
df['current_risk_num'] = df['current_risk_level'].map(risk_map)
df['risk_escalated']   = (
    (df['current_risk_num'] > df['initial_risk_num']) |
    (df['high_incidents_last_90d'] > 0)
).astype(int)

print(f"Regression risk rate: {df['risk_escalated'].mean():.2%}")

# Model B label
df['reintegration_ready'] = (
    (df['reintegration_status'].isin(['ready', 'in_progress'])) &
    (df['avg_edu_progress'].fillna(0) >= 70) &
    (df['high_incidents_last_90d'].fillna(0) == 0)
).astype(int)

print(f"Reintegration readiness rate: {df['reintegration_ready'].mean():.2%}")
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
# Watch for: high missing rates in sessions, visitations (some residents may have few records)
# Watch for: |Skew| > 1 on incident counts, session counts → log1p transform
```

---

## PHASE 4 — Data Preparation

> Reference: `chapter_seven_implementation.md`

### Feature Set

| Feature | Source | Type | Notes |
|---|---|---|---|
| `avg_health_score` | health records | Numeric | Core wellbeing signal |
| `health_trend` | health records | Numeric | Negative = declining — critical flag |
| `avg_nutrition` | health records | Numeric | |
| `avg_sleep` | health records | Numeric | |
| `psych_checkup_rate` | health records | Numeric | Low = neglected psych care |
| `avg_edu_progress` | education records | Numeric | |
| `edu_trend` | education records | Numeric | Slope over time |
| `avg_attendance` | education records | Numeric | |
| `enrollment_rate` | education records | Numeric | |
| `total_incidents` | incident reports | Numeric | Log-transform |
| `high_severity_incidents` | incident reports | Numeric | High weight feature |
| `incidents_last_30d` | incident reports | Numeric | Recency matters |
| `open_incidents` | incident reports | Numeric | Unresolved = active risk |
| `sessions_last_30d` | process recordings | Numeric | Low = disengaged |
| `sessions_with_concerns` | process recordings | Numeric | |
| `sessions_with_progress` | process recordings | Numeric | Positive signal |
| `avg_family_cooperation` | home visitations | Numeric | Low = reintegration risk |
| `visits_with_safety_concerns` | home visitations | Numeric | |
| `positive_visit_rate` | home visitations | Numeric | |
| `plan_completion_rate` | intervention plans | Numeric | Low = falling behind |
| `overdue_plans` | intervention plans | Numeric | |
| `length_of_stay` | residents | Numeric | |
| `present_age` | residents | Numeric | |
| `case_category` | residents | Categorical | Encode |
| `sub_trafficked` et al. | residents | Binary | Already encoded in query |
| `is_pwd` | residents | Binary | |
| `has_special_needs` | residents | Binary | |
| `family_is_4ps` | residents | Binary | |

**Drop before modeling:** `resident_id`, `safehouse_id` (identifier), `assigned_social_worker` (identifier), raw risk level columns (used only for label construction)

### Prep Pipeline

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Log-transform skewed count features
for col in ['total_incidents', 'high_severity_incidents', 'total_sessions', 'total_visits']:
    df[f'{col}_log'] = np.log1p(df[col].fillna(0))

# Collapse rare case categories
freq = df['case_category'].value_counts(normalize=True)
rare = freq[freq < 0.05].index
df['case_category'] = df['case_category'].replace(rare, 'Other')

num_cols = [
    'avg_health_score', 'health_trend', 'avg_nutrition', 'avg_sleep', 'psych_checkup_rate',
    'avg_edu_progress', 'edu_trend', 'avg_attendance', 'enrollment_rate',
    'incidents_last_30d', 'open_incidents', 'sessions_last_30d',
    'sessions_with_concerns', 'sessions_with_progress', 'avg_family_cooperation',
    'visits_with_safety_concerns', 'positive_visit_rate', 'plan_completion_rate',
    'overdue_plans', 'length_of_stay', 'present_age',
    'total_incidents_log', 'high_severity_incidents_log',
    'total_sessions_log', 'total_visits_log',
    'sub_trafficked', 'sub_physical_abuse', 'sub_sexual_abuse',
    'sub_osaec', 'sub_child_labor', 'sub_at_risk',
    'is_pwd', 'has_special_needs', 'family_is_4ps', 'family_solo_parent'
]
cat_cols = ['case_category']

numeric_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])
categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', numeric_pipe, num_cols),
    ('cat', categorical_pipe, cat_cols)
])

# Separate labels
y_risk   = df['risk_escalated']
y_reint  = df['reintegration_ready']
X        = df[num_cols + cat_cols]
```

---

## PHASE 5 — Modeling (Classification)

> Reference: `chapter_thirteen_implementation.md`, `chapter_fourteen_implementation.md`

### Model A — Regression Risk

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(
    X, y_risk, test_size=0.20, random_state=42, stratify=y_risk
)

models_A = [
    (LogisticRegression(max_iter=1000, class_weight='balanced'), 'Logistic Regression'),
    (RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42), 'Random Forest'),
    (GradientBoostingClassifier(n_estimators=100, random_state=42), 'Gradient Boosting'),
]

for Model, name in models_A:
    pipe = Pipeline([('prep', preprocessor), ('model', Model)])
    pipe.fit(X_train_A, y_train_A)
    y_pred  = pipe.predict(X_test_A)
    y_proba = pipe.predict_proba(X_test_A)[:, 1]
    print(f"\nModel A — {name}")
    print(f"ROC-AUC: {roc_auc_score(y_test_A, y_proba):.4f}")
    print(classification_report(y_test_A, y_pred))
    # Recall on class 1 (risk escalated) is the primary metric
    # A missed struggling resident (false negative) is worse than a false alarm
```

### Model B — Reintegration Readiness

```python
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(
    X, y_reint, test_size=0.20, random_state=42, stratify=y_reint
)

models_B = [
    (LogisticRegression(max_iter=1000, class_weight='balanced'), 'Logistic Regression'),
    (RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42), 'Random Forest'),
    (GradientBoostingClassifier(n_estimators=100, random_state=42), 'Gradient Boosting'),
]

for Model, name in models_B:
    pipe = Pipeline([('prep', preprocessor), ('model', Model)])
    pipe.fit(X_train_B, y_train_B)
    y_pred  = pipe.predict(X_test_B)
    y_proba = pipe.predict_proba(X_test_B)[:, 1]
    print(f"\nModel B — {name}")
    print(f"ROC-AUC: {roc_auc_score(y_test_B, y_proba):.4f}")
    print(classification_report(y_test_B, y_pred))
```

**Note on small dataset:** The resident population is small (likely < 500 active residents). With a small N, Random Forest with `class_weight='balanced'` and moderate `max_depth` is preferable to Gradient Boosting, which is more prone to overfitting on small datasets. Monitor the train vs. validation AUC gap closely in Phase 6.

---

## PHASE 6 — Evaluation, Selection & Tuning

> Reference: `chapter_fifteen_implementation.md`

### Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold, cross_validate

# Use fewer folds if dataset is small (n < 200 → use 3-fold)
n_splits = 3 if len(X) < 200 else 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for label_name, y_cv, X_cv in [('Regression Risk', y_risk, X),
                                ('Reintegration Readiness', y_reint, X)]:
    rf_pipe = Pipeline([
        ('prep', preprocessor),
        ('model', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    ])
    cv_results = cross_validate(
        rf_pipe, X_cv, y_cv, cv=cv,
        scoring=['roc_auc', 'recall', 'f1'],
        return_train_score=True
    )
    print(f"\n{label_name}")
    print(f"Val AUC:    {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
    print(f"Val Recall: {cv_results['test_recall'].mean():.4f}")
    # Large train/val gap on a small dataset → reduce max_depth, increase min_samples_leaf
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'model__n_estimators':     [50, 100, 200],
    'model__max_depth':        [3, 4, 5, 6, None],
    'model__min_samples_leaf': [5, 10, 15, 20],  # higher values preferred for small n
    'model__max_features':     ['sqrt', 'log2'],
}

# Tune Model A (regression risk)
rf_A = Pipeline([('prep', preprocessor),
                 ('model', RandomForestClassifier(class_weight='balanced', random_state=42))])
rnd_A = RandomizedSearchCV(rf_A, param_dist, n_iter=20, cv=cv,
                            scoring='recall', random_state=42, n_jobs=-1)
rnd_A.fit(X_train_A, y_train_A)
print(f"Model A best params: {rnd_A.best_params_}")

# Tune Model B (reintegration readiness)
rf_B = Pipeline([('prep', preprocessor),
                 ('model', RandomForestClassifier(class_weight='balanced', random_state=42))])
rnd_B = RandomizedSearchCV(rf_B, param_dist, n_iter=20, cv=cv,
                            scoring='roc_auc', random_state=42, n_jobs=-1)
rnd_B.fit(X_train_B, y_train_B)
print(f"Model B best params: {rnd_B.best_params_}")
```

### Final Test-Set Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

for label_name, model, X_test, y_test in [
    ('Regression Risk',        rnd_A.best_estimator_, X_test_A, y_test_A),
    ('Reintegration Readiness',rnd_B.best_estimator_, X_test_B, y_test_B),
]:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"\n=== {label_name} ===")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(confusion_matrix(y_test, y_pred))

# Gate checks
auc_A    = roc_auc_score(y_test_A, rnd_A.best_estimator_.predict_proba(X_test_A)[:, 1])
recall_A = classification_report(y_test_A, rnd_A.best_estimator_.predict(X_test_A),
                                  output_dict=True)['1']['recall']
assert auc_A    >= 0.75, f"Model A AUC {auc_A:.4f} below threshold"
assert recall_A >= 0.75, f"Model A Recall {recall_A:.4f} below threshold"
```

---

## PHASE 7 — Feature Selection & Explainability

> Reference: `chapter_sixteen_implementation.md`

### Feature Importance

```python
import pandas as pd

def get_feature_names(model, num_cols, cat_cols):
    cat_names = (model.named_steps['prep']
                 .named_transformers_['cat']
                 .named_steps['onehot']
                 .get_feature_names_out(cat_cols).tolist())
    return num_cols + cat_names

for label_name, model in [('Regression Risk',         rnd_A.best_estimator_),
                           ('Reintegration Readiness', rnd_B.best_estimator_)]:
    feat_names   = get_feature_names(model, num_cols, cat_cols)
    importances  = pd.Series(
        model.named_steps['model'].feature_importances_,
        index=feat_names
    ).sort_values(ascending=False)
    print(f"\nTop 10 features — {label_name}")
    print(importances.head(10))
```

### Generating `top_concern_factors` and `top_strength_factors`

```python
CONCERN_LABELS = {
    'health_trend':              'Declining health scores',
    'incidents_last_30d':        'Recent incidents filed',
    'open_incidents':            'Unresolved incidents',
    'sessions_with_concerns':    'Concerns flagged in sessions',
    'overdue_plans':             'Overdue intervention plans',
    'visits_with_safety_concerns':'Safety concerns noted during family visit',
    'avg_attendance':            'Low school attendance',
    'plan_completion_rate':      'Low intervention plan completion',
    'sessions_last_30d':         'Few recent counseling sessions',
}

STRENGTH_LABELS = {
    'sessions_with_progress':    'Progress noted in counseling sessions',
    'avg_edu_progress':          'Education progress improving',
    'edu_trend':                 'Education progress trending upward',
    'positive_visit_rate':       'Positive family visitation outcomes',
    'avg_health_score':          'Stable health scores',
    'plan_completion_rate':      'Intervention plans on track',
    'avg_family_cooperation':    'Good family cooperation',
}

def get_top_factors(model, feature_names, label_map, top_n=3):
    importances = model.named_steps['model'].feature_importances_
    top_indices = importances.argsort()[::-1]
    factors = []
    for i in top_indices:
        name  = feature_names[i]
        label = label_map.get(name)
        if label:
            factors.append(label)
        if len(factors) == top_n:
            break
    return factors
```

---

## PHASE 8 — Deployment

> Reference: `chapter_seventeen_implementation.md`
> **Access restriction:** The `resident_risk_scores` table must be restricted at the Supabase row-level security level. Only users with `Admin` or `SocialWorker` roles may read from it. This must be configured in Supabase before the table is populated.

### Architecture

```
Supabase PostgreSQL
  (residents + health + education + incidents + sessions + visitations + plans)
    ↓  direct psycopg2 connection
Training Job (Azure Container Apps Job — bi-weekly)
    ↓  two model artifacts saved to
Azure Blob Storage
    resident_risk/risk_model.pkl
    resident_risk/reint_model.pkl
    resident_risk/risk_metadata.json
    ↓  loaded by
Inference Job (score_residents.py)
    ↓  upserts scores to
Supabase PostgreSQL
    resident_risk_scores  ← RLS restricted to Admin + SocialWorker
    ↓  read by
.NET API (ResidentsController — authenticated, role-checked)
    ↓  served to
React Admin Dashboard (Resident Case Board — risk tier + reintegration readiness columns)
```

### `resident_risk_scores` Table + Row-Level Security (Supabase — one-time setup)

```sql
CREATE TABLE resident_risk_scores (
    resident_id              TEXT PRIMARY KEY REFERENCES residents(resident_id),
    regression_risk_score    FLOAT NOT NULL,
    regression_risk_tier     TEXT NOT NULL CHECK (regression_risk_tier IN ('high','medium','low')),
    reintegration_score      FLOAT NOT NULL,
    reintegration_tier       TEXT NOT NULL CHECK (reintegration_tier IN ('ready','in_progress','not_ready')),
    top_concern_factors      JSONB,
    top_strength_factors     JSONB,
    scored_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version            TEXT NOT NULL
);

CREATE INDEX idx_risk_scores_tier ON resident_risk_scores (regression_risk_tier);

-- Row-level security: only staff roles may read
ALTER TABLE resident_risk_scores ENABLE ROW LEVEL SECURITY;

CREATE POLICY "staff_read_only" ON resident_risk_scores
    FOR SELECT
    USING (auth.jwt() ->> 'role' IN ('admin', 'social_worker'));
```

### Training Job (`train_risk.py`)

```python
import pickle, json, os
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

# 1. Load + merge all sources (Phase 2)
df = load_and_merge(engine)            # defined in features.py

# 2. Build labels + features (Phases 3–4)
df = build_labels(df)                  # defined in features.py
df = engineer_features(df)
y_risk  = df['risk_escalated']
y_reint = df['reintegration_ready']
X       = df[num_cols + cat_cols]

model_version = datetime.now().strftime('%Y%m%d_%H%M')

for label_name, y, artifact_name, auc_threshold, recall_threshold in [
    ('risk',          y_risk,  'risk_model.pkl',  0.75, 0.75),
    ('reintegration', y_reint, 'reint_model.pkl', 0.70, 0.65),
]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = Pipeline([
        ('prep',  preprocessor),
        ('model', RandomForestClassifier(
            n_estimators=200, max_depth=5,
            min_samples_leaf=10, class_weight='balanced', random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)
    recall  = classification_report(y_test, pipeline.predict(X_test),
                                     output_dict=True)['1']['recall']

    assert auc    >= auc_threshold,    f"{label_name} AUC {auc:.4f} below threshold"
    assert recall >= recall_threshold, f"{label_name} Recall {recall:.4f} below threshold"

    with open(f'/tmp/{artifact_name}', 'wb') as f:
        pickle.dump(pipeline, f)

    upload_artifact(f'/tmp/{artifact_name}', f'resident_risk/{artifact_name}')
    upload_artifact(f'/tmp/{artifact_name}', f'resident_risk/versions/{artifact_name.replace(".pkl", "")}_{model_version}.pkl')

    print(f"{label_name} — AUC: {auc:.4f} | Recall: {recall:.4f}")

# Save shared metadata
metadata = {'trained_at': datetime.now().isoformat(), 'model_version': model_version,
            'n_residents': len(df), 'risk_escalation_rate': float(y_risk.mean()),
            'reintegration_rate': float(y_reint.mean()), 'features': list(X.columns)}
with open('/tmp/risk_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
upload_artifact('/tmp/risk_metadata.json', 'resident_risk/risk_metadata.json')
```

### Inference Job (`score_residents.py`)

```python
import pickle, json, os
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
from dotenv import load_dotenv
from storage.blob_client import load_model_from_blob, load_json_from_blob

load_dotenv()
engine = create_engine(os.getenv("SUPABASE_DB_URL"))

risk_model  = load_model_from_blob('resident_risk/risk_model.pkl')
reint_model = load_model_from_blob('resident_risk/reint_model.pkl')
metadata    = load_json_from_blob('resident_risk/risk_metadata.json')

df = load_and_merge(engine)
df = engineer_features(df)
resident_ids = df['resident_id'].tolist()
X = df[num_cols + cat_cols]

risk_proba  = risk_model.predict_proba(X)[:, 1]
reint_proba = reint_model.predict_proba(X)[:, 1]

feat_names = get_feature_names(risk_model, num_cols, cat_cols)

def risk_tier(p):
    return 'high' if p >= 0.70 else 'medium' if p >= 0.40 else 'low'

def reint_tier(p):
    return 'ready' if p >= 0.70 else 'in_progress' if p >= 0.40 else 'not_ready'

results = pd.DataFrame({
    'resident_id':           resident_ids,
    'regression_risk_score': risk_proba.round(4),
    'regression_risk_tier':  [risk_tier(p) for p in risk_proba],
    'reintegration_score':   reint_proba.round(4),
    'reintegration_tier':    [reint_tier(p) for p in reint_proba],
    'top_concern_factors':   [json.dumps(get_top_factors(risk_model, feat_names, CONCERN_LABELS))
                               for _ in resident_ids],
    'top_strength_factors':  [json.dumps(get_top_factors(reint_model, feat_names, STRENGTH_LABELS))
                               for _ in resident_ids],
    'scored_at':             datetime.now().isoformat(),
    'model_version':         metadata['model_version']
})

with engine.begin() as conn:
    for _, row in results.iterrows():
        conn.execute(text("""
            INSERT INTO resident_risk_scores
                (resident_id, regression_risk_score, regression_risk_tier,
                 reintegration_score, reintegration_tier,
                 top_concern_factors, top_strength_factors, scored_at, model_version)
            VALUES
                (:resident_id, :regression_risk_score, :regression_risk_tier,
                 :reintegration_score, :reintegration_tier,
                 :top_concern_factors::jsonb, :top_strength_factors::jsonb,
                 :scored_at, :model_version)
            ON CONFLICT (resident_id) DO UPDATE SET
                regression_risk_score = EXCLUDED.regression_risk_score,
                regression_risk_tier  = EXCLUDED.regression_risk_tier,
                reintegration_score   = EXCLUDED.reintegration_score,
                reintegration_tier    = EXCLUDED.reintegration_tier,
                top_concern_factors   = EXCLUDED.top_concern_factors,
                top_strength_factors  = EXCLUDED.top_strength_factors,
                scored_at             = EXCLUDED.scored_at,
                model_version         = EXCLUDED.model_version
        """), row.to_dict())

print(f"Scored {len(results)} residents — "
      f"High risk: {(results['regression_risk_tier']=='high').sum()} | "
      f"Ready for reintegration: {(results['reintegration_tier']=='ready').sum()}")
```

### FastAPI Endpoint Addition (`main.py`)

```python
@app.post("/score/residents", dependencies=[Depends(verify_key)])
def trigger_resident_scoring():
    """
    Score all active residents. Internal only — never exposed to donor-facing surfaces.
    Caller must have ML_SERVICE_API_KEY.
    """
    result = subprocess.run(
        ["python", "pipelines/resident_risk/score_residents.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr)
    return {"status": "resident scoring complete", "output": result.stdout}
```

### Retraining Schedule

| Trigger | Action |
|---|---|
| Bi-weekly (1st and 15th of month) | Full retrain + score all active residents |
| New health/education record added | Score only — no retrain |
| New incident filed | Score only — immediate rescore for that resident's safehouse |
| Resident admitted or case status changes | Include in next scoring run |

### Monitoring Checklist

- [ ] **Access control verification** — confirm Supabase RLS policy is active before first data load; test that unauthenticated reads return empty
- [ ] **High-risk alert coverage** — after each scoring run, log count of `high` tier residents per safehouse; alert if any safehouse has > 30% high risk
- [ ] **Label drift** — monitor risk escalation rate over time; sharp increases may indicate data quality issues rather than real deterioration
- [ ] **Small dataset overfitting** — monitor train vs. val AUC gap at each retrain; gap > 0.10 triggers manual review before deploying
- [ ] **Recall gate enforcement** — Model A recall ≥ 0.75 is a hard deploy gate; no model ships below this threshold for the regression risk classifier
- [ ] **Artifact versioning** — both model files versioned in Blob Storage; rollback procedure: copy prior version files to `resident_risk/risk_model.pkl` and `reint_model.pkl`, then trigger scoring run
