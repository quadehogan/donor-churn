# Donation-to-Outcome Attribution Pipeline
## Lighthouse Sanctuary — ML Pipeline Plan

> Follows the phase structure from `MASTER_ML_PIPELINE_GUIDE.md`. This pipeline is not a pure prediction problem — it is a causal linkage and regression analysis that connects donation allocations to measurable resident outcomes. The goal is to produce personalized, data-backed impact statements for donor communications.

---

## Repo Placement

```
lighthouse-ml/
├── Dockerfile
├── requirements.txt
├── main.py
├── pipelines/
│   ├── churn/                        ← donor churn (deployed)
│   └── impact_attribution/           ← THIS PIPELINE
│       ├── features.py               ← SQL queries + feature engineering
│       ├── train_impact.py           ← regression model training
│       ├── score_impact.py           ← generate impact statements + upsert
│       └── statement_builder.py      ← converts model output to readable text
├── storage/
│   └── blob_client.py                ← shared
└── db/
    └── connection.py                 ← shared
```

---

## Quick-Reference Pipeline Map

```
PHASE 1 — PLAN          Define linkage problem, time windows, output format
PHASE 2 — ACQUIRE       Join donation_allocations + outcome tables via Supabase
PHASE 3 — EXPLORE       Explore outcome deltas over time, unistats(), correlation matrix
PHASE 4 — PREPARE       Compute outcome deltas, time-window features, aggregate by safehouse + program area
PHASE 5 — MODEL         OLS regression (statsmodels) — effect size + direction per program area
PHASE 6 — EVALUATE      R², MAE, confidence intervals, effect size significance
PHASE 7 — INTERPRET     Coefficient extraction → personalized statement generation
PHASE 8 — DEPLOY        Azure Container Apps → donor_impact_statements table in Supabase
```

---

## PHASE 1 — Project Planning (CRISP-DM)

### Feasibility Checklist

| Gate | Question | Status |
|---|---|---|
| Business | What specific problem is being solved? | Show each donor how their specific allocation improved resident outcomes — not generic impact claims |
| Data | Is live, updatable data available? | Yes — `donation_allocations`, `education_records`, `health_wellbeing_records`, `safehouse_monthly_metrics` all in Supabase |
| Analytical | Can data support reliable linkage? | Yes — allocations are tagged by `program_area` and `safehouse_id`; outcomes are tracked monthly per safehouse |
| Integration | Can outputs plug into existing systems? | Yes — impact statements stored in `donor_impact_statements` table; .NET API reads and includes in donor communications |
| Risk | Privacy concerns? | Low — outputs are aggregate safehouse-level statistics, never individual resident data |

### Problem Definition

This is a **correlational regression**, not a prediction model. The goal is to quantify and communicate the relationship between donation allocations and measurable outcome improvements at the safehouse level, over a defined time window.

**Unit of analysis:** One row = one (supporter, safehouse, program_area, time_window) combination.

**The core question per row:** For donor X who allocated $Y to program area Z at safehouse SH-N, what was the measured change in the relevant outcome metric over the following M months?

**Output per donor:** One or more personalized impact statements, e.g.:
- *"Your $150 allocated to Education at Safehouse Cebu was associated with a 14% improvement in average education progress over the following 3 months."*
- *"Your $200 to Health & Wellness at Safehouse Davao coincided with an improvement in average health scores from 6.2 to 7.8 over 6 months."*

### Program Area → Outcome Metric Mapping

| Program Area | Primary Outcome Metric | Source Table |
|---|---|---|
| Education | `avg_education_progress` | `safehouse_monthly_metrics` |
| Health & Wellness | `avg_health_score` | `safehouse_monthly_metrics` |
| Counseling & Care | `process_recording_count` (proxy for session activity) | `safehouse_monthly_metrics` |
| Legal / Case Mgmt | `intervention_plan` completion rate | `intervention_plans` |
| Operations | `active_residents` stability | `safehouse_monthly_metrics` |

### Time Windows

Three windows are computed for each allocation — use the one with the strongest signal in Phase 5:
- Short: 1–3 months post-allocation
- Medium: 3–6 months post-allocation
- Long: 6–12 months post-allocation

### Success Criteria

- At least 3 program areas produce statistically significant correlations (p < 0.05) with interpretable effect sizes
- Generated statements are factually grounded — only produced when the correlation is significant and the effect size is at least small (Cohen's f ≥ 0.1)
- Statement generation covers ≥ 60% of active donors (some donors may not have enough allocation history for a statement)

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

### Core Query — Allocation + Safehouse Outcomes

Pull allocation amounts per (supporter, safehouse, program_area, month) and join to monthly outcome metrics. The time window join computes outcome values at the time of donation and N months later.

```python
ALLOCATION_OUTCOMES_QUERY = """
WITH allocation_months AS (
    SELECT
        da.allocation_id,
        da.donation_id,
        da.safehouse_id,
        da.program_area,
        da.amount_allocated,
        da.allocation_date,
        DATE_TRUNC('month', da.allocation_date)                AS alloc_month,
        d.supporter_id
    FROM donation_allocations da
    JOIN donations d ON da.donation_id = d.donation_id
),
outcomes AS (
    SELECT
        safehouse_id,
        month_start,
        avg_education_progress,
        avg_health_score,
        process_recording_count,
        active_residents,
        incident_count
    FROM safehouse_monthly_metrics
)
SELECT
    am.allocation_id,
    am.supporter_id,
    am.safehouse_id,
    am.program_area,
    am.amount_allocated,
    am.allocation_date,
    am.alloc_month,

    -- Outcome at time of allocation
    o_base.avg_education_progress   AS edu_baseline,
    o_base.avg_health_score         AS health_baseline,
    o_base.process_recording_count  AS sessions_baseline,
    o_base.incident_count           AS incidents_baseline,

    -- Outcome 3 months later
    o_3m.avg_education_progress     AS edu_3m,
    o_3m.avg_health_score           AS health_3m,
    o_3m.process_recording_count    AS sessions_3m,
    o_3m.incident_count             AS incidents_3m,

    -- Outcome 6 months later
    o_6m.avg_education_progress     AS edu_6m,
    o_6m.avg_health_score           AS health_6m,
    o_6m.process_recording_count    AS sessions_6m,
    o_6m.incident_count             AS incidents_6m

FROM allocation_months am
LEFT JOIN outcomes o_base
    ON am.safehouse_id = o_base.safehouse_id
    AND am.alloc_month = o_base.month_start
LEFT JOIN outcomes o_3m
    ON am.safehouse_id = o_3m.safehouse_id
    AND (am.alloc_month + INTERVAL '3 months') = o_3m.month_start
LEFT JOIN outcomes o_6m
    ON am.safehouse_id = o_6m.safehouse_id
    AND (am.alloc_month + INTERVAL '6 months') = o_6m.month_start
WHERE am.amount_allocated > 0
ORDER BY am.supporter_id, am.allocation_date
"""

df = pd.read_sql(ALLOCATION_OUTCOMES_QUERY, engine)

print(df.shape)
print(df.dtypes)
df.isnull().sum()
df.head()
```

---

## PHASE 3 — Data Exploration

> Reference: `chapter_two_implementation.md`, `chapter_six_implementation.md`

### Immediate Inspection

```python
print(df['program_area'].value_counts())
print(df['safehouse_id'].value_counts())
print(f"Unique supporters with allocations: {df['supporter_id'].nunique()}")
print(f"Date range: {df['allocation_date'].min()} → {df['allocation_date'].max()}")
print(f"Rows with 3m outcome data: {df['edu_3m'].notna().sum()}")
print(f"Rows with 6m outcome data: {df['edu_6m'].notna().sum()}")
```

### Outcome Delta Exploration

```python
import numpy as np

# Compute deltas
df['edu_delta_3m']     = df['edu_3m']     - df['edu_baseline']
df['health_delta_3m']  = df['health_3m']  - df['health_baseline']
df['edu_delta_6m']     = df['edu_6m']     - df['edu_baseline']
df['health_delta_6m']  = df['health_6m']  - df['health_baseline']

# Summarize deltas by program area
print(df.groupby('program_area')[['edu_delta_3m', 'health_delta_3m']].describe())
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

stats = unistats(df[['amount_allocated', 'edu_delta_3m', 'health_delta_3m',
                      'edu_delta_6m', 'health_delta_6m']])
# Watch for: |Skew| > 1 on amount_allocated → log1p transform
# Watch for: near-zero variance in deltas for certain program areas → insufficient data
```

---

## PHASE 4 — Data Preparation

> Reference: `chapter_seven_implementation.md`

### Feature Engineering

```python
import numpy as np

# Log-transform allocation amount (right-skewed)
df['log_amount'] = np.log1p(df['amount_allocated'])

# Percent change deltas (more interpretable for statements than raw delta)
df['edu_pct_change_3m'] = (
    (df['edu_3m'] - df['edu_baseline']) / df['edu_baseline'].replace(0, np.nan)
) * 100

df['health_pct_change_3m'] = (
    (df['health_3m'] - df['health_baseline']) / df['health_baseline'].replace(0, np.nan)
) * 100

df['edu_pct_change_6m'] = (
    (df['edu_6m'] - df['edu_baseline']) / df['edu_baseline'].replace(0, np.nan)
) * 100

df['health_pct_change_6m'] = (
    (df['health_6m'] - df['health_baseline']) / df['health_baseline'].replace(0, np.nan)
) * 100

# Collapse rare program areas
freq = df['program_area'].value_counts(normalize=True)
rare = freq[freq < 0.05].index
df['program_area'] = df['program_area'].replace(rare, 'Other')

# One-hot encode program area and safehouse
df = pd.get_dummies(df, columns=['program_area', 'safehouse_id'], drop_first=True)
df[df.select_dtypes(bool).columns] = df.select_dtypes(bool).astype(int)

# Drop rows where baseline or outcome is null (can't compute delta)
df_edu    = df.dropna(subset=['edu_baseline', 'edu_3m']).copy()
df_health = df.dropna(subset=['health_baseline', 'health_3m']).copy()
```

### Separate Models per Outcome

Run a separate OLS model per outcome metric. This keeps the coefficients interpretable per program area and avoids mixing signals across unrelated outcomes.

```python
OUTCOME_MODELS = {
    'education_3m':   {'df': df_edu,    'label': 'edu_pct_change_3m',    'window': 3},
    'health_3m':      {'df': df_health, 'label': 'health_pct_change_3m', 'window': 3},
    'education_6m':   {'df': df_edu,    'label': 'edu_pct_change_6m',    'window': 6},
    'health_6m':      {'df': df_health, 'label': 'health_pct_change_6m', 'window': 6},
}
```

---

## PHASE 5 — Modeling (Explanatory OLS Regression)

> Reference: `chapter_nine_implementation.md`, `chapter_ten_implementation.md`

Use **statsmodels OLS** here, not sklearn. The goal is to understand effect directions and magnitudes — the coefficients are the output, not predictions. The coefficient on `log_amount` within each program area group tells us: "a 1-unit increase in log(allocation) is associated with an X% change in the outcome."

```python
import statsmodels.api as sm

results_store = {}

for model_name, config in OUTCOME_MODELS.items():
    df_m  = config['df'].copy()
    label = config['label']

    feature_cols = ['log_amount'] + [c for c in df_m.columns
                                     if c.startswith('program_area_')
                                     or c.startswith('safehouse_id_')]

    y = df_m[label]
    X = df_m[feature_cols].assign(const=1)

    model  = sm.OLS(y, X).fit()
    results_store[model_name] = model
    print(f"\n=== {model_name} ===")
    print(model.summary())
```

### Reading the OLS Summary for Impact Statements

| Field | What it means for impact statements |
|---|---|
| `coef` on `log_amount` | For a 1-unit increase in log(allocation), outcome changes by this much — the core signal |
| `P>|t|` | Only generate statements where p < 0.05 — weak signals should not become donor claims |
| `[0.025, 0.975]` | Confidence interval — use this to bound the statement ("between X% and Y%") |
| `R²` | How much of outcome variance is explained — low R² means other factors dominate (expected) |

### OLS Diagnostics (NMALH)

```python
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

for model_name, model in results_store.items():
    print(f"\n--- Diagnostics: {model_name} ---")

    # 1. Normality
    stat, p = stats.normaltest(model.resid)
    print(f"Normality p={p:.4f} {'OK' if p > 0.05 else 'FAIL — consider transform'}")

    # 2. Durbin-Watson (autocorrelation)
    dw = durbin_watson(model.resid)
    print(f"Durbin-Watson={dw:.2f} {'OK' if 1.5 < dw < 2.5 else 'WARNING'}")

    # 3. Breusch-Pagan (heteroscedasticity)
    _, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
    print(f"Breusch-Pagan p={bp_p:.4f} {'OK' if bp_p > 0.05 else 'Use HC3 SEs'}")

    if bp_p < 0.05:
        results_store[model_name] = model.get_robustcov_results(cov_type='HC3')
        print("→ Refitted with HC3 robust standard errors")
```

---

## PHASE 6 — Evaluation & Selection

> Reference: `chapter_fifteen_implementation.md`

### Metrics for Explanatory Models

For impact attribution, R² matters less than coefficient significance and interpretability. A low R² is expected — safehouse outcomes are influenced by many factors beyond donations alone. What matters is whether the donation signal is detectable and consistent.

```python
for model_name, model in results_store.items():
    print(f"\n{model_name}")
    print(f"R²: {model.rsquared:.4f} | Adj R²: {model.rsquared_adj:.4f}")

    # Extract significant coefficients
    coef_df = pd.DataFrame({
        'coef':   model.params,
        'pvalue': model.pvalues,
        'ci_low': model.conf_int()[0],
        'ci_high':model.conf_int()[1]
    })

    sig = coef_df[coef_df['pvalue'] < 0.05].sort_values('pvalue')
    print("Significant features:")
    print(sig)
```

### Effect Size Check (Cohen's f²)

```python
def cohens_f2(r_squared):
    """Cohen's f² for OLS. Small ≥ 0.02, Medium ≥ 0.15, Large ≥ 0.35"""
    return r_squared / (1 - r_squared)

for model_name, model in results_store.items():
    f2 = cohens_f2(model.rsquared)
    size = 'large' if f2 >= 0.35 else 'medium' if f2 >= 0.15 else 'small' if f2 >= 0.02 else 'negligible'
    print(f"{model_name}: f²={f2:.4f} ({size})")

# Gate: only generate impact statements for program areas with
# significant coefficients (p < 0.05) AND effect size ≥ small (f² ≥ 0.02)
```

---

## PHASE 7 — Interpretation & Statement Generation

### Extracting Significant Effects per Program Area

```python
def extract_significant_effects(results_store: dict, p_threshold=0.05) -> dict:
    """
    Returns a dict of {program_area: {outcome: {coef, ci_low, ci_high, window}}}
    for all effects that pass the significance gate.
    """
    effects = {}
    for model_name, model in results_store.items():
        outcome, window = model_name.rsplit('_', 1)
        coef_df = pd.DataFrame({
            'coef':    model.params,
            'pvalue':  model.pvalues,
            'ci_low':  model.conf_int()[0],
            'ci_high': model.conf_int()[1]
        })
        prog_cols = coef_df[coef_df.index.str.startswith('program_area_')]
        sig       = prog_cols[prog_cols['pvalue'] < p_threshold]
        for feat, row in sig.iterrows():
            prog = feat.replace('program_area_', '').replace('_', ' ').title()
            effects.setdefault(prog, {})[outcome] = {
                'coef':    round(row['coef'], 2),
                'ci_low':  round(row['ci_low'], 2),
                'ci_high': round(row['ci_high'], 2),
                'window':  int(window)
            }
    return effects
```

### Statement Builder (`statement_builder.py`)

```python
OUTCOME_LABELS = {
    'education': 'average education progress',
    'health':    'average health and wellbeing scores',
}

DIRECTION_WORDS = {
    True:  ('improvement in', 'increase in'),
    False: ('decline in',     'decrease in'),
}

def build_impact_statement(
    supporter_id: str,
    safehouse_name: str,
    program_area: str,
    amount: float,
    outcome: str,
    coef: float,
    ci_low: float,
    ci_high: float,
    window: int,
    baseline: float
) -> str:
    """
    Generates a single personalized impact statement for a donor.
    Only called when the underlying effect is statistically significant.
    """
    import math
    # Estimate pct change for this donor's actual allocation amount
    estimated_pct = coef * math.log1p(amount)
    direction      = estimated_pct >= 0
    direction_word = DIRECTION_WORDS[direction][0]
    outcome_label  = OUTCOME_LABELS.get(outcome, outcome)

    statement = (
        f"Your ${amount:,.0f} allocated to {program_area} at {safehouse_name} "
        f"was associated with a {abs(estimated_pct):.1f}% {direction_word} "
        f"{outcome_label} over the following {window} months "
        f"(estimated range: {abs(ci_low * math.log1p(amount)):.1f}%–"
        f"{abs(ci_high * math.log1p(amount)):.1f}%)."
    )
    return statement
```

---

## PHASE 8 — Deployment

> Reference: `chapter_seventeen_implementation.md`
> This pipeline produces text output (impact statements) rather than prediction scores. The deployment pattern is the same — batch job writes to Supabase, .NET API reads and includes in donor communications.

### Architecture

```
Supabase PostgreSQL
    ↓  donation_allocations + safehouse_monthly_metrics
train_impact.py (Azure Container Apps Job — quarterly)
    ↓  saves OLS model coefficients + significant effects
Azure Blob Storage
    impact/impact_effects.json       ← significant effects per program area
    impact/impact_metadata.json
    ↓  loaded by
score_impact.py
    ↓  generates per-donor statements
Supabase PostgreSQL
    donor_impact_statements
    ↓  read by
.NET API → donor communication emails / dashboard
```

### `donor_impact_statements` Table (Supabase — one-time setup)

```sql
CREATE TABLE donor_impact_statements (
    statement_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supporter_id        TEXT NOT NULL REFERENCES supporters(supporter_id),
    safehouse_id        TEXT NOT NULL,
    program_area        TEXT NOT NULL,
    allocation_amount   FLOAT NOT NULL,
    outcome_metric      TEXT NOT NULL,
    time_window_months  INT NOT NULL,
    estimated_pct_change FLOAT NOT NULL,
    statement_text      TEXT NOT NULL,
    generated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version       TEXT NOT NULL
);

CREATE INDEX idx_impact_statements_supporter ON donor_impact_statements (supporter_id);
```

### Training Job (`train_impact.py`)

```python
import json, os, pickle
from datetime import datetime
import pandas as pd
import statsmodels.api as sm
from sqlalchemy import create_engine
from dotenv import load_dotenv
from storage.blob_client import upload_artifact

load_dotenv()
engine = create_engine(os.getenv("SUPABASE_DB_URL"))

# 1. Load and prepare data (Phases 2–4)
df = pd.read_sql(ALLOCATION_OUTCOMES_QUERY, engine)
df = engineer_features(df)   # defined in features.py

# 2. Fit OLS models per outcome (Phase 5)
results_store = {}
for model_name, config in OUTCOME_MODELS.items():
    df_m  = config['df'].copy()
    label = config['label']
    feat  = ['log_amount'] + [c for c in df_m.columns
                              if c.startswith('program_area_')
                              or c.startswith('safehouse_id_')]
    y = df_m[label]
    X = df_m[feat].assign(const=1)
    model = sm.OLS(y, X).fit()
    # Apply HC3 if heteroscedastic
    from statsmodels.stats.diagnostic import het_breuschpagan
    _, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
    if bp_p < 0.05:
        model = model.get_robustcov_results(cov_type='HC3')
    results_store[model_name] = model

# 3. Extract significant effects
effects = extract_significant_effects(results_store)
assert len(effects) >= 3, f"Only {len(effects)} significant effects — check data quality"

# 4. Save to Blob Storage
model_version = datetime.now().strftime('%Y%m%d_%H%M')
metadata = {'trained_at': datetime.now().isoformat(), 'model_version': model_version,
            'n_allocations': len(df), 'significant_program_areas': list(effects.keys())}

with open('/tmp/impact_effects.json',  'w') as f: json.dump(effects,   f, indent=2)
with open('/tmp/impact_metadata.json', 'w') as f: json.dump(metadata,  f, indent=2)

upload_artifact('/tmp/impact_effects.json',  'impact/impact_effects.json')
upload_artifact('/tmp/impact_metadata.json', 'impact/impact_metadata.json')
upload_artifact('/tmp/impact_effects.json',  f'impact/versions/impact_effects_{model_version}.json')

print(f"Significant program areas: {list(effects.keys())}")
```

### Inference Job (`score_impact.py`)

```python
import json, os
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
from dotenv import load_dotenv
from storage.blob_client import load_json_from_blob
from pipelines.impact_attribution.statement_builder import build_impact_statement

load_dotenv()
engine = create_engine(os.getenv("SUPABASE_DB_URL"))

# 1. Load effects from Blob Storage
effects  = load_json_from_blob('impact/impact_effects.json')
metadata = load_json_from_blob('impact/impact_metadata.json')

# 2. Pull all allocation records with safehouse names
df = pd.read_sql(ALLOCATION_OUTCOMES_QUERY, engine)
sh_names = pd.read_sql("SELECT safehouse_id, name FROM safehouses", engine)
df = df.merge(sh_names, on='safehouse_id', how='left')

# 3. Generate statements
statements = []
for _, row in df.iterrows():
    prog = row['program_area']
    if prog not in effects:
        continue
    for outcome, effect in effects[prog].items():
        baseline_col = 'edu_baseline' if 'edu' in outcome else 'health_baseline'
        if pd.isna(row.get(baseline_col)):
            continue
        stmt_text = build_impact_statement(
            supporter_id   = row['supporter_id'],
            safehouse_name = row['name'],
            program_area   = prog,
            amount         = row['amount_allocated'],
            outcome        = outcome.split('_')[0],
            coef           = effect['coef'],
            ci_low         = effect['ci_low'],
            ci_high        = effect['ci_high'],
            window         = effect['window'],
            baseline       = row[baseline_col]
        )
        statements.append({
            'supporter_id':         row['supporter_id'],
            'safehouse_id':         row['safehouse_id'],
            'program_area':         prog,
            'allocation_amount':    row['amount_allocated'],
            'outcome_metric':       outcome,
            'time_window_months':   effect['window'],
            'estimated_pct_change': round(effect['coef'], 4),
            'statement_text':       stmt_text,
            'generated_at':         datetime.now().isoformat(),
            'model_version':        metadata['model_version']
        })

results = pd.DataFrame(statements)

# 4. Upsert to Supabase (truncate + reinsert — statements are regenerated fresh each run)
with engine.begin() as conn:
    conn.execute(text("DELETE FROM donor_impact_statements WHERE model_version != :v"),
                 {'v': metadata['model_version']})
    results.to_sql('donor_impact_statements', conn, if_exists='append', index=False)

print(f"Generated {len(results)} impact statements for {results['supporter_id'].nunique()} donors.")
```

### FastAPI Endpoint Addition (`main.py`)

```python
@app.post("/score/impact", dependencies=[Depends(verify_key)])
def trigger_impact_scoring():
    """Regenerate all donor impact statements."""
    result = subprocess.run(
        ["python", "pipelines/impact_attribution/score_impact.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr)
    return {"status": "impact statements generated", "output": result.stdout}
```

### Retraining Schedule

| Trigger | Action |
|---|---|
| Quarterly (1st of Jan, Apr, Jul, Oct) | Full retrain (`train_impact.py`) + regenerate statements |
| New allocation batch imported | Regenerate statements only (`score_impact.py`) — no retrain |
| New safehouse monthly metrics added | Regenerate statements only |

### Monitoring Checklist

- [ ] **Statement coverage** — log what % of active donors received at least one statement; alert if below 60%
- [ ] **Significance gate** — assert at least 3 program areas produce significant effects before saving artifacts
- [ ] **Sanity check** — review 5 randomly sampled statements after each run to verify they read naturally
- [ ] **Effect direction** — log whether positive or negative effects are found per program area; flag unexpected negatives for review
- [ ] **Artifact versioning** — versioned effects JSON retained in Blob Storage; roll back by copying prior version to `impact/impact_effects.json`

---

## Appendix — Additional `requirements.txt` Dependencies

```
statsmodels==0.14.x    ← required for OLS (not in base churn requirements)
```
