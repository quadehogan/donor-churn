# Master ML Pipeline Guide
## End-to-End Reference: From Raw Dataset to Deployed Model

> This guide synthesizes all 17 chapter implementation guides into a single, ordered workflow. Each phase maps directly to the detailed chapter guide for deeper reference. Individual chapter guides are preserved in the same folder.

---

## Quick-Reference Pipeline Map

```
PHASE 1 — PLAN          Ch 1   Define problem, feasibility, CRISP-DM phases
PHASE 2 — ACQUIRE       Ch 4   Load CSV/Excel/SQL
                         Ch 5   Pull from REST APIs (pagination, auth)
PHASE 3 — EXPLORE       Ch 2   DataFrame structure, indexing, filtering
                         Ch 3   Vectorized ops, recoding, date handling
                         Ch 6   unistats() — automated univariate EDA
PHASE 4 — PREPARE       Ch 7   Full automated prep pipeline
                         Ch 8   bivariate() — automated relationship discovery
PHASE 5A — REGRESSION   Ch 9   OLS / MLR concepts & mechanics (statsmodels)
                         Ch 10  OLS diagnostics (NMALH) — causal inference
                         Ch 11  Predictive MLR (sklearn, Ridge/Lasso/ElasticNet)
PHASE 5B — TREES        Ch 12  Decision tree regression — overfit, prune, visualize
PHASE 5C — CLASSIFICATION Ch 13 Logistic regression + multi-algorithm classification
PHASE 5D — ENSEMBLES    Ch 14  Random Forest, Gradient Boosting, Stacking
PHASE 6 — EVALUATE      Ch 15  Cross-validation, learning curves, GridSearch/Random
PHASE 7 — FEATURE SEL.  Ch 16  Filter, Wrapper (RFE), Embedded (Lasso/Tree importance)
PHASE 8 — DEPLOY        Ch 17  ETL → Train → Inference jobs; artifact versioning; SQLite
```

---

## PHASE 1 — Project Planning (CRISP-DM)
> **Chapter 1 deep-dive:** `chapter_one_implementation.md`

### Feasibility Checklist (before writing a single line of code)

| Gate | Question | Pass Condition |
|---|---|---|
| Business | What specific problem is being solved? | Measurable success criteria defined |
| Data | Is live, updatable data available? | Access confirmed, refresh cadence known |
| Analytical | Can data support reliable predictions? | Domain knowledge / prior work suggests feasibility |
| Integration | Can model outputs plug into existing systems? | API or DB interface identified |
| Risk | Is operational disruption acceptable? | Stakeholder sign-off obtained |

### CRISP-DM Phases at a Glance

```
Business Understanding → Data Understanding → Data Preparation
       ↑                                             ↓
Deployment ← Evaluation ← Modeling ← ─────────────────
```

The process is **iterative** — expect to cycle back as new data issues emerge.

### Deliverables Before Proceeding

- Project scope + success metrics (target accuracy, latency, business KPIs)
- Data inventory (sources, access method, freshness)
- Cost-benefit analysis vs. alternatives
- High-level project plan with iteration milestones

---

## PHASE 2 — Data Acquisition
> **Deep-dives:** `chapter_four_implementation.md`, `chapter_five_implementation.md`

### Loading from Files

```python
import pandas as pd

# CSV (most common)
df = pd.read_csv('data/your_file.csv')

# Key parameters
df = pd.read_csv(
    'data/your_file.csv',
    encoding='UTF-16',      # try 'ISO-8859-1' if this fails
    low_memory=False,       # prevents mixed-type inference on large files
    na_values=['NA', '?']   # treat these strings as NaN
)

# Excel
df = pd.read_excel('file.xlsx', 'Sheet1', na_values=['NA'])

# SQLite
import sqlite3
conn = sqlite3.connect('data/mydb.db')
df   = pd.read_sql_query("SELECT * FROM table_name", conn)

# Always verify immediately after loading
print(df.shape)
print(df.dtypes)
df.head()
```

### Loading from REST APIs

```python
import requests, json, time, pandas as pd

# Single request pattern
response  = requests.get('https://api.example.com/endpoint', params={'key': 'value'})
assert response.status_code == 200, f"Failed: {response.status_code}"
data = response.json()

# Always inspect structure before extracting
print(json.dumps(data, indent=2))

# Paginated loop pattern (limit + offset)
all_rows, limit, offset = [], 200, 1
while True:
    resp = requests.get(BASE_URL, params={**params, 'limit': limit, 'offset': offset})
    if resp.status_code != 200: break
    page = resp.json()
    items = page.get('features', [])          # adapt key to your API
    if len(items) == 0: break
    for item in items:
        all_rows.append({...})                # extract your fields here
    offset += limit
    time.sleep(0.25)                          # be polite — avoid rate limits

df = pd.DataFrame(all_rows)
df.to_csv('output.csv', index=False)

# Authentication patterns
headers = {'Authorization': 'Bearer YOUR_KEY'}        # header auth (preferred)
params  = {'appid': 'YOUR_KEY'}                       # querystring auth
```

**Security:** Store API keys in a separate `api_key.txt` file — never hardcode them in notebooks or commit them to version control.

---

## PHASE 3 — Data Exploration
> **Deep-dives:** `chapter_two_implementation.md`, `chapter_three_implementation.md`, `chapter_six_implementation.md`

### DataFrame Fundamentals

```python
# Key inspection commands — run these immediately after loading
df.shape        # (rows, columns)
df.dtypes       # data types of each column
df.head()       # first 5 rows
df.isnull().sum()  # count of missing values per column
df.describe()   # summary stats for numeric columns

# Filtering (always assign to new variable — never overwrite original)
subset = df[df['age'] > 30].copy()
subset = df[(df['age'] > 30) & (df['smoker'] == 'yes')].copy()

# Joining datasets
merged = pd.merge(df1, df2, on='key_column', how='inner')
stacked = pd.concat([df1, df2], axis=0, ignore_index=True)
```

### Vectorized Operations (always prefer over loops)

```python
import numpy as np

# Column math
df['bmi_per_age'] = df['bmi'] / df['age']
df['log_charges'] = np.log(df['charges'])
df['age_sqrt']    = np.sqrt(df['age'])

# Recoding categoricals
df['smoker_num'] = df['smoker'].map({'yes': 1, 'no': 0})
df['is_male']    = np.where(df['sex'] == 'male', 1, 0)

# Date handling
df['date'] = pd.to_datetime(df['date'])
df['year']    = df['date'].dt.year
df['month']   = df['date'].dt.month
df['weekday'] = df['date'].dt.dayofweek        # Mon=0, Sun=6
df['days_since'] = (pd.Timestamp.today() - df['date']).dt.days
df = df.drop(columns=['date'])                 # drop after extraction
```

### Automated Univariate EDA (`unistats`)

```python
def unistats(df):
    import pandas as pd
    output_df = pd.DataFrame(columns=[
        'Count', 'Unique', 'Type',
        'Min', 'Max', '25%', '50%', '75%',
        'Mean', 'Median', 'Mode', 'Std', 'Skew', 'Kurt'
    ])
    for col in df.columns:
        count  = df[col].count()
        unique = df[col].nunique()
        dtype  = str(df[col].dtype)
        # Reset ALL branch vars every iteration (prevents prior-iteration bug)
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

# Usage
stats = unistats(df)
# Red flags: |Skew| > 1 → consider transform; Unique = 1 → drop (constant column)
```

---

## PHASE 4 — Data Preparation
> **Deep-dives:** `chapter_seven_implementation.md`, `chapter_eight_implementation.md`

### Automated Prep Pipeline (run in this order)

```python
# STEP 1: Remove structurally useless columns
df = basic_wrangling(df)              # drops: >95% missing, ID-like, constant columns

# STEP 2: Parse date columns → numeric features
df = parse_date(df, drop_date=True)   # extracts year/month/day/weekday, drops raw datetime

# STEP 3: Collapse rare categories
df = bin_categories(df, cutoff=0.05)  # collapses <5% frequency categories → 'Other'

# STEP 4: Fix skewed numeric features
for col in df.select_dtypes(include='number').columns:
    if abs(df[col].skew()) > 1:
        df, method = skew_correct(df, col, visualize=False)
        # Creates {col}_skewfix column using best of: cbrt, sqrt, log1p, yeojohnson

# STEP 5: Drop heavily missing data
df = missing_drop(df, label='target', row_threshold=0.9, col_threshold=0.5)

# STEP 6: Impute remaining missing values (with MAR bias test)
df = missing_fill(df, label='target', acceptable=0.1, mar='drop')
# → runs t-test (small data) or z-test (large data) per column
# → p ≤ 0.10 (MAR): drops missing rows; p > 0.10 (MCAR): imputes with IterativeImputer

# STEP 7: Handle per-feature outliers
df = clean_outlier(df, method='replace')  # auto-selects Tukey IQR (skewed) or μ±3σ (normal)

# STEP 8: Multivariate outlier removal
df = clean_outliers(df, drop_percent=0.02)  # DBSCAN — flags at most 2% of rows
```

### Missing Data Decision Logic

```
Missing values in column?
    ↓ Yes
Run t-test (n<200k) or z-test (n≥200k) comparing label means: missing vs. present rows
    ↓ p > 0.10 → MCAR (missingness is random)    → Impute safely
    ↓ p ≤ 0.10 → MAR  (missingness is biased)    → Drop missing rows (or impute + warn)
```

### Bivariate Relationship Discovery (`bivariate`)

```python
# Auto-selects statistic and chart based on data types
bivariate_results = bivariate(df, label='target')
# N2N (both numeric):      Pearson r, Spearman ρ, Kendall τ + scatterplot
# C2N/N2C (mixed):         ANOVA F + Bonferroni pairwise t-tests + bar chart
# C2C (both categorical):  Chi-square X² + contingency heatmap
# Returns table sorted by p-value — highest signal features appear first
```

**Effect size interpretation (p-value alone is not enough):**

| Relationship | Statistic | Small | Medium | Large |
|---|---|---|---|---|
| N2N | Pearson r | 0.1–0.3 | 0.3–0.5 | > 0.5 |
| C2N | Cohen's f | 0.1–0.25 | 0.25–0.4 | > 0.4 |
| C2C | Cramér's V | 0.1–0.3 | 0.3–0.5 | > 0.5 |

---

## PHASE 5A — Regression: Explanatory MLR (statsmodels)
> **Deep-dives:** `chapter_nine_implementation.md`, `chapter_ten_implementation.md`

### When to Use

Use **statsmodels OLS** when the goal is to **understand effect directions and magnitudes** — not just predict. You need p-values, confidence intervals, and diagnostic tests.

```python
import statsmodels.api as sm

# Encode categoricals (always drop_first to avoid perfect multicollinearity)
df = pd.get_dummies(df, columns=df.select_dtypes(['object']).columns, drop_first=True)
df[df.select_dtypes(bool).columns] = df.select_dtypes(bool).astype(int)

y = df['label']
X = df.drop(columns=['label']).assign(const=1)  # statsmodels needs const column

model   = sm.OLS(y, X).fit()
print(model.summary())

# Error metrics (in-sample — informational only)
mae  = abs(model.fittedvalues - y).mean()
rmse = ((model.fittedvalues - y) ** 2).mean() ** 0.5
```

### Reading the OLS Summary

| Field | Interpretation |
|---|---|
| R² | % of label variance explained. Higher is better; compare with Adj R². |
| Adj R² | R² penalized for # features. If Adj R² << R², some features add no value. |
| Cond. No. | Multicollinearity severity. > 1000 = serious. > 1e+6 = near-perfect collinearity. |
| coef | Expected change in label per 1-unit increase in that feature, all else held constant. |
| P>|t| | p < 0.05 = conventionally significant. Threshold, not a guarantee of importance. |
| [0.025, 0.975] | 95% confidence interval for the coefficient. |

### OLS Diagnostic Tests (NMALH — run in this order)

```python
# 1. NORMALITY — check residuals
from scipy import stats
stat, p = stats.normaltest(model.resid)
# Also: check model.summary() Omnibus test and Q-Q plot
stats.probplot(model.resid, plot=plt)

# If non-normal: apply Box-Cox or Yeo-Johnson to the label
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='box-cox', standardize=False)
df['label_bc'] = pt.fit_transform(df[['label']])

# 2. MULTICOLLINEARITY — Variance Inflation Factor
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_vif = sm.add_constant(df.drop(columns=['label']), has_constant='add')
vif_df = pd.DataFrame({
    'feature': X_vif.columns,
    'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
}).query("feature != 'const'").sort_values('VIF', ascending=False)
# VIF > 10 = serious multicollinearity → remove feature or combine with PCA

# 3. AUTOCORRELATION — Durbin-Watson
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(model.resid)
# ≈ 2.0 = no autocorrelation (fine); < 1 or > 3 = concern (common in time-series)

# 4. LINEARITY — residuals vs. fitted plot
plt.scatter(model.fittedvalues, model.resid)
plt.axhline(0, color='red')
plt.xlabel('Fitted'); plt.ylabel('Residuals')
# Fix: add polynomial terms (age²), log-transform features, or interaction terms
# Center before squaring to reduce multicollinearity:
age_c = df['age'] - df['age'].mean()
df['age_sq'] = age_c ** 2

# 5. HOMOSCEDASTICITY — Breusch-Pagan test
from statsmodels.stats.diagnostic import het_breuschpagan
bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(model.resid, X)
# p < 0.05 = heteroscedasticity present
# Fix: HC3 robust standard errors (does not change coefficients — only SEs)
model_hc3 = model.get_robustcov_results(cov_type='HC3')
```

**NMALH Decision Table:**

| Test | Fails if | Quick Fix |
|---|---|---|
| Normality | Omnibus p < 0.05 | Box-Cox or Yeo-Johnson transform on label |
| Multicollinearity | VIF > 10 | Drop one of the correlated features |
| Autocorrelation | DW << 1 or >> 3 | Usually not an issue for cross-sectional data |
| Linearity | Residuals fan/curve in vs. fitted plot | Add polynomial or log-transformed feature terms |
| Homoscedasticity | Breusch-Pagan p < 0.05 | Use HC3 robust standard errors |

---

## PHASE 5B — Regression: Predictive MLR (scikit-learn)
> **Deep-dive:** `chapter_eleven_implementation.md`

### When to Use

Use **sklearn** when the goal is to **minimize out-of-sample error** on new data — not to interpret coefficients.

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Split — ALWAYS before any preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Pipeline (scaler + model — prevents leakage in CV)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  Lasso())
])

# 3. Tune alpha with GridSearchCV
param_grid = {'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid.fit(X_train, y_train)

print(f"Best alpha: {grid.best_params_['model__alpha']}")
print(f"Best CV MAE: ${-grid.best_score_:,.2f}")

# 4. Evaluate on held-out test set
y_pred = grid.best_estimator_.predict(X_test)
print(f"Test MAE:  ${mean_absolute_error(y_test, y_pred):,.2f}")
print(f"Test RMSE: ${mean_squared_error(y_test, y_pred)**0.5:,.2f}")
```

### Regularization Cheat Sheet

| Method | Penalty | Zero Coefficients? | Best When |
|---|---|---|---|
| OLS | None | No | Baseline; small # of features |
| Ridge | L2 (squared) | No — only shrinks | Correlated features; keep all |
| Lasso | L1 (absolute) | Yes — sparse | Many irrelevant features; want selection |
| ElasticNet | L1 + L2 | Yes — some zeros | Correlated features + want selection |

**Rule:** alpha=0 → standard OLS; alpha→∞ → all coefficients zero. Tune with GridSearchCV.

---

## PHASE 5C — Decision Tree Regression
> **Deep-dive:** `chapter_twelve_implementation.md`

### When to Use

Use decision trees when relationships are **nonlinear** or involve **threshold effects and interactions** that would require many engineered features in linear regression.

```python
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# No StandardScaler needed — trees are scale-invariant
preprocessor = ColumnTransformer(transformers=[
    ('num', 'passthrough', num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)
])

# Start with baseline (untuned) to see overfitting
tree_pipe = Pipeline([('prep', preprocessor), ('tree', DecisionTreeRegressor(random_state=42))])
tree_pipe.fit(X_train, y_train)

# Depth sweep — diagnose overfitting
depths = [2, 3, 4, 5, 6, 8, 10]
for d in depths:
    t = Pipeline([('prep', preprocessor),
                  ('tree', DecisionTreeRegressor(max_depth=d, random_state=42))])
    t.fit(X_train, y_train)
    train_rmse = mean_squared_error(y_train, t.predict(X_train)) ** 0.5
    test_rmse  = mean_squared_error(y_test, t.predict(X_test)) ** 0.5
    print(f"depth={d}  train={train_rmse:,.0f}  test={test_rmse:,.0f}  gap={test_rmse-train_rmse:,.0f}")

# Widening gap between train and test = overfitting → choose depth near test-error minimum

# Feature importance
importances = pd.Series(
    tree_pipe.named_steps['tree'].feature_importances_,
    index=num_cols + list(tree_pipe.named_steps['prep']
                          .named_transformers_['cat']
                          .get_feature_names_out(cat_cols))
).sort_values(ascending=False)
print(importances.head(10))
# Importance = magnitude only — does NOT indicate direction
```

### Key Hyperparameters

| Parameter | Effect when increased | Typical starting values |
|---|---|---|
| `max_depth` | More complex, lower bias, higher variance | Start 2–3; increase gradually |
| `min_samples_leaf` | Smoother predictions, greater stability | 10–20 |
| `min_samples_split` | Fewer splits, simpler tree | 20–50 |
| `min_impurity_decrease` | Prevents weak splits | 0.0 (tune upward) |

---

## PHASE 5D — Classification
> **Deep-dives:** `chapter_thirteen_implementation.md`, `chapter_fourteen_implementation.md`

### When to Use

Use classification when the **label is categorical** (binary or multiclass). The pipeline structure is identical to regression; only the model and metrics change.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Three-way stratified split for imbalanced classes
from sklearn.model_selection import train_test_split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
)

# Full preprocessing pipeline
from sklearn.impute import SimpleImputer
numeric_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                          ('scaler', StandardScaler())])
categorical_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                               ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer([('num', numeric_pipe, num_cols),
                                   ('cat', categorical_pipe, cat_cols)])

# Swap model in the same pipeline structure
for Model, name in [
    (LogisticRegression(max_iter=1000), 'Logistic Regression'),
    (DecisionTreeClassifier(max_depth=5), 'Decision Tree'),
    (RandomForestClassifier(n_estimators=100, random_state=42), 'Random Forest'),
    (GradientBoostingClassifier(n_estimators=100, random_state=42), 'Gradient Boosting'),
]:
    pipe = Pipeline([('prep', preprocessor), ('model', Model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(f"\n{name}")
    print(classification_report(y_test, y_pred))
```

### Classification Metrics Quick Reference

| Metric | Formula | Use When |
|---|---|---|
| Accuracy | Correct / Total | Balanced classes only |
| Precision | TP / (TP + FP) | Cost of false positives is high (e.g., spam filter) |
| Recall | TP / (TP + FN) | Cost of false negatives is high (e.g., disease detection) |
| F1 | 2 × (P × R) / (P + R) | Imbalanced classes; balance precision/recall |
| ROC-AUC | Area under ROC curve | General ranking ability; threshold-independent |

### Ensemble Methods

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

# Random Forest — bagging (variance reduction)
rf = Pipeline([('prep', preprocessor),
               ('model', RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42))])

# Gradient Boosting — boosting (bias + variance reduction)
gb = Pipeline([('prep', preprocessor),
               ('model', GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                                     max_depth=4, subsample=0.8,
                                                     random_state=42))])

# Stacking — combine multiple base models via a meta-learner
base_models = [('rf', rf), ('gb', gb)]
stacker = StackingClassifier(estimators=base_models,
                              final_estimator=LogisticRegression(),
                              cv=5, passthrough=False)
```

---

## PHASE 6 — Model Evaluation, Selection & Tuning
> **Deep-dive:** `chapter_fifteen_implementation.md`

### Cross-Validation (always inside a Pipeline)

```python
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = cross_validate(
    pipeline, X_train, y_train,
    cv=cv,
    scoring=['accuracy', 'roc_auc', 'f1'],
    return_train_score=True
)

print(f"Val AUC:   {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
print(f"Train AUC: {cv_results['train_roc_auc'].mean():.4f}")
# Large gap (train >> val) = overfitting → regularize more
# Both low = underfitting → increase model complexity
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import loguniform

# GridSearchCV — exhaustive (small grids)
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth':    [4, 6, 8],
}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")

# RandomizedSearchCV — efficient (large grids)
param_dist = {
    'model__n_estimators': [50, 100, 200, 300, 500],
    'model__learning_rate': loguniform(0.01, 0.3),
    'model__max_depth': [3, 4, 5, 6, 8],
}
rnd = RandomizedSearchCV(pipeline, param_dist, n_iter=30, cv=5,
                          scoring='roc_auc', random_state=42, n_jobs=-1)
rnd.fit(X_train, y_train)
```

### Learning Curves (diagnose overfitting)

```python
from sklearn.model_selection import learning_curve
import numpy as np, matplotlib.pyplot as plt

train_sizes, train_scores, val_scores = learning_curve(
    pipeline, X_train, y_train, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='roc_auc', n_jobs=-1
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Train')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
plt.xlabel('Training Set Size'); plt.ylabel('ROC-AUC')
plt.title('Learning Curve'); plt.legend()
# Converging high → good generalization
# Persistent large gap → overfitting
# Both low → underfitting / need more data or better features
```

### Final Test-Set Evaluation (do this ONCE at the end)

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

best_model = grid.best_estimator_
y_pred      = best_model.predict(X_test)
y_proba     = best_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(confusion_matrix(y_test, y_pred))
```

---

## PHASE 7 — Feature Selection
> **Deep-dive:** `chapter_sixteen_implementation.md`

**Golden rule:** Feature selection must happen **inside the CV loop** — place it inside the Pipeline, never before splitting.

### Three Families of Methods

```python
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, mutual_info_classif,
    RFECV, SelectFromModel, permutation_importance
)
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier

# ── FILTER METHODS (no model required) ────────────────────────────────────────
# 1. Variance threshold — remove near-constant features
filter_pipe = Pipeline([
    ('prep', preprocessor),
    ('var', VarianceThreshold(threshold=0.01)),
    ('model', LogisticRegression(max_iter=1000))
])

# 2. Univariate statistical test (SelectKBest)
anova_pipe = Pipeline([
    ('prep', preprocessor),
    ('select', SelectKBest(f_classif, k=20)),
    ('model', LogisticRegression(max_iter=1000))
])

# ── WRAPPER METHODS (model-dependent, expensive) ──────────────────────────────
# RFECV — recursive feature elimination with CV
rfe_pipe = Pipeline([
    ('prep', preprocessor),
    ('rfe', RFECV(estimator=LogisticRegression(max_iter=1000),
                  cv=5, scoring='roc_auc', min_features_to_select=5)),
    ('model', LogisticRegression(max_iter=1000))
])

# ── EMBEDDED METHODS (selection happens during training) ──────────────────────
# Lasso (L1) — zero coefficients = removed features
lasso_sel = Pipeline([
    ('prep', preprocessor),
    ('select', SelectFromModel(Lasso(alpha=0.01))),
    ('model', LogisticRegression(max_iter=1000))
])

# Tree importance (Random Forest)
tree_sel = Pipeline([
    ('prep', preprocessor),
    ('select', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42),
                               threshold='mean')),
    ('model', LogisticRegression(max_iter=1000))
])

# ── PERMUTATION IMPORTANCE (model-agnostic, post-hoc) ────────────────────────
best_pipe.fit(X_train, y_train)
perm = permutation_importance(best_pipe, X_val, y_val,
                               n_repeats=10, random_state=42, scoring='roc_auc')
perm_df = pd.DataFrame({
    'feature': X_val.columns,
    'importance_mean': perm.importances_mean,
    'importance_std':  perm.importances_std
}).sort_values('importance_mean', ascending=False)
```

### Feature Selection Strategy by Goal

| Goal | Recommended Approach |
|---|---|
| Predictive only | RFECV or SelectFromModel + compare CV scores |
| Causal/explanatory | VIF screening (< 10) + Lasso + manual domain review |
| Very wide data (p >> n) | Variance threshold first, then Lasso |
| Need interpretability | SelectKBest (top k) + Logistic Regression |

---

## PHASE 8 — Deployment
> **Deep-dive:** `chapter_seventeen_implementation.md`

### Architecture

```
App/Dashboard
    ↓ reads predictions from
Operational DB (shop.db)   ←── Inference job writes here on schedule
    ↓ ETL
Warehouse DB (warehouse.db)
    ↓ Training job reads from
Training Job
    ↓ outputs
model.sav + model_metadata.json + metrics.json
```

### Project Folder Structure

```
project/
  data/
    shop.db           ← operational (transactional) database
    warehouse.db      ← analytical (modeling-ready) database
  artifacts/
    model.sav
    model_metadata.json
    metrics.json
  jobs/
    config.py         ← shared path constants
    utils_db.py       ← DB read/write helpers
    etl_build_warehouse.py
    train_model.py
    run_inference.py
```

### Training Job Pattern

```python
import pickle, json
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
import sqlite3, pandas as pd

# 1. Load training data from warehouse
conn = sqlite3.connect('data/warehouse.db')
df   = pd.read_sql_query("SELECT * FROM training_data", conn)
conn.close()

# 2. Prepare features and label
y = df['label']
X = df.drop(columns=['label'])

# 3. Build and fit the full pipeline
pipeline = Pipeline([('prep', preprocessor), ('model', GradientBoostingClassifier(...))])
pipeline.fit(X, y)

# 4. Save model artifact
with open('artifacts/model.sav', 'wb') as f:
    pickle.dump(pipeline, f)

# 5. Save metadata and metrics
metadata = {
    'trained_at': datetime.now().isoformat(),
    'n_samples':  len(df),
    'features':   X.columns.tolist(),
    'model_class': type(pipeline.named_steps['model']).__name__,
    'hyperparams': pipeline.named_steps['model'].get_params()
}
with open('artifacts/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

### Inference Job Pattern

```python
import pickle, sqlite3, pandas as pd
from datetime import datetime

# 1. Load model
with open('artifacts/model.sav', 'rb') as f:
    pipeline = pickle.load(f)

# 2. Load new data to score
conn = sqlite3.connect('data/shop.db')
new_data = pd.read_sql_query("SELECT * FROM new_orders WHERE predicted = 0", conn)
conn.close()

if len(new_data) > 0:
    # 3. Score
    X_new = new_data.drop(columns=['order_id'])
    predictions = pipeline.predict(X_new)
    probabilities = pipeline.predict_proba(X_new)[:, 1]

    # 4. Write predictions back to DB
    results = pd.DataFrame({
        'order_id':   new_data['order_id'],
        'prediction': predictions,
        'probability': probabilities,
        'scored_at':  datetime.now().isoformat()
    })

    conn = sqlite3.connect('data/shop.db')
    results.to_sql('order_predictions', conn, if_exists='append', index=False)
    conn.commit()
    conn.close()
    print(f"Scored {len(results)} records.")
```

### Monitoring Checklist

- [ ] **Model drift detection** — monitor prediction distribution over time; alert if it shifts significantly
- [ ] **Input data validation** — check for null spikes, out-of-range values, new categories before scoring
- [ ] **Retraining schedule** — define trigger: time-based (monthly) or performance-based (AUC drops below threshold)
- [ ] **Artifact versioning** — timestamp `model.sav` and `metrics.json` on each retrain
- [ ] **Rollback plan** — keep previous model artifact so you can revert without downtime

---

## Universal Pipeline Template (Copy-Paste Starter)

```python
# ════════════════════════════════════════════════════════════════
# UNIVERSAL ML PIPELINE TEMPLATE
# Covers: regression (swap scoring/metrics for classification)
# ════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, GridSearchCV)
from sklearn.metrics import mean_absolute_error, mean_squared_error  # regression
# from sklearn.metrics import classification_report, roc_auc_score  # classification

SEED = 42
warnings.filterwarnings('ignore')

# ── 1. LOAD ──────────────────────────────────────────────────────────────────
df = pd.read_csv('your_data.csv')
print(df.shape, '\n', df.dtypes)

# ── 2. TARGET + FEATURES ─────────────────────────────────────────────────────
TARGET  = 'label_column'
LEAKY   = []       # columns derived from or synonymous with the label
DROP    = []       # irrelevant identifier columns

y = df[TARGET].copy()
X = df.drop(columns=[TARGET] + LEAKY + DROP).copy()

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# ── 3. SPLIT ─────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED
    # add stratify=y for classification
)

# ── 4. PREPROCESSOR ──────────────────────────────────────────────────────────
numeric_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])
categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore', drop='first'))
])
preprocessor = ColumnTransformer([
    ('num', numeric_pipe, num_cols),
    ('cat', categorical_pipe, cat_cols)
], remainder='drop')

# ── 5. MODEL PIPELINE ────────────────────────────────────────────────────────
from sklearn.ensemble import GradientBoostingRegressor  # swap for your model

model_pipe = Pipeline([
    ('prep',  preprocessor),
    ('model', GradientBoostingRegressor(random_state=SEED))
])

# ── 6. CROSS-VALIDATE ────────────────────────────────────────────────────────
cv_scores = cross_val_score(
    model_pipe, X_train, y_train,
    cv=5,
    scoring='neg_mean_absolute_error'  # or 'roc_auc', 'f1', etc.
)
print(f"CV MAE: {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 7. TUNE ──────────────────────────────────────────────────────────────────
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth':    [3, 4, 5],
    'model__learning_rate': [0.05, 0.1]
}
grid = GridSearchCV(model_pipe, param_grid, cv=5,
                    scoring='neg_mean_absolute_error', n_jobs=-1)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")

# ── 8. FINAL EVALUATION ──────────────────────────────────────────────────────
best = grid.best_estimator_
y_pred = best.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print(f"\nTest MAE:  {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")

# ── 9. SAVE MODEL ────────────────────────────────────────────────────────────
import pickle, json
from datetime import datetime

with open('artifacts/model.sav', 'wb') as f:
    pickle.dump(best, f)

metadata = {
    'trained_at':  datetime.now().isoformat(),
    'n_samples':   len(X_train),
    'best_params': grid.best_params_,
    'test_mae':    mae,
    'test_rmse':   rmse
}
with open('artifacts/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\nModel saved to artifacts/model.sav")
```

---

## Common Mistakes — Master Checklist

| Mistake | Phase | Fix |
|---|---|---|
| Preprocessing before splitting (data leakage) | Prepare | Always split first; put all transforms in Pipeline |
| Evaluating on training data | Model | Report only test-set metrics as final performance |
| `drop_first=False` in dummy coding | Prepare | Always use `drop_first=True` to avoid perfect multicollinearity |
| Forgetting `const=1` in statsmodels | Regression | Add `.assign(const=1)` or `sm.add_constant(X)` |
| Scaling label y in causal modeling | Regression | Leave y unscaled; scaling destroys coefficient units |
| StandardScaler before decision trees | Trees | Trees are scale-invariant; skip scaler |
| Treating MDI feature importance as causal | Trees | MDI = magnitude in this model; use PFI for robustness |
| No `stratify=y` in classification split | Classification | Always stratify for imbalanced classes |
| Using accuracy on imbalanced classes | Classification | Use F1, ROC-AUC, or Precision-Recall instead |
| Picking alpha/hyperparams by test error | Tuning | Use CV for selection; test set is for FINAL reporting only |
| Feature selection before CV | Feature Sel. | Put feature selection inside the Pipeline |
| No model versioning in deployment | Deploy | Timestamp all saved artifacts; keep previous version |
| No input validation at inference | Deploy | Check for null spikes and schema drift before scoring |

---

## Chapter Reference Index

| Chapter | Topic | File |
|---|---|---|
| 1 | CRISP-DM / Project Planning | `chapter_one_implementation.md` |
| 2 | Pandas DataFrames | `chapter_two_implementation.md` |
| 3 | Data Wrangling (vectorized ops, dates) | `chapter_three_implementation.md` |
| 4 | CSV / Excel / SQL I/O | `chapter_four_implementation.md` |
| 5 | REST APIs | `chapter_five_implementation.md` |
| 6 | Automated EDA (`unistats`) | `chapter_six_implementation.md` |
| 7 | Automated Prep Pipeline | `chapter_seven_implementation.md` |
| 8 | Bivariate Analysis (`bivariate`) | `chapter_eight_implementation.md` |
| 9 | MLR Concepts & Mechanics | `chapter_nine_implementation.md` |
| 10 | OLS Diagnostics (NMALH) | `chapter_ten_implementation.md` |
| 11 | Predictive MLR (Ridge/Lasso) | `chapter_eleven_implementation.md` |
| 12 | Decision Tree Regression | `chapter_twelve_implementation.md` |
| 13 | Classification Modeling | `chapter_thirteen_implementation.md` |
| 14 | Ensemble Methods | `chapter_fourteen_implementation.md` |
| 15 | Model Evaluation & Tuning | `chapter_fifteen_implementation.md` |
| 16 | Feature Selection | `chapter_sixteen_implementation.md` |
| 17 | Deployment | `chapter_seventeen_implementation.md` |
