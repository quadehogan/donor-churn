import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# SQL — Phase 2 (verbatim from spec)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Feature engineering — Phase 4
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame):
    """
    Returns (df_full, df_edu, df_health).

    df_full  — full engineered frame with dummies
    df_edu   — rows with non-null edu baseline + 3m outcome
    df_health — rows with non-null health baseline + 3m outcome
    """
    df = df.copy()

    # Log-transform allocation amount (right-skewed)
    df['log_amount'] = np.log1p(df['amount_allocated'])

    # Percent-change deltas
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

    # Collapse rare program areas (<5% of rows)
    freq = df['program_area'].value_counts(normalize=True)
    rare = freq[freq < 0.05].index
    df['program_area'] = df['program_area'].replace(rare, 'Other')

    # One-hot encode program area and safehouse
    df = pd.get_dummies(df, columns=['program_area', 'safehouse_id'], drop_first=True)
    bool_cols = df.select_dtypes(bool).columns
    df[bool_cols] = df[bool_cols].astype(int)

    df_edu       = df.dropna(subset=['edu_baseline',    'edu_3m']).copy()
    df_health    = df.dropna(subset=['health_baseline', 'health_3m']).copy()
    df_edu_6m    = df.dropna(subset=['edu_baseline',    'edu_6m']).copy()
    df_health_6m = df.dropna(subset=['health_baseline', 'health_6m']).copy()

    return df, df_edu, df_health, df_edu_6m, df_health_6m


def build_outcome_models(
    df_edu: pd.DataFrame,
    df_health: pd.DataFrame,
    df_edu_6m: pd.DataFrame,
    df_health_6m: pd.DataFrame,
) -> dict:
    """Construct the OUTCOME_MODELS config dict used in train_impact.py."""
    return {
        'education_3m': {'df': df_edu,       'label': 'edu_pct_change_3m',    'window': 3},
        'health_3m':    {'df': df_health,    'label': 'health_pct_change_3m', 'window': 3},
        'education_6m': {'df': df_edu_6m,    'label': 'edu_pct_change_6m',    'window': 6},
        'health_6m':    {'df': df_health_6m, 'label': 'health_pct_change_6m', 'window': 6},
    }


# ---------------------------------------------------------------------------
# Effect extraction — Phase 7
# ---------------------------------------------------------------------------

def extract_significant_effects(results_store: dict, p_threshold: float = 0.05) -> dict:
    """
    Returns {program_area: {outcome: {coef, ci_low, ci_high, window}}}
    for all effects where p < p_threshold.
    """
    effects = {}
    for model_name, model in results_store.items():
        try:
            # exog_names is always a plain list — works for both OLS and HC3 results
            names   = model.model.exog_names
            params  = pd.Series(model.params,  index=names)
            pvalues = pd.Series(model.pvalues, index=names)
            ci_raw  = model.conf_int()
            ci      = pd.DataFrame(ci_raw, index=names)
            coef_df = pd.DataFrame({
                'coef':    params,
                'pvalue':  pvalues,
                'ci_low':  ci.iloc[:, 0],
                'ci_high': ci.iloc[:, 1],
            }).dropna(subset=['pvalue'])
        except Exception as e:
            print(f"Skipping {model_name}: {e}")
            continue

        outcome, window_str = model_name.rsplit('_', 1)
        window = int(window_str.rstrip('m'))
        prog_cols = coef_df[coef_df.index.str.startswith('program_area_')]
        sig = prog_cols[prog_cols['pvalue'] < p_threshold]
        for feat, row in sig.iterrows():
            prog = feat.replace('program_area_', '').replace('_', ' ').title()
            effects.setdefault(prog, {})[outcome] = {
                'coef':    round(float(row['coef']),    2),
                'ci_low':  round(float(row['ci_low']),  2),
                'ci_high': round(float(row['ci_high']), 2),
                'window':  int(window),
            }
    return effects
