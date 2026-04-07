import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# SQL — Phase 2 (verbatim from spec)
# ---------------------------------------------------------------------------

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
    CASE WHEN r.sub_cat_trafficked     THEN 1 ELSE 0 END AS sub_trafficked,
    CASE WHEN r.sub_cat_physical_abuse THEN 1 ELSE 0 END AS sub_physical_abuse,
    CASE WHEN r.sub_cat_sexual_abuse   THEN 1 ELSE 0 END AS sub_sexual_abuse,
    CASE WHEN r.sub_cat_osaec          THEN 1 ELSE 0 END AS sub_osaec,
    CASE WHEN r.sub_cat_child_labor    THEN 1 ELSE 0 END AS sub_child_labor,
    CASE WHEN r.sub_cat_at_risk        THEN 1 ELSE 0 END AS sub_at_risk
FROM residents r
WHERE r.case_status = 'Active'
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

EDUCATION_QUERY = """
SELECT
    resident_id,
    COUNT(*)                                                AS edu_record_count,
    MAX(progress_percent)                                   AS max_edu_progress,
    AVG(progress_percent)                                   AS avg_edu_progress,
    REGR_SLOPE(progress_percent,
               EXTRACT(EPOCH FROM record_date))             AS edu_trend,
    MAX(record_date)                                        AS latest_edu_date,
    AVG(CASE WHEN enrollment_status = 'Enrolled' THEN 1.0 ELSE 0.0 END) AS enrollment_rate,
    AVG(attendance_rate)                                    AS avg_attendance,
    MAX(CASE WHEN completion_status = 'Completed' THEN 1 ELSE 0 END) AS has_completed
FROM education_records
GROUP BY resident_id
"""

INCIDENTS_QUERY = """
SELECT
    resident_id,
    COUNT(*)                                                AS total_incidents,
    SUM(CASE WHEN severity = 'High' THEN 1 ELSE 0 END)     AS high_severity_incidents,
    SUM(CASE WHEN resolved = false  THEN 1 ELSE 0 END)      AS open_incidents,
    MAX(incident_date)                                      AS latest_incident_date,
    SUM(CASE WHEN incident_date >= CURRENT_DATE - 30
             THEN 1 ELSE 0 END)                             AS incidents_last_30d,
    SUM(CASE WHEN incident_date >= CURRENT_DATE - 90
             AND severity = 'High'
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
    AVG(CASE WHEN family_cooperation_level = 'Highly Cooperative' THEN 4
             WHEN family_cooperation_level = 'Cooperative'        THEN 3
             WHEN family_cooperation_level = 'Neutral'            THEN 2
             WHEN family_cooperation_level = 'Uncooperative'      THEN 1
             ELSE NULL END)                                 AS avg_family_cooperation,
    SUM(CASE WHEN safety_concerns_noted = true THEN 1 ELSE 0 END) AS visits_with_safety_concerns,
    MAX(visit_date)                                         AS latest_visit_date,
    AVG(CASE WHEN visit_outcome = 'Favorable' THEN 1.0 ELSE 0.0 END) AS positive_visit_rate
FROM home_visitations
GROUP BY resident_id
"""

PLANS_QUERY = """
SELECT
    resident_id,
    COUNT(*)                                                        AS total_plans,
    AVG(CASE WHEN status IN ('Achieved', 'Closed') THEN 1.0
             ELSE 0.0 END)                                          AS plan_completion_rate,
    SUM(CASE WHEN status NOT IN ('Achieved', 'Closed')
             AND target_date < CURRENT_DATE THEN 1 ELSE 0 END)     AS overdue_plans,
    SUM(CASE WHEN status = 'In Progress' THEN 1 ELSE 0 END)        AS active_plans,
    MAX(updated_at)                                                 AS latest_plan_update
FROM intervention_plans
GROUP BY resident_id
"""

# ---------------------------------------------------------------------------
# Feature column definitions — Phase 4
# ---------------------------------------------------------------------------

NUM_COLS = [
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
    'is_pwd', 'has_special_needs', 'family_is_4ps', 'family_solo_parent',
]
CAT_COLS = ['case_category']

# ---------------------------------------------------------------------------
# Explainability label maps — Phase 7
# ---------------------------------------------------------------------------

CONCERN_LABELS = {
    'health_trend':               'Declining health scores',
    'incidents_last_30d':         'Recent incidents filed',
    'open_incidents':             'Unresolved incidents',
    'sessions_with_concerns':     'Concerns flagged in sessions',
    'overdue_plans':              'Overdue intervention plans',
    'visits_with_safety_concerns':'Safety concerns noted during family visit',
    'avg_attendance':             'Low school attendance',
    'plan_completion_rate':       'Low intervention plan completion',
    'sessions_last_30d':          'Few recent counseling sessions',
}

STRENGTH_LABELS = {
    'sessions_with_progress':  'Progress noted in counseling sessions',
    'avg_edu_progress':        'Education progress improving',
    'edu_trend':               'Education progress trending upward',
    'positive_visit_rate':     'Positive family visitation outcomes',
    'avg_health_score':        'Stable health scores',
    'plan_completion_rate':    'Intervention plans on track',
    'avg_family_cooperation':  'Good family cooperation',
}

# ---------------------------------------------------------------------------
# Data loading — Phase 2
# ---------------------------------------------------------------------------

def load_and_merge(engine) -> pd.DataFrame:
    """Load all source tables and merge onto the resident base."""
    df = pd.read_sql(RESIDENT_BASE_QUERY, engine)
    for query in [HEALTH_AGG_QUERY, EDUCATION_QUERY, INCIDENTS_QUERY,
                  SESSIONS_QUERY, VISITATIONS_QUERY, PLANS_QUERY]:
        df_src = pd.read_sql(query, engine)
        df = df.merge(df_src, on='resident_id', how='left')
    print(f"Loaded {len(df)} active residents from {df['safehouse_id'].nunique()} safehouses")
    return df


# ---------------------------------------------------------------------------
# Label construction — Phase 3
# ---------------------------------------------------------------------------

def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Critical is the highest tier in the actual data
    risk_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
    df['initial_risk_num'] = df['initial_risk_level'].map(risk_map)
    df['current_risk_num'] = df['current_risk_level'].map(risk_map)

    df['risk_escalated'] = (
        (df['current_risk_num'] > df['initial_risk_num']) |   # escalated from baseline
        (df['high_incidents_last_90d'].fillna(0) > 0)        |   # recent high incident
        (df['current_risk_num'] >= 2)                             # currently High or Critical
    ).astype(int)

    # Actual values: 'Completed', 'In Progress', 'Not Started', 'On Hold'
    df['reintegration_ready'] = (
        (df['reintegration_status'].isin(['Completed', 'In Progress'])) &
        (df['avg_edu_progress'].fillna(0) >= 70) &
        (df['high_incidents_last_90d'].fillna(0) == 0)
    ).astype(int)

    print(f"Regression risk rate:       {df['risk_escalated'].mean():.2%}")
    print(f"Reintegration readiness rate: {df['reintegration_ready'].mean():.2%}")
    return df


# ---------------------------------------------------------------------------
# Feature engineering — Phase 4
# ---------------------------------------------------------------------------

def _parse_length_of_stay(val) -> float:
    """
    Convert strings like '2 Years 9 months' or '1 Year 0 months' to total months.
    Returns NaN if the value cannot be parsed.
    """
    if pd.isna(val):
        return float('nan')
    if isinstance(val, (int, float)):
        return float(val)
    import re
    val = str(val).lower()
    years  = 0
    months = 0
    y_match = re.search(r'(\d+)\s*year', val)
    m_match = re.search(r'(\d+)\s*month', val)
    if y_match:
        years = int(y_match.group(1))
    if m_match:
        months = int(m_match.group(1))
    if not y_match and not m_match:
        return float('nan')
    return float(years * 12 + months)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse duration/age columns stored as '2 Years 9 months' strings → total months
    for col in ['length_of_stay', 'present_age', 'age_upon_admission']:
        if col in df.columns:
            df[col] = df[col].apply(_parse_length_of_stay)

    # Cast boolean columns to int so sklearn imputer can handle them
    bool_cols = [
        'is_pwd', 'has_special_needs', 'family_is_4ps', 'family_solo_parent', 'family_indigenous',
        'sub_trafficked', 'sub_physical_abuse', 'sub_sexual_abuse',
        'sub_osaec', 'sub_child_labor', 'sub_at_risk',
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # Log-transform skewed count features
    for col in ['total_incidents', 'high_severity_incidents', 'total_sessions', 'total_visits']:
        df[f'{col}_log'] = np.log1p(df[col].fillna(0))

    # Collapse rare case categories
    freq = df['case_category'].value_counts(normalize=True)
    rare = freq[freq < 0.05].index
    df['case_category'] = df['case_category'].replace(rare, 'Other')

    return df


# ---------------------------------------------------------------------------
# Sklearn preprocessor — Phase 4
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('num', numeric_pipe,    NUM_COLS),
        ('cat', categorical_pipe, CAT_COLS),
    ])


# ---------------------------------------------------------------------------
# Explainability helpers — Phase 7
# ---------------------------------------------------------------------------

def get_feature_names(model, num_cols: list, cat_cols: list) -> list:
    cat_names = (
        model.named_steps['prep']
        .named_transformers_['cat']
        .named_steps['onehot']
        .get_feature_names_out(cat_cols)
        .tolist()
    )
    return num_cols + cat_names


def get_top_factors(model, feature_names: list, label_map: dict, top_n: int = 3) -> list:
    importances = model.named_steps['model'].feature_importances_
    top_indices = importances.argsort()[::-1]
    factors = []
    for i in top_indices:
        label = label_map.get(feature_names[i])
        if label:
            factors.append(label)
        if len(factors) == top_n:
            break
    return factors
