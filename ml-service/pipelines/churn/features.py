"""
SQL queries and feature engineering for the donor churn pipeline.

TODO: Column names marked with *** need verification against the real Supabase schema.
      Drop the CSV exports into the repo root and these will be updated to match.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# SQL — Training query (one row per supporter, donations aggregated)
# ---------------------------------------------------------------------------
# *** Assumes columns: supporters(supporter_id, supporter_type, relationship_type,
#     region, country, acquisition_channel, status, created_at, first_donation_date)
#     donations(donation_id, supporter_id, amount, donation_date, is_recurring,
#               campaign_name, channel_source)
# ---------------------------------------------------------------------------

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

    CASE
        WHEN MAX(CASE WHEN d.is_recurring THEN 1 ELSE 0 END) = 1
             AND (CURRENT_DATE - MAX(d.donation_date)) > 120 THEN 1
        WHEN MAX(CASE WHEN d.is_recurring THEN 1 ELSE 0 END) = 0
             AND (CURRENT_DATE - MAX(d.donation_date)) > 90  THEN 1
        ELSE 0
    END                                                 AS churned

FROM supporters s
JOIN donations d ON s.supporter_id = d.supporter_id
WHERE s.status IN ('active', 'lapsed', 'inactive')
GROUP BY
    s.supporter_id, s.supporter_type, s.relationship_type,
    s.region, s.country, s.acquisition_channel,
    s.status, s.created_at, s.first_donation_date
HAVING COUNT(d.donation_id) >= 2
"""

# ---------------------------------------------------------------------------
# SQL — Inference query (active donors only, no label)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# SQL — Donation trend enrichment (slope of amount over time per supporter)
# ---------------------------------------------------------------------------

TREND_QUERY = """
SELECT
    supporter_id,
    REGR_SLOPE(amount, EXTRACT(EPOCH FROM donation_date)) AS donation_trend
FROM donations
GROUP BY supporter_id
"""

# ---------------------------------------------------------------------------
# Feature columns (must stay in sync with engineer_features output)
# ---------------------------------------------------------------------------

NUM_COLS = [
    "recency_days", "frequency", "total_value_log", "avg_value_log",
    "max_value_log", "ever_recurring", "num_campaigns", "num_channels",
    "avg_gap_days", "tenure_days", "active_span_days", "donation_trend",
]

CAT_COLS = ["supporter_type", "relationship_type", "acquisition_channel", "region"]

FEATURE_LABELS = {
    "recency_days":    "No recent donation",
    "frequency":       "Low donation frequency",
    "avg_gap_days":    "Long gaps between donations",
    "total_value_log": "Low total giving",
    "avg_value_log":   "Declining average gift size",
    "donation_trend":  "Decreasing donation amounts",
    "tenure_days":     "Short supporter tenure",
    "num_campaigns":   "Low campaign engagement",
    "ever_recurring":  "Not a recurring donor",
}

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame, engine=None) -> pd.DataFrame:
    """
    Applies date extraction, log transforms, and rare-category collapsing.
    Optionally enriches with donation trend if engine is supplied.
    Returns a copy of df ready for modeling (supporter_id preserved).
    """
    df = df.copy()

    # Enrich with donation trend if a DB engine is available
    if engine is not None:
        df_trend = pd.read_sql(TREND_QUERY, engine)
        df = df.merge(df_trend, on="supporter_id", how="left")
    elif "donation_trend" not in df.columns:
        df["donation_trend"] = 0.0

    # Date conversions
    for col in ["last_donation_date", "first_donation_date",
                "first_donation_date_tx", "supporter_since"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    today = pd.Timestamp.today().normalize()
    df["recency_days"]     = (today - df["last_donation_date"]).dt.days
    df["tenure_days"]      = (today - df["supporter_since"]).dt.days
    df["active_span_days"] = (
        df["last_donation_date"] - df["first_donation_date_tx"]
    ).dt.days.clip(lower=0)
    df["avg_gap_days"] = df["active_span_days"] / (df["frequency"] - 1).clip(lower=1)

    # Drop raw date columns
    df = df.drop(
        columns=[c for c in
                 ["last_donation_date", "first_donation_date",
                  "first_donation_date_tx", "supporter_since"]
                 if c in df.columns]
    )

    # Log transforms for right-skewed monetary columns
    for col in ["total_value", "avg_value", "max_value"]:
        df[f"{col}_log"] = np.log1p(df[col])
        df = df.drop(columns=[col])

    # Collapse rare categories (< 5% frequency → 'Other')
    for col in CAT_COLS:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            rare = freq[freq < 0.05].index
            df[col] = df[col].replace(rare, "Other")

    # Impute donation_trend nulls with 0 (donors with only 2 donations)
    df["donation_trend"] = df["donation_trend"].fillna(0.0)

    return df


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------


def get_top_risk_factors(pipeline, feature_names: list, top_n: int = 3) -> list:
    """
    Returns top_n human-readable risk factor labels based on tree feature importances.
    Same labels applied to every supporter — individual SHAP values are a Phase 2 enhancement.
    """
    importances = pipeline.named_steps["model"].feature_importances_
    top_indices = importances.argsort()[::-1][:top_n]
    return [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in top_indices]
