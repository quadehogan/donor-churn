-- Run once in Supabase SQL editor to create the churn scores table.

CREATE TABLE IF NOT EXISTS donor_churn_scores (
    supporter_id        INTEGER PRIMARY KEY REFERENCES supporters(supporter_id),
    churn_probability   FLOAT        NOT NULL,
    risk_tier           TEXT         NOT NULL CHECK (risk_tier IN ('high', 'medium', 'low')),
    top_risk_factors    JSONB,
    scored_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    model_version       TEXT         NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_churn_scores_risk_tier
    ON donor_churn_scores (risk_tier);
