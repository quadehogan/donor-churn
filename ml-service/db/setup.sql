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

-- -------------------------------------------------------------------------
-- Intervention recommendations (RLS restricted — Admin + SocialWorker only)
-- -------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS intervention_recommendations (
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

ALTER TABLE intervention_recommendations ENABLE ROW LEVEL SECURITY;

CREATE POLICY intervention_recommendations_admin_sw
    ON intervention_recommendations
    FOR ALL
    USING (auth.jwt() ->> 'role' IN ('Admin', 'SocialWorker'));

-- -------------------------------------------------------------------------
-- Social media recommendations (no RLS — not sensitive)
-- -------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS social_media_recommendations (
    recommendation_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    platform                     TEXT        NOT NULL,
    is_boosted                   BOOLEAN     NOT NULL,
    post_type                    TEXT,
    media_type                   TEXT,
    content_topic                TEXT,
    sentiment_tone               TEXT,
    has_call_to_action           BOOLEAN,
    call_to_action_type          TEXT,
    features_resident_story      BOOLEAN,
    best_day_of_week             TEXT,
    best_hour                    INTEGER,
    recommended_hashtag_count    INTEGER,
    predicted_engagement_rate    FLOAT,
    predicted_donation_referrals FLOAT,
    predicted_donation_value_php FLOAT,
    conversion_signal            TEXT,
    sample_count                 INTEGER,
    confidence_tier              TEXT,
    generated_at                 TIMESTAMPTZ,
    model_version                TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS social_media_recs_platform_boosted
    ON social_media_recommendations (platform, is_boosted);
