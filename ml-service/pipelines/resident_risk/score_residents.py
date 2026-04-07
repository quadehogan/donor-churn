"""
Resident Risk & Reintegration Scoring Job
==========================================
-- ONE-TIME SUPABASE SETUP (run before first scoring run) --

    CREATE TABLE resident_risk_scores (
        resident_id              INTEGER PRIMARY KEY REFERENCES residents(resident_id),
        regression_risk_score    FLOAT NOT NULL,
        regression_risk_tier     TEXT  NOT NULL CHECK (regression_risk_tier IN ('high','medium','low')),
        reintegration_score      FLOAT NOT NULL,
        reintegration_tier       TEXT  NOT NULL CHECK (reintegration_tier IN ('ready','in_progress','not_ready')),
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

---

PRIVACY NOTICE: This script processes records of minors who are abuse survivors.
- Logs counts and aggregate statistics only — no individual data
- Outputs are restricted to authenticated Admin and SocialWorker roles via Supabase RLS
- Never expose resident_risk_scores data to any donor-facing or external surface
"""

import json
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import text

from db.connection import get_engine
from storage.blob_client import load_json_from_blob, load_model_from_blob
from pipelines.resident_risk.features import (
    CAT_COLS,
    CONCERN_LABELS,
    NUM_COLS,
    STRENGTH_LABELS,
    build_labels,
    engineer_features,
    get_feature_names,
    get_top_factors,
    load_and_merge,
)


def _risk_tier(p: float) -> str:
    return 'high' if p >= 0.70 else 'medium' if p >= 0.40 else 'low'


def _reint_tier(p: float) -> str:
    return 'ready' if p >= 0.70 else 'in_progress' if p >= 0.40 else 'not_ready'


def main() -> int:
    """Score all active residents. Returns count of residents scored."""
    engine = get_engine()

    # -------------------------------------------------------------------
    # 1. Load models + metadata from Blob Storage
    # -------------------------------------------------------------------
    risk_model  = load_model_from_blob('resident_risk/risk_model.pkl')
    reint_model = load_model_from_blob('resident_risk/reint_model.pkl')
    metadata    = load_json_from_blob('resident_risk/risk_metadata.json')
    model_version = metadata['model_version']

    print(f"Loaded models — version {model_version}")

    # -------------------------------------------------------------------
    # 2. Load + prepare resident data
    # -------------------------------------------------------------------
    df = load_and_merge(engine)
    df = build_labels(df)
    df = engineer_features(df)

    resident_ids = df['resident_id'].tolist()
    X = df[NUM_COLS + CAT_COLS]

    # -------------------------------------------------------------------
    # 3. Score
    # -------------------------------------------------------------------
    risk_proba  = risk_model.predict_proba(X)[:, 1]
    reint_proba = reint_model.predict_proba(X)[:, 1]

    feat_names      = get_feature_names(risk_model,  NUM_COLS, CAT_COLS)
    concern_factors = get_top_factors(risk_model,  feat_names, CONCERN_LABELS)
    strength_factors = get_top_factors(reint_model, feat_names, STRENGTH_LABELS)

    scored_at = datetime.now(timezone.utc).isoformat()

    results = pd.DataFrame({
        'resident_id':           resident_ids,
        'regression_risk_score': risk_proba.round(4),
        'regression_risk_tier':  [_risk_tier(p)  for p in risk_proba],
        'reintegration_score':   reint_proba.round(4),
        'reintegration_tier':    [_reint_tier(p) for p in reint_proba],
        'top_concern_factors':   [json.dumps(concern_factors)  for _ in resident_ids],
        'top_strength_factors':  [json.dumps(strength_factors) for _ in resident_ids],
        'scored_at':             scored_at,
        'model_version':         model_version,
    })

    # -------------------------------------------------------------------
    # 4. Upsert to Supabase (ON CONFLICT DO UPDATE)
    # -------------------------------------------------------------------
    with engine.begin() as conn:
        for _, row in results.iterrows():
            conn.execute(text("""
                INSERT INTO resident_risk_scores
                    (resident_id, regression_risk_score, regression_risk_tier,
                     reintegration_score, reintegration_tier,
                     top_concern_factors, top_strength_factors,
                     scored_at, model_version)
                VALUES
                    (:resident_id, :regression_risk_score, :regression_risk_tier,
                     :reintegration_score, :reintegration_tier,
                     CAST(:top_concern_factors AS jsonb), CAST(:top_strength_factors AS jsonb),
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

    # -------------------------------------------------------------------
    # 5. Summary log (counts only — no individual resident data)
    # -------------------------------------------------------------------
    high_risk  = (results['regression_risk_tier'] == 'high').sum()
    med_risk   = (results['regression_risk_tier'] == 'medium').sum()
    ready      = (results['reintegration_tier'] == 'ready').sum()
    in_prog    = (results['reintegration_tier'] == 'in_progress').sum()
    n_total    = len(results)

    print(
        f"Scored {n_total} residents — "
        f"Risk: {high_risk} high / {med_risk} medium / {n_total - high_risk - med_risk} low | "
        f"Reintegration: {ready} ready / {in_prog} in_progress"
    )

    # Alert if any safehouse has >30% high-risk residents
    safehouse_stats = df.copy()
    safehouse_stats['risk_tier'] = results['regression_risk_tier'].values
    high_rate = (
        safehouse_stats.groupby('safehouse_id')['risk_tier']
        .apply(lambda s: (s == 'high').mean())
    )
    flagged = high_rate[high_rate > 0.30]
    if not flagged.empty:
        print(
            f"WARNING: {len(flagged)} safehouse(s) have >30% high-risk residents. "
            "Review flagged safehouses — counts per safehouse logged above."
        )
        for sh_id, rate in flagged.items():
            count = (safehouse_stats[safehouse_stats['safehouse_id'] == sh_id]['risk_tier'] == 'high').sum()
            print(f"  Safehouse {sh_id}: {count} high-risk residents ({rate:.0%})")

    return n_total


if __name__ == '__main__':
    main()
