"""
Impact Attribution Inference Job
==================================
-- ONE-TIME SUPABASE SETUP (run before first scoring run) --

    CREATE TABLE donor_impact_statements (
        statement_id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
        supporter_id         INTEGER     NOT NULL REFERENCES supporters(supporter_id),
        safehouse_id         TEXT        NOT NULL,
        program_area         TEXT        NOT NULL,
        allocation_amount    FLOAT       NOT NULL,
        outcome_metric       TEXT        NOT NULL,
        time_window_months   INT         NOT NULL,
        estimated_pct_change FLOAT       NOT NULL,
        statement_text       TEXT        NOT NULL,
        generated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        model_version        TEXT        NOT NULL
    );

    CREATE INDEX idx_impact_statements_supporter
        ON donor_impact_statements (supporter_id);

---

Triggered via POST /score/impact or run directly.

Loads the current effects JSON from Blob Storage, pulls all allocation
records, generates personalized impact statements for donors whose
program areas have statistically significant effects, and writes them
to the donor_impact_statements table.

Logs counts and summary stats only — no individual record data.
"""

import pandas as pd
from sqlalchemy import text
from datetime import datetime, timezone

from db.connection import get_engine
from storage.blob_client import load_json_from_blob
from pipelines.impact_attribution.features import ALLOCATION_OUTCOMES_QUERY
from pipelines.impact_attribution.statement_builder import build_impact_statement


def main() -> int:
    """
    Generate impact statements for all donors. Returns count of statements written.
    """
    engine = get_engine()

    # -------------------------------------------------------------------
    # 1. Load effects + metadata from Blob Storage
    # -------------------------------------------------------------------
    effects  = load_json_from_blob('impact/impact_effects.json')
    metadata = load_json_from_blob('impact/impact_metadata.json')
    model_version = metadata['model_version']

    print(f"Loaded effects for model version {model_version}")
    print(f"Significant program areas: {list(effects.keys())}")

    # -------------------------------------------------------------------
    # 2. Pull allocation records + safehouse names
    # -------------------------------------------------------------------
    df = pd.read_sql(ALLOCATION_OUTCOMES_QUERY, engine)
    sh_names = pd.read_sql("SELECT safehouse_id, name FROM safehouses", engine)
    df = df.merge(sh_names, on='safehouse_id', how='left')

    print(f"Loaded {len(df)} allocation rows for {df['supporter_id'].nunique()} supporters")

    # -------------------------------------------------------------------
    # 3. Generate statements
    # -------------------------------------------------------------------
    statements = []
    skipped_no_effect = 0
    skipped_no_baseline = 0

    for _, row in df.iterrows():
        prog = row['program_area']
        if prog not in effects:
            skipped_no_effect += 1
            continue
        for outcome, effect in effects[prog].items():
            baseline_col = 'edu_baseline' if 'education' in outcome else 'health_baseline'
            if pd.isna(row.get(baseline_col)):
                skipped_no_baseline += 1
                continue
            stmt_text = build_impact_statement(
                supporter_id   = row['supporter_id'],
                safehouse_name = row.get('name', row['safehouse_id']),
                program_area   = prog,
                amount         = row['amount_allocated'],
                outcome        = outcome.split('_')[0],
                coef           = effect['coef'],
                ci_low         = effect['ci_low'],
                ci_high        = effect['ci_high'],
                window         = effect['window'],
                baseline       = row[baseline_col],
            )
            statements.append({
                'supporter_id':          row['supporter_id'],
                'safehouse_id':          row['safehouse_id'],
                'program_area':          prog,
                'allocation_amount':     row['amount_allocated'],
                'outcome_metric':        outcome,
                'time_window_months':    effect['window'],
                'estimated_pct_change':  round(float(effect['coef']), 4),
                'statement_text':        stmt_text,
                'generated_at':          datetime.now(timezone.utc).isoformat(),
                'model_version':         model_version,
            })

    results = pd.DataFrame(statements)

    # -------------------------------------------------------------------
    # 4. Write to Supabase — replace previous version's statements
    # -------------------------------------------------------------------
    if results.empty:
        print("No statements generated — check effects JSON and allocation data.")
        return 0

    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM donor_impact_statements WHERE model_version != :v"),
            {'v': model_version},
        )
        results.to_sql('donor_impact_statements', conn, if_exists='append', index=False)

    unique_donors = results['supporter_id'].nunique()
    print(
        f"Generated {len(results)} impact statements for {unique_donors} donors "
        f"(skipped {skipped_no_effect} rows — no effect; "
        f"{skipped_no_baseline} rows — missing baseline)."
    )

    # Coverage warning
    total_donors = df['supporter_id'].nunique()
    coverage = unique_donors / total_donors if total_donors > 0 else 0
    if coverage < 0.60:
        print(
            f"WARNING: Statement coverage {coverage:.1%} is below the 60% threshold. "
            "Review significant program areas and allocation data."
        )
    else:
        print(f"Statement coverage: {coverage:.1%} of active donors.")

    return len(results)


if __name__ == '__main__':
    main()
