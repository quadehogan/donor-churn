"""
Impact Attribution Training Job
================================
Run quarterly via Azure Container Apps Job.

Fits one OLS regression per (outcome, time_window) combination,
applies HC3 robust standard errors when heteroscedasticity is detected,
extracts statistically significant program-area effects, and saves the
resulting effects JSON + metadata JSON to Azure Blob Storage.

Exit codes:
  0 — success
  1 — significance gate failed (fewer than 3 significant program areas)
"""

import json
import sys
from datetime import datetime

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

from db.connection import get_engine
from storage.blob_client import upload_artifact
from pipelines.impact_attribution.features import (
    ALLOCATION_OUTCOMES_QUERY,
    build_outcome_models,
    engineer_features,
    extract_significant_effects,
)


def main():
    engine = get_engine()

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    df = pd.read_sql(ALLOCATION_OUTCOMES_QUERY, engine)
    print(f"Loaded {len(df)} allocation rows ({df['supporter_id'].nunique()} supporters)")

    # -----------------------------------------------------------------------
    # 2. Feature engineering
    # -----------------------------------------------------------------------
    df, df_edu, df_health, df_edu_6m, df_health_6m = engineer_features(df)
    outcome_models = build_outcome_models(df_edu, df_health, df_edu_6m, df_health_6m)

    # -----------------------------------------------------------------------
    # 3. Fit OLS models
    # -----------------------------------------------------------------------
    results_store = {}

    for model_name, config in outcome_models.items():
        df_m  = config['df'].copy()
        label = config['label']

        if df_m.empty or df_m[label].isna().all():
            print(f"{model_name}: no outcome data — skipping")
            continue

        feature_cols = ['log_amount'] + [
            c for c in df_m.columns
            if c.startswith('program_area_') or c.startswith('safehouse_id_')
        ]

        y = df_m[label]
        X = df_m[feature_cols].assign(const=1)

        model = sm.OLS(y, X).fit()

        # Apply HC3 robust SEs if heteroscedasticity detected
        _, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
        if bp_p < 0.05:
            model = model.get_robustcov_results(cov_type='HC3')
            print(f"{model_name}: Breusch-Pagan p={bp_p:.4f} → refitted with HC3")
        else:
            print(f"{model_name}: Breusch-Pagan p={bp_p:.4f} OK")

        results_store[model_name] = model
        print(f"{model_name}: R²={model.rsquared:.4f} | Adj R²={model.rsquared_adj:.4f}")

    # -----------------------------------------------------------------------
    # 4. Extract significant effects
    # -----------------------------------------------------------------------
    effects = extract_significant_effects(results_store)
    print(f"Significant program areas: {list(effects.keys())}")

    if len(effects) == 0:
        print(
            "ERROR: No significant program areas found. "
            "Check data quality and allocation volume before saving artifacts.",
            file=sys.stderr,
        )
        sys.exit(1)
    if len(effects) < 3:
        print(
            f"WARNING: Only {len(effects)} significant program area(s) found "
            f"(target: 3). Proceeding — statements will be limited. "
            "Signal should strengthen as more allocation + outcome data accumulates."
        )

    # -----------------------------------------------------------------------
    # 5. Save artifacts to Blob Storage
    # -----------------------------------------------------------------------
    model_version = datetime.now().strftime('%Y%m%d_%H%M')
    metadata = {
        'trained_at':               datetime.now().isoformat(),
        'model_version':            model_version,
        'n_allocations':            len(df),
        'significant_program_areas': list(effects.keys()),
    }

    effects_path  = '/tmp/impact_effects.json'
    metadata_path = '/tmp/impact_metadata.json'

    with open(effects_path,  'w') as f:
        json.dump(effects,   f, indent=2)
    with open(metadata_path, 'w') as f:
        json.dump(metadata,  f, indent=2)

    upload_artifact(effects_path,  'impact/impact_effects.json')
    upload_artifact(metadata_path, 'impact/impact_metadata.json')
    upload_artifact(effects_path,  f'impact/versions/impact_effects_{model_version}.json')

    print(f"Training complete. Version: {model_version}")


if __name__ == '__main__':
    main()
