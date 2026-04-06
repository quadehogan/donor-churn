"""
Batch inference job — scores all active donors and upserts to Supabase.
Called by the FastAPI /score/churn endpoint and by the training job after retraining.
Can also be run directly: python pipelines/churn/score_churn.py
"""

import json
import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from db.connection import get_engine
from pipelines.churn.features import INFERENCE_QUERY, engineer_features, get_top_risk_factors
from storage.blob_client import load_json_from_blob, load_model_from_blob

load_dotenv()


def main():
    engine = get_engine()

    print("Loading model and metadata from Azure Blob Storage...")
    pipeline = load_model_from_blob("churn/churn_model.pkl")
    metadata = load_json_from_blob("churn/churn_metadata.json")
    model_version = metadata["model_version"]
    feature_names = metadata["features"]

    print("Loading active supporters from Supabase...")
    df_raw = pd.read_sql(INFERENCE_QUERY, engine)
    print(f"Supporters to score: {len(df_raw)}")

    supporter_ids = df_raw["supporter_id"].astype(int).tolist()
    df = engineer_features(df_raw, engine=engine)
    X = df.drop(columns=["supporter_id"])

    probabilities = pipeline.predict_proba(X)[:, 1]

    def assign_tier(prob: float) -> str:
        if prob >= 0.70:
            return "high"
        if prob >= 0.40:
            return "medium"
        return "low"

    top_factors = get_top_risk_factors(pipeline, feature_names, top_n=3)

    results = pd.DataFrame({
        "supporter_id":      supporter_ids,
        "churn_probability": probabilities.round(4),
        "risk_tier":         [assign_tier(p) for p in probabilities],
        "top_risk_factors":  [json.dumps(top_factors)] * len(supporter_ids),
        "scored_at":         datetime.now().isoformat(),
        "model_version":     model_version,
    })

    print("Upserting scores to Supabase...")
    with engine.begin() as conn:
        for _, row in results.iterrows():
            conn.execute(text("""
                INSERT INTO donor_churn_scores
                    (supporter_id, churn_probability, risk_tier,
                     top_risk_factors, scored_at, model_version)
                VALUES
                    (:supporter_id, :churn_probability, :risk_tier,
                     :top_risk_factors::jsonb, :scored_at, :model_version)
                ON CONFLICT (supporter_id) DO UPDATE SET
                    churn_probability = EXCLUDED.churn_probability,
                    risk_tier         = EXCLUDED.risk_tier,
                    top_risk_factors  = EXCLUDED.top_risk_factors,
                    scored_at         = EXCLUDED.scored_at,
                    model_version     = EXCLUDED.model_version
            """), row.to_dict())

    high   = (results["risk_tier"] == "high").sum()
    medium = (results["risk_tier"] == "medium").sum()
    low    = (results["risk_tier"] == "low").sum()
    print(f"Done — scored {len(results)} supporters | High: {high} | Medium: {medium} | Low: {low}")
    return len(results)


if __name__ == "__main__":
    main()
