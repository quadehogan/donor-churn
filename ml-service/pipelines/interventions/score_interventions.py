import json
import os
import pickle
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from db.connection import get_engine
from pipelines.interventions.cluster_builder import (
    assign_cluster,
    assign_confidence_tier,
    build_top_outcome_factors,
    score_social_workers,
)
from pipelines.interventions.features import (
    load_all_data,
    build_age_band,
    build_profile_label,
    derive_sw_roster,
)
from storage.blob_client import load_model_from_blob

load_dotenv()


def main():
    engine = get_engine()

    artifact              = load_model_from_blob("interventions/interventions_model.pkl")
    cluster_recommendations = artifact["cluster_recommendations"]
    sw_roster_df          = pd.DataFrame(artifact["sw_roster"])
    model_version         = artifact["model_version"]

    residents_df, recordings_df, _, _, _ = load_all_data(engine)
    active_residents = residents_df[residents_df["case_status"] == "Active"].copy()

    active_residents["age_band"] = active_residents["present_age"].apply(build_age_band)
    cluster_counts = {k: v["cluster_size"] for k, v in cluster_recommendations.items()}
    active_residents["profile_cluster"] = active_residents.apply(
        lambda row: assign_cluster(row, cluster_counts=cluster_counts), axis=1
    )

    records = []
    for _, resident in active_residents.iterrows():
        cluster_label = resident["profile_cluster"]
        safehouse_id  = resident["safehouse_id"]
        recs          = cluster_recommendations.get(cluster_label, {})
        similar_count = recs.get("cluster_size", 0)

        recommended_sw, sw_outcome_score = score_social_workers(
            cluster_df=None,
            sw_roster=sw_roster_df,
            safehouse_id=safehouse_id,
        )

        sessions_label = recs.get("sessions_per_month", "4–6")
        sessions_int   = {"1–3": 2, "4–6": 5, "7+": 8}.get(sessions_label, 5)

        factors = build_top_outcome_factors(
            cluster_label,
            recs.get("recommended_services", []),
            recs.get("recommended_session_type"),
            sessions_label,
            recommended_sw,
            sw_outcome_score,
            similar_count,
        )

        records.append({
            "resident_id":                     int(resident["resident_id"]),
            "profile_cluster":                 cluster_label,
            "recommended_services":            json.dumps(recs.get("recommended_services", [])),
            "recommended_session_type":        recs.get("recommended_session_type"),
            "recommended_sessions_per_month":  sessions_int,
            "recommended_social_worker":       recommended_sw,
            "sw_outcome_score":                sw_outcome_score,
            "predicted_health_improvement":    recs.get("predicted_health_improvement"),
            "predicted_education_improvement": recs.get("predicted_education_improvement"),
            "similar_resident_count":          similar_count,
            "confidence_tier":                 assign_confidence_tier(similar_count),
            "top_outcome_factors":             json.dumps(factors),
            "scored_at":                       datetime.now(timezone.utc).isoformat(),
            "model_version":                   model_version,
        })

    with engine.begin() as conn:
        for row in records:
            conn.execute(text("""
                INSERT INTO intervention_recommendations
                    (resident_id, profile_cluster, recommended_services,
                     recommended_session_type, recommended_sessions_per_month,
                     recommended_social_worker, sw_outcome_score,
                     predicted_health_improvement, predicted_education_improvement,
                     similar_resident_count, confidence_tier,
                     top_outcome_factors, scored_at, model_version)
                VALUES
                    (:resident_id, :profile_cluster, CAST(:recommended_services AS jsonb),
                     :recommended_session_type, :recommended_sessions_per_month,
                     :recommended_social_worker, :sw_outcome_score,
                     :predicted_health_improvement, :predicted_education_improvement,
                     :similar_resident_count, :confidence_tier,
                     CAST(:top_outcome_factors AS jsonb), :scored_at, :model_version)
                ON CONFLICT (resident_id) DO UPDATE SET
                    profile_cluster                 = EXCLUDED.profile_cluster,
                    recommended_services            = EXCLUDED.recommended_services,
                    recommended_session_type        = EXCLUDED.recommended_session_type,
                    recommended_sessions_per_month  = EXCLUDED.recommended_sessions_per_month,
                    recommended_social_worker       = EXCLUDED.recommended_social_worker,
                    sw_outcome_score                = EXCLUDED.sw_outcome_score,
                    predicted_health_improvement    = EXCLUDED.predicted_health_improvement,
                    predicted_education_improvement = EXCLUDED.predicted_education_improvement,
                    similar_resident_count          = EXCLUDED.similar_resident_count,
                    confidence_tier                 = EXCLUDED.confidence_tier,
                    top_outcome_factors             = EXCLUDED.top_outcome_factors,
                    scored_at                       = EXCLUDED.scored_at,
                    model_version                   = EXCLUDED.model_version
            """), row)

    print(f"Intervention scoring complete — {len(records)} residents scored")
    return len(records)
