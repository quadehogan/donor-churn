import pickle
from datetime import datetime

from db.connection import get_engine
from pipelines.interventions.cluster_builder import (
    assign_cluster,
    rank_services_within_cluster,
    rank_session_type_within_cluster,
    optimal_sessions_per_month,
    score_social_workers,
)
from pipelines.interventions.features import (
    load_all_data,
    build_training_frame,
    derive_sw_roster,
)
from storage.blob_client import upload_artifact


def train():
    engine = get_engine()

    residents_df, recordings_df, health_df, education_df, incidents_df = load_all_data(engine)
    sw_roster   = derive_sw_roster(recordings_df)
    training_df = build_training_frame(residents_df, recordings_df, health_df, education_df, incidents_df)

    cluster_counts = training_df["profile_cluster"].value_counts().to_dict()
    training_df["profile_cluster"] = training_df.apply(
        lambda row: assign_cluster(row, cluster_counts=cluster_counts), axis=1
    )

    cluster_recommendations = {}
    for cluster_label, cluster_df in training_df.groupby("profile_cluster"):
        valid_df = cluster_df.dropna(subset=["composite_score_delta"])
        cluster_recommendations[cluster_label] = {
            "recommended_services":     rank_services_within_cluster(valid_df) if not valid_df.empty else [],
            "recommended_session_type": rank_session_type_within_cluster(valid_df),
            "sessions_per_month":       optimal_sessions_per_month(valid_df),
            "cluster_size":             len(cluster_df),
        }

    model_version = datetime.now().strftime("%Y%m%d_%H%M")
    artifact = {
        "model_version":           model_version,
        "cluster_recommendations": cluster_recommendations,
        "sw_roster":               sw_roster.to_dict(orient="records"),
    }

    model_path = "/tmp/interventions_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    upload_artifact(model_path, f"interventions/versions/interventions_model_{model_version}.pkl")
    upload_artifact(model_path, "interventions/interventions_model.pkl")

    print(f"Trained intervention model — {len(cluster_recommendations)} clusters | version {model_version}")


if __name__ == "__main__":
    train()
