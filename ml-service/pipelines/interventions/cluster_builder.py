import pandas as pd

from pipelines.interventions.features import build_age_band, build_abuse_flag_string, build_profile_label


def assign_cluster(row, min_cluster_size=5, cluster_counts=None):
    """
    Rule-based cluster assignment: category → risk level → age band.
    Falls back to broader groups when cells are too small.
    """
    full_label = build_profile_label(row)
    if cluster_counts and cluster_counts.get(full_label, 0) >= min_cluster_size:
        return full_label

    partial_label = f"{row['current_risk_level']} · {build_abuse_flag_string(row)}"
    if cluster_counts and cluster_counts.get(partial_label, 0) >= min_cluster_size:
        return partial_label

    return row["current_risk_level"]


def rank_services_within_cluster(cluster_df):
    service_perf = (
        cluster_df.groupby("primary_service")["composite_score_delta"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    return service_perf["primary_service"].tolist()


def rank_session_type_within_cluster(cluster_df):
    valid = cluster_df.dropna(subset=["dominant_session_type", "composite_score_delta"])
    if valid.empty:
        return None
    return (
        valid.groupby("dominant_session_type")["composite_score_delta"]
        .mean()
        .idxmax()
    )


def optimal_sessions_per_month(cluster_df):
    cluster_df = cluster_df.copy()
    cluster_df["freq_band"] = pd.cut(
        cluster_df["sessions_per_month"],
        bins=[0, 3, 6, 99],
        labels=["1–3", "4–6", "7+"],
    )
    valid = cluster_df.dropna(subset=["freq_band", "composite_score_delta"])
    if valid.empty:
        return "4–6"
    return (
        valid.groupby("freq_band", observed=False)["composite_score_delta"]
        .mean()
        .idxmax()
    )


def score_social_workers(cluster_df, sw_roster, safehouse_id):
    """
    Returns the SW code with the highest mean composite outcome improvement
    for this cluster at the given safehouse, drawn only from the active roster.
    During scoring (cluster_df=None) returns None — SW lookup is roster-only.
    """
    available_sws = sw_roster[sw_roster["safehouse_id"] == safehouse_id]["social_worker"].tolist()
    if cluster_df is None or cluster_df.empty:
        if available_sws:
            return available_sws[0], None
        return None, None

    sw_perf = (
        cluster_df[cluster_df["primary_sw"].isin(available_sws)]
        .groupby("primary_sw")["composite_score_delta"]
        .agg(["mean", "count"])
        .reset_index()
    )
    sw_perf.columns = ["social_worker", "mean_outcome", "session_count"]
    if sw_perf.empty:
        return None, None
    best = sw_perf.sort_values("mean_outcome", ascending=False).iloc[0]
    return best["social_worker"], round(best["mean_outcome"], 3)


def assign_confidence_tier(similar_count):
    if similar_count >= 15:
        return "high"
    elif similar_count >= 5:
        return "medium"
    else:
        return "low"


def build_top_outcome_factors(cluster_label, recommended_services, recommended_session_type,
                               sessions_per_month, recommended_sw, sw_outcome_score, cluster_size):
    factors = []
    if recommended_services:
        service_name = recommended_services[0]
        factors.append(
            f"{service_name} sessions show the strongest composite improvements "
            f"for residents in the '{cluster_label}' profile"
        )
    if recommended_session_type and sessions_per_month:
        factors.append(
            f"{recommended_session_type} sessions at ~{sessions_per_month} per month "
            f"outperform other formats for this profile"
        )
    if recommended_sw and sw_outcome_score is not None:
        factors.append(
            f"{recommended_sw} has the highest average composite outcome improvement "
            f"for similar profiles at this safehouse (score: {sw_outcome_score:.2f})"
        )
    if not factors:
        factors.append(f"Based on {cluster_size} similar residents in the '{cluster_label}' cluster")
    return factors[:3]
