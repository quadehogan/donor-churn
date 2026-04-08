import os

import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("SUPABASE_DB_URL"))

RESIDENTS_QUERY = """
SELECT
    r.resident_id,
    r.safehouse_id,
    r.case_status,
    r.case_category,
    r.current_risk_level,
    r.present_age,

    -- Abuse sub-type flags
    r.sub_cat_physical_abuse,
    r.sub_cat_sexual_abuse,
    r.sub_cat_trafficked,
    r.sub_cat_osaec,
    r.sub_cat_child_labor,
    r.sub_cat_cicl,
    r.sub_cat_at_risk,
    r.sub_cat_orphaned,
    r.sub_cat_street_child,
    r.sub_cat_child_with_hiv,

    -- Reintegration context
    r.reintegration_type,
    r.reintegration_status,
    r.initial_risk_level,
    r.length_of_stay

FROM residents r
WHERE r.case_status = 'Active'
"""

RECORDINGS_QUERY = """
SELECT
    pr.recording_id,
    pr.resident_id,
    pr.session_date,
    pr.social_worker,
    pr.session_type,
    pr.session_duration_minutes,
    pr.interventions_applied,
    pr.progress_noted,
    pr.concerns_flagged,
    r.safehouse_id
FROM process_recordings pr
JOIN residents r ON pr.resident_id = r.resident_id
ORDER BY pr.resident_id, pr.session_date
"""

HEALTH_QUERY = """
SELECT
    resident_id,
    record_date,
    general_health_score
FROM health_wellbeing_records
ORDER BY resident_id, record_date
"""

EDUCATION_QUERY = """
SELECT
    resident_id,
    record_date,
    progress_percent
FROM education_records
ORDER BY resident_id, record_date
"""

INCIDENTS_QUERY = """
SELECT
    resident_id,
    incident_date,
    severity,
    resolved
FROM incident_reports
ORDER BY resident_id, incident_date
"""


def load_all_data(engine):
    residents_df  = pd.read_sql(RESIDENTS_QUERY, engine)
    recordings_df = pd.read_sql(RECORDINGS_QUERY, engine)
    health_df     = pd.read_sql(HEALTH_QUERY, engine)
    education_df  = pd.read_sql(EDUCATION_QUERY, engine)
    incidents_df  = pd.read_sql(INCIDENTS_QUERY, engine)
    return residents_df, recordings_df, health_df, education_df, incidents_df


def derive_sw_roster(recordings_df, months_lookback=12):
    """
    Returns a DataFrame of (safehouse_id, social_worker, session_count)
    for SWs with >= 3 sessions at a safehouse in the last N months.
    """
    cutoff = pd.Timestamp.now() - pd.DateOffset(months=months_lookback)
    session_dates = pd.to_datetime(recordings_df["session_date"])
    recent = recordings_df[session_dates >= cutoff]
    roster = (
        recent.groupby(["safehouse_id", "social_worker"])
        .size()
        .reset_index(name="session_count")
    )
    return roster[roster["session_count"] >= 3]


def build_age_band(age_str):
    try:
        years = int(str(age_str).split()[0])
        if years < 10:
            return "Under 10"
        elif years <= 13:
            return "10–13"
        elif years <= 17:
            return "14–17"
        else:
            return "18+"
    except (ValueError, IndexError):
        return "Unknown"


def build_abuse_flag_string(row):
    flags = []
    flag_cols = [
        ("sub_cat_physical_abuse", "Physical Abuse"),
        ("sub_cat_sexual_abuse", "Sexual Abuse"),
        ("sub_cat_trafficked", "Trafficked"),
        ("sub_cat_osaec", "OSAEC"),
        ("sub_cat_child_labor", "Child Labor"),
        ("sub_cat_cicl", "CICL"),
    ]
    for col, label in flag_cols:
        if row.get(col) is True or row.get(col) == "True":
            flags.append(label)
    return " · ".join(flags) if flags else "No Sub-type"


def build_profile_label(row):
    risk  = row["current_risk_level"]
    abuse = build_abuse_flag_string(row)
    age   = build_age_band(row["present_age"])
    return f"{risk} · {abuse} · Age {age}"


def compute_health_delta(health_df, session_date, resident_id, window_days=90):
    resident_health = health_df[health_df["resident_id"] == resident_id].copy()
    resident_health["record_date"] = pd.to_datetime(resident_health["record_date"])
    session_date = pd.to_datetime(session_date)

    baseline = resident_health[resident_health["record_date"] <= session_date]
    followup = resident_health[
        (resident_health["record_date"] > session_date) &
        (resident_health["record_date"] <= session_date + pd.Timedelta(days=window_days))
    ]
    if baseline.empty or followup.empty:
        return None
    return followup["general_health_score"].mean() - baseline["general_health_score"].iloc[-1]


def compute_edu_delta(education_df, session_date, resident_id, window_days=90):
    resident_edu = education_df[education_df["resident_id"] == resident_id].copy()
    resident_edu["record_date"] = pd.to_datetime(resident_edu["record_date"])
    session_date = pd.to_datetime(session_date)

    baseline = resident_edu[resident_edu["record_date"] <= session_date]
    followup = resident_edu[
        (resident_edu["record_date"] > session_date) &
        (resident_edu["record_date"] <= session_date + pd.Timedelta(days=window_days))
    ]
    if baseline.empty or followup.empty:
        return None
    return followup["progress_percent"].mean() - baseline["progress_percent"].iloc[-1]


def compute_incident_delta(incidents_df, session_date, resident_id, window_days=90):
    resident_inc = incidents_df[incidents_df["resident_id"] == resident_id].copy()
    resident_inc["incident_date"] = pd.to_datetime(resident_inc["incident_date"])
    session_date = pd.to_datetime(session_date)

    before = resident_inc[resident_inc["incident_date"] <= session_date]
    after  = resident_inc[
        (resident_inc["incident_date"] > session_date) &
        (resident_inc["incident_date"] <= session_date + pd.Timedelta(days=window_days))
    ]
    return len(after) - len(before)


def compute_composite_delta(resident_id, session_date, health_df, education_df, incidents_df,
                             w_health=0.40, w_edu=0.35, w_incidents=0.25, window_days=90):
    delta_health = compute_health_delta(health_df, session_date, resident_id, window_days)
    delta_edu    = compute_edu_delta(education_df, session_date, resident_id, window_days)
    delta_inc    = compute_incident_delta(incidents_df, session_date, resident_id, window_days)

    inc_component = -delta_inc if delta_inc is not None else None
    components = [(delta_health, w_health), (delta_edu, w_edu), (inc_component, w_incidents)]
    valid = [(v, w) for v, w in components if v is not None]
    if not valid:
        return None
    total_weight = sum(w for _, w in valid)
    return sum(v * w for v, w in valid) / total_weight


def summarize_intervention_window(recordings_window):
    services     = recordings_window["interventions_applied"].value_counts()
    session_types = recordings_window["session_type"].value_counts()
    sw_counts    = recordings_window["social_worker"].value_counts()

    return {
        "primary_service":       services.index[0] if not services.empty else None,
        "session_count":         len(recordings_window),
        "sessions_per_month":    len(recordings_window) / 3,
        "dominant_session_type": session_types.index[0] if not session_types.empty else None,
        "primary_sw":            sw_counts.index[0] if not sw_counts.empty else None,
    }


def build_training_frame(residents_df, recordings_df, health_df, education_df, incidents_df,
                          window_days=90):
    """
    For each resident × intervention window, build one training row with profile
    features, intervention summary, and composite outcome delta.
    """
    recordings_df = recordings_df.copy()
    recordings_df["session_date"] = pd.to_datetime(recordings_df["session_date"])

    rows = []
    for resident_id, res_recordings in recordings_df.groupby("resident_id"):
        resident_row = residents_df[residents_df["resident_id"] == resident_id]
        if resident_row.empty:
            continue
        resident_row = resident_row.iloc[0]

        for i in range(0, len(res_recordings), max(1, len(res_recordings) // 3)):
            window = res_recordings.iloc[i:i + max(1, len(res_recordings) // 3)]
            if window.empty:
                continue
            session_date = window["session_date"].min()
            summary = summarize_intervention_window(window)
            composite = compute_composite_delta(
                resident_id, session_date, health_df, education_df, incidents_df, window_days=window_days
            )
            row = {
                "resident_id":         resident_id,
                "safehouse_id":        resident_row["safehouse_id"],
                "case_category":       resident_row["case_category"],
                "current_risk_level":  resident_row["current_risk_level"],
                "present_age":         resident_row["present_age"],
                "age_band":            build_age_band(resident_row["present_age"]),
                "profile_cluster":     build_profile_label(resident_row),
                "composite_score_delta": composite,
                **summary,
            }
            for col in [
                "sub_cat_physical_abuse", "sub_cat_sexual_abuse", "sub_cat_trafficked",
                "sub_cat_osaec", "sub_cat_child_labor", "sub_cat_cicl",
                "sub_cat_at_risk", "sub_cat_orphaned",
            ]:
                row[col] = resident_row.get(col, False)
            rows.append(row)

    return pd.DataFrame(rows)
