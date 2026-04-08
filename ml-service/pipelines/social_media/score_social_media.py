import json
import os
import pickle
import uuid
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text

from db.connection import get_engine
from pipelines.social_media.features import (
    load_posts, clean_posts, FEATURE_COLS_MODEL,
)
from storage.blob_client import load_model_from_blob

load_dotenv()

PLATFORMS = ["Facebook", "Instagram", "TikTok", "WhatsApp", "LinkedIn"]


def assign_confidence_tier_social(sample_count):
    if sample_count >= 30:
        return "high"
    elif sample_count >= 10:
        return "medium"
    else:
        return "low"


def assign_conversion_signal(predicted_engagement, predicted_referrals, platform_median_referrals):
    if predicted_referrals >= 1.5 * platform_median_referrals:
        return "converts"
    elif predicted_engagement >= 0.12 and predicted_referrals < 0.8 * platform_median_referrals:
        return "noise"
    else:
        return "balanced"


def find_best_combination(platform, is_boosted, posts_df, engagement_model, referrals_model, value_model):
    MIN_SAMPLES_PER_COMBO = 3

    subset = posts_df[(posts_df["platform"] == platform) & (posts_df["is_boosted"] == is_boosted)]
    if len(subset) < 5:
        return None

    combo_cols = [
        "post_type", "media_type", "content_topic", "sentiment_tone",
        "has_call_to_action", "call_to_action_type", "features_resident_story",
    ]
    combo_counts = subset.groupby(combo_cols).size().reset_index(name="sample_count")
    valid_combos = combo_counts[combo_counts["sample_count"] >= MIN_SAMPLES_PER_COMBO]

    if valid_combos.empty:
        return None

    best_day      = subset.groupby("day_of_week")["donation_referrals"].mean().idxmax()
    best_hour     = subset.groupby("post_hour")["donation_referrals"].mean().idxmax()
    best_hashtags = int(subset.groupby("num_hashtags")["donation_referrals"].mean().idxmax())

    best_score = -999
    best_rec   = None

    for _, combo_row in valid_combos.iterrows():
        candidate = {
            "platform":                platform,
            "is_boosted":              is_boosted,
            "post_type":               combo_row["post_type"],
            "media_type":              combo_row["media_type"],
            "content_topic":           combo_row["content_topic"],
            "sentiment_tone":          combo_row["sentiment_tone"],
            "has_call_to_action":      combo_row["has_call_to_action"],
            "call_to_action_type":     combo_row["call_to_action_type"],
            "features_resident_story": combo_row["features_resident_story"],
            "day_of_week":             best_day,
            "post_hour":               best_hour,
            "num_hashtags":            best_hashtags,
            "caption_length":          int(subset["caption_length"].median()),
        }
        candidate_df   = pd.DataFrame([candidate])[FEATURE_COLS_MODEL]
        pred_referrals = float(referrals_model.predict(candidate_df)[0])

        if pred_referrals > best_score:
            best_score      = pred_referrals
            pred_engagement = float(engagement_model.predict(candidate_df)[0])
            pred_value      = float(value_model.predict(candidate_df)[0])
            best_rec = {
                **candidate,
                "predicted_engagement_rate":    round(pred_engagement, 4),
                "predicted_donation_referrals": round(pred_referrals, 4),
                "predicted_donation_value_php": round(pred_value, 2),
                "sample_count":                 int(combo_row["sample_count"]),
            }

    return best_rec


def main():
    engine = get_engine()

    artifact         = load_model_from_blob("social_media/social_media_model.pkl")
    engagement_model = artifact["engagement_model"]
    referrals_model  = artifact["referrals_model"]
    value_model      = artifact["value_model"]
    model_version    = artifact["model_version"]
    platform_medians = artifact["platform_medians"]

    posts_df = clean_posts(load_posts(engine))
    records  = []

    for platform in PLATFORMS:
        for is_boosted in [False, True]:
            rec = find_best_combination(
                platform, is_boosted, posts_df,
                engagement_model, referrals_model, value_model,
            )
            if rec is None:
                continue

            platform_median   = platform_medians.get(platform, 1.0)
            conversion_signal = assign_conversion_signal(
                rec["predicted_engagement_rate"],
                rec["predicted_donation_referrals"],
                platform_median,
            )

            records.append({
                "recommendation_id":            str(uuid.uuid4()),
                "platform":                     platform,
                "is_boosted":                   rec["is_boosted"],
                "post_type":                    rec["post_type"],
                "media_type":                   rec["media_type"],
                "content_topic":                rec["content_topic"],
                "sentiment_tone":               rec["sentiment_tone"],
                "has_call_to_action":           bool(rec["has_call_to_action"]),
                "call_to_action_type":          rec["call_to_action_type"] if rec["has_call_to_action"] else None,
                "features_resident_story":      bool(rec["features_resident_story"]),
                "best_day_of_week":             rec["day_of_week"],
                "best_hour":                    int(rec["post_hour"]),
                "recommended_hashtag_count":    int(rec["num_hashtags"]),
                "predicted_engagement_rate":    rec["predicted_engagement_rate"],
                "predicted_donation_referrals": rec["predicted_donation_referrals"],
                "predicted_donation_value_php": rec["predicted_donation_value_php"],
                "conversion_signal":            conversion_signal,
                "sample_count":                 rec["sample_count"],
                "confidence_tier":              assign_confidence_tier_social(rec["sample_count"]),
                "generated_at":                 datetime.now(timezone.utc).isoformat(),
                "model_version":                model_version,
            })

    with engine.begin() as conn:
        # Replace all recommendations fresh each run
        conn.execute(text("DELETE FROM social_media_recommendations"))
        for row in records:
            conn.execute(text("""
                INSERT INTO social_media_recommendations
                    (recommendation_id, platform, is_boosted, post_type, media_type,
                     content_topic, sentiment_tone, has_call_to_action, call_to_action_type,
                     features_resident_story, best_day_of_week, best_hour,
                     recommended_hashtag_count, predicted_engagement_rate,
                     predicted_donation_referrals, predicted_donation_value_php,
                     conversion_signal, sample_count, confidence_tier,
                     generated_at, model_version)
                VALUES
                    (:recommendation_id, :platform, :is_boosted, :post_type, :media_type,
                     :content_topic, :sentiment_tone, :has_call_to_action, :call_to_action_type,
                     :features_resident_story, :best_day_of_week, :best_hour,
                     :recommended_hashtag_count, :predicted_engagement_rate,
                     :predicted_donation_referrals, :predicted_donation_value_php,
                     :conversion_signal, :sample_count, :confidence_tier,
                     :generated_at, :model_version)
            """), row)

    print(f"Social media recommendations written — {len(records)} rows")
    return len(records)
