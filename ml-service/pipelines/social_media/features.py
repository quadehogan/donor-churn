import os

import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("SUPABASE_DB_URL"))

SOCIAL_MEDIA_QUERY = """
SELECT
    post_id,
    platform,
    day_of_week,
    post_hour,
    post_type,
    media_type,
    content_topic,
    sentiment_tone,
    has_call_to_action,
    call_to_action_type,
    features_resident_story,
    num_hashtags,
    caption_length,
    is_boosted,

    -- Outcome targets
    engagement_rate,
    donation_referrals,
    estimated_donation_value_php,

    -- Context / metadata
    platform_post_id,
    created_at
FROM social_media_posts
ORDER BY created_at
"""


def load_posts(engine):
    df = pd.read_sql(SOCIAL_MEDIA_QUERY, engine)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def clean_posts(df):
    df = df.copy()
    df["call_to_action_type"] = df["call_to_action_type"].fillna("None")
    df["is_boosted"] = df["is_boosted"].fillna(False)
    df = df.dropna(subset=["engagement_rate", "donation_referrals"])
    return df


CATEGORICAL_FEATURES = [
    "platform", "post_type", "media_type", "content_topic",
    "sentiment_tone", "call_to_action_type", "day_of_week",
]
BINARY_FEATURES  = ["has_call_to_action", "features_resident_story", "is_boosted"]
NUMERIC_FEATURES = ["post_hour", "num_hashtags", "caption_length"]

FEATURE_COLS_MODEL = CATEGORICAL_FEATURES + BINARY_FEATURES + NUMERIC_FEATURES
