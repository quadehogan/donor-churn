import pickle
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from db.connection import get_engine
from pipelines.social_media.features import (
    clean_posts, load_posts,
    CATEGORICAL_FEATURES, BINARY_FEATURES, NUMERIC_FEATURES, FEATURE_COLS_MODEL,
)
from storage.blob_client import upload_artifact


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),
        ("num", "passthrough", NUMERIC_FEATURES),
    ])


def train():
    engine = get_engine()

    posts_df  = clean_posts(load_posts(engine))
    posts_df  = posts_df.sort_values("created_at")
    split_idx = int(len(posts_df) * 0.80)
    train_df  = posts_df.iloc[:split_idx]
    X_train   = train_df[FEATURE_COLS_MODEL]

    engagement_pipeline = Pipeline([
        ("prep", build_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1,
        )),
    ])

    referrals_pipeline = Pipeline([
        ("prep", build_preprocessor()),
        ("model", GradientBoostingRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.05, min_samples_leaf=5, random_state=42,
        )),
    ])

    value_pipeline = Pipeline([
        ("prep", build_preprocessor()),
        ("model", GradientBoostingRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.05, min_samples_leaf=5, random_state=42,
        )),
    ])

    engagement_pipeline.fit(X_train, train_df["engagement_rate"])
    referrals_pipeline.fit(X_train, train_df["donation_referrals"])
    value_pipeline.fit(X_train, train_df["estimated_donation_value_php"])

    model_version = datetime.now().strftime("%Y%m%d_%H%M")
    artifact = {
        "engagement_model": engagement_pipeline,
        "referrals_model":  referrals_pipeline,
        "value_model":      value_pipeline,
        "model_version":    model_version,
        "platform_medians": posts_df.groupby("platform")["donation_referrals"].median().to_dict(),
    }

    model_path = "/tmp/social_media_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    upload_artifact(model_path, f"social_media/versions/social_media_model_{model_version}.pkl")
    upload_artifact(model_path, "social_media/social_media_model.pkl")

    print(f"Social media models trained | version {model_version}")


if __name__ == "__main__":
    train()
