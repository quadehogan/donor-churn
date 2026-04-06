"""
Training job — run by Azure Container Apps Job on a monthly cron schedule.
Can also be run locally: python pipelines/churn/train_churn.py
"""

import json
import os
import pickle
import sys
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from db.connection import get_engine
from pipelines.churn.features import (
    CAT_COLS,
    NUM_COLS,
    TRAINING_QUERY,
    engineer_features,
)
from storage.blob_client import upload_artifact

load_dotenv()

os.makedirs("/tmp", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)


def build_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUM_COLS),
        ("cat", categorical_pipe, CAT_COLS),
    ])


def main():
    engine = get_engine()

    print("Loading training data from Supabase...")
    df_raw = pd.read_sql(TRAINING_QUERY, engine)
    print(f"Raw shape: {df_raw.shape}")
    print(f"Churn rate: {df_raw['churned'].mean():.2%}")

    df = engineer_features(df_raw, engine=engine)

    y = df["churned"]
    X = df.drop(columns=["churned", "supporter_id"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print("Training...")
    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)
    report  = classification_report(y_test, y_pred, output_dict=True)
    recall  = report["1"]["recall"]

    print(f"\nROC-AUC : {auc:.4f}  (threshold ≥ 0.75)")
    print(f"Recall  : {recall:.4f}  (threshold ≥ 0.70)")
    print(classification_report(y_test, y_pred))

    # Hard gate — do not save if thresholds not met
    if auc < 0.75:
        print(f"ABORT: AUC {auc:.4f} below threshold. Model not saved.")
        sys.exit(1)
    if recall < 0.70:
        print(f"ABORT: Recall {recall:.4f} below threshold. Model not saved.")
        sys.exit(1)

    model_version = datetime.now().strftime("%Y%m%d_%H%M")

    metadata = {
        "trained_at":    datetime.now().isoformat(),
        "model_version": model_version,
        "n_samples":     len(df),
        "churn_rate":    float(y.mean()),
        "features":      X.columns.tolist(),
        "model_class":   type(pipeline.named_steps["model"]).__name__,
        "hyperparams":   pipeline.named_steps["model"].get_params(),
    }
    metrics = {
        "roc_auc":      round(auc, 4),
        "recall_churn": round(recall, 4),
    }

    model_path    = "/tmp/churn_model.pkl"
    metadata_path = "/tmp/churn_metadata.json"
    metrics_path  = "/tmp/churn_metrics.json"

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Versioned copies + overwrite latest pointer
    upload_artifact(model_path,    f"churn/versions/churn_model_{model_version}.pkl")
    upload_artifact(metadata_path, f"churn/versions/churn_metadata_{model_version}.json")
    upload_artifact(metrics_path,  f"churn/versions/churn_metrics_{model_version}.json")
    upload_artifact(model_path,    "churn/churn_model.pkl")
    upload_artifact(metadata_path, "churn/churn_metadata.json")
    upload_artifact(metrics_path,  "churn/churn_metrics.json")

    print(f"\nDone — version: {model_version} | AUC: {auc:.4f} | Recall: {recall:.4f}")


if __name__ == "__main__":
    main()
