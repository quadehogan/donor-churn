"""
Reintegration Readiness — Standalone Retraining Script
=======================================================
Retrains Model B (reintegration readiness) only, without touching the
regression risk model. Use when reintegration labels have shifted but
risk labels are stable.

For a full retrain of both models, use train_risk.py instead.
"""

import json
import pickle
import sys
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from db.connection import get_engine
from storage.blob_client import load_json_from_blob, upload_artifact
from pipelines.resident_risk.features import (
    CAT_COLS,
    NUM_COLS,
    build_labels,
    build_preprocessor,
    engineer_features,
    load_and_merge,
)

AUC_THRESHOLD    = 0.70
RECALL_THRESHOLD = 0.65


def main():
    engine = get_engine()

    df = load_and_merge(engine)
    df = build_labels(df)
    df = engineer_features(df)

    X      = df[NUM_COLS + CAT_COLS]
    y_reint = df['reintegration_ready']

    n_splits = 3 if len(X) < 200 else 5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    param_dist = {
        'model__n_estimators':     [50, 100, 200],
        'model__max_depth':        [3, 4, 5, 6, None],
        'model__min_samples_leaf': [5, 10, 15, 20],
        'model__max_features':     ['sqrt', 'log2'],
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reint, test_size=0.2, random_state=42, stratify=y_reint
    )

    pipe = Pipeline([
        ('prep',  build_preprocessor()),
        ('model', RandomForestClassifier(class_weight='balanced', random_state=42)),
    ])
    search = RandomizedSearchCV(pipe, param_dist, n_iter=20, cv=cv,
                                scoring='roc_auc', random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    pipeline = search.best_estimator_

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)
    report  = classification_report(y_test, y_pred, output_dict=True)
    recall  = report.get('1', {}).get('recall', 0.0)

    print(f"Reintegration — AUC: {auc:.4f} | Recall(1): {recall:.4f}")
    print(classification_report(y_test, y_pred))

    if auc < AUC_THRESHOLD:
        print(f"ERROR: AUC {auc:.4f} below threshold {AUC_THRESHOLD}", file=sys.stderr)
        sys.exit(1)
    if recall < RECALL_THRESHOLD:
        print(f"ERROR: Recall {recall:.4f} below threshold {RECALL_THRESHOLD}", file=sys.stderr)
        sys.exit(1)

    # Preserve model_version from the existing risk metadata so both models stay in sync
    try:
        existing_meta = load_json_from_blob('resident_risk/risk_metadata.json')
        model_version = existing_meta['model_version']
    except Exception:
        model_version = datetime.now().strftime('%Y%m%d_%H%M')

    with open('/tmp/reint_model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    versioned = f'reint_model_{model_version}.pkl'
    upload_artifact('/tmp/reint_model.pkl', 'resident_risk/reint_model.pkl')
    upload_artifact('/tmp/reint_model.pkl', f'resident_risk/versions/{versioned}')

    print(f"Reintegration model saved. Version: {model_version}")


if __name__ == '__main__':
    main()
