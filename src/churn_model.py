import os
import joblib
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .preprocess import load_data, create_features


def _prepare_xy(df: pd.DataFrame, label_column: str = "churn") -> Tuple[np.ndarray, np.ndarray, list]:
    if label_column not in df.columns:
        # Try common alternatives
        for alt in ["is_churn", "label", "target"]:
            if alt in df.columns:
                label_column = alt
                break
    y = df[label_column].values if label_column in df.columns else None

    # Prefer numeric columns only; drop identifiers
    candidate_cols = [c for c in df.columns if c not in {label_column, "player_id"}]
    X_df = df[candidate_cols].copy()

    # Coerce all features to numeric; non-convertible become NaN, then fill
    for c in X_df.columns:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
    X_df = X_df.fillna(0.0)

    feature_cols = list(X_df.columns)
    X = X_df.values.astype(np.float32)

    # Clean labels if needed
    if y is not None:
        y = pd.Series(y).fillna(0).astype(int).values

    return X, y, feature_cols


def train_xgb_and_save(data_dir: str = "data", model_path: str = "models/churn_xgb.pkl") -> None:
    dfs = load_data(data_dir)
    train_df, dev_df, _ = create_features(dfs)

    # Focus training on train_df; reserve a validation split
    X, y, feature_cols = _prepare_xy(train_df)
    if y is None:
        raise ValueError("Training labels not found in train.csv. Expected 'churn' or similar column.")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(X_train, y_train)

    # Metrics
    valid_pred = model.predict(X_valid)
    valid_proba = model.predict_proba(X_valid)[:, 1]
    acc = accuracy_score(y_valid, valid_pred)
    auc = roc_auc_score(y_valid, valid_proba)

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation ROC-AUC: {auc:.4f}")

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"model": model, "features": feature_cols}, model_path)
    print(f"Saved XGBoost model to: {model_path}")


if __name__ == "__main__":
    train_xgb_and_save()


