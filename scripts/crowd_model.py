"""
Training and prediction utilities for SmartHeritage crowd intelligence.

The app uses a RandomForestRegressor to predict visitor counts from the
available tourism data, then converts predicted visitors into crowd levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from scripts.recommendations import (
    HIGH_CROWD_THRESHOLD,
    MEDIUM_CROWD_THRESHOLD,
    build_recommendation_dataset,
    load_tourism_data,
)


NUMERIC_FEATURES = [
    "domestic_2019",
    "foreign_2019",
    "domestic_2020",
    "foreign_2020",
    "domestic_growth_pct",
    "foreign_growth_pct",
]
CATEGORICAL_FEATURES = ["monument", "circle"]
TARGET_VISITORS = "total_visitors"


@dataclass
class CrowdModelBundle:
    prepared_df: pd.DataFrame
    regressor: Pipeline
    metrics: dict[str, float]
    feature_importance: pd.DataFrame


def classify_crowd(visitors: float) -> str:
    """Convert predicted annual visitors into crowd categories."""
    if visitors > HIGH_CROWD_THRESHOLD:
        return "High"
    if visitors > MEDIUM_CROWD_THRESHOLD:
        return "Medium"
    return "Low"


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def _prepare_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = build_recommendation_dataset(df).copy()
    for column in NUMERIC_FEATURES:
        if column not in prepared.columns:
            prepared[column] = 0.0
    return prepared.dropna(subset=["monument", "circle"]).reset_index(drop=True)


def _aggregate_feature_importance(model: Pipeline) -> pd.DataFrame:
    preprocessor: ColumnTransformer = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    raw_names = preprocessor.get_feature_names_out()
    raw_importance = estimator.feature_importances_

    grouped: dict[str, float] = {}
    for name, score in zip(raw_names, raw_importance, strict=True):
        cleaned_name = name.split("__", 1)[-1]
        if cleaned_name.startswith("monument_"):
            cleaned_name = "monument"
        if cleaned_name.startswith("circle_"):
            cleaned_name = "circle"
        grouped[cleaned_name] = grouped.get(cleaned_name, 0.0) + float(score)

    return pd.DataFrame(
        [{"feature": feature, "importance": value} for feature, value in grouped.items()]
    ).sort_values("importance", ascending=False, ignore_index=True)


def train_crowd_models(df: pd.DataFrame, random_state: int = 42) -> CrowdModelBundle:
    """
    Train a RandomForestRegressor to predict annual visitor counts.

    The model uses monument identity plus available domestic/foreign counts and
    growth signals to learn the main tourism baseline used in the dataset.
    """
    prepared = _prepare_base_dataframe(df)
    X = prepared[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y_visitors = prepared[TARGET_VISITORS]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_visitors,
        test_size=0.2,
        random_state=random_state,
    )

    regressor = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=120,
                    random_state=random_state,
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    regressor.fit(X_train, y_train)
    test_predictions = regressor.predict(X_test)
    regressor.fit(X, y_visitors)

    metrics = {
        "training_rows": float(len(prepared)),
        "visitor_mae": float(mean_absolute_error(y_test, test_predictions)),
        "visitor_r2": float(r2_score(y_test, test_predictions)),
    }

    return CrowdModelBundle(
        prepared_df=prepared,
        regressor=regressor,
        metrics=metrics,
        feature_importance=_aggregate_feature_importance(regressor),
    )


def predict_monument_with_model(
    bundle: CrowdModelBundle,
    monument_name: str,
) -> dict[str, Any]:
    """Predict baseline visitor count and crowd level for a monument."""
    prepared = bundle.prepared_df
    row = prepared[
        prepared["monument"].astype(str).str.strip().str.lower()
        == str(monument_name).strip().lower()
    ]
    if row.empty:
        raise ValueError(f"Monument '{monument_name}' was not found in the dataset.")

    actual_row = row.iloc[0]
    feature_row = row.iloc[[0]][CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    predicted_visitors = float(bundle.regressor.predict(feature_row)[0])
    predicted_crowd = classify_crowd(predicted_visitors)
    actual_visitors = float(actual_row["total_visitors"])
    actual_crowd = classify_crowd(actual_visitors)

    return {
        "monument": actual_row["monument"],
        "circle": actual_row["circle"],
        "actual_visitors": int(round(actual_visitors)),
        "actual_crowd": actual_crowd,
        "predicted_visitors": int(round(predicted_visitors)),
        "predicted_crowd": predicted_crowd,
        "best_time_to_visit": "Early morning"
        if predicted_crowd == "High"
        else ("Morning" if predicted_crowd == "Medium" else "Anytime"),
        "top_driver": str(bundle.feature_importance.iloc[0]["feature"]),
    }


def summarize_training(bundle: CrowdModelBundle) -> pd.DataFrame:
    """Expose key training metrics in a compact table for the UI."""
    return pd.DataFrame(
        [
            {"metric": "Training rows", "value": int(bundle.metrics["training_rows"])},
            {"metric": "Visitor MAE", "value": round(bundle.metrics["visitor_mae"], 2)},
            {"metric": "Visitor R2", "value": round(bundle.metrics["visitor_r2"], 4)},
            {"metric": "Forecast basis", "value": "Processed tourism baseline"},
        ]
    )


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parents[1] / "data" / "processed_tourism.csv"
    dataset = load_tourism_data(data_path)
    bundle = train_crowd_models(dataset)

    print("Crowd model training summary:")
    print(summarize_training(bundle).to_string(index=False))
    print("\nTop feature importances:")
    print(bundle.feature_importance.head(8).to_string(index=False))