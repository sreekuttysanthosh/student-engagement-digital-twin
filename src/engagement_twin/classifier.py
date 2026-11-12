
"""XGBoost training and evaluation on a real test set."""

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from xgboost import XGBClassifier


def check_columns(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    label: str,
) -> pd.DataFrame:
    """Validate required columns and return them in a fixed order."""

    expected = features + [target]

    missing = [col for col in expected if col not in df.columns]

    if missing:
        raise ValueError(f"{label}: missing columns {missing}")

    return df[expected].copy()


def validate_target(y: pd.Series, label: str) -> None:
    """Validate that the target contains only binary, non-missing labels."""

    if y.isna().any():
        raise ValueError(f"{label}: target contains missing values.")

    if not y.isin([0, 1]).all():
        raise ValueError(f"{label}: target must contain only 0 and 1.")

    if y.nunique() < 2:
        raise ValueError(f"{label}: target must contain both classes 0 and 1.")


def build_model(
    xgb_params: dict,
    seed: int,
) -> XGBClassifier:
    """Create an XGBoost classifier with fixed, pre-set parameters."""

    return XGBClassifier(**xgb_params, random_state=seed)


def train(
    model: XGBClassifier,
    train_df: pd.DataFrame,
    features: list[str],
    target: str,
) -> XGBClassifier:
    """Fit on training data while preserving native NaN handling."""

    train_data = check_columns(
        train_df,
        features,
        target,
        "Training data",
    )

    validate_target(train_data[target], "Training data")

    model.fit(
        train_data[features],
        train_data[target].astype(int),
    )

    return model


def evaluate(
    model: XGBClassifier,
    test: pd.DataFrame,
    features: list[str],
    target: str,
) -> dict:
    """Evaluate on real test data, including its genuine missing values."""

    test_data = check_columns(
        test,
        features,
        target,
        "Test data",
    )

    validate_target(test_data[target], "Test data")

    y_true = test_data[target].astype(int)
    y_pred = model.predict(test_data[features])

    p, r, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    per_class = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        output_dict=True,
        zero_division=0,
    )

    return {
        "weighted": {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
        },
        "per_class": {
            label: per_class[label]
            for label in ("0", "1")
        },
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).tolist(),
    }
