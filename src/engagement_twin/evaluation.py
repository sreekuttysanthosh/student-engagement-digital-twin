
"""Synthetic-data fidelity metrics and prediction-input validation."""

import logging
import math
from itertools import combinations
from numbers import Real

import numpy as np
import pandas as pd
from sdmetrics.column_pairs import CorrelationSimilarity
from sdmetrics.single_column import MissingValueSimilarity
from sdv.evaluation import evaluate_quality

log = logging.getLogger(__name__)

TABLE = "students"
MIN_PAIR_OBSERVATIONS = 3


def _validate_columns(
    df: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    """Validate that a DataFrame contains the required unique columns."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame.")

    if df.columns.duplicated().any():
        raise ValueError(f"{label} contains duplicate column names.")

    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(f"{label}: missing columns {missing}")


def _validate_comparable_frames(
    real: pd.DataFrame,
    fake: pd.DataFrame,
    columns: list[str],
) -> None:
    """Validate real and synthetic inputs for fidelity comparisons."""

    if not columns:
        raise ValueError("At least one column must be provided.")

    if len(columns) != len(set(columns)):
        raise ValueError("Column names must be unique.")

    _validate_columns(real, columns, "Real data")
    _validate_columns(fake, columns, "Synthetic data")

    for col in columns:
        if not pd.api.types.is_numeric_dtype(real[col]):
            raise TypeError(f"Real data column '{col}' must be numeric.")

        if not pd.api.types.is_numeric_dtype(fake[col]):
            raise TypeError(
                f"Synthetic data column '{col}' must be numeric."
            )


def _validate_score(score: float, label: str) -> float:
    """Ensure a similarity or quality score is finite and within 0..1."""

    try:
        value = float(score)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not a valid numeric score.") from exc

    if not math.isfinite(value):
        raise ValueError(f"{label} is not finite.")

    if not 0 <= value <= 1:
        raise ValueError(f"{label} must be between 0 and 1.")

    return value


def pair_trends(
    real: pd.DataFrame,
    fake: pd.DataFrame,
    features: list[str],
) -> tuple[float, dict]:
    """Mean CorrelationSimilarity over all feature pairs.

    Each pair uses pairwise-complete observations. Both real and synthetic
    datasets must contain at least MIN_PAIR_OBSERVATIONS complete rows.
    """

    if len(features) < 2:
        raise ValueError("At least two features are required for pair trends.")

    _validate_comparable_frames(real, fake, features)

    scores = {}

    for a, b in combinations(features, 2):
        pair = [a, b]

        real_pair = real[pair].dropna()
        fake_pair = fake[pair].dropna()

        if len(real_pair) < MIN_PAIR_OBSERVATIONS:
            raise ValueError(
                f"Real data has fewer than {MIN_PAIR_OBSERVATIONS} "
                f"complete observations for pair {a}|{b}."
            )

        if len(fake_pair) < MIN_PAIR_OBSERVATIONS:
            raise ValueError(
                f"Synthetic data has fewer than {MIN_PAIR_OBSERVATIONS} "
                f"complete observations for pair {a}|{b}."
            )

        raw_score = CorrelationSimilarity.compute(
            real_pair,
            fake_pair,
        )

        scores[f"{a}|{b}"] = _validate_score(
            raw_score,
            f"Correlation similarity for {a}|{b}",
        )

    mean_score = float(np.mean(list(scores.values())))

    return mean_score, scores


def column_shapes(
    real: pd.DataFrame,
    fake: pd.DataFrame,
    metadata,
) -> float:
    """Return the Column Shapes score from the SDV quality report."""

    columns = list(real.columns)

    if not columns:
        raise ValueError("Real data must contain at least one column.")

    _validate_comparable_frames(real, fake, columns)

    if list(fake.columns) != columns:
        raise ValueError(
            "Real and synthetic data must have the same column order."
        )

    report = evaluate_quality(
        {TABLE: real},
        {TABLE: fake},
        metadata,
        verbose=False,
    )

    props = report.get_properties()

    if not isinstance(props, pd.DataFrame):
        raise TypeError("SDV quality report properties must be a DataFrame.")

    if not {"Property", "Score"}.issubset(props.columns):
        raise ValueError(
            "SDV quality report does not contain Property and Score columns."
        )

    matching = props.loc[
        props["Property"] == "Column Shapes",
        "Score",
    ]

    if len(matching) != 1:
        raise ValueError(
            "Expected exactly one Column Shapes score in the SDV report."
        )

    return _validate_score(
        matching.iloc[0],
        "Column Shapes score",
    )


def missing_similarity(
    real: pd.DataFrame,
    fake: pd.DataFrame,
    features: list[str],
) -> dict:
    """Calculate MissingValueSimilarity for each feature."""

    if not features:
        raise ValueError("At least one feature is required.")

    if len(features) != len(set(features)):
        raise ValueError("Feature names must be unique.")

    _validate_columns(real, features, "Real data")
    _validate_columns(fake, features, "Synthetic data")

    scores = {}

    for col in features:
        raw_score = MissingValueSimilarity.compute(
            real[col],
            fake[col],
        )

        scores[col] = _validate_score(
            raw_score,
            f"Missing-value similarity for '{col}'",
        )

    return scores


def overall_quality(shapes: float, trends: float) -> float:
    """Return the custom mean of Column Shapes and Pair Trends scores."""

    shapes = _validate_score(shapes, "Column Shapes score")
    trends = _validate_score(trends, "Pair Trends score")

    return float((shapes + trends) / 2)


def class_metrics(confusion: list[list[int]]) -> dict:
    """Calculate class-1 metrics from a [[TN, FP], [FN, TP]] matrix."""

    try:
        matrix = np.asarray(confusion, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Confusion matrix must contain numeric counts."
        ) from exc

    if matrix.shape != (2, 2):
        raise ValueError(
            "Confusion matrix must have exactly 2 rows and 2 columns."
        )

    if not np.isfinite(matrix).all():
        raise ValueError("Confusion matrix must contain finite values.")

    if (matrix < 0).any():
        raise ValueError("Confusion matrix counts cannot be negative.")

    if not np.equal(matrix, np.floor(matrix)).all():
        raise ValueError("Confusion matrix counts must be integers.")

    (tn, fp), (fn, tp) = matrix.astype(int)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def validate_prediction_input(
    values: dict,
    features: list[str],
    max_value: dict,
) -> list[float]:
    """Validate prediction inputs and return values in fixed feature order.

    NaN and None are allowed to preserve native missing-value handling.
    Negative values, values above their maximum, and invalid types are rejected.
    """

    if not isinstance(values, dict):
        raise TypeError("Prediction values must be provided as a dictionary.")

    if not features:
        raise ValueError("At least one feature must be specified.")

    if len(features) != len(set(features)):
        raise ValueError("Feature names must be unique.")

    if not isinstance(max_value, dict):
        raise TypeError("max_value must be a dictionary.")

    missing_limits = [
        feature for feature in features
        if feature not in max_value
    ]

    if missing_limits:
        raise ValueError(
            f"Missing maximum values for features: {missing_limits}"
        )

    unknown = set(values) - set(features)

    if unknown:
        raise ValueError(f"Unknown features: {sorted(unknown)}")

    missing = [
        feature for feature in features
        if feature not in values
    ]

    if missing:
        raise ValueError(f"Missing features: {missing}")

    out = []

    for name in features:
        limit = max_value[name]

        if (
            isinstance(limit, bool)
            or not isinstance(limit, Real)
            or not math.isfinite(float(limit))
            or limit < 0
        ):
            raise ValueError(
                f"Maximum value for '{name}' must be a finite "
                "non-negative number."
            )

        value = values[name]

        if value is None:
            out.append(float("nan"))
            continue

        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(
                f"{name} must be a numeric value, None, or NaN."
            )

        value = float(value)

        if math.isnan(value):
            out.append(float("nan"))
            continue

        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite or NaN.")

        if value < 0 or value > limit:
            raise ValueError(
                f"{name}={value} outside 0..{limit}"
            )

        out.append(value)

    return out
