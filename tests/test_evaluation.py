
import math

import numpy as np
import pandas as pd
import pytest

from engagement_twin.evaluation import (
    class_metrics,
    missing_similarity,
    overall_quality,
    pair_trends,
    validate_prediction_input,
)

FEATURES = ["nevents", "ndays_act", "ncontent", "nforum_posts"]

MAX_VALUE = {
    "nevents": 50000,
    "ndays_act": 500,
    "ncontent": 100,
    "nforum_posts": 5000,
}


# ---------- Class metrics ----------

def test_class_metrics_known_values():
    metrics = class_metrics([[90, 10], [20, 80]])

    assert metrics["precision"] == pytest.approx(80 / 90)
    assert metrics["recall"] == pytest.approx(0.8)

    expected_f1 = (
        2 * (80 / 90) * 0.8 / ((80 / 90) + 0.8)
    )

    assert metrics["f1"] == pytest.approx(expected_f1)


def test_class_metrics_no_positive_predictions():
    result = class_metrics([[10, 0], [5, 0]])

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


@pytest.mark.parametrize(
    "confusion",
    [
        [[1, 2, 3], [4, 5, 6]],
        [[1, 2], [3]],
        [[1, -2], [3, 4]],
        [[1.5, 2], [3, 4]],
        [[float("nan"), 2], [3, 4]],
        [[float("inf"), 2], [3, 4]],
    ],
)
def test_class_metrics_rejects_invalid_confusion_matrix(confusion):
    with pytest.raises(ValueError):
        class_metrics(confusion)


# ---------- Overall quality ----------

def test_overall_quality_is_mean():
    assert overall_quality(0.9, 0.8) == pytest.approx(0.85)


@pytest.mark.parametrize(
    "shapes,trends",
    [
        (float("nan"), 0.8),
        (0.8, float("inf")),
        (-0.1, 0.8),
        (0.8, 1.1),
    ],
)
def test_overall_quality_rejects_invalid_scores(shapes, trends):
    with pytest.raises(ValueError):
        overall_quality(shapes, trends)


# ---------- Pair trends ----------

def test_pair_trends_identical_data_scores_one():
    rng = np.random.default_rng(0)

    df = pd.DataFrame(
        rng.normal(size=(300, 4)),
        columns=FEATURES,
    )

    score, per_pair = pair_trends(df, df.copy(), FEATURES)

    assert score == pytest.approx(1.0)
    assert len(per_pair) == 6


def test_pair_trends_rejects_too_few_complete_observations():
    df = pd.DataFrame(
        np.random.default_rng(42).normal(size=(10, 4)),
        columns=FEATURES,
    )

    real = df.copy()
    fake = df.copy()

    real.loc[2:, "nevents"] = np.nan
    fake.loc[2:, "nevents"] = np.nan

    with pytest.raises(ValueError, match="fewer than 3"):
        pair_trends(real, fake, FEATURES)


def test_pair_trends_rejects_missing_feature():
    df = pd.DataFrame(
        np.random.default_rng(42).normal(size=(20, 4)),
        columns=FEATURES,
    )

    real = df.drop(columns=["ncontent"])

    with pytest.raises(ValueError, match="missing columns"):
        pair_trends(real, df, FEATURES)


# ---------- Missing-value similarity ----------

def test_missing_similarity_identical_data():
    df = pd.DataFrame({
        "nevents": [10.0, np.nan, 30.0, 40.0],
        "ndays_act": [1.0, 2.0, np.nan, 4.0],
    })

    result = missing_similarity(
        df,
        df.copy(),
        ["nevents", "ndays_act"],
    )

    assert result["nevents"] == pytest.approx(1.0)
    assert result["ndays_act"] == pytest.approx(1.0)


# ---------- Prediction input validation ----------

def test_validate_accepts_nan_and_keeps_order():
    result = validate_prediction_input(
        {
            "nforum_posts": float("nan"),
            "nevents": 5,
            "ncontent": 80,
            "ndays_act": 3,
        },
        FEATURES,
        MAX_VALUE,
    )

    assert result[:3] == [5.0, 3.0, 80.0]
    assert math.isnan(result[3])


@pytest.mark.parametrize(
    "bad",
    [
        {
            "nevents": -1,
            "ndays_act": 1,
            "ncontent": 1,
            "nforum_posts": 1,
        },
        {
            "nevents": 1,
            "ndays_act": 1,
            "ncontent": 101,
            "nforum_posts": 1,
        },
        {
            "nevents": 1,
            "ndays_act": 1,
            "ncontent": 1,
        },
        {
            "nevents": 1,
            "ndays_act": 1,
            "ncontent": 1,
            "nforum_posts": 1,
            "extra": 1,
        },
    ],
)
def test_validate_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        validate_prediction_input(bad, FEATURES, MAX_VALUE)


def test_validate_prediction_rejects_string_value():
    values = {
        "nevents": "five",
        "ndays_act": 1,
        "ncontent": 80,
        "nforum_posts": 2,
    }

    with pytest.raises(ValueError, match="numeric value"):
        validate_prediction_input(values, FEATURES, MAX_VALUE)


def test_validate_prediction_rejects_infinite_value():
    values = {
        "nevents": float("inf"),
        "ndays_act": 1,
        "ncontent": 80,
        "nforum_posts": 2,
    }

    with pytest.raises(ValueError, match="finite"):
        validate_prediction_input(values, FEATURES, MAX_VALUE)


def test_validate_prediction_rejects_boolean_value():
    values = {
        "nevents": True,
        "ndays_act": 1,
        "ncontent": 80,
        "nforum_posts": 2,
    }

    with pytest.raises(ValueError, match="numeric value"):
        validate_prediction_input(values, FEATURES, MAX_VALUE)


def test_validate_prediction_rejects_missing_maximum():
    incomplete_limits = {
        "nevents": 50000,
        "ndays_act": 500,
        "ncontent": 100,
    }

    values = {
        "nevents": 5,
        "ndays_act": 3,
        "ncontent": 80,
        "nforum_posts": 2,
    }

    with pytest.raises(ValueError, match="Missing maximum values"):
        validate_prediction_input(
            values,
            FEATURES,
            incomplete_limits,
        )
