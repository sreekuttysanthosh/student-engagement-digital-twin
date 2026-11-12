
import numpy as np
import pandas as pd
import pytest

from engagement_twin.classifier import (
    build_model,
    check_columns,
    evaluate,
    train,
)
from engagement_twin.config import load_config

FEATURES = ["nevents", "ndays_act", "ncontent", "nforum_posts"]
TARGET = "Engagement_Level"


def _toy(n=200, seed=0):
    rng = np.random.default_rng(seed)

    df = pd.DataFrame(
        rng.integers(0, 50, size=(n, 4)),
        columns=FEATURES,
    ).astype(float)

    df[TARGET] = (df["ncontent"] > 25).astype(int)
    df.loc[::10, "nforum_posts"] = np.nan

    return df


def _model():
    cfg = load_config()
    return build_model(cfg["xgboost"], cfg["seed"])


def test_check_columns_fixes_order():
    df = _toy()[[TARGET, *reversed(FEATURES)]]

    assert list(
        check_columns(df, FEATURES, TARGET, "toy").columns
    ) == FEATURES + [TARGET]


def test_check_columns_rejects_missing_column():
    with pytest.raises(ValueError):
        check_columns(
            _toy().drop(columns=["ncontent"]),
            FEATURES,
            TARGET,
            "toy",
        )


def test_train_and_evaluate_with_nans():
    model = train(_model(), _toy(), FEATURES, TARGET)

    result = evaluate(model, _toy(seed=1), FEATURES, TARGET)

    assert sum(map(sum, result["confusion_matrix"])) == 200
    assert result["weighted"]["f1"] > 0.9


def test_train_rejects_missing_target():
    df = _toy()
    df.loc[0, TARGET] = np.nan

    with pytest.raises(ValueError, match="target contains missing"):
        train(_model(), df, FEATURES, TARGET)


def test_train_rejects_invalid_target_labels():
    df = _toy()
    df[TARGET] = df[TARGET].astype(int)
    df.loc[0, TARGET] = 2

    with pytest.raises(ValueError, match="only 0 and 1"):
        train(_model(), df, FEATURES, TARGET)


def test_train_rejects_single_class():
    df = _toy()
    df[TARGET] = 0

    with pytest.raises(ValueError, match="both classes"):
        train(_model(), df, FEATURES, TARGET)


def test_evaluate_rejects_missing_target():
    model = train(_model(), _toy(), FEATURES, TARGET)

    test_df = _toy(seed=1)
    test_df.loc[0, TARGET] = np.nan

    with pytest.raises(ValueError, match="target contains missing"):
        evaluate(model, test_df, FEATURES, TARGET)


def test_evaluate_rejects_invalid_target_labels():
    model = train(_model(), _toy(), FEATURES, TARGET)

    test_df = _toy(seed=1)
    test_df[TARGET] = test_df[TARGET].astype(int)
    test_df.loc[0, TARGET] = 2

    with pytest.raises(ValueError, match="only 0 and 1"):
        evaluate(model, test_df, FEATURES, TARGET)


def test_evaluate_rejects_single_class():
    model = train(_model(), _toy(), FEATURES, TARGET)

    test_df = _toy(seed=1)
    test_df[TARGET] = 0

    with pytest.raises(ValueError, match="both classes"):
        evaluate(model, test_df, FEATURES, TARGET)
