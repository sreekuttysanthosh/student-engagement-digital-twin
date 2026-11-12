import numpy as np
import pandas as pd
import pytest

from engagement_twin.data import (
    SOURCE_COL,
    assert_no_leakage,
    clean,
    make_target,
    model_table,
    split,
    validate_schema,
)

FEATURES = ["nevents", "ndays_act", "ncontent", "nforum_posts"]
TARGET = "Engagement_Level"


def _raw(n=100):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.integers(0, 50, size=(n, 4)), columns=FEATURES).astype(float)
    df["nevents"] = np.arange(n, dtype=float)  # unique, so rows can be tracked
    df[SOURCE_COL] = np.linspace(0.01, 1.0, n)
    df["userid_DI"] = range(n)
    df["course_id_DI"] = "c"
    return df


def test_target_threshold_boundary():
    df = pd.DataFrame({SOURCE_COL: [0.49, 0.5, 0.51]})
    assert make_target(df, 0.5, TARGET)[TARGET].tolist() == [0, 1, 1]


def test_validate_schema_rejects_missing_column():
    with pytest.raises(ValueError):
        validate_schema(_raw().drop(columns=["ncontent"]), FEATURES)


def test_clean_drops_missing_completion():
    df = _raw()
    df.loc[:9, SOURCE_COL] = np.nan
    assert len(clean(df, FEATURES)) == 90


def test_clean_rejects_percentage_scale():
    df = _raw()
    df[SOURCE_COL] = df[SOURCE_COL] * 100
    with pytest.raises(ValueError):
        clean(df, FEATURES)


def test_model_table_has_no_source_column_and_fixed_order():
    table = model_table(make_target(_raw(), 0.5, TARGET), FEATURES, TARGET)
    assert list(table.columns) == FEATURES + [TARGET]


def test_leakage_guard():
    with pytest.raises(ValueError):
        assert_no_leakage([*FEATURES, SOURCE_COL], TARGET)


def test_split_isolation_and_stratification():
    table = model_table(make_target(_raw(), 0.5, TARGET), FEATURES, TARGET)
    train, test = split(table, TARGET, 0.2, 42)
    assert len(train) + len(test) == len(table)
    assert set(train["nevents"]).isdisjoint(set(test["nevents"]))
    assert abs(train[TARGET].mean() - test[TARGET].mean()) < 0.05