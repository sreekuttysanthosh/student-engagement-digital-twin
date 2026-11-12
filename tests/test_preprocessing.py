
import numpy as np
import pandas as pd
import pytest

from engagement_twin.preprocessing import (
    build_generator_inputs,
    missingness_flags,
    restore_from_flags,
    restore_random,
)

FEATURES = ["nevents", "ndays_act", "ncontent", "nforum_posts"]
TARGET = "Engagement_Level"


def _train(n=200, seed=0):
    rng = np.random.default_rng(seed)

    df = pd.DataFrame(
        rng.integers(1, 100, size=(n, 4)),
        columns=FEATURES,
    ).astype(float)

    df.loc[::5, "nevents"] = np.nan
    df.loc[::3, "nforum_posts"] = np.nan
    df[TARGET] = (rng.random(n) > 0.8).astype(int)

    return df


def test_inputs_have_no_nans_and_integer_features():
    tables, _ = build_generator_inputs(_train(), FEATURES, TARGET, 5)

    for table in tables.values():
        assert not table.isna().any().any()

    assert all(
        pd.api.types.is_integer_dtype(tables["random"][c])
        for c in FEATURES
    )


def test_inputs_columns_and_no_flag_for_complete_column():
    tables, _ = build_generator_inputs(_train(), FEATURES, TARGET, 5)

    assert list(tables["random"].columns) == FEATURES
    assert TARGET not in tables["from_column"].columns

    assert list(tables["from_column_target"].columns) == [
        *FEATURES,
        "nevents_missing",
        "nforum_posts_missing",
        TARGET,
    ]


def test_flags_only_for_columns_with_gaps():
    flags = missingness_flags(_train(), FEATURES)

    assert list(flags.columns) == [
        "nevents_missing",
        "nforum_posts_missing",
    ]


def test_restore_random_rates_label_and_reproducibility():
    fake = pd.DataFrame(
        np.ones((5000, 4)),
        columns=FEATURES,
    )
    fake[TARGET] = 1

    rates = pd.Series({
        "nevents": 0.2,
        "ndays_act": 0.1,
        "ncontent": 0.0,
        "nforum_posts": 0.7,
    })

    a = restore_random(fake, rates, FEATURES, seed=42, target=TARGET)
    b = restore_random(fake, rates, FEATURES, seed=42, target=TARGET)

    assert a.equals(b)
    assert a[TARGET].equals(fake[TARGET])
    assert a["ncontent"].isna().sum() == 0
    assert abs(a["nforum_posts"].isna().mean() - 0.7) < 0.03


def test_restore_from_flags_positions_and_drops_flags():
    df = pd.DataFrame({
        c: [1.0, 2.0, 3.0] for c in FEATURES
    })

    df["nevents_missing"] = [0, 1, 0]
    df[TARGET] = [0, 1, 0]

    out = restore_from_flags(
        df,
        FEATURES,
        flag_features=["nevents"],
        target=TARGET,
    )

    assert out["nevents"].isna().tolist() == [False, True, False]
    assert out["ncontent"].isna().sum() == 0
    assert list(out.columns) == [*FEATURES, TARGET]


def test_restore_from_flags_rejects_bad_flags():
    df = pd.DataFrame({
        c: [1.0, 2.0] for c in FEATURES
    })

    df["nevents_missing"] = [0, 2]

    with pytest.raises(ValueError, match="must contain only 0/1"):
        restore_from_flags(
            df,
            FEATURES,
            flag_features=["nevents"],
        )


def test_restore_from_flags_rejects_missing_expected_flag():
    df = pd.DataFrame({
        c: [1.0, 2.0] for c in FEATURES
    })

    df["nevents_missing"] = [0, 1]

    with pytest.raises(ValueError, match="Missing expected flag"):
        restore_from_flags(
            df,
            FEATURES,
            flag_features=["nevents", "nforum_posts"],
        )


def test_restore_random_rejects_missing_rate():
    fake = pd.DataFrame({
        c: [1.0, 2.0] for c in FEATURES
    })

    rates = pd.Series({"nevents": 0.2})

    with pytest.raises(ValueError, match="Missing missingness rates"):
        restore_random(fake, rates, FEATURES, seed=42)


def test_restore_random_rejects_invalid_rates():
    fake = pd.DataFrame({
        c: [1.0, 2.0] for c in FEATURES
    })

    rates = pd.Series({
        "nevents": 0.2,
        "ndays_act": 1.2,
        "ncontent": 0.0,
        "nforum_posts": 0.7,
    })

    with pytest.raises(ValueError, match="between 0 and 1"):
        restore_random(fake, rates, FEATURES, seed=42)


def test_restore_random_rejects_nan_rate():
    fake = pd.DataFrame({
        c: [1.0, 2.0] for c in FEATURES
    })

    rates = pd.Series({
        "nevents": 0.2,
        "ndays_act": 0.1,
        "ncontent": np.nan,
        "nforum_posts": 0.7,
    })

    with pytest.raises(ValueError, match="cannot contain NaN"):
        restore_random(fake, rates, FEATURES, seed=42)
