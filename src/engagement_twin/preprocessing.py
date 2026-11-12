
"""Imputation, missingness flags and NaN restoration for generator inputs."""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

FLAG_SUFFIX = "_missing"


def fit_imputer(
    train: pd.DataFrame,
    features: list[str],
    k: int,
) -> KNNImputer:
    """Fit a KNN imputer on real training features only (never the target)."""
    return KNNImputer(n_neighbors=k).fit(train[features])


def impute(
    df: pd.DataFrame,
    features: list[str],
    imputer: KNNImputer,
) -> pd.DataFrame:
    """Fill missing values and round features to whole numbers."""
    filled = pd.DataFrame(
        imputer.transform(df[features]),
        columns=features,
        index=df.index,
    )
    return filled.round().astype(int)


def missingness_flags(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Create 0/1 flags only for features that actually have missing values."""
    mask = df[features].isna()
    cols = [col for col in features if mask[col].any()]
    return mask[cols].astype(int).add_suffix(FLAG_SUFFIX)


def build_generator_inputs(
    train: pd.DataFrame,
    features: list[str],
    target: str,
    k: int,
) -> tuple[dict[str, pd.DataFrame], KNNImputer]:
    """Build four generator input tables using real training data only.

    Keys:
        random
        from_column
        random_target
        from_column_target
    """
    imputer = fit_imputer(train, features, k)
    imputed = impute(train, features, imputer)
    flags = missingness_flags(train, features)
    label = train[target].astype(int)

    tables = {
        "random": imputed,
        "from_column": pd.concat([imputed, flags], axis=1),
        "random_target": pd.concat([imputed, label], axis=1),
        "from_column_target": pd.concat(
            [imputed, flags, label],
            axis=1,
        ),
    }

    return tables, imputer


def restore_random(
    df: pd.DataFrame,
    miss_rates: pd.Series,
    features: list[str],
    seed: int,
    target: str | None = None,
) -> pd.DataFrame:
    """Restore missing values randomly using rates learned from real training data."""

    # Validate feature columns
    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    # Validate missingness-rate entries
    missing_rates = [col for col in features if col not in miss_rates.index]
    if missing_rates:
        raise ValueError(
            f"Missing missingness rates for: {missing_rates}"
        )

    rates = miss_rates[features]

    if rates.isna().any():
        raise ValueError("Missingness rates cannot contain NaN values")

    if not rates.between(0, 1).all():
        raise ValueError("Missingness rates must be between 0 and 1")

    # Validate target column, if supplied
    if target is not None and target not in df.columns:
        raise ValueError(f"Missing target column: {target}")

    # Keep only requested features and target
    keep = features + ([target] if target else [])
    out = df[keep].copy()
    out[features] = out[features].astype(float)

    # Restore NaNs using reproducible random masking
    rng = np.random.default_rng(seed)

    for col in features:
        mask = rng.random(len(out)) < rates[col]
        out.loc[mask, col] = np.nan

    return out


def restore_from_flags(
    df: pd.DataFrame,
    features: list[str],
    flag_features: list[str],
    target: str | None = None,
) -> pd.DataFrame:
    """Restore NaNs using explicitly specified generated missingness flags."""

    # Validate requested feature columns
    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    # Validate the expected flag schema
    invalid_flags = [col for col in flag_features if col not in features]
    if invalid_flags:
        raise ValueError(
            f"Flag features must be included in features: {invalid_flags}"
        )

    expected_flags = [
        f"{col}{FLAG_SUFFIX}" for col in flag_features
    ]

    missing_flags = [
        col for col in expected_flags if col not in df.columns
    ]
    if missing_flags:
        raise ValueError(
            f"Missing expected flag columns: {missing_flags}"
        )

    # Validate target column, if supplied
    if target is not None and target not in df.columns:
        raise ValueError(f"Missing target column: {target}")

    keep = features + ([target] if target else [])
    out = df[keep].copy()
    out[features] = out[features].astype(float)

    # Restore missing values
    for col in flag_features:
        flag_col = f"{col}{FLAG_SUFFIX}"
        flag = df[flag_col]

        if not flag.isin([0, 1]).all():
            raise ValueError(f"{flag_col} must contain only 0/1")

        out.loc[flag == 1, col] = np.nan

    return out
