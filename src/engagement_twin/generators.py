
"""Load saved SDV synthesizers and sample from them (no training at import time)."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sdv.sampling import Condition
from sdv.utils import load_synthesizer

log = logging.getLogger(__name__)

TABLE = "students"


def unwrap(result: pd.DataFrame | dict) -> pd.DataFrame:
    """Return a DataFrame from SDV output using the expected table name."""

    if isinstance(result, pd.DataFrame):
        return result

    if isinstance(result, dict):
        if TABLE not in result:
            raise ValueError(
                f"Expected table '{TABLE}' in SDV output. "
                f"Received keys: {list(result.keys())}"
            )

        dataframe = result[TABLE]

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Expected table '{TABLE}' to contain a pandas DataFrame, "
                f"got {type(dataframe).__name__}"
            )

        return dataframe

    raise TypeError(
        f"Expected a pandas DataFrame or dictionary, "
        f"got {type(result).__name__}"
    )


def load_generator(path: str | Path):
    """Load an existing saved SDV synthesizer."""

    model_path = Path(path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved generator model not found: {model_path}"
        )

    if not model_path.is_file():
        raise ValueError(
            f"Generator model path is not a file: {model_path}"
        )

    return load_synthesizer(str(model_path))


def _validate_positive_integer(value: int, name: str) -> None:
    """Ensure a requested sample count is a positive integer."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")


def sample_unconditional(
    synth,
    n_rows: int,
    seed: int,
) -> pd.DataFrame:
    """Generate an unconditional synthetic sample with fixed seeds."""

    _validate_positive_integer(n_rows, "n_rows")

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer.")

    np.random.seed(seed)
    torch.manual_seed(seed)

    result = unwrap(synth.sample(num_rows=n_rows))

    if len(result) != n_rows:
        raise ValueError(
            f"Expected {n_rows} synthetic rows, received {len(result)}."
        )

    return result


def sample_balanced(
    synth,
    target: str,
    n_per_class: int,
    seed: int,
) -> pd.DataFrame:
    """Conditionally sample equal numbers of class 0 and 1, then shuffle."""

    _validate_positive_integer(n_per_class, "n_per_class")

    if not isinstance(target, str) or not target.strip():
        raise ValueError("target must be a non-empty column name.")

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer.")

    parts = []

    for label in (0, 1):
        np.random.seed(seed)
        torch.manual_seed(seed)

        condition = Condition(
            column_values={target: label},
            num_rows=n_per_class,
        )

        generated = unwrap(
            synth.sample_from_conditions(conditions=[condition])
        )

        if target not in generated.columns:
            raise ValueError(
                f"Generated data is missing target column '{target}'."
            )

        if len(generated) != n_per_class:
            raise ValueError(
                f"Expected {n_per_class} rows for class {label}, "
                f"received {len(generated)}."
            )

        if not generated[target].eq(label).all():
            raise ValueError(
                f"Conditional sampling returned unexpected labels "
                f"for class {label}."
            )

        parts.append(generated)

    both = pd.concat(parts, ignore_index=True)

    # Shuffle the combined balanced dataset
    both = both.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    # Final class-balance validation
    counts = both[target].value_counts().to_dict()

    expected = {0: n_per_class, 1: n_per_class}

    if counts != expected:
        raise ValueError(
            f"Expected balanced class counts {expected}, "
            f"received {counts}."
        )

    return both
