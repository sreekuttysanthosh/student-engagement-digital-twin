"""Load, validate, clean, label and split the Canvas Network Person-Course data."""
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from engagement_twin.config import find_project_root, load_config

log = logging.getLogger(__name__)

SOURCE_COL = "completed_%"
IDENTIFIERS = ["userid_DI", "course_id_DI"]


def resolve(path: str | Path, root: Path) -> Path:
    """Turn a relative path from the config into an absolute path under root."""
    path = Path(path)
    return path if path.is_absolute() else root / path


def load_raw(path: str | Path, root: Path) -> pd.DataFrame:
    """Load the tab-separated Canvas dataset."""
    return pd.read_csv(resolve(path, root), sep="\t", low_memory=False)


def validate_schema(df: pd.DataFrame, features: list[str]) -> None:
    """Raise ValueError if a required column is missing."""
    required = [*features, SOURCE_COL, *IDENTIFIERS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Keep rows with a non-missing completed_% and check it is a 0-1 fraction."""
    validate_schema(df, features)
    out = df[df[SOURCE_COL].notna()].reset_index(drop=True)
    low, high = out[SOURCE_COL].min(), out[SOURCE_COL].max()
    if low < 0 or high > 1:
        raise ValueError(f"{SOURCE_COL} outside 0-1 (min={low}, max={high})")
    log.info("Cleaning: %d -> %d rows", len(df), len(out))
    return out


def make_target(df: pd.DataFrame, threshold: float, target: str) -> pd.DataFrame:
    """Label 1 if completed_% >= threshold, else 0."""
    out = df.copy()
    out[target] = (out[SOURCE_COL] >= threshold).astype(int)
    return out


def assert_no_leakage(features: list[str], target: str) -> None:
    """Raise ValueError if target information is used as an input feature."""
    leaked = {SOURCE_COL, target}.intersection(features)
    if leaked:
        raise ValueError(f"Target leakage: {leaked} used as feature")


def model_table(df: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    """Keep only the model features and the target, in fixed order."""
    assert_no_leakage(features, target)
    return df[[*features, target]].copy()


def split(
    df: pd.DataFrame, target: str, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified, reproducible train/test split."""
    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[target]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_audit(
    raw: pd.DataFrame,
    table: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    cfg: dict,
) -> dict:
    """Collect measured dataset statistics."""
    features, target = cfg["features"], cfg["target"]
    return {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(table)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "seed": cfg["seed"],
        "threshold": cfg["target_threshold"],
        "missing_pct_train": (train[features].isna().mean() * 100).round(2).to_dict(),
        "missing_pct_test": (test[features].isna().mean() * 100).round(2).to_dict(),
        "class_share_train": train[target].value_counts(normalize=True).round(4).to_dict(),
        "class_share_test": test[target].value_counts(normalize=True).round(4).to_dict(),
    }


def prepare(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the full pipeline, save train/test CSVs and an audit JSON."""
    root = root or find_project_root()
    cfg = load_config(root)
    features, target = cfg["features"], cfg["target"]

    raw = load_raw(cfg["raw_path"], root)
    labelled = make_target(clean(raw, features), cfg["target_threshold"], target)
    table = model_table(labelled, features, target)
    train, test = split(table, target, cfg["test_size"], cfg["seed"])

    out_dir = resolve(cfg["processed_dir"], root)
    rep_dir = resolve(cfg["reports_dir"], root)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    audit = build_audit(raw, table, train, test, cfg)
    (rep_dir / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return train, test, audit