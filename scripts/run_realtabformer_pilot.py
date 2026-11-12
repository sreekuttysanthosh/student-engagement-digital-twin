"""Run a CPU-limited RealTabFormer pilot on the random-missingness strategy."""
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from realtabformer import REaLTabFormer
from sdv.metadata import Metadata

from engagement_twin.classifier import build_model, train
from engagement_twin.classifier import evaluate as evaluate_classifier
from engagement_twin.config import load_config
from engagement_twin.evaluation import (
    TABLE,
    column_shapes,
    missing_similarity,
    overall_quality,
    pair_trends,
)
from engagement_twin.preprocessing import restore_random

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
TRAINING_STEPS = 300
SAMPLE_ROWS = 5_000
FEATURES = ["nevents", "ndays_act", "ncontent", "nforum_posts"]
TARGET = "Engagement_Level"
MODEL_DIR = ROOT / "artifacts" / "realtabformer" / "pilot_random_seed_42"
REPORT_PATH = ROOT / "reports" / "realtabformer_pilot.json"
SYNTHETIC_PATH = ROOT / "artifacts" / "realtabformer" / "pilot_random_seed_42_restored.csv"


def main() -> None:
    config = load_config(ROOT)
    train_real = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
    test_real = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
    generator_input = pd.read_csv(
        ROOT / "data" / "processed" / "gen_input_random_target.csv"
    )

    expected_columns = FEATURES + [TARGET]
    if generator_input.columns.tolist() != expected_columns:
        raise ValueError(
            f"Unexpected RealTabFormer input columns: {generator_input.columns.tolist()}"
        )
    if generator_input.isna().any().any():
        raise ValueError("RealTabFormer training input must be fully imputed.")
    if len(generator_input) != len(train_real):
        raise ValueError("Generator input and real training data row counts differ.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = REaLTabFormer(
        model_type="tabular",
        epochs=10,
        batch_size=8,
        random_state=SEED,
        checkpoints_dir=str(MODEL_DIR / "checkpoints"),
        samples_save_dir=str(MODEL_DIR / "samples"),
        full_save_dir=str(MODEL_DIR / "full"),
        max_steps=TRAINING_STEPS,
    )
    model.fit(
        generator_input,
        device="cpu",
        num_bootstrap=20,
        frac=0.1,
        frac_max_data=10_000,
        qt_max="compute",
        gen_rounds=1,
        n_critic=10,
        gen_kwargs={},
    )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    generated = model.sample(n_samples=SAMPLE_ROWS, device="cpu", gen_batch=128)
    if generated.columns.tolist() != expected_columns:
        raise ValueError(f"Unexpected generated columns: {generated.columns.tolist()}")
    if generated[TARGET].isna().any() or not generated[TARGET].isin([0, 1]).all():
        raise ValueError("Generated engagement labels must be non-missing binary values.")

    miss_rates = train_real[FEATURES].isna().mean()
    restored = restore_random(
        generated,
        miss_rates,
        FEATURES,
        seed=SEED,
        target=TARGET,
    )
    restored.to_csv(SYNTHETIC_PATH, index=False)

    metadata = Metadata.detect_from_dataframes(
        {TABLE: train_real[FEATURES]}
    )
    shapes = column_shapes(train_real[FEATURES], restored[FEATURES], metadata)
    trends, pair_scores = pair_trends(
        train_real[FEATURES],
        restored[FEATURES],
        FEATURES,
    )
    missing = missing_similarity(
        train_real[FEATURES],
        restored[FEATURES],
        FEATURES,
    )

    classifier = build_model(config["xgboost"], seed=SEED)
    train(classifier, restored, FEATURES, TARGET)
    classifier_results = evaluate_classifier(classifier, test_real, FEATURES, TARGET)
    model.full_save_dir = model.full_save_dir.as_posix()
    model.save(MODEL_DIR / "realtabformer", allow_overwrite=True)
    joblib.dump(classifier, MODEL_DIR / "xgb_realtabformer_pilot.joblib")

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pilot_completed",
        "generator": {
            "name": "RealTabFormer",
            "version": "0.2.4",
            "transformers_version": "4.57.6",
            "datasets_version": "4.4.1",
            "model_type": "tabular",
            "device": "cpu",
            "seed": SEED,
            "epochs_requested": 10,
            "max_steps": TRAINING_STEPS,
            "rows_used_for_fit": len(generator_input),
            "rows_sampled": SAMPLE_ROWS,
            "strategy": "random",
        },
        "fidelity_vs_real_train": {
            "column_shapes": shapes,
            "pair_trends": trends,
            "overall_quality": overall_quality(shapes, trends),
            "pair_trends_per_pair": pair_scores,
            "missing_value_similarity": missing,
        },
        "synthetic_class_share": (
            restored[TARGET].value_counts(normalize=True).sort_index().to_dict()
        ),
        "classification_on_real_test": classifier_results,
        "outputs": {
            "synthetic_csv": str(SYNTHETIC_PATH.relative_to(ROOT)),
            "generator_model": str((MODEL_DIR / "realtabformer").relative_to(ROOT)),
            "classifier_model": str(
                (MODEL_DIR / "xgb_realtabformer_pilot.joblib").relative_to(ROOT)
            ),
        },
        "limitations": [
            "Exploratory pilot, not a paper-scale replication: training was capped at 300 steps.",
            "Only the random missingness strategy was run.",
            "The 5,000-row sample is smaller than the 21,328-row real training set.",
            "This is a single-seed result and should not support model-winner claims.",
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
