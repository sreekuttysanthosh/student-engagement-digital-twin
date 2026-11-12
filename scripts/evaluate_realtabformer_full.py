"""Evaluate full RealTabFormer synthetic datasets against the fixed real test set."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sdv.metadata import Metadata

from engagement_twin.classifier import build_model, evaluate, train
from engagement_twin.config import load_config
from engagement_twin.evaluation import (
    TABLE,
    column_shapes,
    missing_similarity,
    overall_quality,
    pair_trends,
)

plt.switch_backend("Agg")

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["nevents", "ndays_act", "ncontent", "nforum_posts"]
TARGET = "Engagement_Level"
OUTPUT_DIR = ROOT / "artifacts" / "realtabformer" / "full"
STRATEGIES = ("random", "from_column")
FIGURE_DIR = ROOT / "reports" / "figures"
SHAP_REPORT_PATH = ROOT / "reports" / "shap_global_importance.json"


def explain_classifier(classifier, test_features: pd.DataFrame, name: str) -> dict:
    """Create and verify SHAP explanations on the held-out real test features."""
    explainer = shap.TreeExplainer(classifier)
    values = explainer.shap_values(test_features)
    if isinstance(values, list):
        values = values[-1]
    values = pd.DataFrame(values, columns=FEATURES, index=test_features.index)

    base_value = float(pd.Series(explainer.expected_value).iloc[-1])
    margins = classifier.predict(test_features, output_margin=True)
    max_additivity_gap = float(
        (base_value + values.sum(axis=1).to_numpy() - margins).__abs__().max()
    )
    if max_additivity_gap > 1e-3:
        raise ValueError(
            f"SHAP additivity check failed for {name}: "
            f"maximum gap {max_additivity_gap:.6g}"
        )

    mean_abs = values.abs().mean().round(4).to_dict()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(5, 3))
    pd.Series(mean_abs).sort_values().plot.barh(ax=axis, color="#4c72b0")
    axis.set_xlabel("mean |SHAP| (log-odds of high engagement)")
    axis.set_title(name)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / f"shap_importance_{name}.png", dpi=150)
    plt.close(figure)

    shap.summary_plot(
        values.to_numpy(),
        test_features,
        feature_names=FEATURES,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"shap_summary_{name}.png", dpi=150)
    plt.close()

    return {
        "mean_abs_shap": mean_abs,
        "max_additivity_gap": max_additivity_gap,
    }


def update_shap_report(name: str, shap_result: dict) -> None:
    """Add the new model's SHAP metrics without removing existing models."""
    SHAP_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = (
        json.loads(SHAP_REPORT_PATH.read_text(encoding="utf-8"))
        if SHAP_REPORT_PATH.is_file()
        else {"mean_abs_shap": {}}
    )
    report.setdefault(
        "explained_data",
        "real test features (SHAP only reads the data; nothing is fitted or tuned)",
    )
    report.setdefault("units", "mean |SHAP| in log-odds of class 1")
    report.setdefault("mean_abs_shap", {})[name] = shap_result["mean_abs_shap"]
    report.setdefault("max_additivity_gap", {})[name] = (
        shap_result["max_additivity_gap"]
    )
    SHAP_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


def evaluate_strategy(strategy: str) -> dict:
    """Evaluate one saved synthetic dataset for fidelity and classification."""
    data_dir = ROOT / "data" / "processed"
    train_real = pd.read_csv(data_dir / "train.csv")
    test_real = pd.read_csv(data_dir / "test.csv")
    test_features = test_real[FEATURES]
    synthetic_path = OUTPUT_DIR / f"full_{strategy}_seed_42_restored.csv"

    if not synthetic_path.is_file():
        raise FileNotFoundError(f"Synthetic data not found: {synthetic_path}")

    synthetic = pd.read_csv(synthetic_path)
    expected_columns = FEATURES + [TARGET]
    if synthetic.columns.tolist() != expected_columns:
        raise ValueError(
            f"{synthetic_path.name}: expected columns {expected_columns}, "
            f"received {synthetic.columns.tolist()}"
        )
    if len(synthetic) != len(train_real):
        raise ValueError(
            f"{synthetic_path.name}: expected {len(train_real)} rows, "
            f"received {len(synthetic)}"
        )
    if synthetic[TARGET].isna().any() or not synthetic[TARGET].isin([0, 1]).all():
        raise ValueError(f"{synthetic_path.name}: target must be non-missing and binary")
    if synthetic[TARGET].nunique() != 2:
        raise ValueError(f"{synthetic_path.name}: both target classes must be present")

    metadata = Metadata.detect_from_dataframes({TABLE: train_real[FEATURES]})
    shapes = column_shapes(train_real[FEATURES], synthetic[FEATURES], metadata)
    trends, pair_scores = pair_trends(
        train_real[FEATURES],
        synthetic[FEATURES],
        FEATURES,
    )
    missing_scores = missing_similarity(
        train_real[FEATURES],
        synthetic[FEATURES],
        FEATURES,
    )

    model_dir = OUTPUT_DIR / f"full_{strategy}_seed_42"
    model_dir.mkdir(parents=True, exist_ok=True)
    classifier_path = model_dir / "xgb_realtabformer_full.joblib"
    generation_report_path = OUTPUT_DIR / f"full_{strategy}_seed_42_report.json"
    classifier = build_model(load_config(ROOT)["xgboost"], seed=42)
    train(classifier, synthetic, FEATURES, TARGET)
    classification = evaluate(classifier, test_real, FEATURES, TARGET)
    joblib.dump(classifier, classifier_path)
    shap_name = f"realtabformer_{strategy}_seed_42"
    shap_result = explain_classifier(classifier, test_features, shap_name)
    update_shap_report(shap_name, shap_result)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "evaluation_completed",
        "generator": "RealTabFormer",
        "strategy": strategy,
        "seed": 42,
        "generation_report": (
            str(generation_report_path.relative_to(ROOT).as_posix())
            if generation_report_path.is_file()
            else None
        ),
        "synthetic_rows": len(synthetic),
        "synthetic_class_distribution": (
            synthetic[TARGET].value_counts(normalize=True).sort_index().to_dict()
        ),
        "fidelity_vs_real_train": {
            "column_shapes": shapes,
            "pair_trends": trends,
            "overall_quality": overall_quality(shapes, trends),
            "pair_trends_per_pair": pair_scores,
            "missing_value_similarity": missing_scores,
            "real_missing_values": (
                train_real[FEATURES].isna().sum().to_dict()
            ),
            "synthetic_missing_values": (
                synthetic[FEATURES].isna().sum().to_dict()
            ),
        },
        "classification_on_real_test": classification,
        "shap_on_real_test": shap_result,
        "classification_training": (
            "XGBoost trained on this synthetic dataset only; evaluated on the "
            "fixed real held-out test set with original missing values."
        ),
        "outputs": {
            "synthetic_csv": str(synthetic_path.relative_to(ROOT).as_posix()),
            "classifier_model": str(classifier_path.relative_to(ROOT).as_posix()),
            "shap_importance_figure": (
                f"reports/figures/shap_importance_{shap_name}.png"
            ),
            "shap_summary_figure": f"reports/figures/shap_summary_{shap_name}.png",
            "shap_global_report": str(SHAP_REPORT_PATH.relative_to(ROOT).as_posix()),
        },
    }

    report_path = OUTPUT_DIR / f"full_{strategy}_seed_42_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"{strategy}: fidelity={report['fidelity_vs_real_train']['overall_quality']:.4f}, "
        f"weighted F1={classification['weighted']['f1']:.4f}, "
        f"class-1 F1={classification['per_class']['1']['f1-score']:.4f}"
    )
    print(f"Report: {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        choices=[*STRATEGIES, "both"],
        default="from_column",
        help="Which full RealTabFormer synthetic dataset to evaluate.",
    )
    args = parser.parse_args()
    strategies = STRATEGIES if args.strategy == "both" else (args.strategy,)
    for strategy in strategies:
        evaluate_strategy(strategy)


if __name__ == "__main__":
    main()
