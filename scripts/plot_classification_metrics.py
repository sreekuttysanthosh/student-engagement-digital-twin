"""Plot class-1 precision, recall, and F1 for the saved model evaluations."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "reports" / "classification_summary.json"
REALTABFORMER_PATHS = {
    "RealTabFormer · Random": (
        ROOT / "artifacts" / "realtabformer" / "full" / "full_random_seed_42_evaluation.json"
    ),
    "RealTabFormer · From-column": (
        ROOT
        / "artifacts"
        / "realtabformer"
        / "full"
        / "full_from_column_seed_42_evaluation.json"
    ),
}
OUTPUT_PATH = ROOT / "reports" / "figures" / "class1_precision_recall_f1.png"


def load_json(path: Path) -> dict:
    """Load a required JSON file and report missing inputs explicitly."""
    if not path.is_file():
        raise FileNotFoundError(f"Required evaluation file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    summary = load_json(SUMMARY_PATH)["summary"]
    model_specs = [
        ("Real training reference", "real_train_reference", False),
        ("CTGAN · Random", "ctgan_random", False),
        ("CTGAN · From-column", "ctgan_from_column", False),
        ("TVAE · Random", "tvae_random", False),
        ("TVAE · From-column", "tvae_from_column", False),
        ("CTGAN · Random (balanced)", "ctgan_random_balanced", True),
        ("TVAE · Random (balanced)", "tvae_random_balanced", True),
    ]
    models = [
        {
            "label": label,
            "balanced": balanced,
            "metrics": {
                "Precision": summary[key]["class1_precision"],
                "Recall": summary[key]["class1_recall"],
                "F1": summary[key]["class1_f1"],
            },
        }
        for label, key, balanced in model_specs
    ]

    for offset, (label, path) in enumerate(REALTABFORMER_PATHS.items()):
        evaluation = load_json(path)["classification_on_real_test"]["per_class"]["1"]
        models.insert(
            5 + offset,
            {
                "label": label,
                "balanced": False,
                "metrics": {
                    "Precision": evaluation["precision"],
                    "Recall": evaluation["recall"],
                    "F1": evaluation["f1-score"],
                },
            },
        )

    metric_names = ("Precision", "Recall", "F1")
    colors = {"Precision": "#4c78a8", "Recall": "#59a14f", "F1": "#f28e2b"}
    y_positions = list(range(len(models)))
    offsets = (-0.23, 0, 0.23)

    figure, axis = plt.subplots(figsize=(11, 7))
    figure.suptitle("Class-1 precision, recall, and F1 on the real test set", y=0.98)
    for position, model in zip(y_positions, models, strict=True):
        if model["balanced"]:
            axis.axhspan(
                position - 0.43,
                position + 0.43,
                color="#fff1cc",
                zorder=0,
            )

    for metric, offset in zip(metric_names, offsets, strict=True):
        for balanced in (False, True):
            selected = [
                (position, model)
                for position, model in zip(y_positions, models, strict=True)
                if model["balanced"] == balanced
            ]
            axis.barh(
                [position + offset for position, _ in selected],
                [model["metrics"][metric] for _, model in selected],
                height=0.2,
                color=colors[metric],
                edgecolor="#8a5a00" if balanced else colors[metric],
                hatch="//" if balanced else None,
            )

    axis.set_yticks(y_positions, [model["label"] for model in models])
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xlabel("Score")
    axis.xaxis.grid(True, color="#d9d9d9", linewidth=0.8)
    axis.set_axisbelow(True)
    legend_handles = [
        Patch(facecolor=colors[metric], label=metric) for metric in metric_names
    ]
    legend_handles.append(
        Patch(
            facecolor="#fff1cc",
            edgecolor="#8a5a00",
            hatch="//",
            label="Balanced training",
        )
    )
    axis.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=4,
        frameon=True,
    )
    axis.text(
        0.01,
        -0.12,
        "Amber-highlighted rows and hatched bars indicate balanced synthetic training; "
        "real training is the reference.",
        transform=axis.transAxes,
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.92))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
