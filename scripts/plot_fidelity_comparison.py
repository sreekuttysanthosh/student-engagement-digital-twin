"""Plot synthetic-data fidelity metrics from the saved evaluations."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "reports" / "fidelity_multi_seed_summary.json"
REALTABFORMER_PATHS = {
    "random": ROOT
    / "artifacts"
    / "realtabformer"
    / "full"
    / "full_random_seed_42_evaluation.json",
    "from_column": ROOT
    / "artifacts"
    / "realtabformer"
    / "full"
    / "full_from_column_seed_42_evaluation.json",
}
OUTPUT_PATH = ROOT / "reports" / "figures" / "fidelity_comparison.png"

METRICS = (
    ("Column-shape similarity", "column_shapes_mean", "column_shapes_std"),
    ("Pair-trend similarity", "pair_trends_mean", "pair_trends_std"),
    ("Missingness similarity", "missing_sim_mean_mean", "missing_sim_mean_std"),
)
GENERATOR_COLORS = {
    "CTGAN": "#4c78a8",
    "TVAE": "#f28e2b",
    "RealTabFormer": "#59a14f",
}
STRATEGY_MARKERS = {"random": "o", "from_column": "s"}


def load_json(path: Path) -> dict:
    """Load a required JSON file and report missing inputs explicitly."""
    if not path.is_file():
        raise FileNotFoundError(f"Required evaluation file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_multiseed_rows(summary: dict) -> list[dict]:
    """Read the four-seed means and sample SDs for CTGAN and TVAE."""
    rows = []
    for generator in ("ctgan", "tvae"):
        for strategy in ("random", "from_column"):
            key = f"('{strategy}', '{generator}')"
            statistics = summary["multi_seed_mean_std"][key]
            rows.append(
                {
                    "generator": generator.upper(),
                    "strategy": strategy,
                    "means": {
                        mean_key: statistics[mean_key]
                        for _, mean_key, _ in METRICS
                    },
                    "stds": {
                        std_key: statistics[std_key]
                        for _, _, std_key in METRICS
                    },
                    "single_seed": False,
                }
            )
    return rows


def get_realtabformer_rows() -> list[dict]:
    """Read the single-seed RealTabFormer fidelity values."""
    rows = []
    for strategy, path in REALTABFORMER_PATHS.items():
        fidelity = load_json(path)["fidelity_vs_real_train"]
        rows.append(
            {
                "generator": "RealTabFormer",
                "strategy": strategy,
                "means": {
                    "column_shapes_mean": fidelity["column_shapes"],
                    "pair_trends_mean": fidelity["pair_trends"],
                    "missing_sim_mean_mean": sum(
                        fidelity["missing_value_similarity"].values()
                    )
                    / len(fidelity["missing_value_similarity"]),
                },
                "stds": {},
                "single_seed": True,
            }
        )
    return rows


def main() -> None:
    summary = load_json(SUMMARY_PATH)
    rows = get_multiseed_rows(summary) + get_realtabformer_rows()
    labels = [
        f"{row['generator']} · "
        f"{'From-column' if row['strategy'] == 'from_column' else 'Random'}"
        for row in rows
    ]

    figure, axes = plt.subplots(1, 3, figsize=(15, 6.5), sharey=True)
    figure.suptitle("Synthetic-data fidelity vs. real training data", y=0.98)
    y_positions = list(range(len(rows)))

    for axis, (title, mean_key, std_key) in zip(axes, METRICS, strict=True):
        for position, row in zip(y_positions, rows, strict=True):
            axis.errorbar(
                row["means"][mean_key],
                position,
                xerr=None if row["single_seed"] else row["stds"][std_key],
                fmt=STRATEGY_MARKERS[row["strategy"]],
                markersize=7,
                color=GENERATOR_COLORS[row["generator"]],
                ecolor=GENERATOR_COLORS[row["generator"]],
                capsize=3,
                elinewidth=1.5,
                linestyle="none",
            )
        axis.set_title(title)
        axis.set_xlim(0.88, 1.005)
        axis.set_xticks([0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00])
        axis.xaxis.grid(True, color="#dddddd", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.set_xlabel("Similarity score")

    axes[0].set_yticks(y_positions, labels)
    axes[0].invert_yaxis()

    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=color,
            label=generator,
            markersize=7,
        )
        for generator, color in GENERATOR_COLORS.items()
    ]
    legend_handles.extend(
        [
            Line2D(
                [],
                [],
                marker=marker,
                linestyle="none",
                color="#555555",
                label="From-column" if strategy == "from_column" else "Random",
                markersize=7,
            )
            for strategy, marker in STRATEGY_MARKERS.items()
        ]
    )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.89),
        ncols=5,
        frameon=True,
    )
    figure.text(
        0.5,
        0.025,
        "CTGAN/TVAE: mean ± sample SD over four seeds. "
        "RealTabFormer: single-seed points (no SD estimate). "
        "X-axes start at 0.88; higher similarity is better.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.84))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
