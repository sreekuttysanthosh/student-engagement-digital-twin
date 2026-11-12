
"""Streamlit dashboard. Reads saved reports and models only; nothing is trained here."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Locate the project root and make the src package available.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from engagement_twin.config import find_project_root, load_config  # noqa: E402

ROOT = find_project_root(PROJECT_ROOT)
CFG = load_config(ROOT)

FEATURES = CFG["features"]
REPORTS = ROOT / CFG["reports_dir"]
MODELS = ROOT / CFG["artifacts_dir"] / "models"
REALTABFORMER = ROOT / CFG["artifacts_dir"] / "realtabformer" / "full"

MAX_VALUE = {
    "nevents": 50000,
    "ndays_act": 500,
    "ncontent": 100,
    "nforum_posts": 5000,
}

st.set_page_config(
    page_title="Student Engagement Twin",
    layout="wide",
)


def load_json(name: str | Path) -> dict | list | None:
    """Read a saved JSON report using a reliable project-relative path."""
    path = Path(name)

    if not path.is_absolute():
        path = REPORTS / path

    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


def unavailable(what: str) -> None:
    """Show a clear unavailable state instead of displaying invented values."""
    st.warning(
        f"{what} is unavailable: the required saved file could not be found."
    )


@st.cache_resource
def load_model(path: str):
    """Load a saved classifier model once."""
    return joblib.load(path)


def available_models() -> dict[str, Path]:
    """Return saved classifier models, including full RealTabFormer runs."""
    paths = {
        path.stem: path
        for path in MODELS.glob("*.joblib")
    }

    for strategy in ("random", "from_column"):
        model = (
            REALTABFORMER
            / f"full_{strategy}_seed_42"
            / "xgb_realtabformer_full.joblib"
        )

        if model.is_file():
            paths[f"realtabformer_{strategy}_seed_42"] = model

    return paths


def feature_inputs(prefix: str) -> list[float]:
    """Collect four validated inputs; each can also be marked missing."""
    values = []
    states = {}

    cols = st.columns(len(FEATURES))

    for col, name in zip(cols, FEATURES, strict=True):
        with col:
            mode = st.selectbox(
                name,
                ["Value", "Missing"],
                index=0,
                key=f"{prefix}_{name}_mode",
            )

            value = st.number_input(
                f"{name} value",
                min_value=0,
                max_value=MAX_VALUE[name],
                value=0,
                step=1,
                key=f"{prefix}_{name}_val",
                disabled=(mode == "Missing"),
            )

            states[name] = mode
            values.append(
                np.nan if mode == "Missing" else float(value)
            )

    st.caption(f"Input state read by the page: {states}")

    return values


def page_overview() -> None:
    st.title("Student engagement digital twin")

    st.write(
        "Synthetic student records (CTGAN, TVAE) are used to train "
        "an XGBoost classifier, which is tested on real, unseen students. "
        "Replication of Kumi et al. (2025), using the Canvas Network "
        "Person-Course data."
    )

    st.info(
        "Exploratory decision support only. Not for high-stakes decisions "
        "about individual students. This is a synthetic representation "
        "of learning activity, not a live LMS twin, and it makes no "
        "privacy or anonymity claim."
    )

    audit = load_json("data_audit.json")

    if audit is None:
        unavailable("Data audit")
    else:
        st.subheader("Data")

        a, b, c, d = st.columns(4)

        a.metric("Raw rows", f"{audit['raw_rows']:,}")
        b.metric("Clean rows", f"{audit['clean_rows']:,}")
        c.metric("Train rows", f"{audit['train_rows']:,}")
        d.metric("Test rows", f"{audit['test_rows']:,}")

        st.caption(
            f"Class share (train): {audit['class_share_train']}"
        )

    summary = load_json("classification_summary.json")

    if summary is None:
        unavailable("Classification summary")
    else:
        st.subheader("XGBoost on the real test set (one seed)")

        st.dataframe(
            pd.DataFrame(summary["summary"]).round(4)
        )

        st.caption(
            "Weighted scores are dominated by class 0 (about 82.5%). "
            "Read class-1 metrics and error counts alongside them."
        )

    realtabformer = []

    for strategy in ("random", "from_column"):
        report_path = (
            REALTABFORMER
            / f"full_{strategy}_seed_42_evaluation.json"
        )

        report = load_json(report_path)

        if report is not None:
            metrics = report["classification_on_real_test"]

            realtabformer.append(
                {
                    "Model": f"RealTabFormer ({strategy})",
                    "Weighted F1": metrics["weighted"]["f1"],
                    "Class-1 Precision": (
                        metrics["per_class"]["1"]["precision"]
                    ),
                    "Class-1 Recall": (
                        metrics["per_class"]["1"]["recall"]
                    ),
                    "Class-1 F1": (
                        metrics["per_class"]["1"]["f1-score"]
                    ),
                }
            )

            if report.get("generation_report") is None:
                st.warning(
                    "The RealTabFormer evaluation has no generation "
                    "report; its full-size CSV was evaluated, but "
                    "training provenance is unverified."
                )

    if realtabformer:
        st.subheader("RealTabFormer on the same real test set")

        st.dataframe(
            pd.DataFrame(realtabformer)
            .set_index("Model")
            .round(4)
        )


def page_prediction() -> None:
    st.title("Prediction")

    models = available_models()

    if not models:
        unavailable("Saved models")
        return

    choice = st.selectbox("Model", sorted(models))

    st.caption(
        "Each model is trained on the real training set or its "
        "named synthetic set."
    )

    values = feature_inputs("pred")

    if st.button("Predict"):
        model = load_model(str(models[choice]))

        row = pd.DataFrame(
            [values],
            columns=FEATURES,
        )

        st.caption(
            f"Values sent to {choice}: {row.iloc[0].to_dict()}"
        )

        proba = float(model.predict_proba(row)[0, 1])

        st.metric(
            "Estimated probability of high engagement",
            f"{proba:.1%}",
        )

        st.write(
            "Predicted class:",
            "high (1)" if proba >= 0.5 else "low (0)",
        )

        st.caption(
            "Missing forum activity is not proof of low engagement. "
            "This output describes model behaviour, not a decision "
            "about a student."
        )


def page_fidelity() -> None:
    st.title("Real vs synthetic")

    report = load_json("fidelity_summary.json")

    if report is None:
        unavailable("Fidelity summary")
        return

    if isinstance(report, list):
        table = pd.DataFrame(report)

    elif (
        isinstance(report, dict)
        and isinstance(report.get("summary"), dict)
    ):
        table = pd.DataFrame.from_dict(
            report["summary"],
            orient="index",
        )

    else:
        raise ValueError(
            "Fidelity summary must be a list or contain "
            "a summary mapping."
        )

    column_names = {
        "Column Shapes": "shapes",
        "Pair Trends": "pairs",
        "Overall Quality": "overall",
        "Mean Missing Similarity": "missing",
    }

    table = table.rename(columns=column_names)

    if {"Variant", "Generator"}.issubset(table.columns):
        table = table.set_index(["Variant", "Generator"])

    table.index.name = table.index.name or "Experiment"

    rows = []

    for strategy in ("random", "from_column"):
        evaluation_path = (
            REALTABFORMER
            / f"full_{strategy}_seed_42_evaluation.json"
        )

        evaluation = load_json(evaluation_path)

        if evaluation is not None:
            fidelity = evaluation["fidelity_vs_real_train"]

            rows.append(
                {
                    "Variant": strategy,
                    "Generator": "RealTabFormer",
                    "shapes": fidelity["column_shapes"],
                    "pairs": fidelity["pair_trends"],
                    "overall": fidelity["overall_quality"],
                    "missing": np.mean(
                        list(
                            fidelity[
                                "missing_value_similarity"
                            ].values()
                        )
                    ),
                }
            )

    if rows:
        realtabformer_table = (
            pd.DataFrame(rows)
            .set_index(["Variant", "Generator"])
        )

        table = pd.concat(
            [table, realtabformer_table]
        )

        for strategy in ("random", "from_column"):
            evaluation_path = (
                REALTABFORMER
                / f"full_{strategy}_seed_42_evaluation.json"
            )

            evaluation = load_json(evaluation_path)

            if (
                evaluation is not None
                and evaluation.get("generation_report") is None
            ):
                st.warning(
                    "Random RealTabFormer generation provenance "
                    "is unverified: its full-size CSV exists, "
                    "but its generation report is missing."
                )

    st.dataframe(table.round(4))

    st.caption(
        "Scores compare each synthetic set with real train. "
        "Pair trends are computed manually because the SDV report "
        "returned NaN in SDV 2.0.0 / SDMetrics 0.32.0. "
        "Baseline vs target columns are not strictly like-for-like. "
        "RealTabFormer is shown separately; these are single-seed "
        "comparisons."
    )

    st.bar_chart(
        table[["shapes", "pairs", "overall"]]
        .reset_index(drop=True)
    )


def page_explainability() -> None:
    st.title("Explainability (SHAP)")

    report = load_json("shap_global_importance.json")

    if report is None:
        unavailable("SHAP importance")
        return

    table = pd.DataFrame(
        report["mean_abs_shap"]
    ).loc[FEATURES]

    st.write(
        "Mean |SHAP| per feature (log-odds), "
        "on the real test features."
    )

    st.dataframe(table.round(3))
    st.bar_chart(table)

    st.caption(
        "SHAP explains model behaviour, not causes. "
        "Ranks 3 and 4 swap between models and are not interpreted."
    )

    figures = REPORTS / "figures"

    model = st.selectbox(
        "Saved summary plot",
        list(report["mean_abs_shap"]),
    )

    image = figures / f"shap_summary_{model}.png"

    if image.exists():
        st.image(str(image))
    else:
        unavailable("This SHAP plot")


def page_whatif() -> None:
    st.title("What-if (exploratory, non-causal)")

    st.warning(
        "Changing one input shows how the model's output moves. "
        "It does not show what would happen to a real student."
    )

    models = available_models()

    if not models:
        unavailable("Saved models")
        return

    choice = st.selectbox(
        "Model",
        sorted(models),
        key="wi_model",
    )

    base = feature_inputs("wi")

    vary = st.selectbox(
        "Feature to vary",
        FEATURES,
    )

    high = st.number_input(
        "Sweep up to",
        min_value=1,
        max_value=MAX_VALUE[vary],
        value=min(100, MAX_VALUE[vary]),
    )

    if st.button("Run sweep"):
        grid = np.linspace(0, high, 40)

        frame = pd.DataFrame(
            [base] * len(grid),
            columns=FEATURES,
        )

        frame[vary] = grid

        model = load_model(str(models[choice]))

        proba = model.predict_proba(frame)[:, 1]

        chart_data = pd.DataFrame(
            {"P(high engagement)": proba},
            index=grid,
        )

        chart_data.index.name = vary

        st.line_chart(chart_data)


PAGES = {
    "Overview": page_overview,
    "Prediction": page_prediction,
    "Real vs synthetic": page_fidelity,
    "Explainability": page_explainability,
    "What-if": page_whatif,
}

PAGES[st.sidebar.radio("Page", list(PAGES))]()

