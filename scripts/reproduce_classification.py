"""Re-evaluate all seven training sets on the fixed real test set.

Retrains XGBoost with the parameters stored in
reports/classification_summary.json, compares the result with that file,
and adds ROC-AUC, PR-AUC, macro F1 and bootstrap confidence intervals.
Writes reports/classification_reproduction.json. Overwrites nothing else.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "artifacts" / "generators"
FEATURES = ["nevents", "ndays_act", "ncontent", "nforum_posts"]
TARGET = "Engagement_Level"
N_BOOT = 1000

EXPERIMENTS = {
    "real_train_reference": ROOT / "data" / "processed" / "train.csv",
    "ctgan_random": GEN / "ctgan_random_target_restored.csv",
    "tvae_random": GEN / "tvae_random_target_restored.csv",
    "ctgan_random_balanced": GEN / "ctgan_random_target_balanced_restored.csv",
    "tvae_random_balanced": GEN / "tvae_random_target_balanced_restored.csv",
    "ctgan_from_column": GEN / "ctgan_from_column_target_restored.csv",
    "tvae_from_column": GEN / "tvae_from_column_target_restored.csv",
}

# Comparisons beyond "each synthetic set vs the real reference".
PAIRS = [
    ("ctgan_from_column", "ctgan_random"),
    ("tvae_from_column", "tvae_random"),
    ("tvae_random", "ctgan_random"),
    ("tvae_from_column", "ctgan_from_column"),
]


def load_xy(path):
    """Load a CSV, validate columns and target, return (features, target)."""
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    if df[TARGET].isna().any() or not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError(f"{path.name}: target must be 0/1 with no NaN")
    return df[FEATURES].astype(float), df[TARGET].astype(int)


def point_metrics(y, pred, proba):
    """Compute point estimates on the real test set."""
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, pred, labels=[0, 1], zero_division=0
    )
    wp, wr, wf, _ = precision_recall_fscore_support(
        y, pred, average="weighted", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "weighted": {
            "precision": float(wp), "recall": float(wr), "f1": float(wf),
        },
        "class1": {
            "precision": float(prec[1]),
            "recall": float(rec[1]),
            "f1": float(f1[1]),
        },
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def bootstrap_f1(y, preds, seed):
    """Paired bootstrap over test rows (same resamples for every model)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    yb = y[idx]
    n1 = yb.sum(axis=1)
    out = {}
    for name, pred in preds.items():
        pb = pred[idx]
        tp = ((yb == 1) & (pb == 1)).sum(axis=1)
        fp = ((yb == 0) & (pb == 1)).sum(axis=1)
        fn = ((yb == 1) & (pb == 0)).sum(axis=1)
        tn = ((yb == 0) & (pb == 0)).sum(axis=1)
        f1_1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        f1_0 = 2 * tn / np.maximum(2 * tn + fp + fn, 1)
        out[name] = {
            "class1_f1": f1_1,
            "weighted_f1": (n1 * f1_1 + (n - n1) * f1_0) / n,
        }
    return out


def ci(values):
    """Return the 95% percentile interval, rounded."""
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def main():
    prev_path = ROOT / "reports" / "classification_summary.json"
    prev = json.loads(prev_path.read_text(encoding="utf-8-sig"))
    params, seed = prev["xgb_params"], prev["seed"]

    X_test, y_test = load_xy(ROOT / "data" / "processed" / "test.csv")
    y = y_test.to_numpy()

    results, preds = {}, {}
    for name, path in EXPERIMENTS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing training file for {name}: {path}")
        X, yt = load_xy(path)
        model = xgb.XGBClassifier(**params).fit(X, yt)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        preds[name] = pred

        res = point_metrics(y, pred, proba)
        res["train_rows"] = int(len(X))
        res["train_class1_share"] = round(float(yt.mean()), 4)
        res["train_file"] = path.relative_to(ROOT).as_posix()
        res["train_sha256_12"] = hashlib.sha256(path.read_bytes()).hexdigest()[:12]

        old = prev["summary"].get(name)
        if old is None:
            res["matches_previous"] = None
        else:
            (_, fp), (fn, _) = res["confusion_matrix"]
            res["matches_previous"] = bool(
                round(res["weighted"]["f1"], 4) == old["w_f1"]
                and round(res["class1"]["f1"], 4) == old["class1_f1"]
                and fp == int(old["false_pos"])
                and fn == int(old["false_neg"])
            )
        results[name] = res

    boot = bootstrap_f1(y, preds, seed)
    for name in results:
        results[name]["ci95"] = {k: ci(v) for k, v in boot[name].items()}

    pairs = [
        (n, "real_train_reference")
        for n in EXPERIMENTS if n != "real_train_reference"
    ] + PAIRS
    diffs = []
    for a, b in pairs:
        lo, hi = ci(boot[a]["class1_f1"] - boot[b]["class1_f1"])
        point = results[a]["class1"]["f1"] - results[b]["class1"]["f1"]
        diffs.append({
            "a": a, "b": b,
            "class1_f1_diff": round(point, 4),
            "ci95": [lo, hi],
            "excludes_zero": bool(lo > 0 or hi < 0),
        })

    report = {
        "note": "Bootstrap CIs reflect real-test sampling noise only. "
                "Generator training variance (single seed) is not captured.",
        "seed": seed,
        "n_boot": N_BOOT,
        "xgb_params": params,
        "versions": {
            "xgboost": xgb.__version__,
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "results": results,
        "paired_class1_f1_differences": diffs,
    }
    out_path = ROOT / "reports" / "classification_reproduction.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    header = (f"{'experiment':24s} {'wF1':>7s} {'c1F1':>7s} "
              f"{'c1F1 95% CI':>15s} {'ROC':>6s} {'PR':>6s}  matches_prev")
    print(header)
    for name, r in results.items():
        lo, hi = r["ci95"]["class1_f1"]
        print(f"{name:24s} {r['weighted']['f1']:7.4f} {r['class1']['f1']:7.4f} "
              f"[{lo:.3f},{hi:.3f}] {r['roc_auc']:6.3f} {r['pr_auc']:6.3f}  "
              f"{r['matches_previous']}")
    print("\nPaired class-1 F1 differences (a - b):")
    for d in diffs:
        print(f"  {d['a']:22s} - {d['b']:22s} {d['class1_f1_diff']:+.4f} "
              f"{d['ci95']} excludes_zero={d['excludes_zero']}")
    print(f"\nWrote {out_path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()