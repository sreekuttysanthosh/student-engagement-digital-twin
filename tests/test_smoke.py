import numpy as np
import pandas as pd

from engagement_twin.classifier import build_model, evaluate, train
from engagement_twin.config import load_config
from engagement_twin.preprocessing import build_generator_inputs, restore_random

FEATURES = ["nevents", "ndays_act", "ncontent", "nforum_posts"]
TARGET = "Engagement_Level"


def test_pipeline_smoke_without_generator_training():
    """Impute -> restore NaNs -> train XGBoost -> evaluate, on tiny toy data."""
    cfg = load_config()
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.integers(1, 100, size=(300, 4)), columns=FEATURES).astype(float)
    df.loc[::4, "nforum_posts"] = np.nan
    df[TARGET] = (df["ncontent"] > 50).astype(int)

    tables, _ = build_generator_inputs(df, FEATURES, TARGET, cfg["imputer_k"])
    fake = tables["random_target"]  # stands in for generator output
    rates = df[FEATURES].isna().mean()
    restored = restore_random(fake, rates, FEATURES, cfg["seed"], target=TARGET)

    model = train(build_model(cfg["xgboost"], cfg["seed"]), restored, FEATURES, TARGET)
    result = evaluate(model, df, FEATURES, TARGET)
    assert sum(map(sum, result["confusion_matrix"])) == 300