"""Step 4: find the best model WITHOUT fooling ourselves.

Run it with:  uv run python src/model_search.py

With only ~143 trials it's easy to "overfit": try enough models and one
will look great on this data just by luck. Three guards against that:

1. Repeated cross-validation. We reshuffle which surgeons land in each of
   the 5 groups, 10 times over, and average. One lucky split can't fool us.
2. Simple, regularized models. Ridge regression shrinks weights so no single
   feature can dominate; tree models are kept shallow.
3. Nested cross-validation for the final number. Picking the best model is
   itself a kind of training, so we re-run the whole "pick the best" process
   inside each outer fold and score it on surgeons it never saw.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train import SEVERE, load_table

warnings.filterwarnings("ignore")

TOOLS = ["suction_frac", "cottonoid_frac", "tools_per_frame"]
BLOOD = ["blood_mean", "blood_max", "blood_trend", "blood_late"]
FEATURE_SETS = {
    "tools": TOOLS,
    "blood": BLOOD,
    "tools+blood_mean": TOOLS + ["blood_mean"],
    "tools+blood": TOOLS + BLOOD,
    "tools+blood+motion": TOOLS + BLOOD + ["suction_speed"],
}
MODELS = {
    "linear": lambda: LinearRegression(),
    "ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
    "forest": lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=8, n_jobs=-1, random_state=0),
    "boosting": lambda: GradientBoostingRegressor(max_depth=2, n_estimators=150,
                                                  learning_rate=0.05, subsample=0.8, random_state=0),
}
REPEATS = 10


def fit_predict(model_name, log_target, X_train, y_train, X_test):
    """Train one model. log_target=True predicts log(blood loss), which
    stops a few huge bleeds (up to 1,890 mL) from dominating the fit."""
    model = MODELS[model_name]()
    if log_target:
        model.fit(X_train, np.log(y_train))
        return np.exp(model.predict(X_test))
    model.fit(X_train, y_train)
    return model.predict(X_test)


def score(y, pred):
    severe = y > SEVERE
    return {
        "error_mL": np.abs(pred - y).mean(),
        # AUC: chance the model ranks a random severe bleed above a random
        # non-severe one. 0.5 = coin flip, 1.0 = perfect.
        "severe_auc": roc_auc_score(severe, pred),
        "caught": (pred[severe] > SEVERE).mean(),
        "false_alarms": (pred[~severe] > SEVERE).mean(),
    }


def repeated_cv(table, features, model_name, log_target, repeats=REPEATS):
    X, y = table[features].values, table["blood_loss"].values
    groups = table["participant_id"].values
    results = []
    for repeat in range(repeats):
        pred = np.zeros(len(y))
        folds = GroupKFold(n_splits=5, shuffle=True, random_state=repeat)
        for train_idx, test_idx in folds.split(X, y, groups):
            pred[test_idx] = fit_predict(model_name, log_target, X[train_idx], y[train_idx], X[test_idx])
        results.append(score(y, pred))
    return pd.DataFrame(results).mean()


def all_candidates():
    for set_name in FEATURE_SETS:
        for model_name in MODELS:
            for log_target in [False, True]:
                yield set_name, model_name, log_target


def nested_cv(table):
    """Honest estimate of the WHOLE process, including picking the winner."""
    y, groups = table["blood_loss"].values, table["participant_id"].values
    pred = np.zeros(len(y))
    chosen = []
    outer = GroupKFold(n_splits=5, shuffle=True, random_state=99)
    for train_idx, test_idx in outer.split(table, y, groups):
        inner_table = table.iloc[train_idx]
        best = min(all_candidates(), key=lambda c: -repeated_cv(inner_table, FEATURE_SETS[c[0]], c[1], c[2], repeats=3)["severe_auc"])
        chosen.append(best)
        features = FEATURE_SETS[best[0]]
        pred[test_idx] = fit_predict(best[1], best[2], table.iloc[train_idx][features].values,
                                     y[train_idx], table.iloc[test_idx][features].values)
    return score(y, pred), chosen


if __name__ == "__main__":
    table = load_table()
    print(f"{len(table)} trials, {table['participant_id'].nunique()} surgeons, {REPEATS}x repeated 5-fold CV\n")

    rows = []
    for set_name, model_name, log_target in all_candidates():
        result = repeated_cv(table, FEATURE_SETS[set_name], model_name, log_target)
        rows.append({"features": set_name, "model": model_name,
                     "target": "log" if log_target else "mL", **result})
    board = pd.DataFrame(rows).sort_values("severe_auc", ascending=False)
    pd.set_option("display.width", 140)
    print(board.round(3).to_string(index=False))

    print("\nNested CV (honest score for 'pick the best model' as a whole):")
    honest, chosen = nested_cv(table)
    for k, v in honest.items():
        print(f"  {k}: {v:.3f}")
    print("  winners picked in each outer fold:", chosen)
