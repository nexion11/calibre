"""Train BleedWatch's blood-loss predictor, test it fairly, and save it.

Run it with:  uv run python src/train.py

The model choice (ridge regression on log blood loss, tools + blood features)
came from model_search.py, which compared 40 options with nested CV.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_feature_table

FIRST_FEATURES = ["suction_frac", "cottonoid_frac", "tools_per_frame"]
FEATURES = FIRST_FEATURES + ["blood_mean", "blood_max", "blood_trend", "blood_late"]
SEVERE = 500  # mL; a "severe bleed" in the SOCAL papers
TARGET_CATCH = 0.75  # alert threshold is set to catch about 3 in 4 severe bleeds


def load_table():
    table = build_feature_table()
    participants = pd.read_csv("data/socal_outcomes.csv")[["trial_id", "participant_id"]]
    return table.merge(participants, on="trial_id")


def make_ridge():
    # StandardScaler puts every feature on the same scale so ridge can shrink
    # them fairly; RidgeCV picks how much to shrink using the training data only.
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))


def cross_validate(table, kind, features):
    """Every trial gets a prediction from a model that never saw its surgeon."""
    X = table[features].values
    y = table["blood_loss"].values
    groups = table["participant_id"].values
    predictions = np.zeros(len(y))

    for train_idx, test_idx in GroupKFold(n_splits=5).split(X, y, groups):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[test_idx]

        if kind == "baseline":
            predictions[test_idx] = y_train.mean()
        elif kind == "linear":
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions[test_idx] = model.predict(X_test)
        else:
            # Predict log(blood loss) so a few huge bleeds don't dominate the fit.
            model = make_ridge()
            model.fit(X_train, np.log(y_train))
            predictions[test_idx] = np.exp(model.predict(X_test))

    return y, predictions


def report(name, y, predictions, threshold=SEVERE):
    severe = y > SEVERE
    caught = (predictions[severe] > threshold).sum()
    false_alarms = (predictions[~severe] > threshold).sum()
    print(f"{name}")
    print(f"  Average error:        {np.abs(predictions - y).mean():.0f} mL")
    if predictions.std() > 0:
        print(f"  Severe-bleed AUC:     {roc_auc_score(severe, predictions):.2f}  (0.5 = coin flip, 1 = perfect)")
    print(f"  Severe bleeds caught: {caught} of {severe.sum()} ({caught / severe.sum():.0%})")
    print(f"  False alarms:         {false_alarms} of {(~severe).sum()} ({false_alarms / (~severe).sum():.0%})\n")


def save_model(table, threshold):
    """Fit on all trials and save plain numbers the iPhone app can use:
    prediction = exp(intercept + sum(weight * (feature - mean) / scale))."""
    model = make_ridge()
    model.fit(table[FEATURES].values, np.log(table["blood_loss"].values))
    scaler, ridge = model.named_steps["standardscaler"], model.named_steps["ridgecv"]
    os.makedirs("models", exist_ok=True)
    with open("models/bleedwatch_model.json", "w") as f:
        json.dump({
            "features": FEATURES,
            "means": scaler.mean_.tolist(),
            "scales": scaler.scale_.tolist(),
            "weights": ridge.coef_.tolist(),
            "intercept": float(ridge.intercept_),
            "alert_threshold_mL": float(threshold),
            "window_seconds": 60,
        }, f, indent=2)
    print("Saved models/bleedwatch_model.json")
    print("Feature weights (bigger = pushes predicted blood loss up):")
    for name, weight in sorted(zip(FEATURES, ridge.coef_), key=lambda p: -abs(p[1])):
        print(f"  {name:16s} {weight:+.2f}")


if __name__ == "__main__":
    table = load_table()
    print(f"Training on {len(table)} trials from {table['participant_id'].nunique()} surgeons\n")
    report("BASELINE (always guess the average)", *cross_validate(table, "baseline", FIRST_FEATURES))
    report("FIRST MODEL (linear, 3 tool features)", *cross_validate(table, "linear", FIRST_FEATURES))

    y, predictions = cross_validate(table, "ridge", FEATURES)
    # Log predictions run low, so ">500 mL" would miss most bleeds. Instead,
    # set the alert line so it catches TARGET_CATCH of the severe bleeds.
    threshold = np.quantile(predictions[y > SEVERE], 1 - TARGET_CATCH)
    report(f"BLEEDWATCH (ridge, tools + blood, alert above {threshold:.0f} mL predicted)",
           y, predictions, threshold)
    save_model(table, threshold)
