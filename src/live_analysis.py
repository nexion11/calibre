"""Step 5: when is BleedWatch's prediction most useful?

Run it with:  uv run python src/live_analysis.py

SOCALNet made one prediction after 60 seconds. BleedWatch updates live, so
we test it at many moments: after 15 s, 30 s, ... of video. At each moment
the model may only use frames up to that second.

Fairness rule: at time t we only score trials whose video is still running.
A video that already ended tells you the bleeding is over, which would be cheating.

Also answers: does the VIDEO tell us more than the surgeon's experience does?
"""
import warnings

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_feature_table
from train import FEATURES, SEVERE, make_ridge

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

TIMES = [15, 30, 45, 60, 90, 120, 150, 180]
REPEATS = 5


def make_classifier():
    return make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=10, max_iter=2000))


def grouped_auc(table, features, target):
    """Average AUC over repeated surgeon-grouped 5-fold CV.
    target="severe": rank by predicted blood loss (ridge on log mL).
    target="failure": chance the bleed is NOT controlled in 5 min (logistic)."""
    X = table[features].values
    groups = table["participant_id"].values
    y_ml = table["blood_loss"].values
    labels = y_ml > SEVERE if target == "severe" else table["success"].values == 0
    aucs = []
    for repeat in range(REPEATS):
        scores = np.zeros(len(table))
        for train_idx, test_idx in GroupKFold(5, shuffle=True, random_state=repeat).split(X, labels, groups):
            if target == "severe":
                model = make_ridge().fit(X[train_idx], np.log(y_ml[train_idx]))
                scores[test_idx] = model.predict(X[test_idx])
            else:
                model = make_classifier().fit(X[train_idx], labels[train_idx])
                scores[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(labels, scores))
    return np.mean(aucs)


def with_participants(table):
    outcomes = pd.read_csv("data/socal_outcomes.csv")[["trial_id", "participant_id"]]
    return table.merge(outcomes, on="trial_id")


def experience_baseline():
    """Predict from the surgeon's resume alone: years of experience and attending vs trainee."""
    outcomes = pd.read_csv("data/socal_outcomes.csv")
    people = pd.read_csv("data/socal_participant_demographics.csv")
    table = outcomes.merge(people, on="participant_id")
    table["is_attending"] = (table["training_status"] == "Attending").astype(float)
    table = table.dropna(subset=["total_years_experience"])
    features = ["total_years_experience", "is_attending"]
    return len(table), grouped_auc(table, features, "severe"), grouped_auc(table, features, "failure")


if __name__ == "__main__":
    rows = []
    for t in TIMES:
        table = with_participants(build_feature_table(window=t))
        still_running = table[table["video_seconds"] >= t]
        rows.append({
            "seconds": t,
            "trials": len(still_running),
            "severe_auc": grouped_auc(still_running, FEATURES, "severe"),
            "failure_auc": grouped_auc(still_running, FEATURES, "failure"),
        })
        print(f"t={t:3d}s  trials={rows[-1]['trials']:3d}  "
              f"severe-bleed AUC={rows[-1]['severe_auc']:.2f}  failure AUC={rows[-1]['failure_auc']:.2f}")
    results = pd.DataFrame(rows)
    results.to_csv("models/live_accuracy.csv", index=False)

    n, exp_severe, exp_failure = experience_baseline()
    print(f"\nSurgeon experience only ({n} trials): severe-bleed AUC={exp_severe:.2f}  failure AUC={exp_failure:.2f}")

    # Chart for the demo video
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(results["seconds"], results["severe_auc"], "o-", lw=2.5, label="Predicting a severe bleed (>500 mL)")
    ax.plot(results["seconds"], results["failure_auc"], "s-", lw=2.5, label="Predicting the bleed won't be controlled")
    ax.axhline(exp_severe, color="gray", ls="--", lw=1.5, label="Surgeon's experience alone")
    ax.axhline(0.5, color="lightgray", ls=":", lw=1.5, label="Coin flip")
    ax.set_xlabel("Seconds of video BleedWatch has watched")
    ax.set_ylabel("AUC (tested on unseen surgeons)")
    ax.set_title("BleedWatch's warning is strongest in the first 45-90 seconds")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("models/live_accuracy.png", dpi=200)
    print("Saved models/live_accuracy.png")
