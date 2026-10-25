import warnings
import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from calibre.components.vid import build_feature_table # type: ignore
from calibre.models.risk_model_training import model_features, severe_bleed_amt_ml, ridge_model # type: ignore

matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
warnings.filterwarnings("ignore")

checkpoint_secs = [15, 30, 45, 60, 90, 120, 150, 180]
reshuffles = 5


def failure_classifier():
    return make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=10, max_iter=2000))


def auc_on_unseen_surgeons(table, feature_names, question):
    inputs = table[feature_names].values
    surgeon_ids = table["participant_id"].values
    blood_loss = table["blood_loss"].values
    answers = blood_loss > severe_bleed_amt_ml if question == "severe" else table["success"].values == 0
    aucs = []

    for reshuffle in range(reshuffles):
        risk_scores = np.zeros(len(table))
        folds = GroupKFold(5, shuffle=True, random_state=reshuffle)
       
        for train_rows, test_rows in folds.split(inputs, answers, surgeon_ids):
            if question == "severe":
                model = ridge_model().fit(inputs[train_rows], np.log(blood_loss[train_rows]))
                risk_scores[test_rows] = model.predict(inputs[test_rows])
            else:
                model = failure_classifier().fit(inputs[train_rows], answers[train_rows])
                risk_scores[test_rows] = model.predict_proba(inputs[test_rows])[:, 1]
        aucs.append(roc_auc_score(answers, risk_scores))
    
    return np.mean(aucs)


def add_surgeon_ids(table):
    surgeons = pd.read_csv("data/socal_outcomes.csv")[["trial_id", "participant_id"]]
    
    return table.merge(surgeons, on="trial_id")


def experience_baseline():
    outcomes = pd.read_csv("data/socal_outcomes.csv")
    surgeons = pd.read_csv("data/socal_participant_demographics.csv")
    
    table = outcomes.merge(surgeons, on="participant_id")
    table["is_attending"] = (table["training_status"] == "Attending").astype(float)
    table = table.dropna(subset=["total_years_experience"])
    
    resume_features = ["total_years_experience", "is_attending"]
    
    return (len(table), auc_on_unseen_surgeons(table, resume_features, "severe"),
            auc_on_unseen_surgeons(table, resume_features, "failure"))


if __name__ == "__main__":
    rows = []
    for seconds in checkpoint_secs:
        table = add_surgeon_ids(build_feature_table(watch_seconds=seconds))
        still_running = table[table["video_seconds"] >= seconds]
        
        rows.append({
            "seconds": seconds,
            "trials": len(still_running),
            "severe_auc": auc_on_unseen_surgeons(still_running, model_features, "severe"),
            "failure_auc": auc_on_unseen_surgeons(still_running, model_features, "failure"),
        })
    
    results = pd.DataFrame(rows)
    results.to_csv("models/live_accuracy.csv", index=False)

    trial_count, experience_severe_auc, experience_failure_auc = experience_baseline()
   

   #chart for slideshow use if needed
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(results["seconds"], results["severe_auc"], "o-", lw=2.5, label="Predicting a severe bleed (>500 mL)")
    ax.plot(results["seconds"], results["failure_auc"], "s-", lw=2.5, label="Predicting the bleed won't be controlled")
    ax.axhline(experience_severe_auc, color="gray", ls="--", lw=1.5, label="Surgeon's experience alone")
    ax.axhline(0.5, color="lightgray", ls=":", lw=1.5, label="Coin flip")
    ax.set_xlabel("Seconds of video CALIBRE has watched")
    ax.set_ylabel("AUC (tested on unseen surgeons)")
    ax.set_title("CALIBRE's warning is strongest in the first 45-90 seconds")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("models/live_accuracy.png", dpi=200)

    print(results.round(2).to_string(index=False))
    print("experience only auc:", round(experience_severe_auc, 2))
