import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from calibre.components.vid import watch_seconds, build_feature_table


tool_features = ["suction_amt", "cottonoid_amt", "tools_per_frame"]
blood_features = ["blood_mean", "blood_max", "blood_trend", "blood_late"]
model_features = tool_features + blood_features
severe_bleed_amt_ml = 500  # severe bleed is 500 ml for socal
severe_catch_rate = 0.75  


def load_training_table():
    feature_table = build_feature_table()
    surgeons = pd.read_csv("data/socal_outcomes.csv")[["trial_id", "participant_id"]]

    return feature_table.merge(surgeons, on="trial_id")


def ridge_model():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))


def unseen_surgeon_tests(training_table, model_type, feature_names):
    inputs = training_table[feature_names].values
    blood_loss = training_table["blood_loss"].values
    surgeon_ids = training_table["participant_id"].values
    predictions = np.zeros(len(blood_loss))

    for train_rows, test_rows in GroupKFold(n_splits=5).split(inputs, blood_loss, surgeon_ids):
        if model_type == "baseline":
            predictions[test_rows] = blood_loss[train_rows].mean()
        
        elif model_type == "linear":
            model = LinearRegression()
            model.fit(inputs[train_rows], blood_loss[train_rows])
            predictions[test_rows] = model.predict(inputs[test_rows])
        
        else:
            model = ridge_model()
            model.fit(inputs[train_rows], np.log(blood_loss[train_rows]))
            predictions[test_rows] = np.exp(model.predict(inputs[test_rows]))

    return blood_loss, predictions


def report(title, blood_loss, predictions, alert_line_ml=severe_bleed_amt_ml):
    is_severe = blood_loss > severe_bleed_amt_ml
    severe_caught = (predictions[is_severe] > alert_line_ml).sum()
    false_alarms = (predictions[~is_severe] > alert_line_ml).sum()
    
    print(f"{title}")
    
    print(f"avg error:{np.abs(predictions - blood_loss).mean():.0f} mL")
   
    if predictions.std() > 0:
        print(f"severe bleed auc: {roc_auc_score(is_severe, predictions):.2f}")
    print(f"caught severe bleeds: {severe_caught} of {is_severe.sum()} ({severe_caught / is_severe.sum():.0%})")
    print(f" false alarms:{false_alarms} of {(~is_severe).sum()} ({false_alarms / (~is_severe).sum():.0%})\n")


def save_model(training_table, alert_line_ml):
    model = ridge_model()
    model.fit(training_table[model_features].values, np.log(training_table["blood_loss"].values))
    
    scaler, ridge = model.named_steps["standardscaler"], model.named_steps["ridgecv"]
    os.makedirs("models", exist_ok=True)
    
    with open("models/calibre_risk_model.json", "w") as f:
        
        json.dump({
            "features": model_features,
            "means": scaler.mean_.tolist(),
            "scales": scaler.scale_.tolist(),
            "weights": ridge.coef_.tolist(),
            "intercept": float(ridge.intercept_),
            "alert_threshold_mL": float(alert_line_ml),
            "window_seconds": watch_seconds,
        }, f, indent=2)
    
    print("feature weights:")
    
    for name, weight in sorted(zip(model_features, ridge.coef_), key=lambda pair: -abs(pair[1])):
        print(f"{name:16s} {weight:+.2f}")


if __name__ == "__main__":
    training_table = load_training_table()
    report("base:",
                 *unseen_surgeon_tests(training_table, "baseline", tool_features))
    report("model v1:",
                 *unseen_surgeon_tests(training_table, "linear", tool_features))

    blood_loss, predictions = unseen_surgeon_tests(training_table, "ridge", model_features)
    
    alert_line_ml = np.quantile(predictions[blood_loss > severe_bleed_amt_ml], 1 - severe_catch_rate)
    
    report("calibre:",
                 blood_loss, predictions, alert_line_ml)
    save_model(training_table, alert_line_ml)
