# Runs every experiment under the same rules and writes the numbers to results/.
#   uv run python -m calibre.experiments.run_experiments human      (SOCAL's hand-drawn tool boxes)
#   uv run python -m calibre.experiments.run_experiments auto       (our detector's boxes, no human labels)
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from calibre.components.vid import auto_boxes_file, human_boxes_file
from calibre.experiments import harness
from calibre.experiments.harness import (severe_ml, evaluate, evaluate_leave_one_site_out, failure_classifier, load_table,
                                         out_of_fold, paired_difference, ridge_on_log, scores)
from calibre.experiments.image_models import (SocalnetStyle, embedding_failure, embedding_regressor,
                                              sequence_inputs, trial_embedding_summary)
from calibre.experiments.new_features import add_new_features, new_feature_names
from calibre.experiments.new_models import (add_outcome_times, add_surgeon_features, survival_failure_risk, survival_rate_times_time, surgeon_feature_names,
                                            with_missing_filled)
from calibre.models.risk_model_training import model_features as original_features

warnings.filterwarnings("ignore")
video_features = original_features + new_feature_names
last_attempt = ["second_attempt", "last_attempt_log_blood_loss", "last_attempt_controlled"]


def build_table(boxes_file, watch_seconds=60):
    table = load_table(boxes_file, watch_seconds)
    table = add_new_features(table, boxes_file, watch_seconds)
    return add_surgeon_features(add_outcome_times(table))


def tabular_methods():
    return {
        "A. Original CALIBRE (7 features)": (ridge_on_log(original_features), failure_classifier(original_features)),
        "B. + 10 new bleed-handling features": (ridge_on_log(video_features), failure_classifier(video_features)),
        "C. B with survival (rate x time) model": (survival_rate_times_time(video_features),
                                                   survival_failure_risk(video_features)),
        "D. B + surgeon's previous attempt": (with_missing_filled(ridge_on_log, video_features + last_attempt),
                                              with_missing_filled(failure_classifier, video_features + last_attempt)),
        "E. B + all surgeon information": (
            with_missing_filled(ridge_on_log, video_features + surgeon_feature_names),
            with_missing_filled(failure_classifier, video_features + surgeon_feature_names)),
    }


def image_methods(boxes_file):
    methods = {}
    if os.path.exists("data/embeddings_dinov2.npz"):
        dino = trial_embedding_summary("dinov2")
        methods["F. DINOv2 frame features only"] = (embedding_regressor(dino), embedding_failure(dino))
        methods["G. DINOv2 + B"] = (embedding_regressor(dino, video_features), embedding_failure(dino, video_features))
    if os.path.exists("data/embeddings_resnet50.npz"):
        socalnet = SocalnetStyle(sequence_inputs(boxes_file))
        methods["H. SOCALNet-style ResNet + LSTM (our re-build)"] = (socalnet.predict_ml, socalnet.predict_failure)
    return methods


def main_comparison(table, boxes_file):
    rows, results = [], {}
    methods = {**tabular_methods(), **image_methods(boxes_file)}
    for name, (predict_ml, predict_failure) in methods.items():
        print("running", name, flush=True)
        results[name] = evaluate(table, predict_ml, predict_failure, name)
    baseline = results["A. Original CALIBRE (7 features)"]
    for name, summary in results.items():
        row = {"method": name}
        for metric, values in summary.iterrows():
            row[metric], row[f"{metric}_low"], row[f"{metric}_high"] = values["score"], values["low"], values["high"]
        for metric in ["severe_auc", "failure_auc"]:
            gap = paired_difference(table, summary, baseline, metric)
            row[f"{metric}_gap_vs_A"] = gap["gap"]
            row[f"{metric}_gap_low"], row[f"{metric}_gap_high"] = gap["low"], gap["high"]
        rows.append(row)
    return pd.DataFrame(rows)


def site_test(table):
    rows = []
    for name, (predict_ml, predict_failure) in tabular_methods().items():
        summary = evaluate_leave_one_site_out(table, predict_ml, predict_failure, name)
        rows.append({"method": name, **{m: summary.loc[m, "score"] for m in ["severe_auc", "failure_auc"]},
                     "severe_auc_low": summary.loc["severe_auc", "low"], "severe_auc_high": summary.loc["severe_auc", "high"]})
    return pd.DataFrame(rows)


def accuracy_over_time(boxes_file):
    harness.bootstrap_rounds, rows = 300, []
    keep = ["A. Original CALIBRE (7 features)", "B. + 10 new bleed-handling features", "D. B + surgeon's previous attempt"]
    for seconds in [15, 30, 45, 60, 90, 120]:
        table = build_table(boxes_file, seconds)
        table = table[table["video_seconds"] >= seconds].reset_index(drop=True)
        for name, (predict_ml, predict_failure) in tabular_methods().items():
            if name in keep:
                summary = evaluate(table, predict_ml, predict_failure, name)
                rows.append({"seconds": seconds, "trials": len(table), "method": name,
                             "severe_auc": summary.loc["severe_auc", "score"],
                             "failure_auc": summary.loc["failure_auc", "score"]})
    harness.bootstrap_rounds = 2000
    return pd.DataFrame(rows)


def nested_best_of(table):
    """Honest score for 'try every tabular method, keep the best': the choice is made inside each training set."""
    methods = tabular_methods()
    predicted_ml, failure_risk, winners = np.zeros(len(table)), np.zeros(len(table)), []
    outer = GroupKFold(n_splits=5, shuffle=True, random_state=99)
    for train_rows, test_rows in outer.split(table, groups=table["participant_id"]):
        inner = table.iloc[train_rows].reset_index(drop=True)

        def inner_score(name):
            predict_ml, predict_failure = methods[name]
            return np.mean([scores(inner, out_of_fold(inner, predict_ml, r), out_of_fold(inner, predict_failure, r))["severe_auc"]
                            for r in range(3)])
        best = max(methods, key=inner_score)
        winners.append(best)
        predict_ml, predict_failure = methods[best]
        predicted_ml[test_rows] = predict_ml(table.iloc[train_rows], table.iloc[test_rows])
        failure_risk[test_rows] = predict_failure(table.iloc[train_rows], table.iloc[test_rows])
    return pd.DataFrame([{**scores(table, predicted_ml, failure_risk), "winners": " | ".join(winners)}])


def twenty_video_spread(table, repeats=500):
    """How much can a score on 20 random test videos (SOCALNet's test size) swing by luck alone?"""
    predict_failure = failure_classifier(video_features)
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(repeats):
        test_rows = rng.choice(len(table), size=20, replace=False)
        train_rows = np.setdiff1d(np.arange(len(table)), test_rows)
        test = table.iloc[test_rows]
        risk = predict_failure(table.iloc[train_rows], test)
        rows.append({"accuracy_on_20": ((risk > 0.5) == (test["success"].values == 0)).mean()})
    return pd.DataFrame(rows)

def socalnet_protocol(table, boxes_file, repeats=200, deep_repeats=40):
    """Score our models the way SOCALNet was scored: 20 random test videos, the rest for training,
    split by video (the same surgeon can be in both). Repeated many times so we see the spread."""
    from sklearn.metrics import r2_score
    methods = {**tabular_methods(), **image_methods(boxes_file)}
    keep = ["A. Original CALIBRE (7 features)", "B. + 10 new bleed-handling features",
            "D. B + surgeon's previous attempt", "G. DINOv2 + B", "H. SOCALNet-style ResNet + LSTM (our re-build)"]
    rng = np.random.default_rng(0)
    splits = [rng.choice(len(table), size=20, replace=False) for _ in range(repeats)]
    rows = []
    for name in keep:
        predict_ml, predict_failure = methods[name]
        for split_number, test_rows in enumerate(splits[: deep_repeats if name.startswith("H") else repeats]):
            train_rows = np.setdiff1d(np.arange(len(table)), test_rows)
            train, test = table.iloc[train_rows], table.iloc[test_rows]
            ml, failure = predict_ml(train, test), predict_failure(train, test)
            blood = test["blood_loss"].values
            severe = blood > severe_ml
            rows.append({"method": name, "split": split_number,
                         "success_accuracy": ((failure > 0.5) == (test["success"].values == 0)).mean(),
                         "r2_ml": r2_score(blood, ml),
                         "rmse_ml": np.sqrt(np.mean((ml - blood) ** 2)),
                         "over_500_caught": (ml[severe] > severe_ml).mean() if severe.any() else np.nan})
        print("socalnet protocol done:", name, flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "human"
    parts = sys.argv[2:] or ["main", "site", "time", "nested", "twenty"]
    boxes_file = human_boxes_file if which == "human" else auto_boxes_file
    os.makedirs("results", exist_ok=True)
    table = build_table(boxes_file)
    print(f"{which} boxes: {len(table)} trials, {table['participant_id'].nunique()} surgeons")
    if "main" in parts:
        main_comparison(table, boxes_file).to_csv(f"results/{which}_main.csv", index=False)
    if "site" in parts:
        site_test(table).to_csv(f"results/{which}_site_out.csv", index=False)
    if "time" in parts:
        accuracy_over_time(boxes_file).to_csv(f"results/{which}_over_time.csv", index=False)
    if "nested" in parts:
        nested_best_of(table).to_csv(f"results/{which}_nested.csv", index=False)
    if "twenty" in parts:
        spread = twenty_video_spread(table)
        spread.describe(percentiles=[0.05, 0.5, 0.95]).to_csv(f"results/{which}_twenty_video_spread.csv")
    if "socalnet_protocol" in parts:
        socalnet_protocol(table, boxes_file).to_csv(f"results/{which}_socalnet_protocol.csv", index=False)
    print("done")
