# One fixed set of test rules for every method we try, so we can't fool ourselves:
#  - whole surgeons are held out (a surgeon is never in both training and test)
#  - 10 different random surgeon splits, averaged
#  - 95% confidence ranges from resampling surgeons (bootstrap), not trials
# A method is a function (train_table, test_table) -> predictions for the test rows.
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from calibre.components.vid import build_feature_table, human_boxes_file
from calibre.models.risk_model_training import ridge_model

warnings.filterwarnings("ignore")
severe_ml = 500
reshuffles = 10
bootstrap_rounds = 2000


def load_table(boxes_file=human_boxes_file, watch_seconds=60):
    table = build_feature_table(watch_seconds=watch_seconds, boxes_file=boxes_file)
    surgeons = pd.read_csv("data/socal_outcomes.csv")[["trial_id", "participant_id"]]
    return table.merge(surgeons, on="trial_id").sort_values("trial_id").reset_index(drop=True)


def ridge_on_log(features):
    def fit_predict(train, test):
        model = ridge_model().fit(train[features].values, np.log(train["blood_loss"].values))
        return np.exp(model.predict(test[features].values))
    return fit_predict


def failure_classifier(features):
    def fit_predict(train, test):
        model = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=10, max_iter=2000))
        model.fit(train[features].values, train["success"].values == 0)
        return model.predict_proba(test[features].values)[:, 1]
    return fit_predict


def out_of_fold(table, fit_predict, reshuffle):
    predictions = np.zeros(len(table))
    folds = GroupKFold(n_splits=5, shuffle=True, random_state=reshuffle)
    for train_rows, test_rows in folds.split(table, groups=table["participant_id"]):
        predictions[test_rows] = fit_predict(table.iloc[train_rows], table.iloc[test_rows])
    return predictions


def scores(table, predicted_ml, failure_risk):
    blood_loss = table["blood_loss"].values
    is_severe = blood_loss > severe_ml
    alert_line = np.quantile(predicted_ml[is_severe], 0.25)  # the line that catches 75% of severe bleeds
    return {
        "severe_auc": roc_auc_score(is_severe, predicted_ml),
        "failure_auc": roc_auc_score(table["success"].values == 0, failure_risk),
        "false_alarms_at_75pct_caught": (predicted_ml[~is_severe] > alert_line).mean(),
        "log_r2": r2_score(np.log(blood_loss), np.log(predicted_ml)),
        "spearman": spearmanr(blood_loss, predicted_ml)[0],
        "median_ml_error": np.median(np.abs(predicted_ml - blood_loss)),
        "r2_ml": r2_score(blood_loss, predicted_ml),  # in plain mL, the way SOCALNet's error is reported
        "rmse_ml": np.sqrt(np.mean((predicted_ml - blood_loss) ** 2)),
    }


def surgeon_bootstrap(table, predicted_ml, failure_risk, seed=0):
    rng = np.random.default_rng(seed)
    rows_by_surgeon = list(table.groupby("participant_id").indices.values())
    samples = []
    for _ in range(bootstrap_rounds):
        picks = rng.integers(len(rows_by_surgeon), size=len(rows_by_surgeon))
        rows = np.concatenate([rows_by_surgeon[i] for i in picks])
        sample = table.iloc[rows]
        if (sample["blood_loss"] > severe_ml).nunique() < 2 or sample["success"].nunique() < 2:
            continue
        samples.append(scores(sample, predicted_ml[rows], failure_risk[rows]))
    return pd.DataFrame(samples).quantile([0.025, 0.975])


def evaluate(table, predict_ml, predict_failure, name=""):
    all_log_ml, all_failure, per_split = [], [], []
    for reshuffle in range(reshuffles):
        ml = out_of_fold(table, predict_ml, reshuffle)
        failure = out_of_fold(table, predict_failure, reshuffle)
        all_log_ml.append(np.log(ml))
        all_failure.append(failure)
        per_split.append(scores(table, ml, failure))
    averaged_ml, averaged_failure = np.exp(np.mean(all_log_ml, axis=0)), np.mean(all_failure, axis=0)
    ranges = surgeon_bootstrap(table, averaged_ml, averaged_failure)
    summary = pd.DataFrame({"score": pd.DataFrame(per_split).mean(),
                            "low": ranges.loc[0.025], "high": ranges.loc[0.975]})
    summary.attrs.update(name=name, trials=len(table), predicted_ml=averaged_ml, failure_risk=averaged_failure)
    return summary


def paired_difference(table, new, old, metric="severe_auc", seed=1):
    """Is `new` really better than `old`? Resample surgeons and look at the gap each time."""
    rng = np.random.default_rng(seed)
    rows_by_surgeon = list(table.groupby("participant_id").indices.values())
    gaps = []
    for _ in range(bootstrap_rounds):
        rows = np.concatenate([rows_by_surgeon[i] for i in rng.integers(len(rows_by_surgeon), size=len(rows_by_surgeon))])
        sample = table.iloc[rows]
        if (sample["blood_loss"] > severe_ml).nunique() < 2 or sample["success"].nunique() < 2:
            continue
        new_score = scores(sample, new.attrs["predicted_ml"][rows], new.attrs["failure_risk"][rows])[metric]
        old_score = scores(sample, old.attrs["predicted_ml"][rows], old.attrs["failure_risk"][rows])[metric]
        gaps.append(new_score - old_score)
    gaps = np.array(gaps)
    full_new = scores(table, new.attrs["predicted_ml"], new.attrs["failure_risk"])[metric]
    full_old = scores(table, old.attrs["predicted_ml"], old.attrs["failure_risk"])[metric]
    return {"gap": full_new - full_old,
            "low": np.quantile(gaps, 0.025), "high": np.quantile(gaps, 0.975),
            "chance_new_is_better": (gaps > 0).mean()}


def paired_differences(table, new, old, seed=1):
    """paired_difference for every score at once. Same resampled surgeons, so the numbers match it."""
    rng = np.random.default_rng(seed)
    rows_by_surgeon = list(table.groupby("participant_id").indices.values())
    gaps = []
    for _ in range(bootstrap_rounds):
        rows = np.concatenate([rows_by_surgeon[i] for i in rng.integers(len(rows_by_surgeon), size=len(rows_by_surgeon))])
        sample = table.iloc[rows]
        if (sample["blood_loss"] > severe_ml).nunique() < 2 or sample["success"].nunique() < 2:
            continue
        new_scores = pd.Series(scores(sample, new.attrs["predicted_ml"][rows], new.attrs["failure_risk"][rows]))
        old_scores = pd.Series(scores(sample, old.attrs["predicted_ml"][rows], old.attrs["failure_risk"][rows]))
        gaps.append(new_scores - old_scores)
    gaps = pd.DataFrame(gaps)
    full = pd.Series(scores(table, new.attrs["predicted_ml"], new.attrs["failure_risk"])) \
        - pd.Series(scores(table, old.attrs["predicted_ml"], old.attrs["failure_risk"]))
    return pd.DataFrame({"gap": full, "low": gaps.quantile(0.025), "high": gaps.quantile(0.975)})


def show(summary):
    print(f"\n{summary.attrs['name']}  ({summary.attrs['trials']} trials)")
    for metric, row in summary.iterrows():
        print(f"  {metric:30s} {row['score']:7.3f}   [{row['low']:.3f} to {row['high']:.3f}]")


if __name__ == "__main__":
    from calibre.models.risk_model_training import model_features
    table = load_table()
    show(evaluate(table, ridge_on_log(model_features), failure_classifier(model_features),
                  "CALIBRE today (human tool labels)"))


def evaluate_leave_one_site_out(table, predict_ml, predict_failure, name=""):
    """Train on every SOCAL course site but one, test on the one left out. Repeated for each site."""
    predicted_ml, failure_risk = np.zeros(len(table)), np.zeros(len(table))
    for site in table["cohort"].unique():
        test_rows = np.where(table["cohort"] == site)[0]
        train_rows = np.where(table["cohort"] != site)[0]
        predicted_ml[test_rows] = predict_ml(table.iloc[train_rows], table.iloc[test_rows])
        failure_risk[test_rows] = predict_failure(table.iloc[train_rows], table.iloc[test_rows])
    ranges = surgeon_bootstrap(table, predicted_ml, failure_risk)
    summary = pd.DataFrame({"score": pd.Series(scores(table, predicted_ml, failure_risk)),
                            "low": ranges.loc[0.025], "high": ranges.loc[0.975]})
    summary.attrs.update(name=name, trials=len(table), predicted_ml=predicted_ml, failure_risk=failure_risk)
    return summary
