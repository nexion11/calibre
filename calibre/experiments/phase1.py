# Phase 1 of docs/improvement_plan.md: a fairer comparison with SOCALNet, plus modelling changes that need no GPU.
#   uv run python -m calibre.experiments.phase1 [check targets held_out decisions everyone protocol]
#   CALIBRE_LABELS=auto uv run python -m ...   scores everything on our detector's tool labels instead of the hand-drawn ones
# Same rules as run_experiments: whole surgeons held out, 5 folds x 10 reshuffles, 95% ranges from resampling surgeons.
import os
import sys
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import GammaRegressor, HuberRegressor, LogisticRegressionCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from calibre.components.vid import boxes_files
from calibre.experiments import harness
from calibre.experiments.harness import evaluate, failure_classifier, paired_differences, scores, severe_ml, surgeon_bootstrap
from calibre.experiments.new_models import add_surgeon_features, surgeon_feature_names, with_missing_filled
from calibre.experiments.run_experiments import build_table, last_attempt, video_features
from calibre.models.risk_model_training import ridge_model

warnings.filterwarnings("ignore")
jobs = 8
labels = os.environ.get("CALIBRE_LABELS", "human")  # an environment setting, so the worker processes see it too
boxes_file = boxes_files[labels]
socalnet_r2, socalnet_rmse = 0.74, 295  # published: Pangal et al., Sci Rep 2022


# ---------------- 1.2 what the blood-loss model learns ----------------

targets = {  # name -> (mL to what the model learns, and back)
    "log": (np.log, np.exp),
    "sqrt": (np.sqrt, lambda learned: np.clip(learned, 0, None) ** 2),
    "ml": (lambda ml: ml, lambda learned: learned),
}


def ridge_on(target, corrected=False):
    """Ridge on log, square root or plain mL. Converting back from log gives a typical value, not the average,
    so `corrected` rescales the predictions until the training-set average comes out right."""
    to_target, to_ml = targets[target]

    def maker(features):
        def fit_predict(train, test):
            model = ridge_model().fit(train[features].values, to_target(train["blood_loss"].values))
            predicted = to_ml(model.predict(test[features].values))
            if corrected:
                predicted = predicted * train["blood_loss"].mean() / to_ml(model.predict(train[features].values)).mean()
            return np.clip(predicted, 1, None)
        return fit_predict
    return maker


def tuned(model, setting, values):
    """A model fitted straight on mL, with its penalty picked on the training surgeons only."""
    def maker(features):
        def fit_predict(train, test):
            search = GridSearchCV(make_pipeline(StandardScaler(), model), {f"{type(model).__name__.lower()}__{setting}": values},
                                  cv=GroupKFold(3), scoring="neg_root_mean_squared_error")
            search.fit(train[features].values, train["blood_loss"].values, groups=train["participant_id"].values)
            return np.clip(search.predict(test[features].values), 1, None)
        return fit_predict
    return maker


gamma = tuned(GammaRegressor(max_iter=1000), "alpha", [0.01, 0.03, 0.1, 0.3, 1, 3])
huber = tuned(HuberRegressor(max_iter=1000), "alpha", [0.1, 1, 10, 100, 1000])


# ---------------- 1.3 two-part blood model ----------------

def two_part(features):
    """Blood loss falls in two groups: bleeds that were controlled, and bleeds that ran the full 5 minutes.
    Expected loss = chance of failure x loss if it fails + chance of control x loss if it is controlled."""
    def fit_predict(train, test):
        risk = failure_classifier(features)(train, test)
        if_failed = ridge_on("sqrt")(features)(train[train["success"] == 0], test)
        if_controlled = ridge_on("sqrt")(features)(train[train["success"] == 1], test)
        return risk * if_failed + (1 - risk) * if_controlled
    return fit_predict


# ---------------- 1.4 yes/no calls with the cutoff picked inside the training set ----------------

def is_severe(table):
    return table["blood_loss"].values > severe_ml


def is_failure(table):
    return table["success"].values == 0


def severe_classifier(features):
    def fit_predict(train, test):
        model = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=10, max_iter=2000))
        model.fit(train[features].values, is_severe(train))
        return model.predict_proba(test[features].values)[:, 1]
    return fit_predict


def accuracy(truth, flagged):
    return (truth == flagged).mean()


def caught_minus_false_alarms(truth, flagged):
    return flagged[truth].mean() - flagged[~truth].mean()


def above(score_model, cutoff):
    return lambda train, test: (score_model(train, test) > cutoff).astype(float)


def tuned_cutoff(score_model, label, rule):
    """Flag test rows whose score passes a cutoff. The cutoff is chosen from held-out scores inside the training set."""
    def fit_predict(train, test):
        inner = np.zeros(len(train))
        for fit_rows, held_rows in GroupKFold(5).split(train, groups=train["participant_id"]):
            inner[held_rows] = score_model(train.iloc[fit_rows], train.iloc[held_rows])
        truth = label(train)
        cutoff = max(np.unique(inner), key=lambda line: rule(truth, inner >= line))
        return (score_model(train, test) >= cutoff).astype(float)
    return fit_predict


# ---------------- 1.5 surgeon prior learned from every SOCAL trial ----------------

class SurgeonPrior:
    """What the surgeon's record alone predicts for this attempt. SOCAL has 365 trials with outcomes but only 143 with
    video, so this part is learned from the training trials plus every trial of the surgeons who have no video at all.
    Those surgeons can never be test surgeons, so nothing leaks. Training rows get a held-out prior (their own
    surgeon left out) so the video model sees priors of the same quality as it will at test time."""

    def __init__(self, features, name, video_surgeons):
        self.features, self.cache = list(features), {}
        self.columns = [f"{name}_prior_sqrt_ml", f"{name}_prior_failure"]
        self.everyone = add_surgeon_features(pd.read_csv("data/socal_outcomes.csv"))
        self.no_video = self.everyone[~self.everyone["participant_id"].isin(video_surgeons)]

    def add(self, train, test):
        key = (tuple(train["trial_id"]), tuple(test["trial_id"]))
        if key not in self.cache:
            pool = pd.concat([self.everyone[self.everyone["trial_id"].isin(train["trial_id"])], self.no_video]).reset_index(drop=True)
            blood = with_missing_filled(ridge_on("sqrt"), self.features)
            failure = with_missing_filled(failure_classifier, self.features)
            held_out = np.zeros((len(pool), 2))
            for fit_rows, held_rows in GroupKFold(5).split(pool, groups=pool["participant_id"]):
                held_out[held_rows, 0] = np.sqrt(blood(pool.iloc[fit_rows], pool.iloc[held_rows]))
                held_out[held_rows, 1] = failure(pool.iloc[fit_rows], pool.iloc[held_rows])
            held_out = pd.DataFrame(held_out, index=pool["trial_id"])
            self.cache[key] = (held_out.loc[train["trial_id"]].values,
                               np.column_stack([np.sqrt(blood(pool, test)), failure(pool, test)]))
        train_prior, test_prior = self.cache[key]
        return (train.assign(**dict(zip(self.columns, train_prior.T))), test.assign(**dict(zip(self.columns, test_prior.T))))

    def wrap(self, maker, features):
        def fit_predict(train, test):
            train, test = self.add(train, test)
            return with_missing_filled(maker, list(features) + self.columns)(train, test)
        return fit_predict


def with_embedding(summaries, maker, features, components=10):
    """Add the 10 main directions of the trial's DINOv2 frame features as extra inputs."""
    names = [f"embedding_{i}" for i in range(components)]

    def fit_predict(train, test):
        pca = make_pipeline(StandardScaler(), PCA(n_components=components, random_state=0))
        pca.fit(np.stack([summaries[t] for t in train["trial_id"]]))
        train = train.assign(**dict(zip(names, pca.transform(np.stack([summaries[t] for t in train["trial_id"]])).T)))
        test = test.assign(**dict(zip(names, pca.transform(np.stack([summaries[t] for t in test["trial_id"]])).T)))
        return maker(list(features) + names)(train, test)
    return fit_predict


# ---------------- the methods compared ----------------

current, socalnet_rebuild = "D. B + previous attempt, log target (current)", "H. SOCALNet-style ResNet + LSTM (our re-build)"
_methods = {}


def methods(table):
    if _methods:
        return _methods
    from calibre.experiments.image_models import SocalnetStyle, sequence_inputs, trial_embedding_summary
    b, d, e = video_features, video_features + last_attempt, video_features + surgeon_feature_names
    video_surgeons = set(table["participant_id"])
    last_prior = SurgeonPrior(last_attempt, "last", video_surgeons)
    surgeon_prior = SurgeonPrior(surgeon_feature_names, "surgeon", video_surgeons)
    dino = trial_embedding_summary("dinov2")

    def filled(maker, features):
        return with_missing_filled(maker, features)

    def with_dino(wrap):
        return lambda maker, features: with_embedding(dino, lambda more: wrap(maker, more), features)

    def entry(wrap, features, blood=ridge_on("sqrt"), tune=False):
        return {"ml": wrap(blood, features), "failure": wrap(failure_classifier, features),
                "severe": wrap(severe_classifier, features), "tune": tune}

    _methods.update({
        "B. Video only, log target (current)": entry(filled, b, ridge_on("log")),
        current: entry(filled, d, ridge_on("log")),
        "D1. D with square-root target": entry(filled, d, tune=True),
        "D2. D with two-part blood model": entry(filled, d, two_part),
        "P1. B + previous-attempt prior from all 365 trials": entry(last_prior.wrap, b, tune=True),
        "P2. D + previous-attempt prior from all 365 trials": entry(last_prior.wrap, d),
        "P3. B + all-surgeon-information prior from all 365 trials": entry(surgeon_prior.wrap, b),
        "E1. B + all surgeon information (143 trials only)": entry(filled, e),
        "G1. DINOv2 + B": entry(with_dino(filled), b),
        "DG. DINOv2 + D": entry(with_dino(filled), d),
        "PG. DINOv2 + P1": entry(with_dino(last_prior.wrap), b),
    })
    if os.path.exists("data/embeddings_resnet50.npz"):
        import torch
        torch.set_num_threads(1)
        rebuild = SocalnetStyle(sequence_inputs(boxes_file))
        _methods[socalnet_rebuild] = {"ml": rebuild.predict_ml, "failure": rebuild.predict_failure, "tune": False}
    return _methods


def method_names(table):
    return list(methods(table))


# ---------------- 1.1 could SOCALNet's two blood-loss numbers come from one test set? ----------------

def test_sets(table, design, repeats=200, seed=0):
    """'random' = 20 videos at random (same 200 sets as run_experiments.socalnet_protocol).
    'as_published' = 11 controlled + 9 failed, the make-up of SOCALNet's test set."""
    rng = np.random.default_rng(seed)
    if design == "random":
        return [rng.choice(len(table), size=20, replace=False) for _ in range(repeats)]
    controlled, failed = np.where(table["success"] == 1)[0], np.where(table["success"] == 0)[0]
    return [np.concatenate([rng.choice(controlled, 11, replace=False), rng.choice(failed, 9, replace=False)])
            for _ in range(repeats)]


def test_set_check(table):
    """R2 = 1 - (RMSE / spread)^2, so a true R2 of 0.74 with RMSE 295 needs a test-set spread of 579 mL."""
    needed_spread = socalnet_rmse / np.sqrt(1 - socalnet_r2)
    blood, rows = table["blood_loss"].values, []
    for design in ["random", "as_published"]:
        sets = test_sets(table, design, repeats=20000, seed=1)
        spread = np.array([blood[rows_].std() for rows_ in sets])
        average = np.array([blood[rows_].mean() for rows_ in sets])
        rows.append({"test_set": design, "spread_needed_for_true_r2": needed_spread,
                     "spread_5pct": np.quantile(spread, 0.05), "spread_median": np.median(spread),
                     "spread_95pct": np.quantile(spread, 0.95), "share_of_sets_reaching_needed_spread": (spread >= needed_spread).mean(),
                     "true_r2_at_rmse_295_on_median_set": 1 - (socalnet_rmse / np.median(spread)) ** 2,
                     "mean_ml_5pct": np.quantile(average, 0.05), "mean_ml_median": np.median(average),
                     "mean_ml_95pct": np.quantile(average, 0.95)})
    return pd.DataFrame(rows)


# ---------------- surgeons held out ----------------

def _evaluate(table, name, predict_ml, predict_failure):
    summary = evaluate(table, predict_ml, predict_failure, name)
    print("done:", name, flush=True)
    return summary, dict(summary.attrs)


def _evaluate_method(table, name):
    method = methods(table)[name]
    return _evaluate(table, name, method["ml"], method["failure"])


def _restore(result):
    summary, attrs = result
    summary.attrs.update(attrs)
    return summary


def _row(table, summary, baselines):
    row = {"method": summary.attrs["name"]}
    for metric, values in summary.iterrows():
        row[metric], row[f"{metric}_low"], row[f"{metric}_high"] = values["score"], values["low"], values["high"]
    for label, baseline in baselines.items():
        if baseline.attrs["name"] == summary.attrs["name"]:
            continue
        gaps = paired_differences(table, summary, baseline)
        for metric in ["severe_auc", "failure_auc", "rmse_ml", "r2_ml"]:
            row[f"{metric}_gap_vs_{label}"] = gaps.loc[metric, "gap"]
            row[f"{metric}_gap_vs_{label}_low"], row[f"{metric}_gap_vs_{label}_high"] = gaps.loc[metric, "low"], gaps.loc[metric, "high"]
    return row


def target_test(table):
    """1.2 and 1.3: the same inputs, eight ways of learning blood loss. Failure risk is the same model every time."""
    rows = []
    for label, features in [("D", video_features + last_attempt), ("B", video_features)]:
        failure = with_missing_filled(failure_classifier, features)
        blood_models = {
            f"{label}: log (current)": ridge_on("log"), f"{label}: log, corrected": ridge_on("log", corrected=True),
            f"{label}: square root": ridge_on("sqrt"), f"{label}: square root, corrected": ridge_on("sqrt", corrected=True),
            f"{label}: plain mL": ridge_on("ml"), f"{label}: gamma model": gamma, f"{label}: Huber loss": huber,
            f"{label}: two-part": two_part,
        }
        results = Parallel(n_jobs=jobs)(delayed(_evaluate)(table, name, with_missing_filled(maker, features), failure)
                                        for name, maker in blood_models.items())
        summaries = [_restore(result) for result in results]
        rows += Parallel(n_jobs=jobs)(delayed(_row)(table, summary, {"log": summaries[0]}) for summary in summaries)
    return pd.DataFrame(rows)


def held_out_comparison(table):
    """1.3, 1.5 and 1.6: every method scored with surgeons held out, with paired gaps against today's model D
    and against our SOCALNet re-build."""
    results = Parallel(n_jobs=jobs)(delayed(_evaluate_method)(table, name) for name in method_names(table))
    summaries = {name: _restore(result) for name, result in zip(method_names(table), results)}
    baselines = {"D": summaries[current]}
    if socalnet_rebuild in summaries:
        baselines["H"] = summaries[socalnet_rebuild]
    rows = Parallel(n_jobs=jobs)(delayed(_row)(table, summary, baselines) for summary in summaries.values())
    predictions = table[["trial_id", "participant_id", "blood_loss", "success"]].copy()
    for name, summary in summaries.items():
        short = name.split(".")[0]
        predictions[f"{short}_predicted_ml"], predictions[f"{short}_failure_risk"] = summary.attrs["predicted_ml"], summary.attrs["failure_risk"]
    return pd.DataFrame(rows), predictions


def _decision(table, name, what, rule_name, decide, label):
    """Share of yes/no calls that are right, with surgeons held out. Averaged over the 10 reshuffles."""
    truth = label(table)
    right = np.mean([harness.out_of_fold(table, decide, reshuffle) == truth for reshuffle in range(harness.reshuffles)], axis=0)
    rng, rows_by_surgeon, samples = np.random.default_rng(0), list(table.groupby("participant_id").indices.values()), []
    for _ in range(harness.bootstrap_rounds):
        rows = np.concatenate([rows_by_surgeon[i] for i in rng.integers(len(rows_by_surgeon), size=len(rows_by_surgeon))])
        samples.append((right[rows].mean(), right[rows][truth[rows]].mean(), 1 - right[rows][~truth[rows]].mean()))
    low, high = np.nanquantile(samples, 0.025, axis=0), np.nanquantile(samples, 0.975, axis=0)
    print("done:", name, what, rule_name, flush=True)
    return {"method": name, "call": what, "rule": rule_name,
            "accuracy": right.mean(), "accuracy_low": low[0], "accuracy_high": high[0],
            "caught": right[truth].mean(), "caught_low": low[1], "caught_high": high[1],
            "false_alarms": 1 - right[~truth].mean(), "false_alarms_low": low[2], "false_alarms_high": high[2]}


def _decision_for(table, name, what, rule_name):
    method = methods(table)[name]
    rules = {
        ("failure", "risk above 0.5"): (above(method["failure"], 0.5), is_failure),
        ("failure", "cutoff tuned in training"): (tuned_cutoff(method["failure"], is_failure, accuracy), is_failure),
        ("over 500 mL", "mL estimate above 500"): (above(method["ml"], severe_ml), is_severe),
        ("over 500 mL", "mL cutoff tuned in training"): (tuned_cutoff(method["ml"], is_severe, caught_minus_false_alarms), is_severe),
        ("over 500 mL", "own classifier, risk above 0.5"): (above(method["severe"], 0.5), is_severe),
        ("over 500 mL", "own classifier, cutoff tuned in training"): (tuned_cutoff(method["severe"], is_severe, caught_minus_false_alarms), is_severe),
    }
    decide, label = rules[(what, rule_name)]
    return _decision(table, name, what, rule_name, decide, label)


decision_rules = [("failure", "risk above 0.5"), ("failure", "cutoff tuned in training"),
                  ("over 500 mL", "mL estimate above 500"), ("over 500 mL", "mL cutoff tuned in training"),
                  ("over 500 mL", "own classifier, risk above 0.5"), ("over 500 mL", "own classifier, cutoff tuned in training")]


def decision_test(table):
    """1.4: how many failures and big bleeds do we call correctly, and at what false-alarm cost?"""
    names = [current] + [name for name in method_names(table) if methods(table)[name]["tune"]]
    return pd.DataFrame(Parallel(n_jobs=jobs)(delayed(_decision_for)(table, name, what, rule_name)
                                              for name in names for what, rule_name in decision_rules))


# ---------------- all 365 trials: does the surgeon's record predict the outcome without any video? ----------------

def all_trials_test(table):
    everyone = add_surgeon_features(pd.read_csv("data/socal_outcomes.csv")).sort_values("trial_id").reset_index(drop=True)
    later = everyone[everyone["last_attempt_log_blood_loss"].notna()].reset_index(drop=True)
    previous = ["last_attempt_log_blood_loss", "last_attempt_controlled"]

    def models(features):
        return with_missing_filled(ridge_on("sqrt"), features), with_missing_filled(failure_classifier, features)

    tests = [(f"All {len(everyone)} trials: previous attempt only", everyone, last_attempt),
             (f"All {len(everyone)} trials: all surgeon information", everyone, surgeon_feature_names),
             (f"The {len(later)} second and third attempts: previous attempt only", later, previous)]
    rows = []
    for name, trials, features in tests:
        summary = evaluate(trials, *models(features), name)
        rows.append({"test": name, "trials": len(trials), "surgeons": trials["participant_id"].nunique(),
                     **{f"{metric}{end}": summary.loc[metric, column] for metric in ["severe_auc", "failure_auc", "spearman", "rmse_ml"]
                        for end, column in [("", "score"), ("_low", "low"), ("_high", "high")]}})

    # the strictest version: learn from the surgeons we have video for, test on surgeons who were never in any analysis
    seen = later["participant_id"].isin(set(table["participant_id"]))
    train, test = later[seen], later[~seen].reset_index(drop=True)
    predict_ml, predict_failure = models(previous)
    ml, risk = predict_ml(train, test), predict_failure(train, test)
    ranges = surgeon_bootstrap(test, ml, risk)
    result = scores(test, ml, risk)
    rows.append({"test": f"Trained on surgeons with video ({len(train)} trials), tested on surgeons with no video",
                 "trials": len(test), "surgeons": test["participant_id"].nunique(),
                 **{f"{metric}{end}": value for metric in ["severe_auc", "failure_auc", "spearman", "rmse_ml"]
                    for end, value in [("", result[metric]), ("_low", ranges.loc[0.025, metric]), ("_high", ranges.loc[0.975, metric])]}})
    return pd.DataFrame(rows)


# ---------------- SOCALNet's protocol ----------------

def _protocol_row(table, name, design, split_number, test_rows):
    method = methods(table)[name]
    train, test = table.iloc[np.setdiff1d(np.arange(len(table)), test_rows)], table.iloc[test_rows]
    ml, risk = method["ml"](train, test), method["failure"](train, test)
    blood, failed, severe = test["blood_loss"].values, is_failure(test), is_severe(test)
    row = {"method": name, "test_set": design, "split": split_number,
           "success_accuracy": ((risk > 0.5) == failed).mean(),
           "controlled_called_right": (risk[~failed] <= 0.5).mean(), "failures_called_right": (risk[failed] > 0.5).mean(),
           "r2_ml": r2_score(blood, ml), "squared_correlation": np.corrcoef(blood, ml)[0, 1] ** 2,
           "rmse_ml": np.sqrt(np.mean((ml - blood) ** 2)), "mean_error_ml": (ml - blood).mean(), "test_spread_ml": blood.std(),
           "over_500_caught": (ml[severe] > severe_ml).mean() if severe.any() else np.nan,
           "over_500_false_alarms": (ml[~severe] > severe_ml).mean()}
    if method["tune"]:
        called_failed = tuned_cutoff(method["failure"], is_failure, accuracy)(train, test) > 0
        flagged = tuned_cutoff(method["severe"], is_severe, caught_minus_false_alarms)(train, test) > 0
        row.update({"success_accuracy_tuned": (called_failed == failed).mean(),
                    "over_500_caught_tuned": flagged[severe].mean() if severe.any() else np.nan,
                    "over_500_false_alarms_tuned": flagged[~severe].mean()})
    return row


def protocol_test(table, repeats=200, deep_repeats=40):
    """Our models scored the way SOCALNet was scored: 20 test videos, split by video (not by surgeon)."""
    tasks = []
    for design in ["random", "as_published"]:
        sets = test_sets(table, design, repeats)
        for name in method_names(table):
            for split_number, test_rows in enumerate(sets[: deep_repeats if name == socalnet_rebuild else repeats]):
                tasks.append(delayed(_protocol_row)(table, name, design, split_number, test_rows))
    return pd.DataFrame(Parallel(n_jobs=jobs, verbose=1)(tasks))


def main(parts):
    parts = parts or ["check", "targets", "held_out", "decisions", "everyone", "protocol"]
    os.makedirs("results", exist_ok=True)
    table = build_table(boxes_file)
    print(f"{labels} labels: {len(table)} trials, {table['participant_id'].nunique()} surgeons", flush=True)
    if "check" in parts:
        test_set_check(table).to_csv(f"results/{labels}_phase1_test_set_check.csv", index=False)
    if "targets" in parts:
        target_test(table).to_csv(f"results/{labels}_phase1_targets.csv", index=False)
    if "held_out" in parts:
        comparison, predictions = held_out_comparison(table)
        comparison.to_csv(f"results/{labels}_phase1_held_out.csv", index=False)
        predictions.to_csv(f"results/{labels}_phase1_predictions.csv", index=False)
    if "decisions" in parts:
        decision_test(table).to_csv(f"results/{labels}_phase1_decisions.csv", index=False)
    if "everyone" in parts:
        all_trials_test(table).to_csv(f"results/{labels}_phase1_all_365_trials.csv", index=False)
    if "protocol" in parts:
        protocol_test(table).to_csv(f"results/{labels}_phase1_protocol.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    # run through the imported module so the worker processes load the models themselves instead of being sent copies
    from calibre.experiments.phase1 import main as run
    run(sys.argv[1:])
