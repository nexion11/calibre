# Next tests (docs/paper_plan.md), all on the 5 fixed surgeon folds, paired against D and DG on the same folds.
#   stabilize  HemoSurv (video inputs) with 10 GRU starts, with and without the surgeon prior; plus how well each
#              model ranks the TIME to control (C-index), against a Weibull survival model on D's inputs
#   combine    HemoSurv's two outputs (failure risk, expected time to control), cross-fitted inside the training
#              folds, added as inputs to D and DG; and blood loss = DG's bleeding rate x HemoSurv's expected time
#   baselines  SOCALNet-style LSTM with and without the previous attempt; gradient-boosted trees on D's inputs
# Run per label set: CALIBRE_LABELS=clean (default) or human.
#   nice -n 19 uv run python -m calibre.experiments.next_tests stabilize combine baselines
import os
import sys
import numpy as np
import pandas as pd
import torch
from lifelines.utils import concordance_index
from sklearn.model_selection import GroupKFold
from calibre.components.vid import boxes_files
from calibre.experiments import hemosurv_stage2 as hs
from calibre.experiments.harness import failure_classifier, paired_differences, scores, surgeon_bootstrap
from calibre.experiments.new_models import with_missing_filled
from calibre.experiments.per_second_finetune import evaluate_fixed_folds
from calibre.experiments.phase1 import ridge_on, with_embedding
from calibre.experiments.run_experiments import build_table, last_attempt, video_features
from calibre.experiments.surgeon_folds import load_surgeon_folds, n_folds

torch.set_num_threads(int(os.environ.get("CALIBRE_THREADS", "3")))
labels = os.environ.get("CALIBRE_LABELS", "clean")
out_dir = "results/next_tests"
time_limit = 300


def summary_from(table, ml, failure, name, **extra):
    ranges = surgeon_bootstrap(table, ml, failure)
    s = pd.DataFrame({"score": pd.Series(scores(table, ml, failure)), "low": ranges.loc[0.025], "high": ranges.loc[0.975]})
    s.attrs.update(name=name, trials=len(table), predicted_ml=ml, failure_risk=failure, **extra)
    return s


def time_c_index(table, predicted_time):
    """How well predicted time to control ranks the real one (failures censored at 300 s). 0.5 = chance."""
    return concordance_index(table["tth"].clip(upper=time_limit), predicted_time, table["success"].values == 1)


def weibull_time(table, folds, features):
    """Baseline time predictor: Weibull survival model on D's inputs, fixed folds."""
    from lifelines import WeibullAFTFitter
    from sklearn.preprocessing import StandardScaler
    fold_of = folds.set_index("participant_id")["fold"]
    tf = table["participant_id"].astype(int).map(fold_of).values
    out = np.zeros(len(table))
    for k in range(n_folds):
        train, test = table[tf != k], table[tf == k]
        x_tr = train[features].fillna(train[features].median()).fillna(0)
        x_te = test[features].fillna(train[features].median()).fillna(0)
        scaler = StandardScaler().fit(x_tr.values)
        cols = [f"f{i}" for i in range(len(features))]
        frame = pd.DataFrame(scaler.transform(x_tr.values), columns=cols)
        frame["tth"], frame["controlled"] = np.minimum(train["tth"].values, time_limit), (train["success"].values == 1).astype(int)
        model = WeibullAFTFitter(penalizer=1.0).fit(frame, "tth", "controlled")
        out[tf == k] = np.minimum(model.predict_median(pd.DataFrame(scaler.transform(x_te.values), columns=cols)).values, time_limit)
    return out


# ------------------------------------------------------------------------------------------------ cross-fitted outputs
def hemosurv_outputs(table, folds, inputs, use_prior, seeds):
    """For outer fold k: test trials from a GRU fit on all training trials; training trials from GRUs fit on the
    other training surgeons only (4 inner folds), so their outputs are as 'unseen' as the test trials' are."""
    outcomes = pd.read_csv("data/socal_outcomes.csv").set_index("trial_id")
    outcomes["tth"] = outcomes["tth"].fillna(time_limit)
    fold_of = folds.set_index("participant_id")["fold"]
    tf = table["participant_id"].astype(int).map(fold_of).values
    by_fold = {}

    def fit(train_rows, test_rows, inputs):
        train_ids = [t for t in table["trial_id"].iloc[train_rows].tolist() if t in inputs]
        if len(train_ids) < len(train_rows):  # keep the prior rows lined up with the trials that have inputs
            train_rows = np.array([r for r in train_rows if table["trial_id"].iloc[r] in inputs])
        test_ids = table["trial_id"].iloc[test_rows].tolist()
        if use_prior:
            prior_train, prior_test = hs.surgeon_prior_columns(table, train_rows, test_rows)
        else:
            prior_train, prior_test = np.zeros((len(train_ids), 0)), np.zeros((len(test_ids), 0))
        runs = [hs.fit_and_predict(train_ids, test_ids, inputs, prior_train, prior_test, outcomes, seed) for seed in seeds]
        failure = np.clip(np.mean([r[0] for r in runs], axis=0), 1e-4, 1 - 1e-4)
        expected = np.mean([r[2] for r in runs], axis=0)
        return pd.DataFrame({"hs_failure_logit": np.log(failure / (1 - failure)), "hs_log_time": np.log(expected)}, index=test_ids)

    for k in range(n_folds):
        fold_inputs = inputs(k) if callable(inputs) else inputs  # image-model inputs depend on the outer fold
        train_rows, test_rows = np.where(tf != k)[0], np.where(tf == k)[0]
        parts = [fit(train_rows, test_rows, fold_inputs)]
        inner = GroupKFold(4).split(train_rows, groups=table["participant_id"].iloc[train_rows])
        for fit_idx, held_idx in inner:
            parts.append(fit(train_rows[fit_idx], train_rows[held_idx], fold_inputs))
        by_fold[k] = pd.concat(parts)
        print(f"  cross-fitted HemoSurv outputs, fold {k}", flush=True)
    return by_fold


def rate_times_time(features):
    """Blood loss = bleeding rate (ridge on the model's inputs, learned as log mL per second) x HemoSurv's time."""
    def maker(more):
        def fit_predict(train, test):
            from calibre.models.risk_model_training import ridge_model
            used = [f for f in more if f != "hs_log_time"]
            tr = train[used].fillna(train[used].median()).fillna(0).values
            te = test[used].fillna(train[used].median()).fillna(0).values
            rate = np.log(train["blood_loss"].values / np.minimum(train["tth"].values, time_limit))
            model = ridge_model().fit(tr, rate)
            return np.exp(model.predict(te)) * np.exp(test["hs_log_time"].values)
        return fit_predict
    return maker


# ------------------------------------------------------------------------------------------------ baselines
def lstm_with(table, folds, extra):
    """SOCALNet-style rebuild (ResNet-50 features + tools per second, LSTM), optionally given per-trial extras."""
    from calibre.experiments.image_models import SocalnetStyle, sequence_inputs
    seqs = sequence_inputs(boxes_files[labels])
    if extra:
        x = table.set_index("trial_id")[extra]
        flags = x.isna().astype(float).add_suffix("_missing")
        x = pd.concat([x.fillna(0), flags], axis=1)
        seqs = {t: np.hstack([s, np.repeat(x.loc[t].values[None].astype(np.float32), len(s), 0)]) for t, s in seqs.items() if t in x.index}
    model = SocalnetStyle(seqs)
    return evaluate_fixed_folds(table, folds, model.predict_ml, model.predict_failure, "LSTM" + (" + previous attempt" if extra else ""))


def boosted_trees(features):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    settings = dict(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=10, l2_regularization=1.0, random_state=0)

    def ml(train, test):
        return HistGradientBoostingRegressor(**settings).fit(train[features], np.sqrt(train["blood_loss"])).predict(test[features]).clip(1) ** 2

    def fail(train, test):
        return HistGradientBoostingClassifier(**settings).fit(train[features], train["success"] == 0).predict_proba(test[features])[:, 1]
    return ml, fail


# ------------------------------------------------------------------------------------------------ run
def report(table, results, file):
    rows = []
    bases = [b for b in ["D. B + previous attempt", "DG. DINOv2 + D"] if b in results]
    for name, s in results.items():
        row = {"method": name, **{m: s.loc[m, "score"] for m in ["severe_auc", "failure_auc", "rmse_ml", "r2_ml"]}}
        if "expected_time" in s.attrs:
            row["time_c_index"] = time_c_index(table, s.attrs["expected_time"])
        for base in bases:
            if name == base:
                continue
            gap = paired_differences(table, s, results[base])
            tag = base.split(".")[0]
            for m in ["severe_auc", "failure_auc", "rmse_ml"]:
                row[f"{m}_vs_{tag}"], row[f"{m}_vs_{tag}_low"], row[f"{m}_vs_{tag}_high"] = gap.loc[m, "gap"], gap.loc[m, "low"], gap.loc[m, "high"]
        rows.append(row)
    out = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(f"{out_dir}/{labels}_{file}.csv", index=False)
    show = ["method", "severe_auc", "failure_auc", "rmse_ml"] + (["time_c_index"] if "time_c_index" in out else [])
    print(out[show].round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    parts = sys.argv[1:] or ["stabilize", "combine", "baselines"]
    quick = os.environ.get("CALIBRE_QUICK") == "1"
    folds = load_surgeon_folds()
    table = build_table(boxes_files[labels])
    d = video_features + last_attempt
    dino = hs_dino = None
    from calibre.experiments.image_models import trial_embedding_summary
    dino = trial_embedding_summary("dinov2")
    base = {"D. B + previous attempt": evaluate_fixed_folds(table, folds, with_missing_filled(ridge_on("log"), d),
                                                            with_missing_filled(failure_classifier, d), "D"),
            "DG. DINOv2 + D": evaluate_fixed_folds(table, folds, with_embedding(dino, lambda f: with_missing_filled(ridge_on("sqrt"), f), d),
                                                   with_embedding(dino, lambda f: with_missing_filled(failure_classifier, f), d), "DG")}
    video = hs.video_inputs(boxes_files[labels], set(table["trial_id"]))
    seeds = range(2) if quick else range(10)

    if "stabilize" in parts:
        results = dict(base)
        weib = weibull_time(table, folds, d)
        results["D. B + previous attempt"].attrs["expected_time"] = weib  # D itself predicts no time; Weibull on D's inputs does
        results["HemoSurv-video + surgeon prior (10 starts)"] = hs.run_variant(table, folds, lambda k: video, True, seeds=seeds, name="hs+prior")
        results["HemoSurv-video, video only (10 starts)"] = hs.run_variant(table, folds, lambda k: video, False, seeds=seeds, name="hs-video")
        report(table, results, "stabilize")

    if "combine" in parts:
        results = dict(base)
        cf_seeds = range(2) if quick else range(5)
        outputs = hemosurv_outputs(table, folds, video, False, cf_seeds)  # video-only HemoSurv: its information is new to D
        hsf = ["hs_failure_logit", "hs_log_time"]
        results["D + HemoSurv outputs"] = evaluate_fixed_folds(
            table, folds, with_missing_filled(ridge_on("log"), d + hsf), with_missing_filled(failure_classifier, d + hsf), "D+HS", outputs)
        results["DG + HemoSurv outputs"] = evaluate_fixed_folds(
            table, folds, with_embedding(dino, lambda f: with_missing_filled(ridge_on("sqrt"), f), d + hsf),
            with_embedding(dino, lambda f: with_missing_filled(failure_classifier, f), d + hsf), "DG+HS", outputs)
        results["DG rate x HemoSurv time (failure from D + HS)"] = evaluate_fixed_folds(
            table, folds, with_embedding(dino, rate_times_time(d), d + ["hs_log_time"]),
            with_missing_filled(failure_classifier, d + hsf), "rate x time", outputs)
        report(table, results, "combine")

    if "baselines" in parts:
        results = dict(base)
        results["LSTM (SOCALNet-style rebuild)"] = lstm_with(table, folds, [])
        results["LSTM + previous attempt"] = lstm_with(table, folds, last_attempt)
        ml, fail = boosted_trees(d)
        results["Gradient-boosted trees on D's inputs"] = evaluate_fixed_folds(table, folds, ml, fail, "GBT")
        report(table, results, "baselines")
