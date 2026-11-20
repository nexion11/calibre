# Published SOCAL methods, re-run under our test: whole surgeons held out (5 fixed folds), all trials, 95% ranges,
# paired against our models on the same folds. See docs/related_work.md.
#   entropy   Kugener et al., Oper Neurosurg 2023 (doi 10.1227/ons.0000000000000888): Shannon entropy of the
#             sequence of tools on screen -> logistic regression for success
#   apm       Pangal et al., J Neurosurg 2022 (doi 10.3171/2021.10.JNS211064): automated performance metrics from
#             tool boxes (time to phase, share of frames, disappearances, area, distance, speed, deviation,
#             instrument interactions) -> elastic net for log blood loss (and a logistic version for success)
# Each is run on the first 60 s (early prediction, as SOCALNet) and, with human labels, on the whole trial as
# published. Whole-trial inputs include how long the video runs, which is the time to control, so those rows
# describe a trial after it is over: they are not predictions and are reported apart.
#   uv run python -m calibre.experiments.prior_methods          (CALIBRE_LABELS=clean or human)
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from calibre.components.vid import boxes_files, load_tool_boxes
from calibre.experiments.harness import failure_classifier, paired_differences
from calibre.experiments.new_models import with_missing_filled
from calibre.experiments.per_second_finetune import evaluate_fixed_folds
from calibre.experiments.phase1 import ridge_on
from calibre.experiments.run_experiments import build_table, last_attempt, video_features
from calibre.experiments.surgeon_folds import load_surgeon_folds
from calibre.models.tool_detector_training import tool_types

labels = os.environ.get("CALIBRE_LABELS", "clean")
out_dir = "results/prior_methods"


# ----------------------------------------------------------------------------------------------- per-trial metrics
def per_second(boxes, seconds):
    """For each watched second: set of tools present and the largest box of each tool."""
    states, best = [], []
    by_second = {s: g for s, g in boxes.groupby("frame_number")}
    for s in seconds:
        g = by_second.get(s)
        if g is None:
            states.append(frozenset())
            best.append({})
            continue
        g = g.assign(area=(g["x2"] - g["x1"]) * (g["y2"] - g["y1"]))
        g = g.sort_values("area", ascending=False).drop_duplicates("label")
        states.append(frozenset(g["label"]))
        best.append({r.label: (r.x1, r.y1, r.x2, r.y2) for r in g.itertuples()})
    return states, best


def shannon_entropy(states):
    """Entropy of which combination of the 5 tools is on screen, normalised to 0-1 (32 possible combinations)."""
    if not states:
        return np.nan
    counts = pd.Series([tuple(sorted(s)) for s in states]).value_counts(normalize=True).values
    return float(-(counts * np.log2(counts)).sum() / np.log2(2 ** len(tool_types)))


def overlap(a, b):
    return a is not None and b is not None and min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def apm_features(states, best, width, height, window):
    n = len(states)
    out = {}
    for t in tool_types:
        present = np.array([t in s for s in states])
        centres = [((b[t][0] + b[t][2]) / 2 / width, (b[t][1] + b[t][3]) / 2 / height) if t in b else None for b in best]
        moves = [np.hypot(c2[0] - c1[0], c2[1] - c1[1]) for c1, c2 in zip(centres[:-1], centres[1:]) if c1 and c2]
        xs = [c[0] for c in centres if c]
        ys = [c[1] for c in centres if c]
        areas = [(b[t][2] - b[t][0]) * (b[t][3] - b[t][1]) / (width * height) for b in best if t in b]
        out[f"apm_{t}_first_second"] = (int(np.argmax(present)) + 1) if present.any() else window + 1
        out[f"apm_{t}_share"] = present.mean() if n else 0.0
        out[f"apm_{t}_disappearances"] = int((present[:-1] & ~present[1:]).sum()) if n > 1 else 0
        out[f"apm_{t}_area"] = float(np.mean(areas)) if areas else 0.0
        out[f"apm_{t}_distance"] = float(np.sum(moves)) if moves else 0.0
        out[f"apm_{t}_speed"] = float(np.mean(moves)) if moves else 0.0
        out[f"apm_{t}_x_spread"] = float(np.std(xs)) if len(xs) > 1 else 0.0
        out[f"apm_{t}_y_spread"] = float(np.std(ys)) if len(ys) > 1 else 0.0
    counts = np.array([len(s) for s in states])
    out["apm_disappearances_total"] = sum(out[f"apm_{t}_disappearances"] for t in tool_types)
    out["apm_no_tools_share"] = float((counts == 0).mean()) if n else 0.0
    out["apm_three_tools_share"] = float((counts >= 3).mean()) if n else 0.0
    out["apm_grasper_muscle_overlap"] = float(np.mean([overlap(b.get("grasper"), b.get("muscle")) for b in best])) if n else 0.0
    out["apm_suction_cottonoid_overlap"] = float(np.mean([overlap(b.get("suction"), b.get("cottonoid")) for b in best])) if n else 0.0
    out["apm_reached_muscle_phase"] = float(any("grasper" in s and "muscle" in s for s in states))
    out["apm_reached_cottonoid_phase"] = float(any("suction" in s and "cottonoid" in s for s in states))
    return out


def trial_metrics(table, boxes_file, window):
    """window: seconds watched (60) or None for the whole video."""
    boxes = load_tool_boxes(boxes_file)
    boxes = boxes[boxes["label"].isin(tool_types)]
    frames = pd.read_csv("data/frame_to_trial_mapping.csv")
    sizes = pd.read_csv("data/socal_outcomes.csv").set_index("trial_id")[["trial_video_width", "trial_video_height"]]
    rows = []
    box_groups = dict(list(boxes.groupby("trial_id")))
    frame_groups = frames.groupby("trial_id")["frame_number"].apply(lambda s: sorted(s.tolist())).to_dict()
    for trial_id in table["trial_id"]:
        seconds = [s for s in frame_groups.get(trial_id, []) if window is None or s <= window]
        b = box_groups.get(trial_id, boxes.iloc[:0])
        b = b[b["frame_number"].isin(set(seconds))]
        w = sizes["trial_video_width"].get(trial_id, np.nan)
        h = sizes["trial_video_height"].get(trial_id, np.nan)
        w, h = (1920 if pd.isna(w) else w), (1080 if pd.isna(h) else h)
        states, best = per_second(b, seconds)
        rows.append({"trial_id": trial_id, "shannon_entropy": shannon_entropy(states),
                     **apm_features(states, best, w, h, window or 300), "watched_seconds": len(seconds)})
    return pd.DataFrame(rows).set_index("trial_id")


# ----------------------------------------------------------------------------------------------- the published models
def entropy_logistic(features):
    def fit(train, test):
        model = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=10, max_iter=2000))
        model.fit(train[features].values, train["success"] == 0)
        return model.predict_proba(test[features].values)[:, 1]
    return fit


def mean_blood_loss(train, test):
    return np.full(len(test), np.exp(np.log(train["blood_loss"]).mean()))


def elastic_net_blood(features):
    def fit(train, test):
        model = make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, max_iter=20000, random_state=0))
        model.fit(train[features].values, np.log(train["blood_loss"].values))
        return np.exp(model.predict(test[features].values))
    return fit


def elastic_net_failure(features):
    def fit(train, test):
        model = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=10, penalty="elasticnet", solver="saga",
                                                                     l1_ratios=[0.5], max_iter=5000))
        model.fit(train[features].values, train["success"] == 0)
        return model.predict_proba(test[features].values)[:, 1]
    return fit


# ----------------------------------------------------------------------------------------------- report
def their_metrics(table, s):
    """Also the numbers the SOCAL papers report: accuracy, sensitivity (successes found), specificity (failures found)
    at a 0.5 cutoff, and Pearson r between predicted and actual blood loss."""
    failed = table["success"].values == 0
    called_fail = s.attrs["failure_risk"] > 0.5
    return {"accuracy": float((called_fail == failed).mean()),
            "sensitivity_success": float((~called_fail[~failed]).mean()),
            "specificity_failure": float(called_fail[failed].mean()),
            "pearson_r": float(np.corrcoef(s.attrs["predicted_ml"], table["blood_loss"].values)[0, 1])}


def report(table, results, file, bases):
    rows = []
    for name, s in results.items():
        row = {"method": name, **{m: s.loc[m, "score"] for m in ["severe_auc", "failure_auc", "rmse_ml"]},
               "failure_low": s.loc["failure_auc", "low"], "failure_high": s.loc["failure_auc", "high"], **their_metrics(table, s)}
        for base in bases:
            if base == name or base not in results:
                continue
            gap = paired_differences(table, s, results[base])
            tag = base.split(".")[0]
            for m in ["severe_auc", "failure_auc", "rmse_ml"]:
                row[f"{m}_vs_{tag}"], row[f"{m}_vs_{tag}_low"], row[f"{m}_vs_{tag}_high"] = gap.loc[m, "gap"], gap.loc[m, "low"], gap.loc[m, "high"]
        rows.append(row)
    out = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(f"{out_dir}/{file}.csv", index=False)
    print(out[["method", "severe_auc", "failure_auc", "rmse_ml", "accuracy", "pearson_r"]].round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    folds = load_surgeon_folds()
    table = build_table(boxes_files[labels])
    d = video_features + last_attempt
    early = trial_metrics(table, boxes_files[labels], 60)
    apm = [c for c in early.columns if c.startswith("apm_")]
    t60 = table.join(early, on="trial_id")
    results = {"D. B + previous attempt": evaluate_fixed_folds(t60, folds, with_missing_filled(ridge_on("log"), d),
                                                               with_missing_filled(failure_classifier, d), "D")}
    results["Entropy (Kugener 2023), first 60 s"] = evaluate_fixed_folds(t60, folds, mean_blood_loss, entropy_logistic(["shannon_entropy"]), "entropy")
    results["APM elastic net (Pangal 2022), first 60 s"] = evaluate_fixed_folds(t60, folds, elastic_net_blood(apm), elastic_net_failure(apm), "APM")
    borrowed = ["apm_disappearances_total", "apm_suction_disappearances", "shannon_entropy", "apm_muscle_first_second", "apm_no_tools_share"]
    results["D + borrowed signals (disappearances, entropy, time to muscle, no-tool share)"] = evaluate_fixed_folds(
        t60, folds, with_missing_filled(ridge_on("log"), d + borrowed), with_missing_filled(failure_classifier, d + borrowed), "D+borrowed")
    report(t60, results, f"{labels}_first60s", ["D. B + previous attempt"])

    if labels == "human":  # as published: the whole trial, human labels. Includes trial length = time to control.
        whole = trial_metrics(table, boxes_files["human"], None)
        tw = table.join(whole, on="trial_id")
        post = {"D. B + previous attempt": results["D. B + previous attempt"]}
        post["Entropy (Kugener 2023), whole trial, as published"] = evaluate_fixed_folds(tw, folds, mean_blood_loss, entropy_logistic(["shannon_entropy"]), "entropy-whole")
        post["APM elastic net (Pangal 2022), whole trial, as published"] = evaluate_fixed_folds(tw, folds, elastic_net_blood(apm), elastic_net_failure(apm), "APM-whole")
        post["Video length alone (= time to control), whole trial"] = evaluate_fixed_folds(tw, folds, elastic_net_blood(["watched_seconds"]), entropy_logistic(["watched_seconds"]), "length")
        report(tw, post, "human_whole_trial", ["D. B + previous attempt"])
