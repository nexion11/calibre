# Checks asked for by the outside review (docs/related_work.md):
#   prefix    track cleanup that only sees the first 60 s (fills gaps / drops blips without looking ahead), written
#             as the label set "cleanpre60" (seconds after 60 keep the normal cleanup; only training uses them)
#   models    D, DG and B on the 5 fixed surgeon folds, for the current label set and CALIBRE_AT_RISK setting
#   history   what the surgeon's previous attempt adds: D minus B on the same folds, B + attempt number only,
#             D with the previous attempt shuffled between surgeons (within course cohort, 20 times), and second
#             attempts only
#   CALIBRE_LABELS=clean CALIBRE_AT_RISK=60 uv run python -m calibre.experiments.validity_checks prefix models history
import os
import sys
import numpy as np
import pandas as pd
from calibre.components.vid import boxes_files, cleanpre_boxes_file
from calibre.experiments.harness import failure_classifier, paired_differences, scores, surgeon_bootstrap
from calibre.experiments.new_models import with_missing_filled
from calibre.experiments.per_second_finetune import evaluate_fixed_folds
from calibre.experiments.phase1 import ridge_on, with_embedding
from calibre.experiments.run_experiments import build_table, last_attempt, video_features
from calibre.experiments.surgeon_folds import load_surgeon_folds, n_folds

labels = os.environ.get("CALIBRE_LABELS", "clean")
at_risk = os.environ.get("CALIBRE_AT_RISK", "")
out_dir = "results/validity"
tag = f"{labels}_{'atrisk' + at_risk if at_risk else 'all'}"


def make_prefix_clean(window=60):
    from calibre.experiments.track_cleanup import apply_settings, load_low_confidence
    frame_to_trial = pd.read_csv("data/frame_to_trial_mapping.csv")
    folds = load_surgeon_folds().set_index("participant_id")["fold"]
    auto = load_low_confidence(frame_to_trial)
    auto = auto[auto["frame_number"] <= window]
    auto["fold"] = auto["trial_id"].str[1:4].astype(int).map(folds)
    frames = auto[["frame", "trial_id", "frame_number", "fold"]].drop_duplicates("frame")
    settings = pd.read_csv("results/clean_track_settings.csv")
    parts = []
    for k in range(n_folds):
        by_tool = {r.tool: {"cutoff": r.cutoff, "gap_fill": int(r.gap_fill), "min_run": int(r.min_run)}
                   for r in settings[settings["fold"] == k].itertuples()}
        parts.append(apply_settings(auto[auto["fold"] == k], frames[frames["fold"] == k], by_tool))
    early = pd.concat(parts, ignore_index=True)
    full = pd.read_csv(boxes_files["clean"]).merge(frame_to_trial, on="frame", how="left")
    late = full[full["frame_number"] > window][["frame", "label", "x1", "y1", "x2", "y2", "confidence"]]
    out = pd.concat([early[["frame", "label", "x1", "y1", "x2", "y2", "confidence"]], late], ignore_index=True)
    out.sort_values("frame", kind="stable").to_csv(cleanpre_boxes_file, index=False)
    print("wrote", cleanpre_boxes_file, len(out), "rows")


def evaluate_models(table, folds, d):
    from calibre.experiments.image_models import trial_embedding_summary
    dino = trial_embedding_summary("dinov2")
    return {"B. video only": evaluate_fixed_folds(table, folds, with_missing_filled(ridge_on("log"), video_features),
                                                  with_missing_filled(failure_classifier, video_features), "B"),
            "D. B + previous attempt": evaluate_fixed_folds(table, folds, with_missing_filled(ridge_on("log"), d),
                                                            with_missing_filled(failure_classifier, d), "D"),
            "DG. DINOv2 + D": evaluate_fixed_folds(table, folds, with_embedding(dino, lambda f: with_missing_filled(ridge_on("sqrt"), f), d),
                                                   with_embedding(dino, lambda f: with_missing_filled(failure_classifier, f), d), "DG")}


def row(name, s, table, base=None, base_name=""):
    out = {"method": name, "n": len(table), "failures": int((table["success"] == 0).sum()),
           **{m: s.loc[m, "score"] for m in ["severe_auc", "failure_auc", "rmse_ml"]},
           "failure_low": s.loc["failure_auc", "low"], "failure_high": s.loc["failure_auc", "high"]}
    if base is not None:
        gap = paired_differences(table, s, base)
        for m in ["severe_auc", "failure_auc", "rmse_ml"]:
            out[f"{m}_vs_{base_name}"] = gap.loc[m, "gap"]
            out[f"{m}_vs_{base_name}_low"], out[f"{m}_vs_{base_name}_high"] = gap.loc[m, "low"], gap.loc[m, "high"]
    return out


def history_tests(table, folds, base_results):
    d = video_features + last_attempt
    b_plus_attempt = video_features + ["second_attempt"]
    rows = [row("B. video only", base_results["B. video only"], table),
            row("D. B + previous attempt (D minus B)", base_results["D. B + previous attempt"], table, base_results["B. video only"], "B"),
            row("B + attempt number only", evaluate_fixed_folds(table, folds, with_missing_filled(ridge_on("log"), b_plus_attempt),
                                                                with_missing_filled(failure_classifier, b_plus_attempt), "B+n"),
                table, base_results["B. video only"], "B")]
    # negative control: the previous attempt's outcome moved to another surgeon's later attempt in the same course
    rng = np.random.default_rng(0)
    shuffled = []
    later = table.index[table["second_attempt"] == 1]
    for _ in range(20):
        t = table.copy()
        for _, group in t.loc[later].groupby("cohort"):
            perm = rng.permutation(group.index)
            t.loc[group.index, ["last_attempt_log_blood_loss", "last_attempt_controlled"]] = \
                table.loc[perm, ["last_attempt_log_blood_loss", "last_attempt_controlled"]].values
        s = evaluate_fixed_folds(t, folds, with_missing_filled(ridge_on("log"), d), with_missing_filled(failure_classifier, d), "D-shuffled")
        shuffled.append({m: s.loc[m, "score"] for m in ["severe_auc", "failure_auc", "rmse_ml"]})
    sh = pd.DataFrame(shuffled)
    rows.append({"method": "D with previous attempt shuffled within cohort (20 runs, mean [2.5-97.5%])", "n": len(table),
                 **{m: sh[m].mean() for m in sh}, **{f"{m}_shuffle_low": sh[m].quantile(0.025) for m in sh},
                 **{f"{m}_shuffle_high": sh[m].quantile(0.975) for m in sh}})
    # second attempts only: the rows where a previous attempt exists
    sub = table[table["second_attempt"] == 1].reset_index(drop=True)
    if sub["success"].nunique() == 2 and (sub["blood_loss"] > 500).nunique() == 2:
        b = evaluate_fixed_folds(sub, folds, with_missing_filled(ridge_on("log"), video_features), with_missing_filled(failure_classifier, video_features), "B-2nd")
        dd = evaluate_fixed_folds(sub, folds, with_missing_filled(ridge_on("log"), d), with_missing_filled(failure_classifier, d), "D-2nd")
        rows += [row("Second attempts only: B", b, sub), row("Second attempts only: D (D minus B)", dd, sub, b, "B")]
    first = table[table["second_attempt"] == 0].reset_index(drop=True)
    if first["success"].nunique() == 2:
        b1 = evaluate_fixed_folds(first, folds, with_missing_filled(ridge_on("log"), video_features), with_missing_filled(failure_classifier, video_features), "B-1st")
        rows.append(row("First attempts only: B (no history exists)", b1, first))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parts = sys.argv[1:] or ["prefix", "models", "history"]
    if "prefix" in parts:
        make_prefix_clean()
    if "models" in parts or "history" in parts:
        folds = load_surgeon_folds()
        table = build_table(boxes_files[labels])
        print(f"{labels}, at risk {at_risk or 'off'}: {len(table)} trials, {(table['success'] == 0).sum()} failures", flush=True)
        results = evaluate_models(table, folds, video_features + last_attempt)
        os.makedirs(out_dir, exist_ok=True)
        if "models" in parts:
            m = pd.DataFrame([row(k, v, table) for k, v in results.items()])
            m.to_csv(f"{out_dir}/{tag}_models.csv", index=False)
            print(m[["method", "n", "severe_auc", "failure_auc", "rmse_ml"]].round(3).to_string(index=False), flush=True)
        if "history" in parts:
            h = history_tests(table, folds, results)
            h.to_csv(f"{out_dir}/{tag}_history.csv", index=False)
            print(h[["method", "severe_auc", "failure_auc", "rmse_ml"]].round(3).to_string(index=False), flush=True)
