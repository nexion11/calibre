# Full HemoSurv: the sequence model reads, every second, the detected tools and visible blood PLUS the stage-1
# image model's summary of the frame. Compared on the same 5 surgeon folds against D and DG, alone and as two added
# inputs (failure risk, expected time to control) to D and DG. One row per image-model set found in data/hemosurv:
#   seed0                  image model also taught time to control (leak rule: 25 encoders)
#   notime_seed0..2        image model taught only tools and blood (5 encoders, never sees an outcome)
#   dinov2_notime_seed0..2 the same with a DINOv2 backbone
# Sets with several seeds also get an ensemble row (predictions averaged over the seeds).
#   CALIBRE_LABELS=clean uv run python -m calibre.experiments.hemosurv_full
import functools
import os
import numpy as np
import pandas as pd
import torch
from calibre.components.vid import boxes_files
from calibre.experiments import hemosurv_stage2 as hs
from calibre.experiments.harness import failure_classifier, paired_differences, scores, surgeon_bootstrap
from calibre.experiments.new_models import with_missing_filled
from calibre.experiments.next_tests import hemosurv_outputs
from calibre.experiments.per_second_finetune import evaluate_fixed_folds
from calibre.experiments.phase1 import ridge_on, with_embedding
from calibre.experiments.run_experiments import build_table, last_attempt, video_features
from calibre.experiments.surgeon_folds import load_surgeon_folds, n_folds

torch.set_num_threads(int(os.environ.get("CALIBRE_THREADS", "3")))
labels = os.environ.get("CALIBRE_LABELS", "clean")
root = os.environ.get("CALIBRE_HEMO_ROOT", "data/hemosurv")
out_dir = "results/hemosurv_full"
gru_seeds = range(int(os.environ.get("CALIBRE_GRU_SEEDS", "3")))


@functools.lru_cache(maxsize=None)
def encoder_features(path, with_time):
    """trial -> (seconds, per-second features): the 32-number frame summary, plus the time-left bins if taught."""
    saved = np.load(path)
    x = saved["summary"].astype(np.float32)
    if with_time:
        x = np.column_stack([x, saved["time_left"].astype(np.float32)])
    order = np.lexsort((saved["second"], saved["trial_id"]))
    trial_ids, seconds, x = saved["trial_id"][order], saved["second"][order], x[order]
    out, starts = {}, np.flatnonzero(np.r_[True, trial_ids[1:] != trial_ids[:-1], True])
    for a, b in zip(starts[:-1], starts[1:]):
        out[str(trial_ids[a])] = (seconds[a:b], x[a:b])
    return out


def inputs_maker(folder, table, folds, video):
    with_time = not os.path.basename(folder).startswith(("notime", "dinov2_notime"))
    fold_of = folds.set_index("participant_id")["fold"]

    @functools.lru_cache(maxsize=None)
    def for_fold(k):
        described = {}
        for t in table["trial_id"]:
            j = fold_of[int(t[1:4])]
            inner = "x" if (j == k or not with_time) else j  # without the time task no encoder ever saw an outcome
            enc = encoder_features(f"{folder}/outer{k}_inner{inner}.npz", with_time)
            if t in enc:
                described[t] = enc[t]
        return hs.join_inputs(video, described)
    return for_fold


def complete(folder):
    with_time = not os.path.basename(folder).startswith(("notime", "dinov2_notime"))
    needed = [f"outer{k}_inner{'x' if (j is None) else j}.npz" for k in range(n_folds)
              for j in ([None] + ([i for i in range(n_folds) if i != k] if with_time else []))]
    return all(os.path.exists(f"{folder}/{n}") for n in needed)


def averaged(summaries, table, name):
    ml = np.exp(np.mean([np.log(s.attrs["predicted_ml"]) for s in summaries], axis=0))
    failure = np.mean([s.attrs["failure_risk"] for s in summaries], axis=0)
    ranges = surgeon_bootstrap(table, ml, failure)
    s = pd.DataFrame({"score": pd.Series(scores(table, ml, failure)), "low": ranges.loc[0.025], "high": ranges.loc[0.975]})
    s.attrs.update(name=name, trials=len(table), predicted_ml=ml, failure_risk=failure)
    return s


if __name__ == "__main__":
    folds = load_surgeon_folds()
    table = build_table(boxes_files[labels])
    d, hsf = video_features + last_attempt, ["hs_failure_logit", "hs_log_time"]
    from calibre.experiments.image_models import trial_embedding_summary
    dino = trial_embedding_summary("dinov2")
    results = {
        "D. B + previous attempt": evaluate_fixed_folds(table, folds, with_missing_filled(ridge_on("log"), d),
                                                        with_missing_filled(failure_classifier, d), "D"),
        "DG. DINOv2 + D": evaluate_fixed_folds(table, folds, with_embedding(dino, lambda f: with_missing_filled(ridge_on("sqrt"), f), d),
                                               with_embedding(dino, lambda f: with_missing_filled(failure_classifier, f), d), "DG")}
    video = hs.video_inputs(boxes_files[labels], set(table["trial_id"]))
    groups = {}
    for name in sorted(os.listdir(root)) if os.path.exists(root) else []:
        folder = f"{root}/{name}"
        if not os.path.isdir(folder) or not complete(folder):
            print("skipped (incomplete):", name)
            continue
        make = inputs_maker(folder, table, folds, video)
        alone = hs.run_variant(table, folds, make, True, seeds=gru_seeds, name=f"HemoSurv[{name}] + prior")
        outputs = hemosurv_outputs(table, folds, make, False, gru_seeds)  # video-only HemoSurv: new information for D
        with_d = evaluate_fixed_folds(table, folds, with_missing_filled(ridge_on("log"), d + hsf),
                                      with_missing_filled(failure_classifier, d + hsf), f"D+HS[{name}]", outputs)
        with_dg = evaluate_fixed_folds(table, folds, with_embedding(dino, lambda f: with_missing_filled(ridge_on("sqrt"), f), d + hsf),
                                       with_embedding(dino, lambda f: with_missing_filled(failure_classifier, f), d + hsf), f"DG+HS[{name}]", outputs)
        results[f"HemoSurv alone [{name}]"], results[f"D + HemoSurv [{name}]"], results[f"DG + HemoSurv [{name}]"] = alone, with_d, with_dg
        group = name.rsplit("seed", 1)[0]
        groups.setdefault(group, []).append((alone, with_d, with_dg))
        print(name, "done", flush=True)
    for group, runs in groups.items():
        if len(runs) > 1:
            label = (group or "time_") + "ensemble"
            for i, kind in enumerate(["HemoSurv alone", "D + HemoSurv", "DG + HemoSurv"]):
                results[f"{kind} [{label}]"] = averaged([r[i] for r in runs], table, f"{kind} [{label}]")

    rows = []
    for name, s in results.items():
        row = {"method": name, **{m: s.loc[m, "score"] for m in ["severe_auc", "failure_auc", "rmse_ml"]}}
        for base in ["D. B + previous attempt", "DG. DINOv2 + D"]:
            if name == base:
                continue
            gap = paired_differences(table, s, results[base])
            tag = base.split(".")[0]
            for m in ["severe_auc", "failure_auc", "rmse_ml"]:
                row[f"{m}_vs_{tag}"], row[f"{m}_vs_{tag}_low"], row[f"{m}_vs_{tag}_high"] = gap.loc[m, "gap"], gap.loc[m, "low"], gap.loc[m, "high"]
        rows.append(row)
    out = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(f"{out_dir}/{labels}.csv", index=False)
    print(out[["method", "severe_auc", "failure_auc", "rmse_ml"]].round(3).to_string(index=False))
