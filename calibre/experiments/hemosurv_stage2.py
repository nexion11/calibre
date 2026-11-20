# HemoSurv stage 2: a small survival sequence model that reads the bleed second by second.
# At every second it predicts the time left until control (11 bins, censored at 300 s for failed trials) and a
# bleeding rate. Trained on every second of the training trials (up to 180 s), not on 143 trial totals.
# Prediction at 60 s: failure risk = chance control is not reached by 300 s; blood loss = rate x expected time.
#
# Inputs per second (variants):
#   video     detected tools (label set CALIBRE_LABELS, default "clean") + visible blood, no GPU needed
#   encoder   + the HemoSurv stage-1 encoder's per-second outputs (leak rule as in Phase 4)
# Surgeon prior (constant over the trial): the previous attempt (as in model D) and the prior learned from all
# 365 SOCAL trials (phase1.SurgeonPrior). Variants without it test video alone.
# Scored on the 5 fixed surgeon folds, paired against D and DG rerun on the same folds.
import os
import numpy as np
import pandas as pd
import torch
from calibre.experiments.surgeon_folds import load_surgeon_folds, n_folds
from calibre.models.tool_detector_training import tool_types

time_limit, watch_seconds, train_up_to = 300, 60, 180
bin_edges = np.array([0, 10, 20, 30, 45, 60, 90, 120, 180, 240, 300])
n_bins = len(bin_edges)
bin_middle = np.append((bin_edges[:-1] + bin_edges[1:]) / 2, time_limit)  # the last bin: not within the window


# --------------------------------------------------------------------------------------------- per-second inputs
def video_inputs(boxes_file, trial_ids):
    """trial -> (seconds, [5 tools present, sqrt visible blood]) for seconds 1..180."""
    from calibre.components.vid import load_tool_boxes
    boxes = load_tool_boxes(boxes_file).dropna(subset=["label"])
    boxes = boxes[boxes["label"].isin(tool_types) & (boxes["frame_number"] <= train_up_to)]
    if "confidence" in boxes:  # cleaned automatic labels keep only boxes that passed cleanup; NaN = gap filled
        boxes = boxes[boxes["confidence"].isna() | (boxes["confidence"] > 0)]
    present = set(zip(boxes["trial_id"], boxes["frame_number"], boxes["label"]))
    blood = pd.read_csv("data/blood_coverage.csv")
    blood = blood[blood["frame_number"] <= train_up_to]
    out = {}
    for trial_id, b in blood.groupby("trial_id"):
        if trial_id not in trial_ids:
            continue
        b = b.sort_values("frame_number")
        seconds = b["frame_number"].values
        flags = np.array([[float((trial_id, s, t) in present) for t in tool_types] for s in seconds])
        out[trial_id] = (seconds, np.column_stack([flags, np.sqrt(b["blood_amt"].clip(lower=0).values)]).astype(np.float32))
    return out


def encoder_inputs(npz):
    """trial -> (seconds, [32 summary, 11 time-left probabilities]) from one stage-1 encoder."""
    saved = np.load(npz)
    features = np.column_stack([saved["summary"].astype(np.float32), saved["time_left"].astype(np.float32)])
    order = np.lexsort((saved["second"], saved["trial_id"]))
    trial_ids, seconds, features = saved["trial_id"][order], saved["second"][order], features[order]
    out, starts = {}, np.flatnonzero(np.r_[True, trial_ids[1:] != trial_ids[:-1], True])
    for a, b in zip(starts[:-1], starts[1:]):
        out[str(trial_ids[a])] = (seconds[a:b], features[a:b])
    return out


def join_inputs(*parts):
    """Second-by-second join of several input sets (seconds every set has)."""
    out = {}
    for trial_id in set.intersection(*[set(p) for p in parts]):
        common = sorted(set.intersection(*[set(p[trial_id][0].tolist()) for p in parts]))
        cols = []
        for p in parts:
            seconds, x = p[trial_id]
            where = {s: i for i, s in enumerate(seconds.tolist())}
            cols.append(x[[where[s] for s in common]])
        out[trial_id] = (np.array(common), np.hstack(cols))
    return out


# --------------------------------------------------------------------------------------------- targets
def per_second_targets(seconds, tth, controlled):
    left = np.clip(tth - seconds, 0, None)
    event = np.clip(np.searchsorted(bin_edges, left, side="right") - 1, 0, n_bins - 1) if controlled else np.full(len(seconds), -1)
    window_left = np.clip(time_limit - seconds, 0, None)
    first_allowed = np.clip(np.searchsorted(bin_edges, window_left, side="right") - 1, 0, n_bins - 1)
    # seconds after control (video runs a little past it) carry no new information about the time left
    useful = seconds <= tth if controlled else seconds <= time_limit
    return event, first_allowed, useful


def prediction_at(probs, second):
    """Failure risk and expected time to control from the time-left distribution seen at `second`."""
    window_left = time_limit - second
    upper = np.append(bin_edges[1:], np.inf)
    within = upper <= window_left
    failure = 1 - probs[within].sum()
    expected = second + float(np.dot(probs, np.minimum(bin_middle, window_left)))
    return failure, min(expected, time_limit)


# --------------------------------------------------------------------------------------------- the model
class SurvivalGRU(torch.nn.Module):
    def __init__(self, n_inputs, n_prior, hidden=32):
        super().__init__()
        self.inp = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(n_inputs + n_prior, hidden), torch.nn.ReLU())
        self.gru = torch.nn.GRU(hidden, hidden, batch_first=True)
        self.start = torch.nn.Linear(n_prior, hidden) if n_prior else None  # the surgeon's starting state
        self.time_left = torch.nn.Linear(hidden, n_bins)
        self.log_rate = torch.nn.Linear(hidden, 1)

    def forward(self, x, prior):
        steps = x.shape[1]
        h0 = torch.tanh(self.start(prior)).unsqueeze(0) if self.start is not None else None
        if prior.shape[1]:
            x = torch.cat([x, prior.unsqueeze(1).expand(-1, steps, -1)], dim=2)
        out, _ = self.gru(self.inp(x), h0)
        return self.time_left(out), self.log_rate(out).squeeze(2)


def pad(trials, inputs, outcomes, mean, std):
    length = max(len(inputs[t][0]) for t in trials)
    width = inputs[trials[0]][1].shape[1]
    x = np.zeros((len(trials), length, width), np.float32)
    event = np.full((len(trials), length), -1, np.int64)
    allowed = np.zeros((len(trials), length), np.int64)
    mask = np.zeros((len(trials), length), bool)
    for i, t in enumerate(trials):
        seconds, features = inputs[t]
        n = len(seconds)
        x[i, :n] = (features - mean) / std
        e, a, useful = per_second_targets(seconds, outcomes.loc[t, "tth"], outcomes.loc[t, "success"] == 1)
        event[i, :n], allowed[i, :n], mask[i, :n] = e, a, useful
    return torch.from_numpy(x), torch.from_numpy(event), torch.from_numpy(allowed), torch.from_numpy(mask)


def fit_and_predict(train_ids, test_ids, inputs, prior_train, prior_test, outcomes, seed, epochs=120, dense=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    stacked = np.concatenate([inputs[t][1] for t in train_ids])
    mean, std = stacked.mean(0), stacked.std(0) + 1e-6
    p_mean, p_std = prior_train.mean(0), prior_train.std(0) + 1e-6
    prior_tr = torch.tensor((prior_train - p_mean) / p_std, dtype=torch.float32)
    prior_te = torch.tensor((prior_test - p_mean) / p_std, dtype=torch.float32)
    x, event, allowed, mask = pad(train_ids, inputs, outcomes, mean, std)
    if not dense:  # ablation: supervise only the 60-second state, like a trial-level model
        keep = torch.zeros_like(mask)
        seconds = torch.tensor([[s for s in inputs[t][0]] + [10 ** 6] * (x.shape[1] - len(inputs[t][0])) for t in train_ids])
        last = (seconds <= watch_seconds).sum(1) - 1
        keep[torch.arange(len(train_ids)), last] = True
        mask = mask & keep
    tth = outcomes.loc[train_ids, "tth"].values
    rate = torch.tensor(np.log(outcomes.loc[train_ids, "blood_loss"].values / np.minimum(tth, time_limit)), dtype=torch.float32)
    rate_mean, rate_std = rate.mean(), rate.std()
    net = SurvivalGRU(x.shape[2], prior_tr.shape[1])
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
    for _ in range(epochs):
        net.train()
        for batch in torch.randperm(len(train_ids)).split(16):
            logits, log_rate = net(x[batch], prior_tr[batch])
            m = mask[batch]
            if not m.any():
                continue
            log_p = torch.log_softmax(logits, 2)
            bins = torch.arange(n_bins).view(1, 1, -1)
            ev, al = event[batch].unsqueeze(2), allowed[batch].unsqueeze(2)
            ok = torch.where(ev >= 0, bins == ev, bins >= al)
            nll = -torch.logsumexp(log_p.masked_fill(~ok, -1e4), 2)
            rate_err = (log_rate - ((rate[batch] - rate_mean) / rate_std).unsqueeze(1)) ** 2
            loss = nll[m].mean() + 0.5 * rate_err[m].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    net.eval()
    # read each test trial up to 60 s only
    test_inputs = {t: (inputs[t][0][inputs[t][0] <= watch_seconds], inputs[t][1][inputs[t][0] <= watch_seconds]) for t in test_ids}
    xt, *_ = pad(test_ids, test_inputs, outcomes, mean, std)
    with torch.no_grad():
        logits, log_rate = net(xt, prior_te)
    failure, blood, expected_time = np.zeros(len(test_ids)), np.zeros(len(test_ids)), np.zeros(len(test_ids))
    for i, t in enumerate(test_ids):
        last = len(test_inputs[t][0]) - 1
        probs = torch.softmax(logits[i, last], 0).numpy()
        failure[i], expected_time[i] = prediction_at(probs, int(test_inputs[t][0][last]))
        blood[i] = float(np.exp(log_rate[i, last].item() * rate_std + rate_mean)) * expected_time[i]
    return failure, blood, expected_time


# --------------------------------------------------------------------------------------------- evaluation
def surgeon_prior_columns(table, train_rows, test_rows):
    """Previous attempt (filled from training trials) + prior learned from all 365 trials, for train and test rows."""
    from calibre.experiments.phase1 import SurgeonPrior
    from calibre.experiments.run_experiments import last_attempt
    prior = SurgeonPrior(last_attempt, "last", set(table["participant_id"]))
    train, test = prior.add(table.iloc[train_rows], table.iloc[test_rows])
    columns = []
    for c in last_attempt:
        fill = train[c].median()
        columns.append((train[c].fillna(0 if np.isnan(fill) else fill).values, test[c].fillna(0 if np.isnan(fill) else fill).values))
    columns.append((train[last_attempt[1]].isna().astype(float).values, test[last_attempt[1]].isna().astype(float).values))
    for c in prior.columns:
        columns.append((train[c].values, test[c].values))
    return np.column_stack([c[0] for c in columns]), np.column_stack([c[1] for c in columns])


def run_variant(table, folds, inputs_for_fold, use_prior, seeds=(0, 1, 2), dense=True, name=""):
    from calibre.experiments.harness import scores, surgeon_bootstrap
    outcomes = pd.read_csv("data/socal_outcomes.csv").set_index("trial_id")
    outcomes["tth"] = outcomes["tth"].fillna(time_limit)
    fold_of = folds.set_index("participant_id")["fold"]
    table_fold = table["participant_id"].astype(int).map(fold_of).values
    failure, ml, expected_time = np.zeros(len(table)), np.zeros(len(table)), np.zeros(len(table))
    for k in range(n_folds):
        train_rows, test_rows = np.where(table_fold != k)[0], np.where(table_fold == k)[0]
        inputs = inputs_for_fold(k)
        train_ids = [t for t in table["trial_id"].iloc[train_rows] if t in inputs]
        test_ids = table["trial_id"].iloc[test_rows].tolist()
        if use_prior:
            prior_train, prior_test = surgeon_prior_columns(table, train_rows, test_rows)
            keep = [i for i, t in enumerate(table["trial_id"].iloc[train_rows]) if t in inputs]
            prior_train = prior_train[keep]
        else:
            prior_train, prior_test = np.zeros((len(train_ids), 0)), np.zeros((len(test_ids), 0))
        runs = [fit_and_predict(train_ids, test_ids, inputs, prior_train, prior_test, outcomes, seed, dense=dense) for seed in seeds]
        failure[test_rows] = np.mean([r[0] for r in runs], axis=0)
        ml[test_rows] = np.exp(np.mean([np.log(r[1]) for r in runs], axis=0))
        expected_time[test_rows] = np.mean([r[2] for r in runs], axis=0)
        print(f"  {name} fold {k} done", flush=True)
    ranges = surgeon_bootstrap(table, ml, failure)
    summary = pd.DataFrame({"score": pd.Series(scores(table, ml, failure)), "low": ranges.loc[0.025], "high": ranges.loc[0.975]})
    summary.attrs.update(name=name, trials=len(table), predicted_ml=ml, failure_risk=failure, expected_time=expected_time)
    print(f"{name}: severe {summary.loc['severe_auc', 'score']:.3f}  failure {summary.loc['failure_auc', 'score']:.3f}  "
          f"rmse {summary.loc['rmse_ml', 'score']:.0f}", flush=True)
    return summary


def evaluate(quick=False, parts=("video",)):
    from calibre.components.vid import boxes_files
    from calibre.experiments.harness import failure_classifier, paired_differences
    from calibre.experiments.image_models import trial_embedding_summary
    from calibre.experiments.new_models import with_missing_filled
    from calibre.experiments.per_second_finetune import evaluate_fixed_folds
    from calibre.experiments.phase1 import ridge_on, with_embedding
    from calibre.experiments.run_experiments import build_table, last_attempt, video_features
    torch.set_num_threads(4)
    labels = os.environ.get("CALIBRE_LABELS", "clean")
    folds = load_surgeon_folds()
    table = build_table(boxes_files[labels])
    d = video_features + last_attempt
    results = {"D. B + previous attempt": evaluate_fixed_folds(
        table, folds, with_missing_filled(ridge_on("log"), d), with_missing_filled(failure_classifier, d), "D")}
    dino = trial_embedding_summary("dinov2")
    results["DG. DINOv2 + D"] = evaluate_fixed_folds(
        table, folds, with_embedding(dino, lambda f: with_missing_filled(ridge_on("sqrt"), f), d),
        with_embedding(dino, lambda f: with_missing_filled(failure_classifier, f), d), "DG")
    results["B. video only (ridge)"] = evaluate_fixed_folds(
        table, folds, with_missing_filled(ridge_on("log"), video_features), with_missing_filled(failure_classifier, video_features), "B")

    video = video_inputs(boxes_files[labels], set(table["trial_id"]))
    if "video" in parts:
        results["HemoSurv-video, with surgeon prior"] = run_variant(table, folds, lambda k: video, True, name="HemoSurv-video+prior")
        results["HemoSurv-video, video only"] = run_variant(table, folds, lambda k: video, False, name="HemoSurv-video")
        results["Ablation: supervised at 60 s only, with prior"] = run_variant(table, folds, lambda k: video, True, dense=False, name="no-dense")
    if "encoder" in parts:
        from calibre.experiments.hemosurv import encoder_file, hemo_paths
        p = hemo_paths(quick)
        fold_of = folds.set_index("participant_id")["fold"]
        for seed in sorted(int(s[4:]) for s in os.listdir(p["out"]) if s.startswith("seed")):
            def inputs_for_fold(k, seed=seed):
                described = {}
                for t in table["trial_id"]:
                    j = fold_of[int(t[1:4])]
                    enc = encoder_inputs(encoder_file(p, seed, k, None if j == k else j)) if (k, j) not in described else described[(k, j)]
                    described[(k, j)] = enc
                return join_inputs(video, {t: described[(k, fold_of[int(t[1:4])])][t] for t in table["trial_id"]
                                           if t in described[(k, fold_of[int(t[1:4])])]})
            results[f"HemoSurv full (encoder seed {seed}), with prior"] = run_variant(table, folds, inputs_for_fold, True, name=f"HemoSurv-full-s{seed}")
            results[f"HemoSurv full (encoder seed {seed}), video only"] = run_variant(table, folds, inputs_for_fold, False, name=f"HemoSurv-full-video-s{seed}")

    rows = []
    for name, s in results.items():
        row = {"method": name, **{m: s.loc[m, "score"] for m in ["severe_auc", "failure_auc", "rmse_ml", "r2_ml"]}}
        for base in ["D. B + previous attempt", "DG. DINOv2 + D", "B. video only (ridge)"]:
            if name == base:
                continue
            gap = paired_differences(table, s, results[base])
            tag = base.split(".")[0]
            for m in ["severe_auc", "failure_auc", "rmse_ml"]:
                row[f"{m}_vs_{tag}"] = gap.loc[m, "gap"]
                row[f"{m}_vs_{tag}_low"], row[f"{m}_vs_{tag}_high"] = gap.loc[m, "low"], gap.loc[m, "high"]
        rows.append(row)
    out = pd.DataFrame(rows)
    p = "results/hemosurv_quick" if quick else "results/hemosurv"
    os.makedirs(p, exist_ok=True)
    out.to_csv(f"{p}/{labels}_comparison.csv", index=False)
    print(out[["method", "severe_auc", "failure_auc", "rmse_ml"]].round(3).to_string(index=False))
    return out


if __name__ == "__main__":
    import sys
    evaluate("--quick" in sys.argv, tuple(a for a in sys.argv[1:] if not a.startswith("--")) or ("video",))
