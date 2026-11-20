# SOCALNet rebuilt as its paper describes (Pangal et al., Sci Rep 2022, doi 10.1038/s41598-022-11549-2; no code was
# published), and scored on our 5 surgeon folds. Their methods, as written:
#   - ResNet pretrained on ImageNet; "the final three layers of the ResNet were retrained on SOCAL images to predict
#     blood loss or task success"; we read this as the last residual stage plus new output layers (the paper does
#     not name the ResNet depth; ResNet-50 is used here)
#   - "The values of the four output nodes from the penultimate layer of the ResNet were extracted" per frame
#   - combined with 8 binary tool-presence flags per frame (human annotations) and passed to a two-layer LSTM
#   - 1 frame per second, videos truncated after 60 s; outputs: success/failure and blood loss
# Variants (stage B): SOCALNet (4 values + 8 human tool flags); frames only (4 values; the "automated" model of
# Kugener et al., Neurosurgery 2022, doi 10.1227/neu.0000000000001906); 4 values + our automatic tool flags.
# As in their pipeline, stage A and stage B are trained on the same training trials (no cross-fitting).
#   uv run python -m calibre.experiments.socalnet_faithful train --seed 0     (GPU, 5 encoders, ~10 min)
#   uv run python -m calibre.experiments.socalnet_faithful pack --seed 0      (one small zip)
#   uv run python -m calibre.experiments.socalnet_faithful evaluate           (CPU)
import argparse
import json
import os
import random
import time
import zipfile
import numpy as np
import pandas as pd
import torch
from calibre.experiments.hemosurv import GpuBatches
from calibre.experiments.per_second_finetune import device_and_settings, paths, resnet50_imagenet
from calibre.experiments.surgeon_folds import load_surgeon_folds, n_folds

watch_seconds = 60
all_tools = ["suction", "grasper", "cottonoid", "muscle", "string", "drill", "scalpel", "tool"]  # SOCAL's 8 classes
out_root = os.environ.get("CALIBRE_SOCALNET_ROOT", "data/socalnet_faithful")


class FrameNet(torch.nn.Module):
    """ImageNet ResNet-50, only its last residual stage trainable, then a 4-node layer and the two outputs."""

    def __init__(self):
        super().__init__()
        self.backbone = resnet50_imagenet()
        for name, weight in self.backbone.named_parameters():
            weight.requires_grad = name.startswith("layer4")
        self.four = torch.nn.Linear(2048, 4)
        self.out = torch.nn.Linear(4, 2)  # failure logit, standardised log blood loss

    def forward(self, images):
        four = self.four(self.backbone(images))
        return four, self.out(torch.relu(four))


def train_encoder(table, folds, seed, outer, quick=False):
    out_file = f"{out_root}/seed{seed}/outer{outer}.npz"
    if os.path.exists(out_file):
        print("done already:", out_file)
        return
    outcomes = pd.read_csv("data/socal_outcomes.csv").set_index("trial_id")
    known = outcomes.index[outcomes["blood_loss"].notna() & outcomes["success"].notna()]
    table = table[table["frame_number"] <= watch_seconds].reset_index(drop=True)
    describe_rows = table.copy()  # every trial is described, but only trials with outcomes are trained on
    table = table[table["trial_id"].isin(known)].reset_index(drop=True)
    fold_of = folds.set_index("participant_id")["fold"]
    surgeon_fold = table["participant_id"].map(fold_of).values
    train_surgeons = sorted(table.loc[surgeon_fold != outer, "participant_id"].unique())
    random.Random(100 * outer + seed).shuffle(train_surgeons)
    val_surgeons = set(train_surgeons[: max(1, len(train_surgeons) // 8)])
    fit_rows = np.where(table["participant_id"].isin(set(train_surgeons) - val_surgeons))[0]
    val_rows = np.where(table["participant_id"].isin(val_surgeons))[0]
    if quick:
        fit_rows, val_rows = fit_rows[::5], val_rows[::5]
    failed = torch.tensor((table["trial_id"].map(outcomes["success"]).values == 0).astype(np.float32))
    log_blood = np.log(table["trial_id"].map(outcomes["blood_loss"]).values.astype(float))
    mean, std = log_blood[fit_rows].mean(), log_blood[fit_rows].std()
    blood = torch.tensor((log_blood - mean) / std, dtype=torch.float32)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device, batch_size, mixed = device_and_settings()
    gpu = GpuBatches(device, names=table["frame"].tolist(), folder=paths(quick)["small"])
    net = FrameNet().to(device)
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": net.backbone.layer4.parameters(), "lr": 1e-4},
                                   {"params": [*net.four.parameters(), *net.out.parameters()], "lr": 1e-3}], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=mixed)
    epochs = 1 if quick else 5
    best, best_state, best_epoch, started = np.inf, None, 0, time.time()

    def loss_of(outputs, rows):
        f, b = failed[rows].to(device), blood[rows].to(device)
        return torch.nn.functional.binary_cross_entropy_with_logits(outputs[:, 0], f) + torch.nn.functional.mse_loss(outputs[:, 1], b)

    for epoch in range(1, epochs + 1):
        net.train()
        net.backbone.eval()  # frozen stages keep ImageNet batch-norm statistics
        net.backbone.layer4.train()
        for rows, images in gpu.batches(fit_rows, batch_size, train=True, shuffle=True, drop_last=True):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=mixed):
                _, outputs = net(images)
            loss = loss_of(outputs.float(), rows)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        net.eval()
        total = 0.0
        with torch.no_grad():
            for rows, images in gpu.batches(val_rows, batch_size * 2, train=False):
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=mixed):
                    _, outputs = net(images)
                total += loss_of(outputs.float(), rows).item() * len(rows)
        val = total / len(val_rows)
        print(f"socalnet seed {seed} outer {outer} epoch {epoch}: check loss {val:.4f}  {(time.time() - started) / 60:.1f} min", flush=True)
        if val < best:
            best, best_epoch, best_state = val, epoch, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    fours = []
    gpu = GpuBatches(device, names=describe_rows["frame"].tolist(), folder=paths(quick)["small"])
    with torch.no_grad():
        for rows, images in gpu.batches(np.arange(len(describe_rows)), batch_size * 2, train=False):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=mixed):
                four, _ = net(images)
            fours.append(four.float().cpu().numpy())
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    np.savez_compressed(out_file, trial_id=describe_rows["trial_id"].to_numpy().astype(str), second=describe_rows["frame_number"].values,
                        four=np.concatenate(fours).astype(np.float32))
    json.dump({"seed": seed, "outer": outer, "best_epoch": best_epoch, "minutes": (time.time() - started) / 60},
              open(out_file.replace(".npz", ".json"), "w"))
    print(f"kept round {best_epoch}; saved {out_file}", flush=True)


# ------------------------------------------------------------------------------------------------ stage B (CPU)
class TwoLayerLSTM(torch.nn.Module):
    def __init__(self, n_inputs, hidden=32):
        super().__init__()
        self.lstm = torch.nn.LSTM(n_inputs, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.failure = torch.nn.Linear(hidden, 1)
        self.blood = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.failure(h[-1]).squeeze(1), self.blood(h[-1]).squeeze(1)


def sequences(fours, flags):
    """trial -> 60 x (4 + flags) array. flags: (trial -> second -> 0/1 vector, number of classes) or None."""
    out = {}
    for t, (seconds, x) in fours.items():
        if flags is None:
            out[t] = x.astype(np.float32)
            continue
        by_second, width = flags
        f = by_second.get(t, {})  # a trial with no tool boxes at all gets all-zero flags
        out[t] = np.hstack([x, np.array([f.get(s, np.zeros(width)) for s in seconds])]).astype(np.float32)
    return out


def tool_flags(boxes_file, classes):
    from calibre.components.vid import load_tool_boxes
    b = load_tool_boxes(boxes_file).dropna(subset=["label"])
    b = b[b["frame_number"] <= watch_seconds]
    out = {}
    for (t, s), g in b.groupby(["trial_id", "frame_number"]):
        out.setdefault(t, {})[s] = np.array([float(c in set(g["label"])) for c in classes])
    return out, len(classes)


def fit_predict_lstm(seqs, train, test, seed, epochs=80):
    torch.manual_seed(seed)
    x_tr = np.stack([seqs[t] for t in train["trial_id"]])
    mean, std = x_tr.reshape(-1, x_tr.shape[2]).mean(0), x_tr.reshape(-1, x_tr.shape[2]).std(0) + 1e-6
    x_tr = torch.tensor((x_tr - mean) / std)
    x_te = torch.tensor((np.stack([seqs[t] for t in test["trial_id"]]) - mean) / std)
    failed = torch.tensor((train["success"].values == 0).astype(np.float32))
    log_b = np.log(train["blood_loss"].values)
    b_mean, b_std = log_b.mean(), log_b.std()
    blood = torch.tensor((log_b - b_mean) / b_std, dtype=torch.float32)
    net = TwoLayerLSTM(x_tr.shape[2])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    for _ in range(epochs):
        net.train()
        for batch in torch.randperm(len(x_tr)).split(16):
            f, b = net(x_tr[batch])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(f, failed[batch]) + torch.nn.functional.mse_loss(b, blood[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        f, b = net(x_te)
    return torch.sigmoid(f).numpy(), np.exp(b.numpy() * b_std + b_mean)


def evaluate(seeds=(0, 1, 2)):
    from calibre.components.vid import boxes_files
    from calibre.experiments.harness import paired_differences, scores, surgeon_bootstrap
    from calibre.experiments.new_models import with_missing_filled
    from calibre.experiments.harness import failure_classifier
    from calibre.experiments.phase1 import ridge_on
    from calibre.experiments.per_second_finetune import evaluate_fixed_folds
    from calibre.experiments.prior_methods import their_metrics
    from calibre.experiments.run_experiments import build_table, last_attempt, video_features
    from calibre.models.tool_detector_training import tool_types
    torch.set_num_threads(3)
    labels = os.environ.get("CALIBRE_LABELS", "clean")
    folds = load_surgeon_folds()
    table = build_table(boxes_files[labels])
    fold_of = folds.set_index("participant_id")["fold"]
    tf = table["participant_id"].astype(int).map(fold_of).values
    d = video_features + last_attempt
    results = {"D. B + previous attempt": evaluate_fixed_folds(table, folds, with_missing_filled(ridge_on("log"), d),
                                                               with_missing_filled(failure_classifier, d), "D")}
    variants = {"SOCALNet, faithful (4 values + 8 human tool flags)": tool_flags(boxes_files["human"], all_tools),
                "Frames only (Kugener 2022 automated)": None,
                "SOCALNet with our automatic tool flags": tool_flags(boxes_files["clean"], tool_types)}
    for name, flags in variants.items():
        runs = []
        for seed in seeds:
            if not all(os.path.exists(f"{out_root}/seed{seed}/outer{k}.npz") for k in range(n_folds)):
                continue
            failure, ml = np.zeros(len(table)), np.zeros(len(table))
            for k in range(n_folds):
                saved = np.load(f"{out_root}/seed{seed}/outer{k}.npz")
                fours = {}
                for t in np.unique(saved["trial_id"]):
                    m = saved["trial_id"] == t
                    order = np.argsort(saved["second"][m])
                    fours[str(t)] = (saved["second"][m][order], saved["four"][m][order])
                seqs = sequences(fours, flags)
                length = 60
                seqs = {t: np.pad(s[:length], ((0, max(0, length - len(s))), (0, 0))) for t, s in seqs.items()}
                train, test = table[tf != k], table[tf == k]
                failure[tf == k], ml[tf == k] = fit_predict_lstm(seqs, train, test, seed)
            ranges = surgeon_bootstrap(table, ml, failure)
            s = pd.DataFrame({"score": pd.Series(scores(table, ml, failure)), "low": ranges.loc[0.025], "high": ranges.loc[0.975]})
            s.attrs.update(name=name, trials=len(table), predicted_ml=ml, failure_risk=failure)
            runs.append(s)
        if runs:
            ml = np.exp(np.mean([np.log(r.attrs["predicted_ml"]) for r in runs], axis=0))
            failure = np.mean([r.attrs["failure_risk"] for r in runs], axis=0)
            ranges = surgeon_bootstrap(table, ml, failure)
            s = pd.DataFrame({"score": pd.Series(scores(table, ml, failure)), "low": ranges.loc[0.025], "high": ranges.loc[0.975]})
            s.attrs.update(name=name, trials=len(table), predicted_ml=ml, failure_risk=failure)
            results[name] = s
    rows = []
    for name, s in results.items():
        row = {"method": name, **{m: s.loc[m, "score"] for m in ["severe_auc", "failure_auc", "rmse_ml"]},
               "failure_low": s.loc["failure_auc", "low"], "failure_high": s.loc["failure_auc", "high"], **their_metrics(table, s)}
        if name != "D. B + previous attempt":
            gap = paired_differences(table, results["D. B + previous attempt"], s)
            for m in ["severe_auc", "failure_auc", "rmse_ml"]:
                row[f"D_minus_this_{m}"], row[f"D_minus_this_{m}_low"], row[f"D_minus_this_{m}_high"] = gap.loc[m, "gap"], gap.loc[m, "low"], gap.loc[m, "high"]
        rows.append(row)
    out = pd.DataFrame(rows)
    os.makedirs("results/socalnet_faithful", exist_ok=True)
    out.to_csv(f"results/socalnet_faithful/{labels}.csv", index=False)
    print(out[["method", "severe_auc", "failure_auc", "rmse_ml", "accuracy", "pearson_r"]].round(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["train", "pack", "evaluate"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.step == "train":
        table = pd.read_csv(paths(args.quick)["targets"])
        for k in range(n_folds):
            train_encoder(table, load_surgeon_folds(), args.seed, k, args.quick)
    elif args.step == "pack":
        with zipfile.ZipFile(f"socalnet_seed{args.seed}.zip", "w") as z:
            for name in sorted(os.listdir(f"{out_root}/seed{args.seed}")):
                z.write(f"{out_root}/seed{args.seed}/{name}", name)
        print(f"wrote socalnet_seed{args.seed}.zip")
    else:
        evaluate()
