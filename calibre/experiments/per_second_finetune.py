# Phase 4: teach an image model what happens in EVERY second of video, then predict the outcome from that.
#
# Stage 1 (GPU). A ResNet-50 (the backbone SOCALNet used, ImageNet start, safetensors weights) is fine-tuned on
# every second of every training video to predict four things about that second:
#   tools    which of the 5 tools are on screen        (human boxes, every frame is labeled)
#   blood    how much of the view is blood              (calibre/components/blood.py colour rule)
#   time     log(1 + seconds left until the bleed is controlled); failed trials only give a lower bound
#   muscle   has the muscle patch been placed yet
# That is ~25,000 training signals per encoder instead of 143 outcomes.
#
# Leak rule. "time" comes from the outcome, so an encoder must never have seen the surgeons it describes:
#   outer encoder k      trained on folds != k             -> describes fold k (the test surgeons)
#   inner encoder k,j    trained on folds not in {k, j}    -> describes fold j when fold k is the test fold
# so every training row of the outcome model is also described by an encoder that never saw that surgeon.
# 5 outer + 20 inner = 25 encoders per variant and seed. Variants without the "time" task do not need the
# inner ones (their targets never touch the outcome) and use only the 5 outer encoders.
#
# Stage 2 (CPU). Per trial, the per-second outputs over the first 60 s are summarised (mean, last 15 s, trend)
# and fed to the same ridge / logistic models as before, on the 5 fixed surgeon folds the encoders were built on.
# Every comparison (D, DG, frozen ImageNet ResNet, each task removed) uses the same folds and paired gaps.
#
#   uv run python -m calibre.experiments.per_second_finetune prepare                      (CPU, ~10 min, once)
#   uv run python -m calibre.experiments.per_second_finetune jobs --variant all            (lists the encoders)
#   uv run python -m calibre.experiments.per_second_finetune train --variant all --seed 0 [--job N]   (GPU)
#   uv run python -m calibre.experiments.per_second_finetune evaluate                     (CPU, after training)
#   add --quick to any of them for a tiny smoke test in separate folders
import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import torch
from PIL import Image
from calibre.experiments.surgeon_folds import load_surgeon_folds, n_folds
from calibre.models.tool_detector_training import frames_folder, tool_types

small_size = (288, 160)  # whole 16:9 view, no cropping; ResNet pools over any size
watch_seconds = 60
time_limit = 300         # SOCAL stops the clock at 5 minutes
variants = {             # name: (tasks, backbone trained?)
    "all": (("tools", "blood", "time", "muscle"), True),
    "frozen": (("tools", "blood", "time", "muscle"), False),
    "no_time": (("tools", "blood", "muscle"), True),
    "no_tools": (("blood", "time", "muscle"), True),
    "no_blood": (("tools", "time", "muscle"), True),
    "no_muscle": (("tools", "blood", "time"), True),
}
head_names = [f"{t}_present" for t in tool_types] + ["blood", "time_left", "muscle_placed"]


def paths(quick):
    tag = "_quick" if quick else ""
    return {"small": f"data/frames_small{tag}", "targets": f"data/finetune{tag}/targets.csv",
            "encoders": f"data/finetune{tag}", "results": f"results/finetune{tag}"}


# ------------------------------------------------------------------------------------------------ prepare
def shrink(job):
    source, target = job
    if not os.path.exists(target):
        Image.open(source).convert("RGB").resize(small_size, Image.BILINEAR).save(target, quality=90)


def prepare(quick):
    p = paths(quick)
    frames = pd.read_csv("data/frame_to_trial_mapping.csv")
    if quick:
        frames = frames[frames["frame_number"] % 10 == 1]
    boxes = pd.read_csv("data/socal.csv", header=None, names=["frame", "x1", "y1", "x2", "y2", "label"])
    present = boxes[boxes["label"].isin(tool_types)].groupby("frame")["label"].apply(set)
    blood = pd.read_csv("data/blood_coverage.csv").set_index("frame")["blood_amt"]
    outcomes = pd.read_csv("data/socal_outcomes.csv").set_index("trial_id")
    t = frames.copy()
    t["participant_id"] = t["trial_id"].str[1:4].astype(int)
    for tool in tool_types:
        t[f"{tool}_present"] = [float(tool in present.get(f, set())) for f in t["frame"]]
    t["blood"] = np.sqrt(t["frame"].map(blood).fillna(0).clip(lower=0))
    tth = t["trial_id"].map(outcomes["tth"]).fillna(time_limit)
    t["controlled"] = t["trial_id"].map(outcomes["success"]).fillna(0).astype(float)
    t["time_left"] = np.log1p((tth - t["frame_number"]).clip(lower=0))
    first_muscle = t[t["muscle_present"] > 0].groupby("trial_id")["frame_number"].min()
    t["muscle_placed"] = (t["frame_number"] >= t["trial_id"].map(first_muscle).fillna(np.inf)).astype(float)
    os.makedirs(p["small"], exist_ok=True)
    os.makedirs(os.path.dirname(p["targets"]), exist_ok=True)
    jobs = [(f"{frames_folder}/{f}", f"{p['small']}/{f}") for f in t["frame"]]
    with ProcessPoolExecutor() as workers:
        list(workers.map(shrink, jobs, chunksize=64))
    t.to_csv(p["targets"], index=False)
    print(len(t), "frames,", t["trial_id"].nunique(), "trials; targets in", p["targets"])


# ------------------------------------------------------------------------------------------------ the network
def resnet50_imagenet():
    """torchvision ResNet-50 with ImageNet weights from a safetensors file (no pickled code is loaded)."""
    import torchvision
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    net = torchvision.models.resnet50(weights=None)
    state = load_file(hf_hub_download("timm/resnet50.tv_in1k", "model.safetensors"))
    state = {k: v for k, v in state.items() if not k.startswith("fc.")}
    missing, unexpected = net.load_state_dict(state, strict=False)
    assert not unexpected and all(k.startswith("fc.") for k in missing), (missing, unexpected)
    net.fc = torch.nn.Identity()
    return net


class PerSecondNet(torch.nn.Module):
    def __init__(self, train_backbone):
        super().__init__()
        self.backbone = resnet50_imagenet()
        for weight in self.backbone.parameters():
            weight.requires_grad = train_backbone
        self.drop = torch.nn.Dropout(0.2)
        self.heads = torch.nn.Linear(2048, len(head_names))  # 5 tools, blood, time left, muscle placed

    def forward(self, images):
        features = self.backbone(images)
        return self.heads(self.drop(features)), features


class Frames(torch.utils.data.Dataset):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, table, folder, train):
        self.names, self.folder, self.train = table["frame"].tolist(), folder, train
        self.targets = torch.tensor(table[head_names + ["controlled"]].values, dtype=torch.float32)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        image = Image.open(f"{self.folder}/{self.names[i]}").convert("RGB")
        x = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).permute(2, 0, 1)
        if self.train:
            if random.random() < 0.5:
                x = x.flip(-1)
            x = (x * random.uniform(0.85, 1.15) + random.uniform(-0.05, 0.05)).clamp(0, 1)
        return (x - self.mean) / self.std, self.targets[i]


def losses(outputs, targets, tasks):
    """One loss per task. 'time' on failed trials is censored: only predicting too LITTLE time left is wrong."""
    parts = {}
    if "tools" in tasks:
        parts["tools"] = torch.nn.functional.binary_cross_entropy_with_logits(outputs[:, :5], targets[:, :5])
    if "blood" in tasks:
        parts["blood"] = torch.nn.functional.smooth_l1_loss(outputs[:, 5], targets[:, 5] * 4) / 4  # sqrt scale x4
    if "time" in tasks:
        guess, truth, controlled = outputs[:, 6], targets[:, 6], targets[:, 8]
        exact = torch.nn.functional.smooth_l1_loss(guess, truth, reduction="none")
        at_least = torch.nn.functional.relu(truth - guess) ** 2 / 2
        parts["time"] = torch.where(controlled > 0, exact, at_least).mean()
    if "muscle" in tasks:
        parts["muscle"] = torch.nn.functional.binary_cross_entropy_with_logits(outputs[:, 7], targets[:, 7])
    return parts


# ------------------------------------------------------------------------------------------------ stage 1
def encoder_jobs(variant):
    """(outer, inner) for every encoder this variant needs. inner None = the outer encoder."""
    tasks, _ = variants[variant]
    jobs = [(k, None) for k in range(n_folds)]
    if "time" in tasks:
        jobs += [(k, j) for k in range(n_folds) for j in range(n_folds) if j != k]
    return jobs


def encoder_file(p, variant, seed, outer, inner):
    return f"{p['encoders']}/{variant}_seed{seed}/outer{outer}_inner{'x' if inner is None else inner}.npz"


def device_and_settings():
    if torch.cuda.is_available():
        return "cuda", 96, True
    if torch.backends.mps.is_available():
        return "mps", 32, False
    return "cpu", 16, False


def train_encoder(table, folds, variant, seed, outer, inner, quick):
    p = paths(quick)
    out_file = encoder_file(p, variant, seed, outer, inner)
    if os.path.exists(out_file):
        print("done already:", out_file)
        return
    tasks, train_backbone = variants[variant]
    unseen = {outer} | ({inner} if inner is not None else set())
    fold_of = folds.set_index("participant_id")["fold"]
    table = table.assign(fold=table["participant_id"].map(fold_of))
    train_surgeons = sorted(table.loc[~table["fold"].isin(unseen), "participant_id"].unique())
    random.Random(1000 * outer + (inner or 0) + seed).shuffle(train_surgeons)
    val_surgeons = set(train_surgeons[: max(1, len(train_surgeons) // 8)])  # only for picking the best epoch
    fit_rows = table[table["participant_id"].isin(set(train_surgeons) - val_surgeons)]
    val_rows = table[table["participant_id"].isin(val_surgeons)]
    if quick:
        fit_rows, val_rows = fit_rows.iloc[::3], val_rows.iloc[::3]

    torch.manual_seed(seed)
    random.seed(seed)
    device, batch_size, mixed = device_and_settings()
    workers = int(os.environ.get("CALIBRE_WORKERS", "4"))
    loader = torch.utils.data.DataLoader(Frames(fit_rows, p["small"], True), batch_size=batch_size, shuffle=True,
                                         num_workers=workers, drop_last=True, persistent_workers=workers > 0)
    val_loader = torch.utils.data.DataLoader(Frames(val_rows, p["small"], False), batch_size=batch_size * 2,
                                             num_workers=workers)
    net = PerSecondNet(train_backbone).to(device)
    groups = [{"params": net.heads.parameters(), "lr": 1e-3}]
    if train_backbone:
        groups.append({"params": net.backbone.parameters(), "lr": 1e-4})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    epochs = 1 if quick else 6
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[g["lr"] for g in groups],
                                                   total_steps=epochs * len(loader), pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=mixed)
    best_loss, best_state, history, started = np.inf, None, [], time.time()
    for epoch in range(1, epochs + 1):
        net.train()
        if not train_backbone:
            net.backbone.eval()  # frozen: batch-norm statistics stay ImageNet's
        for images, targets in loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=mixed):
                outputs, _ = net(images)
            loss = sum(losses(outputs.float(), targets, tasks).values())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
        net.eval()
        totals, count = {}, 0
        with torch.no_grad():
            for images, targets in val_loader:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=mixed):
                    outputs, _ = net(images.to(device))
                for task, value in losses(outputs.float(), targets.to(device), tasks).items():
                    totals[task] = totals.get(task, 0.0) + value.item() * len(targets)
                count += len(targets)
        val = {task: total / count for task, total in totals.items()}
        history.append({"epoch": epoch, **val})
        print(f"{variant} seed {seed} outer {outer} inner {inner} epoch {epoch}:",
              {k: round(v, 4) for k, v in val.items()}, flush=True)
        if sum(val.values()) < best_loss:
            best_loss, best_state = sum(val.values()), {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)

    # describe the first minute of every trial; the evaluator picks which encoder describes which surgeon
    first_minute = table[table["frame_number"] <= watch_seconds]
    describe = torch.utils.data.DataLoader(Frames(first_minute, p["small"], False), batch_size=batch_size * 2,
                                           num_workers=workers)
    heads, embeddings = [], []
    with torch.no_grad():
        for images, _ in describe:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=mixed):
                outputs, features = net(images.to(device))
            outputs = outputs.float()
            outputs[:, :5], outputs[:, 7] = torch.sigmoid(outputs[:, :5]), torch.sigmoid(outputs[:, 7])
            heads.append(outputs.cpu().numpy())
            embeddings.append(features.float().cpu().numpy().astype(np.float16))
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    np.savez_compressed(out_file, trial_id=first_minute["trial_id"].values.astype("U12"), second=first_minute["frame_number"].values,
                        heads=np.concatenate(heads), embedding=np.concatenate(embeddings))
    json.dump({"variant": variant, "seed": seed, "outer": outer, "inner": inner, "tasks": list(tasks),
               "train_backbone": train_backbone, "train_frames": len(fit_rows), "val_frames": len(val_rows),
               "epochs": epochs, "minutes": (time.time() - started) / 60, "history": history},
              open(out_file.replace(".npz", ".json"), "w"), indent=2)
    print("saved", out_file, flush=True)


# ------------------------------------------------------------------------------------------------ stage 2
def trial_summaries(encoder_npz):
    """trial_id -> per-trial numbers from one encoder's per-second outputs over the first minute."""
    saved = np.load(encoder_npz)
    heads = pd.DataFrame(saved["heads"], columns=head_names)
    heads["trial_id"], heads["second"] = saved["trial_id"], saved["second"]
    out = {}
    for trial_id, h in heads.groupby("trial_id"):
        h = h.sort_values("second")
        late = h[h["second"] > watch_seconds - 15]
        late = late if len(late) else h
        row = {}
        for name in head_names:
            row[f"ft_{name}_mean"] = h[name].mean()
            row[f"ft_{name}_late"] = late[name].mean()
            row[f"ft_{name}_trend"] = np.polyfit(h["second"], h[name], 1)[0] if len(h) > 2 else 0.0
        row["ft_time_left_at_end"] = h["time_left"].iloc[-1]
        out[trial_id] = row
    return out


def finetuned_feature_names():
    return [f"ft_{n}_{s}" for n in head_names for s in ["mean", "late", "trend"]] + ["ft_time_left_at_end"]


def features_for_fold(p, variant, seed, outer, folds, trial_ids):
    """Rows for outer fold k: test surgeons described by encoder k; training surgeons in fold j by encoder (k, j)
    when the variant uses the outcome-derived task, else by encoder k (its targets never touch the outcome)."""
    tasks, _ = variants[variant]
    fold_of = folds.set_index("participant_id")["fold"]
    cache = {}

    def described(inner):
        if inner not in cache:
            cache[inner] = trial_summaries(encoder_file(p, variant, seed, outer, inner))
        return cache[inner]
    rows = {}
    for trial_id in trial_ids:
        surgeon_fold = fold_of[int(trial_id[1:4])]
        inner = None if (surgeon_fold == outer or "time" not in tasks) else surgeon_fold
        rows[trial_id] = described(inner)[trial_id]
    return pd.DataFrame.from_dict(rows, orient="index")


def evaluate_fixed_folds(table, folds, fit_predict_ml, fit_predict_failure, name, features_by_fold=None):
    """Like harness.evaluate, but on the 5 fixed surgeon folds, so fold-specific encoders can be used."""
    from calibre.experiments.harness import scores, surgeon_bootstrap
    fold_of = folds.set_index("participant_id")["fold"]
    table_fold = table["participant_id"].astype(int).map(fold_of).values  # the outcome table keeps ids as text
    assert not pd.isna(table_fold).any(), "every surgeon must belong to one of the 5 folds"
    ml, failure = np.zeros(len(table)), np.zeros(len(table))
    for k in range(n_folds):
        data = table if features_by_fold is None else table.join(features_by_fold[k], on="trial_id")
        train, test = data[table_fold != k], data[table_fold == k]
        ml[table_fold == k] = fit_predict_ml(train, test)
        failure[table_fold == k] = fit_predict_failure(train, test)
    ranges = surgeon_bootstrap(table, ml, failure)
    summary = pd.DataFrame({"score": pd.Series(scores(table, ml, failure)),
                            "low": ranges.loc[0.025], "high": ranges.loc[0.975]})
    summary.attrs.update(name=name, trials=len(table), predicted_ml=ml, failure_risk=failure)
    return summary


def average_runs(summaries, table, name):
    """Average several seeds' predictions (blood loss on the log scale), then score once."""
    from calibre.experiments.harness import scores, surgeon_bootstrap
    ml = np.exp(np.mean([np.log(s.attrs["predicted_ml"]) for s in summaries], axis=0))
    failure = np.mean([s.attrs["failure_risk"] for s in summaries], axis=0)
    ranges = surgeon_bootstrap(table, ml, failure)
    summary = pd.DataFrame({"score": pd.Series(scores(table, ml, failure)), "low": ranges.loc[0.025], "high": ranges.loc[0.975]})
    summary.attrs.update(name=name, trials=len(table), predicted_ml=ml, failure_risk=failure)
    return summary


def evaluate(quick):
    from calibre.components.vid import boxes_files
    from calibre.experiments.harness import failure_classifier, paired_differences
    from calibre.experiments.image_models import trial_embedding_summary
    from calibre.experiments.new_models import with_missing_filled
    from calibre.experiments.phase1 import ridge_on, with_embedding
    from calibre.experiments.run_experiments import build_table, last_attempt, video_features
    p, folds = paths(quick), load_surgeon_folds()
    labels = os.environ.get("CALIBRE_LABELS", "clean")  # no human labels at test time by default
    table = build_table(boxes_files[labels])
    d, ft = video_features + last_attempt, finetuned_feature_names()

    def ridge_and_logistic(features, target="sqrt"):
        return with_missing_filled(ridge_on(target), features), with_missing_filled(failure_classifier, features)

    results = {"D. B + previous attempt": evaluate_fixed_folds(table, folds, *ridge_and_logistic(d, "log"), "D")}
    if os.path.exists("data/embeddings_dinov2.npz"):
        dino = trial_embedding_summary("dinov2")
        dg = (with_embedding(dino, lambda f: with_missing_filled(ridge_on("sqrt"), f), d),
              with_embedding(dino, lambda f: with_missing_filled(failure_classifier, f), d))
        results["DG. DINOv2 + D (current best)"] = evaluate_fixed_folds(table, folds, *dg, "DG")

    for variant in variants:
        seeds = sorted({int(f.split("seed")[1]) for f in os.listdir(p["encoders"])
                        if f.startswith(f"{variant}_seed")}) if os.path.exists(p["encoders"]) else []
        for extra, label in [([], "fine-tuned outputs only"), (d, "fine-tuned outputs + D")]:
            runs = []
            for seed in seeds:
                if not all(os.path.exists(encoder_file(p, variant, seed, k, j)) for k, j in encoder_jobs(variant)):
                    print(f"{variant} seed {seed}: encoders missing, skipped")
                    continue
                by_fold = {k: features_for_fold(p, variant, seed, k, folds, table["trial_id"]) for k in range(n_folds)}
                runs.append(evaluate_fixed_folds(table, folds, *ridge_and_logistic(list(extra) + ft),
                                                 f"{variant} {label} seed {seed}", by_fold))
            if runs:
                results[f"P4 {variant}: {label}"] = average_runs(runs, table, f"P4 {variant}: {label}")

    os.makedirs(p["results"], exist_ok=True)
    rows = []
    baselines = {k: v for k, v in results.items() if k.startswith(("D.", "DG."))}
    for name, summary in results.items():
        row = {"method": name, **{m: summary.loc[m, "score"] for m in ["severe_auc", "failure_auc", "rmse_ml", "r2_ml"]},
               "severe_low": summary.loc["severe_auc", "low"], "severe_high": summary.loc["severe_auc", "high"],
               "failure_low": summary.loc["failure_auc", "low"], "failure_high": summary.loc["failure_auc", "high"]}
        for base_name, base in baselines.items():
            if base is summary:
                continue
            gap = paired_differences(table, summary, base)
            tag = base_name.split(".")[0]
            for metric in ["severe_auc", "failure_auc", "rmse_ml"]:
                row[f"{metric}_gap_vs_{tag}"] = gap.loc[metric, "gap"]
                row[f"{metric}_gap_vs_{tag}_low"], row[f"{metric}_gap_vs_{tag}_high"] = gap.loc[metric, "low"], gap.loc[metric, "high"]
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(f"{p['results']}/{labels}_comparison.csv", index=False)
    print(out[["method", "severe_auc", "failure_auc", "rmse_ml"]].round(3).to_string(index=False))


# ------------------------------------------------------------------------------------------------ command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["prepare", "jobs", "train", "evaluate"])
    parser.add_argument("--variant", default="all", choices=list(variants))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--job", type=int, help="train only this encoder (index from 'jobs'); for job arrays")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.step == "prepare":
        prepare(args.quick)
    elif args.step == "jobs":
        for i, (k, j) in enumerate(encoder_jobs(args.variant)):
            print(i, f"outer {k}", "outer encoder" if j is None else f"inner {j}")
    elif args.step == "train":
        table, folds = pd.read_csv(paths(args.quick)["targets"]), load_surgeon_folds()
        jobs = encoder_jobs(args.variant)
        for i in ([args.job] if args.job is not None else range(len(jobs))):
            train_encoder(table, folds, args.variant, args.seed, *jobs[i], args.quick)
    else:
        evaluate(args.quick)
