# HemoSurv: predict hemorrhage control second by second, from video, with the surgeon's own starting risk.
#
# What it fixes. SOCALNet and our earlier models predict a trial's outcome from 143 trial-level examples.
# Every second of video, though, carries its own answer: "from here, how long until the bleed is controlled?"
# (exactly known for controlled trials, at least 300 - t for failed ones). HemoSurv learns from those ~25,000
# per-second answers in BOTH stages, so nothing is squeezed into a few trial totals.
#
# Stage 1 (GPU) - per-second encoder. ResNet-50 (the backbone SOCALNet used, ImageNet start) reads one frame and
#   predicts (a) which tools are on screen, (b) visible blood, (c) a distribution over the time left until control,
#   in 11 bins, trained with a censoring-aware likelihood (failed trials only say "more than 300 - t seconds").
#   A 32-number summary of each second is kept for stage 2. Leak rule as in Phase 4: an encoder never describes a
#   surgeon whose outcomes it trained on (5 outer + 20 inner encoders per seed).
# Stage 2 (CPU) - survival sequence model. A small GRU reads the seconds in order (encoder summary, detected tools,
#   visible blood) plus the surgeon's prior (previous attempt, and a prior learned from all 365 SOCAL trials) and,
#   at EVERY second, predicts the time-left distribution again and a bleeding rate. At 60 s:
#     failure risk = chance control is not reached by 300 s
#     blood loss   = bleeding rate x expected time to control (SOCAL: time to control explains ~half of blood loss)
#
#   uv run python -m calibre.experiments.hemosurv cache                  (once: frames into one array, ~2 min)
#   uv run python -m calibre.experiments.hemosurv train --seed 0         (GPU: 25 encoders)
#   uv run python -m calibre.experiments.hemosurv pack --seed 0          (zip parts under 9 MB for Drive)
#   uv run python -m calibre.experiments.hemosurv evaluate               (CPU: stage 2 + comparisons)
#   --quick on any step runs a tiny version in separate folders
import argparse
import json
import os
import random
import time
import zipfile
import numpy as np
import pandas as pd
import torch
from PIL import Image
from calibre.experiments.per_second_finetune import device_and_settings, paths, resnet50_imagenet
from calibre.experiments.surgeon_folds import load_surgeon_folds, n_folds
from calibre.models.tool_detector_training import tool_types

time_limit, watch_seconds, describe_up_to = 300, 60, 180
bin_edges = np.array([0, 10, 20, 30, 45, 60, 90, 120, 180, 240, 300])  # seconds left; last bin = beyond 300
n_bins = len(bin_edges)  # 10 bins between the edges + 1 for "not within the window"
summary_size = 32


def hemo_paths(quick):
    p = paths(quick)
    tag = "_quick" if quick else ""
    return {**p, "cache": f"data/frames_small{tag}.npy", "out": f"data/hemosurv{tag}", "results": f"results/hemosurv{tag}"}


# --------------------------------------------------------------------------------------------------- data
def survival_targets(table):
    """Per second: the bin the control time falls in (controlled trials), or the first bin still possible
    (failed trials: control, if ever, comes after 300 s). Returns (event_bin, first_allowed_bin); event_bin = -1
    means censored, so any bin from first_allowed_bin on is consistent with what happened."""
    outcomes = pd.read_csv("data/socal_outcomes.csv").set_index("trial_id")
    tth = table["trial_id"].map(outcomes["tth"]).fillna(time_limit).values.astype(float)
    controlled = table["trial_id"].map(outcomes["success"]).fillna(0).values == 1
    left = np.clip(tth - table["frame_number"].values, 0, None)
    event_bin = np.where(controlled, np.clip(np.searchsorted(bin_edges, left, side="right") - 1, 0, n_bins - 1), -1)
    window_left = np.clip(time_limit - table["frame_number"].values, 0, None)
    first_allowed = np.clip(np.searchsorted(bin_edges, window_left, side="right") - 1, 0, n_bins - 1)
    return event_bin.astype(np.int64), first_allowed.astype(np.int64)


def build_cache(quick):
    """All small frames in one uint8 array, in targets.csv order, so training never decodes a JPEG again."""
    p = hemo_paths(quick)
    table = pd.read_csv(p["targets"])
    if os.path.exists(p["cache"]):
        cached = np.load(p["cache"], mmap_mode="r")
        if cached.shape[0] == len(table):
            print("cache ready:", p["cache"], cached.shape)
            return
    width, height = Image.open(f"{p['small']}/{table['frame'].iloc[0]}").size
    out = np.lib.format.open_memmap(p["cache"] + ".tmp.npy", mode="w+", dtype=np.uint8, shape=(len(table), height, width, 3))
    for i, name in enumerate(table["frame"]):
        out[i] = np.asarray(Image.open(f"{p['small']}/{name}").convert("RGB"))
        if i % 5000 == 0:
            print(f"cache {i}/{len(table)}", flush=True)
    out.flush()
    del out
    os.replace(p["cache"] + ".tmp.npy", p["cache"])
    print("cache written:", p["cache"], (len(table), height, width, 3))


channels_last = os.environ.get("CALIBRE_CHANNELS_LAST", "0") == "1"


class RowImages(torch.utils.data.Dataset):
    """Reads small JPEG frames in worker processes (the loader Phase 4 ran at ~1.2 min per round on the PC)."""

    def __init__(self, names, folder, rows):
        self.names, self.folder, self.rows = names, folder, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = int(self.rows[i])
        return r, torch.from_numpy(np.asarray(Image.open(f"{self.folder}/{self.names[r]}").convert("RGB")).copy())


class GpuBatches:
    """Batches of frames as uint8, moved to the GPU; augmentation and normalisation happen there.
    source "jpeg" (default) decodes small JPEGs in worker processes; "memory" indexes one in-memory array."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, device, names=None, folder=None, frames=None):
        self.device, self.names, self.folder, self.frames = device, names, folder, frames
        self.mean, self.std = self.mean.to(device), self.std.to(device)
        self.workers = int(os.environ.get("CALIBRE_WORKERS", "4"))

    def batches(self, rows, batch_size, train, shuffle=False, drop_last=False):
        rows = np.random.permutation(rows) if shuffle else np.asarray(rows)
        if self.frames is not None:
            stop = len(rows) - len(rows) % batch_size if drop_last else len(rows)
            for start in range(0, stop, batch_size):
                chunk = rows[start:start + batch_size]
                yield chunk, self.images(torch.from_numpy(self.frames[chunk]), train)
            return
        loader = torch.utils.data.DataLoader(RowImages(self.names, self.folder, rows), batch_size=batch_size,
                                             num_workers=self.workers, drop_last=drop_last, pin_memory=self.device == "cuda")
        for chunk, x in loader:
            yield chunk.numpy(), self.images(x, train)

    def images(self, x, train):
        rows_in_batch = x.shape[0]
        x = x.to(self.device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
        if train:
            flip = (torch.rand(rows_in_batch, 1, 1, 1, device=self.device) < 0.5).float()
            x = flip * x.flip(-1) + (1 - flip) * x  # no boolean indexing: avoids a GPU-to-CPU sync per batch
            gain = torch.empty(rows_in_batch, 1, 1, 1, device=self.device).uniform_(0.85, 1.15)
            shift = torch.empty(rows_in_batch, 1, 1, 1, device=self.device).uniform_(-0.05, 0.05)
            x = (x * gain + shift).clamp_(0, 1)
        x = (x - self.mean) / self.std
        return x.contiguous(memory_format=torch.channels_last) if channels_last else x


# --------------------------------------------------------------------------------------------------- stage 1
backbone_name = os.environ.get("CALIBRE_BACKBONE", "resnet50")  # resnet50 (SOCALNet's, the fair comparison) or dinov2


class Dinov2Small(torch.nn.Module):
    """DINOv2-small (self-supervised ViT, 22M weights, safetensors). Frames are resized to 280x154, a multiple of
    its 14-pixel patches; the class token is the frame's summary."""

    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained("facebook/dinov2-small", use_safetensors=True)
        self.out_size = self.model.config.hidden_size

    def forward(self, images):
        images = torch.nn.functional.interpolate(images, size=(154, 280), mode="bilinear", align_corners=False)
        return self.model(pixel_values=images).pooler_output


class HemoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if backbone_name == "dinov2":
            self.backbone = Dinov2Small()
            width = self.backbone.out_size
        else:
            self.backbone, width = resnet50_imagenet(), 2048
        self.summary = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(width, summary_size), torch.nn.ReLU())
        self.tools = torch.nn.Linear(summary_size, len(tool_types))
        self.blood = torch.nn.Linear(summary_size, 1)
        self.time_left = torch.nn.Linear(summary_size, n_bins)

    def forward(self, images):
        s = self.summary(self.backbone(images))
        return s, self.tools(s), self.blood(s).squeeze(1), self.time_left(s)


def censored_bin_nll(logits, event_bin, first_allowed):
    """-log P(what happened): the event's bin if controlled, else any bin from first_allowed on."""
    log_p = torch.log_softmax(logits, dim=1)
    bins = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
    allowed = torch.where(event_bin.unsqueeze(1) >= 0, bins == event_bin.unsqueeze(1), bins >= first_allowed.unsqueeze(1))
    return -torch.logsumexp(log_p.masked_fill(~allowed, -1e4), dim=1).mean()


def stage1_losses(outputs, targets):
    _, tools, blood, time_left = outputs
    tool_truth, blood_truth, event_bin, first_allowed = targets
    return {"tools": torch.nn.functional.binary_cross_entropy_with_logits(tools, tool_truth),
            "blood": torch.nn.functional.smooth_l1_loss(blood, blood_truth * 4) / 4,
            "time": censored_bin_nll(time_left, event_bin, first_allowed)}


def encoder_jobs():
    return [(k, None) for k in range(n_folds)] + [(k, j) for k in range(n_folds) for j in range(n_folds) if j != k]


def seed_folder(p, seed):
    return f"{p['out']}/seed{seed}" if backbone_name == "resnet50" else f"{p['out']}/{backbone_name}_seed{seed}"


def encoder_file(p, seed, outer, inner):
    return f"{seed_folder(p, seed)}/outer{outer}_inner{'x' if inner is None else inner}.npz"


def train_encoder(table, frames, folds, seed, outer, inner, quick):
    """frames: the in-memory array (CALIBRE_LOADER=memory) or None to read small JPEGs with worker processes."""
    p = hemo_paths(quick)
    out_file = encoder_file(p, seed, outer, inner)
    if os.path.exists(out_file):
        print("done already:", out_file)
        return
    unseen = {outer} | ({inner} if inner is not None else set())
    fold_of = folds.set_index("participant_id")["fold"]
    surgeon_fold = table["participant_id"].map(fold_of).values
    train_surgeons = sorted(table.loc[~np.isin(surgeon_fold, list(unseen)), "participant_id"].unique())
    random.Random(1000 * outer + (inner or 0) + seed).shuffle(train_surgeons)
    val_surgeons = set(train_surgeons[: max(1, len(train_surgeons) // 8)])
    fit_rows = np.where(table["participant_id"].isin(set(train_surgeons) - val_surgeons))[0]
    val_rows = np.where(table["participant_id"].isin(val_surgeons))[0]
    if quick:
        fit_rows, val_rows = fit_rows[::3], val_rows[::3]

    torch.manual_seed(seed)
    np.random.seed(seed)
    device, batch_size, mixed = device_and_settings()
    # same settings as Phase 4, which ran at ~1.2 min per round on the RTX 4060: batch 96 already fills ~6 of 8 GB,
    # and on Windows a bigger batch silently spills into system memory and runs ~5x slower instead of erroring
    batch_size = int(os.environ.get("CALIBRE_BATCH", batch_size))
    torch.backends.cudnn.benchmark = os.environ.get("CALIBRE_BENCHMARK", "0") == "1"
    print(f"backbone {backbone_name}, device {device}, batch {batch_size}, loader {'memory' if frames is not None else 'jpeg'}, "
          f"channels_last {channels_last}", flush=True)
    gpu = GpuBatches(device, names=table["frame"].tolist(), folder=p["small"], frames=frames)
    event_bin, first_allowed = survival_targets(table)
    targets_all = (torch.tensor(table[[f"{t}_present" for t in tool_types]].values, dtype=torch.float32),
                   torch.tensor(table["blood"].values, dtype=torch.float32),
                   torch.from_numpy(event_bin), torch.from_numpy(first_allowed))

    def targets_for(rows):
        return tuple(t[rows].to(device, non_blocking=True) for t in targets_all)

    net = HemoEncoder().to(device)
    if channels_last:
        net = net.to(memory_format=torch.channels_last)
    heads = [*net.summary.parameters(), *net.tools.parameters(), *net.blood.parameters(), *net.time_left.parameters()]
    backbone_lr = 2e-5 if backbone_name == "dinov2" else 1e-4  # transformers fine-tune best with a smaller step
    groups = [{"params": heads, "lr": 1e-3}, {"params": net.backbone.parameters(), "lr": backbone_lr}]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    epochs = 1 if quick else 6
    steps = len(fit_rows) // batch_size
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[1e-3, backbone_lr], total_steps=epochs * steps, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=mixed)
    best, best_state, best_epoch, first_val, history, started = np.inf, None, 0, None, [], time.time()
    for epoch in range(1, epochs + 1):
        net.train()
        for rows, images in gpu.batches(fit_rows, batch_size, train=True, shuffle=True, drop_last=True):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=mixed):
                outputs = net(images)
            loss = sum(stage1_losses(tuple(o.float() for o in outputs), targets_for(rows)).values())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
        net.eval()
        totals = {}
        with torch.no_grad():
            for rows, images in gpu.batches(val_rows, batch_size * 2, train=False):
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=mixed):
                    outputs = net(images)
                for task, value in stage1_losses(tuple(o.float() for o in outputs), targets_for(rows)).items():
                    totals[task] = totals.get(task, 0.0) + value.item() * len(rows)
        val = {task: total / len(val_rows) for task, total in totals.items()}
        first_val = first_val if epoch > 1 else dict(val)
        relative = float(np.mean([val[t] / max(first_val[t], 1e-9) for t in val]))
        history.append({"epoch": epoch, **val, "relative": relative, "minutes": (time.time() - started) / 60})
        if device == "cuda" and epoch == 1:
            print(f"GPU memory used at most: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB of "
                  f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
        print(f"hemosurv seed {seed} outer {outer} inner {inner} epoch {epoch}:", {k: round(v, 4) for k, v in val.items()},
              f"relative {relative:.3f}  {(time.time() - started) / 60:.1f} min", flush=True)
        if relative < best:
            best, best_epoch = relative, epoch
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    print(f"kept round {best_epoch}", flush=True)

    # describe every second up to 180 s of every trial; stage 2 picks which encoder describes which surgeon
    rows_out = np.where(table["frame_number"].values <= describe_up_to)[0]
    summaries, probs, tools, blood = [], [], [], []
    with torch.no_grad():
        for rows, images in gpu.batches(rows_out, batch_size * 2, train=False):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=mixed):
                s, t, b, tl = net(images)
            summaries.append(s.float().cpu().numpy().astype(np.float16))
            probs.append(torch.softmax(tl.float(), 1).cpu().numpy().astype(np.float16))
            tools.append(torch.sigmoid(t.float()).cpu().numpy().astype(np.float16))
            blood.append(b.float().cpu().numpy().astype(np.float16))
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    np.savez_compressed(out_file, trial_id=table["trial_id"].to_numpy().astype(str)[rows_out],
                        second=table["frame_number"].values[rows_out], summary=np.concatenate(summaries),
                        time_left=np.concatenate(probs), tools=np.concatenate(tools), blood=np.concatenate(blood))
    json.dump({"seed": seed, "outer": outer, "inner": inner, "train_frames": int(len(fit_rows)), "val_frames": int(len(val_rows)),
               "batch_size": batch_size, "epochs": epochs, "best_epoch": best_epoch, "minutes": (time.time() - started) / 60,
               "history": history}, open(out_file.replace(".npz", ".json"), "w"), indent=2)
    print("saved", out_file, flush=True)


def pack(seed, quick, limit_mb=9):
    """Zip one seed's encoder outputs into parts small enough to download from Drive."""
    p = hemo_paths(quick)
    folder = seed_folder(p, seed)
    files = sorted(os.listdir(folder))
    part, size, zipped = 0, 0, None
    for name in files:
        path = f"{folder}/{name}"
        if zipped is None or size + os.path.getsize(path) > limit_mb * 1e6:
            if zipped:
                zipped.close()
            part, size = part + 1, 0
            zipped = zipfile.ZipFile(f"hemosurv_{os.path.basename(folder)}_part{part}.zip", "w", zipfile.ZIP_STORED)
        zipped.write(path, name)
        size += os.path.getsize(path)
    if zipped:
        zipped.close()
    print(f"{len(files)} files in {part} zip parts: hemosurv_{os.path.basename(folder)}_part1..{part}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["cache", "train", "pack", "evaluate"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--job", type=int)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.step == "cache":
        build_cache(args.quick)
    elif args.step == "train":
        p = hemo_paths(args.quick)
        table = pd.read_csv(p["targets"])
        frames = None
        if os.environ.get("CALIBRE_LOADER", "jpeg") == "memory":
            build_cache(args.quick)
            frames = np.load(p["cache"])  # whole array in memory (~4.3 GB for every frame)
        jobs = encoder_jobs()
        for i in ([args.job] if args.job is not None else range(len(jobs))):
            train_encoder(table, frames, load_surgeon_folds(), args.seed, *jobs[i], args.quick)
    elif args.step == "pack":
        pack(args.seed, args.quick)
    else:
        from calibre.experiments.hemosurv_stage2 import evaluate
        evaluate(args.quick)
