# Phase 2, Part B (PC GPU): find a better tool detector on one surgeon split (fold 0).
#   Training surgeons: folds 1-4. 1/8 of them are held back as VALIDATION surgeons; every choice below
#   (model size, image size, rare-tool fix, RetinaNet's best round) is made on them only.
#   Test surgeons: fold 0's 15 surgeons, never used for any choice, only reported.
# Stages, in order (each finished run is kept, so the command can be stopped and started again):
#   sizes       YOLO11 n, s, m at 640 px, all frames, 30 rounds                      (plan item 2.2)
#   resolution  the best phone-speed model (n or s) at 960 px                        (plan item 2.2)
#   rare        the same with muscle/cottonoid frames repeated + pasted tool crops   (plan item 2.3)
#   retinanet   RetinaNet (ResNet-50), trained by us on the same split               (plan item 2.4)
#   report      one table + paired gaps with surgeon-level ranges                    (results/detector_search*.csv)
#   uv run python -m calibre.experiments.detector_search            (all stages)
#   uv run python -m calibre.experiments.detector_search --quick    (tiny smoke test, a few minutes)
import json
import os
import random
import shutil
import sys
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from calibre.experiments.detection_scoring import Scorer, human_truth_boxes, paired_gap, score_with_range
from calibre.experiments.surgeon_folds import load_surgeon_folds
from calibre.models.tool_detector_training import frames_folder, tool_types

quick = "--quick" in sys.argv
fold = 0
epochs = 1 if quick else 30
retinanet_epochs = 1 if quick else 12
device = 0 if torch.cuda.is_available() else "mps"
suffix = "_quick" if quick else ""
data_root = os.path.abspath(f"data/yolo_search{suffix}")
models_root = os.path.abspath(f"models/detector_search{suffix}")
results_root = f"results/detector_search{suffix}"
choice_file = f"{results_root}/choices.json"
rare_tools = {"muscle": 3, "cottonoid": 2}  # frames with these tools appear this many times in training
pasted_images = 60 if quick else 3000


def link_or_copy(source, destination):
    for make in (os.symlink, os.link):  # hard links work on Windows without admin rights
        try:
            make(source, destination)
            return
        except OSError:
            pass
    shutil.copy(source, destination)


def yolo_label_lines(boxes, width, height):
    lines = []
    for box in boxes.itertuples():
        if box.label in tool_types:
            lines.append(f"{tool_types.index(box.label)} {(box.x1 + box.x2) / 2 / width:.5f} "
                         f"{(box.y1 + box.y2) / 2 / height:.5f} {(box.x2 - box.x1) / width:.5f} "
                         f"{(box.y2 - box.y1) / height:.5f}")
    return lines


def load_split():
    folds = load_surgeon_folds()
    boxes = pd.read_csv("data/socal.csv", header=None, names=["frame", "x1", "y1", "x2", "y2", "label"])
    boxes["participant_id"] = boxes["frame"].str[1:4].astype(int)
    train_surgeons = sorted(folds.loc[folds["fold"] != fold, "participant_id"])
    random.Random(fold).shuffle(train_surgeons)
    val_surgeons = set(train_surgeons[: len(train_surgeons) // 8])  # same validation surgeons as fold_detectors.py
    test_surgeons = set(folds.loc[folds["fold"] == fold, "participant_id"])
    frames = {
        "train": sorted(boxes.loc[boxes["participant_id"].isin(set(train_surgeons) - val_surgeons), "frame"].unique()),
        "val": sorted(boxes.loc[boxes["participant_id"].isin(val_surgeons), "frame"].unique()),
        "test": sorted(boxes.loc[boxes["participant_id"].isin(test_surgeons), "frame"].unique()),
    }
    if quick:
        frames = {split: names[::50] for split, names in frames.items()}
    return boxes, frames


def write_split(folder, split, frame_names, boxes_by_frame, repeats=None):
    os.makedirs(f"{folder}/images/{split}", exist_ok=True)
    os.makedirs(f"{folder}/labels/{split}", exist_ok=True)
    for frame_name in frame_names:
        frame_boxes = boxes_by_frame.get_group(frame_name)
        width, height = Image.open(f"{frames_folder}/{frame_name}").size
        lines = "\n".join(yolo_label_lines(frame_boxes, width, height))
        copies = 1
        if repeats:
            copies = max([repeats.get(label, 1) for label in frame_boxes["label"]] + [1])
        for copy in range(copies):
            stem = frame_name.replace(".jpeg", "" if copy == 0 else f"_repeat{copy}")
            link_or_copy(f"{frames_folder}/{frame_name}", f"{folder}/images/{split}/{stem}.jpeg")
            with open(f"{folder}/labels/{split}/{stem}.txt", "w") as f:
                f.write(lines)


def paste_rare_tools(folder, train_frames, boxes, count, seed=0):
    """Copy-paste: cut muscle/cottonoid boxes out of one training frame and paste them into another."""
    rng = random.Random(seed)
    train_boxes = boxes[boxes["frame"].isin(set(train_frames)) & boxes["label"].isin(tool_types)]
    crops = train_boxes[train_boxes["label"].isin(rare_tools)].reset_index(drop=True)
    by_frame = train_boxes.groupby("frame")
    for number in range(count):
        base_name = rng.choice(train_frames)
        image = Image.open(f"{frames_folder}/{base_name}").convert("RGB")
        width, height = image.size
        frame_boxes = by_frame.get_group(base_name) if base_name in by_frame.groups else train_boxes.iloc[:0]
        frame_boxes = frame_boxes[["x1", "y1", "x2", "y2", "label"]].copy()
        for _ in range(rng.choice([1, 2])):
            crop_row = crops.iloc[rng.randrange(len(crops))]
            source = Image.open(f"{frames_folder}/{crop_row.frame}").convert("RGB")
            crop = source.crop((int(crop_row.x1), int(crop_row.y1), int(crop_row.x2), int(crop_row.y2)))
            scale = rng.uniform(0.8, 1.2)
            crop = crop.resize((max(8, int(crop.width * scale)), max(8, int(crop.height * scale))))
            if crop.width >= width or crop.height >= height:
                continue
            for _attempt in range(20):  # find a spot that does not cover an existing tool much
                x, y = rng.randrange(width - crop.width), rng.randrange(height - crop.height)
                new = np.array([x, y, x + crop.width, y + crop.height], float)
                old = frame_boxes[["x1", "y1", "x2", "y2"]].values.astype(float)
                if len(old) == 0:
                    break
                ix = np.clip(np.minimum(new[2], old[:, 2]) - np.maximum(new[0], old[:, 0]), 0, None)
                iy = np.clip(np.minimum(new[3], old[:, 3]) - np.maximum(new[1], old[:, 1]), 0, None)
                if (ix * iy / ((old[:, 2] - old[:, 0]) * (old[:, 3] - old[:, 1]) + 1e-9)).max() < 0.3:
                    break
            image.paste(crop, (x, y))
            frame_boxes.loc[len(frame_boxes) + 1000] = [x, y, x + crop.width, y + crop.height, crop_row.label]
        stem = f"pasted_{number:05d}"
        image.save(f"{folder}/images/train/{stem}.jpeg", quality=92)
        with open(f"{folder}/labels/train/{stem}.txt", "w") as f:
            f.write("\n".join(yolo_label_lines(frame_boxes, width, height)))


def build_data(variant, boxes, frames):
    folder = f"{data_root}/{variant}"
    if os.path.exists(f"{folder}/data.yaml"):
        return folder
    shutil.rmtree(folder, ignore_errors=True)
    boxes_by_frame = boxes.groupby("frame")
    if variant == "test":
        write_split(folder, "val", frames["test"], boxes_by_frame)
    else:
        write_split(folder, "train", frames["train"], boxes_by_frame, rare_tools if variant == "rare" else None)
        write_split(folder, "val", frames["val"], boxes_by_frame)
        if variant == "rare":
            paste_rare_tools(folder, frames["train"], boxes, pasted_images)
    with open(f"{folder}/data.yaml.tmp", "w") as f:
        f.write(f"path: {folder}\ntrain: images/{'val' if variant == 'test' else 'train'}\n"
                f"val: images/val\nnames: {tool_types}\n")
    os.replace(f"{folder}/data.yaml.tmp", f"{folder}/data.yaml")  # written last, so a half-built folder is rebuilt
    return folder


def load_choices():
    return json.load(open(choice_file)) if os.path.exists(choice_file) else {}


def save_choices(choices):
    os.makedirs(results_root, exist_ok=True)
    json.dump(choices, open(choice_file, "w"), indent=2)


def train_yolo(name, start_weights, data_folder, imgsz):
    from ultralytics import YOLO  # type: ignore
    run_folder = f"{models_root}/{name}"
    if os.path.exists(f"{run_folder}/done.json"):
        return json.load(open(f"{run_folder}/done.json"))
    started = time.time()
    last = f"{run_folder}/weights/last.pt"
    if os.path.exists(last):
        YOLO(last).train(resume=True)
    else:
        YOLO(start_weights).train(data=f"{data_folder}/data.yaml", epochs=epochs, imgsz=imgsz,
                                  batch=-1 if device == 0 else 8, device=device, workers=8,
                                  project=models_root, name=name, exist_ok=True, plots=False, verbose=False)
    info = {"name": name, "start_weights": start_weights, "imgsz": imgsz, "data": os.path.basename(data_folder),
            "epochs": epochs, "train_hours": (time.time() - started) / 3600,
            "train_images": len(os.listdir(f"{data_folder}/images/train"))}
    json.dump(info, open(f"{run_folder}/done.json", "w"), indent=2)
    return info


def ultralytics_scores(weights, data_folder, imgsz, name):
    from ultralytics import YOLO  # type: ignore
    result = YOLO(weights).val(data=f"{data_folder}/data.yaml", split="val", imgsz=imgsz, device=device,
                               plots=False, verbose=False, project=os.path.abspath("runs"), name=name, exist_ok=True)
    row = {"map50": result.box.map50}
    for tool_index, ap in zip(result.box.ap_class_index, result.box.ap50):
        row[f"{tool_types[tool_index]}_ap50"] = ap
    return row


def yolo_predictions(weights, frame_names, imgsz):
    from ultralytics import YOLO  # type: ignore
    detector, rows = YOLO(weights), []
    for start in range(0, len(frame_names), 32):
        batch = frame_names[start:start + 32]
        results = detector.predict([f"{frames_folder}/{n}" for n in batch], conf=0.001, iou=0.7, max_det=300,
                                   imgsz=imgsz, device=device, verbose=False)
        for frame_name, result in zip(batch, results):
            for box, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
                rows.append((frame_name, tool_types[int(cls)], conf, *box))
    return pd.DataFrame(rows, columns=["frame", "label", "confidence", "x1", "y1", "x2", "y2"])


def score_yolo_run(name, weights, imgsz, data_folders, frames):
    """Validation and test scores (Ultralytics' own), plus saved test predictions for the paired gaps."""
    score_file = f"{results_root}/runs/{name}.json"
    if os.path.exists(score_file):
        return json.load(open(score_file))
    os.makedirs(f"{results_root}/runs", exist_ok=True)
    os.makedirs(f"{results_root}/test_predictions", exist_ok=True)
    val = ultralytics_scores(weights, data_folders["plain"], imgsz, f"search_val_{name}")
    test = ultralytics_scores(weights, data_folders["test"], imgsz, f"search_test_{name}")
    yolo_predictions(weights, frames["test"], imgsz).to_csv(f"{results_root}/test_predictions/{name}.csv.gz", index=False)
    row = {"name": name, "imgsz": imgsz, **{f"val_{k}": v for k, v in val.items()},
           **{f"test_{k}": v for k, v in test.items()}}
    json.dump(row, open(score_file, "w"), indent=2)
    print(name, "validation mAP50", round(row["val_map50"], 3), "test mAP50", round(row["test_map50"], 3), flush=True)
    return row


# ----------------------------------------------------------------------------------------------------------- RetinaNet
class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, frame_names, boxes_by_frame, train):
        self.frame_names, self.boxes_by_frame, self.train = frame_names, boxes_by_frame, train

    def __len__(self):
        return len(self.frame_names)

    def __getitem__(self, index):
        import torchvision.transforms.functional as F
        name = self.frame_names[index]
        image = F.to_tensor(Image.open(f"{frames_folder}/{name}").convert("RGB"))
        boxes = self.boxes_by_frame.get(name, np.zeros((0, 5)))
        xyxy = torch.tensor(boxes[:, :4], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(boxes[:, 4], dtype=torch.int64) + 1  # 0 is background
        if self.train and random.random() < 0.5:
            width = image.shape[-1]
            image = image.flip(-1)
            xyxy = torch.stack([width - xyxy[:, 2], xyxy[:, 1], width - xyxy[:, 0], xyxy[:, 3]], dim=1)
        return image, {"boxes": xyxy, "labels": labels}, name


def collate(batch):
    return tuple(zip(*batch))


def build_retinanet():
    """RetinaNet with the ResNet-50 + FPN design of SOCALNet's detector (v2 head). Backbone starts from
    torchvision's ImageNet ResNet-50, loaded from a safetensors file (no pickled code)."""
    from functools import partial
    import torchvision
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
    from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead, _default_anchorgen
    from torchvision.ops.feature_pyramid_network import LastLevelP6P7
    from torchvision.ops.misc import FrozenBatchNorm2d
    resnet = torchvision.models.resnet50(weights=None, norm_layer=FrozenBatchNorm2d)
    state = load_file(hf_hub_download("timm/resnet50.tv_in1k", "model.safetensors"))
    state = {k: v for k, v in state.items() if not k.startswith("fc.") and not k.endswith("num_batches_tracked")}
    missing, unexpected = resnet.load_state_dict(state, strict=False)
    assert not unexpected and all(k.startswith("fc.") for k in missing), (missing, unexpected)
    backbone = _resnet_fpn_extractor(resnet, 3, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(2048, 256))
    anchors = _default_anchorgen()
    head = RetinaNetHead(backbone.out_channels, anchors.num_anchors_per_location()[0], len(tool_types) + 1,
                         norm_layer=partial(torch.nn.GroupNorm, 32))
    head.regression_head._loss_type = "giou"
    return RetinaNet(backbone, num_classes=len(tool_types) + 1, anchor_generator=anchors, head=head,
                     min_size=640, max_size=1138, score_thresh=0.01, detections_per_img=300)


def retinanet_predictions(model, loader, run_device):
    model.eval()
    rows = []
    with torch.no_grad():
        for images, _targets, names in loader:
            outputs = model([image.to(run_device) for image in images])
            for name, output in zip(names, outputs):
                for box, label, score in zip(output["boxes"].tolist(), output["labels"].tolist(), output["scores"].tolist()):
                    if 1 <= label <= len(tool_types):
                        rows.append((name, tool_types[label - 1], score, *box))
    return pd.DataFrame(rows, columns=["frame", "label", "confidence", "x1", "y1", "x2", "y2"])


def train_retinanet(boxes, frames):
    run_folder = f"{models_root}/retinanet"
    score_file = f"{results_root}/runs/retinanet.json"
    if os.path.exists(score_file):
        return json.load(open(score_file))
    os.makedirs(run_folder, exist_ok=True)
    os.makedirs(f"{results_root}/runs", exist_ok=True)
    run_device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    tool_boxes = boxes[boxes["label"].isin(tool_types)].copy()
    tool_boxes["class"] = tool_boxes["label"].map(tool_types.index)
    boxes_by_frame = {f: g[["x1", "y1", "x2", "y2", "class"]].values.astype(float) for f, g in tool_boxes.groupby("frame")}
    train_frames = frames["train"][::2]  # every 2nd second: neighbouring seconds are near copies, RetinaNet is slow
    workers = 0 if quick else 4
    train_loader = torch.utils.data.DataLoader(FrameDataset(train_frames, boxes_by_frame, True), batch_size=8,
                                               shuffle=True, num_workers=workers, collate_fn=collate, drop_last=True)
    val_loader = torch.utils.data.DataLoader(FrameDataset(frames["val"], boxes_by_frame, False), batch_size=8,
                                             num_workers=workers, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(FrameDataset(frames["test"], boxes_by_frame, False), batch_size=8,
                                              num_workers=workers, collate_fn=collate)
    model = build_retinanet().to(run_device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=0.01, momentum=0.9, weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    drops = [int(retinanet_epochs * 2 / 3) + 1, int(retinanet_epochs * 11 / 12) + 1]  # epochs 9 and 12 of 12: rate x 0.1
    use_amp = run_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    val_truth = human_truth_boxes(frames["val"])
    best_val, history, started, step = -1.0, [], time.time(), 0
    for epoch in range(1, retinanet_epochs + 1):
        model.train()
        decay = 0.1 ** sum(epoch >= d for d in drops)
        for images, targets, _names in train_loader:
            for group in optimizer.param_groups:
                group["lr"] = 0.01 * decay * min(1.0, (step + 1) / 500)  # warm up over the first 500 steps
            images = [image.to(run_device) for image in images]
            targets = [{k: v.to(run_device) for k, v in t.items()} for t in targets]
            with torch.autocast("cuda", enabled=use_amp):
                loss = sum(model(images, targets).values())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 10.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if step % 200 == 0:
                print(f"retinanet epoch {epoch} step {step % steps_per_epoch}/{steps_per_epoch} loss {loss.item():.3f}",
                      flush=True)
        if epoch % 2 == 0 or epoch == retinanet_epochs:
            val_score = Scorer(retinanet_predictions(model, val_loader, run_device), val_truth).score()["map50"]
            history.append({"epoch": epoch, "val_map50": val_score})
            print(f"retinanet epoch {epoch} validation mAP50 {val_score:.3f}", flush=True)
            if val_score > best_val:
                best_val = val_score
                torch.save(model.state_dict(), f"{run_folder}/best.pt")
    model.load_state_dict(torch.load(f"{run_folder}/best.pt", map_location=run_device, weights_only=True))
    predictions = retinanet_predictions(model, test_loader, run_device)
    os.makedirs(f"{results_root}/test_predictions", exist_ok=True)
    predictions.to_csv(f"{results_root}/test_predictions/retinanet.csv.gz", index=False)
    test = Scorer(predictions, human_truth_boxes(frames["test"])).score()
    best_epoch = max(history, key=lambda h: h["val_map50"])["epoch"]
    row = {"name": "retinanet", "imgsz": 640, "val_map50": best_val, "best_epoch": best_epoch,
           "train_hours": (time.time() - started) / 3600, "train_images": len(train_frames),
           **{f"test_{k}": v for k, v in test.items()}}
    json.dump(row, open(score_file, "w"), indent=2)
    print("retinanet validation mAP50", round(best_val, 3), "test mAP50", round(test["map50"], 3), flush=True)
    return row


# --------------------------------------------------------------------------------------------------------------- stages
def best_of(rows, names):
    return max((r for r in rows if r["name"] in names), key=lambda r: r["val_map50"])["name"]


if __name__ == "__main__":
    stages = [a for a in sys.argv[1:] if not a.startswith("--")] or ["sizes", "resolution", "rare", "retinanet", "report"]
    boxes, frames = load_split()
    print({split: len(names) for split, names in frames.items()}, "frames (train / validation / test)", flush=True)
    data_folders = {"plain": build_data("plain", boxes, frames), "test": build_data("test", boxes, frames)}
    choices, scored = load_choices(), {}

    def run(name, start_weights, data_variant, imgsz):
        info = train_yolo(name, start_weights, data_folders[data_variant], imgsz)
        scored[name] = {**info, **score_yolo_run(name, f"{models_root}/{name}/weights/best.pt", imgsz, data_folders, frames)}
        return scored[name]

    baseline_weights = os.path.abspath(f"models/fold_detectors/fold{fold}/weights/best.pt")
    if os.path.exists(baseline_weights):  # the detector already used for the automatic labels (n, 640, every 4th frame, 20 rounds)
        scored["baseline"] = {"name": "baseline", "start_weights": "yolo11n.pt", "epochs": 20,
                              **score_yolo_run("baseline", baseline_weights, 640, data_folders, frames)}

    if "sizes" in stages or "resolution" in stages or "rare" in stages:
        for size in ["n", "s", "m"]:
            run(f"{size}_640", f"yolo11{size}.pt", "plain", 640)
        choices["phone_model"] = best_of(list(scored.values()), ["n_640", "s_640"])
        choices["largest_model_wins"] = best_of(list(scored.values()), ["n_640", "s_640", "m_640"]) == "m_640"
        save_choices(choices)

    if "resolution" in stages or "rare" in stages:
        size = choices["phone_model"][0]
        run(f"{size}_960", f"yolo11{size}.pt", "plain", 960)
        choices["phone_setup"] = best_of(list(scored.values()), [f"{size}_640", f"{size}_960"])
        save_choices(choices)

    if "rare" in stages:
        data_folders["rare"] = build_data("rare", boxes, frames)
        setup = choices["phone_setup"]
        size, imgsz = setup[0], int(setup.split("_")[1])
        run(f"{setup}_rare", f"yolo11{size}.pt", "rare", imgsz)
        choices["final_setup"] = best_of(list(scored.values()), [setup, f"{setup}_rare"])
        save_choices(choices)

    if "retinanet" in stages:
        scored["retinanet"] = train_retinanet(boxes, frames)

    if "report" in stages:
        for name in os.listdir(f"{results_root}/runs"):
            row = json.load(open(f"{results_root}/runs/{name}"))
            done = f"{models_root}/{row['name']}/done.json"
            scored.setdefault(row["name"], {**(json.load(open(done)) if os.path.exists(done) else {}), **row})
        test_truth = human_truth_boxes(frames["test"])
        scorers = {name: Scorer(pd.read_csv(f"{results_root}/test_predictions/{name}.csv.gz"), test_truth)
                   for name in scored}
        rows = []
        for name, row in scored.items():
            ranged = score_with_range(scorers[name])
            rows.append({**row, "our_test_map50": ranged["map50"], "our_test_map50_low": ranged["map50_low"],
                         "our_test_map50_high": ranged["map50_high"]})
        pd.DataFrame(rows).to_csv(f"{results_root}/runs.csv", index=False)
        gaps = []
        final = choices.get("final_setup")
        pairs = [(name, "baseline") for name in scored if name != "baseline" and "baseline" in scored]
        if final and "retinanet" in scored:
            pairs.append(("retinanet", final))
        for new, old in pairs:
            gap = paired_gap(scorers[new], scorers[old])
            gaps.append({"new": new, "old": old, **{f"{metric}_{k}": gap.loc[metric, k]
                                                    for metric in gap.index for k in ["gap", "low", "high"]}})
        pd.DataFrame(gaps).to_csv(f"{results_root}/gaps.csv", index=False)
        print(pd.DataFrame(rows)[["name", "val_map50", "test_map50", "our_test_map50", "our_test_map50_low",
                                  "our_test_map50_high"]].round(3).to_string(index=False))
        print("choices:", choices)
