# Fully automatic test: train one tool detector per surgeon fold, then let each detector
# label only the surgeons it never saw. The result (data/auto_boxes.csv) replaces the
# hand-drawn SOCAL boxes, so no human labels are used for any trial we score.
import os
import random
import shutil
import sys
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO  # type: ignore
from calibre.experiments.surgeon_folds import load_surgeon_folds, n_folds
from calibre.models.tool_detector_training import frames_folder, tool_types, use_every_nth_frame

predict_up_to_second = 180  # enough for the 60 s model and the accuracy-over-time test
app_confidence = 0.35       # same cutoff the iPhone app uses
device = 0 if torch.cuda.is_available() else "mps"  # NVIDIA GPU on the PC, Apple GPU on the Mac


def link_or_copy(source, destination):
    try:
        os.symlink(source, destination)
    except OSError:  # Windows blocks symlinks by default
        shutil.copy(source, destination)


def build_fold_data(fold, tool_boxes, folds):
    data_folder = os.path.abspath(f"data/yolo_folds/fold{fold}")
    train_surgeons = sorted(folds.loc[folds["fold"] != fold, "participant_id"])
    random.Random(fold).shuffle(train_surgeons)
    val_surgeons = set(train_surgeons[: len(train_surgeons) // 8])  # only for picking the best epoch

    shutil.rmtree(data_folder, ignore_errors=True)
    for split in ["train", "val"]:
        os.makedirs(f"{data_folder}/images/{split}")
        os.makedirs(f"{data_folder}/labels/{split}")

    boxes_by_frame = tool_boxes.groupby("frame")
    fold_frames = tool_boxes.loc[tool_boxes["participant_id"].isin(train_surgeons), "frame"].unique()
    for frame_name in sorted(fold_frames)[::use_every_nth_frame]:
        split = "val" if int(frame_name[1:4]) in val_surgeons else "train"
        width, height = Image.open(f"{frames_folder}/{frame_name}").size
        label_lines = []
        for box in boxes_by_frame.get_group(frame_name).itertuples():
            if box.label not in tool_types:
                continue
            center_x, center_y = (box.x1 + box.x2) / 2 / width, (box.y1 + box.y2) / 2 / height
            box_width, box_height = (box.x2 - box.x1) / width, (box.y2 - box.y1) / height
            label_lines.append(f"{tool_types.index(box.label)} {center_x:.5f} {center_y:.5f} "
                               f"{box_width:.5f} {box_height:.5f}")
        link_or_copy(f"{frames_folder}/{frame_name}", f"{data_folder}/images/{split}/{frame_name}")
        with open(f"{data_folder}/labels/{split}/{frame_name.replace('.jpeg', '.txt')}", "w") as f:
            f.write("\n".join(label_lines))

    with open(f"{data_folder}/data.yaml", "w") as f:
        f.write(f"path: {data_folder}\ntrain: images/train\nval: images/val\nnames: {tool_types}\n")
    return data_folder


def label_unseen_surgeons(fold, weights, frame_to_trial, folds):
    test_surgeons = set(folds.loc[folds["fold"] == fold, "participant_id"])
    frames = frame_to_trial[frame_to_trial["trial_id"].str[1:4].astype(int).isin(test_surgeons)
                            & (frame_to_trial["frame_number"] <= predict_up_to_second)]
    detector = YOLO(weights)
    rows = []
    frame_names = frames["frame"].tolist()
    for start in range(0, len(frame_names), 64):
        batch = frame_names[start:start + 64]
        results = detector.predict([f"{frames_folder}/{name}" for name in batch], conf=app_confidence,
                                   imgsz=640, device=device, verbose=False)
        for frame_name, result in zip(batch, results):
            if len(result.boxes) == 0:
                rows.append({"frame": frame_name, "label": None})  # empty frame still counts as watched
            for box, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                      result.boxes.conf.tolist()):
                rows.append({"frame": frame_name, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                             "label": tool_types[int(cls)], "confidence": conf})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    only_folds = [int(f) for f in sys.argv[1:]] or range(n_folds)
    folds = load_surgeon_folds()
    tool_boxes = pd.read_csv("data/socal.csv", header=None, names=["frame", "x1", "y1", "x2", "y2", "label"])
    tool_boxes["participant_id"] = tool_boxes["frame"].str[1:4].astype(int)
    frame_to_trial = pd.read_csv("data/frame_to_trial_mapping.csv")
    os.makedirs("data/auto_boxes", exist_ok=True)

    for fold in only_folds:
        out_file = f"data/auto_boxes/fold{fold}.csv"
        if os.path.exists(out_file):
            print(f"fold {fold} already done")
            continue
        weights = os.path.abspath(f"models/fold_detectors/fold{fold}/weights/best.pt")
        if not os.path.exists(weights):
            data_folder = build_fold_data(fold, tool_boxes, folds)
            YOLO("yolo11n.pt").train(data=f"{data_folder}/data.yaml", epochs=20, imgsz=640, batch=32,
                                     device=device, workers=8, project=os.path.abspath("models/fold_detectors"),
                                     name=f"fold{fold}", exist_ok=True, plots=False, verbose=False)
        label_unseen_surgeons(fold, weights, frame_to_trial, folds).to_csv(out_file, index=False)
        print(f"fold {fold} done", flush=True)

    pd.concat([pd.read_csv(f"data/auto_boxes/fold{fold}.csv") for fold in range(n_folds)
               if os.path.exists(f"data/auto_boxes/fold{fold}.csv")]).to_csv("data/auto_boxes.csv", index=False)
