# Phase 2, Part C: the final detector setup, trained once per surgeon fold, then used to label only the
# surgeons it never saw. The setup was picked on fold 0 in detector_search.py (YOLO11s, 960 px, rare-tool
# repeats + pasted crops, 30 rounds); nothing about it is re-tuned here.
#   train   fold k: train on folds != k (same validation surgeons and recipe as fold 0), save test predictions
#   label   every fold: label its unseen surgeons up to 180 s, keeping weak boxes (data/auto_boxes_v2_low/)
#           and the app cutoff (data/auto_boxes_v2.csv, label set "auto2")
#   score   5-fold detector table, plus old (Phase 1) vs new detector paired on the same 75 surgeons
#   uv run python -m calibre.experiments.final_detectors train 1 2 3 4    (GPU; one fold per call suits job arrays)
#   uv run python -m calibre.experiments.final_detectors label            (GPU, about 10 min per fold)
#   uv run python -m calibre.experiments.final_detectors score            (CPU)
#   then: uv run python -m calibre.experiments.track_cleanup tune --v2    (label set "clean2")
#   --quick runs a tiny version of everything for a smoke test (separate folders, nothing real is touched)
import json
import os
import shutil
import sys
import pandas as pd
from calibre.experiments import detector_search as search
from calibre.experiments.detection_scoring import Scorer, human_truth_boxes, paired_gap, score_with_range
from calibre.experiments.surgeon_folds import load_surgeon_folds, n_folds
from calibre.models.tool_detector_training import frames_folder, tool_types

quick = search.quick
suffix = "_quick" if quick else ""
setup_name, start_weights, imgsz = "s_960_rare", "yolo11s.pt", 960
low_confidence, app_confidence, predict_up_to_second = 0.05, 0.35, 180
results_root = f"results/final_detectors{suffix}"
low_folder = f"data/auto_boxes_v2_low{suffix}"
auto_file = f"data/auto_boxes_v2{suffix}.csv"
fold0_search_weights = os.path.abspath(f"models/detector_search/{setup_name}/weights/best.pt")


def fold_paths(fold):
    return (os.path.abspath(f"data/yolo_final{suffix}/fold{fold}"),
            os.path.abspath(f"models/final_detectors{suffix}/fold{fold}"))


def weights_of(fold):
    return f"{fold_paths(fold)[1]}/{setup_name}/weights/best.pt"


def train_fold(fold):
    data_root, models_root = fold_paths(fold)
    search.configure(fold, data_root, models_root, results_root)
    boxes, frames = search.load_split()
    print(f"fold {fold}:", {split: len(names) for split, names in frames.items()}, "frames", flush=True)
    if fold == 0 and not quick and os.path.exists(fold0_search_weights) and not os.path.exists(weights_of(0)):
        # fold 0 was already trained with exactly this split and setup in the detector search
        os.makedirs(os.path.dirname(weights_of(0)), exist_ok=True)
        shutil.copy(fold0_search_weights, weights_of(0))
        print("fold 0: reusing the detector-search weights", flush=True)
    if not os.path.exists(weights_of(fold)):
        rare_folder = search.build_data("rare", boxes, frames)
        search.train_yolo(setup_name, start_weights, rare_folder, imgsz)
    prediction_file = f"{results_root}/test_predictions/fold{fold}.csv.gz"
    if not os.path.exists(prediction_file):
        os.makedirs(os.path.dirname(prediction_file), exist_ok=True)
        search.yolo_predictions(weights_of(fold), frames["test"], imgsz).to_csv(prediction_file, index=False)
    print(f"fold {fold} trained and test predictions saved", flush=True)


def label_fold(fold, frame_to_trial, folds):
    """Every second up to 180 s of the surgeons this fold's detector never saw, boxes down to 0.05 confidence.
    Every frame gets one empty row too, so seconds with no tool still count as watched."""
    from ultralytics import YOLO  # type: ignore
    test_surgeons = set(folds.loc[folds["fold"] == fold, "participant_id"])
    frames = frame_to_trial[frame_to_trial["trial_id"].str[1:4].astype(int).isin(test_surgeons)
                            & (frame_to_trial["frame_number"] <= predict_up_to_second)]
    names = frames["frame"].tolist()[:: 40 if quick else 1]
    detector, rows = YOLO(weights_of(fold)), []
    for start in range(0, len(names), 32):
        batch = names[start:start + 32]
        results = detector.predict([f"{frames_folder}/{n}" for n in batch], conf=low_confidence, imgsz=imgsz,
                                   device=search.device, verbose=False)
        for frame_name, result in zip(batch, results):
            rows.append({"frame": frame_name, "label": None})
            for box, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
                rows.append({"frame": frame_name, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                             "label": tool_types[int(cls)], "confidence": conf})
    return pd.DataFrame(rows)


def score_folds(folds):
    """Each fold's new detector on its own unseen surgeons; then all 75 surgeons pooled, old vs new, paired."""
    rows, new_parts, old_parts, truth_parts = [], [], [], []
    for fold in range(n_folds):
        prediction_file = f"{results_root}/test_predictions/fold{fold}.csv.gz"
        if not os.path.exists(prediction_file):
            print(f"fold {fold}: no test predictions yet, skipped")
            continue
        new = pd.read_csv(prediction_file)
        data_root, models_root = fold_paths(fold)
        search.configure(fold, data_root, models_root, results_root)
        _, frames = search.load_split()
        truth = human_truth_boxes(frames["test"])
        rows.append({"fold": fold, "frames": len(frames["test"]), **score_with_range(Scorer(new, truth))})
        new_parts.append(new)
        truth_parts.append(truth)
        old_file = f"{results_root}/old_predictions/fold{fold}.csv.gz"
        if not os.path.exists(old_file):  # the Phase 1 detector of the same fold, scored the same way
            old_weights = os.path.abspath(f"models/fold_detectors/fold{fold}/weights/best.pt")
            if os.path.exists(old_weights):
                os.makedirs(os.path.dirname(old_file), exist_ok=True)
                search.yolo_predictions(old_weights, frames["test"], 640).to_csv(old_file, index=False)
        if os.path.exists(old_file):
            old_parts.append(pd.read_csv(old_file))
        print(rows[-1], flush=True)
    if not rows:
        return
    table = pd.DataFrame(rows)
    average = table.drop(columns="fold").mean().to_dict()
    table = pd.concat([table, pd.DataFrame([{**average, "fold": "average"}])])
    truth = pd.concat(truth_parts)
    pooled_new = Scorer(pd.concat(new_parts), truth)
    table = pd.concat([table, pd.DataFrame([{"fold": "pooled", "frames": sum(r["frames"] for r in rows),
                                             **score_with_range(pooled_new)}])])
    table.to_csv(f"{results_root}/folds.csv", index=False)
    print(table.round(3).to_string(index=False))
    if len(old_parts) == len(new_parts):
        gap = paired_gap(pooled_new, Scorer(pd.concat(old_parts), truth))
        gap.to_csv(f"{results_root}/new_minus_old.csv")
        print("new minus old (Phase 1) detector, same surgeons:\n", gap.round(3).to_string())


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    step, chosen = (args[0] if args else "train"), [int(a) for a in args[1:]]
    folds = load_surgeon_folds()
    os.makedirs(results_root, exist_ok=True)
    json.dump({"setup": setup_name, "start_weights": start_weights, "imgsz": imgsz, "epochs": search.epochs,
               "low_confidence": low_confidence, "app_confidence": app_confidence},
              open(f"{results_root}/setup.json", "w"), indent=2)

    if step == "train":
        for fold in chosen or range(n_folds):
            train_fold(fold)

    if step == "label":
        frame_to_trial = pd.read_csv("data/frame_to_trial_mapping.csv")
        os.makedirs(low_folder, exist_ok=True)
        for fold in chosen or range(n_folds):
            out_file = f"{low_folder}/fold{fold}.csv"
            if not os.path.exists(weights_of(fold)):
                print(f"fold {fold}: not trained yet, skipped")
                continue
            if not os.path.exists(out_file):
                label_fold(fold, frame_to_trial, folds).to_csv(out_file, index=False)
            print(f"fold {fold} labeled", flush=True)
        parts = [pd.read_csv(f"{low_folder}/fold{f}.csv") for f in range(n_folds) if os.path.exists(f"{low_folder}/fold{f}.csv")]
        if len(parts) == n_folds:  # the app-cutoff label set only once every surgeon is covered
            low = pd.concat(parts)
            kept = low[low["label"].isna() | (low["confidence"] >= app_confidence)]
            kept.to_csv(auto_file, index=False)
            print("wrote", auto_file, len(kept), "rows")

    if step == "score":
        score_folds(folds)
