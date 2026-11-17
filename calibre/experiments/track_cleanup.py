# Phase 2, Part A: clean up the detector's tool tracks before turning them into features.
#  1. Re-run each fold's detector on the surgeons it never saw, keeping weak boxes too (confidence >= 0.05).
#  2. For each tool pick (a) a confidence cutoff instead of the app's single 0.35, (b) how long a gap in
#     the tool's track to fill in, (c) how short a lone blip to drop. Judged by how well "is this tool on
#     screen this second?" matches the human labels (F1 score).
#  3. The settings used for one fold's surgeons are picked only on the OTHER four folds' surgeons, so the
#     human labels of the surgeons being scored are never looked at.
#  4. Write data/auto_boxes_clean.csv, which the outcome experiments read with the label set "clean".
#   uv run python -m calibre.experiments.track_cleanup [predict] [tune]
#   uv run python -m calibre.experiments.track_cleanup tune --v2    (Part C detectors -> data/auto_boxes_v2_clean.csv)
import os
import sys
import numpy as np
import pandas as pd
from calibre.experiments.surgeon_folds import load_surgeon_folds, n_folds
from calibre.models.tool_detector_training import frames_folder, tool_types

low_confidence = 0.05
predict_up_to_second = 180
cutoffs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
gap_fills = [0, 1, 2]     # fill a hole of up to this many seconds when the tool is seen on both sides
min_runs = [1, 2]         # drop a run of detections shorter than this many seconds
app_settings = {"cutoff": 0.35, "gap_fill": 0, "min_run": 1}
v2 = "--v2" in sys.argv  # Part C: tune on the final detectors' boxes (final_detectors.py label) instead of Phase 1's
low_folder = "data/auto_boxes_v2_low" if v2 else "data/auto_boxes_low"
clean_boxes_file = "data/auto_boxes_v2_clean.csv" if v2 else "data/auto_boxes_clean.csv"
result_tag = "_v2" if v2 else ""


def predict_low_confidence(fold, frame_to_trial, folds):
    from ultralytics import YOLO  # type: ignore
    from calibre.experiments.fold_detectors import device
    test_surgeons = set(folds.loc[folds["fold"] == fold, "participant_id"])
    frames = frame_to_trial[frame_to_trial["trial_id"].str[1:4].astype(int).isin(test_surgeons)
                            & (frame_to_trial["frame_number"] <= predict_up_to_second)]
    detector = YOLO(os.path.abspath(f"models/fold_detectors/fold{fold}/weights/best.pt"))
    rows, frame_names = [], frames["frame"].tolist()
    for start in range(0, len(frame_names), 64):
        batch = frame_names[start:start + 64]
        results = detector.predict([f"{frames_folder}/{name}" for name in batch], conf=low_confidence,
                                   imgsz=640, device=device, verbose=False)
        for frame_name, result in zip(batch, results):
            rows.append({"frame": frame_name, "label": None})  # every frame listed, so empty seconds count
            for box, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
                rows.append({"frame": frame_name, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                             "label": tool_types[int(cls)], "confidence": conf})
    return pd.DataFrame(rows)


def runs(flags):
    """(start, end, value) for each run of equal values, end exclusive."""
    out, start = [], 0
    for i in range(1, len(flags) + 1):
        if i == len(flags) or flags[i] != flags[start]:
            out.append((start, i, flags[start]))
            start = i
    return out


def clean_presence(present, gap_fill, min_run):
    present = present.copy()
    for start, end, value in runs(present):
        if value and end - start < min_run:
            present[start:end] = False
    for start, end, value in runs(present):
        if not value and start > 0 and end < len(present) and end - start <= gap_fill:
            present[start:end] = True
    return present


def presence_table(boxes, frames):
    """One row per (trial, second), one True/False column per tool."""
    table = frames[["trial_id", "frame_number"]].drop_duplicates().copy()
    for tool in tool_types:
        seen = boxes.loc[boxes["label"] == tool, ["trial_id", "frame_number"]].drop_duplicates()
        table[tool] = pd.MultiIndex.from_frame(table[["trial_id", "frame_number"]]).isin(
            pd.MultiIndex.from_frame(seen))
    return table.sort_values(["trial_id", "frame_number"]).reset_index(drop=True)


def cleaned_tool_presence(auto, frames, tool, settings):
    kept = auto[(auto["label"] == tool) & (auto["confidence"] >= settings["cutoff"])]
    raw = presence_table(kept, frames)
    out = []
    for _, trial in raw.groupby("trial_id", sort=True):
        out.append(clean_presence(trial[tool].values, settings["gap_fill"], settings["min_run"]))
    return np.concatenate(out) if out else np.array([], bool)


def f1(truth, guess):
    hits = (truth & guess).sum()
    return 2 * hits / max(truth.sum() + guess.sum(), 1)


def pick_settings(auto, truth, frames, tool):
    best, best_f1 = None, -1
    for cutoff in cutoffs:
        for gap_fill in gap_fills:
            for min_run in min_runs:
                settings = {"cutoff": cutoff, "gap_fill": gap_fill, "min_run": min_run}
                score = f1(truth[tool].values, cleaned_tool_presence(auto, frames, tool, settings))
                if score > best_f1:
                    best, best_f1 = settings, score
    return best


def apply_settings(auto, frames, settings_by_tool):
    """Boxes after cleanup: weak boxes dropped, blips dropped, short gaps filled with an in-between box."""
    kept_rows = [auto[auto["label"].isna()][["frame", "label"]]]  # the empty row for every watched frame
    frame_of = frames.set_index(["trial_id", "frame_number"])["frame"]
    for tool in tool_types:
        settings = settings_by_tool[tool]
        boxes = auto[(auto["label"] == tool) & (auto["confidence"] >= settings["cutoff"])]
        best_box = boxes.sort_values("confidence").groupby(["trial_id", "frame_number"]).tail(1)
        best_box = best_box.set_index(["trial_id", "frame_number"])
        raw = presence_table(boxes, frames)
        for trial_id, trial in raw.groupby("trial_id", sort=True):
            seconds = trial["frame_number"].values
            before, after = trial[tool].values, clean_presence(trial[tool].values, settings["gap_fill"], settings["min_run"])
            dropped = set(seconds[before & ~after])
            trial_boxes = boxes[(boxes["trial_id"] == trial_id) & ~boxes["frame_number"].isin(dropped)]
            kept_rows.append(trial_boxes[["frame", "x1", "y1", "x2", "y2", "label", "confidence"]])
            for i in np.where(after & ~before)[0]:
                left = max(j for j in range(i) if before[j] and after[j])
                right = min(j for j in range(i + 1, len(seconds)) if before[j] and after[j])
                share = (i - left) / (right - left)
                a, b = best_box.loc[(trial_id, seconds[left])], best_box.loc[(trial_id, seconds[right])]
                filled = {c: a[c] + share * (b[c] - a[c]) for c in ["x1", "y1", "x2", "y2"]}
                kept_rows.append(pd.DataFrame([{"frame": frame_of[(trial_id, seconds[i])], **filled, "label": tool,
                                                "confidence": np.nan}]))
    return pd.concat(kept_rows, ignore_index=True)


def load_low_confidence(frame_to_trial):
    auto = pd.concat([pd.read_csv(f"{low_folder}/fold{fold}.csv") for fold in range(n_folds)])
    return auto.merge(frame_to_trial, on="frame", how="left")


if __name__ == "__main__":
    parts = [a for a in sys.argv[1:] if not a.startswith("--")] or (["tune"] if v2 else ["predict", "tune"])
    assert not (v2 and "predict" in parts), "with --v2 the boxes come from final_detectors.py label"
    folds = load_surgeon_folds()
    frame_to_trial = pd.read_csv("data/frame_to_trial_mapping.csv")

    if "predict" in parts:
        os.makedirs("data/auto_boxes_low", exist_ok=True)
        for fold in range(n_folds):
            out_file = f"data/auto_boxes_low/fold{fold}.csv"
            if not os.path.exists(out_file):
                predict_low_confidence(fold, frame_to_trial, folds).to_csv(out_file, index=False)
            print(f"fold {fold} predicted", flush=True)

    if "tune" in parts:
        auto = load_low_confidence(frame_to_trial)
        auto["participant_id"] = auto["trial_id"].str[1:4].astype(int)
        fold_of = folds.set_index("participant_id")["fold"]
        auto["fold"] = auto["participant_id"].map(fold_of)
        frames = auto[["frame", "trial_id", "frame_number", "fold"]].drop_duplicates("frame")
        human = pd.read_csv("data/socal.csv", header=None, names=["frame", "x1", "y1", "x2", "y2", "label"])
        human = human.merge(frame_to_trial, on="frame")
        truth = presence_table(human[human["frame"].isin(frames["frame"])], frames)
        truth["fold"] = truth["trial_id"].str[1:4].astype(int).map(fold_of)

        chosen, cleaned_parts, f1_rows = [], [], []
        for fold in range(n_folds):
            train_auto, test_auto = auto[auto["fold"] != fold], auto[auto["fold"] == fold]
            train_frames, test_frames = frames[frames["fold"] != fold], frames[frames["fold"] == fold]
            train_truth = truth[truth["fold"] != fold].reset_index(drop=True)
            settings = {tool: pick_settings(train_auto, train_truth, train_frames, tool) for tool in tool_types}
            for tool, s in settings.items():
                chosen.append({"fold": fold, "tool": tool, **s})
            cleaned_parts.append(apply_settings(test_auto, test_frames, settings))
            print(f"fold {fold} settings:", {t: tuple(s.values()) for t, s in settings.items()}, flush=True)

        # per-second presence agreement with the human labels, every surgeon scored with settings picked without them
        for tool in tool_types:
            app_guess, clean_guess = [], []
            for fold in range(n_folds):
                test_auto, test_frames = auto[auto["fold"] == fold], frames[frames["fold"] == fold]
                fold_settings = next(c for c in chosen if c["fold"] == fold and c["tool"] == tool)
                app_guess.append(cleaned_tool_presence(test_auto, test_frames, tool, app_settings))
                clean_guess.append(cleaned_tool_presence(test_auto, test_frames, tool,
                                                         {k: fold_settings[k] for k in app_settings}))
            truth_by_fold = np.concatenate([truth.loc[truth["fold"] == fold, tool].values for fold in range(n_folds)])
            app_guess, clean_guess = np.concatenate(app_guess), np.concatenate(clean_guess)
            f1_rows.append({"tool": tool, "seconds": len(truth_by_fold), "seconds_tool_on_screen": truth_by_fold.sum(),
                            "f1_app_setting": f1(truth_by_fold, app_guess), "f1_cleaned": f1(truth_by_fold, clean_guess)})

        os.makedirs("results", exist_ok=True)
        pd.DataFrame(chosen).to_csv(f"results/clean_track_settings{result_tag}.csv", index=False)
        f1_table = pd.DataFrame(f1_rows)
        f1_table.to_csv(f"results/clean_track_presence_f1{result_tag}.csv", index=False)
        print(f1_table.round(3).to_string(index=False))
        cleaned = pd.concat(cleaned_parts, ignore_index=True).sort_values("frame", kind="stable")
        cleaned[["frame", "label", "x1", "y1", "x2", "y2", "confidence"]].to_csv(clean_boxes_file, index=False)
        print("wrote", clean_boxes_file, len(cleaned), "rows")
