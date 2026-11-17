# Phase 3.1: how the tools MOVE in the first minute, not only whether they are on screen.
# Built from per-second tool boxes (human or detector, whichever label set is used) and the location of the
# blood in each frame (same colour rule as calibre/components/blood.py). Positions are fractions of the frame
# width, so videos of different sizes compare. A feature is left empty (NaN) when the tool never shows up;
# with_missing_filled then fills it from the training trials only and adds a "was missing" flag.
#   uv run python -m calibre.experiments.motion_features      (writes data/blood_location.csv, about 5 min)
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image
from calibre.components.blood import max_brightness, min_brightness_in_view, min_saturation
from calibre.components.vid import load_tool_boxes
from calibre.models.tool_detector_training import frames_folder

watch_seconds = 60
blood_location_file = "data/blood_location.csv"
idle_move = 0.01      # moving less than 1% of the frame width in a second counts as holding still
close_together = 0.15  # suction and grasper centres within 15% of the frame width = working at one spot

motion_feature_names = [
    "suction_path",          # total distance the suction travelled (frame widths)
    "suction_speed_spread",  # how uneven the suction movement is (sd of per-second distance)
    "suction_idle_frac",     # share of seconds the suction is on screen but holding still
    "grasper_speed",         # mean grasper movement per second
    "grasper_idle_frac",
    "suction_to_blood",      # mean distance from the suction to the middle of the blood
    "suction_on_blood_frac", # share of suction seconds with the middle of the blood inside the suction box (+25%)
    "tools_close_frac",      # share of seconds with suction and grasper working close together
    "blood_spot_drift",      # how far the middle of the blood moves per second (spreading or moving bleed)
]


def blood_location(image_path):
    """Middle of the blood region and how big it is, in fractions of the frame (same rule as blood_amount)."""
    image = Image.open(image_path).resize((480, 270))
    hsv = np.asarray(image.convert("HSV")).astype(np.float32) / 255
    hue, saturation, brightness = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    inside_view = brightness > min_brightness_in_view
    is_red = (hue < 0.04) | (hue > 0.94)
    is_blood = inside_view & is_red & (saturation > min_saturation) & (brightness < max_brightness)
    if is_blood.sum() < 20:  # a few stray pixels are not a bleed
        return np.nan, np.nan
    rows, cols = np.nonzero(is_blood)
    return cols.mean() / 480, rows.mean() / 270


def make_blood_locations(watch=watch_seconds):
    frames = pd.read_csv("data/frame_to_trial_mapping.csv")
    frames = frames[frames["frame_number"] <= watch]
    with ProcessPoolExecutor() as workers:
        spots = list(workers.map(blood_location, [f"{frames_folder}/{n}" for n in frames["frame"]], chunksize=64))
    frames[["blood_x", "blood_y"]] = np.array(spots)
    frames.to_csv(blood_location_file, index=False)
    return frames


def best_box_per_second(boxes):
    """One box per tool per second: the most confident one (human boxes have no confidence, so the first)."""
    boxes = boxes.dropna(subset=["label"]).copy()
    if "confidence" not in boxes:
        boxes["confidence"] = 1.0
    boxes["confidence"] = boxes["confidence"].fillna(1.0)
    return boxes.sort_values("confidence", ascending=False).drop_duplicates(["frame_number", "label"])


def tool_track(best, tool, width, height):
    """second -> (x, y, half width, half height) of the tool's box, as fractions of the frame."""
    t = best[best["label"] == tool]
    return pd.DataFrame({"x": (t["x1"] + t["x2"]).values / 2 / width, "y": (t["y1"] + t["y2"]).values / 2 / height,
                         "half_w": (t["x2"] - t["x1"]).values / 2 / width, "half_h": (t["y2"] - t["y1"]).values / 2 / height},
                        index=t["frame_number"].values).sort_index()


def step_distances(track):
    """Distance moved between back-to-back seconds only (a tool leaving the screen is not a move)."""
    if len(track) < 2:
        return np.array([])
    moves = np.hypot(np.diff(track["x"].values), np.diff(track["y"].values))
    return moves[np.diff(track.index.values) == 1]


def trial_motion_features(boxes, blood, width, height):
    best = best_box_per_second(boxes)
    suction, grasper = tool_track(best, "suction", width, height), tool_track(best, "grasper", width, height)
    suction_moves, grasper_moves = step_distances(suction), step_distances(grasper)
    spot = blood.set_index("frame_number")[["blood_x", "blood_y"]].dropna().sort_index()
    nan = np.nan

    both = suction.join(spot, how="inner")
    if len(both):
        distance = np.hypot(both["x"] - both["blood_x"], both["y"] - both["blood_y"])
        on_blood = ((both["blood_x"] - both["x"]).abs() <= 1.25 * both["half_w"]) & \
                   ((both["blood_y"] - both["y"]).abs() <= 1.25 * both["half_h"])
    pair = suction.join(grasper, how="inner", lsuffix="_s", rsuffix="_g")
    spot_moves = step_distances(spot.rename(columns={"blood_x": "x", "blood_y": "y"}))
    return {
        "suction_path": suction_moves.sum() if len(suction) else nan,
        "suction_speed_spread": suction_moves.std() if len(suction_moves) > 1 else nan,
        "suction_idle_frac": (suction_moves < idle_move).mean() if len(suction_moves) else nan,
        "grasper_speed": grasper_moves.mean() if len(grasper_moves) else nan,
        "grasper_idle_frac": (grasper_moves < idle_move).mean() if len(grasper_moves) else nan,
        "suction_to_blood": distance.mean() if len(both) else nan,
        "suction_on_blood_frac": on_blood.mean() if len(both) else nan,
        "tools_close_frac": (np.hypot(pair["x_s"] - pair["x_g"], pair["y_s"] - pair["y_g"]) < close_together).sum()
        / watch_seconds if len(suction) else nan,
        "blood_spot_drift": spot_moves.mean() if len(spot_moves) else nan,
    }


def add_motion_features(table, boxes_file):
    if not os.path.exists(blood_location_file):
        make_blood_locations()
    blood = pd.read_csv(blood_location_file)
    boxes = load_tool_boxes(boxes_file)
    boxes = boxes[boxes["frame_number"] <= watch_seconds]
    sizes = pd.read_csv("data/socal_outcomes.csv").set_index("trial_id")[["trial_video_width", "trial_video_height"]]
    blood_groups, box_groups = dict(list(blood.groupby("trial_id"))), dict(list(boxes.groupby("trial_id")))
    empty_boxes = pd.DataFrame(columns=["frame_number", "label", "x1", "y1", "x2", "y2"])
    empty_blood = pd.DataFrame(columns=["frame_number", "blood_x", "blood_y"])
    rows = []
    for trial_id in table["trial_id"]:
        width = sizes["trial_video_width"].get(trial_id, np.nan)
        height = sizes["trial_video_height"].get(trial_id, np.nan)
        width, height = (1920 if np.isnan(width) else width), (1080 if np.isnan(height) else height)
        rows.append({"trial_id": trial_id, **trial_motion_features(box_groups.get(trial_id, empty_boxes),
                                                                   blood_groups.get(trial_id, empty_blood), width, height)})
    return table.merge(pd.DataFrame(rows), on="trial_id")


if __name__ == "__main__":
    spots = make_blood_locations()
    print(spots[["blood_x", "blood_y"]].describe().round(3))
