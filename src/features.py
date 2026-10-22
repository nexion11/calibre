"""Step 2: turn each trial's video labels into a few numbers ("features")
that a model can learn from.

Run it with:  uv run python src/features.py

IMPORTANT RULE: BleedWatch has to predict *early*, so we only look at the
first WINDOW seconds of each video. If we used the whole video, the model
could "cheat": for example, the moment muscle appears is basically the
moment the bleeding stops, which gives away the answer.
"""

#only looking and first few sections of video so model knows how and when to predict bleeding
import os

import numpy as np
import pandas as pd

WINDOW = 60  # seconds of video the model is allowed to see (1 frame = 1 second)


def load_boxes():
    """Load every tool box, tagged with its trial and its time in the video."""
    boxes = pd.read_csv(
        "data/socal.csv",
        header=None,
        names=["frame", "x1", "y1", "x2", "y2", "label"],
    )
    mapping = pd.read_csv("data/frame_to_trial_mapping.csv")
    return boxes.merge(mapping, on="frame", how="left")


def suction_speed(trial_boxes, video_width):
    """How far the suction tip moves per second, on average.
    Fast, jumpy suction may mean the surgeon is struggling to see."""
    suction = trial_boxes[trial_boxes["label"] == "suction"].copy()

    suction["cx"] = (suction["x1"] + suction["x2"]) / 2 / video_width
    suction["cy"] = (suction["y1"] + suction["y2"]) / 2 / video_width

    per_frame = suction.groupby("frame_number")[["cx", "cy"]].mean().sort_index()
    dx = per_frame["cx"].diff()
    dy = per_frame["cy"].diff()

    distance = (dx**2 + dy**2) ** 0.5
    gap = per_frame.index.to_series().diff()
    distance = distance[gap == 1]
    if len(distance) == 0:
        return 0 
    return distance.mean()


def blood_shape(early_blood, window=WINDOW):
    """How the blood changes over the window, not just its average:
    the worst moment, whether it's rising or falling, and where it ends up."""
    rows = []
    for trial_id, b in early_blood.groupby("trial_id"):
        b = b.sort_values("frame_number")
        # Slope of a straight line through blood vs. time: > 0 means it's getting worse.
        trend = np.polyfit(b["frame_number"], b["blood_frac"], 1)[0] if len(b) > 1 else 0
        rows.append({
            "trial_id": trial_id,
            "blood_max": b["blood_frac"].max(),
            "blood_trend": trend,
            "blood_late": b.loc[b["frame_number"] > window - 15, "blood_frac"].mean(),
        })
    return pd.DataFrame(rows).fillna(0)


def trial_features(trial_boxes, video_width):
    """Given the boxes for ONE trial (first WINDOW seconds only),
    return a dictionary of features describing it."""
    n_frames = trial_boxes["frame_number"].nunique()

    # Example feature (done for you):
    # in what fraction of frames is the suction visible?
    suction_frames = trial_boxes.loc[trial_boxes["label"] == "suction", "frame_number"].nunique()
    suction_frac = suction_frames / n_frames

    cottonoid_frames = trial_boxes.loc[
        trial_boxes["label"] == "cottonoid", 
        "frame_number",
    ].nunique()
    cottonoid_frac = cottonoid_frames / n_frames 

    tool_boxes = trial_boxes.dropna(subset=["label"])
    tools_per_frame = len(tool_boxes) / n_frames

    return {
        "suction_frac": suction_frac,
        "cottonoid_frac": cottonoid_frac,
        "tools_per_frame": tools_per_frame,
        "suction_speed": suction_speed(trial_boxes, video_width),
    }


def build_feature_table(window=WINDOW, start=0):
    """One row per trial, using video from second `start` up to second `window`."""
    boxes = load_boxes()
    early = boxes[(boxes["frame_number"] > start) & (boxes["frame_number"] <= window)]
    outcomes = pd.read_csv("data/socal_outcomes.csv")

    # Video width per trial. A few trials don't list it; most videos are 1920 wide.
    widths = outcomes.set_index("trial_id")["trial_video_width"].fillna(1920)

    rows = []
    for trial_id, trial_boxes in early.groupby("trial_id"):
        feats = trial_features(trial_boxes, widths.get(trial_id, 1920))
        feats["trial_id"] = trial_id
        rows.append(feats)
    features = pd.DataFrame(rows)

    # Blood coverage comes from the Colab notebook. Skip it until that file exists.
    blood_path = "data/blood_coverage.csv"
    if os.path.exists(blood_path):
        blood = pd.read_csv(blood_path)
        early_blood = blood[(blood["frame_number"] > start) & (blood["frame_number"] <= window)]
        # blood_mean: each trial's average blood coverage in the window.
        blood_mean = early_blood.groupby("trial_id")["blood_frac"].mean().reset_index(name="blood_mean")
        features = features.merge(blood_mean, on="trial_id")
        features = features.merge(blood_shape(early_blood, window), on="trial_id")

    # How long each video runs, so analyses can skip trials that ended before `window`.
    lengths = boxes.groupby("trial_id")["frame_number"].max().rename("video_seconds")
    features = features.merge(lengths, on="trial_id")

    return features.merge(outcomes[["trial_id", "blood_loss", "success"]], on="trial_id")


if __name__ == "__main__":
    table = build_feature_table()
    print(table.head(10))
    print("\nTrials in table:", len(table))
    print("\nHow each feature relates to blood loss (correlation, -1 to 1):")
    print(table.drop(columns=["trial_id"]).corr()["blood_loss"].round(2))
