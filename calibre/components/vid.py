import os
import numpy as np
import pandas as pd

watch_seconds = 60  # only wacth 60 sec
default_vid_width = 1920 


def load_tool_boxes():
    tool_boxes = pd.read_csv(
        "data/socal.csv",
        header=None,
        names=["frame", "x1", "y1", "x2", "y2", "label"],
    )
    
    frame_to_trial = pd.read_csv("data/frame_to_trial_mapping.csv")
    return tool_boxes.merge(frame_to_trial, on="frame", how="left")



def suction_speed(trial_boxes, video_width):
    suction = trial_boxes[trial_boxes["label"] == "suction"].copy()

    
    suction["center_x"] = (suction["x1"] + suction["x2"]) / 2 / video_width
    suction["center_y"] = (suction["y1"] + suction["y2"]) / 2 / video_width

    position_each_second = suction.groupby("frame_number")[["center_x", "center_y"]].mean().sort_index()
    move_x = position_each_second["center_x"].diff()
    move_y = position_each_second["center_y"].diff()

    distance_moved = (move_x**2 + move_y**2) ** 0.5
    seconds_between = position_each_second.index.to_series().diff()
    distance_moved = distance_moved[seconds_between == 1]  # skip when the suction leaves the screen
    
    if len(distance_moved) == 0:
        return 0
    return distance_moved.mean()



def blood_pattern_features(blood_by_second, watch_seconds=watch_seconds):
    rows = []
    
    for trial_id, trial_blood in blood_by_second.groupby("trial_id"):
        trial_blood = trial_blood.sort_values("frame_number")
        
        if len(trial_blood) > 1:
            blood_trend = np.polyfit(trial_blood["frame_number"], trial_blood["blood_amt"], 1)[0]
        else:
            blood_trend = 0
        last_15_seconds = trial_blood["frame_number"] > watch_seconds - 15
        rows.append({
            "trial_id": trial_id,
            "blood_max": trial_blood["blood_amt"].max(),
            "blood_trend": blood_trend,
            "blood_late": trial_blood.loc[last_15_seconds, "blood_amt"].mean(),
        })
    
    return pd.DataFrame(rows).fillna(0)


def tool_features_trial(trial_boxes, video_width):

    seconds_watched = trial_boxes["frame_number"].nunique()

    # seconds of suction and cottnoids 
    suction_seconds = trial_boxes.loc[trial_boxes["label"] == "suction", "frame_number"].nunique()
    suction_frac = suction_seconds / seconds_watched

    cottonoid_seconds = trial_boxes.loc[
        trial_boxes["label"] == "cottonoid",
        "frame_number",
    ].nunique()
    cottonoid_frac = cottonoid_seconds / seconds_watched

    tools_on_screen = trial_boxes.dropna(subset=["label"])
    tools_per_frame = len(tools_on_screen) / seconds_watched

    return {
        "suction_amt": suction_frac,
        "cottonoid_amt": cottonoid_frac,
        "tools_per_frame": tools_per_frame,
        "suction_speed": suction_speed(trial_boxes, video_width),
    }



def build_feature_table(watch_seconds=watch_seconds, start_second=0):

    tool_boxes = load_tool_boxes()
    in_window = (tool_boxes["frame_number"] > start_second) & (tool_boxes["frame_number"] <= watch_seconds)
    early_boxes = tool_boxes[in_window]
    outcomes = pd.read_csv("data/socal_outcomes.csv")

    video_widths = outcomes.set_index("trial_id")["trial_video_width"].fillna(default_vid_width)

    rows = []
    for trial_id, trial_boxes in early_boxes.groupby("trial_id"):
        trial_row = tool_features_trial(trial_boxes, video_widths.get(trial_id, default_vid_width))
        trial_row["trial_id"] = trial_id
        rows.append(trial_row)
    
    feature_table = pd.DataFrame(rows)

    blood_file = "data/blood_coverage.csv"
    if os.path.exists(blood_file):
        blood = pd.read_csv(blood_file)
        in_window = (blood["frame_number"] > start_second) & (blood["frame_number"] <= watch_seconds)
        blood_by_second = blood[in_window]
        
        blood_mean = blood_by_second.groupby("trial_id")["blood_amt"].mean().reset_index(name="blood_mean")
        feature_table = feature_table.merge(blood_mean, on="trial_id")
        feature_table = feature_table.merge(blood_pattern_features(blood_by_second, watch_seconds), on="trial_id")

    
    video_lengths = tool_boxes.groupby("trial_id")["frame_number"].max().rename("video_seconds")
    feature_table = feature_table.merge(video_lengths, on="trial_id")

    return feature_table.merge(outcomes[["trial_id", "blood_loss", "success"]], on="trial_id")


if __name__ == "__main__":
    
    feature_table = build_feature_table()
    print(feature_table.head(10))
    print("\ntrials in the table:", len(feature_table))
    print("\nFeature correleation to blood loss (-1 to 1):")
    print(feature_table.drop(columns=["trial_id"]).corr()["blood_loss"].round(2))
