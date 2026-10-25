#using socal dataset

import pandas as pd

data_outcomes = pd.read_csv("data/socal_outcomes.csv")

print("outcomes table:")
print(data_outcomes.head(), "\n")
print("# of trials:", len(data_outcomes))
print("blood loss (in millimeters):")
print(data_outcomes["blood_loss"].describe(), "\n")
print("1 = bleeding was controlled within 5 minutes:")
print(data_outcomes["success"].value_counts(), "\n")

tool_boxes = pd.read_csv(
    "data/socal.csv",
    header=None,
    names=["frame", "x1", "y1", "x2", "y2", "label"],
)
print("tool boxes table:")
print(tool_boxes.head(), "\n")
print("boxes per tool:")
print(tool_boxes["label"].value_counts(dropna=False), "\n")

# link the frames to their respective trials and seconds 

frame_to_trial = pd.read_csv("data/frame_to_trial_mapping.csv")

tool_boxes = tool_boxes.merge(frame_to_trial, on="frame", how="left")
print("trials with videos:", tool_boxes["trial_id"].nunique())

# video length
seconds_per_trial = frame_to_trial.groupby("trial_id")["frame_number"].max()
print("video len:")
print(seconds_per_trial.describe())
