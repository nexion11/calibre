"""Step 1: get to know the SOCAL data before building anything.

Run it with:  uv run python src/explore.py
"""
import pandas as pd

# --- 1. Outcomes: one row per trial (one surgeon's attempt at stopping the bleed) ---
outcomes = pd.read_csv("data/socal_outcomes.csv")
print("OUTCOMES TABLE")
print(outcomes.head(), "\n")
print("Number of trials:", len(outcomes))
print("Blood loss (mL) summary:")
print(outcomes["blood_loss"].describe(), "\n")
print("Success (1 = bleeding controlled within 5 min):")
print(outcomes["success"].value_counts(), "\n")

# --- 2. Boxes: one row per tool per video frame ---
# socal.csv has no header row, so we give the columns names ourselves.
boxes = pd.read_csv(
    "data/socal.csv",
    header=None,
    names=["frame", "x1", "y1", "x2", "y2", "label"],
)
print("BOXES TABLE")
print(boxes.head(), "\n")
print("Boxes per tool label:")
print(boxes["label"].value_counts(dropna=False), "\n")

# --- 3. Link each frame to its trial and its position in time ---
mapping = pd.read_csv("data/frame_to_trial_mapping.csv")
boxes = boxes.merge(mapping, on="frame", how="left")
print("Trials that have video:", boxes["trial_id"].nunique())

# How long is each video? (frames are 1 per second)
frames_per_trial = mapping.groupby("trial_id")["frame_number"].max()
print("Video length in frames (seconds):")
print(frames_per_trial.describe())
