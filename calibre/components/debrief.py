import json
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from calibre.components.vid import load_tool_boxes


appfolder = "phone/CALIBRE/Resources"
severe_bleed_amt_ml = 500

# the habits we compare against. "higher" = more is better, "lower" = sooner is better
habits = {
    "cottonoid_first": "lower",
    "cottonoid_share_60": "higher",
    "muscle_first": "lower",
    "tools_per_frame": "higher",
}


def first_second(trial_boxes, tool):
    seconds = trial_boxes.loc[trial_boxes["label"] == tool, "frame_number"]
    return seconds.min() if len(seconds) else np.nan


def habits_per_trial():
    tool_boxes = load_tool_boxes().dropna(subset=["trial_id"])
    rows = []

    for trial_id, trial_boxes in tool_boxes.groupby("trial_id"):
        first_minute = trial_boxes[trial_boxes["frame_number"] <= 60]
        rows.append({
            "trial_id": trial_id,
            "cottonoid_first": first_second(trial_boxes, "cottonoid"),
            "cottonoid_share_60": first_minute.loc[first_minute["label"] == "cottonoid", "frame_number"].nunique()
                                  / max(first_minute["frame_number"].nunique(), 1),
            "muscle_first": first_second(trial_boxes, "muscle"),
            "tools_per_frame": trial_boxes["label"].notna().sum() / trial_boxes["frame_number"].nunique(),
        })

    outcomes = pd.read_csv("data/socal_outcomes.csv")
    table = pd.DataFrame(rows).merge(outcomes, on="trial_id")
    # best runs = stopped the bleed and kept it under 500 ml
    table["best"] = (table["success"] == 1) & (table["blood_loss"] < severe_bleed_amt_ml)
    return table


if __name__ == "__main__":
    table = habits_per_trial()
    best, other = table[table["best"]], table[~table["best"]]
    print(f"best runs: {len(best)} of {len(table)}")

    reference = {"best_runs": int(len(best)), "total_runs": int(len(table)), "habits": {}}
    for habit, better in habits.items():
        best_values, other_values = best[habit].dropna(), other[habit].dropna()
        p = mannwhitneyu(best_values, other_values).pvalue
        print(f"{habit:20s} best {best_values.median():.2f}  other {other_values.median():.2f}  p={p:.4f}")

        reference["habits"][habit] = {
            "better": better,
            "best_median": float(best_values.median()),
            "best_low": float(best_values.quantile(0.25)),  # middle half of best runs
            "best_high": float(best_values.quantile(0.75)),
            "other_median": float(other_values.median()),
            "p_value": float(p),
        }

    # pick one real good run to show side by side: a best run that hits all 4 habits
    # and is closest to the typical best run. skip the demo videos so its not comparing to itself
    tool_boxes = load_tool_boxes().dropna(subset=["trial_id"])
    candidates = best[~best["trial_id"].isin(["S814T1", "S103T2", "S807T1"])].dropna(subset=list(habits)).copy()
    for habit, better in habits.items():
        info = reference["habits"][habit]
        if better == "lower":
            candidates = candidates[candidates[habit] <= info["best_high"]]
        else:
            candidates = candidates[candidates[habit] >= info["best_low"]]
    spread = best[list(habits)].std()
    medians = best[list(habits)].median()
    candidates["distance"] = (((candidates[list(habits)] - medians) / spread) ** 2).sum(axis=1)
    good_run = candidates.sort_values("distance").iloc[0]

    good_boxes = tool_boxes[tool_boxes["trial_id"] == good_run["trial_id"]]
    reference["good_run"] = {
        "trial_id": good_run["trial_id"],
        "blood_loss": float(good_run["blood_loss"]),
        "length": int(good_boxes["frame_number"].max()),
        "first_used": {tool: int(first_second(good_boxes, tool))
                       for tool in ["suction", "grasper", "cottonoid", "muscle"]
                       if not np.isnan(first_second(good_boxes, tool))},
        "habits": {habit: float(good_run[habit]) for habit in habits},
    }
    print("good run:", good_run["trial_id"], good_run["blood_loss"], "mL")

    with open(f"{appfolder}/debrief_reference.json", "w") as f:
        json.dump(reference, f, indent=2)
