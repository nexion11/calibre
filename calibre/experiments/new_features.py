# New features from the first minute that describe HOW the surgeon handles the bleed,
# not just how much blood is on screen. Everything here comes from the per-second blood
# measurement (automatic) and the tool boxes (human or detector, whichever file is passed).
import numpy as np
import pandas as pd
from calibre.components.vid import human_boxes_file, load_tool_boxes

red_out_level = 0.6  # more than 60% of the view covered in blood = surgeon is basically blind

new_feature_names = [
    "red_out_seconds",        # seconds the view was mostly blood
    "longest_red_out",        # longest run of red-out seconds in a row
    "clearing_with_suction",  # how fast blood coverage drops per second while suction is in view
    "suction_benefit",        # clearing with suction minus clearing without it
    "first_cottonoid_second", # how quickly the first cottonoid patty went in (61 = never)
    "first_grasper_second",
    "two_handed_frac",        # share of seconds with suction plus another tool (bimanual work)
    "tool_switches",          # changes in which tools are on screen, per second
    "blood_volatility",       # how jumpy the blood coverage is second to second
    "blood_recovery",         # worst moment minus the last 15 s (did they get the view back?)
]


def longest_run(flags):
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def trial_new_features(blood, boxes, watch_seconds):
    seconds = pd.RangeIndex(1, watch_seconds + 1)
    blood_by_second = blood.set_index("frame_number")["blood_amt"].reindex(seconds).interpolate().bfill().ffill()
    labels_by_second = boxes.dropna(subset=["label"]).groupby("frame_number")["label"].apply(set)
    labels_by_second = labels_by_second.reindex(seconds).apply(lambda s: s if isinstance(s, set) else set())

    has_suction = labels_by_second.apply(lambda s: "suction" in s).values
    change_next = -np.diff(blood_by_second.values)  # positive = blood clearing
    with_suction, without_suction = change_next[has_suction[:-1]], change_next[~has_suction[:-1]]
    clearing_with = with_suction.mean() if len(with_suction) else 0.0
    clearing_without = without_suction.mean() if len(without_suction) else 0.0

    def first_second(tool):
        hits = [second for second, tools in labels_by_second.items() if tool in tools]
        return hits[0] if hits else watch_seconds + 1

    red_out = (blood_by_second > red_out_level).values
    tool_sets = labels_by_second.tolist()
    late = blood_by_second[blood_by_second.index > watch_seconds - 15]
    return {
        "red_out_seconds": red_out.sum(),
        "longest_red_out": longest_run(red_out),
        "clearing_with_suction": clearing_with,
        "suction_benefit": clearing_with - clearing_without,
        "first_cottonoid_second": first_second("cottonoid"),
        "first_grasper_second": first_second("grasper"),
        "two_handed_frac": np.mean([("suction" in s) and len(s) >= 2 for s in tool_sets]),
        "tool_switches": np.mean([a != b for a, b in zip(tool_sets[:-1], tool_sets[1:])]),
        "blood_volatility": np.std(np.diff(blood_by_second.values)),
        "blood_recovery": blood_by_second.max() - late.mean(),
    }


def add_new_features(table, boxes_file=human_boxes_file, watch_seconds=60):
    blood = pd.read_csv("data/blood_coverage.csv")
    blood = blood[blood["frame_number"] <= watch_seconds]
    boxes = load_tool_boxes(boxes_file)
    boxes = boxes[boxes["frame_number"] <= watch_seconds]
    blood_groups, box_groups = dict(list(blood.groupby("trial_id"))), dict(list(boxes.groupby("trial_id")))
    rows = []
    for trial_id in table["trial_id"]:
        empty = pd.DataFrame(columns=["frame_number", "label"])
        row = trial_new_features(blood_groups[trial_id], box_groups.get(trial_id, empty), watch_seconds)
        rows.append({"trial_id": trial_id, **row})
    return table.merge(pd.DataFrame(rows), on="trial_id")
