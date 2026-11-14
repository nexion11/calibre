# One mAP50 calculation for every detector (YOLO or RetinaNet), so they are compared the same way,
# plus surgeon-level resampling for 95% ranges and paired gaps between two detectors.
# Same recipe as Ultralytics: a box counts if it overlaps a human box of the same tool by IoU >= 0.5,
# precision-recall curve made monotone, area under it from 101 recall points.
import numpy as np
import pandas as pd
from calibre.models.tool_detector_training import tool_types


def iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    overlap = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return overlap / (area + areas - overlap + 1e-9)


def match(predictions, truths):
    """Mark each predicted box right or wrong. Both tables: frame, label, x1, y1, x2, y2 (+ confidence)."""
    predictions = predictions.sort_values("confidence", ascending=False).reset_index(drop=True)
    correct = np.zeros(len(predictions), bool)
    truth_groups = {key: group[["x1", "y1", "x2", "y2"]].values for key, group in truths.groupby(["frame", "label"])}
    used = {key: np.zeros(len(boxes), bool) for key, boxes in truth_groups.items()}
    for row, (frame, label, x1, y1, x2, y2) in enumerate(predictions[["frame", "label", "x1", "y1", "x2", "y2"]].values):
        boxes = truth_groups.get((frame, label))
        if boxes is None:
            continue
        overlaps = iou(np.array([x1, y1, x2, y2]), boxes)
        overlaps[used[(frame, label)]] = 0
        best = overlaps.argmax()
        if overlaps[best] >= 0.5:
            correct[row] = True
            used[(frame, label)][best] = True
    predictions["correct"] = correct
    return predictions


def average_precision(correct, weights, n_truth):
    if n_truth == 0:
        return np.nan
    true_pos, false_pos = np.cumsum(correct * weights), np.cumsum(~correct * weights)
    recall = true_pos / n_truth
    precision = true_pos / np.maximum(true_pos + false_pos, 1e-9)
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[1.0], precision, [0.0]])
    precision = np.flip(np.maximum.accumulate(np.flip(precision)))
    points = np.linspace(0, 1, 101)
    return np.trapezoid(np.interp(points, recall, precision), points)


def surgeon_of(frames):
    return frames.str[1:4].astype(int)


class Scorer:
    """Precompute matches once; then score the full set or any resampled set of surgeons quickly."""

    def __init__(self, predictions, truths):
        self.surgeons = np.sort(surgeon_of(truths["frame"]).unique())
        matched = match(predictions, truths)
        matched["surgeon"] = surgeon_of(matched["frame"])
        truths = truths.assign(surgeon=surgeon_of(truths["frame"]))
        self.by_tool = {}
        for tool in tool_types:
            tool_preds = matched[matched["label"] == tool]
            surgeon_index = np.searchsorted(self.surgeons, tool_preds["surgeon"].values)
            truth_per_surgeon = truths[truths["label"] == tool].groupby("surgeon").size() \
                .reindex(self.surgeons, fill_value=0).values
            self.by_tool[tool] = (tool_preds["correct"].values, surgeon_index, truth_per_surgeon)

    def score(self, surgeon_counts=None):
        if surgeon_counts is None:
            surgeon_counts = np.ones(len(self.surgeons))
        row = {}
        for tool, (correct, surgeon_index, truth_per_surgeon) in self.by_tool.items():
            row[f"{tool}_ap50"] = average_precision(correct, surgeon_counts[surgeon_index],
                                                    (truth_per_surgeon * surgeon_counts).sum())
        row["map50"] = np.nanmean([row[f"{tool}_ap50"] for tool in tool_types])
        return row


def resampled_counts(n_surgeons, rounds=2000, seed=0):
    rng = np.random.default_rng(seed)
    return [np.bincount(rng.integers(n_surgeons, size=n_surgeons), minlength=n_surgeons) for _ in range(rounds)]


def score_with_range(scorer, rounds=2000):
    full = scorer.score()
    samples = pd.DataFrame([scorer.score(c) for c in resampled_counts(len(scorer.surgeons), rounds)])
    return {**full, "map50_low": samples["map50"].quantile(0.025), "map50_high": samples["map50"].quantile(0.975)}


def paired_gap(new, old, rounds=2000):
    """new minus old on the same resampled surgeons. Both scorers must cover the same surgeons."""
    assert np.array_equal(new.surgeons, old.surgeons)
    gaps = pd.DataFrame([pd.Series(new.score(c)) - pd.Series(old.score(c))
                         for c in resampled_counts(len(new.surgeons), rounds)])
    full = pd.Series(new.score()) - pd.Series(old.score())
    return pd.DataFrame({"gap": full, "low": gaps.quantile(0.025), "high": gaps.quantile(0.975)})


def human_truth_boxes(frames):
    truths = pd.read_csv("data/socal.csv", header=None, names=["frame", "x1", "y1", "x2", "y2", "label"])
    return truths[truths["frame"].isin(set(frames)) & truths["label"].isin(tool_types)]
