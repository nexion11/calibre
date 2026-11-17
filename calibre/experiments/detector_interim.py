# Interim scoring of the PC detector search while it is still running.
# Reads whatever finished runs are in results/detector_search (copied from the PC), makes the current app
# detector's test predictions here if the PC did not, then scores every run the same way:
# our mAP50 with a 95% surgeon-resampled range, and paired gaps (each run minus the current detector,
# s minus n, m minus s) on the same resampled surgeons.
#   uv run python -m calibre.experiments.detector_interim
import json
import os
import pandas as pd
from calibre.experiments.detection_scoring import Scorer, human_truth_boxes, paired_gap, score_with_range
from calibre.experiments.detector_search import load_split, results_root, yolo_predictions

if __name__ == "__main__":
    boxes, frames = load_split()
    print({split: len(names) for split, names in frames.items()}, "frames (train / validation / test)", flush=True)
    os.makedirs(f"{results_root}/test_predictions", exist_ok=True)
    baseline_file = f"{results_root}/test_predictions/baseline.csv.gz"
    if not os.path.exists(baseline_file):
        weights = os.path.abspath("models/fold_detectors/fold0/weights/best.pt")
        yolo_predictions(weights, frames["test"], 640).to_csv(baseline_file, index=False)

    names = ["baseline"] + sorted(f.removesuffix(".csv.gz") for f in os.listdir(f"{results_root}/test_predictions")
                                  if f.endswith(".csv.gz") and f != "baseline.csv.gz")
    truth = human_truth_boxes(frames["test"])
    scorers = {n: Scorer(pd.read_csv(f"{results_root}/test_predictions/{n}.csv.gz"), truth) for n in names}

    rows = []
    for n in names:
        pc = json.load(open(f"{results_root}/runs/{n}.json")) if os.path.exists(f"{results_root}/runs/{n}.json") else {}
        rows.append({"name": n, **score_with_range(scorers[n]),
                     "ultralytics_val_map50": pc.get("val_map50"), "ultralytics_test_map50": pc.get("test_map50")})
    table = pd.DataFrame(rows)
    table.to_csv(f"{results_root}/interim_runs.csv", index=False)
    print(table.round(3).to_string(index=False))

    pairs = [(n, "baseline") for n in names if n != "baseline"]
    pairs += [(a, b) for a, b in [("s_640", "n_640"), ("m_640", "s_640"), ("m_640", "n_640"),
                                  ("s_960", "s_640"), ("s_960_rare", "s_960"), ("m_640", "s_960_rare"),
                                  ("s_960_rare", "retinanet"), ("m_640", "retinanet")]
              if a in scorers and b in scorers]
    gaps = []
    for new, old in pairs:
        gap = paired_gap(scorers[new], scorers[old])
        for metric, r in gap.iterrows():
            gaps.append({"new": new, "old": old, "metric": metric, **r.to_dict()})
    gaps = pd.DataFrame(gaps)
    gaps.to_csv(f"{results_root}/interim_gaps.csv", index=False)
    print(gaps.round(3).to_string(index=False))
