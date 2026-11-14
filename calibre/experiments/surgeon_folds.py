import random
import pandas as pd

n_folds = 5
folds_file = "data/surgeon_folds.csv"


def make_surgeon_folds():
    # every surgeon goes in exactly one of 5 folds, so a model tested on fold k never saw those surgeons
    tool_boxes = pd.read_csv("data/socal.csv", header=None, names=["frame", "x1", "y1", "x2", "y2", "label"])
    surgeons = sorted({int(frame[1:4]) for frame in tool_boxes["frame"]})  # "S201T1_frame..." -> 201
    random.Random(0).shuffle(surgeons)
    folds = pd.DataFrame({"participant_id": surgeons, "fold": [i % n_folds for i in range(len(surgeons))]})
    folds.to_csv(folds_file, index=False)
    return folds


def load_surgeon_folds():
    return pd.read_csv(folds_file)


if __name__ == "__main__":
    folds = make_surgeon_folds()
    print(folds["fold"].value_counts().sort_index().to_dict())
