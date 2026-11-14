# How good is each fold's tool detector on the surgeons it never saw?
# Every hand-labeled frame of those surgeons is scored, so all 5 folds together cover every surgeon once.
#   uv run python -m calibre.experiments.detector_check
import os
import shutil
import pandas as pd
from PIL import Image
from ultralytics import YOLO  # type: ignore
from calibre.experiments.fold_detectors import device, link_or_copy
from calibre.experiments.surgeon_folds import load_surgeon_folds, n_folds
from calibre.models.tool_detector_training import frames_folder, tool_types


def build_unseen_data(fold, tool_boxes, folds):
    data_folder = os.path.abspath(f"data/yolo_folds/unseen{fold}")
    unseen_surgeons = set(folds.loc[folds["fold"] == fold, "participant_id"])
    shutil.rmtree(data_folder, ignore_errors=True)
    os.makedirs(f"{data_folder}/images/val")
    os.makedirs(f"{data_folder}/labels/val")
    unseen_boxes = tool_boxes[tool_boxes["participant_id"].isin(unseen_surgeons)]
    for frame_name, boxes in unseen_boxes.groupby("frame"):
        width, height = Image.open(f"{frames_folder}/{frame_name}").size
        label_lines = []
        for box in boxes.itertuples():
            if box.label not in tool_types:
                continue
            center_x, center_y = (box.x1 + box.x2) / 2 / width, (box.y1 + box.y2) / 2 / height
            box_width, box_height = (box.x2 - box.x1) / width, (box.y2 - box.y1) / height
            label_lines.append(f"{tool_types.index(box.label)} {center_x:.5f} {center_y:.5f} "
                               f"{box_width:.5f} {box_height:.5f}")
        link_or_copy(f"{frames_folder}/{frame_name}", f"{data_folder}/images/val/{frame_name}")
        with open(f"{data_folder}/labels/val/{frame_name.replace('.jpeg', '.txt')}", "w") as f:
            f.write("\n".join(label_lines))
    with open(f"{data_folder}/data.yaml", "w") as f:
        f.write(f"path: {data_folder}\ntrain: images/val\nval: images/val\nnames: {tool_types}\n")
    return data_folder, len(unseen_surgeons), unseen_boxes["frame"].nunique()


if __name__ == "__main__":
    folds = load_surgeon_folds()
    tool_boxes = pd.read_csv("data/socal.csv", header=None, names=["frame", "x1", "y1", "x2", "y2", "label"])
    tool_boxes["participant_id"] = tool_boxes["frame"].str[1:4].astype(int)
    rows = []
    for fold in range(n_folds):
        data_folder, surgeons, frames = build_unseen_data(fold, tool_boxes, folds)
        detector = YOLO(os.path.abspath(f"models/fold_detectors/fold{fold}/weights/best.pt"))
        result = detector.val(data=f"{data_folder}/data.yaml", imgsz=640, device=device, plots=False, verbose=False,
                              project=os.path.abspath("runs"), name=f"unseen{fold}", exist_ok=True)
        row = {"fold": fold, "unseen_surgeons": surgeons, "frames": frames, "map50": result.box.map50}
        for tool_index, tool_ap50 in zip(result.box.ap_class_index, result.box.ap50):
            row[f"{tool_types[tool_index]}_ap50"] = tool_ap50
        rows.append(row)
        print(row, flush=True)
    table = pd.DataFrame(rows)
    average = table.drop(columns="fold").mean().to_dict()
    average.update(fold="average", unseen_surgeons=table["unseen_surgeons"].sum(), frames=table["frames"].sum())
    pd.concat([table, pd.DataFrame([average])]).to_csv("results/auto_detector_unseen_surgeons.csv", index=False)
    print(pd.DataFrame([average]).round(3).to_string(index=False))
