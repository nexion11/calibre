import os
import random
import shutil
import pandas as pd
from PIL import Image
from ultralytics import YOLO # type: ignore


frames_folder = os.path.abspath("data/frames/JPEGImages")
yolo_data_folder = os.path.abspath("data/yolo")
tool_types = ["suction", "grasper", "cottonoid", "muscle", "string"]
use_every_nth_frame = 4 


def build_yolo_data():
    tool_boxes = pd.read_csv("data/socal.csv", header=None, names=["frame", "x1", "y1", "x2", "y2", "label"])
    surgeon_ids = sorted({frame_name[:4] for frame_name in tool_boxes["frame"]})  # "S201T1_frame..." -> "S201"
    random.Random(0).shuffle(surgeon_ids)
   
    test_surgeons = set(surgeon_ids[: len(surgeon_ids) // 5])
    
    shutil.rmtree(yolo_data_folder, ignore_errors=True)
    for split in ["train", "val"]:
        os.makedirs(f"{yolo_data_folder}/images/{split}")
        os.makedirs(f"{yolo_data_folder}/labels/{split}")

    boxes_by_frame = tool_boxes.groupby("frame")
    
    for frame_name in sorted(tool_boxes["frame"].unique())[::use_every_nth_frame]:
        split = "val" if frame_name[:4] in test_surgeons else "train"
        width, height = Image.open(f"{frames_folder}/{frame_name}").size
        label_lines = []
        for box in boxes_by_frame.get_group(frame_name).itertuples():
            if box.label not in tool_types:
                continue
            # YOLO format: class, box center x, center y, width, height (all as a share of the image)
            center_x, center_y = (box.x1 + box.x2) / 2 / width, (box.y1 + box.y2) / 2 / height
            box_width, box_height = (box.x2 - box.x1) / width, (box.y2 - box.y1) / height
            label_lines.append(f"{tool_types.index(box.label)} {center_x:.5f} {center_y:.5f} "
                               f"{box_width:.5f} {box_height:.5f}")
       
        os.symlink(f"{frames_folder}/{frame_name}", f"{yolo_data_folder}/images/{split}/{frame_name}")
       
        with open(f"{yolo_data_folder}/labels/{split}/{frame_name.replace('.jpeg', '.txt')}", "w") as f:
            f.write("\n".join(label_lines))

    with open(f"{yolo_data_folder}/data.yaml", "w") as f:
        f.write(f"path: {yolo_data_folder}\ntrain: images/train\nval: images/val\nnames: {tool_types}\n")
    
    print("train images:", len(os.listdir(f"{yolo_data_folder}/images/train")),
          "test images:", len(os.listdir(f"{yolo_data_folder}/images/val")))


if __name__ == "__main__":
    build_yolo_data()
    detector = YOLO("yolo11n.pt")  # "n" = nano, the smallest YOLO11, fast enough for an iPhone
    detector.train(data=f"{yolo_data_folder}/data.yaml", epochs=20, imgsz=640, batch=32, device="mps",
                   workers=8, project=os.path.abspath("models"), name="detector", exist_ok=True, plots=False)
    best_detector = YOLO("models/detector/weights/best.pt")
    
    results = best_detector.val(data=f"{yolo_data_folder}/data.yaml", device="mps")
    
    print(round(results.box.map50, 3))
   
    best_detector.export(format="coreml", nms=True, imgsz=640)
