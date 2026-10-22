"""Train the YOLO tool detector on this Mac (Apple GPU), then export it for iPhone.
Same steps as the Colab notebook.

Run it with:  uv run python src/train_detector.py
"""
import os
import random
import shutil

import pandas as pd
from PIL import Image
from ultralytics import YOLO

FRAMES = os.path.abspath("data/frames/JPEGImages")
YOLO_DIR = os.path.abspath("data/yolo")
CLASSES = ["suction", "grasper", "cottonoid", "muscle", "string"]
EVERY = 4  # back-to-back frames look alike, so every 4th frame is plenty


def build_dataset():
    boxes = pd.read_csv("data/socal.csv", header=None, names=["frame", "x1", "y1", "x2", "y2", "label"])
    surgeons = sorted({f[:4] for f in boxes["frame"]})
    random.Random(0).shuffle(surgeons)
    val_surgeons = set(surgeons[: len(surgeons) // 5])  # validate on unseen surgeons

    shutil.rmtree(YOLO_DIR, ignore_errors=True)
    for split in ["train", "val"]:
        os.makedirs(f"{YOLO_DIR}/images/{split}")
        os.makedirs(f"{YOLO_DIR}/labels/{split}")

    groups = boxes.groupby("frame")
    for name in sorted(boxes["frame"].unique())[::EVERY]:
        split = "val" if name[:4] in val_surgeons else "train"
        width, height = Image.open(f"{FRAMES}/{name}").size
        lines = []
        for b in groups.get_group(name).itertuples():
            if b.label not in CLASSES:
                continue
            cx, cy = (b.x1 + b.x2) / 2 / width, (b.y1 + b.y2) / 2 / height
            bw, bh = (b.x2 - b.x1) / width, (b.y2 - b.y1) / height
            lines.append(f"{CLASSES.index(b.label)} {cx:.5f} {cy:.5f} {bw:.5f} {bh:.5f}")
        os.symlink(f"{FRAMES}/{name}", f"{YOLO_DIR}/images/{split}/{name}")
        with open(f"{YOLO_DIR}/labels/{split}/{name.replace('.jpeg', '.txt')}", "w") as f:
            f.write("\n".join(lines))

    with open(f"{YOLO_DIR}/data.yaml", "w") as f:
        f.write(f"path: {YOLO_DIR}\ntrain: images/train\nval: images/val\nnames: {CLASSES}\n")
    print("train:", len(os.listdir(f"{YOLO_DIR}/images/train")), "val:", len(os.listdir(f"{YOLO_DIR}/images/val")))


if __name__ == "__main__":
    build_dataset()
    model = YOLO("yolo11n.pt")
    model.train(data=f"{YOLO_DIR}/data.yaml", epochs=20, imgsz=640, batch=32, device="mps",
                workers=8, project=os.path.abspath("models"), name="detector", exist_ok=True, plots=False)
    best = YOLO("models/detector/weights/best.pt")
    print("mAP50 on unseen surgeons:", round(best.val(data=f"{YOLO_DIR}/data.yaml", device="mps").box.map50, 3))
    best.export(format="coreml", nms=True, imgsz=640)
