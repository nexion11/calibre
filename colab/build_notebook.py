"""Builds bleedwatch_colab.ipynb from the cells below.
Run it with:  uv run python colab/build_notebook.py
"""
import json

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip()})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip()})


md("""
# BleedWatch: blood coverage + tool detector

Run each cell in order with **Shift+Enter**. Before you start:
**Runtime → Change runtime type → T4 GPU → Save.**

What this notebook does:
1. Downloads the SOCAL frames (3 GB) into Colab, not onto your laptop.
2. Measures how much of each frame is covered in blood.
3. Trains a YOLO model to find surgical tools (about 40–60 minutes).
4. Exports the model for the iPhone app.
""")

md("## 1. Setup\nConnects your Google Drive so results survive if Colab disconnects. Click **Allow** when asked.")
code("""
from google.colab import drive
drive.mount("/content/drive")

import os
OUT = "/content/drive/MyDrive/bleedwatch"   # results are saved here
DATA = "/content/data"                      # big files live here (deleted when Colab closes)
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

!nvidia-smi --query-gpu=name --format=csv,noheader
!pip install -q ultralytics
""")

md("## 2. Download the data\nAbout 3–5 minutes.")
code("""
%cd /content/data
!wget -q -O frames.zip https://ndownloader.figshare.com/files/29078499
!wget -q -O socal.csv https://ndownloader.figshare.com/files/29078493
!wget -q -O frame_to_trial_mapping.csv https://ndownloader.figshare.com/files/29078496
!unzip -q frames.zip && rm frames.zip
!ls JPEGImages | wc -l
""")

md("""
## 3. Blood coverage

The endoscope view is pink-red tissue even with no bleeding, so "count red pixels" isn't enough.
Blood is **deep, saturated, darker** red. We keep pixels that are:
- inside the endoscope circle (not the black border),
- red in hue,
- very saturated (`S_MIN`), and not too bright (`V_MAX`).

First, look at a few frames. Green = what the rule calls blood.
""")
code("""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

FRAMES = f"{DATA}/JPEGImages"
S_MIN, V_MAX = 0.75, 0.75


def blood_fraction(path, show=False):
    img = Image.open(path).resize((480, 270))
    hsv = np.asarray(img.convert("HSV")).astype(np.float32) / 255
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    in_view = v > 0.12
    blood = in_view & ((h < 0.04) | (h > 0.94)) & (s > S_MIN) & (v < V_MAX)
    if show:
        overlay = np.asarray(img).copy()
        overlay[blood] = [0, 255, 0]
        return blood.sum() / max(in_view.sum(), 1), overlay
    return blood.sum() / max(in_view.sum(), 1)


examples = ["S201T1_frame_00000030.jpeg", "S107T1_frame_00000040.jpeg",
            "S103T1_frame_00000050.jpeg", "S104T2_frame_00000010.jpeg"]
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, name in zip(axes, examples):
    frac, overlay = blood_fraction(f"{FRAMES}/{name}", show=True)
    ax.imshow(overlay)
    ax.set_title(f"{name[:6]}  blood = {frac:.0%}")
    ax.axis("off")
plt.show()
""")

md("Now measure every frame (about 5–10 minutes) and save the results to Drive.")
code("""
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

names = sorted(os.listdir(FRAMES))
with ProcessPoolExecutor() as pool:
    fractions = list(pool.map(blood_fraction, [f"{FRAMES}/{n}" for n in names], chunksize=64))

mapping = pd.read_csv(f"{DATA}/frame_to_trial_mapping.csv")
blood = pd.DataFrame({"frame": names, "blood_frac": fractions}).merge(mapping, on="frame")
blood.to_csv(f"{OUT}/blood_coverage.csv", index=False)
print(blood.describe())
""")

md("""
## 4. Tool detector (YOLO)

YOLO needs one small text file per image listing its boxes. We:
- keep the 5 common tools (the rare ones have too few examples to learn),
- use **every 3rd frame**: frames 1 second apart look almost the same, so this trains 3× faster with little loss,
- split by **surgeon**, so validation uses surgeons the model never trained on (same rule as before).
""")
code("""
import random
import pandas as pd

CLASSES = ["suction", "grasper", "cottonoid", "muscle", "string"]
boxes = pd.read_csv(f"{DATA}/socal.csv", header=None,
                    names=["frame", "x1", "y1", "x2", "y2", "label"])

surgeons = sorted({f[:4] for f in boxes["frame"]})    # "S201T1_frame..." -> "S201"
random.Random(0).shuffle(surgeons)
val_surgeons = set(surgeons[: len(surgeons) // 5])     # 20% of surgeons for validation

YOLO_DIR = f"{DATA}/yolo"
for split in ["train", "val"]:
    os.makedirs(f"{YOLO_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{YOLO_DIR}/labels/{split}", exist_ok=True)

frames = sorted(boxes["frame"].unique())[::3]
groups = boxes.groupby("frame")
for name in frames:
    split = "val" if name[:4] in val_surgeons else "train"
    width, height = Image.open(f"{FRAMES}/{name}").size
    lines = []
    for _, b in groups.get_group(name).iterrows():
        if b["label"] not in CLASSES:
            continue
        # YOLO format: class, box center x, center y, width, height (all 0-1)
        cx = (b.x1 + b.x2) / 2 / width
        cy = (b.y1 + b.y2) / 2 / height
        bw = (b.x2 - b.x1) / width
        bh = (b.y2 - b.y1) / height
        lines.append(f"{CLASSES.index(b['label'])} {cx:.5f} {cy:.5f} {bw:.5f} {bh:.5f}")
    os.symlink(f"{FRAMES}/{name}", f"{YOLO_DIR}/images/{split}/{name}")
    with open(f"{YOLO_DIR}/labels/{split}/{name.replace('.jpeg', '.txt')}", "w") as f:
        f.write("\\n".join(lines))

with open(f"{YOLO_DIR}/data.yaml", "w") as f:
    f.write(f"path: {YOLO_DIR}\\ntrain: images/train\\nval: images/val\\nnames: {CLASSES}\\n")

print("train images:", len(os.listdir(f"{YOLO_DIR}/images/train")))
print("val images:  ", len(os.listdir(f"{YOLO_DIR}/images/val")))
""")

md("""
Train. This takes about **40–60 minutes**. Leave the tab open.
Each epoch is one pass through all training images; watch **mAP50** rise
(the share of tools it finds with a well-placed box; higher is better).

If Colab disconnects, re-run sections 1, 2 and 4's first cell, then run the **Resume** cell below.
""")
code("""
from ultralytics import YOLO

model = YOLO("yolo11n.pt")   # "n" = nano, the smallest model, fast enough for an iPhone
model.train(data=f"{YOLO_DIR}/data.yaml", epochs=25, imgsz=640, batch=32,
            project=OUT, name="tools", exist_ok=True)
""")
code("""
# Resume (only if training was interrupted)
# model = YOLO(f"{OUT}/tools/weights/last.pt")
# model.train(resume=True)
""")

md("## 5. Check it and export for iPhone")
code("""
best = YOLO(f"{OUT}/tools/weights/best.pt")
metrics = best.val(data=f"{YOLO_DIR}/data.yaml")
print("mAP50:", round(metrics.box.map50, 3))

# Show predictions on a few validation frames
val_images = sorted(os.listdir(f"{YOLO_DIR}/images/val"))[::200][:4]
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, name in zip(axes, val_images):
    result = best(f"{YOLO_DIR}/images/val/{name}", verbose=False)[0]
    ax.imshow(result.plot()[..., ::-1])
    ax.axis("off")
plt.show()
""")
code("""
# Core ML = Apple's format for running models on iPhone.
best.export(format="coreml", nms=True, imgsz=640)
!cd {OUT}/tools/weights && zip -qr bleedwatch_tools.mlpackage.zip best.mlpackage
print("Saved to Google Drive: bleedwatch/blood_coverage.csv and bleedwatch/tools/weights/")
""")

notebook = {"cells": cells, "metadata": {"accelerator": "GPU",
            "kernelspec": {"name": "python3", "display_name": "Python 3"}},
            "nbformat": 4, "nbformat_minor": 5}
with open("colab/bleedwatch_colab.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
print("Wrote colab/bleedwatch_colab.ipynb with", len(cells), "cells")
