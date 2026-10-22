"""Measure how much of each video frame is covered in blood.
Same rule as the Colab notebook, run locally.

Run it with:  uv run python src/blood_coverage.py
Needs the frames in data/frames/JPEGImages (3 GB download).
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

FRAMES = "data/frames/JPEGImages"
S_MIN, V_MAX = 0.75, 0.75


def blood_fraction(path):
    """Share of the endoscope view that looks like blood: red, very
    saturated and not too bright (pink tissue is less saturated)."""
    img = Image.open(path).resize((480, 270))
    hsv = np.asarray(img.convert("HSV")).astype(np.float32) / 255
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    in_view = v > 0.12  # inside the endoscope circle, not the black border
    blood = in_view & ((h < 0.04) | (h > 0.94)) & (s > S_MIN) & (v < V_MAX)
    return blood.sum() / max(in_view.sum(), 1)


if __name__ == "__main__":
    names = sorted(os.listdir(FRAMES))
    with ProcessPoolExecutor() as pool:
        fractions = list(pool.map(blood_fraction, [f"{FRAMES}/{n}" for n in names], chunksize=64))

    mapping = pd.read_csv("data/frame_to_trial_mapping.csv")
    blood = pd.DataFrame({"frame": names, "blood_frac": fractions}).merge(mapping, on="frame")
    blood.to_csv("data/blood_coverage.csv", index=False)
    print(blood["blood_frac"].describe())
