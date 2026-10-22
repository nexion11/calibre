"""Export what the iPhone app needs: the risk model and two sample videos.

Run it with:  uv run python src/export_app_assets.py

The app's model is trained WITHOUT the demo surgeons, so the demo shows
honest predictions on cases the model has never seen.
"""
import json
import subprocess
import tempfile

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupKFold

from train import FEATURES, SEVERE, TARGET_CATCH, cross_validate, load_table, make_ridge

# Surgeons the tool detector also never saw (its validation split), so both models are honest here.
DEMO_TRIALS = {"S814T1": "Severe bleed", "S103T2": "Controlled bleed", "S807T1": "Missed case"}
DEMO_SECONDS = 75
OUT = "ios/BleedWatch/Resources"

table = load_table()
demo_surgeons = {t[:4] for t in DEMO_TRIALS}
train_table = table[~table["trial_id"].str[:4].isin(demo_surgeons)]

y, predictions = cross_validate(train_table, "ridge", FEATURES)
threshold = np.quantile(predictions[y > SEVERE], 1 - TARGET_CATCH)

# Honest uncertainty, measured on surgeons each model never saw (10 reshuffles):
# 1. an 80% range: actual blood loss fell between low x and high x the estimate 80% of the time
# 2. risk tiers: how often trials with that estimate turned out to be severe (>500 mL)
oof_pred, oof_true = [], []
X, groups = train_table[FEATURES].values, train_table["participant_id"].values
for repeat in range(10):
    folds = GroupKFold(n_splits=5, shuffle=True, random_state=repeat)
    for tr, te in folds.split(X, y, groups):
        fold_model = make_ridge().fit(X[tr], np.log(y[tr]))
        oof_pred += list(np.exp(fold_model.predict(X[te])))
        oof_true += list(y[te])
oof_pred, oof_true = np.array(oof_pred), np.array(oof_true)
ratio = np.log(oof_true / oof_pred)
range_low, range_high = np.exp(np.quantile(ratio, [0.1, 0.9]))
LOW_BELOW = 150
tiers = []
for name, lo, hi in [("LOW", 0, LOW_BELOW), ("MODERATE", LOW_BELOW, threshold), ("HIGH", threshold, np.inf)]:
    in_tier = (oof_pred >= lo) & (oof_pred < hi)
    tiers.append({"name": name, "below": None if np.isinf(hi) else float(hi),
                  "severeRate": float((oof_true[in_tier] > SEVERE).mean())})
print(f"80% range: {range_low:.2f}x to {range_high:.2f}x the estimate")
for tier in tiers:
    print(f"  {tier['name']}: {tier['severeRate']:.0%} of trials were severe")
model = make_ridge().fit(train_table[FEATURES].values, np.log(train_table["blood_loss"].values))
scaler, ridge = model.named_steps["standardscaler"], model.named_steps["ridgecv"]
with open(f"{OUT}/risk_model.json", "w") as f:
    json.dump({"features": FEATURES, "means": scaler.mean_.tolist(), "scales": scaler.scale_.tolist(),
               "weights": ridge.coef_.tolist(), "intercept": float(ridge.intercept_),
               "alertThreshold": float(threshold), "windowSeconds": 60,
               "rangeLow": float(range_low), "rangeHigh": float(range_high), "tiers": tiers}, f, indent=2)
print(f"risk model: trained on {len(train_table)} trials, alert above {threshold:.0f} mL")

# Sample videos: the first DEMO_SECONDS frames of each demo trial at 1 frame per
# second (the dataset's rate), encoded as an .mp4 the app plays like any other video.
mapping = pd.read_csv("data/frame_to_trial_mapping.csv")
for trial_id in DEMO_TRIALS:
    frames = mapping[(mapping["trial_id"] == trial_id) & (mapping["frame_number"] <= DEMO_SECONDS)]
    with tempfile.TemporaryDirectory() as tmp:
        for row in frames.itertuples():
            img = Image.open(f"data/frames/JPEGImages/{row.frame}").convert("RGB").resize((640, 360))
            img.save(f"{tmp}/{row.frame_number:03d}.jpg", quality=90)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "1", "-i", f"{tmp}/%03d.jpg",
                        "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-crf", "23",
                        f"{OUT}/sample_{trial_id}.mp4"], check=True)
    print(f"{OUT}/sample_{trial_id}.mp4: {len(frames)} s")
