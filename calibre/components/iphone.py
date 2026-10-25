import json
import subprocess
import tempfile

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupKFold
from calibre.components.vid import watch_seconds # type: ignore
from calibre.models.risk_model_training import (model_features, severe_bleed_amt_ml, severe_catch_rate, # type: ignore
                                                 load_training_table, ridge_model,
                                                 unseen_surgeon_tests)

samplecases = {"S814T1": "Severe bleed", "S103T2": "Controlled bleed", "S807T1": "Missed case"}
samplevidsecs = 75
appfolder = "phone/CALIBRE/Resources"
lowrisk = 150

training_table = load_training_table()
sample_surgeons = {trial_id[:4] for trial_id in samplecases}
training_table = training_table[~training_table["trial_id"].str[:4].isin(sample_surgeons)]

blood_loss, predictions = unseen_surgeon_tests(training_table, "ridge", model_features)
alert_line_ml = np.quantile(predictions[blood_loss > severe_bleed_amt_ml], 1 - severe_catch_rate)

held_out_estimates, held_out_actuals = [], []
inputs, surgeon_ids = training_table[model_features].values, training_table["participant_id"].values

for reshuffle in range(10):
    folds = GroupKFold(n_splits=5, shuffle=True, random_state=reshuffle)
    
    for train_rows, test_rows in folds.split(inputs, blood_loss, surgeon_ids):
        fold_model = ridge_model().fit(inputs[train_rows], np.log(blood_loss[train_rows]))
        held_out_estimates += list(np.exp(fold_model.predict(inputs[test_rows])))
        held_out_actuals += list(blood_loss[test_rows])
held_out_estimates, held_out_actuals = np.array(held_out_estimates), np.array(held_out_actuals)


log_ratio = np.log(held_out_actuals / held_out_estimates)
range_low, range_high = np.exp(np.quantile(log_ratio, [0.25, 0.75]))  # middle 50% of outcomes

risk_tiers = []

for tier_name, lower, upper in [("LOW", 0, lowrisk), ("MODERATE", lowrisk, alert_line_ml), ("HIGH", alert_line_ml, np.inf)]:
    in_tier = (held_out_estimates >= lower) & (held_out_estimates < upper)
    risk_tiers.append({"name": tier_name, "below": None if np.isinf(upper) else float(upper),
                       "severeRate": float((held_out_actuals[in_tier] > severe_bleed_amt_ml).mean())})
    
for tier in risk_tiers:
    print(f"  {tier['name']}: {tier['severeRate']:.0%} of trials severe")

final_model = ridge_model().fit(training_table[model_features].values, np.log(training_table["blood_loss"].values))
scaler, ridge = final_model.named_steps["standardscaler"], final_model.named_steps["ridgecv"]

with open(f"{appfolder}/risk_model.json", "w") as f:
    json.dump({"features": model_features, "means": scaler.mean_.tolist(), "scales": scaler.scale_.tolist(),
               "weights": ridge.coef_.tolist(), "intercept": float(ridge.intercept_),
               "alertThreshold": float(alert_line_ml), "windowSeconds": watch_seconds,
               "rangeLow": float(range_low), "rangeHigh": float(range_high), "tiers": risk_tiers}, f, indent=2)



frame_to_trial = pd.read_csv("data/frame_to_trial_mapping.csv")

for trial_id in samplecases:
    trial_frames = frame_to_trial[(frame_to_trial["trial_id"] == trial_id)
                                  & (frame_to_trial["frame_number"] <= samplevidsecs)]
    
    with tempfile.TemporaryDirectory() as temp_folder:
        for frame in trial_frames.itertuples():
            image = Image.open(f"data/frames/JPEGImages/{frame.frame}").convert("RGB").resize((640, 360))
            image.save(f"{temp_folder}/{frame.frame_number:03d}.jpg", quality=90)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "1", "-i", f"{temp_folder}/%03d.jpg",
                        "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-crf", "23",
                        f"{appfolder}/sample_{trial_id}.mp4"], check=True)
    
    print(f"{appfolder}/sample_{trial_id}.mp4: {len(trial_frames)}")
