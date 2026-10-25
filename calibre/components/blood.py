import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image

frames = "data/frames/JPEGImages"
min_saturation = 0.75 
max_brightness = 0.75
min_brightness_in_view = 0.12 
#nice balanced values maybe change later though 

#amount of blood
def blood_amount(image_path):
    image = Image.open(image_path).resize((480, 270))
    hsv = np.asarray(image.convert("HSV")).astype(np.float32) / 255

    hue, saturation, brightness = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    inside_view = brightness > min_brightness_in_view
    is_red = (hue < 0.04) | (hue > 0.94)  
    is_blood = inside_view & is_red & (saturation > min_saturation) & (brightness < max_brightness)
    
    return is_blood.sum() / max(inside_view.sum(), 1)


if __name__ == "__main__":
    frame_names = sorted(os.listdir(frames))
    
    with ProcessPoolExecutor() as workers:
        fractions = list(workers.map(blood_amount, [f"{frames}/{name}" for name in frame_names], chunksize=64))

    frame_to_trial = pd.read_csv("data/frame_to_trial_mapping.csv")
    blood = pd.DataFrame({"frame": frame_names, "blood_amt": fractions}).merge(frame_to_trial, on="frame")
    blood.to_csv("data/blood_coverage.csv", index=False)
    print(blood["blood_amt"].describe())
