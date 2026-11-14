# Turn every frame of the first minute into numbers with two pretrained image models:
#  - DINOv2 (self-supervised, never told what a tool or blood is)
#  - ResNet-50 (ImageNet), the backbone SOCALNet used, for the head-to-head
# Saved to data/embeddings_<model>.npz with one row per (trial, second).
import sys
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms
from calibre.models.tool_detector_training import frames_folder

seconds_kept = 60
device = "cuda" if torch.cuda.is_available() else "mps"
prepare = transforms.Compose([
    transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_backbone(name):
    if name == "dinov2":
        from transformers import AutoModel  # weights only (safetensors), no downloaded code is run
        dino = AutoModel.from_pretrained("facebook/dinov2-small", use_safetensors=True).eval().to(device)
        return lambda batch: dino(pixel_values=batch).pooler_output
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    resnet.fc = torch.nn.Identity()
    return resnet.eval().to(device)


def embed_frames(name, batch_size=64):
    frames = pd.read_csv("data/frame_to_trial_mapping.csv")
    labeled_trials = pd.read_csv("data/socal.csv", header=None)[0].str[:6].unique()
    frames = frames[frames["trial_id"].isin(labeled_trials) & (frames["frame_number"] <= seconds_kept)]
    backbone = load_backbone(name)
    outputs = []
    names = frames["frame"].tolist()
    with torch.no_grad():
        for start in range(0, len(names), batch_size):
            batch = torch.stack([prepare(Image.open(f"{frames_folder}/{n}").convert("RGB"))
                                 for n in names[start:start + batch_size]]).to(device)
            outputs.append(backbone(batch).float().cpu().numpy())
            if start % (batch_size * 20) == 0:
                print(f"{name}: {start}/{len(names)}", flush=True)
    np.savez_compressed(f"data/embeddings_{name}.npz", embeddings=np.concatenate(outputs),
                        trial_id=frames["trial_id"].values, second=frames["frame_number"].values)


if __name__ == "__main__":
    for name in sys.argv[1:] or ["dinov2", "resnet50"]:
        embed_frames(name)
