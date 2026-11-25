"""Inference code for the final model: an ensemble of EfficientNet-B0 and MobileNetV3-Large.

The ensemble score of a class is the mean of the members' softmax outputs.
These are model scores, not calibrated probabilities.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import models, transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def build_model(arch: str, n_classes: int = len(CLASSES)) -> nn.Module:
    """Same head layout as in training (weights come from the checkpoint, not from ImageNet)."""
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, n_classes))
    elif arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=None)
        m.classifier = nn.Sequential(nn.Linear(m.classifier[0].in_features, 1280), nn.Hardswish(),
                                     nn.Dropout(0.3), nn.Linear(1280, n_classes))
    else:
        raise ValueError(f"unknown architecture: {arch}")
    return m


class WasteClassifier:
    def __init__(self, config_path=None, device=None, members=None):
        config_path = Path(config_path) if config_path else REPO_ROOT / "models" / "config.json"
        self.cfg = json.loads(config_path.read_text())
        self.classes = self.cfg["classes"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        h, w = self.cfg["input_size_hw"]
        self.size = (w, h)  # PIL wants (width, height)
        self.tf = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(self.cfg["mean"], self.cfg["std"])])
        self.members = []
        for spec in self.cfg["members"]:
            if members is not None and spec["name"] not in members:
                continue
            m = build_model(spec["arch"], len(self.classes))
            state = torch.load(config_path.parent / spec["file"], map_location="cpu", weights_only=True)
            m.load_state_dict(state)
            self.members.append((spec["name"], m.to(self.device).eval()))

    def load_image(self, src) -> Image.Image:
        img = src if isinstance(src, Image.Image) else Image.open(src)
        img = ImageOps.exif_transpose(img).convert("RGB")      # phone photos carry a rotation flag
        return img.resize(self.size)                           # same resize as in training (256 x 192, 4:3)

    @torch.no_grad()
    def predict_batch(self, images):
        """images: list of paths or PIL images -> array (n, n_classes) of ensemble softmax scores."""
        x = torch.stack([self.tf(self.load_image(i)) for i in images]).to(self.device)
        probs = [F.softmax(m(x), dim=1).float().cpu().numpy() for _, m in self.members]
        return np.mean(probs, axis=0)

    def predict(self, src, top_k=3):
        scores = self.predict_batch([src])[0]
        order = np.argsort(-scores)[:top_k]
        return {"label": self.classes[order[0]],
                "top": [{"label": self.classes[i], "score": float(scores[i])} for i in order],
                "scores": {c: float(s) for c, s in zip(self.classes, scores)}}
