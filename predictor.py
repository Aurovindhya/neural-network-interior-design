"""
predictor.py

Inference logic: loads the trained model, preprocesses an image,
runs a forward pass, and attaches a palette recommendation.
"""

import io
import time
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from model.model import InteriorStyleClassifier, STYLE_CLASSES, load_model
from model.dataset import get_inference_transforms
from api.schemas import PaletteRecommendation


# Curated palettes per style — hex codes from real interior photography
STYLE_PALETTES: Dict[str, PaletteRecommendation] = {
    "Mid-Century Modern": PaletteRecommendation(
        primary=["#8B4513", "#D2691E", "#F5DEB3", "#A0522D"],
        accent=["#2F4F4F", "#708090", "#B8860B"],
        description="Warm walnut tones with slate and gold accents — hallmarks of mid-century modern.",
    ),
    "Scandinavian": PaletteRecommendation(
        primary=["#F8F8F0", "#E8E8E0", "#D4CFC8", "#FFFFFF"],
        accent=["#4A4A4A", "#7D9B8A", "#C4956A"],
        description="Crisp whites and warm neutrals with sage and natural wood accents.",
    ),
    "Industrial": PaletteRecommendation(
        primary=["#4A4A4A", "#696969", "#808080", "#2C2C2C"],
        accent=["#B87333", "#8B6914", "#A0522D"],
        description="Raw concrete greys anchored by copper and aged brass.",
    ),
    "Bohemian": PaletteRecommendation(
        primary=["#C4956A", "#D4956A", "#E8B89A", "#A0522D"],
        accent=["#6B8E6B", "#8B6914", "#9B4E63", "#4682B4"],
        description="Warm terracottas and spice tones layered with jewel accents — rich and eclectic.",
    ),
    "Minimalist": PaletteRecommendation(
        primary=["#FAFAFA", "#F0F0F0", "#E0E0E0", "#FFFFFF"],
        accent=["#1A1A1A", "#A0A0A0", "#D4C5B0"],
        description="Near-whites and soft greys with a single warm stone accent — restrained and calm.",
    ),
}


class Predictor:
    def __init__(self, weights_path: str, device: str = "cpu"):
        self.device = device
        self.model = load_model(weights_path, device=device)
        self.transform = get_inference_transforms()
        self.class_names = STYLE_CLASSES
        print(f"Model loaded on {device} | Classes: {self.class_names}")

    def predict(self, image_bytes: bytes) -> Tuple[str, float, Dict[str, float], float]:
        """
        Run inference on raw image bytes.

        Returns:
            style: predicted class name
            confidence: softmax confidence for top class
            all_scores: dict of class -> confidence
            inference_ms: latency
        """
        t0 = time.time()

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        probs_list = probs.cpu().tolist()
        all_scores = {cls: round(float(p), 4) for cls, p in zip(self.class_names, probs_list)}

        top_idx = int(probs.argmax())
        style = self.class_names[top_idx]
        confidence = float(probs[top_idx])
        inference_ms = (time.time() - t0) * 1000

        return style, confidence, all_scores, round(inference_ms, 2)

    def get_palette(self, style: str) -> PaletteRecommendation:
        return STYLE_PALETTES.get(style, STYLE_PALETTES["Minimalist"])
