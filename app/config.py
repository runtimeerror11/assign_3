"""
Configuration for Cotton Plant Disease Detection App.
=====================================================

*** SINGLE FILE TO CHANGE for model / ensemble updates ***

To swap models or add ensemble:
  1. Change MODEL_PATH / MODEL_ARCH
  2. Update `load_model()` to return a new nn.Module
  3. Update `predict()` if ensemble logic is needed
  4. Update CLASS_NAMES if dataset changes.

Everything else (routes, UI) stays untouched.
"""

import os, torch, torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "efficientnet_b3_cotton.pth")

# ── Dataset meta ────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "diseased cotton leaf",
    "diseased cotton plant",
    "fresh cotton leaf",
    "fresh cotton plant",
]
NUM_CLASSES = len(CLASS_NAMES)

# ── Image pre-processing (must match training!) ────────────────────────────
IMG_SIZE = 224
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model loader ────────────────────────────────────────────────────────────
def _build_efficientnet_b3(num_classes: int) -> nn.Module:
    """Build EfficientNet‑B3 with the same head used during training."""
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


def load_model() -> nn.Module:
    """
    Load the trained model from disk and return it in eval mode.

    *** Modify this function to swap architectures or add ensembling ***
    For ensemble, return an nn.Module whose forward() averages multiple
    sub‑models' logits.
    """
    model = _build_efficientnet_b3(NUM_CLASSES)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# ── Prediction helper ──────────────────────────────────────────────────────
@torch.no_grad()
def predict(model: nn.Module, image: Image.Image):
    """
    Run inference on a single PIL Image.

    Returns
    -------
    class_name : str
    confidence : float  (0‑100 %)
    all_probs  : dict   {class_name: probability}
    """
    tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1).squeeze().cpu().tolist()

    idx        = int(torch.argmax(torch.tensor(probs)))
    class_name = CLASS_NAMES[idx]
    confidence = probs[idx] * 100.0
    all_probs  = {CLASS_NAMES[i]: round(probs[i] * 100, 2) for i in range(NUM_CLASSES)}

    return class_name, confidence, all_probs
