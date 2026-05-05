"""
Flask routes for the Cotton Plant Disease Detection web app.
"""

import os, io, base64, uuid
from flask import Blueprint, render_template, request, jsonify
from PIL import Image
from .config import load_model, predict, CLASS_NAMES

main = Blueprint("main", __name__)

# Load model once at startup (kept in module scope)
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


@main.route("/", methods=["GET"])
def index():
    return render_template("index.html", class_names=CLASS_NAMES)


@main.route("/predict", methods=["POST"])
def predict_route():
    """Accept an image upload and return prediction JSON."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    model = _get_model()
    class_name, confidence, all_probs = predict(model, img)

    # Encode thumbnail for display
    buf = io.BytesIO()
    img_thumb = img.copy()
    img_thumb.thumbnail((400, 400))
    img_thumb.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    # Determine health status
    is_healthy = "fresh" in class_name.lower()

    return jsonify({
        "class_name":  class_name,
        "confidence":  round(confidence, 2),
        "all_probs":   all_probs,
        "is_healthy":  is_healthy,
        "image_b64":   img_b64,
    })
