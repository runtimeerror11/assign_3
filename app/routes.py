"""
Flask routes for the Cotton Plant Disease Detection web app.
"""

import io
import base64
from flask import Blueprint, render_template, request, jsonify
from PIL import Image
from .config import load_model, predict, CLASS_NAMES
from .gemini_advisor import get_advice
from .translator import translate_text, LANGUAGES

main = Blueprint("main", __name__)

# Load model once at startup
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


@main.route("/", methods=["GET"])
def index():
    return render_template("index.html", class_names=CLASS_NAMES, languages=LANGUAGES)


@main.route("/predict", methods=["POST"])
def predict_route():
    """Accept an image upload and return prediction JSON + LLM advice."""
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

    is_healthy = "fresh" in class_name.lower()

    # Get farmer-friendly advice
    advice_result = get_advice(class_name, confidence, img)

    return jsonify({
        "class_name":    class_name,
        "confidence":    round(confidence, 2),
        "all_probs":     all_probs,
        "is_healthy":    is_healthy,
        "image_b64":     img_b64,
        "advice":        advice_result["advice"],
        "filename":      file.filename,
    })


@main.route("/translate", methods=["POST"])
def translate_route():
    """Translate advice text to an Indian language via Sarvam AI."""
    data = request.get_json()
    if not data or "text" not in data or "lang" not in data:
        return jsonify({"error": "Missing text or lang"}), 400

    result = translate_text(data["text"], data["lang"])
    return jsonify(result)
