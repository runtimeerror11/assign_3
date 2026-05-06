"""
Gemini LLM Advisor for Cotton Plant Disease Detection
====================================================

Uses Google Gemini (google-genai SDK) to generate short,
farmer-friendly advice based on:
1. Predicted cotton disease class
2. Confidence score
3. Uploaded image (leaf or plant)

Falls back to hardcoded advice if:
- No API key
- API failure
- Invalid response

SETUP (PowerShell)
------------------
$env:GEMINI_API_KEY="your_api_key_here"

INSTALL
-------
pip uninstall google google-generativeai -y
pip install -U google-genai pillow
"""

import os
import json
import io
import traceback
from PIL import Image
from dotenv import load_dotenv

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_FILE = os.path.join(BASE_DIR, "advice_cache.json")
DOTENV_FILE = os.path.join(BASE_DIR, ".env")

load_dotenv(DOTENV_FILE)


# ── Prompt Template ─────────────────────────────────────────────────────────
_PROMPT = """
You are a helpful farming advisor talking to a cotton farmer.

An AI checked the attached cotton plant or leaf photo and found:
{class_name}
Confidence: {confidence:.1f}%

Write a short reply for the farmer (3-4 simple sentences).

Rules:
- Use very simple farmer-friendly words.
- Avoid scientific terms unless very common.
- Clearly say if the plant is healthy or unhealthy.
- If unhealthy, explain what may be wrong.
- Give easy practical next steps.
- Give one prevention tip.
- Be kind, short, and encouraging.
"""


# ── Cache Helpers ───────────────────────────────────────────────────────────
def _load_cache() -> dict:
    """Load advice cache from JSON file."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache(cache: dict):
    """Save advice cache to JSON file."""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception:
        print("⚠️ Could not save cache file.")


# ── Public API ──────────────────────────────────────────────────────────────
def get_advice(class_name: str, confidence: float, image: Image.Image) -> dict:
    """
    Returns:
    {
        "advice": str,
        "source": "cache" | "gemini" | "fallback"
    }
    """

    # Normalize class name
    cache_key = class_name.strip().lower()

    # 1. Check cache first
    cache = _load_cache()
    if cache_key in cache:
        return {
            "advice": cache[cache_key],
            "source": "cache"
        }

    # 2. Get API Key
    api_key = get_gemini_api_key()

    if not api_key:
        print("⚠️ Gemini API key not found. Using fallback.")
        return {
            "advice": _fallback_advice(class_name),
            "source": "fallback"
        }

    # 3. Call Gemini
    try:
        advice_text = _call_gemini(
            api_key=api_key,
            class_name=class_name,
            confidence=confidence,
            image=image
        )

        # Cache successful result
        cache[cache_key] = advice_text
        _save_cache(cache)

        return {
            "advice": advice_text,
            "source": "gemini"
        }

    except Exception as e:
        print("\n⚠️ FULL GEMINI ERROR:")
        print(str(e))
        traceback.print_exc()

        return {
            "advice": _fallback_advice(class_name),
            "source": "fallback"
        }


def get_gemini_api_key() -> str:
    """Return the Gemini API key from the environment or .env file."""
    return (
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )


# ── Gemini API Call ─────────────────────────────────────────────────────────
# ── Replace ONLY your _call_gemini() function with this EXACT version ──

def _call_gemini(api_key: str, class_name: str, confidence: float, image: Image.Image) -> str:
    from google import genai
    from google.genai import types
    import io

    client = genai.Client(api_key=api_key)

    # Resize and convert image to bytes
    img_copy = image.copy()
    img_copy.thumbnail((512, 512))
    buf = io.BytesIO()
    img_copy.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    prompt = _PROMPT.format(class_name=class_name, confidence=confidence)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            types.Part.from_text(text=prompt),
        ]
    )

    if not response or not response.text:
        raise ValueError("Empty response from Gemini.")

    return response.text.strip()

# ── Fallback Advice ─────────────────────────────────────────────────────────
_FALLBACK = {
    "diseased cotton leaf": (
        "This cotton leaf looks unhealthy. It may have disease, fungus, or moisture damage. "
        "Remove badly affected leaves and spray neem or proper crop medicine. "
        "Check nearby leaves daily so the problem does not spread."
    ),

    "diseased cotton plant": (
        "This cotton plant does not look healthy. It may be affected by pests, disease, or watering problems. "
        "Keep infected plants away if possible and check soil and insects carefully. "
        "Visit your local agriculture officer for the right treatment."
    ),

    "fresh cotton leaf": (
        "Good news! This cotton leaf looks healthy. Keep regular watering, monitor insects, "
        "and continue proper care. Your crop looks in good condition."
    ),

    "fresh cotton plant": (
        "This cotton plant looks healthy and strong. Continue proper sunlight, watering, "
        "and regular pest checks. Keep watching for yellow or brown spots early."
    ),
}


def _fallback_advice(class_name: str) -> str:
    """Return hardcoded fallback advice."""
    return _FALLBACK.get(
        class_name.strip().lower(),
        "Unable to generate advice right now. Please consult a local agriculture expert."
    )
