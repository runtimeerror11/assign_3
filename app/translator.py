"""
Sarvam AI Translator for Indian Languages.
==========================================

Set env var:  SARVAM_API_KEY=your-key-here
Get a key at: https://dashboard.sarvam.ai
"""

import os, requests, traceback

SARVAM_API_URL = "https://api.sarvam.ai/translate"

LANGUAGES = {
    "en":    "English",
    "hi-IN": "हिन्दी (Hindi)",
    "bn-IN": "বাংলা (Bengali)",
    "gu-IN": "ગુજરાતી (Gujarati)",
    "kn-IN": "ಕನ್ನಡ (Kannada)",
    "ml-IN": "മലയാളം (Malayalam)",
    "mr-IN": "मराठी (Marathi)",
    "od-IN": "ଓଡ଼ିଆ (Odia)",
    "pa-IN": "ਪੰਜਾਬੀ (Punjabi)",
    "ta-IN": "தமிழ் (Tamil)",
    "te-IN": "తెలుగు (Telugu)",
}


def translate_text(text: str, target_lang: str) -> dict:
    """
    Translate text to an Indian language using Sarvam AI.

    Returns dict with:
        "translated_text": str
        "success": bool
    """
    if target_lang == "en" or not target_lang:
        return {"translated_text": text, "success": True}

    api_key = os.environ.get("SARVAM_API_KEY", "").strip()
    if not api_key:
        return {"translated_text": text, "success": False}

    try:
        resp = requests.post(
            SARVAM_API_URL,
            json={
                "input": text,
                "source_language_code": "en-IN",
                "target_language_code": target_lang,
                "speaker_gender": "Male",
                "mode": "formal",
                "model": "mayura:v1",
                "enable_preprocessing": True,
            },
            headers={
                "api-subscription-key": api_key,
                "Content-Type": "application/json",
            },
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {"translated_text": data.get("translated_text", text), "success": True}
        else:
            print(f"Sarvam API error {resp.status_code}: {resp.text}")
            return {"translated_text": text, "success": False}
    except Exception:
        traceback.print_exc()
        return {"translated_text": text, "success": False}
