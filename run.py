"""
Cotton Plant Disease Detection – entry point.

Usage
-----
    python run.py          # launches Flask dev server on http://localhost:5000
"""

from flask import Flask
from app.routes import main

def create_app():
    app = Flask(
        __name__,
        template_folder="app/templates",
        static_folder="app/static",
    )
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload
    app.register_blueprint(main)
    return app


if __name__ == "__main__":
    app = create_app()
    print("\n🌿  Cotton Plant Disease Detection")
    print("   → http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
