"""
backend/config.py
-----------------
Application configuration settings for SmartSpam.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
MODELS_DIR = BASE_DIR / "models"
FRONTEND_DIR = BASE_DIR / "frontend"

# SQLite Database path (can be overridden via SMARTSPAM_DB_PATH for isolated automated tests)
def get_database_path() -> Path:
    env_path = os.environ.get("SMARTSPAM_DB_PATH")
    if env_path:
        return Path(env_path)
    return BACKEND_DIR / "smartspam.db"

DATABASE_PATH = get_database_path()

# Model and Vectorizer paths
# Primary model: Calibrated LinearSVC (highest F1 and Recall)
PRIMARY_MODEL_PATH = MODELS_DIR / "linearsvc_model.pkl"
# Baseline model: Multinomial Naive Bayes (preserved for academic comparison)
BASELINE_MODEL_PATH = MODELS_DIR / "spam_model.pkl"
# Shared fitted TF-IDF Vectorizer
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
# Metrics file paths
PRIMARY_METRICS_PATH = MODELS_DIR / "linearsvc_metrics.json"
BASELINE_METRICS_PATH = MODELS_DIR / "metrics.json"
COMPARISON_METRICS_PATH = MODELS_DIR / "model_comparison.json"

# Validation constraints
MAX_MESSAGE_LENGTH = 5000
MIN_MESSAGE_LENGTH = 1

# Flask configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "smartspam-insecure-dev-key-academic-nlp")
DEBUG = os.environ.get("FLASK_DEBUG", "True").lower() in ("true", "1")
CORS_ORIGINS = [
    "http://127.0.0.1:5000",
    "http://localhost:5000"
]

