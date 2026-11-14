"""
backend/predictor.py
--------------------
Inference predictor for SmartSpam.
Loads the calibrated LinearSVC model and TF-IDF vectorizer,
reusing the exact text preprocessing pipeline from ml_pipeline.preprocess.
"""

import os
import sys
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root is accessible
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.config import PRIMARY_MODEL_PATH, BASELINE_MODEL_PATH, VECTORIZER_PATH
from ml_pipeline.preprocess import clean_text


class SpamPredictor:
    """
    Spam classification inference engine.
    Loads calibrated model and vectorizer with graceful fallback and error handling.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        vectorizer_path: Optional[Path] = None
    ):
        self.model_path = Path(model_path or PRIMARY_MODEL_PATH)
        self.vectorizer_path = Path(vectorizer_path or VECTORIZER_PATH)
        self.model = None
        self.vectorizer = None
        self.model_name = "Calibrated LinearSVC"
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads model and vectorizer from disk with fallback handling."""
        # 1. Load Vectorizer
        if not self.vectorizer_path.exists():
            raise FileNotFoundError(
                f"TF-IDF Vectorizer artifact not found at: {self.vectorizer_path}. "
                "Please run model training pipeline first."
            )
        try:
            with open(self.vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load vectorizer artifact: {e}")

        # 2. Load Model (Primary Calibrated LinearSVC, with fallback to Naive Bayes if needed)
        target_path = self.model_path
        if not target_path.exists():
            print(f"[PREDICTOR WARNING] Primary model not found at {target_path}. Checking baseline...")
            if BASELINE_MODEL_PATH.exists():
                target_path = BASELINE_MODEL_PATH
                self.model_name = "Multinomial Naive Bayes (Baseline)"
            else:
                raise FileNotFoundError(
                    f"No trained model found at {self.model_path} or {BASELINE_MODEL_PATH}."
                )

        try:
            with open(target_path, "rb") as f:
                self.model = pickle.load(f)
                if target_path == PRIMARY_MODEL_PATH:
                    self.model_name = "Calibrated LinearSVC"
        except Exception as e:
            raise RuntimeError(f"Failed to load model artifact from {target_path}: {e}")

        print(f"[PREDICTOR] Successfully loaded {self.model_name} and Vectorizer.")

    def predict(self, raw_message: str) -> Dict[str, Any]:
        """
        Classifies a raw text message.
        
        Returns:
            Dict containing classification ('Spam' or 'Ham'), confidence percentage,
            probabilities, cleaned representation, and model identifier.
        """
        if not raw_message or not isinstance(raw_message, str) or not raw_message.strip():
            raise ValueError("Input message must be a non-empty string.")

        # Preprocess using exact training pipeline
        cleaned_text = clean_text(raw_message)

        # Handle edge case: message contained only stripped punctuation/stopwords
        if not cleaned_text.strip():
            # If all content was stripped (e.g. emoticons/greetings), treat as legitimate Ham
            # with moderate default confidence
            return {
                "classification": "Ham",
                "is_spam": False,
                "confidence": 95.0,
                "probability_spam": 0.05,
                "probability_ham": 0.95,
                "cleaned_text": "<empty / non-lexical tokens>",
                "model_name": self.model_name
            }

        # Vectorize
        features = self.vectorizer.transform([cleaned_text])

        # Inference
        pred_label = int(self.model.predict(features)[0])
        is_spam = bool(pred_label == 1)
        classification = "Spam" if is_spam else "Ham"

        # Probability / Confidence
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)[0]
            prob_ham = float(proba[0])
            prob_spam = float(proba[1])
            confidence = round((prob_spam if is_spam else prob_ham) * 100, 2)
        elif hasattr(self.model, "decision_function"):
            decision = float(self.model.decision_function(features)[0])
            # Sigmoid conversion
            import math
            prob_spam = 1.0 / (1.0 + math.exp(-decision))
            prob_ham = 1.0 - prob_spam
            confidence = round((prob_spam if is_spam else prob_ham) * 100, 2)
        else:
            prob_spam = 1.0 if is_spam else 0.0
            prob_ham = 0.0 if is_spam else 1.0
            confidence = 100.0

        return {
            "classification": classification,
            "is_spam": is_spam,
            "confidence": confidence,
            "probability_spam": round(prob_spam, 4),
            "probability_ham": round(prob_ham, 4),
            "cleaned_text": cleaned_text,
            "model_name": self.model_name
        }


# Global singleton predictor instance
_predictor_instance = None


def get_predictor() -> SpamPredictor:
    """Returns or initializes the singleton predictor."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = SpamPredictor()
    return _predictor_instance


if __name__ == "__main__":
    predictor = get_predictor()
    test_msg = "URGENT! You have won £1,000 cash. Call 09061743806 now to claim!"
    result = predictor.predict(test_msg)
    print("Test Prediction:")
    for k, v in result.items():
        print(f"  {k}: {v}")

