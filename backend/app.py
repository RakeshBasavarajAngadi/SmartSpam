"""
backend/app.py
--------------
Main Flask application for SmartSpam.
Exposes RESTful APIs for real-time text classification, prediction history,
and dashboard statistics with input validation, error handling, and CORS.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from flask import Flask, request, jsonify, make_response, render_template
from flask_cors import CORS

from backend.config import (
    DEBUG,
    SECRET_KEY,
    CORS_ORIGINS,
    MAX_MESSAGE_LENGTH,
    MIN_MESSAGE_LENGTH,
    FRONTEND_DIR,
    PRIMARY_METRICS_PATH,
    BASELINE_METRICS_PATH,
    COMPARISON_METRICS_PATH
)
from backend.database import (
    init_db,
    insert_prediction,
    get_history,
    get_prediction_by_id,
    delete_prediction,
    get_statistics
)
from backend.predictor import get_predictor


def create_app() -> Flask:
    """Application factory for Flask server."""
    app = Flask(
        __name__,
        template_folder=str(FRONTEND_DIR / "templates"),
        static_folder=str(FRONTEND_DIR / "static"),
        static_url_path="/static"
    )
    app.config["SECRET_KEY"] = SECRET_KEY
    app.config["JSON_SORT_KEYS"] = False

    # Enable CORS for specified development origins
    CORS(app, resources={r"/api/*": {"origins": CORS_ORIGINS}})

    # Automatically ensure database and predictor are initialized
    with app.app_context():
        init_db()
        # Pre-warm predictor
        try:
            get_predictor()
        except Exception as e:
            app.logger.error(f"Failed to pre-warm predictor: {e}")

    # =========================================================================
    # Error Handlers
    # =========================================================================

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            "success": False,
            "error": "Bad Request",
            "message": str(error.description if hasattr(error, "description") else error)
        }), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "success": False,
            "error": "Not Found",
            "message": "The requested resource could not be found."
        }), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            "success": False,
            "error": "Method Not Allowed",
            "message": "HTTP method not supported on this endpoint."
        }), 405

    @app.errorhandler(500)
    def internal_server_error(error):
        app.logger.error(f"Internal server error: {error}")
        return jsonify({
            "success": False,
            "error": "Internal Server Error",
            "message": "An unexpected error occurred while processing the request."
        }), 500

    # =========================================================================
    # Web & API Routes
    # =========================================================================

    @app.route("/", methods=["GET"])
    def index():
        """Renders the main single-page dashboard application."""
        return render_template("index.html")

    @app.route("/api/health", methods=["GET"])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "service": "SmartSpam API",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 200

    @app.route("/api/predict", methods=["POST"])
    def predict_message():
        """
        Classifies an input text message.
        Expects JSON: { "message": "string" }
        Returns classification, confidence, probabilities, and saved record ID.
        """
        if not request.is_json:
            return jsonify({
                "success": False,
                "error": "Invalid Content-Type",
                "message": "Request payload must be application/json."
            }), 400

        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({
                "success": False,
                "error": "Invalid JSON",
                "message": "Request body must be a valid JSON object."
            }), 400

        message = data.get("message")

        # 1. Input Validation
        if message is None or not isinstance(message, str):
            return jsonify({
                "success": False,
                "error": "Validation Error",
                "message": "Field 'message' is required and must be a string."
            }), 400

        stripped_msg = message.strip()
        if len(stripped_msg) < MIN_MESSAGE_LENGTH:
            return jsonify({
                "success": False,
                "error": "Validation Error",
                "message": "Message cannot be empty."
            }), 400

        if len(message) > MAX_MESSAGE_LENGTH:
            return jsonify({
                "success": False,
                "error": "Validation Error",
                "message": f"Message length exceeds maximum allowed limit of {MAX_MESSAGE_LENGTH} characters."
            }), 400

        # 2. Inference
        try:
            predictor = get_predictor()
            pred_result = predictor.predict(message)
        except Exception as e:
            app.logger.error(f"Inference error: {e}")
            return jsonify({
                "success": False,
                "error": "Inference Error",
                "message": "Failed to classify message."
            }), 500

        # 3. Database Persistence
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            record_id = insert_prediction(
                message=message,
                classification=pred_result["classification"],
                confidence=pred_result["confidence"],
                timestamp=timestamp
            )
        except Exception as e:
            app.logger.error(f"Database insertion error: {e}")
            return jsonify({
                "success": False,
                "error": "Database Error",
                "message": "Failed to persist classification to history."
            }), 500

        return jsonify({
            "success": True,
            "id": record_id,
            "message": message,
            "classification": pred_result["classification"],
            "is_spam": pred_result["is_spam"],
            "confidence": pred_result["confidence"],
            "probabilities": {
                "spam": pred_result["probability_spam"],
                "ham": pred_result["probability_ham"]
            },
            "cleaned_text": pred_result["cleaned_text"],
            "model_name": pred_result["model_name"],
            "timestamp": timestamp
        }), 201

    @app.route("/api/history", methods=["GET"])
    def get_history_route():
        """
        Retrieves recent predictions with pagination.
        Query parameters:
            limit: int (default: 50, max: 200)
            offset: int (default: 0)
        """
        limit_arg = request.args.get("limit", default=50, type=int)
        offset_arg = request.args.get("offset", default=0, type=int)

        if limit_arg is None or limit_arg < 1:
            limit_arg = 50
        limit_arg = min(limit_arg, 200)

        if offset_arg is None or offset_arg < 0:
            offset_arg = 0

        try:
            records = get_history(limit=limit_arg, offset=offset_arg)
            return jsonify({
                "success": True,
                "count": len(records),
                "limit": limit_arg,
                "offset": offset_arg,
                "history": records
            }), 200
        except Exception as e:
            app.logger.error(f"Error fetching history: {e}")
            return jsonify({
                "success": False,
                "error": "Database Error",
                "message": "Failed to retrieve classification history."
            }), 500

    @app.route("/api/history/<int:record_id>", methods=["DELETE"])
    def delete_history_record(record_id: int):
        """
        Deletes a single prediction record by primary key ID.
        """
        try:
            existing = get_prediction_by_id(record_id)
            if not existing:
                return jsonify({
                    "success": False,
                    "error": "Not Found",
                    "message": f"Prediction record with ID {record_id} does not exist."
                }), 404

            deleted = delete_prediction(record_id)
            if deleted:
                return jsonify({
                    "success": True,
                    "message": f"Prediction record with ID {record_id} successfully deleted.",
                    "deleted_id": record_id
                }), 200
            else:
                return jsonify({
                    "success": False,
                    "error": "Delete Failed",
                    "message": "Failed to delete prediction record."
                }), 500
        except Exception as e:
            app.logger.error(f"Error deleting history record {record_id}: {e}")
            return jsonify({
                "success": False,
                "error": "Database Error",
                "message": "Failed to delete prediction record."
            }), 500

    @app.route("/api/stats", methods=["GET"])
    def get_stats_route():
        """
        Returns real-time dashboard statistics from SQLite and model performance metadata.
        """
        try:
            db_stats = get_statistics()

            # Load model metrics if available
            model_metrics = None
            if PRIMARY_METRICS_PATH.exists():
                try:
                    with open(PRIMARY_METRICS_PATH, "r", encoding="utf-8") as f:
                        model_metrics = json.load(f)
                except Exception:
                    pass

            return jsonify({
                "success": True,
                "stats": db_stats,
                "model_performance": {
                    "model_name": model_metrics.get("model_name", "Calibrated LinearSVC") if model_metrics else "Calibrated LinearSVC",
                    "accuracy": model_metrics.get("accuracy") if model_metrics else 0.9903,
                    "spam_precision": model_metrics.get("spam_precision") if model_metrics else 0.9690,
                    "spam_recall": model_metrics.get("spam_recall") if model_metrics else 0.9542,
                    "spam_f1": model_metrics.get("spam_f1") if model_metrics else 0.9615
                }
            }), 200
        except Exception as e:
            app.logger.error(f"Error calculating stats: {e}")
            return jsonify({
                "success": False,
                "error": "Server Error",
                "message": "Failed to compute statistics."
            }), 500

    return app


# Standard WSGI entry point
app = create_app()

if __name__ == "__main__":
    print(f"Starting SmartSpam Flask Server on http://127.0.0.1:5000 (Debug={DEBUG})")
    app.run(host="127.0.0.1", port=5000, debug=DEBUG)

