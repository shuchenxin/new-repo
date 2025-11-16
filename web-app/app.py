"""
Flask web application for displaying music classification results.
Music classification: Vocal vs Instrumental.
"""

from datetime import datetime
import os

from flask import Flask, render_template, jsonify, request, redirect, url_for
from werkzeug.utils import secure_filename

from database import get_database
from gridfs import GridFS

from flask_cors import CORS

app = Flask(__name__)
app.config.from_object("config.Config")

CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=False,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"],
    max_age=86400,
)


@app.route("/")
def index():
    """Display the main dashboard page with music classification results."""
    db = get_database()
    # Get recent classification results from ML client
    # Expected data structure:
    # {
    #   "filename": "song.mp3",
    #   "filepath": "/path/to/song.mp3",
    #   "classification": "vocal" or "instrumental",
    #   "confidence": 0.95,
    #   "timestamp": datetime,
    #   "features": {...}  # optional ML features
    # }
    recent_results = list(db.classifications.find().sort("timestamp", -1).limit(20))

    # Get statistics
    total_count = db.classifications.count_documents({})
    vocal_count = db.classifications.count_documents({"classification": "vocal"})
    instrumental_count = db.classifications.count_documents(
        {"classification": "instrumental"}
    )

    stats = {
        "total": total_count,
        "vocal": vocal_count,
        "instrumental": instrumental_count,
    }

    return render_template("index.html", results=recent_results, stats=stats)


@app.route("/api/results")
def api_results():
    """API endpoint to get classification results."""
    db = get_database()
    limit = int(app.config.get("API_LIMIT", 100))
    results = list(db.classifications.find().sort("timestamp", -1).limit(limit))

    # Convert ObjectId and datetime to string for JSON serialization
    for item in results:
        item["_id"] = str(item["_id"])
        if "timestamp" in item and isinstance(item["timestamp"], datetime):
            item["timestamp"] = item["timestamp"].isoformat()

    return jsonify({"results": results})


@app.route("/api/stats")
def api_stats():
    """API endpoint to get classification statistics."""
    db = get_database()
    total = db.classifications.count_documents({})
    vocal = db.classifications.count_documents({"classification": "vocal"})
    instrumental = db.classifications.count_documents(
        {"classification": "instrumental"}
    )

    return jsonify(
        {
            "total": total,
            "vocal": vocal,
            "instrumental": instrumental,
            "vocal_percentage": round(vocal / total * 100, 2) if total > 0 else 0,
            "instrumental_percentage": (
                round(instrumental / total * 100, 2) if total > 0 else 0
            ),
        }
    )


@app.route("/upload", methods=["POST", "OPTIONS"])
def upload_audio():
    """Handle audio file uploads from the dashboard."""
    # Handle preflight (already handled in before_request), keep for clarity
    if request.method == "OPTIONS":
        return ("", 204)

    # Accept either 'file' (from frontend) or 'audio_file' (legacy)
    file = request.files.get("file") or request.files.get("audio_file")
    if file is None or file.filename == "":
        return jsonify({"ok": False, "error": "No file provided"}), 400

    filename = secure_filename(file.filename)

    upload_dir = app.config.get(
        "UPLOAD_FOLDER",
        os.path.join(os.path.dirname(__file__), "uploads"),
    )
    os.makedirs(upload_dir, exist_ok=True)

    save_path = os.path.join(upload_dir, filename)
    file.save(save_path)

    # Save the uploaded audio into MongoDB (GridFS)
    db = get_database()
    fs = GridFS(db)
    try:
        with open(save_path, "rb") as f:
            mongo_file_id = fs.put(
                f,
                filename=filename,
                content_type=getattr(file, "mimetype", None),
                upload_time=datetime.utcnow(),
                status="pending",
                source="web-upload",
                original_path=save_path,
            )
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Failed to store file in MongoDB: {exc}"}), 500

    # Optionally enqueue for ML processing here
    return (
        jsonify(
            {
                "ok": True,
                "filename": filename,
                "path": save_path,
                "mongo_file_id": str(mongo_file_id),
                "status": "pending",
            }
        ),
        201,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
