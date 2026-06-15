# ============================================================
# Member 3 - Step 3: Flask Backend API
# Run: python backend.py
# API at: http://localhost:5000
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
from search import search_patents, get_novelty_label

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status" : "running",
        "message": "Patent Novelty API is live",
        "routes" : {
            "POST /search": "Search patents by idea",
            "GET  /health": "Health check"
        }
    })


@app.route("/search", methods=["POST"])
def search():
    """
    Input:  { "idea": "your invention description", "top_k": 3 }
    Output: { "results": [...], "novelty_score": 16.0, "assessment": {...} }
    """
    data = request.get_json()

    if not data or "idea" not in data:
        return jsonify({"error": "Please provide an 'idea' field"}), 400

    user_idea = data["idea"].strip()
    top_k     = data.get("top_k", 3)

    if not user_idea:
        return jsonify({"error": "Idea cannot be empty"}), 400

    results, novelty = search_patents(user_idea, top_k=top_k)
    label = get_novelty_label(novelty)

    return jsonify({
        "query"        : user_idea,
        "results"      : results,
        "novelty_score": novelty,
        "assessment"   : label
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("=" * 50)
    print("  Patent Novelty Flask API")
    print("  Running at: http://localhost:5000")
    print("  POST /search  with body: {'idea': 'your idea here'}")
    print("=" * 50)
    app.run(debug=True, port=5000)