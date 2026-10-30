"""
SafeNest AI - Intelligent Accommodation Discovery Server
Author: AI/ML Engineering Lead
"""

import os
from flask import Flask, render_template, request, jsonify
from core.ml_engine import SafeNestRecommenderEngine

app = Flask(__name__)

# Initialize recommendation engine with dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "pg_dataset.json")

engine = SafeNestRecommenderEngine(DATASET_PATH)


@app.route("/", methods=["GET"])
def index():
    query = request.args.get("query", "").strip()
    try:
        budget = int(request.args.get("budget", 15000))
    except ValueError:
        budget = 15000

    try:
        min_safety = float(request.args.get("min_safety", 40.0))
    except ValueError:
        min_safety = 40.0

    results = engine.recommend(
        query=query,
        max_budget=budget,
        min_safety_score=min_safety,
        alpha_semantic=0.6,
        alpha_safety=0.4
    )

    return render_template(
        "index.html",
        results=results,
        query=query,
        budget=budget,
        min_safety=int(min_safety)
    )


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """REST API endpoint for programmatic access or mobile clients."""
    data = request.get_json(silent=True) or {}
    query = data.get("query", "")
    budget = int(data.get("budget", 15000))
    min_safety = float(data.get("min_safety", 40.0))

    results = engine.recommend(
        query=query,
        max_budget=budget,
        min_safety_score=min_safety
    )
    return jsonify({
        "status": "success",
        "count": len(results),
        "results": results
    })


if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 SafeNest AI Server running on http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=True)
