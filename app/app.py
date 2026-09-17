from flask import Flask, render_template, request, jsonify
from recommender import MovieRecommender

app = Flask(__name__, template_folder="../templates", static_folder="../static")

# Load the recommender once at startup
recommender = MovieRecommender()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/search")
def search():
    query = request.args.get("q", "")
    if not query:
        return jsonify([])
    results = recommender.search_titles(query, limit=10)
    return jsonify(results)


@app.route("/api/recommend")
def recommend():
    title = request.args.get("title", "")
    if not title:
        return jsonify({"error": "No title provided"}), 400

    results = recommender.recommend(title, top_n=10)
    if results is None:
        return jsonify({"error": f"Movie '{title}' not found"}), 404

    return jsonify({"title": title, "recommendations": results})


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)