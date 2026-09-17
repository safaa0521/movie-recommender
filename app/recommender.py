import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Path to the dataset (adjust if your folder structure differs)
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ml-25m", "movies.csv")

class MovieRecommender:
    def __init__(self, data_path=DATA_PATH, max_movies=20000):
        # Load the movies dataset
        self.movies = pd.read_csv(data_path)

        # For speed, use a subset (25M dataset has 62k+ movies; TF-IDF on all is fine but slow to reload each run)
        self.movies = self.movies.head(max_movies).reset_index(drop=True)

        # Clean genres: MovieLens uses "|" as separator, replace with space for TF-IDF
        self.movies["genres_clean"] = self.movies["genres"].fillna("").str.replace("|", " ", regex=False)

        # Build TF-IDF matrix over genres
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies["genres_clean"])

        # Precompute cosine similarity matrix
        self.similarity = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        # Map title (lowercase) -> index, for lookup
        self.title_to_index = pd.Series(
            self.movies.index, index=self.movies["title"].str.lower()
        )

    def search_titles(self, query, limit=10):
        """Return movie titles containing the query string (case-insensitive)."""
        query = query.lower()
        matches = self.movies[self.movies["title"].str.lower().str.contains(query, na=False)]
        return matches["title"].head(limit).tolist()

    def recommend(self, title, top_n=10):
        """Given an exact movie title, return top_n similar movies by genre similarity."""
        title_lower = title.lower()
        if title_lower not in self.title_to_index:
            return None  # not found

        idx = self.title_to_index[title_lower]
        # Handle duplicate titles (take first match)
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        sim_scores = list(enumerate(self.similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]  # skip itself

        movie_indices = [i[0] for i in sim_scores]
        results = self.movies.iloc[movie_indices][["title", "genres"]].to_dict(orient="records")
        return results


if __name__ == "__main__":
    # Quick manual test
    rec = MovieRecommender()
    print("Loaded", len(rec.movies), "movies")
    test_title = rec.movies.iloc[0]["title"]
    print(f"Recommendations for '{test_title}':")
    for r in rec.recommend(test_title, top_n=5):
        print(" -", r["title"], "|", r["genres"])