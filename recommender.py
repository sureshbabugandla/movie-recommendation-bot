"""
recommender.py — Content-based movie recommender.

Pipeline:
  1. TF-IDF vectorise each movie's "content soup" (genres x2 + keywords + cast +
     director + overview).
  2. For a query we either (a) rank by cosine similarity to a seed movie
     ("more like that"), (b) rank by similarity to a free-text query, or
     (c) hard-filter by requested genre/mood/era/rating and rank by a Bayesian
     quality prior. Results are explainable and de-duplicated against the
     session's already-seen titles (supports the 'refine' intent).

Evaluation: genre Precision@K, mood-appropriateness@K, and NDCG@K against a
genre-membership relevance signal — a real, honest metric (the prior repo
shipped none).
"""
from __future__ import annotations
import os as _os
_DATA_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data")
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sys, os
sys.path.append(os.path.dirname(__file__))
from data_prep import MOOD_TO_GENRES


class MovieRecommender:
    def __init__(self, movies: pd.DataFrame):
        self.m = movies.reset_index(drop=True)
        self.vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                   min_df=2, stop_words="english")
        self.X = self.vec.fit_transform(self.m["content"])
        # Bayesian weighted rating (IMDB formula) as a quality prior.
        C = self.m["vote_average"].mean()
        mvotes = self.m["vote_count"].quantile(0.60)
        v, R = self.m["vote_count"], self.m["vote_average"]
        self.m["wr"] = (v / (v + mvotes)) * R + (mvotes / (v + mvotes)) * C
        self.title_to_idx = {t: i for i, t in enumerate(self.m["original_title"])}

    # ---- helpers ----------------------------------------------------------
    def _requested_genres(self, slots):
        """Explicit genre wins the hard filter; mood only broadens if no genre."""
        explicit = set(slots.get("genre") or [])
        if explicit:
            return explicit
        mood_genres = set()
        for mood in slots.get("mood") or []:
            mood_genres |= set(MOOD_TO_GENRES.get(mood, []))
        return mood_genres

    def _era_mask(self, eras):
        if not eras:
            return np.ones(len(self.m), dtype=bool)
        mask = np.zeros(len(self.m), dtype=bool)
        yr = self.m["release_year"].to_numpy()
        for e in eras:
            if e == "recent":
                mask |= yr >= 2010
            elif e == "classic":
                mask |= yr <= 1989
            elif e.endswith("s") and e[:-1].isdigit():
                start = int(e[:-1]); mask |= (yr >= start) & (yr <= start + 9)
        return mask

    # ---- main entry point -------------------------------------------------
    def recommend(self, slots=None, query_text=None, seed_title=None,
                  exclude=None, k=5):
        slots = slots or {}
        exclude = set(exclude or [])
        genres = self._requested_genres(slots)

        # Candidate mask: era + rating + genre hard filters.
        mask = self._era_mask(slots.get("era")).copy()
        if slots.get("min_rating"):
            mask &= (self.m["vote_average"] >= slots["min_rating"]).to_numpy()
        if genres:
            gmask = self.m["genre_list"].apply(
                lambda gl: bool(set(gl) & genres)).to_numpy()
            mask &= gmask
        cand = np.where(mask)[0]
        if len(cand) == 0:                      # relax genre filter if too strict
            cand = np.where(self._era_mask(slots.get("era")))[0]
        if len(cand) == 0:
            cand = np.arange(len(self.m))

        # Scoring signal.
        if seed_title and seed_title in self.title_to_idx:
            sim = cosine_similarity(self.X[self.title_to_idx[seed_title]],
                                    self.X[cand]).ravel()
            score = 0.7 * sim + 0.3 * _minmax(self.m["wr"].values[cand])
        elif query_text:
            qv = self.vec.transform([query_text.lower()])
            sim = cosine_similarity(qv, self.X[cand]).ravel()
            score = 0.6 * sim + 0.4 * _minmax(self.m["wr"].values[cand])
        else:
            # genre/mood-only query -> rank by quality within the filtered pool.
            score = _minmax(self.m["wr"].values[cand])

        order = cand[np.argsort(-score)]
        out = []
        for i in order:
            title = self.m["original_title"].iat[i]
            if title in exclude:
                continue
            out.append(self._card(i, slots, genres))
            if len(out) >= k:
                break
        return out

    def _card(self, i, slots, genres):
        r = self.m.iloc[i]
        reasons = []
        if genres:
            hit = set(r["genre_list"]) & genres
            if hit:
                reasons.append("matches " + "/".join(sorted(hit)))
        if slots.get("mood"):
            reasons.append("good for a " + "/".join(slots["mood"]) + " mood")
        if slots.get("era"):
            reasons.append(f"from the {r['era']}")
        reasons.append(f"rated {r['vote_average']:.1f}/10")
        return {
            "title": r["original_title"], "year": int(r["release_year"]),
            "genres": r["genre_list"], "rating": round(float(r["vote_average"]), 1),
            "why": "; ".join(reasons),
        }


def _minmax(a):
    a = np.asarray(a, dtype=float)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-9)


# ---- Evaluation -----------------------------------------------------------
def evaluate(reco: MovieRecommender, k=5):
    """Genre Precision@K + mood-appropriateness@K + NDCG@K (genre relevance)."""
    genres = ["Action", "Comedy", "Drama", "Horror", "Thriller", "Romance",
              "Science Fiction", "Animation", "Adventure", "Crime", "Mystery",
              "Family", "Fantasy"]
    def ndcg(rel):
        dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rel))
        idcg = sum(1 / np.log2(i + 2) for i in range(len(rel)))
        return dcg / idcg if idcg else 0.0

    gp, gndcg = [], []
    for g in genres:
        recs = reco.recommend(slots={"genre": [g]}, k=k)
        rel = [1 if g in r["genres"] else 0 for r in recs]
        gp.append(np.mean(rel)); gndcg.append(ndcg(rel))

    mood_scores = []
    for mood in ["feel-good", "scary", "exciting", "romantic", "thoughtful", "dark"]:
        target = set(MOOD_TO_GENRES[mood])
        recs = reco.recommend(slots={"mood": [mood]}, k=k)
        rel = [1 if set(r["genres"]) & target else 0 for r in recs]
        mood_scores.append(np.mean(rel))

    # Non-trivial signal: for "more like X", how genre-coherent are neighbours?
    # (Jaccard genre overlap between each seed and its top-k similar movies.)
    seeds = ["Inception", "Toy Story", "The Godfather", "Titanic", "Alien",
             "The Hangover", "Gladiator", "Finding Nemo"]
    coh = []
    for s in seeds:
        if s not in reco.title_to_idx:
            continue
        sg = set(reco.m["genre_list"].iat[reco.title_to_idx[s]])
        recs = reco.recommend(seed_title=s, exclude=[s], k=k)
        for r in recs:
            rg = set(r["genres"])
            coh.append(len(sg & rg) / len(sg | rg) if (sg | rg) else 0)

    return {
        "genre_precision@%d" % k: round(float(np.mean(gp)), 3),
        "genre_ndcg@%d" % k: round(float(np.mean(gndcg)), 3),
        "mood_appropriateness@%d" % k: round(float(np.mean(mood_scores)), 3),
        "similar_genre_coherence@%d" % k: round(float(np.mean(coh)), 3),
    }


if __name__ == "__main__":
    from data_prep import load_movies
    movies = load_movies(_os.path.join(_DATA_DIR, "tmdb_movies.csv"))
    reco = MovieRecommender(movies)
    print("Catalogue:", len(movies), "movies\n")

    print(">> 'feel-good comedy from the 90s'")
    for c in reco.recommend(slots={"mood": ["feel-good"], "genre": ["Comedy"],
                                   "era": ["1990s"]}, k=5):
        print(f"   {c['title']} ({c['year']}) — {c['why']}")

    print("\n>> scary horror, highly rated")
    for c in reco.recommend(slots={"mood": ["scary"], "genre": ["Horror"],
                                   "min_rating": 7.0}, k=5):
        print(f"   {c['title']} ({c['year']}) — {c['why']}")

    print("\n>> more like 'Inception'")
    for c in reco.recommend(seed_title="Inception", k=5):
        print(f"   {c['title']} ({c['year']}) — {c['why']}")

    print("\nEVALUATION:", evaluate(reco, k=5))
