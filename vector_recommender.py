from __future__ import annotations
import os
import numpy as np
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

import sys
sys.path.append(os.path.dirname(__file__))
from data_prep import MOOD_TO_GENRES

class VectorRecommender:
    def __init__(self, movies: pd.DataFrame, db_path="./chroma_db"):
        self.m = movies.reset_index(drop=True)
        
        # Bayesian weighted rating (IMDB formula) as a quality prior.
        C = self.m["vote_average"].mean()
        mvotes = self.m["vote_count"].quantile(0.60)
        v, R = self.m["vote_count"], self.m["vote_average"]
        self.m["wr"] = (v / (v + mvotes)) * R + (mvotes / (v + mvotes)) * C
        self.title_to_idx = {t: i for i, t in enumerate(self.m["original_title"])}

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        # Using the default all-MiniLM-L6-v2 which is highly efficient for semantic search
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.client.get_or_create_collection(
            name="movies",
            embedding_function=self.emb_fn
        )
        self.build_index()

    def build_index(self):
        if self.collection.count() > 0:
            return
        
        print("Building Chroma vector index... this will take a few moments.")
        batch_size = 250
        for i in range(0, len(self.m), batch_size):
            batch = self.m.iloc[i:i+batch_size]
            
            ids = [str(idx) for idx in batch.index]
            # Embed the 'content soup' which includes genres, keywords, cast, overview
            documents = batch["content"].tolist()
            metadatas = []
            
            for _, row in batch.iterrows():
                metadatas.append({
                    "title": str(row["original_title"]),
                    "year": int(row["release_year"]) if pd.notna(row["release_year"]) else 0,
                    "rating": float(row["vote_average"]) if pd.notna(row["vote_average"]) else 0.0,
                    "era": str(row["era"])
                })
                
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        print(f"Indexed {self.collection.count()} movies successfully.")

    def _requested_genres(self, slots):
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

    def recommend(self, slots=None, query_text=None, seed_title=None, exclude=None, k=5):
        slots = slots or {}
        exclude = set(exclude or [])
        genres = self._requested_genres(slots)

        # Candidate mask: era + rating + genre hard filters.
        mask = self._era_mask(slots.get("era")).copy()
        if slots.get("min_rating"):
            mask &= (self.m["vote_average"] >= slots.get("min_rating")).to_numpy()
        if genres:
            gmask = self.m["genre_list"].apply(
                lambda gl: bool(set(gl) & genres)).to_numpy()
            mask &= gmask
        
        cand = np.where(mask)[0]
        if len(cand) == 0:
            cand = np.where(self._era_mask(slots.get("era")))[0]
        if len(cand) == 0:
            cand = np.arange(len(self.m))

        score = np.zeros(len(self.m))
        
        # 1. Base score is normalized WR (quality prior)
        wr_norm = _minmax(self.m["wr"].values)
        
        if seed_title and seed_title in self.title_to_idx:
            # Semantic search using seed movie content
            seed_idx = self.title_to_idx[seed_title]
            query_content = self.m["content"].iloc[seed_idx]
            
            # Query ChromaDB.
            results = self.collection.query(
                query_texts=[query_content],
                n_results=min(100, len(self.m)),
                include=["distances"]
            )
            # distances are typically L2 distance. lower is better. We convert to similarity
            for c_id, dist in zip(results["ids"][0], results["distances"][0]):
                idx = int(c_id)
                score[idx] += 0.7 * (1.0 / (1.0 + dist))
                
            score += 0.3 * wr_norm
            
        elif query_text or slots.get("mood") or slots.get("genre"):
            # Construct a semantic query
            parts = []
            if query_text: parts.append(query_text)
            if slots.get("mood"): parts.append(" ".join(slots["mood"]))
            if slots.get("genre"): parts.append(" ".join(slots["genre"]))
            
            search_str = " ".join(parts)
            
            if search_str.strip():
                results = self.collection.query(
                    query_texts=[search_str],
                    n_results=min(200, len(self.m)),
                    include=["distances"]
                )
                for c_id, dist in zip(results["ids"][0], results["distances"][0]):
                    idx = int(c_id)
                    score[idx] += 0.6 * (1.0 / (1.0 + dist))
                
            score += 0.4 * wr_norm
        else:
            score = wr_norm
            
        # Apply mask
        masked_score = np.full(len(self.m), -1.0)
        masked_score[cand] = score[cand]
        
        order = np.argsort(-masked_score)
        
        out = []
        for i in order:
            if masked_score[i] < 0:
                break
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

def evaluate(reco: VectorRecommender, k=5):
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
    _DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    csv_path = os.path.join(_DATA_DIR, "tmdb_movies.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Ensure data is downloaded first.")
        sys.exit(1)
        
    movies = load_movies(csv_path)
    reco = VectorRecommender(movies)
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
