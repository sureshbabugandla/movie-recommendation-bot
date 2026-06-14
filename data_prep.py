"""
data_prep.py — Load and clean the TMDB movie catalogue, derive mood tags,
era buckets, and a text "content soup" used by the content-based recommender.

This logic is inlined into the capstone notebook; it lives here as a script so
each piece can be unit-checked in isolation before assembly.
"""
from __future__ import annotations
import os as _os
_DATA_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data")
import re
import pandas as pd

# ----------------------------------------------------------------------------
# Mood system: map a user-expressed mood to the genres that satisfy it.
# A user saying "I feel sad" wants to be cheered up -> feel-good genres.
# This mapping is the single source of truth for mood-aware recommendation.
# ----------------------------------------------------------------------------
MOOD_TO_GENRES = {
    "feel-good":   ["Comedy", "Family", "Animation", "Romance", "Music"],
    "happy":       ["Comedy", "Family", "Animation", "Adventure", "Music"],
    "sad":         ["Comedy", "Family", "Animation", "Romance"],          # cheer-up default
    "uplifting":   ["Comedy", "Family", "Drama", "Music", "Animation"],
    "exciting":    ["Action", "Adventure", "Thriller", "Science Fiction"],
    "thrilling":   ["Thriller", "Action", "Mystery", "Crime"],
    "scary":       ["Horror", "Thriller", "Mystery"],
    "romantic":    ["Romance", "Drama"],
    "funny":       ["Comedy", "Family", "Animation"],
    "thoughtful":  ["Drama", "History", "War", "Documentary"],
    "relaxed":     ["Family", "Comedy", "Animation", "Romance"],
    "dark":        ["Crime", "Thriller", "Horror", "Drama"],
    "adventurous": ["Adventure", "Action", "Fantasy", "Science Fiction"],
    "tense":       ["Thriller", "Mystery", "Crime"],
    "mind-bending":["Science Fiction", "Mystery", "Thriller"],
    "nostalgic":   ["Family", "Animation", "Adventure", "Comedy"],
    "emotional":   ["Drama", "Romance"],
}

CANONICAL_MOODS = sorted(MOOD_TO_GENRES.keys())


def era_bucket(year: int) -> str:
    """Map a release year to a human decade label, e.g. 1994 -> '1990s'."""
    if pd.isna(year):
        return "unknown"
    decade = int(year) // 10 * 10
    return f"{decade}s"


def load_movies(path: str, min_votes: int = 30) -> pd.DataFrame:
    """Load TMDB catalogue and produce a clean, recommendation-ready frame."""
    df = pd.read_csv(path)

    # Keep only rows usable for content-based recommendation.
    df = df.dropna(subset=["original_title", "genres", "overview"]).copy()
    df = df[df["vote_count"].fillna(0) >= min_votes].copy()
    df = df.drop_duplicates(subset=["original_title", "release_year"]).copy()

    # Normalise genres: "Action|Adventure" -> ["Action", "Adventure"]
    df["genre_list"] = df["genres"].str.split("|").apply(
        lambda gs: [g.strip() for g in gs if g.strip()]
    )

    # Keywords / cast / director cleaned for the content soup.
    for col in ["keywords", "cast", "director", "tagline"]:
        df[col] = df[col].fillna("").str.replace("|", " ", regex=False)

    df["era"] = df["release_year"].apply(era_bucket)

    # Content soup: weight genres (x2) so genre dominates similarity, then
    # keywords, director, top cast, and the plot overview.
    def soup(r):
        genres = " ".join(r["genre_list"])
        return " ".join([
            genres, genres,                       # double-weight genre signal
            str(r["keywords"]),
            str(r["director"]),
            str(r["cast"]),
            str(r["overview"]),
        ]).lower()

    df["content"] = df.apply(soup, axis=1)

    # Per-movie mood set: a movie "supports" a mood if it shares a genre with it.
    genre_to_moods = {}
    for mood, genres in MOOD_TO_GENRES.items():
        for g in genres:
            genre_to_moods.setdefault(g, set()).add(mood)

    def movie_moods(genre_list):
        out = set()
        for g in genre_list:
            out |= genre_to_moods.get(g, set())
        return sorted(out)

    df["moods"] = df["genre_list"].apply(movie_moods)

    df = df.reset_index(drop=True)
    cols = ["original_title", "genre_list", "genres", "overview", "keywords",
            "director", "cast", "release_year", "era", "vote_average",
            "vote_count", "popularity", "moods", "content"]
    return df[cols]


if __name__ == "__main__":
    m = load_movies(_os.path.join(_DATA_DIR, "tmdb_movies.csv"))
    print("Clean catalogue shape:", m.shape)
    print("Year range:", int(m.release_year.min()), "-", int(m.release_year.max()))
    print("\nGenre frequency (top 12):")
    from collections import Counter
    c = Counter(g for gl in m.genre_list for g in gl)
    for g, n in c.most_common(12):
        print(f"  {g:18} {n}")
    print("\nEra distribution:")
    print(m.era.value_counts().sort_index().to_string())
    print("\nSample rows:")
    print(m[["original_title", "release_year", "genres", "vote_average"]].head(5).to_string())
    print("\nExample mood tags for first 3 movies:")
    for _, r in m.head(3).iterrows():
        print(f"  {r.original_title:28} -> {r.moods}")
