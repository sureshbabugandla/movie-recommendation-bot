"""
entities.py — Rule-based slot extraction (mood, genre, era, min_rating) from a
user utterance. A gazetteer + regex approach is deliberate: it gives tight
precision on the curated movie vocabulary without needing a large annotated NER
corpus, and it is fully transparent/explainable for the report.
"""
from __future__ import annotations
import os as _os
_DATA_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data")
import re
from typing import Dict, List

# Canonical moods understood by the recommender (mirrors data_prep.MOOD_TO_GENRES).
MOOD_LEXICON = {
    "feel-good": ["feel good", "feel-good", "feelgood", "cheer me up", "cheerful"],
    "happy": ["happy", "joyful", "good mood"],
    "sad": ["sad", "down", "blue", "depressed", "low", "unhappy"],
    "uplifting": ["uplifting", "inspiring", "inspirational", "hopeful"],
    "exciting": ["exciting", "exhilarating", "high energy", "high-energy"],
    "thrilling": ["thrilling", "thrill", "edge of my seat", "edge-of-seat"],
    "scary": ["scary", "spooky", "frightening", "creepy", "horror"],
    "romantic": ["romantic", "romance", "lovey", "date night"],
    "funny": ["funny", "hilarious", "laugh", "comedic", "lighthearted", "light-hearted"],
    "thoughtful": ["thoughtful", "deep", "meaningful", "profound", "serious"],
    "relaxed": ["relaxed", "relaxing", "chill", "cozy", "easy watch", "laid back"],
    "dark": ["dark", "gritty", "grim", "bleak", "disturbing"],
    "adventurous": ["adventurous", "adventure", "epic"],
    "tense": ["tense", "suspenseful", "suspense", "nail biting", "nail-biting"],
    "nostalgic": ["nostalgic", "nostalgia", "throwback"],
    "emotional": ["emotional", "moving", "tearjerker", "tear-jerker", "heartfelt"],
    "intense": ["intense", "gripping"],
}

# Genre surface forms -> canonical TMDB genre.
GENRE_LEXICON = {
    "Action": ["action"],
    "Adventure": ["adventure"],
    "Animation": ["animation", "animated", "cartoon", "anime"],
    "Comedy": ["comedy", "comedies", "comedic", "funny movie"],
    "Crime": ["crime", "gangster", "heist", "mafia"],
    "Documentary": ["documentary", "documentaries", "docu"],
    "Drama": ["drama", "dramatic"],
    "Family": ["family", "kids", "children", "kid friendly", "kid-friendly"],
    "Fantasy": ["fantasy", "magical"],
    "History": ["history", "historical"],
    "Horror": ["horror", "slasher"],
    "Music": ["music", "musical", "musicals"],
    "Mystery": ["mystery", "whodunit", "detective"],
    "Romance": ["romance", "romantic", "rom com", "rom-com", "romcom", "love story"],
    "Science Fiction": ["sci-fi", "scifi", "sci fi", "science fiction", "space"],
    "Thriller": ["thriller", "thrillers", "suspense thriller"],
    "War": ["war", "wartime", "military"],
    "Western": ["western", "westerns", "cowboy"],
}

# Map decade phrases to a (start, end) year window.
_DECADE_WORDS = {
    "sixties": 1960, "seventies": 1970, "eighties": 1980, "nineties": 1990,
}


def _extract_eras(text: str) -> List[str]:
    eras = set()
    # "90s", "1990s", "'90s"
    for m in re.finditer(r"\b(?:19|20)?(\d0)s\b", text):
        two = m.group(1)
        # heuristic: 00,10,20 -> 2000s..., 30-90 -> 1930s...
        decade = int(two)
        if decade <= 20:
            year = 2000 + decade
        else:
            year = 1900 + decade
        eras.add(f"{year}s")
    # explicit 4-digit decade like "1980s" already caught above via prefix; also bare years
    for word, year in _DECADE_WORDS.items():
        if word in text:
            eras.add(f"{year}s")
    if re.search(r"\bclassic|old\b", text):
        eras.add("classic")
    if re.search(r"\brecent|new(er)?|modern|latest\b", text):
        eras.add("recent")
    return sorted(eras)


def _phrase_hit(text: str, phrase: str) -> bool:
    return re.search(r"\b" + re.escape(phrase) + r"\b", text) is not None


def extract_entities(text: str) -> Dict[str, List]:
    t = " " + text.lower().strip() + " "

    moods = [m for m, forms in MOOD_LEXICON.items()
             if any(_phrase_hit(t, f) for f in forms)]

    # Longest-match-wins for genres so "science fiction" beats "fiction".
    genres = []
    for g, forms in GENRE_LEXICON.items():
        if any(_phrase_hit(t, f) for f in sorted(forms, key=len, reverse=True)):
            genres.append(g)

    eras = _extract_eras(t)

    # Minimum rating, e.g. "at least 7", "rated above 8", "high rated".
    min_rating = None
    m = re.search(r"(?:at least|above|over|rated?|minimum|min)\s*(\d(?:\.\d)?)", t)
    if m:
        try:
            v = float(m.group(1))
            if 0 < v <= 10:
                min_rating = v
        except ValueError:
            pass
    if re.search(r"\b(highly rated|top rated|critically acclaimed|best)\b", t):
        min_rating = max(min_rating or 0, 7.0)

    return {"mood": moods, "genre": genres, "era": eras, "min_rating": min_rating}


# Lightweight self-test (acts as the notebook's entity unit-test cell).
TEST_CASES = [
    ("I'm feeling sad, suggest something feel-good from the 90s",
     {"mood_has": "sad", "genre_empty": True, "era_has": "1990s"}),
    ("recommend a scary horror movie", {"mood_has": "scary", "genre_has": "Horror"}),
    ("a science fiction film from the eighties",
     {"genre_has": "Science Fiction", "era_has": "1980s"}),
    ("something funny and romantic, highly rated",
     {"mood_has": "funny", "genre_has": "Romance", "min_rating": 7.0}),
    ("a recent thriller rated above 8",
     {"genre_has": "Thriller", "era_has": "recent", "min_rating": 8.0}),
]


def run_tests():
    passed = 0
    for text, exp in TEST_CASES:
        e = extract_entities(text)
        ok = True
        if "mood_has" in exp:    ok &= exp["mood_has"] in e["mood"]
        if "genre_has" in exp:   ok &= exp["genre_has"] in e["genre"]
        if "genre_empty" in exp: ok &= (len(e["genre"]) == 0) == exp["genre_empty"]
        if "era_has" in exp:     ok &= exp["era_has"] in e["era"]
        if "min_rating" in exp:  ok &= e["min_rating"] == exp["min_rating"]
        print(f"[{'PASS' if ok else 'FAIL'}] {text}\n        -> {e}")
        passed += ok
    print(f"\n{passed}/{len(TEST_CASES)} entity tests passed")
    return passed == len(TEST_CASES)


if __name__ == "__main__":
    run_tests()
