"""
dialogue.py — Dialogue manager: turns a raw user message into a CineBot reply.

It composes the three NLP/recommender pieces and keeps per-session state so the
conversation is coherent across turns:
  intent (DL model)  +  entities (rules)  ->  slot memory  ->  recommender
  -> templated natural-language response.

The 'refine' intent reuses remembered slots and excludes already-shown titles;
'more_info' answers about the last shown movie; 'feedback'/'greet'/'goodbye'/
'oos' are handled conversationally.
"""
from __future__ import annotations
import os as _os
_DATA_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data")
from dataclasses import dataclass, field
from typing import Callable, Dict, List


@dataclass
class Session:
    slots: Dict = field(default_factory=lambda: {"mood": [], "genre": [],
                                                 "era": [], "min_rating": None})
    shown: List[str] = field(default_factory=list)   # titles already recommended
    last_results: List[Dict] = field(default_factory=list)


class DialogueManager:
    def __init__(self, intent_fn: Callable[[str], str], entity_fn, recommender):
        self.intent_fn = intent_fn        # text -> intent label
        self.entity_fn = entity_fn        # text -> slot dict
        self.reco = recommender

    def _merge_slots(self, sess: Session, ent: Dict, reset=False):
        if reset:
            sess.slots = {"mood": [], "genre": [], "era": [], "min_rating": None}
        for key in ["mood", "genre", "era"]:
            if ent.get(key):
                sess.slots[key] = ent[key]          # newest mention wins
        if ent.get("min_rating") is not None:
            sess.slots["min_rating"] = ent["min_rating"]

    def _format(self, results):
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"  {i}. {r['title']} ({r['year']}) — {r['why']}")
        return "\n".join(lines)

    def reply(self, text: str, sess: Session) -> str:
        intent = self.intent_fn(text)
        ent = self.entity_fn(text)

        if intent == "greet":
            return ("Hi! I'm CineBot 🎬 — tell me a mood or genre "
                    "(e.g. \"a feel-good comedy from the 90s\") and I'll suggest films.")

        if intent == "goodbye":
            return "Enjoy the movie! 👋 Come back anytime for more picks."

        if intent == "oos":
            return ("I'm a movie-recommendation bot, so I can't help with that — "
                    "but tell me a mood or genre and I'll find you something to watch.")

        if intent == "feedback":
            low = text.lower()
            if any(w in low for w in ["didn't", "not", "hate", "boring", "seen",
                                      "already", "meh", "terrible"]):
                # treat as implicit refine
                results = self.reco.recommend(slots=sess.slots, exclude=sess.shown, k=5)
                sess.shown += [r["title"] for r in results]
                sess.last_results = results
                return "No problem — here are different options:\n" + self._format(results)
            return "Glad you liked it! Want more in the same vein? Just say \"more\"."

        if intent == "more_info":
            if not sess.last_results:
                return "Tell me what you're in the mood for first, then I can give details."
            r = sess.last_results[0]
            return (f"{r['title']} ({r['year']}) — genres: {', '.join(r['genres'])}; "
                    f"rated {r['rating']}/10. Want something similar or different?")

        if intent == "refine":
            self._merge_slots(sess, ent, reset=False)
            results = self.reco.recommend(slots=sess.slots, exclude=sess.shown, k=5)
            sess.shown += [r["title"] for r in results]
            sess.last_results = results
            if not results:
                return "I've shown the best matches I have for that — try a different mood or genre?"
            return "Here are some more:\n" + self._format(results)

        # default: recommend
        self._merge_slots(sess, ent, reset=True)
        results = self.reco.recommend(slots=sess.slots, exclude=sess.shown, k=5)
        sess.shown += [r["title"] for r in results]
        sess.last_results = results
        desc = []
        if sess.slots["mood"]:  desc.append("/".join(sess.slots["mood"]))
        if sess.slots["genre"]: desc.append("/".join(sess.slots["genre"]))
        if sess.slots["era"]:   desc.append("/".join(sess.slots["era"]))
        tag = (" for a " + ", ".join(desc)) if desc else ""
        return f"Here are 5 picks{tag}:\n" + self._format(results)


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    from data_prep import load_movies
    from recommender import MovieRecommender
    from entities import extract_entities

    movies = load_movies(_os.path.join(_DATA_DIR, "tmdb_movies.csv"))
    reco = MovieRecommender(movies)

    # Lightweight keyword intent stub so this script runs without TF;
    # the notebook plugs in the trained Keras model instead.
    import re
    def stub_intent(t):
        tl = t.lower()
        def has(words): return any(re.search(r"\b"+w+r"\b", tl) for w in words)
        if has(["bye", "goodbye", "see you", "that's all", "thats all"]): return "goodbye"
        if has(["hi", "hello", "hey", "morning", "evening", "howdy"]): return "greet"
        if has(["plot", "about", "director", "cast", "year", "rating", "long", "synopsis"]): return "more_info"
        if has(["more", "else", "different", "another", "similar", "instead"]): return "refine"
        if has(["loved", "liked", "great", "hate", "boring", "seen"]): return "feedback"
        if has(["weather", "restaurant", "flight", "recipe", "wifi"]): return "oos"
        return "recommend"

    dm = DialogueManager(stub_intent, extract_entities, reco)
    sess = Session()
    convo = [
        "hey there",
        "I'm feeling down, something feel-good from the 90s",
        "show me more",
        "tell me about it",
        "actually I want a scary horror movie instead",
        "thanks, that's all",
    ]
    for u in convo:
        print(f"\nUSER: {u}\nBOT : {dm.reply(u, sess)}")
