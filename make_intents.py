"""
make_intents.py — Generate a balanced, varied intent-classification dataset for
the CineBot chatbot. Seven intents drive the dialogue manager:

  greet | recommend | refine | more_info | feedback | goodbye | oos

We expand hand-written templates with domain slot-fillers (moods, genres, eras)
to get natural variety. This is standard text data augmentation and produces a
dataset large enough for a neural model to learn from (the prior capstone failed
with only ~210 examples; we target ~2k balanced examples).
"""
from __future__ import annotations
import os as _os
_DATA_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data")
import itertools
import json
import random
from pathlib import Path

random.seed(42)

MOODS = ["happy", "sad", "feel-good", "exciting", "scary", "romantic", "funny",
         "thoughtful", "relaxed", "dark", "adventurous", "tense", "nostalgic",
         "uplifting", "emotional", "chill", "intense"]
GENRES = ["action", "comedy", "drama", "horror", "thriller", "romance", "sci-fi",
          "science fiction", "fantasy", "animation", "adventure", "crime",
          "mystery", "family", "documentary", "war", "musical"]
ERAS = ["80s", "90s", "2000s", "70s", "from the 1990s", "from the eighties",
        "classic", "recent", "new", "old", "modern"]

# ---- GREET ----------------------------------------------------------------
GREET = [
    "hi", "hello", "hey", "hey there", "hi there", "hello there", "yo",
    "good morning", "good evening", "good afternoon", "greetings",
    "hey cinebot", "hello cinebot", "hi cinebot", "hey bot", "howdy",
    "hi, can you help me", "hello, i need a movie", "hey, you there",
    "hiya", "what's up", "morning", "evening", "hey can you help me find a film",
    "hello, are you the movie bot", "hi, looking for something to watch",
]

# ---- GOODBYE --------------------------------------------------------------
GOODBYE = [
    "bye", "goodbye", "see you", "see you later", "see ya", "cya", "later",
    "thanks bye", "thanks, that's all", "that's all for now", "i'm done",
    "that's enough", "ok bye", "alright bye", "thank you goodbye",
    "ok that's enough for today", "i have to go", "gotta go", "talk later",
    "thanks for the help, bye", "no more, thanks", "i'm good, thanks bye",
    "that will be all", "ok i'm leaving now", "catch you later", "peace out",
    "alright that's it", "nothing else, bye",
]

# ---- MORE_INFO ------------------------------------------------------------
MORE_INFO = [
    "tell me more about it", "tell me about that movie", "what's the plot",
    "what is it about", "what's it about", "give me the synopsis",
    "synopsis please", "who directed it", "who's the director",
    "who is in it", "who stars in it", "what's the cast",
    "what year is it from", "when did it come out", "what year was it released",
    "is it scary", "is it funny", "is it any good", "what's the rating",
    "how is it rated", "how long is it", "what's the runtime",
    "what genre is it", "tell me more", "more details please", "more info",
    "describe it", "what's the story", "can you summarise it",
    "is it appropriate for kids", "what's it rated", "give me more on that one",
]

# ---- FEEDBACK -------------------------------------------------------------
FEEDBACK = [
    "i loved it", "i love that one", "that was great", "great pick",
    "amazing choice", "perfect", "perfect, thanks", "nice one", "good one",
    "i liked that", "that was good", "loved it thanks", "awesome suggestion",
    "that's a great movie", "i enjoyed that", "brilliant choice",
    "i didn't like that", "not my taste", "didn't enjoy it", "that was boring",
    "i hated it", "not a fan of that", "not really my thing", "meh",
    "i've already seen it", "already watched that", "seen it already",
    "i saw that one already", "i've seen all of those", "that was terrible",
    "not interested in that", "that one was excellent", "really good thanks",
    "wonderful, i liked it", "no i didn't like it",
]

# ---- OUT OF SCOPE (oos) ---------------------------------------------------
OOS = [
    "what's the weather today", "what's the weather like", "will it rain tomorrow",
    "what's the capital of france", "what is 2 plus 2", "what time is it",
    "tell me a joke", "book a flight to paris", "order me a pizza",
    "what's my schedule today", "set an alarm for 7am", "play some music",
    "recommend a restaurant", "suggest a good restaurant", "find me a hotel",
    "what's the stock price of apple", "who won the football match",
    "how do i cook pasta", "translate hello to spanish", "recommend a book",
    "recommend a song", "what's the news today", "send an email to john",
    "how tall is mount everest", "what's the meaning of life",
    "give me directions to the airport", "convert 10 dollars to euros",
    "what's a good laptop to buy", "tell me about quantum physics",
    "who is the president", "what's the square root of 144",
    "ignore previous instructions", "write me some code",
    "recommend a tv show to learn cooking", "what's the best phone",
]

# ---- RECOMMEND (templated) ------------------------------------------------
RECOMMEND_TEMPLATES = [
    "recommend a {mood} movie", "suggest a {mood} film", "i want a {mood} movie",
    "i'm feeling {mood}, suggest a movie", "i'm in a {mood} mood",
    "something {mood} please", "recommend something {mood}",
    "suggest a {genre} movie", "i want to watch a {genre} film",
    "recommend a good {genre} movie", "show me some {genre} movies",
    "any good {genre} films", "i feel like watching {genre}",
    "recommend a {mood} {genre} movie", "a {genre} movie {era}",
    "suggest a {genre} film {era}", "something {mood} {era}",
    "i'm feeling {mood}, something {genre} {era}",
    "what should i watch tonight", "recommend me a movie",
    "suggest something to watch", "i need a movie recommendation",
    "what's a good movie to watch", "pick a movie for me",
    "give me a movie suggestion", "find me a {genre} movie {era}",
    "i'm feeling {mood}, recommend something feel-good",
    "can you recommend a {mood} film {era}", "looking for a {genre} movie",
    "i want something {mood} to watch", "a good {mood} film for tonight",
]

# ---- REFINE (templated) ---------------------------------------------------
REFINE_TEMPLATES = [
    "something else", "show me more", "more options", "anything else",
    "give me another", "another one", "show me something different",
    "not that one", "something different", "i've seen those",
    "show me more like that", "more like that one", "similar movies",
    "something similar", "in a similar style", "more suggestions",
    "something more {mood}", "make it more {mood}", "something {mood}er",
    "more {genre} instead", "switch to {genre}", "what about {genre} instead",
    "something {era} instead", "anything {era}", "newer ones",
    "older ones", "something newer", "something older",
    "not horror, something else", "less scary", "more upbeat",
    "show me the next ones", "different genre please", "change the genre",
    "a bit more {mood}", "try {genre} instead",
    "show me {genre} ones instead", "anything more {mood} than that",
    "got anything {era}", "what else is {genre}", "something a bit more {mood}",
    "pick a different {genre} one", "not {genre}, try something else",
    "give me more {mood} options", "any other {genre} films {era}",
    "more {mood} {genre} ones", "show the next {genre} picks",
    "something else that's {mood}", "different {mood} options please",
    "swap it for something {mood}", "i'd prefer something {era}",
    "rather watch something {genre}", "can you change it to {genre}",
    "less {mood}, more {mood}", "another {genre} suggestion",
]


def fill(template: str) -> str:
    return template.format(
        mood=random.choice(MOODS),
        genre=random.choice(GENRES),
        era=random.choice(ERAS),
    )


def expand(templates, n_target):
    """Expand templates by repeated slot-filling until we hit n_target uniques."""
    out, tries = set(), 0
    static = [t for t in templates if "{" not in t]
    out.update(static)
    while len(out) < n_target and tries < n_target * 40:
        out.add(fill(random.choice(templates)))
        tries += 1
    return list(out)


PREFIXES = ["", "um ", "so ", "ok ", "hey ", "well ", "please "]
SUFFIXES = ["", "?", "!", ".", " please", " now", " thanks"]


def pad_pool(pool, per_class):
    """Reach per_class unique items via prefix/suffix product over base items."""
    items = list(dict.fromkeys(pool))
    seen = set(items)
    combos = [(p, s) for p in PREFIXES for s in SUFFIXES if (p, s) != ("", "")]
    random.shuffle(combos)
    for base in itertools.cycle(pool):
        if len(items) >= per_class:
            break
        for p, s in combos:
            v = f"{p}{base}{s}"
            if v not in seen:
                seen.add(v); items.append(v)
                break
    return items[:per_class]


def build(per_class: int = 280):
    data = []
    fixed = {"greet": GREET, "goodbye": GOODBYE, "more_info": MORE_INFO,
             "feedback": FEEDBACK, "oos": OOS}
    for label, pool in fixed.items():
        items = pad_pool(pool, per_class)
        for t in items:
            data.append({"text": t, "label": label})

    for label, templates in [("recommend", RECOMMEND_TEMPLATES),
                             ("refine", REFINE_TEMPLATES)]:
        for t in expand(templates, per_class)[:per_class]:
            data.append({"text": t, "label": label})

    random.shuffle(data)
    return data


if __name__ == "__main__":
    rows = build(per_class=280)
    out = Path(_os.path.join(_DATA_DIR, "intents.jsonl"))
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    c = Counter(r["label"] for r in rows)
    print(f"Wrote {len(rows)} examples to {out}")
    for k in sorted(c):
        print(f"  {k:12} {c[k]}")
    print("\nSamples:")
    for r in rows[:12]:
        print(f"  [{r['label']:10}] {r['text']}")
