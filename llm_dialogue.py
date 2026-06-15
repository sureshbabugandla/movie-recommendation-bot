from __future__ import annotations
import os
from pydantic import BaseModel, Field
from typing import List, Optional
from openai import OpenAI

class ExtractedIntent(BaseModel):
    intent: str = Field(description="One of: 'greet', 'recommend', 'refine', 'more_info', 'feedback', 'goodbye', 'oos'")
    mood: Optional[List[str]] = Field(default=[], description="List of moods like 'feel-good', 'scary', 'romantic'")
    genre: Optional[List[str]] = Field(default=[], description="List of genres like 'Action', 'Comedy', 'Horror', 'Sci-Fi'")
    era: Optional[List[str]] = Field(default=[], description="List of era constraints like '1990s', 'recent', 'classic'")
    min_rating: Optional[float] = Field(default=None, description="Minimum rating mentioned (0-10)")

class Session:
    def __init__(self):
        self.slots = {"mood": [], "genre": [], "era": [], "min_rating": None}
        self.shown = []
        self.last_results = []
        self.messages = [] # For conversation history

class LLMDialogueManager:
    def __init__(self, recommender, api_key=None):
        self.reco = recommender
        # Use provided key or fallback to OPENAI_API_KEY from environment
        key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=key) if key else None
        
    def _merge_slots(self, sess: Session, ent: ExtractedIntent, reset=False):
        if reset:
            sess.slots = {"mood": [], "genre": [], "era": [], "min_rating": None}
        
        if ent.mood: sess.slots["mood"] = ent.mood
        if ent.genre: sess.slots["genre"] = ent.genre
        if ent.era: sess.slots["era"] = ent.era
        if ent.min_rating is not None: sess.slots["min_rating"] = ent.min_rating

    def _format(self, results):
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"  {i}. {r['title']} ({r['year']}) — {r['why']}")
        return "\n".join(lines)

    def extract_intent(self, text: str, sess: Session) -> ExtractedIntent:
        if not self.client:
            raise ValueError("OpenAI API Key is not set. Please set the OPENAI_API_KEY environment variable.")
            
        history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in sess.messages[-4:]])
        
        system_prompt = """You are CineBot, an intelligent movie recommendation assistant.
Extract the user's intent and any movie constraints from their message.
If the user specifies a constraint, ensure it matches the provided schema fields.
Intents:
- greet: User says hello or introduces themselves
- recommend: User wants a new recommendation
- refine: User wants different recommendations or adjusts constraints (e.g. "show me older ones", "not horror")
- more_info: User wants details about a specific movie recently recommended
- feedback: User gives feedback on recommendations (e.g. "I loved it", "I've seen it")
- goodbye: User says bye
- oos: Out of scope request (e.g. asking about weather, booking flights)
"""
        response = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"History:\n{history}\n\nUser: {text}"}
            ],
            response_format=ExtractedIntent,
        )
        return response.choices[0].message.parsed

    def reply(self, text: str, sess: Session) -> str:
        sess.messages.append({"role": "user", "content": text})
        
        try:
            parsed = self.extract_intent(text, sess)
        except Exception as e:
            err = f"Error connecting to LLM API: {e}"
            sess.messages.append({"role": "assistant", "content": err})
            return err
            
        intent = parsed.intent
        
        if intent == "greet":
            bot_reply = "Hi! I'm CineBot 🎬 — tell me a mood or genre (e.g. \"a feel-good comedy from the 90s\") and I'll suggest films."
        
        elif intent == "goodbye":
            bot_reply = "Enjoy the movie! 👋 Come back anytime for more picks."
            
        elif intent == "oos":
            bot_reply = "I'm a movie-recommendation bot, so I can't help with that — but tell me a mood or genre and I'll find you something to watch."
            
        elif intent == "feedback":
            if any(w in text.lower() for w in ["didn't", "not", "hate", "boring", "seen"]):
                results = self.reco.recommend(slots=sess.slots, exclude=sess.shown, k=5)
                sess.shown += [r["title"] for r in results]
                sess.last_results = results
                if not results:
                    bot_reply = "I've shown all the best matches I have for that."
                else:
                    bot_reply = "No problem — here are different options:\n" + self._format(results)
            else:
                bot_reply = "Glad you liked it! Want more in the same vein? Just say \"more\"."
                
        elif intent == "more_info":
            if not sess.last_results:
                bot_reply = "Tell me what you're in the mood for first, then I can give details."
            else:
                r = sess.last_results[0]
                bot_reply = f"{r['title']} ({r['year']}) — genres: {', '.join(r['genres'])}; rated {r['rating']}/10. Want something similar or different?"
                
        elif intent == "refine":
            self._merge_slots(sess, parsed, reset=False)
            results = self.reco.recommend(slots=sess.slots, exclude=sess.shown, k=5)
            sess.shown += [r["title"] for r in results]
            sess.last_results = results
            if not results:
                bot_reply = "I've shown the best matches I have for that — try a different mood or genre?"
            else:
                bot_reply = "Here are some more:\n" + self._format(results)
                
        else: # recommend
            self._merge_slots(sess, parsed, reset=True)
            results = self.reco.recommend(slots=sess.slots, exclude=sess.shown, query_text=text, k=5)
            sess.shown += [r["title"] for r in results]
            sess.last_results = results
            
            desc = []
            if sess.slots["mood"]:  desc.append("/".join(sess.slots["mood"]))
            if sess.slots["genre"]: desc.append("/".join(sess.slots["genre"]))
            if sess.slots["era"]:   desc.append("/".join(sess.slots["era"]))
            tag = (" for a " + ", ".join(desc)) if desc else ""
            bot_reply = f"Here are 5 picks{tag}:\n" + self._format(results)
            
        sess.messages.append({"role": "assistant", "content": bot_reply})
        return bot_reply

if __name__ == "__main__":
    import sys
    from vector_recommender import VectorRecommender
    from data_prep import load_movies
    
    # Load dataset
    _DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    csv_path = os.path.join(_DATA_DIR, "tmdb_movies.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Make sure data is present.")
        sys.exit(1)
        
    print("Loading recommender (and Chroma DB if not already initialized)...")
    movies = load_movies(csv_path)
    reco = VectorRecommender(movies)
    
    print("\nStarting Chatbot (ensure OPENAI_API_KEY is set in your environment)...")
    dm = LLMDialogueManager(reco)
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
