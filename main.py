import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

import sys
sys.path.append(os.path.dirname(__file__))

from data_prep import load_movies
from vector_recommender import VectorRecommender
from llm_dialogue import LLMDialogueManager, Session

# Global state
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup on startup
    _DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    csv_path = os.path.join(_DATA_DIR, "tmdb_movies.csv")
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Ensure data is present.")
        yield
        return
        
    print("Loading movies and VectorRecommender...")
    movies = load_movies(csv_path)
    reco = VectorRecommender(movies)
    dm = LLMDialogueManager(reco)
    
    app_state["reco"] = reco
    app_state["dm"] = dm
    app_state["sessions"] = {} # session_id -> Session
    
    yield
    # Cleanup on shutdown
    pass

app = FastAPI(title="CineBot API", lifespan=lifespan)

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str
    session_id: str

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    if "dm" not in app_state:
        raise HTTPException(status_code=500, detail="Dialogue Manager not initialized. Missing API key or Data.")
        
    dm: LLMDialogueManager = app_state["dm"]
    sessions = app_state["sessions"]
    
    if req.session_id not in sessions:
        sessions[req.session_id] = Session()
        
    sess = sessions[req.session_id]
    reply_text = dm.reply(req.message, sess)
    
    return ChatResponse(reply=reply_text, session_id=req.session_id)

@app.get("/health")
def health_check():
    return {"status": "ok", "loaded": "dm" in app_state}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
