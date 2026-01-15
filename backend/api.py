import os
import sys
from pathlib import Path

# Prevent thread conflicts between FAISS / torch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent))

from config import ARCHETYPES
from inference.recommender import MovieRecommender

app = FastAPI()

# Let the React app call the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# create one global recommender and keep it in memory
rec_engine = None


# Request body for /recommend
class SearchRequest(BaseModel):
    user_id: int
    query: str
    top_k: Optional[int] = 10


@app.on_event("startup")
def startup_event():
    # This loads FAISS index, movie metadata, and the SVD model
    global rec_engine
    print("Initializing Engine...")
    rec_engine = MovieRecommender()


@app.get("/personas")
def get_personas():
    # This provides dropdown options for the frontend
    persona_list = []

    # Standard Search mode (semantic only)
    persona_list.append({
        "id": 0,
        "label": "Standard Search",
        "description": "Raw Semantic Search."
    })

    # Add archetypes
    for i, (label, keyword) in enumerate(ARCHETYPES.items()):
        persona_list.append({
            "id": 1000000 + i,
            "label": label,
            "description": f" Biased towards {keyword} movies."
        })

    return persona_list


@app.get("/random_user")
def get_random_user():
    # Returns a valid user_id that exists in the CF model trainset
    if not rec_engine:
        raise HTTPException(status_code=503, detail="Engine loading...")
    return {"user_id": rec_engine.random_known_user_id()}


@app.post("/recommend")
def recommend(req: SearchRequest):
    # Main endpoint used by the frontend search button
    # Logic lives in MovieRecommender.recommend()
    if not rec_engine:
        raise HTTPException(status_code=503, detail="Engine loading...")

    results = rec_engine.recommend(req.user_id, req.query, req.top_k)
    return results
