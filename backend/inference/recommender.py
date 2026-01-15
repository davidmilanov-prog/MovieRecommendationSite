import sys
import faiss
import pickle
import pandas as pd
import random
from pathlib import Path
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    CLEANED_DATA_PATH, 
    INDEX_PATH, 
    MAPPING_PATH, 
    MODEL_PATH,
    ARCHETYPES
)

class MovieRecommender:
    def __init__(self):
        print("Initializing Recommender Engine")
        print("Loading Artifacts")
        
        try:
            # load the FAISS index
            self.index = faiss.read_index(str(INDEX_PATH))
            # load the mapping file to map vector ids to movie ids
            with open(MAPPING_PATH, "rb") as f:
                self.movie_ids = pickle.load(f)
        except Exception as e:
            print(f"CRITICAL ERROR loading FAISS: {e}")
            raise e

        # load the cleaned metadata for display info
        self.movies_df = pd.read_parquet(CLEANED_DATA_PATH)
        self.movies_df = self.movies_df.set_index('movieId')

        # initialize the sentence transformer
        self.encoder = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

        # load the trained SVD model
        with open(MODEL_PATH, "rb") as f:
            self.cf_model = pickle.load(f)
            
        self._initialized = True
        print("System Ready")

    def random_known_user_id(self) -> int:
        trainset = self.cf_model.trainset
        inner_uid = random.choice(list(trainset.all_users()))
        raw_uid = trainset.to_raw_uid(inner_uid)
        return int(raw_uid)
    
    def recommend(self, user_id: int, query_text: str, top_k: int = 10):
        persona_keyword = None
        is_archetype = False
        is_standard = (user_id == 0)
        # Reconstruct the list logic from api.py to find the matching index
        if user_id >= 1000000:
            is_archetype = True
            index = user_id - 1000000
            keys = list(ARCHETYPES.keys())
            if index < len(keys):
                label = keys[index]
                persona_keyword = ARCHETYPES[label]

        # prompt engineering
        final_query = query_text
        if persona_keyword:
            final_query = f"{query_text} {persona_keyword} style movie"

        # Generate Embeddings
        query_vector = self.encoder.encode([final_query], normalize_embeddings=True)
        
        # search the FAISS index for the nearest neighbors
        D, I = self.index.search(query_vector, 100)
        candidate_indices = I[0]
        candidate_sims = D[0]
        
        candidates = []
        for idx, sim in zip(candidate_indices, candidate_sims):
            if idx != -1:
                mid = self.movie_ids[idx]
                candidates.append((mid, float(sim)))

        # Determine if real user exists in CF model (only if not archetype)
        
        user_known = False
        if (not is_archetype) and (not is_standard):
            try:
                self.cf_model.trainset.to_inner_uid(user_id)
                user_known = True
            except ValueError:
                user_known = False

        final_results = []

        if user_known:
            # True Hybrid: re-rank by SVD predicted rating
            predictions = []
            for mid, sim in candidates:
                pred = self.cf_model.predict(uid=user_id, iid=mid)
                predictions.append((mid, pred.est, "Hybrid"))

            predictions.sort(key=lambda x: x[1], reverse=True)
            final_results = predictions[:top_k]
            score_label = "Predicted rating"

        elif is_archetype:
            # Archetype: rank by semantic similarity (not SVD)
            candidates.sort(key=lambda x: x[1], reverse=True)
            final_results = [(mid, sim, "Archetype") for mid, sim in candidates[:top_k]]
            score_label = "Semantic similarity"

        elif is_standard:
            # Standard Search: purely content-based (semantic similarity)
            candidates.sort(key=lambda x: x[1], reverse=True)
            final_results = [(mid, sim, "Semantic") for mid, sim in candidates[:top_k]]
            score_label = "Semantic similarity"

        else:
            # Cold start: rank by vote_count among retrieved candidates
            candidates_with_votes = []
            for mid, sim in candidates:
                if mid in self.movies_df.index:
                    votes = self.movies_df.loc[mid]["vote_count"]
                    candidates_with_votes.append((mid, float(votes), "Popularity"))

            candidates_with_votes.sort(key=lambda x: x[1], reverse=True)
            final_results = [(mid, votes, "Popularity") for mid, votes, _ in candidates_with_votes[:top_k]]
            score_label = "Popularity (votes)"

        # Format output
        results = []
        for mid, score, strategy in final_results:
            if mid in self.movies_df.index:
                movie_info = self.movies_df.loc[mid]
                results.append({
                    "movieId": mid,
                    "title": movie_info["title"],
                    "overview": movie_info["overview"],
                    "poster_path": movie_info["poster_path"],
                    "score": int(score) if strategy == "Popularity" else round(float(score), 4),
                    "score_label": score_label,
                    "votes": int(movie_info["vote_count"]),
                    "match_score": strategy
                })

        return results
    

if __name__ == "__main__":
    rec_engine = MovieRecommender()
    
    # Test with a user (Cold Start scenario to see votes)
    # Using a fake user ID (999999) triggers the "Popularity" logic
    try:
        recommendations = rec_engine.recommend(user_id=999999, query_text="Toy Story")
        
        print(f"{'ID':<6} | {'Rating':<6} | {'Votes':<6} | {'Title'}")
        print("-" * 60)
        for r in recommendations:
            print(f"{r['movieId']:<6} | {r['score']:<6} | {r['votes']:<6} | {r['title']}")
            
    except Exception as e:
        print(f"Error: {e}")