import sys
import faiss
import pickle
import pandas as pd
import threading
from pathlib import Path
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    CLEANED_DATA_PATH, 
    INDEX_PATH, 
    MAPPING_PATH, 
    MODEL_PATH
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

    def recommend(self, user_id: int, query_text: str, top_k: int = 10):
        # generate embedding for the query text
        query_vector = self.encoder.encode([query_text], normalize_embeddings=True)
        # search the FAISS index for the nearest neighbors
        D, I = self.index.search(query_vector, 100)
        candidate_indices = I[0]
        # retrieve valid movie IDs from the mapping
        candidate_movie_ids = [self.movie_ids[i] for i in candidate_indices if i != -1]

        # Check User Existence
        user_known = False
        try:
            # check if the user exists in the cf model trainset
            self.cf_model.trainset.to_inner_uid(user_id)
            user_known = True
        except ValueError:
            user_known = False

        # Ranking Strategy
        final_results = []
        
        if user_known:
            # Hybrid (Semantic + SVD)
            predictions = []
            for mid in candidate_movie_ids:
                # predict the rating the user would give this movie
                pred = self.cf_model.predict(uid=user_id, iid=mid)
                predictions.append((mid, pred.est))
            
            # sort by predicted rating
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            for mid, score in predictions[:top_k]:
                final_results.append((mid, score, "Hybrid"))
                
        else:
            # Cold start with content only
            # Sort by vote_count for new users so they get popular relevant movies instead of obscure ones.
            candidates_with_votes = []
            for mid in candidate_movie_ids:
                if mid in self.movies_df.index:
                    # get the vote count for popularity ranking
                    votes = self.movies_df.loc[mid]['vote_count']
                    candidates_with_votes.append((mid, votes))
            
            # Sort by votes (Descending)
            candidates_with_votes.sort(key=lambda x: x[1], reverse=True)
            
            for mid, votes in candidates_with_votes[:top_k]:
                final_results.append((mid, 0.0, "Content"))

        # formatting output 
        results = []
        for mid, pred_rating, strategy in final_results:
            if mid in self.movies_df.index:
                # get movie details from dataframe
                movie_info = self.movies_df.loc[mid]
                results.append({
                    "movieId": mid,
                    "title": movie_info['title'],
                    "overview": movie_info['overview'],
                    "predicted_rating": round(pred_rating, 2),
                    "votes": int(movie_info['vote_count']),
                    "match_score": strategy
                })
        
        return results

if __name__ == "__main__":
    rec_engine = MovieRecommender()
    
    # Test with a user (Cold Start scenario to see votes)
    # Using a fake user ID (999999) triggers the "Content (Popularity)" logic
    try:
        recommendations = rec_engine.recommend(user_id=999999, query_text="Toy Story")
        
        print(f"{'ID':<6} | {'Rating':<6} | {'Votes':<6} | {'Title'}")
        print("-" * 60)
        for r in recommendations:
            print(f"{r['movieId']:<6} | {r['predicted_rating']:<6} | {r['votes']:<6} | {r['title']}")
            
    except Exception as e:
        print(f"Error: {e}")