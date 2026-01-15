## Backend Recommendation Algorithm

### Summary
This folder contains all backend logic for the recommendation system. It is a **hybrid recommender**:
1. Semantic Search (Sentence Transformers + FAISS) to retrieve candidates
2. Collaborative Filtering (SVD) to re-rank for real users

### What the API exposes
The FastAPI server provides:
- `/personas` (dropdown options)
- `/random_user` (returns a valid userId from the CF trainset)
- `/recommend` (main recommendation endpoint)

### How it works (Pipeline)
These are the artifacts the backend depends on:

**1. Data Preprocessing (`data/preprocess_data.py`)**
- Loads MovieLens movies + ratings
- Filters to movies with enough votes and users with enough ratings
- Adds genome tags and TMDB overview/poster
- Writes:
  - `backend/data/cleaned_movies.parquet`
  - `backend/data/cleaned_ratings.parquet`

**2. Vector Indexing (`data/build_index.py`)**
- Embeds each movie “soup” using `multi-qa-mpnet-base-dot-v1`
- Builds a FAISS index for fast similarity search
- Writes:
  - `backend/data/movies.index`
  - `backend/data/movie_ids.pkl`

**3. Model Training (`models/train_cf.py`)**
- Trains an SVD model on the cleaned ratings
- Writes:
  - `backend/models/svd_model.pkl`

**4. Inference Engine (`inference/recommender.py`)**
When you call `/recommend`:
- Encode the query text
- Retrieve top ~100 from FAISS
- Decide which strategy to use:
  - `user_id == 0` -> semantic only
  - `user_id >= 1000000` -> archetype prompt bias + semantic only
  - real known user -> semantic retrieval + SVD re-rank (hybrid)
  - unknown user -> popularity fallback within retrieved set
