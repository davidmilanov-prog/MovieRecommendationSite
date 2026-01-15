## Hybrid Movie Recommendation Engine 
A movie recommendation system that combines Semantic Retrieval with Collaborative Filtering.

This repo includes:
- `backend/` FastAPI + recommendation engine
- `frontend/` React UI

### What it does
1. Semantic Retrieval (Content-Based)
- Encode each movie (overview + tags) into vectors
- Use FAISS to retrieve the most relevant movies for a text query

2. Collaborative Filtering (SVD)
- For real users, predict ratings and re-rank the retrieved movies
- For unknown users, fall back to popularity within the retrieved set

### Tech Stack

**Backend / ML**
- **Python** – Core language for data processing, modeling, and inference
- **FastAPI** – High-performance REST API for serving recommendations
- **FAISS (CPU)** – Approximate nearest-neighbor search for semantic retrieval at scale
- **Sentence Transformers (multi-qa-mpnet-base-dot-v1)** – Dense embeddings for natural-language movie search
- **scikit-surprise (SVD)** – Collaborative filtering model for user-based rating prediction
- **pandas / NumPy / PyArrow** – Data preprocessing, feature engineering, and parquet storage

**Models & Data**
- **MovieLens (ml-latest)** – User–movie ratings for collaborative filtering
- **TMDB Metadata** – Movie overviews and poster paths for semantic context and UI display
- **Hybrid Retrieval Architecture** – Semantic candidate generation + CF re-ranking

**Frontend**
- **React** – Client-side UI
- **Axios** – HTTP client for backend communication

**Infrastructure / Tooling**
- **FAISS + Torch threading controls** – Prevents BLAS / Torch thread contention
- **Parquet artifacts** – Efficient on-disk storage for cleaned datasets
- **Git + .gitignore** – Models and large data excluded; reproducible pipeline via scripts


### Modes supported

- Standard Search (id 0)
* Pure semantic search. Movies are retrieved based on how similar their embeddings are to the natural-language query. No user data is used.
- Archetypes (ids 1000000+)
* Semantic search with a light prompt bias toward a specific genre. Internally, a genre keyword is appended to the query to steer retrieval without hard filtering.
- Real User (MovieLens userId)
* Hybrid recommendation. The system retrieves semantically relevant movies, then re-ranks them using collaborative filtering based on a real MovieLens user’s rating history. This simulates personalized recommendations.
- Cold Start
* Automatic fallback for unknown users. Movies are ranked by popularity (vote count) within the semantically retrieved set.

### Data you must download
Datasets are third-party and not included in this repo. Download separately and respect their licenses.

1. MovieLens (ml-latest)
* https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
Required files (put inside `backend/data/ml-latest/`):
- movies.csv
- ratings.csv
- genome-scores.csv
- genome-tags.csv
- links.csv

2. TMDB dataset
* https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies
Required file (put inside `backend/data/`):
- TMDB_movie_dataset_v11.csv

### Folder setup (important)
The paths are hardcoded in `backend/config.py` to expect:

backend/
  data/
    ml-latest/
      movies.csv
      ratings.csv
      genome-scores.csv
      genome-tags.csv
      links.csv
    TMDB_movie_dataset_v11.csv

### Install backend dependencies
From repo root:

1. Create venv (optional but recommended)
- `python -m venv venv`
- `source venv/bin/activate` (Mac/Linux)
- `venv\Scripts\activate` (Windows)

2. Install packages
- `pip install -r requirements.txt`

### Build the models + index (one time)
This generates the cleaned datasets, FAISS index, and SVD model.

From repo root:
- `python backend/main.py`

If you want to force rebuild everything:
- `python backend/main.py --force`

This should create:
- `backend/data/cleaned_movies.parquet`
- `backend/data/cleaned_ratings.parquet`
- `backend/data/movies.index`
- `backend/data/movie_ids.pkl`
- `backend/models/svd_model.pkl`

### Run the backend API
From repo root:
- `uvicorn backend.api:app --reload --port 8000`

Test it in browser:
- `http://localhost:8000/personas`

### Run the frontend
In a second terminal:

- `cd frontend`
- `npm install`
- `npm start`

Then open:
- `http://localhost:3000`

### Common problems
- If `/personas` fails: backend is not running or CORS/origin mismatch
- If backend crashes on startup: you are missing artifacts (run `python backend/main.py`)
- If it says it cannot find CSVs: your data is not in the exact folders above
