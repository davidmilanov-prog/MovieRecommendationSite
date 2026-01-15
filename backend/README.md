## Backend Recommendation Algorithm

### Summary
This folder contains all logic for the recommendation system. We use a **hybrid approach**, combining Semantic Search (Content-Based) and Collaborative Filtering (SVD) to recommend movies.

### How it works (The Pipeline)
The `main.py` file runs the following 5 steps in order to build the engine:

**1. Data Preprocessing (`preprocess_data.py`)**
We merge two datasets: **MovieLens** (which contains user ratings and tags) and **TMDB** (which contains movie posters and plot summaries). We filter out movies with low vote counts to ensure quality and clean up the text descriptions.

**2. Vector Indexing (`build_index.py`)**
We use a Sentence Transformer (`multi-qa-mpnet-base-dot-v1`) to convert every movie's plot summary and genome tags into a vector. We then store these in a **FAISS Index**, which allows us to search through movies in quickly to find semantically similar ones.

**3. Model Training (`train_cf.py`)**
We train a **Singular Value Decomposition (SVD)** model on user ratings. This model learns the patterns of user preferences. It predicts how a specific user would rate it based on their history.

**4. Persona Generation (`generate_personas.py`)**
The system mines the dataset to find users who represent specific "Archetypes." It identifies users who have excessively high ratings for specific genres. These are saved as JSON profiles to be used in the frontend.

**5. Inference Engine (`inference/recommender.py`)**
This is the system that combines everything. When a request comes in:
* It converts the text query into a vector.
* It retrieves the top 100 matches from the FAISS index.
* It uses the SVD model to predict how the *current user* would rate those 100 movies.
* It re-ranks the list and returns the best hybrid matches.