## Hybrid Movie Recommendation Engine ##
A sophisticated recommendation system that combines Semantic Content Retrieval with Collaborative Filtering to provide highly personalized movie suggestions.

_Note: This repository currently contains the backend logic. The frontend UI is currently in development._ 

Unlike standard recommenders that rely solely on what other users liked (Collaborative Filtering) or solely on movie tags (Content-Based), this system uses a hybrid two-stage pipeline:

1. Retrieval (Content-Based): Uses Sentence Transformers to embed movie plot summaries into a vector space, allowing the system to find 50-100 movies that are semantically similar to the user's query or history.

2. Ranking (Collaborative Filtering): Uses Singular Value Decomposition (SVD) to predict how the specific user would rate those candidate movies, re-ranking them to ensure the recommendations match their personal taste.

### **Data Sources**
This project relies on two primary datasets. Because the data files are large, they are not included in this repository. You must download them locally.
1. MovieLens (ml-latest): Used for user ratings, tags, and genome scores.
  * https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset 
  * Required Files: movies.csv, ratings.csv, genome-scores.csv, genome-tags.csv, links.csv
2. The Movie Database (TMDB): Used for high-quality plot summaries and poster paths.
  * https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies
  * Required File: TMDB_movie_dataset_v11.csv