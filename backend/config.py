from pathlib import Path

# directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ML_LATEST_DIR = DATA_DIR / "ml-latest"
MODEL_DIR = BASE_DIR / "models"

# file paths
CLEANED_DATA_PATH = DATA_DIR / "cleaned_movies.parquet"
CLEANED_RATINGS_PATH = DATA_DIR / "cleaned_ratings.parquet"
RATINGS_PATH = ML_LATEST_DIR / "ratings.csv"
TMDB_PATH = DATA_DIR / "TMDB_movie_dataset_v11.csv" 
INDEX_PATH = DATA_DIR / "movies.index"              
MAPPING_PATH = DATA_DIR / "movie_ids.pkl"           
MODEL_PATH = MODEL_DIR / "svd_model.pkl"

# hyperparameters
MIN_VOTES = 50
MIN_USER_RATINGS = 10
MAX_RATINGS_TRAIN = 4_000_000
TAG_RELEVANCE_THRESHOLD = 0.5
N_FACTORS = 100
N_EPOCHS = 30
LR_ALL = 0.005
REG_ALL = 0.02

# archetypes for prompt engineering
ARCHETYPES = {
    "The Horror Fan": "Horror",
    "The Sci-Fi Geek": "Sci-Fi",
    "The Action Hero": "Action",
    "The Hopeless Romantic": "Romance",
    "The Comedy Club": "Comedy",
    "The Drama Critic": "Drama",
    "The 90s Kid": "Children's 90s"
}