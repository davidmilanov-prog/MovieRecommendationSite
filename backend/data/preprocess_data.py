import pandas as pd
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
MOVIELENS_DIR = SCRIPT_DIR / "ml-latest"
TMDB_FILE = SCRIPT_DIR.parent  / "data" / "TMDB_movie_dataset_v11.csv"
OUTPUT_FILE = 'cleaned_movies.parquet'

# Keep movies with >50 rating to ensure quality
MIN_VOTES = 50
# Keep genome scores with >0.5 score only
TAG_RELEVANCE_THRESHOLD = 0.5

# Goal: create a dataset with a "soup" column designed for embedding that is genome tags + overview
def load_base_movies():
    # load the movies from MOVIELENS_DIR / "movies.csv"
    movies = pd.read_csv(MOVIELENS_DIR / "movies.csv")
    # load the rating from MOVIELENS_DIR / "ratings.csv" and usecols ]"movieId"]
    ratings = pd.read_csv(MOVIELENS_DIR / "ratings.csv", usecols=['movieId'])
    # create a vote counts column, merge it into the movies dataset, filter movies to ensure vote count >= MIN VOTES
    votes = ratings["movieId"].value_counts().reset_index()
    votes.columns = ["movieId", "vote_count"]

    df = movies.merge(votes, on="movieId")
    df = df[df["vote_count"]>=MIN_VOTES]
    # return the dataframe
    return df
  

def get_genome_tags(valid_movie_ids):
    # load scores and load tags
    scores = pd.read_csv(MOVIELENS_DIR / "genome-scores.csv")
    tags = pd.read_csv(MOVIELENS_DIR / "genome-tags.csv")
    # scores- keep rows for valid movies (in valid_movie_ids)
    scores = scores[scores["movieId"].isin(valid_movie_ids)]
    # scores- keep rows with relavancy > 0.5
    scores = scores[scores["relevance"] >= TAG_RELEVANCE_THRESHOLD]
    # merge scores with tags using on="tagId"
    merged = scores.merge(tags, on="tagId")
    # sort values by relevance (add , ascending=[True, False])
    merged = merged.sort_values("relevance", ascending=False)
    # group movie tags into one column and merge them into one string
    movie_tags = merged.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()
    # rename the columns so it's clear
    movie_tags.columns = ["movieId", "genome_tags"]
    # return tags
    return movie_tags

def merge_tmdb_data(df):
    # link the movie lens id with the tmdb file to get the overview and poster using link file
    links = pd.read_csv(MOVIELENS_DIR / "links.csv")
    # focus only on tmdb column
    links = links.dropna(subset=['tmdbId'])
    # convert to int for tmdb dataset
    links['tmdbId'] = links['tmdbId'].astype(int)

    df = df.merge(links[['movieId', 'tmdbId']], on='movieId', how='left')
    # only read id, overview, and psoter path from the tmdb dataset and merge the dataframe and return
    tmdb_cols = ['id', 'overview', 'poster_path']
    tmdb = pd.read_csv(TMDB_FILE, usecols=tmdb_cols, low_memory=False)
    tmdb.rename(columns={'id': 'tmdbId'}, inplace=True)
    
    df = df.merge(tmdb, on='tmdbId', how='left')
    
    return df

def create_soup(df):
    # create soup column by adding genome tags and overview columns. 
    df['genome_tags'] = df['genome_tags'].fillna('')
    df['overview'] = df['overview'].fillna('')
    df["soup"] = df['genome_tags'] + " " + df['overview']
    
    return df

def main():
    # load the base movies
    df = load_base_movies()
    # get valid ids
    valid_ids = df["movieId"].unique()
    # create a tags df
    tags = get_genome_tags(valid_ids)
    # merge the base movies and tags
    df = df.merge(tags, on="movieId")
    # merge the above with tmdb
    df = merge_tmdb_data(df)
    # create soup column
    df = create_soup(df)
    # keep ['movieId', 'tmdbId', 'title', 'poster_path', 'soup', 'vote_count']
    final_cols = ['movieId', 'tmdbId', 'title', 'poster_path', 'soup', 'vote_count']
    df[final_cols].to_parquet(OUTPUT_FILE, index=False)
    # done

if __name__ == "__main__":
    main()