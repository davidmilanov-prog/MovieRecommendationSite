import sys
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    ML_LATEST_DIR, 
    TMDB_PATH, 
    CLEANED_DATA_PATH, 
    CLEANED_RATINGS_PATH,
    MIN_VOTES, 
    MIN_USER_RATINGS, 
    TAG_RELEVANCE_THRESHOLD
)

def load_base_movies():
    print("Loading the Movies")
    # load the movies
    movies = pd.read_csv(ML_LATEST_DIR / "movies.csv")
    # load the rating from MOVIELENS_DIR / "ratings.csv" and usecols ]"movieId"]
    ratings = pd.read_csv(ML_LATEST_DIR / "ratings.csv", usecols=['userId', 'movieId'])
    user_counts = ratings['userId'].value_counts()
    valid_users = user_counts[user_counts >= MIN_USER_RATINGS].index
    ratings = ratings[ratings['userId'].isin(valid_users)]
    # create a vote counts column, merge it into the movies dataset, filter movies to ensure vote count >= MIN VOTES
    votes = ratings["movieId"].value_counts().reset_index()
    votes.columns = ["movieId", "vote_count"]

    df = movies.merge(votes, on="movieId")
    df = df[df["vote_count"]>=MIN_VOTES]
    # return the dataframe
    return df
  

def get_genome_tags(valid_movie_ids):
    print("Getting the Genome Tags")
    # load scores and load tags
    scores = pd.read_csv(ML_LATEST_DIR / "genome-scores.csv")
    tags = pd.read_csv(ML_LATEST_DIR / "genome-tags.csv")
    # scores- keep rows for valid movies (in valid_movie_ids)
    scores = scores[scores["movieId"].isin(valid_movie_ids)]
    # scores- keep rows with relavancy > 0.5
    scores = scores[scores["relevance"] >= TAG_RELEVANCE_THRESHOLD]
    # merge scores with tags using on="tagId"
    merged = scores.merge(tags, on="tagId")
    # sort values by relevance (add , ascending=[True, False])
    merged = merged.sort_values("relevance", ascending=False)
    # group movie tags into one column and merge them into one string
    movie_tags = merged.groupby("movieId")["tag"].agg(" ".join).reset_index()
    # rename the columns so it's clear
    movie_tags.columns = ["movieId", "genome_tags"]
    # return tags
    return movie_tags

def merge_tmdb_data(df):
    print("Merging the TMDB Data")
    # link the movie lens id with the tmdb file to get the overview and poster using link file
    links = pd.read_csv(ML_LATEST_DIR / "links.csv")
    # focus only on tmdb column
    links = links.dropna(subset=['tmdbId'])
    # convert to int for tmdb dataset
    links['tmdbId'] = links['tmdbId'].astype(int)

    df = df.merge(links[['movieId', 'tmdbId']], on='movieId', how='left')
    # only read id, overview, and psoter path from the tmdb dataset and merge the dataframe and return
    tmdb_cols = ['id', 'overview', 'poster_path']
    tmdb = pd.read_csv(TMDB_PATH, usecols=tmdb_cols, low_memory=False)
    tmdb.rename(columns={'id': 'tmdbId'}, inplace=True)
    
    df = df.merge(tmdb, on='tmdbId', how='left')
    
    return df

def create_soup(df):
    print("Combining Genome Tags with Summaries")
    df['overview'] = df['overview'].fillna('')
    df['genome_tags'] = df['genome_tags'].apply(lambda x: " ".join(x.split()[:20]))
    
    df["soup"] = (
        "Title: " + df['title'] + 
        " Overview: " + df['overview'] + 
        " Tags: " + df['genome_tags']
    )
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
    df = df[df["soup"].str.strip().str.len() > 0] # remove empty entries
    # keep ['movieId', 'tmdbId', 'title', 'poster_path', 'soup', 'vote_count']
    final_cols = ['movieId', 'tmdbId', 'title', 'poster_path', 'soup', 'overview', 'vote_count']
    df[final_cols].to_parquet(CLEANED_DATA_PATH, index=False)
    
    print("Saving Cleaned Ratings...")
    ratings = pd.read_csv(ML_LATEST_DIR / "ratings.csv")

    # Only keep ratings for movies that actually exist in our cleaned movies df
    ratings = ratings[ratings['movieId'].isin(df['movieId'])]

    # Filter users (min ratings)
    user_counts = ratings['userId'].value_counts()
    valid_users = user_counts[user_counts >= MIN_USER_RATINGS].index
    ratings = ratings[ratings['userId'].isin(valid_users)]
    
    # Save
    ratings.to_parquet(CLEANED_RATINGS_PATH, index=False)

    print("Success. Cleaned Dataset.")

if __name__ == "__main__":
    main()