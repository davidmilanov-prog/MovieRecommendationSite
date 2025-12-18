import pandas as pd
import os

# Configuration
INPUT_FILE = 'TMDB_movie_dataset_v11.csv'
OUTPUT_FILE = 'cleaned_movies.parquet'
MIN_VOTES = 50

def create_soup(row):
    # Seperate keywords by a space
    kw = str(row['keywords']).replace(',', ' ') if pd.notna(row['keywords']) else ""

    # Return the title, descsription, and keywords
    return f"{row['title']}, {row['overview']} Keywords: {kw}"
  
def clean_data():
    print(f"Loading {INPUT_FILE}")

    # Inspecting the dataset
    needed_cols = [
          'id', 'title', 'overview', 'genres', 'poster_path', 
          'vote_average', 'vote_count', 'release_date', 'keywords'
      ]
    df = pd.read_csv(INPUT_FILE, usecols=needed_cols, low_memory=False)

    initial_count = len(df)
    print(f"Initial Rows: {initial_count}")

    print(f"Filtering out movies without overviews or low in popularity")
    # Filtering the dataset
    # Remove NaNs and empty strings
    df = df[df['overview'].notna() & df['overview'].str.strip() != '']
    # filter by popularity
    df = df[df['vote_count'] > MIN_VOTES]
    # create an embedding column for the LLM
    df['content_for_embedding'] = df.apply(create_soup, axis=1)
    
    # drop keywords column
    cols_to_keep = [
        'id', 'title', 'overview', 'genres', 'poster_path', 
        'vote_average', 'vote_count', 'release_date', 'content_for_embedding'
    ]
    df = df[cols_to_keep]
    print(f"Final Rows: {len(df):,}")

    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Cleaned data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    clean_data()