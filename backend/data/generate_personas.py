import sys
import pandas as pd
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import ML_LATEST_DIR, DATA_DIR

def generate_personas():
    print("Creating User Personas")
    
    # Load Data
    print("Loading Movies and Ratings")
    movies = pd.read_csv(ML_LATEST_DIR / "movies.csv")
    ratings = pd.read_csv(ML_LATEST_DIR / "ratings.csv")

    # We only care about what they love (4.0+), not what they hate.
    liked_ratings = ratings[ratings['rating'] >= 4.0]

    # Merge with movies to get genres
    # This creates a massive dataframe of User | Movie | Genre
    merged = liked_ratings.merge(movies[['movieId', 'genres']], on='movieId')

    # Explode Genres (because "Action|Sci-Fi" needs to count for both)
    # This splits the pipe-separated string into individual rows
    merged['genres'] = merged['genres'].str.split('|')
    exploded = merged.explode('genres')

    # Count Genres per User
    user_genre_counts = exploded.groupby(['userId', 'genres']).size().reset_index(name='count')

    # Define the Archetypes we want to find
    target_genres = [
        "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", 
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", 
        "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]

    persona_list = []
    
    # Add a "Generic" average user that is the most active
    most_active_user = ratings['userId'].value_counts().idxmax()
    persona_list.append({"id": int(most_active_user), "label": "The Movie Buff"})

    print("Finding Archetypes.")
    used_user_ids = {p['id'] for p in persona_list}

    for genre in target_genres:
        # Filter for the specific genre
        genre_data = user_genre_counts[user_genre_counts['genres'] == genre]
        
        # Sort by count (descending)
        potential_fans = genre_data.sort_values('count', ascending=False)
        
        # Iterate through the top fans until we find one not already used
        for _, row in potential_fans.iterrows():
            user_id = int(row['userId'])
            count = int(row['count'])
            
            if user_id not in used_user_ids:
                label = f"{genre} Enthusiast"
                
                persona_list.append({
                    "id": user_id, 
                    "label": label,
                    "description": f"Rated {count} {genre} movies highly."
                })
                
                # Mark this user as used
                used_user_ids.add(user_id)
                
                # We found our archetype for this genre, stop looking at users
                break

    # Save to JSON
    output_path = DATA_DIR / "user_personas.json"
    with open(output_path, "w") as f:
        json.dump(persona_list, f, indent=2)
        
    print(f"Generated {len(persona_list)} personas saved to {output_path}")

if __name__ == "__main__":
    generate_personas()