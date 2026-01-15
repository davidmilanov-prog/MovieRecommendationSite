import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# pytorch and faiss were colliding so the above code prevents this (forces libraries to use a single thread)

import argparse
from config import (
    CLEANED_DATA_PATH, 
    CLEANED_RATINGS_PATH,
    INDEX_PATH, 
    MAPPING_PATH, 
    MODEL_PATH, 
    PERSONAS_PATH,
    MODEL_DIR
)

def run_pipeline(force_rerun=False):
    print("Starting Movie Recommendation Pipeline")

    # Check if we still need to clean the raw data
    if not CLEANED_DATA_PATH.exists() or not CLEANED_RATINGS_PATH.exists() or force_rerun:
        print("\n[1/5] Cleaned data not found. Preprocessing now.")
        from data import preprocess_data
        preprocess_data.main()
    else:
        print(f"\n[1/5] Found Cleaned Data. Skipping preprocessing.")

    # Check if the vector index and ID mapping exist
    if not INDEX_PATH.exists() or not MAPPING_PATH.exists() or force_rerun:
        print("\n[2/5] Index not found. Building FAISS Index.")
        from data import build_index
        build_index.main()
    else:
        print(f"\n[2/5] Found FAISS artifacts. Skipping index build.")

    # Check if the collaborative filtering model exists
    if not MODEL_PATH.exists() or force_rerun:
        print("\n[3/5] SVD Model not found. Training Model.")
        # Check if the directory exists first
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        from models import train_cf
        train_cf.train_model()
    else:
        print(f"\n[3/5] Found {MODEL_PATH.name}. Skipping training.")

    if not PERSONAS_PATH.exists() or force_rerun:
        print("\n[4/5] Archetypes not found. Creating Now.")
        from data import generate_personas
        generate_personas.generate_personas()
    else:
        print(f"\n[4/5] Found {PERSONAS_PATH.name}. Skipping.")

    print("\n[5/5] Loading Inference Engine.")
    from inference.recommender import MovieRecommender
    
    # Initialize the singleton engine (loads all artifacts into memory)
    rec_engine = MovieRecommender()

    print("\nSystem Ready.")
    print("Interactive Mode: Type 'exit' or 'quit' to stop.")
    # Interactive CLI Loop TEMPORARY TO PROVE SYSTEM WORKS
    while True:
        try:
            uid_input = input("\nEnter User ID (default: 1): ").strip()
            if uid_input.lower() in ['exit', 'quit', 'q']: 
                break
            
            # Handle default or empty input
            user_id = int(uid_input) if uid_input else 1
            
            query = input("Describe the movie: ").strip()
            if not query:
                print("Please enter a description.")
                continue

            print(f"Searching for '{query}' for User {user_id}...")
            results = rec_engine.recommend(user_id, query)
            
            if results:
                # Define column headers
                header = f"{'ID':<7} | {'Rating':<6} | {'Votes':<6} | {'Match Type':<10} | {'Title'}"
                separator = "-" * len(header)
                
                # Print Header
                print(separator)
                print(header)
                print(separator)
                
                # Print Rows
                for r in results:
                    title = r['title'][:45] + "..." if len(r['title']) > 45 else r['title']
                    
                    print(f"{r['movieId']:<7} | {r['predicted_rating']:<6} | {r['votes']:<6} | {r['match_score']:<10} | {title}")
                
                print(separator)
            else:
                print("No results found.")

        except ValueError:
            print("Error: User ID must be an integer.")
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            break
        except Exception as e:
            print(f"Unexpected Error: {e}")

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force regenerate all data and models")
    args = parser.parse_args()
    
    # start the pipeline
    run_pipeline(force_rerun=args.force)