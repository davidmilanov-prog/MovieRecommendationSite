import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
from pathlib import Path

# dynamic path
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR.parent / "data" / "cleaned_movies.parquet"

INDEX_FILE = 'movie_embeddings.index'
MODEL_NAME = 'all-mpnet-base-v2' # best, but slow
# MODEL_NAME = 'all-MiniLM-L6-v2' # fast, but slightly worse

def generate_embeddings():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run preprocess_data.py first.")
        return
    
    df = pd.read_parquet(INPUT_FILE)

# Download the LLM (it will happen automatically the first time you run it).

# Process all 27,000+ movie descriptions into a high-dimensional vector space.

# Save that space into a .index file.

if __name__ == "__main__":
    generate_embeddings()
