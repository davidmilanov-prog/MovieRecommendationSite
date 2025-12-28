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
    
    print(f"Loading cleaned dataset")
    df = pd.read_parquet(INPUT_FILE)

    print(f"Loading pretrained model")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Calculating embeddings")
    embeddings = model.encode(df['content_for_embedding'].tolist(),
                              show_progress_bar =True, 
                              convert_to_numpy=True
                              )
    # Convert into type float32 as FAISS is in c++ and requires it
    embeddings = embeddings.astype('float32')

    # Normalize so dot product operation becomes a cosine similarity calculation
    print(f"Normalizing embeddings")
    faiss.normalize_L2(embeddings)

    # Creating the high dimensional vector space. Inner product finds movies similar to eachother
    print(f"Creating vector space")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    # Populating vector space with all embeddings and save
    print(f"Populating space and saving file")
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_FILE))
    
if __name__ == "__main__":
    generate_embeddings()
