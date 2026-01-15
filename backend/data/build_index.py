import sys
import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import CLEANED_DATA_PATH, INDEX_PATH, MAPPING_PATH

def main():
    
    # check for input data
    if not CLEANED_DATA_PATH.exists():
        # fallback check if path resolution behaves differently
        potential_path = CLEANED_DATA_PATH
        if potential_path.exists():
            df = pd.read_parquet(potential_path)
        else:
            return
    else:
        df = pd.read_parquet(CLEANED_DATA_PATH)

    # extract necessary columns
    movies_soup = df["soup"].tolist()
    movie_ids = df["movieId"].tolist()
    print(f"Loaded {len(movies_soup)} movies.")

    # generate embeddings
    print("Generating Embeddings")
    # we use multi-qa-mpnet-base-dot-v1 because it is optimized for semantic matching and handles asymmetric query/doc lengths well
    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
    # normalize_embeddings=True ensures that dot product equals cosine similarity
    embeddings = model.encode(movies_soup, 
    show_progress_bar=True, normalize_embeddings=True)

    # build FAISS index
    print("Building FAISS Index...")
    # get the dimension size of the vectors (768)
    d = embeddings.shape[1] 
    # IndexFlatIP calculates Inner Product (Dot Product). Since vectors are normalized, this is equivalent to Cosine Similarity.
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    
    # save artifacts
    print("Saving output files...")
    # save the FAISS index
    faiss.write_index(index, str(INDEX_PATH))
    print(f"   Saved {INDEX_PATH}")
    
    # save the ID mapping (so we know which vector belongs to which movie)
    with open(MAPPING_PATH, 'wb') as f:
        pickle.dump(movie_ids, f)
    print(f"   Saved {MAPPING_PATH}")
    print("Success. Built the Index.")

if __name__ == "__main__":
    main()