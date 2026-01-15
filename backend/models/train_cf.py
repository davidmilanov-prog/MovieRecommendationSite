import sys
import pandas as pd
import pickle
from pathlib import Path
from surprise import SVD, Dataset, Reader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    CLEANED_RATINGS_PATH,
    MODEL_PATH, 
    MAX_RATINGS_TRAIN,
    N_FACTORS,
    N_EPOCHS,
    LR_ALL,
    REG_ALL
)

def train_model():
    print("Loading Ratings...")
    
    if CLEANED_RATINGS_PATH.exists():
        df = pd.read_parquet(CLEANED_RATINGS_PATH)
    else:
        print("Error: Cleaned ratings not found. Run preprocess_data.py first.")
        return

    # subsample data for speed
    if len(df) > MAX_RATINGS_TRAIN:
        # take a random sample of rows if dataset is too big (ensure MAX_RATINGS_TRAIN is statistically sufficient for SVD)
        print(f'Sampling top {MAX_RATINGS_TRAIN} ratings.')
        df = df.sample(n=MAX_RATINGS_TRAIN, random_state=42)

    # define the rating scale
    reader = Reader(rating_scale=(0.5, 5.0))
    # load the dataframe into the surprise dataset format
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

    print("Training SVD Model...")
    
    # convert the dataset into a trainset object
    trainset = data.build_full_trainset()

    # initialize the SVD algorithm with optimized hyperparameters
    algo = SVD(n_factors=N_FACTORS, n_epochs=N_EPOCHS, lr_all=LR_ALL, reg_all=REG_ALL)
    # fit the model on the training set
    algo.fit(trainset)

    output_path = MODEL_PATH
    print(f"Saving model to {output_path}...")
    # save the trained model to a pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(algo, f)
        
    print("Success. CF Model Trained.")

if __name__ == "__main__":
    train_model()