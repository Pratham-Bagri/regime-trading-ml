import pandas as pd
import os
import pickle
from hmmlearn.hmm import GaussianHMM
import config

def train_hmm():
    print("Loading PCA Feature Space...")
    df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col='Date', parse_dates=True)
    
    # Extract just the PCA columns (ignoring the target_return for unsupervised learning)
    pca_cols = [col for col in df.columns if col.startswith('PC')]
    X_train = df[pca_cols].values
    
    print(f"Training Hidden Markov Model with K=3 regimes on {X_train.shape[1]} components...")
    
    # Initialize the HMM
    # covariance_type='full' allows the clusters to be elliptical (handling varying volatility)
    hmm_model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
    
    # The EM Algorithm fits the Gaussians and the Markov Transition Matrix
    hmm_model.fit(X_train)
    
    # Decode the hidden states using the Viterbi Algorithm
    hidden_states = hmm_model.predict(X_train)
    
    # Append the decoded states back to our dataframe
    df['Regime'] = hidden_states
    
    # Save the updated dataframe
    df.to_csv(config.PROCESSED_DATA_PATH)
    print(f"HMM complete. Regimes appended to {config.PROCESSED_DATA_PATH}")
    
    # Save the actual trained model to disk so we can use it on future data
    model_path = os.path.join(config.BASE_DIR, "data", "processed", "hmm_model.pkl")
    with open(model_path, "wb") as file:
        pickle.dump(hmm_model, file)
    print(f"Trained HMM saved to {model_path}")

if __name__ == "__main__":
    train_hmm()