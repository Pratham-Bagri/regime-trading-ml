import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
import config

def train_classifier():
    print("Loading Feature Space and Regime States...")
    df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col='Date', parse_dates=True)
    
    # 1. Define the Binary Target
    # If tomorrow's return is > 0, Buy (1). Else, Sell (0).
    df['Signal'] = np.where(df['target_return'] > 0, 1, 0)
    
    # 2. One-Hot Encode the Regime
    # Regimes 0, 1, 2 are independent categories, not ordinal magnitudes.
    regime_dummies = pd.get_dummies(df['Regime'], prefix='Regime')
    df = pd.concat([df, regime_dummies], axis=1)
    
    # 3. Define the final Feature Matrix (X) and Target (y)
    pca_cols = [col for col in df.columns if col.startswith('PC')]
    regime_cols = [col for col in df.columns if col.startswith('Regime_')]
    feature_cols = pca_cols + regime_cols
    
    X = df[feature_cols].values
    y = df['Signal'].values
    
    print(f"Executing Walk-Forward Validation on {len(df)} trading days...")
    
    # 4. Walk-Forward Validation (Time Series Split)
    # n_splits=5 slices the timeline into 5 chronological testing windows
    tscv = TimeSeriesSplit(n_splits=5)
    precisions = []
    recalls = []
    
    for train_index, test_index in tscv.split(X):
        # Strictly split past data (train) from future data (test)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # 5. Fit the Logistic Regression (Sigmoid Mapping)
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Predict tomorrow's signal based solely on today's trained knowledge
        y_pred = model.predict(X_test)
        
        # 6. Evaluate using strictly quantitative finance metrics
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        
    print("\n--- Walk-Forward Validation Results ---")
    print(f"Average Precision: {np.mean(precisions):.4f}")
    print(f"Average Recall:    {np.mean(recalls):.4f}")

if __name__ == "__main__":
    train_classifier()