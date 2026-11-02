import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
import config

def train_classifier():
    print("Loading Feature Space and Regime States...")
    df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col='Date', parse_dates=True)
    
    # 1. Define the Binary Target
    df['Signal'] = np.where(df['target_return'] > 0, 1, 0)
    
    # 2. One-Hot Encode the Regime
    regime_dummies = pd.get_dummies(df['Regime'], prefix='Regime')
    df = pd.concat([df, regime_dummies], axis=1)
    
    # 3. Define the final Feature Matrix (X) and Target (y)
    pca_cols = [col for col in df.columns if col.startswith('PC')]
    regime_cols = [col for col in df.columns if col.startswith('Regime_')]
    feature_cols = pca_cols + regime_cols
    
    X = df[feature_cols].values
    y = df['Signal'].values
    
    print(f"Executing Walk-Forward Validation with SVM on {len(df)} trading days...")
    
    # 4. Walk-Forward Validation
    tscv = TimeSeriesSplit(n_splits=5)
    precisions = []
    recalls = []
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # 5. Fit the Support Vector Machine
        # kernel='rbf' gives us the non-linear, curved decision boundaries
        # C=1.0 is the regularization parameter (controls overfitting vs underfitting)
        model = SVC(kernel='rbf', C=1.0, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        
    print("\n--- Walk-Forward Validation Results (SVM) ---")
    print(f"Average Precision: {np.mean(precisions):.4f}")
    print(f"Average Recall:    {np.mean(recalls):.4f}")

if __name__ == "__main__":
    train_classifier()