import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import config

def train_classifier():
    print("Loading Feature Space and Regime States...")
    df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col='Date', parse_dates=True)
    
    df['Signal'] = np.where(df['target_return'] > 0, 1, 0)
    
    regime_dummies = pd.get_dummies(df['Regime'], prefix='Regime')
    df = pd.concat([df, regime_dummies], axis=1)
    
    pca_cols = [col for col in df.columns if col.startswith('PC')]
    regime_cols = [col for col in df.columns if col.startswith('Regime_')]
    feature_cols = pca_cols + regime_cols
    
    X = df[feature_cols].values
    y = df['Signal'].values
    
    print(f"Executing Walk-Forward Validation with Grid Search on {len(df)} trading days...")
    print("This may take a minute as it trains dozens of SVMs...")
    
    # Outer Loop: Walk-Forward Validation
    tscv_outer = TimeSeriesSplit(n_splits=5)
    precisions = []
    recalls = []
    
    # The Grid of parameters to search
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.01, 0.1]
    }
    
    for train_index, test_index in tscv_outer.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Inner Loop: Time-Strict Grid Search strictly within the training data
        tscv_inner = TimeSeriesSplit(n_splits=3)
        
        # We optimize specifically for 'precision' because we want to minimize false positives
        grid_search = GridSearchCV(
            SVC(kernel='rbf', random_state=42), 
            param_grid, 
            cv=tscv_inner, 
            scoring='precision',
            n_jobs=-1 # Uses all your CPU cores to speed up the math
        )
        
        # Fit the grid search to find the best parameters for this specific fold
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # Predict on the unseen future data
        y_pred = best_model.predict(X_test)
        
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        
    print("\n--- Walk-Forward Validation Results (Tuned SVM) ---")
    print(f"Average Precision: {np.mean(precisions):.4f}")
    print(f"Average Recall:    {np.mean(recalls):.4f}")

if __name__ == "__main__":
    train_classifier()