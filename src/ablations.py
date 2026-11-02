import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import config

# Import our custom mathematical classes directly from the trading model
from trading_model import CausalFilteredHMM, FeatureConcatenator

def run_ablations():
    print("Loading raw rolling features...")
    df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col='Date', parse_dates=True)
    df = df.dropna()
    
    df['Signal'] = np.where(df['target_return'] > 0, 1, 0)
    feature_cols = [col for col in df.columns if col not in ['target_return', 'Signal', 'Close', 'return']]
    X = df[feature_cols].values
    y = df['Signal'].values
    
    # Establish the untouchable 20% holdout set
    holdout_idx = int(len(X) * 0.8)
    X_cv, y_cv = X[:holdout_idx], y[:holdout_idx]
    
    cv_outer = TimeSeriesSplit(n_splits=5, gap=20)
    cv_inner = TimeSeriesSplit(n_splits=3, gap=20)
    
    # 1. Define the structural variants to test
    experiments = {
        "1. Baseline LR (Raw Features)": {
            "pipeline": Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(random_state=42, max_iter=1000))
            ]),
            "params": {'lr__C': [0.1, 1, 10]}
        },
        "2. SVM (Raw Features Only)": {
            "pipeline": Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(random_state=42))
            ]),
            "params": {'svm__C': [0.1, 1, 10], 'svm__kernel': ['rbf', 'linear']}
        },
        "3. SVM (PCA Only, No HMM)": {
            "pipeline": Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=4)),
                ('svm', SVC(random_state=42))
            ]),
            "params": {'svm__C': [0.1, 1, 10], 'svm__kernel': ['rbf', 'linear']}
        },
        "4. SVM (HMM States Only)": {
            "pipeline": Pipeline([
                ('scaler', StandardScaler()),
                ('hmm', CausalFilteredHMM(n_components=3)),
                ('svm', SVC(random_state=42))
            ]),
            "params": {'svm__C': [0.1, 1, 10], 'svm__kernel': ['rbf', 'linear']}
        },
        "5. SVM (Full Pipeline)": {
            "pipeline": Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=4)),
                ('hmm_concat', FeatureConcatenator(n_components=3)),
                ('svm', SVC(random_state=42))
            ]),
            "params": {'svm__C': [0.1, 1, 10], 'svm__kernel': ['rbf', 'linear']}
        }
    }
    
    print(f"Running Ablation Study on {len(X_cv)} days. This will take a few minutes...\n")
    
    for name, config_dict in experiments.items():
        precisions = []
        for train_idx, test_idx in cv_outer.split(X_cv):
            X_train, X_test = X_cv[train_idx], X_cv[test_idx]
            y_train, y_test = y_cv[train_idx], y_cv[test_idx]
            
            grid = GridSearchCV(
                config_dict["pipeline"], 
                config_dict["params"], 
                cv=cv_inner, 
                scoring='precision', 
                n_jobs=None
            )
            
            grid.fit(X_train, y_train)
            y_pred = grid.predict(X_test)
            
            precisions.append(precision_score(y_test, y_pred, zero_division=0))
            
        print(f"{name}: {np.mean(precisions):.4f} Average Precision")

if __name__ == "__main__":
    run_ablations()