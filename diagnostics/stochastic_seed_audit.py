import os
# Force OpenBLAS and underlying C-libraries to use only 1 thread to prevent memory blowouts
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from trading_model import FeatureConcatenator
import config
import warnings

warnings.filterwarnings('ignore')

def exact_coverage(y_scores, num_buys):
    preds = np.zeros_like(y_scores, dtype=int)
    top_idx = np.argsort(y_scores)[-num_buys:]
    preds[top_idx] = 1
    return preds

def run_stochastic_audit():
    print("Loading features for stochastic model seed audit (RAM-Safe Mode)...")
    df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col='Date', parse_dates=True)
    
    if 'OBV' in df.columns:
        if 'Volume' in df.columns:
            df['OBV'] = df['OBV'].diff() / df['Volume'].rolling(20).mean()
        else:
            df['OBV'] = df['OBV'].diff() / df['OBV'].diff().abs().rolling(20).mean()
            
    # FIXED: Added the column guards back in
    if 'MACD' in df.columns: df['MACD'] = df['MACD'] / df['Close']
    if 'ATR' in df.columns: df['ATR'] = df['ATR'] / df['Close']
    
    df = df.dropna()
    df['Signal'] = np.where(df['target_return'] > 0, 1, 0)
    
    feature_cols = [c for c in df.columns if c not in ['target_return', 'Signal', 'Close', 'return', 'Volume']]
    X, y = df[feature_cols].values, df['Signal'].values
    
    holdout_idx = int(len(X) * 0.8)
    X_cv, y_cv = X[:holdout_idx], y[:holdout_idx]
    
    cv_outer = TimeSeriesSplit(n_splits=5, gap=20)
    cv_inner = TimeSeriesSplit(n_splits=3, gap=20)
    
    seeds = list(range(42, 62))
    results = {m: [] for m in ["RF Raw", "GB Raw", "LR Full", "SVM Full", "SVM HMM-Only"]}
    
    y_true_oos = []
    for train_idx, test_idx in cv_outer.split(X_cv):
        y_true_oos.extend(y_cv[test_idx])
    y_true_oos = np.array(y_true_oos)
    base_rate = np.mean(y_true_oos)
    target_buys = int(len(y_true_oos) * 0.64)
    
    for seed in seeds:
        print(f"Processing Seed {seed} / {seeds[-1]}...")
        
        models = {
            "RF Raw": (Pipeline([('s', StandardScaler()), ('rf', RandomForestClassifier(random_state=seed, n_jobs=1))]), {'rf__max_depth': [3, 5], 'rf__n_estimators': [50, 100]}),
            "GB Raw": (Pipeline([('s', StandardScaler()), ('gb', GradientBoostingClassifier(random_state=seed))]), {'gb__max_depth': [3], 'gb__n_estimators': [50]}),
            "LR Full": (Pipeline([('s', StandardScaler()), ('pca', PCA(n_components=4, random_state=seed)), 
                                  ('hmm', FeatureConcatenator(n_components=3)), 
                                  ('lr', LogisticRegression(random_state=seed))]), {'lr__C': [0.1, 1]}),
            "SVM Full": (Pipeline([('s', StandardScaler()), ('pca', PCA(n_components=4, random_state=seed)), 
                                   ('hmm', FeatureConcatenator(n_components=3)), 
                                   ('svm', SVC(random_state=seed))]), {'svm__C': [0.1, 1], 'svm__kernel': ['linear', 'rbf']}),
            "SVM HMM-Only": (Pipeline([('s', StandardScaler()), ('hmm', FeatureConcatenator(n_components=3)), 
                                       ('svm', SVC(random_state=seed))]), {'svm__C': [0.1, 1], 'svm__kernel': ['linear', 'rbf']})
        }
        
        predictions = {m: [] for m in models}
        
        for train_idx, test_idx in cv_outer.split(X_cv):
            X_tr, X_te = X_cv[train_idx], X_cv[test_idx]
            y_tr, y_te = y_cv[train_idx], y_cv[test_idx]
            
            for name, (pipe, grid) in models.items():
                np.random.seed(seed + test_idx[0]) 
                
                gs = GridSearchCV(pipe, grid, cv=cv_inner, scoring='precision', n_jobs=1)
                gs.fit(X_tr, y_tr)
                
                if hasattr(gs.best_estimator_[-1], 'decision_function'):
                    scores = gs.decision_function(X_te)
                else:
                    scores = gs.predict_proba(X_te)[:, 1]
                predictions[name].extend(scores)
                
        for name, scores in predictions.items():
            preds = exact_coverage(np.array(scores), target_buys)
            tp = np.sum((preds == 1) & (y_true_oos == 1))
            lift = (tp / target_buys - base_rate) * 100
            results[name].append(lift)
            
        # Force garbage collection to free RAM after each seed
        gc.collect()
            
    print("\n=== 20-SEED STOCHASTIC MODEL VARIANCE ===")
    for name, lifts in results.items():
        print(f"{name:15} | Mean Lift: {np.mean(lifts):+.2f} pp | SD: {np.std(lifts):.2f} pp")

if __name__ == '__main__':
    run_stochastic_audit()