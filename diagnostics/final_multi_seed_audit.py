import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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

def run_20_seed_audit():
    print("Loading features and setting up 20-seed deterministic audit...")
    df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col='Date', parse_dates=True)
    
    if 'OBV' in df.columns:
        if 'Volume' in df.columns:
            df['OBV'] = df['OBV'].diff() / df['Volume'].rolling(20).mean()
        else:
            df['OBV'] = df['OBV'].diff() / df['OBV'].diff().abs().rolling(20).mean()
            
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
    
    num_seeds = 20
    seeds = list(range(42, 42 + num_seeds))
    
    results = {"LR Raw": [], "SVM Raw": [], "LR Full": [], "SVM Full": []}
    paired_raw = []
    paired_full = []
    
    y_true_oos = []
    for train_idx, test_idx in cv_outer.split(X_cv):
        y_true_oos.extend(y_cv[test_idx])
    y_true_oos = np.array(y_true_oos)
    base_rate = np.mean(y_true_oos)
    target_buys = int(len(y_true_oos) * 0.64)
    
    for seed in seeds:
        print(f"Processing Seed {seed} / {seeds[-1]}...")
        
        # Enforce deterministic HMM EM initialization via numpy's global random state
        np.random.seed(seed)
        
        models = {
            "LR Raw": (Pipeline([('s', StandardScaler()), ('lr', LogisticRegression(random_state=seed))]), {'lr__C': [0.1, 1, 10]}),
            "SVM Raw": (Pipeline([('s', StandardScaler()), ('svm', SVC(random_state=seed))]), {'svm__C': [0.1, 1, 10], 'svm__kernel': ['linear', 'rbf']}),
            "LR Full": (Pipeline([('s', StandardScaler()), ('pca', PCA(n_components=4, random_state=seed)), 
                                  ('hmm', FeatureConcatenator(n_components=3)), 
                                  ('lr', LogisticRegression(random_state=seed))]), {'lr__C': [0.1, 1]}),
            "SVM Full": (Pipeline([('s', StandardScaler()), ('pca', PCA(n_components=4, random_state=seed)), 
                                   ('hmm', FeatureConcatenator(n_components=3)), 
                                   ('svm', SVC(random_state=seed))]), {'svm__C': [0.1, 1], 'svm__kernel': ['linear', 'rbf']})
        }
        
        predictions = {m: [] for m in models}
        
        for train_idx, test_idx in cv_outer.split(X_cv):
            X_tr, X_te = X_cv[train_idx], X_cv[test_idx]
            y_tr, y_te = y_cv[train_idx], y_cv[test_idx]
            
            for name, (pipe, grid) in models.items():
                gs = GridSearchCV(pipe, grid, cv=cv_inner, scoring='precision', n_jobs=1)
                gs.fit(X_tr, y_tr)
                
                if hasattr(gs.best_estimator_[-1], 'decision_function'):
                    scores = gs.decision_function(X_te)
                else:
                    scores = gs.predict_proba(X_te)[:, 1]
                predictions[name].extend(scores)
                
        matched_preds = {}
        for name, scores in predictions.items():
            preds = exact_coverage(np.array(scores), target_buys)
            matched_preds[name] = preds
            tp = np.sum((preds == 1) & (y_true_oos == 1))
            lift = (tp / target_buys - base_rate) * 100
            results[name].append(lift)
            
        p_raw = np.sum((matched_preds["SVM Raw"] == 1) & (y_true_oos == 1)) - np.sum((matched_preds["LR Raw"] == 1) & (y_true_oos == 1))
        p_full = np.sum((matched_preds["SVM Full"] == 1) & (y_true_oos == 1)) - np.sum((matched_preds["LR Full"] == 1) & (y_true_oos == 1))
        
        paired_raw.append(p_raw / target_buys * 100)
        paired_full.append(p_full / target_buys * 100)
        
    print("\n=== 20-SEED AGGREGATED LIFT OVER BASE RATE ===")
    for name, lifts in results.items():
        print(f"{name:10} | Mean Lift: {np.mean(lifts):+.2f} pp | SD: {np.std(lifts):.2f} pp | Min: {np.min(lifts):+.2f} | Max: {np.max(lifts):+.2f}")
        
    print("\n=== PAIRED DIFFERENCES (SVM - LR) ===")
    print(f"Raw Features  | Mean Diff: {np.mean(paired_raw):+.2f} pp | SD: {np.std(paired_raw):.2f} pp")
    print(f"Full Pipeline | Mean Diff: {np.mean(paired_full):+.2f} pp | SD: {np.std(paired_full):.2f} pp")

if __name__ == '__main__':
    run_20_seed_audit()