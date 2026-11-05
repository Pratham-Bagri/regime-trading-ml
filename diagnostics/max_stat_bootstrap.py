import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
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

def run_max_stat_bootstrap():
    print("Running deterministic predictions for Max-Statistic Bootstrap...")
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
    
    models = {
        "LR Raw": (Pipeline([('s', StandardScaler()), ('lr', LogisticRegression(random_state=42))]), {'lr__C': [0.1, 1, 10]}),
        "SVM Raw": (Pipeline([('s', StandardScaler()), ('svm', SVC(random_state=42))]), {'svm__C': [0.1, 1, 10], 'svm__kernel': ['linear', 'rbf']}),
        "RF Raw": (Pipeline([('s', StandardScaler()), ('rf', RandomForestClassifier(random_state=42, n_jobs=1))]), {'rf__max_depth': [3, 5], 'rf__n_estimators': [50, 100]}),
        "GB Raw": (Pipeline([('s', StandardScaler()), ('gb', GradientBoostingClassifier(random_state=42))]), {'gb__max_depth': [3], 'gb__n_estimators': [50]}),
        "LR Full": (Pipeline([('s', StandardScaler()), ('pca', PCA(n_components=4, random_state=42)), ('hmm', FeatureConcatenator(n_components=3)), ('lr', LogisticRegression(random_state=42))]), {'lr__C': [0.1, 1]}),
        "SVM Full": (Pipeline([('s', StandardScaler()), ('pca', PCA(n_components=4, random_state=42)), ('hmm', FeatureConcatenator(n_components=3)), ('svm', SVC(random_state=42))]), {'svm__C': [0.1, 1], 'svm__kernel': ['linear', 'rbf']}),
        "SVM PCA-Only": (Pipeline([('s', StandardScaler()), ('pca', PCA(n_components=4, random_state=42)), ('svm', SVC(random_state=42))]), {'svm__C': [0.1, 1], 'svm__kernel': ['linear', 'rbf']}),
        "SVM HMM-Only": (Pipeline([('s', StandardScaler()), ('hmm', FeatureConcatenator(n_components=3)), ('svm', SVC(random_state=42))]), {'svm__C': [0.1, 1], 'svm__kernel': ['linear', 'rbf']})
    }
    
    predictions = {m: [] for m in models}
    y_true_oos = []
    
    for train_idx, test_idx in cv_outer.split(X_cv):
        X_tr, X_te = X_cv[train_idx], X_cv[test_idx]
        y_tr, y_te = y_cv[train_idx], y_cv[test_idx]
        y_true_oos.extend(y_te)
        
        for name, (pipe, grid) in models.items():
            np.random.seed(42 + test_idx[0])
            gs = GridSearchCV(pipe, grid, cv=cv_inner, scoring='precision', n_jobs=1)
            gs.fit(X_tr, y_tr)
            
            if hasattr(gs.best_estimator_[-1], 'decision_function'):
                scores = gs.decision_function(X_te)
            else:
                scores = gs.predict_proba(X_te)[:, 1]
            predictions[name].extend(scores)
            
    y_true_oos = np.array(y_true_oos)
    base_rate = np.mean(y_true_oos)
    target_buys = int(len(y_true_oos) * 0.64)
    
    matched_preds = {}
    actual_lifts = {}
    for name, scores in predictions.items():
        preds = exact_coverage(np.array(scores), target_buys)
        matched_preds[name] = preds
        tp = np.sum((preds == 1) & (y_true_oos == 1))
        actual_lifts[name] = (tp / target_buys - base_rate) * 100
        
    max_actual_lift = max(actual_lifts.values())
    
    print("\nRunning Max-Statistic Block Bootstrap (5,000 iterations)...")
    np.random.seed(42)
    n = len(y_true_oos)
    block_size = 20
    iterations = 5000
    
    max_centered_lifts = []
    for _ in range(iterations):
        idx = []
        while len(idx) < n:
            start = np.random.randint(0, max(1, n - block_size))
            idx.extend(range(start, start + block_size))
        idx = idx[:n]
        
        b_true = y_true_oos[idx]
        b_base = np.mean(b_true)
        
        b_centered_lifts = []
        for name in models.keys():
            p = matched_preds[name][idx]
            if np.sum(p) > 0:
                b_prec = np.sum((p == 1) & (b_true == 1)) / np.sum(p)
                b_lift = (b_prec - b_base) * 100
                b_centered_lifts.append(b_lift - actual_lifts[name])
            else:
                b_centered_lifts.append(-999)
        
        max_centered_lifts.append(max(b_centered_lifts))
        
    max_centered_lifts = np.array(max_centered_lifts)
    p_value = np.mean(max_centered_lifts >= max_actual_lift)
    critical_value = np.percentile(max_centered_lifts, 95)
    
    print(f"\n--- REALITY CHECK RESULTS ---")
    print(f"Max Observed Lift: {max_actual_lift:+.2f} pp")
    print(f"Family-Wise Error Rate (FWER) p-value: {p_value:.4f}")
    print(f"95% Critical Value for Max Lift: {critical_value:+.2f} pp")

if __name__ == '__main__':
    run_max_stat_bootstrap()