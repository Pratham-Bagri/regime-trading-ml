import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import warnings
from trading_model import FeatureConcatenator
import config

warnings.filterwarnings('ignore')

def exact_coverage(y_scores, num_buys):
    preds = np.zeros_like(y_scores, dtype=int)
    top_idx = np.argsort(y_scores)[-num_buys:]
    preds[top_idx] = 1
    return preds

def single_bootstrap_lift(y_true, preds, base_rate, block_size=20, iterations=5000):
    n = len(y_true)
    lifts = []
    for _ in range(iterations):
        idx = []
        while len(idx) < n:
            start = np.random.randint(0, max(1, n - block_size))
            idx.extend(range(start, start + block_size))
        idx = idx[:n]
        b_true, b_pred = y_true[idx], preds[idx]
        if np.sum(b_pred) > 0:
            prec = np.sum((b_pred == 1) & (b_true == 1)) / np.sum(b_pred)
            lifts.append((prec - base_rate) * 100)
    return np.percentile(lifts, [2.5, 97.5])

def paired_bootstrap_diff(y_true, pred_A, pred_B, block_size=20, iterations=5000):
    n = len(y_true)
    diffs = []
    for _ in range(iterations):
        idx = []
        while len(idx) < n:
            start = np.random.randint(0, max(1, n - block_size))
            idx.extend(range(start, start + block_size))
        idx = idx[:n]
        b_true = y_true[idx]
        pA, pB = pred_A[idx], pred_B[idx]
        if np.sum(pA) > 0 and np.sum(pB) > 0:
            precA = np.sum((pA == 1) & (b_true == 1)) / np.sum(pA)
            precB = np.sum((pB == 1) & (b_true == 1)) / np.sum(pB)
            diffs.append(precA - precB)
    return np.percentile(diffs, [2.5, 97.5])

def run_nested_cv_audit():
    print("Loading and correctly normalizing features (OBV scaled diff)...")
    df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col='Date', parse_dates=True)
    
    if 'OBV' in df.columns:
        if 'Volume' in df.columns:
            df['OBV'] = df['OBV'].diff() / df['Volume'].rolling(20).mean()
        else:
            # Fallback if Volume was dropped but OBV exists
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
        "RF Raw": (Pipeline([('s', StandardScaler()), ('rf', RandomForestClassifier(random_state=42))]), {'rf__max_depth': [3, 5], 'rf__n_estimators': [50, 100]}),
        "GB Raw": (Pipeline([('s', StandardScaler()), ('gb', GradientBoostingClassifier(random_state=42))]), {'gb__max_depth': [3], 'gb__n_estimators': [50]}),
        "LR Full": (Pipeline([('s', StandardScaler()), ('pca', PCA(n_components=4)), ('hmm', FeatureConcatenator(n_components=3)), ('lr', LogisticRegression(random_state=42))]), {'lr__C': [0.1, 1]}),
        "SVM Full": (Pipeline([('s', StandardScaler()), ('pca', PCA(n_components=4)), ('hmm', FeatureConcatenator(n_components=3)), ('svm', SVC(random_state=42))]), {'svm__C': [0.1, 1], 'svm__kernel': ['linear', 'rbf']}),
        "SVM PCA-Only": (Pipeline([('s', StandardScaler()), ('pca', PCA(n_components=4)), ('svm', SVC(random_state=42))]), {'svm__C': [0.1, 1], 'svm__kernel': ['linear', 'rbf']}),
        "SVM HMM-Only": (Pipeline([('s', StandardScaler()), ('hmm', FeatureConcatenator(n_components=3)), ('svm', SVC(random_state=42))]), {'svm__C': [0.1, 1], 'svm__kernel': ['linear', 'rbf']})
    }
    
    predictions = {m: [] for m in models}
    y_true_oos = []
    
    print("Running Nested CV...")
    for train_idx, test_idx in cv_outer.split(X_cv):
        X_tr, X_te = X_cv[train_idx], X_cv[test_idx]
        y_tr, y_te = y_cv[train_idx], y_cv[test_idx]
        y_true_oos.extend(y_te)
        
        for name, (pipe, grid) in models.items():
            gs = GridSearchCV(pipe, grid, cv=cv_inner, scoring='precision', n_jobs=1)
            gs.fit(X_tr, y_tr)
            predictions[name].extend(gs.decision_function(X_te) if hasattr(gs, 'decision_function') else gs.predict_proba(X_te)[:, 1])
                
    y_true_oos = np.array(y_true_oos)
    total_oos = len(y_true_oos)
    base_rate = np.mean(y_true_oos)
    target_buys = int(total_oos * 0.64)
    print(f"\n--- OOS BASE RATE: {np.sum(y_true_oos)}/{total_oos} = {base_rate:.4%} ---")
    
    print("--- 64% MATCHED COVERAGE (Exactly 1277 Buys) ---")
    matched_preds = {}
    for name, scores in predictions.items():
        preds = exact_coverage(np.array(scores), target_buys)
        matched_preds[name] = preds
        tp = np.sum((preds == 1) & (y_true_oos == 1))
        prec = tp / target_buys
        lift = (prec - base_rate) * 100
        ci = single_bootstrap_lift(y_true_oos, preds, base_rate)
        print(f"{name:15} | TP: {tp}/{target_buys} | Prec: {prec:.4f} | Lift: {lift:+.2f} pp | CI: [{ci[0]:+.2f}, {ci[1]:+.2f}]")

    print("\n--- PAIRED DIFFERENCES (95% CI) ---")
    
    ci_full = paired_bootstrap_diff(y_true_oos, matched_preds["SVM Full"], matched_preds["LR Full"])
    print(f"SVM Full - LR Full: [{ci_full[0]:.4f}, {ci_full[1]:.4f}]")
    
    ci_raw = paired_bootstrap_diff(y_true_oos, matched_preds["SVM Raw"], matched_preds["LR Raw"])
    print(f"SVM Raw - LR Raw:   [{ci_raw[0]:.4f}, {ci_raw[1]:.4f}]")

if __name__ == '__main__':
    run_nested_cv_audit()