import os
os.environ["OMP_NUM_THREADS"] = "1"
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

def run_final_diagnostics():
    print("Loading data for Final Reviewer Diagnostics...")
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
    X_ho, y_ho = X[holdout_idx:], y[holdout_idx:]
    
    cv_outer = TimeSeriesSplit(n_splits=5, gap=20)
    
    # ---------------------------------------------------------
    # PART 1: HOLDOUT DECOUPLING (Scale Shift vs Market Drift)
    # ---------------------------------------------------------
    print("\n--- HOLDOUT DIAGNOSIS DECOUPLING ---")
    models_to_test = {
        "SVM Full": Pipeline([('s', StandardScaler()), ('pca', PCA(n_components=4, random_state=42)), ('hmm', FeatureConcatenator(n_components=3)), ('svm', SVC(C=1, kernel='rbf', random_state=42))]),
        "LR Full": Pipeline([('s', StandardScaler()), ('pca', PCA(n_components=4, random_state=42)), ('hmm', FeatureConcatenator(n_components=3)), ('lr', LogisticRegression(C=1, random_state=42))])
    }
    
    for name, pipe in models_to_test.items():
        print(f"\nAnalyzing {name}:")
        
        # 1. Get pooled OOF scores (Base Scale)
        oof_scores = []
        fold_models = []
        last_cv_idx = None
        for train_idx, test_idx in cv_outer.split(X_cv):
            np.random.seed(42 + test_idx[0])
            pipe.fit(X_cv[train_idx], y_cv[train_idx])
            fold_models.append(pipe)
            if hasattr(pipe[-1], 'decision_function'):
                oof_scores.extend(pipe.decision_function(X_cv[test_idx]))
            else:
                oof_scores.extend(pipe.predict_proba(X_cv[test_idx])[:, 1])
            last_cv_idx = test_idx
            
        oof_scores = np.array(oof_scores)
        oof_mean = np.mean(oof_scores)
        
        # 2. Refit on all CV
        np.random.seed(42)
        pipe.fit(X_cv, y_cv)
        
        # 3. Score Last CV Fold with Refit Model (Isolates Scale Shift)
        if hasattr(pipe[-1], 'decision_function'):
            last_fold_refit_scores = pipe.decision_function(X_cv[last_cv_idx])
        else:
            last_fold_refit_scores = pipe.predict_proba(X_cv[last_cv_idx])[:, 1]
            
        # 4. Score Holdout with averaged Fold Models (Isolates Market Drift)
        ho_fold_scores = np.zeros(len(X_ho))
        for fm in fold_models:
            if hasattr(fm[-1], 'decision_function'):
                ho_fold_scores += fm.decision_function(X_ho)
            else:
                ho_fold_scores += fm.predict_proba(X_ho)[:, 1]
        ho_fold_scores /= len(fold_models)
        
        # 5. Score Holdout with Refit Model (The actual collapse scenario)
        if hasattr(pipe[-1], 'decision_function'):
            ho_refit_scores = pipe.decision_function(X_ho)
        else:
            ho_refit_scores = pipe.predict_proba(X_ho)[:, 1]
            
        print(f"  Pooled OOF Mean (Baseline):        {oof_mean:.4f}")
        print(f"  Last CV Fold via Refit (Scale):    {np.mean(last_fold_refit_scores):.4f}")
        print(f"  Holdout via Fold Models (Drift):   {np.mean(ho_fold_scores):.4f}")
        print(f"  Holdout via Refit Model (Combined):{np.mean(ho_refit_scores):.4f}")

    # ---------------------------------------------------------
    # PART 2: RIGOROUS REALITY CHECK BOOTSTRAP (10,000 Iterations)
    # ---------------------------------------------------------
    print("\n--- GENERATING DETERMINISTIC PREDICTIONS FOR REALITY CHECK ---")
    cv_inner = TimeSeriesSplit(n_splits=3, gap=20)
    
    models = {
        "LR Raw": (Pipeline([('s', StandardScaler()), ('lr', LogisticRegression(random_state=42))]), {'lr__C': [0.1, 1, 10]}),
        "SVM Raw": (Pipeline([('s', StandardScaler()), ('svm', SVC(random_state=42))]), {'svm__C': [0.1, 1, 10], 'svm__kernel': ['linear', 'rbf']}),
        "RF Raw": (Pipeline([('s', StandardScaler()), ('rf', RandomForestClassifier(random_state=42, n_jobs=1))]), {'rf__max_depth': [3, 5], 'rf__n_estimators': [50, 100]}),
        "GB Raw": (Pipeline([('s', StandardScaler()), ('gb', GradientBoostingClassifier(random_state=42))]), {'gb__max_depth': [3], 'gb__n_estimators': [50]}),
        "LR Full": (Pipeline([('s', StandardScaler()), ('pca', PCA(n_components=4, random_state=42)), ('hmm', FeatureConcatenator(n_components=3)), ('lr', LogisticRegression(random_state=42))]), {'lr__C': [0.1, 1]}),
        "SVM Full": (Pipeline([('s', StandardScaler()), ('pca', PCA(n_components=4, random_state=42)), ('hmm', FeatureConcatenator(n_components=3)), ('svm', SVC(random_state=42))]), {'svm__C': [0.1, 1], 'svm__kernel': ['linear', 'rbf']}),
        "SVM PCA": (Pipeline([('s', StandardScaler()), ('pca', PCA(n_components=4, random_state=42)), ('svm', SVC(random_state=42))]), {'svm__C': [0.1, 1], 'svm__kernel': ['linear', 'rbf']}),
        "SVM HMM": (Pipeline([('s', StandardScaler()), ('hmm', FeatureConcatenator(n_components=3)), ('svm', SVC(random_state=42))]), {'svm__C': [0.1, 1], 'svm__kernel': ['linear', 'rbf']})
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
    
    print("\nRunning Reality Check (10,000 resamples)...")
    np.random.seed(42)
    n = len(y_true_oos)
    block_size = 20
    iterations = 10000
    
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
                # Centered exactly at null (lift = 0)
                b_centered_lifts.append(b_lift - actual_lifts[name])
            else:
                b_centered_lifts.append(-999)
        
        max_centered_lifts.append(max(b_centered_lifts))
        
    max_centered_lifts = np.array(max_centered_lifts)
    # One-sided FWER p-value
    fwer_p = np.mean(max_centered_lifts >= max_actual_lift)
    crit_val = np.percentile(max_centered_lifts, 95)
    
    print(f"\n--- RIGOROUS REALITY CHECK RESULTS ---")
    print(f"Target Benchmark (Base Rate): {base_rate*100:.2f}%")
    print(f"Max Observed Lift: {max_actual_lift:+.2f} pp")
    print(f"FWER Adjusted p-value (One-Sided): {fwer_p:.4f}")
    print(f"95% Critical Value for Max Lift: {crit_val:+.2f} pp")

if __name__ == '__main__':
    run_final_diagnostics()