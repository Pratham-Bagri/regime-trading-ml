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
from trading_model import CausalFilteredHMM, FeatureConcatenator
import config

warnings.filterwarnings('ignore')

def exact_coverage(y_scores, num_buys):
    """Forces exactly num_buys to be 1."""
    preds = np.zeros_like(y_scores, dtype=int)
    top_idx = np.argsort(y_scores)[-num_buys:]
    preds[top_idx] = 1
    return preds

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

def economic_bootstrap(ret_model, ret_bench, block_size=20, iterations=5000):
    n = len(ret_model)
    ret_diffs, sharpe_diffs = [], []
    for _ in range(iterations):
        idx = []
        while len(idx) < n:
            start = np.random.randint(0, max(1, n - block_size))
            idx.extend(range(start, start + block_size))
        idx = idx[:n]
        
        rM, rB = ret_model[idx], ret_bench[idx]
        
        ann_M = (1 + np.prod(1 + rM) - 1) ** (252 / n) - 1
        vol_M = np.std(rM) * np.sqrt(252)
        sh_M = ann_M / vol_M if vol_M != 0 else 0
        
        ann_B = (1 + np.prod(1 + rB) - 1) ** (252 / n) - 1
        vol_B = np.std(rB) * np.sqrt(252)
        sh_B = ann_B / vol_B if vol_B != 0 else 0
        
        ret_diffs.append((ann_M - ann_B) * 100)
        sharpe_diffs.append(sh_M - sh_B)
        
    return np.percentile(ret_diffs, [2.5, 97.5]), np.percentile(sharpe_diffs, [2.5, 97.5])

def run_master_audit():
    print("Loading and strictly normalizing features...")
    df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col='Date', parse_dates=True).dropna()
    
    # Strictly normalize non-stationary features
    if 'OBV' in df.columns: df['OBV'] = df['OBV'].pct_change()
    if 'MACD' in df.columns: df['MACD'] = df['MACD'] / df['Close']
    if 'ATR' in df.columns: df['ATR'] = df['ATR'] / df['Close']
    df = df.dropna()
    
    df['Signal'] = np.where(df['target_return'] > 0, 1, 0)
    feature_cols = [c for c in df.columns if c not in ['target_return', 'Signal', 'Close', 'return']]
    
    X, y, rets = df[feature_cols].values, df['Signal'].values, df['target_return'].values
    holdout_idx = int(len(X) * 0.8)
    X_cv, y_cv = X[:holdout_idx], y[:holdout_idx]
    X_ho, y_ho, rets_ho = X[holdout_idx:], y[holdout_idx:], rets[holdout_idx:]
    
    cv_outer = TimeSeriesSplit(n_splits=5, gap=20)
    cv_inner = TimeSeriesSplit(n_splits=3, gap=20)
    
    models = {
        "LR Raw": (Pipeline([('s', StandardScaler()), ('lr', LogisticRegression())]), {'lr__C': [0.1, 1, 10]}),
        "SVM Raw": (Pipeline([('s', StandardScaler()), ('svm', SVC())]), {'svm__C': [0.1, 1, 10], 'svm__kernel': ['linear', 'rbf']}),
        "RF Raw": (Pipeline([('s', StandardScaler()), ('rf', RandomForestClassifier(random_state=42))]), {'rf__max_depth': [3, 5], 'rf__n_estimators': [50, 100]}),
        "GB Raw": (Pipeline([('s', StandardScaler()), ('gb', GradientBoostingClassifier(random_state=42))]), {'gb__max_depth': [3], 'gb__n_estimators': [50]}),
        "LR Full": (Pipeline([('s', StandardScaler()), ('pca', PCA(n_components=4)), ('hmm', FeatureConcatenator(n_components=3)), ('lr', LogisticRegression())]), {'lr__C': [0.1, 1]}),
        "SVM Full": (Pipeline([('s', StandardScaler()), ('pca', PCA(n_components=4)), ('hmm', FeatureConcatenator(n_components=3)), ('svm', SVC())]), {'svm__C': [0.1, 1], 'svm__kernel': ['linear', 'rbf']}),
        "SVM PCA-Only": (Pipeline([('s', StandardScaler()), ('pca', PCA(n_components=4)), ('svm', SVC())]), {'svm__C': [0.1, 1], 'svm__kernel': ['linear', 'rbf']}),
        "SVM HMM-Only": (Pipeline([('s', StandardScaler()), ('hmm', FeatureConcatenator(n_components=3)), ('svm', SVC())]), {'svm__C': [0.1, 1], 'svm__kernel': ['linear', 'rbf']})
    }
    
    predictions = {m: [] for m in models}
    y_true_oos = []
    rbf_count = 0
    
    print("Running Nested CV...")
    for train_idx, test_idx in cv_outer.split(X_cv):
        X_tr, X_te = X_cv[train_idx], X_cv[test_idx]
        y_tr, y_te = y_cv[train_idx], y_cv[test_idx]
        y_true_oos.extend(y_te)
        
        for name, (pipe, grid) in models.items():
            gs = GridSearchCV(pipe, grid, cv=cv_inner, scoring='precision', n_jobs=-1)
            gs.fit(X_tr, y_tr)
            predictions[name].extend(gs.decision_function(X_te) if hasattr(gs, 'decision_function') else gs.predict_proba(X_te)[:, 1])
            
            if name == "SVM Full" and gs.best_params_['svm__kernel'] == 'rbf':
                rbf_count += 1
                
    y_true_oos = np.array(y_true_oos)
    total_oos = len(y_true_oos)
    base_rate = np.mean(y_true_oos)
    target_buys = int(total_oos * 0.64)  # ~1277
    print(f"\n--- OOS BASE RATE: {np.sum(y_true_oos)}/{total_oos} = {base_rate:.4%} ---")
    print(f"SVM Full selected RBF in {rbf_count}/5 folds\n")
    
    print("--- 64% MATCHED COVERAGE (Exactly 1277 Buys) ---")
    matched_preds = {}
    for name, scores in predictions.items():
        preds = exact_coverage(np.array(scores), target_buys)
        matched_preds[name] = preds
        tp = np.sum((preds == 1) & (y_true_oos == 1))
        prec = tp / target_buys
        lift = (prec - base_rate) * 100
        ci = single_bootstrap_lift(y_true_oos, preds, base_rate)
        print(f"{name:15} | TP: {tp}/1277 | Prec: {prec:.4f} | Lift: {lift:+.2f} pp | CI: [{ci[0]:+.2f}, {ci[1]:+.2f}]")

    print("\n--- PAIRED DIFFERENCES (95% CI) ---")
    ci_full = paired_bootstrap_diff(y_true_oos, matched_preds["SVM Full"], matched_preds["LR Full"])
    ci_raw = paired_bootstrap_diff(y_true_oos, matched_preds["SVM Raw"], matched_preds["LR Raw"])
    print(f"SVM Full - LR Full: [{ci_full[0]:.4f}, {ci_full[1]:.4f}]")
    print(f"SVM Raw  - LR Raw:  [{ci_raw[0]:.4f}, {ci_raw[1]:.4f}]")

    print("\n--- HOLDOUT BACKTEST ---")
    cost = 0.0005
    for name in ["LR Full", "SVM Full"]:
        pipe, grid = models[name]
        gs = GridSearchCV(pipe, grid, cv=cv_inner, scoring='precision', n_jobs=-1)
        gs.fit(X_cv, y_cv)
        
        cv_scores = gs.decision_function(X_cv) if hasattr(gs, 'decision_function') else gs.predict_proba(X_cv)[:, 1]
        threshold = np.percentile(cv_scores, 100 - 64)
        
        ho_scores = gs.decision_function(X_ho) if hasattr(gs, 'decision_function') else gs.predict_proba(X_ho)[:, 1]
        ho_preds = (ho_scores >= threshold).astype(int)
        
        exposure = np.mean(ho_preds)
        changes = np.abs(np.diff(np.insert(ho_preds, 0, 0)))
        strat_ret = ho_preds * rets_ho - (changes * cost)
        bench_ret = rets_ho * exposure
        
        ann_S = (1 + np.prod(1 + strat_ret) - 1) ** (252 / len(rets_ho)) - 1
        vol_S = np.std(strat_ret) * np.sqrt(252)
        sh_S = ann_S / vol_S
        max_dd_S = np.min((np.cumprod(1 + strat_ret) - np.maximum.accumulate(np.cumprod(1 + strat_ret))) / np.maximum.accumulate(np.cumprod(1 + strat_ret)))
        
        ann_B = (1 + np.prod(1 + bench_ret) - 1) ** (252 / len(rets_ho)) - 1
        vol_B = np.std(bench_ret) * np.sqrt(252)
        sh_B = ann_B / vol_B
        
        ci_ret, ci_sh = economic_bootstrap(strat_ret, bench_ret)
        
        print(f"\n{name} Holdout (Threshold frozen from CV)")
        print(f"Realized Exposure: {exposure:.2%} | Trades: {np.sum(changes)}")
        print(f"Strategy -> Ret: {ann_S:.2%} | Vol: {vol_S:.2%} | Sharpe: {sh_S:.2f} | MaxDD: {max_dd_S:.2%}")
        print(f"Bench({exposure:.0%}) -> Ret: {ann_B:.2%} | Vol: {vol_B:.2%} | Sharpe: {sh_B:.2f}")
        print(f"Paired Diff CI -> Return: [{ci_ret[0]:.2f}, {ci_ret[1]:.2f}] pp | Sharpe: [{ci_sh[0]:.2f}, {ci_sh[1]:.2f}]")

if __name__ == '__main__':
    run_master_audit()