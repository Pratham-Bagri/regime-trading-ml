import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit
from trading_model import CausalFilteredHMM, FeatureConcatenator
import config
import warnings

warnings.filterwarnings('ignore')

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

def run_full_causal_backtest():
    print("Loading and strictly normalizing features (OBV, MACD, ATR)...")
    df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col='Date', parse_dates=True).dropna()
    
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
    
    models = {
        "LR Full": Pipeline([
            ('s', StandardScaler()), 
            ('pca', PCA(n_components=4)), 
            ('hmm', FeatureConcatenator(n_components=3)), 
            ('lr', LogisticRegression(C=1, max_iter=1000))
        ]),
        "SVM Full": Pipeline([
            ('s', StandardScaler()), 
            ('pca', PCA(n_components=4)), 
            ('hmm', FeatureConcatenator(n_components=3)), 
            ('svm', SVC(C=1, kernel='rbf', probability=True))
        ])
    }
    
    cv_inner = TimeSeriesSplit(n_splits=5, gap=20)
    
    for name, pipeline in models.items():
        print(f"\n--- Processing {name} ---")
        print("Generating pooled out-of-fold scores to seed causal threshold...")
        oof_scores = []
        for train_idx, test_idx in cv_inner.split(X_cv):
            pipeline.fit(X_cv[train_idx], y_cv[train_idx])
            scores = pipeline.predict_proba(X_cv[test_idx])[:, 1]
            oof_scores.extend(scores)
            
        print("Refitting walk-forward through the holdout set (this may take a minute)...")
        historical_scores = list(oof_scores)
        predictions = []
        
        refit_step = 20
        for i in range(len(X_ho)):
            if i % refit_step == 0:
                X_train = np.vstack((X_cv, X_ho[:i])) if i > 0 else X_cv
                y_train = np.concatenate((y_cv, y_ho[:i])) if i > 0 else y_cv
                pipeline.fit(X_train, y_train)
                
            score = pipeline.predict_proba(X_ho[i:i+1])[:, 1][0]
            
            # Causal expanding quantile mapped to 64% coverage target
            threshold = np.percentile(historical_scores, 100 - 64)
            pred = int(score >= threshold)
            
            predictions.append(pred)
            historical_scores.append(score)
            
        predictions = np.array(predictions)
        in_market_days = np.sum(predictions)
        exposure = in_market_days / len(rets_ho)
        
        trades = np.sum(np.abs(np.diff(np.insert(predictions, 0, 0))))
        strat_ret = predictions * rets_ho - (np.abs(np.diff(np.insert(predictions, 0, 0))) * 0.0005)
        bench_ret = rets_ho * exposure
        
        ann_S = (1 + np.prod(1 + strat_ret) - 1) ** (252 / len(rets_ho)) - 1
        vol_S = np.std(strat_ret) * np.sqrt(252)
        sh_S = ann_S / vol_S if vol_S != 0 else 0
        max_dd_S = np.min((np.cumprod(1 + strat_ret) - np.maximum.accumulate(np.cumprod(1 + strat_ret))) / np.maximum.accumulate(np.cumprod(1 + strat_ret)))
        
        ann_B = (1 + np.prod(1 + bench_ret) - 1) ** (252 / len(rets_ho)) - 1
        vol_B = np.std(bench_ret) * np.sqrt(252)
        sh_B = ann_B / vol_B if vol_B != 0 else 0
        
        print("Bootstrapping confidence intervals...")
        ci_ret, ci_sh = economic_bootstrap(strat_ret, bench_ret)
        
        print(f"\n{name} Holdout Metrics:")
        print(f"In-Market Days: {in_market_days}/600 | Realized Exposure: {exposure:.2%} | Trades: {trades}")
        print(f"Strategy -> Ret: {ann_S:.2%} | Vol: {vol_S:.2%} | Sharpe: {sh_S:.2f} | MaxDD: {max_dd_S:.2%}")
        print(f"Bench({exposure:.0%}) -> Ret: {ann_B:.2%} | Vol: {vol_B:.2%} | Sharpe: {sh_B:.2f}")
        print(f"Paired Diff CI -> Return: [{ci_ret[0]:.2f}, {ci_ret[1]:.2f}] pp | Sharpe: [{ci_sh[0]:.2f}, {ci_sh[1]:.2f}]")

if __name__ == '__main__':
    run_full_causal_backtest()