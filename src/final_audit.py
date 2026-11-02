import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import config
from trading_model import CausalFilteredHMM, FeatureConcatenator

def match_coverage(y_scores, target_coverage=0.64):
    """Dynamically scales threshold to ensure exactly target_coverage % of days are traded."""
    threshold = np.percentile(y_scores, 100 - (target_coverage * 100))
    return (y_scores >= threshold).astype(int)

def block_bootstrap_paired(y_true, y_pred_A, y_pred_B, block_size=20, iterations=5000):
    """Bootstraps the precision difference (A - B) on identical test days."""
    n = len(y_true)
    diffs = []
    
    for _ in range(iterations):
        indices = []
        while len(indices) < n:
            start = np.random.randint(0, n - block_size)
            indices.extend(range(start, start + block_size))
        indices = indices[:n]
        
        boot_true = y_true[indices]
        pred_A = y_pred_A[indices]
        pred_B = y_pred_B[indices]
        
        # Avoid zero division if a bootstrap sample happens to have 0 predictions
        if np.sum(pred_A) > 0 and np.sum(pred_B) > 0:
            prec_A = np.sum((pred_A == 1) & (boot_true == 1)) / np.sum(pred_A)
            prec_B = np.sum((pred_B == 1) & (boot_true == 1)) / np.sum(pred_B)
            diffs.append(prec_A - prec_B)
            
    return np.percentile(diffs, [2.5, 97.5])

def run_final_audit():
    print("--- 1. DATE & ARITHMETIC AUDIT ---")
    df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col='Date', parse_dates=True).dropna()
    df['Signal'] = np.where(df['target_return'] > 0, 1, 0)
    
    feature_cols = [col for col in df.columns if col not in ['target_return', 'Signal', 'Close', 'return']]
    X, y = df[feature_cols].values, df['Signal'].values
    
    holdout_idx = int(len(X) * 0.8)
    df_cv, df_holdout = df.iloc[:holdout_idx], df.iloc[holdout_idx:]
    X_cv, y_cv = X[:holdout_idx], y[:holdout_idx]
    
    print(f"Total Dataset: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} days)")
    print(f"CV Set: {df_cv.index[0].date()} to {df_cv.index[-1].date()} ({len(df_cv)} days)")
    print(f"Holdout Set: {df_holdout.index[0].date()} to {df_holdout.index[-1].date()} ({len(df_holdout)} days)")
    
    cv_outer = TimeSeriesSplit(n_splits=5, gap=20)
    test_days_total = sum(len(test) for _, test in cv_outer.split(X_cv))
    print(f"Total Outer OOS Test Days: {test_days_total}")
    
    print("\n--- 2. MATCHED COVERAGE (64%) EVALUATION ---")
    base_rate = np.mean(y_cv)
    print(f"Base Rate (Expected Chance Precision): {base_rate:.4f}\n")
    
    lr_raw = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(random_state=42))])
    svm_raw = Pipeline([('scaler', StandardScaler()), ('svm', SVC(random_state=42))])
    
    # We will use decision_function to match coverage manually
    lr_scores_all, svm_scores_all, y_true_all = [], [], []
    
    for train_idx, test_idx in cv_outer.split(X_cv):
        X_train, X_test = X_cv[train_idx], X_cv[test_idx]
        y_train, y_test = y_cv[train_idx], y_cv[test_idx]
        
        lr_raw.fit(X_train, y_train)
        svm_raw.fit(X_train, y_train)
        
        lr_scores_all.extend(lr_raw.decision_function(X_test))
        svm_scores_all.extend(svm_raw.decision_function(X_test))
        y_true_all.extend(y_test)
        
    y_true_all = np.array(y_true_all)
    lr_preds_matched = match_coverage(np.array(lr_scores_all), 0.64)
    svm_preds_matched = match_coverage(np.array(svm_scores_all), 0.64)
    
    lr_prec = np.sum((lr_preds_matched == 1) & (y_true_all == 1)) / np.sum(lr_preds_matched)
    svm_prec = np.sum((svm_preds_matched == 1) & (y_true_all == 1)) / np.sum(svm_preds_matched)
    
    print(f"LR (Raw)  -> Precision: {lr_prec:.4f} | Lift over Base: {(lr_prec - base_rate)*100:+.2f} pp")
    print(f"SVM (Raw) -> Precision: {svm_prec:.4f} | Lift over Base: {(svm_prec - base_rate)*100:+.2f} pp")
    
    ci_diff = block_bootstrap_paired(y_true_all, svm_preds_matched, lr_preds_matched, iterations=5000)
    print(f"Paired Block Bootstrap CI (SVM - LR) at 64% coverage: [{ci_diff[0]:.4f}, {ci_diff[1]:.4f}]")

    print("\n--- 3. HOLDOUT ECONOMIC BACKTEST ---")
    returns_holdout = df_holdout['target_return'].values
    trading_days = 252
    
    def calc_metrics(ret_array):
        cum_ret = np.prod(1 + ret_array) - 1
        ann_ret = (1 + cum_ret) ** (trading_days / len(ret_array)) - 1
        vol = np.std(ret_array) * np.sqrt(trading_days)
        sharpe = ann_ret / vol if vol != 0 else 0
        return ann_ret, sharpe
        
    # Baseline 1: Buy and Hold (100% Exposure)
    bnh_ann, bnh_sharpe = calc_metrics(returns_holdout)
    
    # Baseline 2: Constant 64% Exposure (Beta-adjusted B&H)
    bnh_64_ann, bnh_64_sharpe = calc_metrics(returns_holdout * 0.64)
    
    # Baseline 3: Random Timing (1000 draws of 64% coverage)
    np.random.seed(42)
    rand_anns, rand_sharpes = [], []
    for _ in range(1000):
        rand_preds = np.random.binomial(1, 0.64, size=len(returns_holdout))
        rand_ret = rand_preds * returns_holdout
        a, s = calc_metrics(rand_ret)
        rand_anns.append(a)
        rand_sharpes.append(s)
        
    print(f"Buy & Hold (100%):     Return {bnh_ann:.2%} | Sharpe {bnh_sharpe:.2f}")
    print(f"Buy & Hold ( 64%):     Return {bnh_64_ann:.2%} | Sharpe {bnh_64_sharpe:.2f}")
    print(f"Random Timing (64%):   Return CI [{np.percentile(rand_anns, 2.5):.2%}, {np.percentile(rand_anns, 97.5):.2%}] | Sharpe CI [{np.percentile(rand_sharpes, 2.5):.2f}, {np.percentile(rand_sharpes, 97.5):.2f}]")

if __name__ == '__main__':
    run_final_audit()