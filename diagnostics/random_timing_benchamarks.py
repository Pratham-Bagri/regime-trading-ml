import pandas as pd
import numpy as np
import config

def random_timing_distribution(rets, num_days, iterations=10000):
    n = len(rets)
    anns, vols, sharpes, maxdds = [], [], [], []
    
    np.random.seed(42) # For reproducibility
    for _ in range(iterations):
        # Exact matching of in-market days to ensure exact same exposure
        preds = np.zeros(n)
        preds[np.random.choice(n, num_days, replace=False)] = 1
        
        strat_ret = preds * rets
        
        cum_ret = np.prod(1 + strat_ret) - 1
        ann = (1 + cum_ret) ** (252 / n) - 1
        vol = np.std(strat_ret) * np.sqrt(252)
        sharpe = ann / vol if vol != 0 else 0
        
        cum_idx = np.cumprod(1 + strat_ret)
        rmax = np.maximum.accumulate(cum_idx)
        # Avoid division by zero on the first day if returns are 0
        dd = np.where(rmax > 0, (cum_idx - rmax) / rmax, 0)
        mdd = np.min(dd)
        
        anns.append(ann)
        vols.append(vol)
        sharpes.append(sharpe)
        maxdds.append(mdd)
        
    return {
        'Return': (np.percentile(anns, 2.5), np.percentile(anns, 97.5)),
        'Vol': (np.percentile(vols, 2.5), np.percentile(vols, 97.5)),
        'Sharpe': (np.percentile(sharpes, 2.5), np.percentile(sharpes, 97.5)),
        'MaxDD': (np.percentile(maxdds, 2.5), np.percentile(maxdds, 97.5))
    }

def run_random_benchmarks():
    print("Loading holdout returns...")
    df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col='Date', parse_dates=True).dropna()
    feature_cols = [c for c in df.columns if c not in ['target_return', 'Signal', 'Close', 'return']]
    X = df[feature_cols].values
    rets = df['target_return'].values
    
    holdout_idx = int(len(X) * 0.8)
    rets_ho = rets[holdout_idx:]
    
    print(f"Total Holdout Days: {len(rets_ho)}\n")
    
    print("--- LR Full Random Timing Benchmark (499 Days, 83.17%) ---")
    lr_bench = random_timing_distribution(rets_ho, 499)
    print(f"Return CI: [{lr_bench['Return'][0]:.2%}, {lr_bench['Return'][1]:.2%}]")
    print(f"Vol CI:    [{lr_bench['Vol'][0]:.2%}, {lr_bench['Vol'][1]:.2%}]")
    print(f"Sharpe CI: [{lr_bench['Sharpe'][0]:.2f}, {lr_bench['Sharpe'][1]:.2f}]")
    print(f"Max DD CI: [{lr_bench['MaxDD'][0]:.2%}, {lr_bench['MaxDD'][1]:.2%}]\n")

    print("--- SVM Full Random Timing Benchmark (490 Days, 81.67%) ---")
    svm_bench = random_timing_distribution(rets_ho, 490)
    print(f"Return CI: [{svm_bench['Return'][0]:.2%}, {svm_bench['Return'][1]:.2%}]")
    print(f"Vol CI:    [{svm_bench['Vol'][0]:.2%}, {svm_bench['Vol'][1]:.2%}]")
    print(f"Sharpe CI: [{svm_bench['Sharpe'][0]:.2f}, {svm_bench['Sharpe'][1]:.2f}]")
    print(f"Max DD CI: [{svm_bench['MaxDD'][0]:.2%}, {svm_bench['MaxDD'][1]:.2%}]")

if __name__ == '__main__':
    run_random_benchmarks()