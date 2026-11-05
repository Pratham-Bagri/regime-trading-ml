import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit
from trading_model import FeatureConcatenator
import config
import warnings

warnings.filterwarnings('ignore')

def run_diagnosis():
    print("Loading features for holdout diagnosis...")
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
    dates = df.index
    
    holdout_idx = int(len(X) * 0.8)
    X_cv, y_cv = X[:holdout_idx], y[:holdout_idx]
    X_ho, y_ho = X[holdout_idx:], y[holdout_idx:]
    dates_ho = dates[holdout_idx:]
    
    cv_outer = TimeSeriesSplit(n_splits=5, gap=20)
    
    print("Generating pooled OOF scores for SVM Full...")
    np.random.seed(42)
    pipe = Pipeline([
        ('s', StandardScaler()), 
        ('pca', PCA(n_components=4, random_state=42)), 
        ('hmm', FeatureConcatenator(n_components=3)), 
        ('svm', SVC(C=1, kernel='rbf', random_state=42))
    ])
    
    oof_scores = []
    for train_idx, test_idx in cv_outer.split(X_cv):
        np.random.seed(42 + test_idx[0])
        pipe.fit(X_cv[train_idx], y_cv[train_idx])
        oof_scores.extend(pipe.decision_function(X_cv[test_idx]))
    
    oof_scores = np.array(oof_scores)
    static_threshold = np.percentile(oof_scores, 100 - 64)
    
    print("Fitting final model on all CV data and generating Holdout scores...")
    np.random.seed(42)
    pipe.fit(X_cv, y_cv)
    ho_scores = pipe.decision_function(X_ho)
    
    print("\n--- DIAGNOSIS: DISTRIBUTION SHIFT ---")
    print(f"OOF Scores - Mean: {np.mean(oof_scores):.4f} | Std: {np.std(oof_scores):.4f}")
    print(f"HO Scores  - Mean: {np.mean(ho_scores):.4f} | Std: {np.std(ho_scores):.4f}")
    print(f"Static OOF Threshold (64% target): {static_threshold:.4f}")
    
    ho_static_exposure = np.mean(ho_scores >= static_threshold)
    print(f"Holdout Exposure under Static Threshold: {ho_static_exposure:.2%}")
    
    # Monthly Exposure Analysis
    ho_df = pd.DataFrame({'Score': ho_scores}, index=dates_ho)
    ho_df['Static_Invested'] = (ho_df['Score'] >= static_threshold).astype(int)
    monthly_exp = ho_df['Static_Invested'].resample('ME').mean() * 100
    
    print("\n--- MONTHLY EXPOSURE (STATIC) ---")
    for date, exp in monthly_exp.dropna().items():
        print(f"{date.strftime('%Y-%m')}: {exp:5.1f}%")
        
    # Causal Alternative: Trailing Window Quantile
    all_scores = np.concatenate([oof_scores, ho_scores])
    trailing_window = 120 # ~6 months
    dynamic_invested = []
    
    for i in range(len(ho_scores)):
        idx = len(oof_scores) + i
        window = all_scores[idx - trailing_window : idx]
        thresh = np.percentile(window, 100 - 64)
        dynamic_invested.append(1 if ho_scores[i] >= thresh else 0)
        
    dynamic_exposure = np.mean(dynamic_invested)
    print(f"\n--- TRAILING CAUSAL THRESHOLD (N={trailing_window}) ---")
    print(f"Holdout Exposure: {dynamic_exposure:.2%}")

if __name__ == '__main__':
    run_diagnosis()