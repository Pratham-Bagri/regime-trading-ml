import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import binomtest
import config

def evaluate_winning_model():
    print("Loading raw features for winning Logistic Regression model...")
    df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col='Date', parse_dates=True)
    df = df.dropna()
    
    df['Signal'] = np.where(df['target_return'] > 0, 1, 0)
    feature_cols = [col for col in df.columns if col not in ['target_return', 'Signal', 'Close', 'return']]
    X = df[feature_cols].values
    y = df['Signal'].values
    
    holdout_idx = int(len(X) * 0.8)
    X_cv, y_cv = X[:holdout_idx], y[:holdout_idx]
    
    # Calculate the Base Rate (Market Drift)
    base_rate = np.mean(y_cv)
    print(f"Market Base Rate (Always Buy): {base_rate:.4f}\n")
    
    cv_outer = TimeSeriesSplit(n_splits=5, gap=20)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        # Using C=0.1 as an example optimized parameter from previous tuning
        ('lr', LogisticRegression(C=0.1, random_state=42, max_iter=1000)) 
    ])
    
    all_y_true = []
    all_y_prob = []
    
    for train_idx, test_idx in cv_outer.split(X_cv):
        X_train, X_test = X_cv[train_idx], X_cv[test_idx]
        y_train, y_test = y_cv[train_idx], y_cv[test_idx]
        
        pipeline.fit(X_train, y_train)
        
        # Get probabilities instead of hard 0/1 labels
        probs = pipeline.predict_proba(X_test)[:, 1] 
        
        all_y_true.extend(y_test)
        all_y_prob.extend(probs)
        
    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    
    print("--- Precision vs. Coverage Threshold Sweep ---")
    thresholds = np.arange(0.50, 0.65, 0.02)
    
    best_threshold = 0.50
    best_precision = 0.0
    total_predictions_at_best = 0
    correct_predictions_at_best = 0
    
    for thresh in thresholds:
        preds = (all_y_prob >= thresh).astype(int)
        
        total_predicted_positive = np.sum(preds)
        if total_predicted_positive == 0:
            continue
            
        true_positives = np.sum((preds == 1) & (all_y_true == 1))
        precision = true_positives / total_predicted_positive
        coverage = total_predicted_positive / len(all_y_true)
        
        print(f"Threshold: {thresh:.2f} | Precision: {precision:.4f} | Coverage: {coverage:.2%} ({total_predicted_positive} trades)")
        
        if thresh == 0.50:
            best_precision = precision
            total_predictions_at_best = total_predicted_positive
            correct_predictions_at_best = true_positives
            
    # Statistical Significance Test (Binomial Test)
    print("\n--- Statistical Significance (Binomial Test) ---")
    # Testing if our standard 0.50 threshold model is significantly better than the base rate
    p_value = binomtest(correct_predictions_at_best, total_predictions_at_best, base_rate, alternative='greater').pvalue
    print(f"P-Value against Base Rate: {p_value:.4f}")
    if p_value < 0.05:
        print("Result: STATISTICALLY SIGNIFICANT edge over the market.")
    else:
        print("Result: NO STATISTICALLY SIGNIFICANT edge. The performance is within random market noise.")

if __name__ == "__main__":
    evaluate_winning_model()