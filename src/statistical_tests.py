import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from statsmodels.stats.contingency_tables import mcnemar
import config

def block_bootstrap_precision(y_true, y_pred, block_size=20, iterations=1000):
    """Generates confidence intervals using a stationary block bootstrap."""
    n = len(y_true)
    precisions = []
    
    for _ in range(iterations):
        indices = []
        while len(indices) < n:
            start = np.random.randint(0, n - block_size)
            indices.extend(range(start, start + block_size))
        
        indices = indices[:n]
        boot_true = y_true[indices]
        boot_pred = y_pred[indices]
        
        if np.sum(boot_pred) > 0:
            precisions.append(precision_score(boot_true, boot_pred, zero_division=0))
            
    return np.percentile(precisions, [2.5, 97.5])

def run_statistical_rigor():
    print("Loading raw features for paired testing...")
    df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col='Date', parse_dates=True)
    df = df.dropna()
    
    df['Signal'] = np.where(df['target_return'] > 0, 1, 0)
    feature_cols = [col for col in df.columns if col not in ['target_return', 'Signal', 'Close', 'return']]
    X = df[feature_cols].values
    y = df['Signal'].values
    
    holdout_idx = int(len(X) * 0.8)
    X_cv, y_cv = X[:holdout_idx], y[:holdout_idx]
    
    cv_outer = TimeSeriesSplit(n_splits=5, gap=20)
    
    # We will test the raw feature pipelines to isolate the model architecture
    lr_pipe = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=1000, random_state=42))])
    svm_pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(random_state=42))])
    
    lr_params = {'lr__C': [0.01, 0.1, 1, 10]}
    svm_params = {'svm__C': [0.1, 1, 10], 'svm__kernel': ['linear', 'rbf']}
    
    global_y_true, lr_preds, svm_preds = [], [], []
    
    print("Generating matched predictions for McNemar's Paired Test...")
    for train_idx, test_idx in cv_outer.split(X_cv):
        X_train, X_test = X_cv[train_idx], X_cv[test_idx]
        y_train, y_test = y_cv[train_idx], y_cv[test_idx]
        
        cv_inner = TimeSeriesSplit(n_splits=3, gap=20)
        
        grid_lr = GridSearchCV(lr_pipe, lr_params, cv=cv_inner, scoring='precision', n_jobs=None)
        grid_svm = GridSearchCV(svm_pipe, svm_params, cv=cv_inner, scoring='precision', n_jobs=None)
        
        grid_lr.fit(X_train, y_train)
        grid_svm.fit(X_train, y_train)
        
        lr_preds.extend(grid_lr.predict(X_test))
        svm_preds.extend(grid_svm.predict(X_test))
        global_y_true.extend(y_test)
        
    global_y_true = np.array(global_y_true)
    lr_preds = np.array(lr_preds)
    svm_preds = np.array(svm_preds)
    
    print("\n--- Block Bootstrap Confidence Intervals (95%) ---")
    lr_ci = block_bootstrap_precision(global_y_true, lr_preds)
    svm_ci = block_bootstrap_precision(global_y_true, svm_preds)
    print(f"Logistic Regression 95% CI: [{lr_ci[0]:.4f}, {lr_ci[1]:.4f}]")
    print(f"SVM 95% CI:                 [{svm_ci[0]:.4f}, {svm_ci[1]:.4f}]")
    
    print("\n--- McNemar's Paired Test (LR vs SVM) ---")
    # Contingency table for McNemar's test
    # [[Both Correct, SVM Correct / LR Wrong], [LR Correct / SVM Wrong, Both Wrong]]
    lr_correct = (lr_preds == global_y_true)
    svm_correct = (svm_preds == global_y_true)
    
    table = [
        [np.sum(lr_correct & svm_correct), np.sum(~lr_correct & svm_correct)],
        [np.sum(lr_correct & ~svm_correct), np.sum(~lr_correct & ~svm_correct)]
    ]
    
    result = mcnemar(table, exact=False, correction=True)
    print(f"P-Value for architecture difference: {result.pvalue:.4f}")
    if result.pvalue < 0.05:
        print("Conclusion: There is a statistically significant difference between the models.")
    else:
        print("Conclusion: NO statistically significant difference. They perform identically on out-of-sample data.")

if __name__ == '__main__':
    run_statistical_rigor()