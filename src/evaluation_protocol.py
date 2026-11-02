import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import config

# Import custom mathematical classes
from trading_model import CausalFilteredHMM, FeatureConcatenator

def run_nested_cv_evaluation():
    print("Loading raw rolling features...")
    df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col='Date', parse_dates=True)
    df = df.dropna()
    
    df['Signal'] = np.where(df['target_return'] > 0, 1, 0)
    feature_cols = [col for col in df.columns if col not in ['target_return', 'Signal', 'Close', 'return']]
    X = df[feature_cols].values
    y = df['Signal'].values
    
    # Isolate the untouchable 20% holdout set
    holdout_idx = int(len(X) * 0.8)
    X_cv, y_cv = X[:holdout_idx], y[:holdout_idx]
    
    # Outer loop for evaluation, inner loop for tuning (gap=20 embargo)
    cv_outer = TimeSeriesSplit(n_splits=5, gap=20)
    
    # Evaluate models on IDENTICAL features and strict outer folds
    models = {
        "1. Logistic Regression (Raw Features)": {
            "pipe": Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=1000, random_state=42))]),
            "params": {'lr__C': [0.01, 0.1, 1, 10]}
        },
        "2. SVM (Raw Features)": {
            "pipe": Pipeline([('scaler', StandardScaler()), ('svm', SVC(random_state=42))]),
            "params": {'svm__C': [0.1, 1, 10], 'svm__kernel': ['linear', 'rbf']}
        },
        "3. Logistic Regression (Full Pipeline)": {
            "pipe": Pipeline([
                ('scaler', StandardScaler()), 
                ('pca', PCA(n_components=4)), 
                ('hmm_concat', FeatureConcatenator(n_components=3)), 
                ('lr', LogisticRegression(max_iter=1000, random_state=42))
            ]),
            "params": {'lr__C': [0.01, 0.1, 1, 10]}
        },
        "4. SVM (Full Pipeline)": {
            "pipe": Pipeline([
                ('scaler', StandardScaler()), 
                ('pca', PCA(n_components=4)), 
                ('hmm_concat', FeatureConcatenator(n_components=3)), 
                ('svm', SVC(random_state=42))
            ]),
            "params": {'svm__C': [0.1, 1, 10], 'svm__kernel': ['linear', 'rbf']}
        },
        "5. Random Forest (Raw Features)": {
            "pipe": Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(random_state=42))]),
            "params": {'rf__max_depth': [3, 5, None], 'rf__n_estimators': [50, 100]}
        }
    }
    
    print("Executing strict Nested Cross-Validation (Outer=5, Inner=3)...\n")
    
    # Collect predictions to calculate a single global Out-Of-Sample (OOS) precision
    global_y_true = []
    
    for name, config_dict in models.items():
        all_y_true = []
        all_y_pred = []
        
        for train_idx, test_idx in cv_outer.split(X_cv):
            X_train, X_test = X_cv[train_idx], X_cv[test_idx]
            y_train, y_test = y_cv[train_idx], y_cv[test_idx]
            
            cv_inner = TimeSeriesSplit(n_splits=3, gap=20)
            
            grid = GridSearchCV(
                config_dict["pipe"], 
                config_dict["params"], 
                cv=cv_inner, 
                scoring='precision', 
                n_jobs=None
            )
            
            # Fit on inner loop, predict strictly on outer test fold
            grid.fit(X_train, y_train)
            y_pred = grid.predict(X_test)
            
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
        
        # Save true labels once to use for the naive baseline comparison
        if not global_y_true:
            global_y_true = all_y_true
            
        oos_precision = precision_score(all_y_true, all_y_pred, zero_division=0)
        coverage = np.sum(all_y_pred) / len(all_y_pred)
        
        print(f"{name} -> OOS Precision: {oos_precision:.4f} | Coverage: {coverage:.2%}")
        
    # Match naive baseline exposure to roughly 64% (LR's previous coverage)
    np.random.seed(42)
    random_preds = np.random.binomial(1, 0.64, size=len(global_y_true))
    random_precision = precision_score(global_y_true, random_preds, zero_division=0)
    print(f"6. Naive Random Baseline (64% Coverage) -> OOS Precision: {random_precision:.4f} | Coverage: {np.sum(random_preds)/len(random_preds):.2%}")

if __name__ == '__main__':
    run_nested_cv_evaluation()