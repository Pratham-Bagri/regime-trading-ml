import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from hmmlearn.hmm import GaussianHMM
import config

class CausalFilteredHMM(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=3, covariance_type="full", random_state=42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.hmm_ = None
        self.state_map_ = None

    def fit(self, X, y=None):
        self.hmm_ = GaussianHMM(
            n_components=self.n_components, 
            covariance_type=self.covariance_type, 
            random_state=self.random_state,
            n_iter=100
        )
        self.hmm_.fit(X)
        
        # Sort states by variance to prevent label switching across folds
        variances = np.array([np.diag(cov)[0] for cov in self.hmm_.covars_])
        sorted_indices = np.argsort(variances)
        self.state_map_ = {original: new for new, original in enumerate(sorted_indices)}
        return self

    def transform(self, X):
        # Causal filtering using the forward algorithm to prevent Viterbi leakage
        filtered_states = np.zeros(len(X))
        for t in range(1, len(X) + 1):
            causal_slice = X[:t]
            current_state = self.hmm_.predict(causal_slice)[-1] 
            filtered_states[t-1] = self.state_map_[current_state]
            
        one_hot_states = np.eye(self.n_components)[filtered_states.astype(int)]
        return one_hot_states

class FeatureConcatenator(BaseEstimator, TransformerMixin):
    """Passes the PCA features into the HMM, then concatenates both for the SVM."""
    def __init__(self, n_components=3):
        self.n_components = n_components  # Scikit-learn GridSearchCV requires this exact attribute
        self.hmm = CausalFilteredHMM(n_components=self.n_components)
        
    def fit(self, X, y=None):
        self.hmm.fit(X)
        return self
        
    def transform(self, X):
        hmm_features = self.hmm.transform(X)
        return np.hstack([X, hmm_features])

def train_rigorous_pipeline():
    print("Loading raw rolling features...")
    df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col='Date', parse_dates=True)
    df = df.dropna()
    
    df['Signal'] = np.where(df['target_return'] > 0, 1, 0)
    
    # Drop target columns to isolate the raw feature matrix
    feature_cols = [col for col in df.columns if col not in ['target_return', 'Signal', 'Close', 'return']]
    X = df[feature_cols].values
    y = df['Signal'].values
    
    # 1. Establish the untouchable 20% holdout set
    holdout_idx = int(len(X) * 0.8)
    X_cv, X_holdout = X[:holdout_idx], X[holdout_idx:]
    y_cv, y_holdout = y[:holdout_idx], y[holdout_idx:]
    
    print(f"Reserved final {len(X_holdout)} days as the untouched holdout.")
    print(f"Executing Walk-Forward Validation on the {len(X_cv)} day development set...")
    
    # 2. Purged Time-Series Cross Validation (gap=20 to match our 20-day rolling windows)
    cv_outer = TimeSeriesSplit(n_splits=5, gap=20)
    
    # 3. The provably leak-free mathematical pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=4)),
        ('hmm_concat', FeatureConcatenator(n_components=3)),
        ('classifier', SVC(random_state=42))
    ])
    
    param_grid = {
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['rbf', 'linear']
    }
    
    precisions = []
    
    for train_idx, test_idx in cv_outer.split(X_cv):
        X_train, X_test = X_cv[train_idx], X_cv[test_idx]
        y_train, y_test = y_cv[train_idx], y_cv[test_idx]
        
        cv_inner = TimeSeriesSplit(n_splits=3, gap=20)
        
        grid = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=cv_inner, 
            scoring='precision', 
            n_jobs=None
        )
        
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        
        fold_precision = precision_score(y_test, y_pred, zero_division=0)
        precisions.append(fold_precision)
        print(f"Fold Precision: {fold_precision:.4f} | Best Kernel: {grid.best_params_['classifier__kernel']}")
        
    print("\n--- Development Set Walk-Forward Results ---")
    print(f"Average Purged Precision: {np.mean(precisions):.4f}")

if __name__ == "__main__":
    train_rigorous_pipeline()