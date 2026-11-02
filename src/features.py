import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import config

def generate_features():
    print("Loading raw data...")
    df = pd.read_csv(config.RAW_DATA_PATH, index_col='Date', parse_dates=True)
    
    # 1. Base Stationary Transformation (Ito's Lemma Logic)
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 2. Rolling Statistical Moments
    # 1st Moment: Trend (10 and 20 day moving average of returns)
    df['ret_ma_10'] = df['log_return'].rolling(window=10).mean()
    df['ret_ma_20'] = df['log_return'].rolling(window=20).mean()
    
    # 2nd Moment: Variance / Realized Volatility
    df['volatility_10'] = df['log_return'].rolling(window=10).var()
    df['volatility_20'] = df['log_return'].rolling(window=20).var()
    
    # 3rd Moment: Skewness (Tail Risk)
    df['skew_20'] = df['log_return'].rolling(window=20).skew()
    
    # 3. Clean up the data
    # Rolling windows create 'NaN' (Not a Number) for the first 20 days. We must drop them.
    df = df.dropna()
    
    # 4. Standardization (Crucial for PCA)
    # PCA is sensitive to scale. We force every feature to have mean=0 and variance=1.
    feature_cols = ['ret_ma_10', 'ret_ma_20', 'volatility_10', 'volatility_20', 'skew_20']
    X_raw = df[feature_cols].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # 5. Dimensionality Reduction (PCA)
    # We ask PCA to retain exactly enough components to explain 95% of the variance
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"PCA reduced {len(feature_cols)} correlated features down to {X_pca.shape[1]} orthogonal components.")
    
    # 6. Save the Processed Feature Space
    # Create a new dataframe with our PCA components and our target variable (tomorrow's return)
    pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    df_processed = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
    
    # We map tomorrow's return today so the supervised model has a target to predict
    df_processed['target_return'] = df['log_return'].shift(-1)
    df_processed = df_processed.dropna()
    
    os.makedirs(os.path.dirname(config.PROCESSED_DATA_PATH), exist_ok=True)
    df_processed.to_csv(config.PROCESSED_DATA_PATH)
    print(f"Feature Space successfully built and saved to {config.PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    generate_features()