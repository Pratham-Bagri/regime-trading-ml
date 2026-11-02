import pandas as pd
import numpy as np
import config

def engineer_features():
    print(f"Loading raw data from {config.RAW_DATA_PATH}...")
    df = pd.read_csv(config.RAW_DATA_PATH, index_col='Date', parse_dates=True)
    
    # 1. Calculate Daily Returns
    df['return'] = df['Close'].pct_change()
    
    # 2. Target Variable (Next day's return)
    # This is what our model will try to predict
    df['target_return'] = df['return'].shift(-1)
    
    # 3. Raw Rolling Statistics (Maximum window = 20 days)
    # We do NOT scale or apply PCA here to prevent data leakage.
    df['roll_mean_5'] = df['return'].rolling(window=5).mean()
    df['roll_mean_20'] = df['return'].rolling(window=20).mean()
    df['roll_vol_5'] = df['return'].rolling(window=5).std()
    df['roll_vol_20'] = df['return'].rolling(window=20).std()
    
    # 4. Price Momentum
    df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    
    # Drop rows with NaN values created by the 20-day rolling windows and the 1-day target shift
    df.dropna(inplace=True)
    
    # Save the raw, unscaled feature matrix for the pipeline to ingest
    df.to_csv(config.PROCESSED_DATA_PATH)
    print(f"Engineered {len(df.columns) - 2} raw features.")
    print(f"Unscaled data saved to {config.PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    engineer_features()