import yfinance as yf
import pandas as pd
import os
import config

def download_raw_data():
    print(f"Downloading {config.TICKER} data from {config.START_DATE} to {config.END_DATE}...")
    
    # Download data using yfinance
    df = yf.download(config.TICKER, start=config.START_DATE, end=config.END_DATE)

    # Flatten the MultiIndex columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure the raw directory exists
    os.makedirs(os.path.dirname(config.RAW_DATA_PATH), exist_ok=True)
    
    # Save purely as a CSV to freeze the state
    df.to_csv(config.RAW_DATA_PATH)
    print(f"Raw data successfully saved to {config.RAW_DATA_PATH}")

if __name__ == "__main__":
    download_raw_data()