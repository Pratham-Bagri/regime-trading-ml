import os

# Asset Definition
TICKER = "SPY"
START_DATE = "2014-01-01"
END_DATE = "2026-01-01"

# Directory Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", f"{TICKER.lower()}_raw.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", f"{TICKER.lower()}_features.csv")