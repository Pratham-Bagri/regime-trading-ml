# Quantitative Finance: Market Regime Detection & Predictive Trading Pipeline

## Overview
This project is a rigorous, end-to-end machine learning pipeline designed to detect latent market regimes and predict the binary directional movement of the S&P 500 (SPY). 

Unlike many beginner machine learning projects that suffer from extreme data leakage (Look-Ahead Bias) and unrealistic accuracy metrics, this system was engineered adhering strictly to professional quantitative finance standards. It implements strict Walk-Forward Validation, nested cross-validation for hyperparameter tuning, and prioritizes capital preservation (Precision) over raw Accuracy.

## The Mathematical Pipeline

The architecture is divided into five distinct mathematical and computational stages:

### 1. Feature Engineering (Stochastic Calculus Logic)
*   **Stationarity:** Transformed raw, non-stationary OHLCV price data into a stationary series of log-returns (inspired by Ito's Lemma).
*   **Statistical Moments:** Extracted rolling 10-day and 20-day statistical moments, including Trend (Moving Averages), Realized Volatility (Variance), and Tail Risk (Skewness).

### 2. Dimensionality Reduction (PCA)
*   Applied Principal Component Analysis (PCA) to the standardized feature space to eliminate multicollinearity.
*   Successfully reduced the feature space to **4 orthogonal principal components** while retaining 95% of the market's behavioral variance.

### 3. Unsupervised Sequence Modeling (HMM)
*   Implemented a **Gaussian Hidden Markov Model (HMM)** using the Expectation-Maximization (EM) algorithm to detect $K=3$ latent market regimes (e.g., Bull, Bear, Sideways).
*   Utilized full covariance matrices to capture elliptical boundary distributions across the 4-dimensional PCA space.

### 4. Supervised Predictive Trading (Non-Linear SVM)
*   Appended the decoded HMM regime states (via One-Hot Encoding) to the PCA feature space.
*   Trained a **Support Vector Machine (SVM)** with a Radial Basis Function (RBF) kernel to classify the next day's market movement (Buy/Sell).
*   Utilized **Grid Search** to mathematically optimize the regularization (`C`) and kernel coefficient (`gamma`) to maximize trading Precision.

### 5. Design of Experiments (Preventing Data Leakage)
*   **Walk-Forward Validation:** Completely avoided random K-Fold cross-validation (which causes Look-Ahead Bias). Evaluated the model using strict chronological `TimeSeriesSplit`.
*   **Nested Cross-Validation:** Confined the Grid Search strictly inside the training folds to prevent Optimism Bias and ensure the testing data remained completely unseen.

## Repository Architecture

The project is structured using professional software engineering practices, ensuring modularity, reproducibility, and a clean Git version control history (utilizing Conventional Commits like `feat`, `fix`, and `chore`).

```text
regime-trading-ml/
├── data/
│   ├── raw/               # Immutable raw data from yfinance
│   └── processed/         # Feature space and trained .pkl models
├── src/
│   ├── config.py          # Centralized configuration and pathing
│   ├── data_ingestion.py  # API extraction and formatting
│   ├── features.py        # Statistical moments and PCA
│   ├── regime_model.py    # Unsupervised Gaussian HMM
│   └── trading_model.py   # Supervised non-linear SVM and evaluation
├── .gitignore
├── requirements.txt       # Frozen dependencies for exact reproducibility
└── README.md
```

## Rigorous Conclusions & Findings

*   **Optimized Metrics:** The final tuned SVM achieved an Average Recall of **~87%** and an Average Precision of **~55%**. 
*   **Market Efficiency:** The mathematical limit reached by this model validates the Efficient Market Hypothesis (EMH). By rigorously preventing data leakage, we proved that basic daily OHLCV data (even when subjected to advanced PCA and HMM transformations) does not contain sufficient signal-to-noise ratio to dramatically outperform the baseline upward drift of the market (~54%).
*   **Real-World Application:** A model displaying 80%+ precision on basic daily price data is almost certainly mathematically compromised by Look-Ahead Bias. This project serves as a mathematically honest, structurally sound foundation. To achieve production-grade alpha (>65% Precision), this exact algorithmic engine can now be fed higher-quality alternative data (e.g., limit order book dynamics, cross-asset correlations, or sentiment analysis).

## How to Run

1. Clone the repository and navigate to the root directory.
2. Initialize and activate a virtual environment (`python -m venv venv`).
3. Install the locked dependencies: `pip install -r requirements.txt`.
4. Execute the pipeline sequentially:
   ```bash
   python src/data_ingestion.py
   python src/features.py
   python src/regime_model.py
   python src/trading_model.py
   ```