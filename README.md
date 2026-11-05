# Market Regime Detection and Predictive Trading
**An Evaluation of Non-Linear Pipelines vs. Market Drift**

## The Question
Can a complex non-linear regime-switching pipeline (PCA + Gaussian HMM + RBF-SVM) generate a statistically significant precision lift over a simple linear baseline (Logistic Regression) at predicting the daily direction of the S&P 500? 

## The Verdict
**No.** Using an expanding-window nested cross-validation protocol over 2,997 trading days (2014–2025), the primary paired comparison (SVM Full minus LR Full) yielded a precision difference of **-1.02 percentage points** (95% CI [-3.18, +1.17] pp, $p = 0.381$). 

Rather than a failure of the model, this null result highlights the mathematical reality of daily directional trading:
* **The Edge is Bounded:** The CI upper bound mathematically proves that the data is incompatible with the non-linear pipeline beating the linear baseline by more than 1.2 pp. 
* **The Power Limit:** Because the empirical lift-versus-base Standard Error (SE) is $\sim$0.84 pp, detecting a realistic sub-1% edge with 80% power requires **~40+ years of daily data** (~11,000 test days).
* **The Illusion of Alpha:** An exploratory SVM Raw model yielded a +2.02 pp lift (unadjusted $p=0.019$). In a naive setup, this would be published as a success. However, after applying a 10,000-resample Romano-Wolf max-statistic block bootstrap to control the Family-Wise Error Rate (FWER), the adjusted $p$-value rose to 0.116. The alpha was a statistical illusion.

## Methodological Rigor & Leakage Prevention
This project diverges from standard retail ML projects by utilizing an aggressively controlled evaluation framework:
* **Temporal Leakage Prevention:** Employed an expanding-window nested CV with a strict 20-day embargo gap between training and testing sets.
* **Causality Checks:** HMM state probabilities were computed strictly via forward-only filtering (avoiding smoothed posteriors). Feature causality was verified via a feature-level truncation test.
* **Label-Shuffle Validation:** A block-permuted label-shuffle test (100 iterations) yielded precision distributions matching the base rate ($\sim$54.49%), confirming the tuning code path does not artificially inflate precision under a null hypothesis.
* **Holdout Diagnostics:** A 2x2 diagnostic matrix on the 600-day economic holdout mathematically separated market covariate drift (LR exposure collapse) from SVM margin scale shift.

## Repository Structure
The repository is strictly organized to separate core production code from research diagnostics.

```text
.
├── config/                 # Global hyperparameter grids and pipeline configuration
├── data/
│   ├── raw/                # Raw S&P 500 OHLCV data
│   └── processed/          # Cleaned and engineered features
├── diagnostics/            # 🔬 RESEARCH TRAIL: Statistical audits & null hypothesis testing
│   ├── final_diagnostic.py
│   ├── final_multi_seed_audit.py
│   ├── holdout_diagnostic.py
│   ├── max_stat_bootstrap.py        # Romano-Wolf FWER correction script
│   ├── nested_cv_obv_audit.py
│   ├── random_timing_benchmarks.py
│   └── stochastic_seed_audit.py     # HMM EM initialization convergence checks
├── notebooks/              # Jupyter notebooks for EDA and visual plotting
└── src/                    # ⚙️ CORE PIPELINE: Production-grade modeling
    ├── data_ingestion.py
    ├── features.py         # 10-variable tech features & non-stationary normalization
    ├── trading_model.py    # PCA -> Gaussian HMM -> RBF-SVM / LR architectures
    ├── evaluation_protocol.py
    ├── ablation.py
    ├── backtest.py
    ├── statistical_tests.py
    └── final_audit.py
```

## How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Core Evaluation Pipeline:**
   ```bash
   python src/final_audit.py
   ```
3. **Run the Multiple-Testing Bootstrap:**
   ```bash
   python diagnostics/max_stat_bootstrap.py
   ```

## Strategic Outlook & Next Steps
Applying a macro-volatility regime filter (HMM) to a micro-directional coin-flip target (next-day sign) represents a plausible target mismatch. Furthermore, estimating $\sim$50 parameters for a 3-state, 4-D covariance HMM on short 80–100 day inner training windows means the pipeline was likely dominated by estimation noise.

**Future Work:**
1. Reframe regime filters toward **volatility state forecasting** (benchmarked against GARCH/EWMA) rather than directional precision.
2. Shift to continuous evaluation metrics (Spearman Rank IC, Log-loss, Brier score) to preserve information.
3. The reserved 2026 data (~210 days) will not be used for precision testing due to severe underpowering (MDE $\sim$7 pp). It remains entirely untouched and is reserved strictly for a planned, pre-registered test of isotonic probability calibration.