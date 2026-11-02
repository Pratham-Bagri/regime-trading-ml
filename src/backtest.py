import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import config

def run_financial_backtest():
    print("Loading raw features for the final financial backtest...")
    df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col='Date', parse_dates=True)
    df = df.dropna()
    
    df['Signal'] = np.where(df['target_return'] > 0, 1, 0)
    feature_cols = [col for col in df.columns if col not in ['target_return', 'Signal', 'Close', 'return']]
    X = df[feature_cols].values
    y = df['Signal'].values
    returns = df['target_return'].values
    
    # Isolate the untouched 20% holdout set
    holdout_idx = int(len(X) * 0.8)
    X_cv, y_cv = X[:holdout_idx], y[:holdout_idx]
    X_holdout, y_holdout = X[holdout_idx:], y[holdout_idx:]
    returns_holdout = returns[holdout_idx:]
    
    # Train final model on the entire cross-validation set
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=0.1, random_state=42, max_iter=1000))
    ])
    pipeline.fit(X_cv, y_cv)
    
    # Generate predictions on the holdout set
    # Using the 50% threshold to maximize market exposure for a fair comparison
    predictions = pipeline.predict(X_holdout)
    
    # Calculate Financial Metrics
    trading_days = 252
    transaction_cost = 0.0005  # 5 bps per trade
    
    # Strategy Returns: If we predict 1, we capture the next day's return. 
    # If 0, we sit in cash (0 return).
    strategy_returns = predictions * returns_holdout
    
    # Subtract transaction costs whenever our position changes
    # Position changes: abs(pred_t - pred_{t-1})
    position_changes = np.abs(np.diff(np.insert(predictions, 0, 0)))
    strategy_returns_net = strategy_returns - (position_changes * transaction_cost)
    
    # Buy and Hold Baseline
    bnh_returns = returns_holdout
    
    def calc_metrics(ret_array):
        cum_ret = np.prod(1 + ret_array) - 1
        ann_ret = (1 + cum_ret) ** (trading_days / len(ret_array)) - 1
        vol = np.std(ret_array) * np.sqrt(trading_days)
        sharpe = ann_ret / vol if vol != 0 else 0
        
        # Max Drawdown
        cum_index = np.cumprod(1 + ret_array)
        running_max = np.maximum.accumulate(cum_index)
        drawdown = (cum_index - running_max) / running_max
        max_dd = np.min(drawdown)
        
        return ann_ret, vol, sharpe, max_dd

    strat_ann, strat_vol, strat_sharpe, strat_dd = calc_metrics(strategy_returns_net)
    bnh_ann, bnh_vol, bnh_sharpe, bnh_dd = calc_metrics(bnh_returns)
    
    print("\n--- Holdout Set Financial Backtest ---")
    print(f"Trading Days: {len(returns_holdout)}")
    print(f"Total Trades Executed: {np.sum(position_changes)}")
    print("\nBuy and Hold Baseline:")
    print(f"Annualized Return: {bnh_ann:.2%}")
    print(f"Volatility: {bnh_vol:.2%}")
    print(f"Sharpe Ratio: {bnh_sharpe:.2f}")
    print(f"Max Drawdown: {bnh_dd:.2%}")
    
    print("\nLogistic Regression Strategy (Net of 5bps Costs):")
    print(f"Annualized Return: {strat_ann:.2%}")
    print(f"Volatility: {strat_vol:.2%}")
    print(f"Sharpe Ratio: {strat_sharpe:.2f}")
    print(f"Max Drawdown: {strat_dd:.2%}")

if __name__ == '__main__':
    run_financial_backtest()