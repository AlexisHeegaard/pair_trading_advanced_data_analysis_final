"""
Backtest Runner

Run ML backtests: Ridge vs LSTM
"""

import pandas as pd
import sys
import os

from src.backtest import SimpleBacktest
from src.analysis import calculate_stats, print_results

# ========== LOAD DATA ==========
print("Loading predictions...")
df = pd.read_csv('../data/processed/05_model_predictions.csv', index_col=0, parse_dates=True)
df = df.sort_index()

print(f"Loaded {len(df)} rows")
print(f"Date range: {df.index.min().date()} to {df.index.max().date()}\n")

# ========== CONFIG ==========
INITIAL_CAPITAL = 10000
POSITION_RISK_PCT = 0.02
MAX_POSITIONS = 3
TRANSACTION_COST = 0.004

# ========== RUN BACKTESTS ==========
print("="*70)
print("BACKTESTING ML PREDICTIONS")
print("="*70)
print(f"Initial Capital:     ${INITIAL_CAPITAL:,.0f}")
print(f"Position Risk:       {POSITION_RISK_PCT*100:.1f}% per trade")
print(f"Max Concurrent:      {MAX_POSITIONS} positions")
print(f"Transaction Cost:    {TRANSACTION_COST*100:.2f}%\n")

# Ridge
print("Running Ridge Regression backtest...")
ridge_bt = SimpleBacktest(INITIAL_CAPITAL, POSITION_RISK_PCT, MAX_POSITIONS, TRANSACTION_COST)
ridge_equity, ridge_trades = ridge_bt.run(df, 'Ridge_Pred')
ridge_stats = calculate_stats(ridge_equity, ridge_trades)

# LSTM
print("Running LSTM backtest...")
lstm_bt = SimpleBacktest(INITIAL_CAPITAL, POSITION_RISK_PCT, MAX_POSITIONS, TRANSACTION_COST)
lstm_equity, lstm_trades = lstm_bt.run(df, 'LSTM_Pred')
lstm_stats = calculate_stats(lstm_equity, lstm_trades)

# ========== PRINT RESULTS ==========
print("\n" + "="*70)
print_results("RIDGE REGRESSION", ridge_stats, INITIAL_CAPITAL)
print_results("LSTM", lstm_stats, INITIAL_CAPITAL)

# Winner
ridge_pnl = ridge_stats.get('total_pnl', 0)
lstm_pnl = lstm_stats.get('total_pnl', 0)

print("="*70)
if ridge_pnl > lstm_pnl:
    print(f"🏆 WINNER: RIDGE (${ridge_pnl:,.2f} vs ${lstm_pnl:,.2f})")
else:
    print(f"🏆 WINNER: LSTM (${lstm_pnl:,.2f} vs ${ridge_pnl:,.2f})")
print("="*70)
