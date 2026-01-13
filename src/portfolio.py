import pandas as pd
import numpy as np

class PortfolioManager:
    """
    Handles financial simulation using realistic returns and transaction costs.
    Complementary to StrategyEngine (which handles signal generation and win rates).
    """

    def __init__(self, strategy_engine):
        """
        Initialize using the data and configuration already loaded by the Engine.
        
        Args:
            strategy_engine (StrategyEngine): Initialized instance of StrategyEngine.
        """
        # Reuse the dataframe and threshold from the Engine to avoid redundancy
        self.df = strategy_engine.df.copy()
        self.z_threshold = strategy_engine.z_threshold

    def calculate_equity_curve(self, capital_per_trade=1000, cost_per_trade=2.0):
        """
        Simulates the portfolio equity curve based on Real Returns.
        
        Args:
            capital_per_trade (float): Capital allocated per trade (e.g., $1000).
            cost_per_trade (float): Total transaction cost per trade (Commission + Slippage).
            
        Returns:
            pd.DataFrame: Daily aggregated equity curve.
        """
        print(f"\n💰 SIMULATING REAL PORTFOLIO (Capital: ${capital_per_trade}, Cost: ${cost_per_trade})")
        
        sim_df = self.df.copy()

        # 1. REPLICATE SIGNAL LOGIC (Consistent with StrategyEngine)
        # Z-Score Triggers
        long_condition = sim_df['Z_Score'] < -self.z_threshold
        short_condition = sim_df['Z_Score'] > self.z_threshold

        # LSTM Filters
        # Note: StrategyEngine defines a trade where (Trigger == True) AND (Pred == 1/0)
        lstm_long = long_condition & (sim_df['LSTM_Pred'] == 1)
        lstm_short = short_condition & (sim_df['LSTM_Pred'] == 0)

        # Hybrid Filters (Consensus)
        hybrid_long = lstm_long & (sim_df['Ridge_Pred'] == 1)
        hybrid_short = lstm_short & (sim_df['Ridge_Pred'] == 0)

        # 2. DEFINE P&L LOGIC (Real Returns - Costs)
        def get_pnl_series(long_mask, short_mask):
            pnl = pd.Series(0.0, index=sim_df.index)
            
            # P&L = (Capital * Target_Return) - Cost
            # Note: Target_Return is the 10-day forward return magnitude.
            # If Long and Direction=1 (Up), we Win.
            # If Long and Direction=0 (Down), we Lose.
            
            # Long Trades
            # We profit if Direction is 1, lose if 0.
            # We assume 'Target_Return' is absolute magnitude or aligned with direction.
            # Standard assumption: Return is the % change. 
            # If we bought, PnL is Return.
            pnl.loc[long_mask] = np.where(
                sim_df.loc[long_mask, 'Target_Direction'] == 1,
                (capital_per_trade * sim_df.loc[long_mask, 'Target_Return']) - cost_per_trade,
                (capital_per_trade * -sim_df.loc[long_mask, 'Target_Return']) - cost_per_trade
            )
            
            # Short Trades 
            # We profit if Direction is 0 (Price went down).
            pnl.loc[short_mask] = np.where(
                sim_df.loc[short_mask, 'Target_Direction'] == 0,
                (capital_per_trade * sim_df.loc[short_mask, 'Target_Return']) - cost_per_trade,
                (capital_per_trade * -sim_df.loc[short_mask, 'Target_Return']) - cost_per_trade
            )
            return pnl

        # 3. CALCULATE & AGGREGATE
        sim_df['LSTM_Pnl'] = get_pnl_series(lstm_long, lstm_short)
        sim_df['Hybrid_Pnl'] = get_pnl_series(hybrid_long, hybrid_short)

        # Group by Date to aggregate portfolio performance across all pairs
        daily_results = sim_df.groupby(sim_df.index)[['LSTM_Pnl', 'Hybrid_Pnl']].sum()

        # Cumulative Sum
        daily_results['Equity_LSTM'] = daily_results['LSTM_Pnl'].cumsum()
        daily_results['Equity_Hybrid'] = daily_results['Hybrid_Pnl'].cumsum()

        # Filter active days
        active_days = daily_results[
            (daily_results['Equity_LSTM'] != 0) | (daily_results['Equity_Hybrid'] != 0)
        ].copy()

        return active_days