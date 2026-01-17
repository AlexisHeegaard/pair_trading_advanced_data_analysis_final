import pandas as pd
import numpy as np

class PortfolioManager:
    """
    Handles financial simulation using realistic returns and transaction costs.
    Complementary to StrategyEngine (which handles signal generation and win rates).
    """

    def __init__(self, strategy_engine, total_capital=1000, max_positions=5):
        """
        Initialize using the data and configuration already loaded by the Engine.
        
        Args:
            strategy_engine (StrategyEngine): Initialized instance of StrategyEngine.
            total_capital (float): Starting portfolio value.
            max_positions (int): Maximum simultaneous open positions.
        """
        self.df = strategy_engine.df.copy()
        self.z_threshold = strategy_engine.z_threshold
        self.total_capital = total_capital
        self.max_positions = max_positions
        
        # Realistic cost parameters
        self.slippage_pct = 0.001       # 0.1% slippage per trade
        self.spread_pct = 0.0005        # 0.05% bid-ask spread
        self.short_borrow_rate = 0.02   # 2% annual borrowing cost

    def calculate_equity_curve(self, capital_per_trade=100, cost_per_trade=2.0, hold_period=10):
        """
        Simulates the portfolio equity curve based on Real Returns.
        
        Args:
            capital_per_trade (float): Capital allocated per trade.
            cost_per_trade (float): Base transaction cost per trade (commission).
            hold_period (int): Days to hold each position.
            
        Returns:
            pd.DataFrame: Daily aggregated equity curve.
        """
        print(f"\nSIMULATING PORTFOLIO")
        print(f"   Starting Capital: ${self.total_capital}")
        print(f"   Capital Per Trade: ${capital_per_trade}")
        print(f"   Commission: ${cost_per_trade}")
        print(f"   Slippage: {self.slippage_pct:.2%}")
        print(f"   Spread: {self.spread_pct:.2%}")
        print(f"   Short Borrow Rate: {self.short_borrow_rate:.2%} annual")
        print(f"   Max Positions: {self.max_positions}")
        print(f"   Hold Period: {hold_period} days")

        # Sort data by date
        sim_df = self.df.sort_index()
        
        # Track state for each strategy
        strategies = ['LSTM', 'Hybrid']
        results = {s: self._run_simulation(sim_df, s, capital_per_trade, 
                                            cost_per_trade, hold_period) 
                   for s in strategies}
        
        # Combine results
        equity_df = pd.DataFrame({
            'Equity_LSTM': results['LSTM']['equity'],
            'Equity_Hybrid': results['Hybrid']['equity'],
            'Positions_LSTM': results['LSTM']['positions'],
            'Positions_Hybrid': results['Hybrid']['positions']
        })
        
        # Print summary
        for s in strategies:
            final_equity = results[s]['equity'].iloc[-1]
            total_return = (final_equity - self.total_capital) / self.total_capital
            print(f"\n   {s} Results:")
            print(f"      Final Equity: ${final_equity:,.2f}")
            print(f"      Total Return: {total_return:.2%}")
            print(f"      Trades Executed: {results[s]['trade_count']}")
            print(f"      Trades Skipped (no capital): {results[s]['skipped']}")
        
        return equity_df

    def _add_trading_days(self, start_date, days):
        """Calculate close date skipping weekends."""
        current = start_date
        added = 0
        while added < days:
            current += pd.Timedelta(days=1)
            if current.weekday() < 5:  # Monday=0 to Friday=4
                added += 1
        return current

    def _run_simulation(self, sim_df, strategy, capital_per_trade, cost_per_trade, hold_period):
        """
        Internal method to run simulation for a single strategy.
        
        Args:
            sim_df: Sorted DataFrame with predictions
            strategy: 'LSTM' or 'Hybrid'
            capital_per_trade: Amount per trade
            cost_per_trade: Base commission cost
            hold_period: Days to hold position
            
        Returns:
            dict: equity series, position count series, trade stats
        """
        # State tracking
        equity = self.total_capital
        available_capital = self.total_capital
        open_positions = []
        
        # Results tracking
        equity_history = []
        position_history = []
        trade_count = 0
        skipped_count = 0
        
        for date, row in sim_df.iterrows():
            
            # 1. CLOSE EXPIRED POSITIONS
            for pos in open_positions[:]:
                if date >= pos['close_date']:
                    available_capital += pos['invested'] + pos['pnl']
                    equity += pos['pnl']
                    open_positions.remove(pos)
            
            # 2. CHECK FOR SIGNALS
            z = row['Z_Score']
            
            if strategy == 'LSTM':
                long_signal = (z < -self.z_threshold) and (row['LSTM_Pred'] == 1)
                short_signal = (z > self.z_threshold) and (row['LSTM_Pred'] == 0)
            else:  # Hybrid
                long_signal = (z < -self.z_threshold) and (row['LSTM_Pred'] == 1) and (row['Ridge_Pred'] == 1)
                short_signal = (z > self.z_threshold) and (row['LSTM_Pred'] == 0) and (row['Ridge_Pred'] == 0)
            
            has_signal = long_signal or short_signal
            
            # 3. EXECUTE IF CONSTRAINTS MET
            if has_signal:
                can_trade = (
                    len(open_positions) < self.max_positions and
                    available_capital >= capital_per_trade * 1.1  # Buffer for costs
                )
                
                if can_trade:
                    # CALCULATE REALISTIC COSTS
                    slippage_cost = capital_per_trade * self.slippage_pct
                    spread_cost = capital_per_trade * self.spread_pct
                    total_cost = cost_per_trade + slippage_cost + spread_cost
                    
                    # Add borrowing cost for short positions
                    if short_signal:
                        borrow_cost = capital_per_trade * (self.short_borrow_rate * hold_period / 365)
                        total_cost += borrow_cost
                    
                    # Deduct capital and costs
                    available_capital -= (capital_per_trade + total_cost)
                    equity -= total_cost
                    
                    # Calculate P&L based on actual target return
                    target_return = row['Target_Return']
                    target_direction = row['Target_Direction']
                    
                    if long_signal:
                        if target_direction == 1:
                            pnl = capital_per_trade * abs(target_return)
                        else:
                            pnl = -capital_per_trade * abs(target_return)
                    else:
                        if target_direction == 0:
                            pnl = capital_per_trade * abs(target_return)
                        else:
                            pnl = -capital_per_trade * abs(target_return)
                    
                    # Record position (skip weekends for close date)
                    close_date = self._add_trading_days(date, hold_period)
                    
                    open_positions.append({
                        'close_date': close_date,
                        'invested': capital_per_trade,
                        'pnl': pnl
                    })
                    
                    trade_count += 1
                else:
                    skipped_count += 1
            
            # 4. RECORD STATE
            equity_history.append(equity)
            position_history.append(len(open_positions))
        
        return {
            'equity': pd.Series(equity_history, index=sim_df.index),
            'positions': pd.Series(position_history, index=sim_df.index),
            'trade_count': trade_count,
            'skipped': skipped_count
        }