"""
Backtest Engine - Core trading logic
"""

import pandas as pd
import numpy as np


class SimpleBacktest:
    """Run multi-pair backtest with mean reversion signals."""
    
    def __init__(self, initial_capital=10000, position_risk_pct=0.02, 
                 max_positions=3, transaction_cost=0.004):
        self.initial_capital = initial_capital
        self.realized_equity = initial_capital
        self.position_risk_pct = position_risk_pct
        self.max_positions = max_positions
        self.transaction_cost = transaction_cost
        
        # Signal thresholds
        self.ENTRY_Z = 1.5
        self.EXIT_Z = 0.5
        self.MODEL_CONFIDENCE = 0.55
        
        self.positions = {}
        self.trades = []
        self.equity_curve = []
    
    def run(self, df, model_col):
        """Run backtest on predictions."""
        
        for date in df.index.unique():
            day_data = df[df.index == date]
            
            # Process exits
            for pair_id in list(self.positions.keys()):
                row = day_data[day_data['Pair_ID'] == pair_id]
                if row.empty:
                    continue
                
                z_score = float(row['Z_Score'].values[0])
                spread_price = float(row['Spread'].values[0])
                
                if abs(z_score) < self.EXIT_Z:
                    self._close_position(pair_id, spread_price, date, reason='Mean Reversion')
            
            # Process entries
            for _, row in day_data.iterrows():
                pair_id = row['Pair_ID']
                
                if pair_id in self.positions:
                    continue
                
                if len(self.positions) >= self.max_positions:
                    continue
                
                z_score = float(row['Z_Score'])
                spread_price = float(row['Spread'])
                pred = float(row[model_col])
                
                # Long: Low spread + Model predicts UP
                if z_score < -self.ENTRY_Z and pred > self.MODEL_CONFIDENCE:
                    self._open_position(pair_id, spread_price, date, direction=1)
                
                # Short: High spread + Model predicts DOWN
                elif z_score > self.ENTRY_Z and pred < (1 - self.MODEL_CONFIDENCE):
                    self._open_position(pair_id, spread_price, date, direction=-1)
            
            # Mark-to-market
            self._mark_to_market(day_data, date)
        
        # Close remaining positions
        last_date = df.index.max()
        last_data = df[df.index == last_date]
        for pair_id in list(self.positions.keys()):
            row = last_data[last_data['Pair_ID'] == pair_id]
            if not row.empty:
                spread = float(row['Spread'].values[0])
                self._close_position(pair_id, spread, last_date, reason='End of Backtest')
        
        return self.equity_curve, self.trades
    
    def _open_position(self, pair_id, spread_price, date, direction):
        """Open a position."""
        capital = self.realized_equity * self.position_risk_pct
        entry_price = spread_price * (1 + self.transaction_cost if direction > 0 else 1 - self.transaction_cost)
        size = capital / spread_price
        
        self.positions[pair_id] = {
            'entry_price': entry_price,
            'entry_spread': spread_price,
            'size': size,
            'direction': direction,
            'capital': capital,
        }
        
        self.trades.append({
            'date': date,
            'pair': pair_id,
            'type': 'entry',
            'direction': 'LONG' if direction > 0 else 'SHORT',
        })
    
    def _close_position(self, pair_id, spread_price, date, reason):
        """Close a position and realize PnL."""
        if pair_id not in self.positions:
            return
        
        pos = self.positions.pop(pair_id)
        exit_price = spread_price * (1 - self.transaction_cost if pos['direction'] > 0 else 1 + self.transaction_cost)
        pnl = (exit_price - pos['entry_price']) * pos['size'] * pos['direction']
        
        self.realized_equity += pnl
        
        self.trades.append({
            'date': date,
            'pair': pair_id,
            'type': 'exit',
            'realized_pnl': pnl,
            'pnl_pct': (pnl / pos['capital'] * 100) if pos['capital'] > 0 else 0,
        })
    
    def _mark_to_market(self, day_data, date):
        """Calculate total equity."""
        unrealized = 0.0
        
        for pair_id, pos in self.positions.items():
            row = day_data[day_data['Pair_ID'] == pair_id]
            if row.empty:
                continue
            
            current_spread = float(row['Spread'].values[0])
            current_price = current_spread * (1 - self.transaction_cost if pos['direction'] > 0 else 1 + self.transaction_cost)
            pnl = (current_price - pos['entry_price']) * pos['size'] * pos['direction']
            unrealized += pnl
        
        total_equity = self.realized_equity + unrealized
        self.equity_curve.append((date, total_equity))