import pandas as pd
import numpy as np
import os
import sys

class StrategyEngine:
    """
    A scalable engine for backtesting pair trading strategies based on
    Z-Score thresholds and Machine Learning signals.
    """

    def __init__(self, predictions_path, z_threshold=1.5):
        """
        Initialize the strategy engine.
        
        Args:
            predictions_path (str): Path to the CSV containing model predictions.
            z_threshold (float): The Z-Score level that triggers a trade entry.
        """
        self.predictions_path = predictions_path
        self.z_threshold = z_threshold
        self.df = self._load_data()

    def _load_data(self):
        """Internal method to safely load and validate data."""
        if not os.path.exists(self.predictions_path):
            raise FileNotFoundError(f" Error: File not found at {self.predictions_path}")
        
        df = pd.read_csv(self.predictions_path, index_col=0, parse_dates=True)
        # Ensure we have the necessary columns
        required_cols = ['Z_Score', 'Target_Direction', 'Ridge_Pred', 'LSTM_Pred']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
        return df

    def run_backtest(self):
        """
        Executes the strategy logic for individual models and the hybrid consensus.
        Returns a summary DataFrame and a dictionary of detailed metrics.
        """
        print(f"RUNNING STRATEGY (Threshold: Z > {self.z_threshold} / Z < -{self.z_threshold})")
        
        # 1. IDENTIFY TRADE OPPORTUNITIES
        #before looking at any ML predictions, filter fo days where market is stretched
        long_opps = self.df[self.df['Z_Score'] < -self.z_threshold].copy() 
        short_opps = self.df[self.df['Z_Score'] > self.z_threshold].copy() 
    
        actionable_count = len(long_opps) + len(short_opps)
        total_days = len(self.df)
        print(f"   Actionable Opportunities: {actionable_count}")

        results = []

        # 2. EVALUATE INDIVIDUAL MODELS
        models = ['Ridge', 'LSTM']
        for model in models:
            metrics = self._evaluate_model(model, long_opps, short_opps)
            results.append(metrics)

        # 3. EVALUATE HYBRID MODEL (Consensus)
        hybrid_metrics = self._evaluate_hybrid(long_opps, short_opps)
        results.append(hybrid_metrics)

        # 4. FORMAT RESULTS
        results_df = pd.DataFrame(results).set_index('Model')
        return results_df

    def _evaluate_model(self, model_name, long_opps, short_opps):
        """Calculates win rates for a single model."""
        pred_col = f"{model_name}_Pred"

        # Long Logic: Z is low (-), Model predicts Up (1), Target is Up (1)
        long_trades = long_opps[long_opps[pred_col] == 1]
        long_wins = long_trades[long_trades['Target_Direction'] == 1]

        # Short Logic: Z is high (+), Model predicts Down (0), Target is Down (0)
        short_trades = short_opps[short_opps[pred_col] == 0]
        short_wins = short_trades[short_trades['Target_Direction'] == 0]

        return self._calculate_metrics(model_name, long_trades, long_wins, short_trades, short_wins)

    def _evaluate_hybrid(self, long_opps, short_opps):
        """Calculates win rates when BOTH models agree."""
        
        # Hybrid Long: Both say "1"
        h_longs = long_opps[(long_opps['Ridge_Pred'] == 1) & (long_opps['LSTM_Pred'] == 1)]
        h_long_wins = h_longs[h_longs['Target_Direction'] == 1]

        # Hybrid Short: Both say "0"
        h_shorts = short_opps[(short_opps['Ridge_Pred'] == 0) & (short_opps['LSTM_Pred'] == 0)]
        h_short_wins = h_shorts[h_shorts['Target_Direction'] == 0]

        return self._calculate_metrics('Hybrid', h_longs, h_long_wins, h_shorts, h_short_wins)

    def _calculate_metrics(self, name, l_trades, l_wins, s_trades, s_wins):
        """Helper to calculate standard trading metrics."""
        total_trades = len(l_trades) + len(s_trades)
        total_wins = len(l_wins) + len(s_wins)
        
        return {
            'Model': name,
            'Total_Trades': total_trades,
            'Win_Rate': total_wins / total_trades if total_trades > 0 else 0.0,
            'Long_WR': len(l_wins) / len(l_trades) if len(l_trades) > 0 else 0.0,
            'Short_WR': len(s_wins) / len(s_trades) if len(s_trades) > 0 else 0.0
        }

if __name__ == "__main__":
    # Quick test if run directly
    path = '../data/processed/05_model_predictions.csv'
    if os.path.exists(path):
        engine = StrategyEngine(path)
        print(engine.run_backtest())
    else:
        print("Please run from project root or check file path.")