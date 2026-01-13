#REFACTORE FEATURE ENGINEERING LOGIC INTO A DEDICATED MODULE
import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Centralizes all feature engineering logic for Pair Trading.
    Generates Mean Reversion, Volatility, and Momentum metrics.
    """
    
    def __init__(self, window=20, horizon=10):
        self.window = window
        self.horizon = horizon
        self.epsilon = 1e-6  # To prevent division by zero

    def calculate_rolling_stats(self, spread):
        """Calculates base rolling metrics needed for other features."""
        return {
            'mean': spread.rolling(self.window).mean(),
            'std': spread.rolling(self.window).std(),
            'min': spread.rolling(self.window).min(),
            'max': spread.rolling(self.window).max()
        }

    def feature_z_score(self, spread, stats):
        """Standard Z-Score: (Spread - Mean) / Std"""
        return (spread - stats['mean']) / (stats['std'] + self.epsilon)

    def feature_extreme_z(self, z_score, threshold=1.5):
        """Binary: Is Z-Score beyond +/- threshold?"""
        return (z_score.abs() > threshold).astype(int)

    def feature_distance_from_mean(self, spread, stats):
        """Absolute distance from mean in standard deviations"""
        return (spread - stats['mean']).abs() / (stats['std'] + self.epsilon)

    def feature_range_position(self, spread, stats):
        """Oscillator: Position within recent Min/Max range (0 to 1)"""
        numerator = spread - stats['min']
        denominator = stats['max'] - stats['min'] + self.epsilon
        return numerator / denominator

    def feature_mr_strength(self, spread, stats):
        """Mean Reversion Strength: Direction * Magnitude"""
        # sign(Mean - Spread) gives direction we WANT it to go
        direction = np.sign(stats['mean'] - spread) 
        magnitude = (spread - stats['mean']).abs() / (stats['std'] + self.epsilon)
        return direction * magnitude

    def feature_volatility_expansion(self, stats):
        """Ratio of current Volatility to 10-day average Volatility"""
        vol_sma = stats['std'].rolling(10).mean()
        return stats['std'] / (vol_sma + self.epsilon)

    def feature_recent_extreme(self, spread, stats):
        """Did the spread touch 2-sigma bands yesterday?"""
        # Shift(1) because we want to know if it WAS extreme yesterday
        prev_spread = spread.shift(1).abs()
        prev_threshold = stats['std'].shift(1) * 2
        return (prev_spread > prev_threshold).astype(int)

    def generate_targets(self, df, spread):
        """Generates the prediction targets based on the horizon."""
        df['Target_Return'] = spread.shift(-self.horizon) - spread
        df['Target_Direction'] = (df['Target_Return'] > 0).astype(int)
        return df

    def generate_all_features(self, spread):
        """
        Main execution function.
        Args:
            spread (pd.Series): The raw price spread.
        Returns:
            pd.DataFrame: Dataframe with all features and targets.
        """
        df = pd.DataFrame(index=spread.index)
        df['Spread'] = spread

        # 1. Base Stats
        stats = self.calculate_rolling_stats(spread)

        # 2. Standardized Metrics
        df['Z_Score'] = self.feature_z_score(spread, stats)
        df['Extreme_Z'] = self.feature_extreme_z(df['Z_Score'])
        df['Distance_Mean'] = self.feature_distance_from_mean(spread, stats)
        df['Volatility'] = stats['std']

        # 3. Oscillators
        df['Range_Position'] = self.feature_range_position(spread, stats)
        df['Recent_Extreme'] = self.feature_recent_extreme(spread, stats)

        # 4. Dynamics
        df['MR_Strength'] = self.feature_mr_strength(spread, stats)
        df['Vol_Expansion'] = self.feature_volatility_expansion(stats)

        # 5. Generate Targets
        df = self.generate_targets(df, spread)

        return df.dropna()