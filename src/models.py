# src/models.py
# PROJECT BLUEPRINT: MACHINE LEARNING MODELS FOR PAIR TRADING
# Two-tier approach: Linear baseline, non-linear ensemble, temporal deep learning

import torch
import torch.nn as nn
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier

#######################################
# TIER 1: LINEAR BASELINE (Ridge Regression) 
#######################################

class BaselineModel:
    """
    Ridge Regression classifier for pair trading.
    Models mean reversion as a linear process with L2 regularization.
    
    Attributes:
        model: RidgeClassifier instance
        alpha: Regularization strength (prevents overfitting)
    """
    def __init__(self, alpha=1.0):
        self.model = RidgeClassifier(alpha=alpha)

    def fit(self, X_train, y_train):
        """Train the Ridge model on training data."""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Return binary predictions (0 or 1)."""
        return self.model.predict(X_test)


#######################################
# TIER 2: DEEP LEARNING (LSTM)
#######################################

class LSTMModel(nn.Module):
    """
    Long Short-Term Memory (LSTM) network for pair trading.
    Captures temporal dependencies in price spread sequences.
    
    Architecture:
        - LSTM layer with memory cells for temporal learning
        - Fully connected output layer with sigmoid activation
        - Outputs probability predictions (0.0 to 1.0)
    
    Attributes:
        lstm: LSTM layer
        fc: Fully connected output layer
        sigmoid: Sigmoid activation for probability calibration
        hidden_dim: Size of LSTM hidden state
        num_layers: Number of stacked LSTM layers
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        """
        Initialize LSTM model.
        
        Args:
            input_dim (int): Number of input features (2: Z_Score, Volatility)
            hidden_dim (int): Number of hidden units in LSTM (default: 64)
            num_layers (int): Number of stacked LSTM layers (default: 1)
        """
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM Layer (The "Memory" Engine)
        # Processes sequences and learns temporal patterns
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully Connected Output Layer (Decision Maker)
        # Maps LSTM hidden state to probability prediction
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through LSTM network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
               Where seq_length = 10 (lookback window)
               
        Returns:
            Probability predictions of shape (batch_size, 1)
        """
        # x shape: (Batch_Size, Sequence_Length, Features)
        
        # Initialize hidden state (h0) and cell state (c0) with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM through entire sequence
        # out shape: (batch_size, seq_length, hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        
        # Use only the last timestep output for prediction
        # out[:, -1, :] extracts the final sequence element
        out = out[:, -1, :]
        
        # Pass through fully connected layer
        out = self.fc(out)
        
        # Apply sigmoid for probability calibration (0 to 1)
        return self.sigmoid(out)