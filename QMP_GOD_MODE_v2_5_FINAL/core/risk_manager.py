import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class RiskManager:
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.max_portfolio_risk = 0.01  # 1% max risk per trade
        self.max_position_size = 0.25   # 25% max position size
        self.volatility_lookback = 30   # days
        
    def calculate_position_size(self, symbol, confidence, history_data):
        """Calculate position size using Kelly Criterion with fat-tail Expected Shortfall"""
        if '1m' not in history_data or history_data['1m'].empty:
            return 0.0
            
        recent_data = history_data['1m'].tail(self.volatility_lookback * 1440)  # 30 days of 1m data
        if len(recent_data) < 100:
            return 0.0
            
        returns = recent_data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252 * 1440)  # Annualized volatility
        
        var_95 = np.percentile(returns, 5)  # 95% confidence VaR
        es_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        
        if es_95 < -0.02:  # If 5% ES worse than -2%
            print(f"CRITICAL: Extreme market conditions detected for {symbol}. Expected Shortfall: {es_95:.4f}")
            return 0.0  # No position during extreme conditions
            
        win_rate = confidence  # Use confidence as win rate proxy
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0.01
        avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0.01
        
        if avg_loss == 0:
            kelly_fraction = 0.01
        else:
            kelly_fraction = (-es_95 / returns.std()) * 0.5  # Half Kelly for safety
            
            kurtosis = returns.kurtosis()
            is_fat_tailed = kurtosis > 3.0  # Normal distribution has kurtosis = 3
            
            if is_fat_tailed:
                kelly_fraction *= (3.0 / kurtosis)
                
            # Cap kelly between 0.1% and 20%
            kelly_fraction = max(0.001, min(0.2, kelly_fraction))
            
            # Add additional safety check for extreme volatility
            if returns.std() > 0.05:  # If daily vol > 5%
                kelly_fraction *= 0.5  # Further reduce position size
            
        # Dynamic volatility scaling - more conservative during high volatility
        vol_scaling = min(1.0, 0.15 / volatility)  # Scale down if vol > 15%
        
        position_size = kelly_fraction * vol_scaling * confidence
        
        position_size = max(0.0, min(position_size, self.max_position_size))
        
        portfolio_value = self.algorithm.Portfolio.TotalPortfolioValue
        max_risk_size = (self.max_portfolio_risk * portfolio_value) / (volatility * portfolio_value)
        position_size = min(position_size, max_risk_size)
        
        return position_size
