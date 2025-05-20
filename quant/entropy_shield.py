
import numpy as np
import pandas as pd
from datetime import datetime
import json
import math

class EntropyShield:
    """
    Entropy Shield Risk Manager
    
    Dynamically adjusts position size based on market volatility and chaos.
    Protects against unexpected market movements by scaling risk exposure.
    """
    
    def __init__(self, max_risk=0.02, min_risk=0.005, volatility_window=20):
        """
        Initialize the Entropy Shield Risk Manager
        
        Parameters:
        - max_risk: Maximum risk per trade as decimal (default: 0.02 = 2%)
        - min_risk: Minimum risk per trade as decimal (default: 0.005 = 0.5%)
        - volatility_window: Window for volatility calculation (default: 20)
        """
        self.max_risk = max_risk
        self.min_risk = min_risk
        self.volatility_window = volatility_window
        self.entropy_history = []
        
    def calc_entropy(self, prices, lookback=None):
        """
        Measures market disorder (0 = calm, 1 = chaotic)
        
        Parameters:
        - prices: Array of price data
        - lookback: Number of periods to look back (default: volatility_window)
        
        Returns:
        - Entropy value between 0 and 1
        """
        if lookback is None:
            lookback = self.volatility_window
            
        if len(prices) < lookback + 1:
            return 0.5  # Default to medium entropy if not enough data
            
        returns = np.diff(np.log(prices[-lookback-1:]))
        
        epsilon = 1e-10
        abs_returns = np.abs(returns) + epsilon
        
        norm_returns = abs_returns / np.sum(abs_returns)
        
        entropy = -np.sum(norm_returns * np.log(norm_returns))
        
        max_entropy = -np.log(1.0 / len(returns))  # Maximum possible entropy
        normalized_entropy = entropy / max_entropy
        
        self.entropy_history.append({
            'timestamp': datetime.now().isoformat(),
            'entropy': float(normalized_entropy),
            'max_entropy': float(max_entropy),
            'raw_entropy': float(entropy)
        })
        
        if len(self.entropy_history) > 100:
            self.entropy_history = self.entropy_history[-100:]
            
        return normalized_entropy
        
    def position_size(self, entropy, account_size, price, stop_loss_pct=None):
        """
        Calculates position size based on current entropy
        
        Parameters:
        - entropy: Current market entropy (0-1)
        - account_size: Total account size in currency units
        - price: Current price of the asset
        - stop_loss_pct: Stop loss percentage (optional)
        
        Returns:
        - Dictionary with position sizing information
        """
        risk_pct = self.max_risk * (1 - entropy)
        
        risk_pct = max(risk_pct, self.min_risk)
        
        risk_amount = account_size * risk_pct
        
        if stop_loss_pct is not None and stop_loss_pct > 0:
            position_size = risk_amount / (price * stop_loss_pct)
        else:
            position_size = (account_size * risk_pct) / price
            
        return {
            'position_size': float(position_size),
            'risk_pct': float(risk_pct),
            'risk_amount': float(risk_amount),
            'entropy': float(entropy),
            'timestamp': datetime.now().isoformat()
        }
        
    def analyze_market_state(self, prices, volumes, high_prices=None, low_prices=None):
        """
        Comprehensive market state analysis
        
        Parameters:
        - prices: Array of closing prices
        - volumes: Array of volume data
        - high_prices: Array of high prices (optional)
        - low_prices: Array of low prices (optional)
        
        Returns:
        - Dictionary with market state analysis
        """
        if len(prices) < self.volatility_window * 2:
            return {'message': 'Insufficient data for analysis'}
            
        entropy = self.calc_entropy(prices)
        
        returns = np.diff(np.log(prices[-self.volatility_window-1:]))
        volatility = np.std(returns) * math.sqrt(252)  # Annualized
        
        volume_trend = np.mean(volumes[-5:]) / np.mean(volumes[-self.volatility_window:])
        
        momentum = prices[-1] / prices[-10] - 1
        
        avg_true_range = None
        if high_prices is not None and low_prices is not None and len(high_prices) == len(low_prices) == len(prices):
            true_ranges = []
            for i in range(1, min(self.volatility_window, len(prices))):
                true_range = max(
                    high_prices[-i] - low_prices[-i],
                    abs(high_prices[-i] - prices[-(i+1)]),
                    abs(low_prices[-i] - prices[-(i+1)])
                )
                true_ranges.append(true_range)
            avg_true_range = np.mean(true_ranges)
        
        if entropy > 0.7:
            market_state = "HIGH CHAOS"
            message = "High entropy detected. Significant risk reduction advised."
        elif entropy > 0.5:
            market_state = "MODERATE CHAOS"
            message = "Moderate entropy detected. Cautious position sizing recommended."
        else:
            market_state = "LOW CHAOS"
            message = "Low entropy detected. Standard position sizing acceptable."
            
        risk_pct = self.max_risk * (1 - entropy)
        risk_pct = max(risk_pct, self.min_risk)
        
        return {
            'market_state': market_state,
            'message': message,
            'entropy': float(entropy),
            'volatility': float(volatility),
            'volume_trend': float(volume_trend),
            'momentum': float(momentum),
            'avg_true_range': float(avg_true_range) if avg_true_range is not None else None,
            'recommended_risk_pct': float(risk_pct),
            'timestamp': datetime.now().isoformat()
        }
        
    def adjust_for_correlation(self, base_position_size, correlation_matrix, current_positions):
        """
        Adjust position size based on correlation with existing positions
        
        Parameters:
        - base_position_size: Initial position size calculation
        - correlation_matrix: Matrix of correlations between assets
        - current_positions: Dictionary of current positions {symbol: size}
        
        Returns:
        - Adjusted position size
        """
        if not correlation_matrix or not current_positions:
            return base_position_size
            
        total_correlation = 0
        for symbol, position in current_positions.items():
            if symbol in correlation_matrix:
                total_correlation += abs(position) * correlation_matrix[symbol]
                
        correlation_factor = 1 / (1 + total_correlation)
        
        return base_position_size * correlation_factor
        
    def save_analysis(self, analysis, filename='entropy_analysis.json'):
        """
        Save entropy analysis to a JSON file
        
        Parameters:
        - analysis: Analysis results dictionary
        - filename: Output filename (default: 'entropy_analysis.json')
        """
        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2)
            
    def load_analysis(self, filename='entropy_analysis.json'):
        """
        Load entropy analysis from a JSON file
        
        Parameters:
        - filename: Input filename (default: 'entropy_analysis.json')
        
        Returns:
        - Analysis results dictionary
        """
        with open(filename, 'r') as f:
            return json.load(f)


if __name__ == "__main__":
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.normal(0, 1, 100))
    volumes = np.random.normal(1000, 200, 100)
    
    shield = EntropyShield()
    
    entropy = shield.calc_entropy(prices)
    print(f"Market Entropy: {entropy:.4f}")
    
    position = shield.position_size(entropy, account_size=10000, price=prices[-1], stop_loss_pct=0.02)
    print(f"Position Size: {position['position_size']:.2f} units")
    print(f"Risk Percentage: {position['risk_pct']*100:.2f}%")
    
    analysis = shield.analyze_market_state(prices, volumes)
    print(f"Market State: {analysis['market_state']}")
    print(f"Message: {analysis['message']}")
