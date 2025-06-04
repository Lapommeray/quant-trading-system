"""
Example integration of advanced indicators with existing system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.indicators import HestonVolatility, ML_RSI, OrderFlowImbalance, RegimeDetector

def create_sample_data():
    """Create sample market data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
    
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 300)
    prices = [100.0]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    df = pd.DataFrame({
        'Close': prices,
        'Volume': np.random.normal(1000000, 200000, 300)
    }, index=dates)
    
    return df

def calculate_traditional_rsi(prices, window=14):
    """Calculate traditional RSI for ML_RSI input"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def demonstrate_integration():
    """Demonstrate how to use new indicators with existing system"""
    df = create_sample_data()
    
    heston_vol = HestonVolatility()
    volatility = heston_vol.calculate(df['Close'])
    print(f"Heston Volatility (latest): {volatility.iloc[-1]:.4f}")
    
    traditional_rsi = calculate_traditional_rsi(df['Close'])
    ml_rsi = ML_RSI()
    ml_predictions = ml_rsi.calculate(df['Close'], traditional_rsi)
    print(f"ML RSI Prediction (latest): {ml_predictions.iloc[-1]:.4f}")
    
    tick_data = pd.DataFrame({
        'price': np.random.normal(100, 1, 1000),
        'quantity': np.random.randint(1, 100, 1000),
        'side': np.random.choice([1, -1], 1000)
    })
    
    order_flow = OrderFlowImbalance()
    try:
        imbalance = order_flow.calculate(tick_data)
    except TypeError:
        try:
            imbalance = order_flow.calculate(tick_data['price'], tick_data['quantity'])
        except TypeError:
            imbalance = order_flow.calculate(tick_data['price'], tick_data['quantity'], tick_data['price'], tick_data['price'])
    print(f"Order Flow Imbalance (latest): {imbalance.iloc[-1]:.4f}")
    
    regime_detector = RegimeDetector()
    regimes = regime_detector.calculate(volatility, traditional_rsi.fillna(50))
    print(f"Current Market Regime: {regimes.iloc[-1]}")
    
    return {
        'volatility': volatility,
        'ml_rsi': ml_predictions,
        'order_flow': imbalance,
        'regimes': regimes
    }

if __name__ == "__main__":
    results = demonstrate_integration()
    print("Integration test completed successfully!")
